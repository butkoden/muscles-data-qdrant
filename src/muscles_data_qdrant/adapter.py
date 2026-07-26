from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Mapping

from muscles_data.config import DataResourceConfig
from muscles_data.errors import (
    DataConfigurationError,
    DataConnectionError,
    DataError,
    DataSchemaMismatchError,
    DataUnsupportedOperationError,
    DataVectorDimensionError,
)
from muscles_data.models import DataCapability, HealthResult, InspectResult, VectorHit, WriteResult


_RANGE_OPERATORS = {"gt", "gte", "lt", "lte"}
_CLIENT_UNSET = object()
_MODELS_UNSET = object()


class QdrantAdapterError(DataError):
    """Base error for Qdrant vector adapter failures."""


class QdrantConfigError(DataConfigurationError, QdrantAdapterError):
    """Raised when Qdrant resource configuration is incomplete or invalid."""


class QdrantClientMissingError(DataConfigurationError, QdrantAdapterError):
    """Raised when Qdrant adapter is used without an available client."""


class QdrantConnectionError(DataConnectionError, QdrantAdapterError):
    """Raised when Qdrant operation cannot reach or use the backend."""


class QdrantFilterError(DataUnsupportedOperationError, QdrantAdapterError):
    """Raised when a data filter cannot be translated to Qdrant."""


class QdrantDimensionError(DataVectorDimensionError, QdrantAdapterError):
    """Raised when vectors are empty or incompatible with the Qdrant collection."""


class QdrantVectorAdapter:
    resource_type = "qdrant"

    def __init__(
        self,
        config: DataResourceConfig,
        *,
        client_factory: Callable[[DataResourceConfig], Any] | None = None,
        models_provider: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._models_provider = models_provider
        self._client: Any = _CLIENT_UNSET
        self._models: Any = _MODELS_UNSET
        self._lock = threading.RLock()
        self._collection_initialized = False
        self.closed = False

    def search_vectors(
        self,
        vector: list[float],
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
        options: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        options = dict(options or {})
        normalized_vector = _normalize_vector(vector)
        self._validate_dimension(normalized_vector)
        self._ensure_collection(len(normalized_vector))
        query_filter = self._filter(filters)
        kwargs: dict[str, Any] = {
            "collection_name": self.collection_name(),
            "query": normalized_vector,
            "limit": max(0, int(limit)),
            "with_payload": options.pop("with_payload", True),
            "with_vectors": options.pop("with_vectors", False),
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        for key in ("score_threshold", "using", "search_params", "params", "consistency", "shard_key_selector"):
            if key in options:
                kwargs[key] = options[key]
        try:
            result = self._client_instance().query_points(**kwargs)
        except (QdrantClientMissingError, QdrantDimensionError, QdrantFilterError):
            raise
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        points = getattr(result, "points", result)
        return [_hit_from_point(point) for point in points]

    def upsert_vectors(self, items: list[Mapping[str, Any]], options: Mapping[str, Any] | None = None) -> WriteResult:
        options = dict(options or {})
        models = self._models_instance()
        try:
            normalized_items = []
            for item in items:
                vector = _normalize_vector(item["vector"])
                self._validate_dimension(vector)
                normalized_items.append((item, vector))
            points = [
                models.PointStruct(
                    id=item["id"],
                    vector=vector,
                    payload=dict(item.get("payload", {}) or {}),
                )
                for item, vector in normalized_items
            ]
        except KeyError as exc:
            raise QdrantDimensionError(f"Vector item requires {exc.args[0]}") from exc
        if normalized_items:
            self._ensure_collection(len(normalized_items[0][1]))
        kwargs: dict[str, Any] = {"collection_name": self.collection_name(), "points": points}
        if "wait" in options:
            kwargs["wait"] = bool(options["wait"])
        try:
            self._client_instance().upsert(**kwargs)
        except (QdrantClientMissingError, QdrantDimensionError, QdrantFilterError):
            raise
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        return WriteResult(written=len(items), matched=len(items))

    def delete_vectors(
        self,
        ids: list[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> WriteResult:
        options = dict(options or {})
        if ids is None and not filters:
            return WriteResult()
        models = self._models_instance()
        self._ensure_collection()
        if ids is not None:
            selector = models.PointIdsList(points=list(ids))
            deleted = len(ids)
        else:
            selector = models.FilterSelector(filter=self._filter(filters))
            deleted = 0
        kwargs: dict[str, Any] = {"collection_name": self.collection_name(), "points_selector": selector}
        if "wait" in options:
            kwargs["wait"] = bool(options["wait"])
        try:
            self._client_instance().delete(**kwargs)
        except (QdrantClientMissingError, QdrantDimensionError, QdrantFilterError):
            raise
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        return WriteResult(deleted=deleted, matched=deleted)

    def inspect(self) -> dict[str, Any]:
        return asdict(
            InspectResult(
                name=self.config.name,
                type=self.config.type,
                capabilities=[],
                initialized=self._client is not _CLIENT_UNSET,
                status="ok",
                options=self.config.safe_options(),
                details={"backend": "qdrant", "collection": self.collection_name()},
            )
        )

    def health(self) -> HealthResult:
        try:
            self._ensure_collection()
            client = self._client_instance()
            collection = self.collection_name()
            exists = self._collection_initialized
            if not exists:
                if hasattr(client, "collection_exists"):
                    exists = bool(client.collection_exists(collection_name=collection))
                else:
                    client.get_collection(collection_name=collection)
                    exists = True
        except DataConfigurationError as exc:
            return HealthResult(status="failed", code="configuration_error", message=str(exc), details={"collection": self.collection_name()})
        except DataSchemaMismatchError as exc:
            return HealthResult(status="failed", code="schema_mismatch", message=str(exc), details={"collection": self.collection_name()})
        except Exception as exc:
            return HealthResult(status="failed", code="connection_failed", message=self._safe_error(exc), details={"collection": self.collection_name()})
        if not exists:
            return HealthResult(
                status="failed",
                code="resource_missing",
                message=f"Qdrant collection '{self.collection_name()}' is not available",
                details={"collection": self.collection_name()},
            )
        return HealthResult(status="ok", code="ok", message="Qdrant collection is available", details={"collection": self.collection_name()})

    def close(self) -> None:
        if self._client is _CLIENT_UNSET:
            self.closed = True
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self.closed = True

    def native_client(self):
        return self._client_instance()

    def collection_name(self) -> str:
        try:
            return str(self.config.options["collection"])
        except KeyError as exc:
            raise QdrantConfigError("Qdrant resource requires collection") from exc

    def vector_size(self) -> int | None:
        value = self.config.options.get("vector_size")
        if value is None:
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise QdrantConfigError("Qdrant vector_size must be a positive integer") from exc
        if normalized <= 0:
            raise QdrantConfigError("Qdrant vector_size must be a positive integer")
        return normalized

    def distance(self) -> str:
        value = str(self.config.options.get("distance", "cosine")).lower()
        if value not in {"cosine", "dot", "euclid"}:
            raise QdrantConfigError("Qdrant distance must be cosine, dot or euclid")
        return value

    def payload_indexes(self) -> list[str]:
        configured = self.config.options.get(
            "payload_indexes",
            ["workspace_uid", "resource_type", "section_uid", "category", "status"],
        )
        if not isinstance(configured, (list, tuple, set)):
            raise QdrantConfigError("Qdrant payload_indexes must be a list")
        return [str(field) for field in configured]

    def _validate_dimension(self, vector: list[float]) -> None:
        expected = self.vector_size()
        if expected is not None and len(vector) != expected:
            raise QdrantDimensionError(
                f"Qdrant vector dimension mismatch: expected {expected}, got {len(vector)}"
            )

    def _ensure_collection(self, vector_size: int | None = None) -> None:
        if self._collection_initialized:
            return
        client = self._client_instance()
        collection = self.collection_name()
        try:
            if hasattr(client, "collection_exists"):
                exists = bool(client.collection_exists(collection_name=collection))
            else:
                client.get_collection(collection_name=collection)
                exists = True
            if not exists:
                size = self.vector_size() or vector_size
                if size is None:
                    raise QdrantConfigError("Qdrant vector_size is required to create a collection")
                models = self._models_instance()
                distance_name = self.distance().upper()
                distance = getattr(getattr(models, "Distance", None), distance_name, distance_name.title())
                client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=size, distance=distance),
                )
            else:
                self._validate_existing_collection(client)
            self._ensure_payload_indexes(client)
            self._collection_initialized = True
        except (QdrantConfigError, QdrantDimensionError):
            raise
        except Exception as exc:
            _raise_qdrant_operation_error(exc)

    def _ensure_payload_indexes(self, client: Any) -> None:
        create = getattr(client, "create_payload_index", None)
        if not callable(create):
            return
        models = self._models_instance()
        schema_type = getattr(getattr(models, "PayloadSchemaType", None), "KEYWORD", "keyword")
        for field in self.payload_indexes():
            try:
                create(
                    collection_name=self.collection_name(),
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise

    def _validate_existing_collection(self, client: Any) -> None:
        expected = self.vector_size()
        get_collection = getattr(client, "get_collection", None)
        if expected is None or not callable(get_collection):
            return
        info = get_collection(collection_name=self.collection_name())
        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        remote_size = getattr(vectors, "size", None)
        if remote_size is not None and int(remote_size) != expected:
            raise QdrantDimensionError(
                f"Qdrant collection dimension mismatch: expected {expected}, got {remote_size}"
            )

    def _client_instance(self):
        if self._client is _CLIENT_UNSET:
            with self._lock:
                if self._client is _CLIENT_UNSET:
                    client = self._client_factory(self.config) if self._client_factory else _default_qdrant_client(self.config)
                    if client is None:
                        raise QdrantClientMissingError("Qdrant client is not available")
                    self._client = client
        return self._client

    def _models_instance(self):
        if self._models is _MODELS_UNSET:
            with self._lock:
                if self._models is _MODELS_UNSET:
                    self._models = self._models_provider() if self._models_provider else _default_qdrant_models()
        return self._models

    def _filter(self, filters: Mapping[str, Any] | None):
        if not filters:
            return None
        return qdrant_filter_from_mapping(filters, models=self._models_instance())

    def _safe_error(self, exc: Exception) -> str:
        message = str(exc)
        try:
            options = self.config.resolved_options()
        except DataConfigurationError:
            options = self.config.options
        for key, value in options.items():
            if key in {"url", "api_key", "token", "password"} and value:
                message = message.replace(str(value), "***")
        return message


class QdrantVectorFactory:
    resource_type = "qdrant"

    def __init__(
        self,
        *,
        client_factory: Callable[[DataResourceConfig], Any] | None = None,
        models_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._models_provider = models_provider

    def capabilities(self, config: DataResourceConfig) -> set[DataCapability]:
        native = {DataCapability.NATIVE_CLIENT} if bool(config.options.get("native_client")) else set()
        return {DataCapability.VECTOR_SEARCH, DataCapability.VECTOR_WRITE, DataCapability.HEALTHCHECK} | native

    def create(self, config: DataResourceConfig) -> QdrantVectorAdapter:
        return QdrantVectorAdapter(
            config,
            client_factory=self._client_factory,
            models_provider=self._models_provider,
        )


def qdrant_filter_from_mapping(filters: Mapping[str, Any], *, models=None):
    models = models or _default_qdrant_models()
    must: list[Any] = []
    should: list[Any] = []
    must_not: list[Any] = []
    for key in sorted(filters, key=str):
        value = filters[key]
        if key == "$and":
            must.extend(_logical_conditions(value, models=models))
        elif key == "$or":
            should.extend(_logical_conditions(value, models=models))
        elif key == "$not":
            must_not.extend(_logical_conditions(value, models=models))
        else:
            must.append(_field_condition(str(key), value, models=models))
    return models.Filter(must=must, should=should, must_not=must_not)


def _logical_conditions(value: Any, *, models) -> list[Any]:
    if isinstance(value, Mapping):
        return _conditions_from_mapping(value, models=models)
    if isinstance(value, list):
        conditions: list[Any] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise QdrantFilterError("Qdrant logical filter items must be mappings")
            conditions.extend(_conditions_from_mapping(item, models=models))
        return conditions
    raise QdrantFilterError("Qdrant logical filters must be mappings or lists of mappings")


def _conditions_from_mapping(filters: Mapping[str, Any], *, models) -> list[Any]:
    conditions: list[Any] = []
    for key in sorted(filters, key=str):
        value = filters[key]
        if key in {"$and", "$or", "$not"}:
            conditions.append(qdrant_filter_from_mapping({key: value}, models=models))
        else:
            conditions.append(_field_condition(str(key), value, models=models))
    return conditions


def _field_condition(key: str, value: Any, *, models):
    if isinstance(value, Mapping):
        operators = set(value)
        if operators <= _RANGE_OPERATORS:
            return models.FieldCondition(key=key, range=models.Range(**dict(value)))
        if operators == {"eq"}:
            return models.FieldCondition(key=key, match=models.MatchValue(value=value["eq"]))
        if operators == {"any"}:
            return models.FieldCondition(key=key, match=models.MatchAny(any=list(value["any"])))
        unsupported = ", ".join(sorted(str(operator) for operator in operators))
        raise QdrantFilterError(f"Unsupported Qdrant filter operator for '{key}': {unsupported}")
    if isinstance(value, (list, tuple, set)):
        return models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _default_qdrant_client(config: DataResourceConfig):
    try:
        client_module = importlib.import_module("qdrant_client")
    except Exception as exc:
        raise QdrantClientMissingError("Install qdrant-client or muscles-data-qdrant to use type=qdrant") from exc
    options = config.resolved_options()
    kwargs = {
        "url": options["url"],
        "api_key": options.get("api_key"),
        "timeout": options.get("timeout"),
        "prefer_grpc": options.get("prefer_grpc"),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        return client_module.QdrantClient(**kwargs)
    except Exception as exc:
        raise QdrantConnectionError(str(exc)) from exc


def _default_qdrant_models():
    try:
        return importlib.import_module("qdrant_client.models")
    except Exception as exc:
        raise QdrantClientMissingError("Install qdrant-client or muscles-data-qdrant to use type=qdrant") from exc


def _normalize_vector(vector: Any) -> list[float]:
    try:
        normalized = [float(value) for value in vector]
    except TypeError as exc:
        raise QdrantDimensionError("Vector must be an iterable of numbers") from exc
    if not normalized:
        raise QdrantDimensionError("Vector must not be empty")
    return normalized


def _hit_from_point(point: Any) -> VectorHit:
    payload = dict(getattr(point, "payload", {}) or {})
    metadata = {"backend": "qdrant"}
    version = getattr(point, "version", None)
    if version is not None:
        metadata["version"] = version
    return VectorHit(
        id=str(getattr(point, "id")),
        score=float(getattr(point, "score", 0.0)),
        payload=payload,
        metadata=metadata,
        text=payload.get("text"),
        title=payload.get("title"),
    )


def _raise_qdrant_operation_error(exc: Exception):
    message = str(exc)
    lowered = message.lower()
    if "dimension" in lowered or "vector size" in lowered or "vector dimension" in lowered:
        raise QdrantDimensionError(message) from exc
    raise QdrantConnectionError(message) from exc
