from __future__ import annotations

import importlib
import hashlib
import inspect
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Mapping
from urllib.parse import urlsplit

from muscles.data.config import DataResourceConfig
from muscles.data.errors import (
    DataConfigurationError,
    DataConnectionError,
    DataError,
    DataSchemaMismatchError,
    DataUnsupportedOperationError,
    DataVectorDimensionError,
    DataWriteOutcomeUnknownError,
)
from muscles.data.models import DataCapability, HealthResult, InspectResult, VectorHit, WriteResult


_RANGE_OPERATORS = {"gt", "gte", "lt", "lte"}
_CLIENT_UNSET = object()
_MODELS_UNSET = object()
_PUBLIC_ID_FIELD = "_muscles_id"
_POLICY_TOKEN_FIELD = "_muscles_policy_operation"
_POLICY_DELETED_FIELD = "_muscles_policy_deleted"


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
        operation_journal: Any = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._models_provider = models_provider
        self._operation_journal = operation_journal
        self._journal_scope: str | None = None
        self._client: Any = _CLIENT_UNSET
        self._models: Any = _MODELS_UNSET
        self._lock = threading.RLock()
        self._collection_initialized = False
        self.closed = False

    def _operation_scope(self) -> str:
        if self._journal_scope is not None:
            return self._journal_scope
        options = self.config.resolved_options()
        url = str(options.get("url", ""))
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.hostname:
            raise QdrantConfigError("Operation recovery requires a configured Qdrant URL")
        namespace = options.get("policy_journal_scope", "")
        if not isinstance(namespace, str):
            raise QdrantConfigError("policy_journal_scope must be a string")
        identity = repr((parsed.scheme.lower(), parsed.hostname.lower(), parsed.port, parsed.path, namespace))
        self._journal_scope = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self._journal_scope

    def policy_get(self, document_id: str) -> Mapping[str, Any] | None:
        """Read current payload for an explicitly enabled non-atomic precheck."""
        self._ensure_collection()
        try:
            points = self._client_instance().retrieve(
                collection_name=self.collection_name(),
                ids=[_qdrant_point_id(str(document_id))],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        if not points:
            return None
        payload = dict(points[0].payload or {})
        if _POLICY_DELETED_FIELD in payload:
            return None
        payload.pop(_POLICY_TOKEN_FIELD, None)
        return payload

    def policy_atomic_supported(self) -> bool:
        """Require an explicit managed-writer invariant and Qdrant 1.17+."""
        if self.config.options.get("policy_atomic_managed") is not True:
            return False
        client = self._client_instance()
        if not callable(getattr(client, "info", None)):
            return False
        info = client.info()
        version = str(getattr(info, "version", ""))
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
        if not match or tuple(map(int, match.groups())) < (1, 17, 0):
            return False
        parameters = inspect.signature(client.upsert).parameters
        models = self._models_instance()
        return ("update_mode" in parameters and "update_filter" in parameters
                and hasattr(models, "UpdateMode") and hasattr(models, "WriteOrdering"))

    def policy_get_atomic(self, document_id: str):
        self._ensure_collection()
        try:
            points = self._client_instance().retrieve(
                collection_name=self.collection_name(), ids=[_qdrant_point_id(str(document_id))],
                with_payload=True, with_vectors=False, consistency="all",
            )
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        if not points:
            return None
        payload = dict(points[0].payload or {})
        if _POLICY_DELETED_FIELD in payload:
            raise DataUnsupportedOperationError("A protected deletion tombstone cannot be overwritten")
        marker = payload.pop(_POLICY_TOKEN_FIELD, None)
        if not isinstance(marker, str) or not re.fullmatch(r"[0-9a-f]{32}", marker):
            raise DataUnsupportedOperationError("Existing point has no managed policy version")
        return payload, marker

    def policy_atomic_upsert(self, item: Mapping[str, Any], expected_marker: str | None,
                             expected_fields: Mapping[str, Any]) -> None:
        if expected_marker is not None and (not isinstance(expected_marker, str) or not re.fullmatch(r"[0-9a-f]{32}", expected_marker)):
            raise ValueError("Invalid expected policy version")
        models = self._models_instance()
        document_id = str(item["id"])
        vector = _normalize_vector(item["vector"])
        self._validate_dimension(vector)
        payload = _payload_with_public_id(item)
        marker = uuid.uuid4().hex
        payload[_POLICY_TOKEN_FIELD] = marker
        self._ensure_collection(len(vector))
        point = models.PointStruct(id=_qdrant_point_id(document_id), vector=vector, payload=payload)
        mode = models.UpdateMode.INSERT_ONLY if expected_marker is None else models.UpdateMode.UPDATE_ONLY
        clauses = self._policy_atomic_conditions(expected_marker, expected_fields)
        condition = None if expected_marker is None else models.Filter(must=clauses)
        if self._operation_journal is not None:
            self._operation_journal.prepare(self.config.name, self.collection_name(), document_id, marker, scope=self._operation_scope())
        try:
            self._client_instance().upsert(
                collection_name=self.collection_name(), points=[point], wait=True,
                ordering=models.WriteOrdering.STRONG, update_mode=mode, update_filter=condition,
            )
        except Exception as exc:
            raise DataWriteOutcomeUnknownError(document_id=document_id, operation_id=marker) from exc
        if self.policy_confirm_upsert(document_id, marker) is not True:
            raise DataWriteOutcomeUnknownError(document_id=document_id, operation_id=marker)

    def _policy_atomic_conditions(self, expected_marker: str | None, expected_fields: Mapping[str, Any]):
        models = self._models_instance()
        clauses = [
            models.FieldCondition(key=_POLICY_TOKEN_FIELD, match=models.MatchValue(value=expected_marker)),
            models.FieldCondition(key=_POLICY_TOKEN_FIELD, values_count=models.ValuesCount(gte=1, lte=1)),
        ] if expected_marker else []
        for field, value in expected_fields.items():
            if value is None:
                clauses.append(models.IsNullCondition(is_null=models.PayloadField(key=field)))
            elif type(value) in (str, int, bool):
                clauses.extend([
                    models.FieldCondition(key=field, match=models.MatchValue(value=value)),
                    models.FieldCondition(key=field, values_count=models.ValuesCount(gte=1, lte=1)),
                ])
            else:
                raise DataUnsupportedOperationError("Qdrant cannot atomically verify this policy field type")
        return clauses

    def policy_atomic_delete(self, document_id: str, expected_marker: str, expected_fields: Mapping[str, Any]) -> None:
        if not isinstance(expected_marker, str) or not re.fullmatch(r"[0-9a-f]{32}", expected_marker):
            raise ValueError("Invalid expected policy version")
        models = self._models_instance()
        marker = uuid.uuid4().hex
        clauses = self._policy_atomic_conditions(expected_marker, expected_fields)
        clauses.append(models.HasIdCondition(has_id=[_qdrant_point_id(str(document_id))]))
        selector = models.Filter(must=clauses, must_not=[models.FieldCondition(
            key=_POLICY_DELETED_FIELD, match=models.MatchValue(value=True),
        )])
        if self._operation_journal is not None:
            self._operation_journal.prepare(self.config.name, self.collection_name(), str(document_id), marker, operation_kind="delete", scope=self._operation_scope())
        try:
            self._client_instance().set_payload(
                collection_name=self.collection_name(), points=selector, wait=True,
                ordering=models.WriteOrdering.STRONG,
                payload={_POLICY_TOKEN_FIELD: marker, _POLICY_DELETED_FIELD: True},
            )
        except Exception as exc:
            raise DataWriteOutcomeUnknownError(document_id=str(document_id), operation_id=marker, operation_kind="delete") from exc
        if self.policy_confirm_delete(str(document_id), marker) is not True:
            raise DataWriteOutcomeUnknownError(document_id=str(document_id), operation_id=marker, operation_kind="delete")

    def policy_adopt_existing(self, document_id: str, expected_fields: Mapping[str, Any]) -> None:
        """Attach a marker conditionally during an externally enforced maintenance window."""
        if not self.policy_atomic_supported():
            raise DataUnsupportedOperationError("Backend does not support managed adoption")
        try:
            existing = self._client_instance().retrieve(
                collection_name=self.collection_name(), ids=[_qdrant_point_id(str(document_id))],
                with_payload=True, with_vectors=False, consistency="all",
            )
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        if not existing or _POLICY_TOKEN_FIELD in (existing[0].payload or {}) or _POLICY_DELETED_FIELD in (existing[0].payload or {}):
            raise DataUnsupportedOperationError("Point is missing, already managed, or carries a reserved marker")
        models = self._models_instance()
        marker = uuid.uuid4().hex
        clauses = self._policy_atomic_conditions(None, expected_fields)
        clauses.extend([
            models.HasIdCondition(has_id=[_qdrant_point_id(str(document_id))]),
            models.IsEmptyCondition(is_empty=models.PayloadField(key=_POLICY_TOKEN_FIELD)),
        ])
        selector = models.Filter(must=clauses, must_not=[models.FieldCondition(
            key=_POLICY_DELETED_FIELD, match=models.MatchValue(value=True),
        )])
        if self._operation_journal is not None:
            self._operation_journal.prepare(self.config.name, self.collection_name(), str(document_id), marker, operation_kind="adopt", scope=self._operation_scope())
        try:
            self._client_instance().set_payload(
                collection_name=self.collection_name(), points=selector, wait=True,
                ordering=models.WriteOrdering.STRONG, payload={_POLICY_TOKEN_FIELD: marker},
            )
        except Exception as exc:
            raise DataWriteOutcomeUnknownError(document_id=str(document_id), operation_id=marker, operation_kind="adopt") from exc
        if self.policy_confirm_upsert(str(document_id), marker) is not True:
            raise DataWriteOutcomeUnknownError(document_id=str(document_id), operation_id=marker, operation_kind="adopt")

    def policy_confirm_upsert(self, document_id: str, operation_id: str) -> bool | None:
        """Return True on matching marker; None means the outcome remains unknown."""
        return self._policy_confirm(document_id, operation_id, deleted=False)

    def policy_confirm_delete(self, document_id: str, operation_id: str) -> bool | None:
        return self._policy_confirm(document_id, operation_id, deleted=True)

    def _policy_confirm(self, document_id: str, operation_id: str, *, deleted: bool) -> bool | None:
        if not isinstance(operation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", operation_id):
            raise ValueError("operation_id must be a lowercase UUID token")
        try:
            records = self._client_instance().retrieve(
                collection_name=self.collection_name(), ids=[_qdrant_point_id(str(document_id))],
                with_payload=True, with_vectors=False, consistency="all",
            )
        except Exception:
            return None
        if not records or (records[0].payload or {}).get(_POLICY_TOKEN_FIELD) != operation_id:
            return None
        payload = records[0].payload or {}
        if deleted and payload.get(_POLICY_DELETED_FIELD) is not True:
            return None
        if not deleted and _POLICY_DELETED_FIELD in payload:
            return None
        if self._operation_journal is not None:
            try:
                self._operation_journal.confirm(operation_id, scope=self._operation_scope())
            except KeyError:
                pass
            except Exception:
                return None
        return True

    def policy_reconcile_pending(self) -> list[dict[str, Any]]:
        if self._operation_journal is None:
            raise DataUnsupportedOperationError("Operation journal is not configured")
        result = []
        for item in self._operation_journal.pending(self.config.name, scope=self._operation_scope()):
            if item["collection"] != self.collection_name():
                continue
            confirm = self.policy_confirm_delete if item["operation_kind"] == "delete" else self.policy_confirm_upsert
            confirmed = confirm(item["document_id"], item["operation_id"])
            result.append({**item, "confirmed": confirmed})
        return result

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
        if self.config.options.get("policy_atomic_managed") is True:
            if options.get("with_payload", True) is not True:
                raise QdrantConfigError("Managed policy search requires full payload")
            if options.get("consistency", "all") != "all":
                raise QdrantConfigError("Managed policy search requires all-replica consistency")
            options["consistency"] = "all"
            filters = {"$and": [dict(filters or {}), {"$not": {_POLICY_DELETED_FIELD: True}}]}
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
        for key in ("score_threshold", "using", "search_params", "params", "consistency", "shard_key_selector", "offset"):
            if key in options:
                kwargs[key] = options[key]
        try:
            result = self._client_instance().query_points(**kwargs)
        except (QdrantClientMissingError, QdrantDimensionError, QdrantFilterError):
            raise
        except Exception as exc:
            _raise_qdrant_operation_error(exc)
        points = getattr(result, "points", result)
        if self.config.options.get("policy_atomic_managed") is True:
            points = [point for point in points if _POLICY_DELETED_FIELD not in (getattr(point, "payload", None) or {})]
        return [_hit_from_point(point) for point in points]

    def upsert_vectors(self, items: list[Mapping[str, Any]], options: Mapping[str, Any] | None = None) -> WriteResult:
        if self.config.options.get("policy_atomic_managed") is True:
            raise DataUnsupportedOperationError("Managed Qdrant writes require a policy-aware vector port")
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
                    id=_qdrant_point_id(str(item["id"])),
                    vector=vector,
                    payload=_payload_with_public_id(item),
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
        if self.config.options.get("policy_atomic_managed") is True:
            raise DataUnsupportedOperationError("Managed Qdrant writes require a policy-aware vector port")
        options = dict(options or {})
        if ids is None and not filters:
            return WriteResult()
        models = self._models_instance()
        self._ensure_collection()
        if ids is not None:
            selector = models.PointIdsList(points=[_qdrant_point_id(str(item_id)) for item_id in ids])
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
        if self.config.options.get("policy_atomic_managed") is True:
            raise DataUnsupportedOperationError("Managed Qdrant resources do not expose the native client")
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
        operation_journal: Any = None,
    ) -> None:
        self._client_factory = client_factory
        self._models_provider = models_provider
        self._operation_journal = operation_journal

    def capabilities(self, config: DataResourceConfig) -> set[DataCapability]:
        native = {DataCapability.NATIVE_CLIENT} if bool(config.options.get("native_client")) and config.options.get("policy_atomic_managed") is not True else set()
        return {DataCapability.VECTOR_SEARCH, DataCapability.VECTOR_WRITE, DataCapability.HEALTHCHECK} | native

    def create(self, config: DataResourceConfig) -> QdrantVectorAdapter:
        return QdrantVectorAdapter(
            config,
            client_factory=self._client_factory,
            models_provider=self._models_provider,
            operation_journal=self._operation_journal,
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
    return models.Filter(must=must or None, should=should or None, must_not=must_not or None)


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
    payload.pop(_POLICY_TOKEN_FIELD, None)
    payload.pop(_POLICY_DELETED_FIELD, None)
    public_id = str(payload.pop(_PUBLIC_ID_FIELD, getattr(point, "id")))
    metadata = {"backend": "qdrant"}
    version = getattr(point, "version", None)
    if version is not None:
        metadata["version"] = version
    return VectorHit(
        id=public_id,
        score=float(getattr(point, "score", 0.0)),
        payload=payload,
        metadata=metadata,
        text=payload.get("text"),
        title=payload.get("title"),
    )


def _payload_with_public_id(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item.get("payload", {}) or {})
    if _POLICY_TOKEN_FIELD in payload:
        raise QdrantConfigError(f"Qdrant payload field '{_POLICY_TOKEN_FIELD}' is reserved")
    if _POLICY_DELETED_FIELD in payload:
        raise QdrantConfigError(f"Qdrant payload field '{_POLICY_DELETED_FIELD}' is reserved")
    item_id = str(item["id"])
    existing_id = payload.get(_PUBLIC_ID_FIELD)
    if existing_id is not None and str(existing_id) != item_id:
        raise QdrantConfigError(f"Qdrant payload field '{_PUBLIC_ID_FIELD}' is reserved")
    payload[_PUBLIC_ID_FIELD] = item_id
    return payload


def _qdrant_point_id(value: str) -> str | int:
    """Map the string port id to a Qdrant-supported stable point id."""
    try:
        return int(value) if value.isdigit() else str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"muscles-data:qdrant:{value}"))


def _raise_qdrant_operation_error(exc: Exception):
    message = str(exc)
    lowered = message.lower()
    if "dimension" in lowered or "vector size" in lowered or "vector dimension" in lowered:
        raise QdrantDimensionError(message) from exc
    raise QdrantConnectionError(message) from exc
