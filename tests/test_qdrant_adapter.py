from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from muscles.data.catalog import DataAdapterCatalog
from muscles.data.config import DataConfig
from muscles.data.models import DataCapability
from muscles.data.ports import VectorSearchPort
from muscles.data.runtime import DataRuntime
from muscles.data.errors import DataPolicyDeniedError, DataUnsupportedOperationError
from muscles.data.policies import DataNonAtomicPolicyMode, DataOperationContext, DataPolicyDecision, DataPolicyPipeline, DataPredicate, DataPredicateOperator
from muscles.data.qdrant import PolicyVectorSearch

from muscles.data.qdrant import (
    QdrantClientMissingError,
    QdrantConfigError,
    QdrantConnectionError,
    QdrantDimensionError,
    QdrantFilterError,
    QdrantVectorFactory,
    qdrant_filter_from_mapping,
)


class FakePoint:
    def __init__(self, point_id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = point_id
        self.score = score
        self.payload = payload
        self.version = 7


class FakeQueryResult:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeQdrantClient:
    def __init__(self, *, fail_health: bool = False) -> None:
        self.fail_health = fail_health
        self.queries: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.collection_checks: list[str] = []
        self.collections_created: list[dict[str, Any]] = []
        self.payload_indexes: list[dict[str, Any]] = []
        self.closed = False

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        points = [
            FakePoint("doc-1", 0.91, {"section": "docs"}),
            FakePoint("doc-2", 0.42, {"section": "other"}),
        ]
        offset = kwargs.get("offset", 0)
        return FakeQueryResult(points[offset:offset + kwargs.get("limit", len(points))])

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        return SimpleNamespace(status="completed")

    def delete(self, **kwargs):
        self.deletes.append(kwargs)
        return SimpleNamespace(status="completed")

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_checks.append(collection_name)
        if self.fail_health:
            raise TimeoutError("qdrant api_key=qdrant-secret timed out")
        return collection_name == "docs"

    def create_collection(self, **kwargs):
        self.collections_created.append(kwargs)

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)

    def close(self) -> None:
        self.closed = True


class FakeModels:
    class Distance:
        COSINE = "Cosine"
        DOT = "Dot"
        EUCLID = "Euclid"

    class PayloadSchemaType:
        KEYWORD = "keyword"

    class VectorParams:
        def __init__(self, *, size: int, distance) -> None:
            self.size = size
            self.distance = distance

    class MatchValue:
        def __init__(self, value) -> None:
            self.value = value

    class MatchAny:
        def __init__(self, any) -> None:
            self.any = any

    class Range:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    class FieldCondition:
        def __init__(self, *, key: str, match=None, range=None) -> None:
            self.key = key
            self.match = match
            self.range = range

    class Filter:
        def __init__(self, *, must=None, should=None, must_not=None) -> None:
            self.must = must or []
            self.should = should or []
            self.must_not = must_not or []

    class PointStruct:
        def __init__(self, *, id, vector, payload=None) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload or {}

    class PointIdsList:
        def __init__(self, *, points) -> None:
            self.points = points

    class FilterSelector:
        def __init__(self, *, filter) -> None:
            self.filter = filter


def _config() -> dict:
    return {
        "data": {
            "resources": {
                "vector.qdrant": {
                    "type": "qdrant",
                    "url": "https://qdrant.example",
                    "api_key": "qdrant-secret",
                    "collection": "docs",
                    "vector_size": 2,
                    "distance": "cosine",
                    "native_client": True,
                }
            }
        }
    }


def _runtime(client: FakeQdrantClient | None):
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(QdrantVectorFactory(client_factory=lambda _config: client, models_provider=lambda: FakeModels))
    return DataRuntime(config=DataConfig.from_raw(_config()), catalog=catalog)


def test_policy_vector_search_filters_before_limit_and_checks_payload():
    class Policy:
        def evaluate(self, request):
            return DataPolicyDecision(predicates=(DataPredicate("section", DataPredicateOperator.EQ, request.context.attributes["section"]),))

    client = FakeQdrantClient()
    runtime = _runtime(client)
    store = PolicyVectorSearch(runtime.require_port("vector.qdrant", VectorSearchPort), "vector.qdrant", policies=DataPolicyPipeline((Policy(),)), context=DataOperationContext(attributes={"section": "docs"}))
    assert [hit.id for hit in store.search_vectors([0.1, 0.2], limit=2)] == ["doc-1"]
    request = client.queries[0]
    assert request["limit"] == store.max_scan
    assert request["with_payload"] is True
    assert any(condition.key == "section" and condition.match.value == "docs" for condition in request["query_filter"].must)


def test_atomic_policy_upsert_uses_version_and_checks_replacement():
    class Policy:
        def evaluate(self, request):
            return DataPolicyDecision(predicates=(DataPredicate("section", DataPredicateOperator.EQ, "docs"),))

    class AtomicBackend:
        def __init__(self):
            self.writes = []

        def policy_atomic_supported(self):
            return True

        def policy_get_atomic(self, document_id):
            return ({"section": "docs"}, "previous") if document_id == "1" else None

        def policy_atomic_upsert(self, item, marker, expected):
            self.writes.append((item["id"], marker, expected))

    backend = AtomicBackend()
    store = PolicyVectorSearch(backend, "vectors", policies=DataPolicyPipeline((Policy(),)))
    assert store.upsert_vectors([{"id": "1", "payload": {"section": "docs"}}]).written == 1
    assert store.upsert_vectors([{"id": "2", "payload": {"section": "docs"}}]).written == 1
    assert backend.writes == [("1", "previous", {"section": "docs"}), ("2", None, {})]
    with pytest.raises(DataPolicyDeniedError):
        store.upsert_vectors([{"id": "1", "payload": {"section": "other"}}])


def test_atomic_policy_delete_checks_existing_value_and_uses_version():
    class Policy:
        def evaluate(self, request):
            return DataPolicyDecision(predicates=(DataPredicate("section", DataPredicateOperator.EQ, "docs"),))

    class AtomicBackend:
        def __init__(self):
            self.deletes = []

        def policy_atomic_supported(self):
            return True

        def policy_get_atomic(self, document_id):
            return ({"section": "docs"}, "previous") if document_id == "1" else ({"section": "other"}, "other")

        def policy_atomic_delete(self, document_id, marker, expected):
            self.deletes.append((document_id, marker, expected))

    backend = AtomicBackend()
    store = PolicyVectorSearch(backend, "vectors", policies=DataPolicyPipeline((Policy(),)))
    assert store.delete_vectors(ids=["1"]).deleted == 1
    assert backend.deletes == [("1", "previous", {"section": "docs"})]
    with pytest.raises(DataPolicyDeniedError):
        store.delete_vectors(ids=["2"])


def test_adopting_existing_point_requires_maintenance_and_policy():
    class Policy:
        def evaluate(self, request):
            return DataPolicyDecision(predicates=(DataPredicate("section", DataPredicateOperator.EQ, "docs"),))

    class Backend:
        def __init__(self):
            self.calls = []

        def policy_atomic_supported(self):
            return True

        def policy_get(self, document_id):
            return {"section": "docs", "revision": 3}

        def policy_adopt_existing(self, document_id, expected):
            self.calls.append((document_id, expected))

    backend = Backend()
    store = PolicyVectorSearch(backend, "vectors", policies=DataPolicyPipeline((Policy(),)))
    with pytest.raises(DataUnsupportedOperationError):
        store.adopt_existing("1", expected_fields={"revision": 3}, maintenance_confirmed=False)
    store.adopt_existing("1", expected_fields={"revision": 3}, maintenance_confirmed=True)
    assert backend.calls == [("1", {"revision": 3, "section": "docs"})]
    with pytest.raises(DataPolicyDeniedError):
        store.adopt_existing("1", expected_fields={"revision": 4}, maintenance_confirmed=True)


def test_atomic_policy_modes_are_not_enabled_for_qdrant_116():
    client = FakeQdrantClient()
    client.info = lambda: SimpleNamespace(version="1.16.3")
    config = _config()
    config["data"]["resources"]["vector.qdrant"]["policy_atomic_managed"] = True
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(QdrantVectorFactory(client_factory=lambda _: client, models_provider=lambda: FakeModels))
    adapter = DataRuntime(config=DataConfig.from_raw(config), catalog=catalog).require_port("vector.qdrant", VectorSearchPort)
    assert adapter.policy_atomic_supported() is False


def test_atomic_policy_requires_explicit_managed_writer_configuration():
    client = FakeQdrantClient()
    client.info = lambda: SimpleNamespace(version="1.17.1")
    adapter = _runtime(client).require_port("vector.qdrant", VectorSearchPort)
    assert adapter.policy_atomic_supported() is False


def test_managed_resource_blocks_public_unprotected_writes_and_native_access():
    config = _config()
    config["data"]["resources"]["vector.qdrant"]["policy_atomic_managed"] = True
    client = FakeQdrantClient()
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(QdrantVectorFactory(client_factory=lambda _: client, models_provider=lambda: FakeModels))
    runtime = DataRuntime(config=DataConfig.from_raw(config), catalog=catalog)
    adapter = runtime.require_port("vector.qdrant", VectorSearchPort)
    with pytest.raises(DataUnsupportedOperationError):
        adapter.upsert_vectors([{"id": "1", "vector": [1.0, 0.0]}])
    with pytest.raises(DataUnsupportedOperationError):
        adapter.delete_vectors(ids=["1"])
    with pytest.raises(DataUnsupportedOperationError):
        adapter.native_client()
    assert not client.upserts and not client.deletes


def test_policy_operation_marker_is_reserved_and_hidden():
    client = FakeQdrantClient()
    adapter = _runtime(client).require_port("vector.qdrant", VectorSearchPort)
    with pytest.raises(QdrantConfigError, match="reserved"):
        adapter.upsert_vectors([{"id": "x", "vector": [1.0, 0.0], "payload": {"_muscles_policy_operation": "forged"}}])
    from muscles.data.qdrant.adapter import _hit_from_point
    hit = _hit_from_point(FakePoint("x", 1.0, {"_muscles_policy_operation": "secret", "name": "safe"}))
    assert hit.payload == {"name": "safe"}


def test_policy_vector_search_with_local_qdrant():
    qdrant = pytest.importorskip("qdrant_client")
    models = pytest.importorskip("qdrant_client.models")

    class Policy:
        def evaluate(self, request):
            return DataPolicyDecision(predicates=(DataPredicate("section", DataPredicateOperator.EQ, "docs"),))

    client = qdrant.QdrantClient(":memory:")
    client.create_collection("docs", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    client.upsert("docs", points=[
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={"section": "docs"}),
        models.PointStruct(id=2, vector=[0.9, 0.1], payload={"section": ["docs", "other"]}),
        models.PointStruct(id=3, vector=[0.8, 0.2], payload={"section": "docs"}),
    ], wait=True)
    runtime = DataRuntime(config=DataConfig.from_raw(_config()), catalog=_catalog_for_local(client, models))
    try:
        store = PolicyVectorSearch(runtime.require_port("vector.qdrant", VectorSearchPort), "vector.qdrant", policies=DataPolicyPipeline((Policy(),)), max_scan=4)
        assert [hit.id for hit in store.search_vectors([1.0, 0.0], limit=2)] == ["1", "3"]
        # Exercise the explicitly opted-in legacy path independently of the
        # installed local client's reported server version.
        runtime.require_port("vector.qdrant", VectorSearchPort).policy_atomic_supported = lambda: False
        prechecked = PolicyVectorSearch(
            runtime.require_port("vector.qdrant", VectorSearchPort), "vector.qdrant",
            policies=DataPolicyPipeline((Policy(),), non_atomic_mode=DataNonAtomicPolicyMode.ALLOW_PRECHECK),
        )
        with pytest.raises(DataPolicyDeniedError):
            prechecked.upsert_vectors([{"id": "2", "vector": [1.0, 0.0], "payload": {"section": "docs"}}])
        prechecked.upsert_vectors([{"id": "1", "vector": [1.0, 0.0], "payload": {"section": "docs"}}], options=None)
        with pytest.raises(DataPolicyDeniedError):
            prechecked.delete_vectors(ids=["2"])
    finally:
        runtime.close()


def _catalog_for_local(client, models):
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(QdrantVectorFactory(client_factory=lambda _config: client, models_provider=lambda: models))
    return catalog


def test_qdrant_external_adapter_maps_vector_operations_and_native_access():
    client = FakeQdrantClient()
    runtime = _runtime(client)

    listed = runtime.list_resources()[0]
    assert listed["type"] == "qdrant"
    assert {"vector_search", "vector_write"} <= set(listed["capabilities"])
    assert listed["initialized"] is False

    vector = runtime.require_port("vector.qdrant", VectorSearchPort)
    hits = vector.search_vectors([0.1, 0.9], filters={"section": "docs"}, limit=2)
    write = vector.upsert_vectors([{"id": "doc-1", "vector": [0.1, 0.9], "payload": {"section": "docs"}}])
    deleted = vector.delete_vectors(ids=["doc-1"])

    assert [hit.id for hit in hits] == ["doc-1", "doc-2"]
    assert client.queries[0]["query_filter"].must[0].key == "section"
    assert write.written == 1
    assert client.upserts[0]["points"][0].payload == {"section": "docs", "_muscles_id": "doc-1"}
    assert deleted.deleted == 1
    assert client.deletes[0]["points_selector"].points == [str(uuid5(NAMESPACE_URL, "muscles-data:qdrant:doc-1"))]
    assert runtime.require_resource("vector.qdrant", DataCapability.NATIVE_CLIENT).native_client() is client
    assert runtime.doctor()["status"] == "ok"
    assert client.collection_checks == ["docs"]
    assert runtime.close()["status"] == "ok"
    assert client.closed is True


def test_qdrant_external_adapter_filters_dimensions_and_safe_failures():
    translated = qdrant_filter_from_mapping({"score": {"gte": 0.5}, "$not": {"archived": True}}, models=FakeModels)
    assert translated.must[0].key == "score"
    assert translated.must[0].range.values == {"gte": 0.5}
    assert translated.must_not[0].key == "archived"

    with pytest.raises(QdrantFilterError):
        qdrant_filter_from_mapping({"score": {"near": 1.0}}, models=FakeModels)
    with pytest.raises(QdrantDimensionError):
        _runtime(FakeQdrantClient()).require_port("vector.qdrant", VectorSearchPort).search_vectors([])
    with pytest.raises(QdrantDimensionError):
        _runtime(FakeQdrantClient()).require_port("vector.qdrant", VectorSearchPort).search_vectors([1.0])
    with pytest.raises(QdrantClientMissingError):
        _runtime(None).require_port("vector.qdrant", VectorSearchPort).search_vectors([1.0, 0.0])

    failing = _runtime(FakeQdrantClient(fail_health=True)).doctor()
    assert failing["status"] == "failed"
    assert "qdrant-secret" not in repr(failing)

    bad_client = FakeQdrantClient()
    bad_client.query_points = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("network unavailable"))
    with pytest.raises(QdrantConnectionError):
        _runtime(bad_client).require_port("vector.qdrant", VectorSearchPort).search_vectors([1.0, 0.0])


def test_qdrant_creates_collection_indexes_payload_and_resolves_url_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    client = FakeQdrantClient()
    client.collection_exists = lambda collection_name: False
    captured: list[dict] = []
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(
        QdrantVectorFactory(
            client_factory=lambda config: captured.append(config.resolved_options()) or client,
            models_provider=lambda: FakeModels,
        )
    )
    runtime = DataRuntime(
        config=DataConfig.from_raw(
            {
                "data": {
                    "resources": {
                        "vector.qdrant": {
                            "type": "qdrant",
                            "url_env": "QDRANT_URL",
                            "collection": "assetforge_rag",
                            "vector_size": 2,
                            "distance": "cosine",
                            "payload_indexes": ["workspace_uid", "status"],
                        }
                    }
                }
            }
        ),
        catalog=catalog,
    )

    runtime.require_port("vector.qdrant", VectorSearchPort).upsert_vectors(
        [{"id": "doc-1", "vector": [1.0, 0.0], "payload": {"status": "ready"}}]
    )

    assert captured[0]["url"] == "https://qdrant.example"
    assert client.collections_created[0]["collection_name"] == "assetforge_rag"
    assert client.collections_created[0]["vectors_config"].size == 2
    assert {item["field_name"] for item in client.payload_indexes} == {"workspace_uid", "status"}


def test_qdrant_normalizes_arbitrary_string_ids_for_backend_and_keeps_public_id():
    client = FakeQdrantClient()
    runtime = _runtime(client)
    vector = runtime.require_port("vector.qdrant", VectorSearchPort)

    vector.upsert_vectors([{"id": "document-alpha", "vector": [1.0, 0.0], "payload": {"kind": "doc"}}])

    point = client.upserts[-1]["points"][0]
    UUID(str(point.id))
    assert point.payload["_muscles_id"] == "document-alpha"


def test_qdrant_rejects_configured_dimension_mismatch_with_existing_collection():
    client = FakeQdrantClient()
    client.get_collection = lambda **_kwargs: SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3)))
    )
    with pytest.raises(QdrantDimensionError, match="expected 2, got 3"):
        _runtime(client).require_port("vector.qdrant", VectorSearchPort).search_vectors([1.0, 0.0])
