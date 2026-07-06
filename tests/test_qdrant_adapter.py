from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from muscles_data.catalog import DataAdapterCatalog
from muscles_data.config import DataConfig
from muscles_data.models import DataCapability
from muscles_data.ports import VectorSearchPort
from muscles_data.runtime import DataRuntime

from muscles_data_qdrant import (
    QdrantClientMissingError,
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
        self.closed = False

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return FakeQueryResult([
            FakePoint("doc-1", 0.91, {"section": "docs"}),
            FakePoint("doc-2", 0.42, {"section": "other"}),
        ])

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

    def close(self) -> None:
        self.closed = True


class FakeModels:
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
                    "native_client": True,
                }
            }
        }
    }


def _runtime(client: FakeQdrantClient | None):
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(QdrantVectorFactory(client_factory=lambda _config: client, models_provider=lambda: FakeModels))
    return DataRuntime(config=DataConfig.from_raw(_config()), catalog=catalog)


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
    assert client.upserts[0]["points"][0].payload == {"section": "docs"}
    assert deleted.deleted == 1
    assert client.deletes[0]["points_selector"].points == ["doc-1"]
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
    with pytest.raises(QdrantClientMissingError):
        _runtime(None).require_port("vector.qdrant", VectorSearchPort).search_vectors([1.0])

    failing = _runtime(FakeQdrantClient(fail_health=True)).doctor()
    assert failing["status"] == "failed"
    assert "qdrant-secret" not in repr(failing)

    bad_client = FakeQdrantClient()
    bad_client.query_points = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("network unavailable"))
    with pytest.raises(QdrantConnectionError):
        _runtime(bad_client).require_port("vector.qdrant", VectorSearchPort).search_vectors([1.0])
