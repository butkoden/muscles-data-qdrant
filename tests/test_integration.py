from __future__ import annotations

import os
from uuid import uuid4

import pytest
from muscles_data.catalog import DataAdapterCatalog
from muscles_data.config import DataConfig
from muscles_data.models import DataCapability
from muscles_data.ports import VectorSearchPort
from muscles_data.runtime import DataRuntime

from muscles_data_qdrant import QdrantVectorFactory


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MUSCLES_DATA_INTEGRATION"), reason="backend integration is disabled"),
]


def test_qdrant_real_collection_vector_lifecycle():
    collection = f"muscles_data_it_{uuid4().hex[:12]}"
    config = DataConfig.from_raw(
        {
            "data": {
                "resources": {
                    "vector.qdrant": {
                        "type": "qdrant",
                        "url_env": "QDRANT_URL",
                        "collection": collection,
                        "vector_size": 3,
                        "distance": "cosine",
                        "payload_indexes": ["status"],
                        "native_client": True,
                    }
                }
            }
        }
    )
    catalog = DataAdapterCatalog.with_defaults()
    catalog.register(QdrantVectorFactory())
    runtime = DataRuntime(config=config, catalog=catalog)

    client = None
    try:
        vector = runtime.require_port("vector.qdrant", VectorSearchPort)
        assert vector.upsert_vectors(
            [
                {"id": "alpha", "vector": [1.0, 0.0, 0.0], "payload": {"status": "ready"}},
                {"id": "beta", "vector": [0.0, 1.0, 0.0], "payload": {"status": "draft"}},
            ],
            options={"wait": True},
        ).written == 2
        hits = vector.search_vectors([1.0, 0.0, 0.0], filters={"status": "ready"}, limit=10)
        assert [hit.id for hit in hits] == ["alpha"]
        assert hits[0].payload["status"] == "ready"
        assert runtime.doctor()["status"] == "ok"
        assert vector.delete_vectors(ids=["alpha"], options={"wait": True}).deleted == 1
        contracts = pytest.importorskip("muscles_data.contracts")
        contract = getattr(contracts, "assert_vector_search_contract", None)
        if contract is not None:
            contract(lambda: vector, dimension=3)
        client = runtime.require_resource("vector.qdrant", DataCapability.NATIVE_CLIENT).native_client()
        client.delete_collection(collection_name=collection)
    finally:
        if client is not None:
            try:
                client.delete_collection(collection_name=collection)
            except Exception:
                pass
        runtime.close()
