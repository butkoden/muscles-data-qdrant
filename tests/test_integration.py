from __future__ import annotations

import os
import inspect
from uuid import uuid4

import pytest
from muscles.data.catalog import DataAdapterCatalog
from muscles.data.config import DataConfig
from muscles.data.models import DataCapability
from muscles.data.ports import VectorSearchPort
from muscles.data.runtime import DataRuntime
from muscles.data.operation_journal import SqliteOperationJournal
from muscles.data.errors import DataUnsupportedOperationError, DataWriteOutcomeUnknownError
from muscles.data.policies import DataPolicyDecision, DataPolicyPipeline, DataPredicate, DataPredicateOperator
from muscles.data.policy_search import PolicyVectorSearch

from muscles.data.qdrant import QdrantVectorFactory


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MUSCLES_DATA_INTEGRATION"), reason="backend integration is disabled"),
]


def test_qdrant_atomic_upsert_and_confirmation(tmp_path):
    models = pytest.importorskip("qdrant_client.models")
    collection = f"muscles_data_policy_it_{uuid4().hex[:12]}"
    config = DataConfig.from_raw({"resources": {"v": {
        "type": "qdrant", "url_env": "QDRANT_URL", "collection": collection, "vector_size": 2,
        "policy_atomic_managed": True,
    }}})
    catalog = DataAdapterCatalog.with_defaults()
    journal_path = tmp_path / "operations.sqlite"
    journal = SqliteOperationJournal(journal_path)
    catalog.register(QdrantVectorFactory(operation_journal=journal))
    runtime = DataRuntime(config=config, catalog=catalog)
    adapter = runtime.require_port("v", VectorSearchPort)
    client = None

    class Policy:
        def evaluate(self, request):
            return DataPolicyDecision(predicates=(DataPredicate("workspace", DataPredicateOperator.EQ, "a"),))

    try:
        client = adapter._client_instance()
        if not adapter.policy_atomic_supported():
            pytest.skip("conditional write modes require Qdrant and client 1.17+")
        store = PolicyVectorSearch(adapter, "v", policies=DataPolicyPipeline((Policy(),)))
        item = {"id": "1", "vector": [1.0, 0.0], "payload": {"workspace": "a"}}
        prepare = journal.prepare
        def fail_prepare(*_args, **_kwargs):
            raise OSError("journal unavailable")
        journal.prepare = fail_prepare
        with pytest.raises(OSError, match="journal unavailable"):
            store.upsert_vectors([item])
        assert client.retrieve(collection, ids=[1], with_payload=True) == []
        journal.prepare = prepare
        assert store.upsert_vectors([item]).written == 1
        original = adapter.policy_get_atomic("1")[1]
        assert store.upsert_vectors([item]).written == 1
        current = adapter.policy_get_atomic("1")[1]
        assert current != original
        assert store.confirm_upsert("1", current) is True
        assert store.confirm_upsert("1", original) is None
        assert "_muscles_policy_operation" not in adapter.policy_get("1")
        confirm = adapter.policy_confirm_upsert
        adapter.policy_confirm_upsert = lambda *_args: None
        with pytest.raises(DataWriteOutcomeUnknownError) as uncertain:
            store.upsert_vectors([item])
        assert SqliteOperationJournal(journal_path).pending("v")[0]["operation_id"] == uncertain.value.operation_id
        adapter.policy_confirm_upsert = confirm
        assert store.reconcile_pending()[0]["confirmed"] is True
        assert SqliteOperationJournal(journal_path).pending("v") == []
        current = adapter.policy_get_atomic("1")[1]
        with pytest.raises(DataWriteOutcomeUnknownError) as conflict:
            adapter.policy_atomic_upsert(item, original, {"workspace": "a"})
        assert conflict.value.document_id == "1"
        assert store.confirm_upsert("1", conflict.value.operation_id) is None
        with pytest.raises(DataWriteOutcomeUnknownError) as stale_delete:
            adapter.policy_atomic_delete("1", original, {"workspace": "a"})
        assert stale_delete.value.operation_kind == "delete"
        assert adapter.policy_get("1")["workspace"] == "a"
        client.set_payload(collection, payload={"workspace": "b"}, points=[1], wait=True)
        with pytest.raises(DataWriteOutcomeUnknownError):
            adapter.policy_atomic_upsert(item, current, {"workspace": "a"})
        assert client.retrieve(collection, ids=[1], with_payload=True)[0].payload["workspace"] == "b"
        client.upsert(collection, points=[models.PointStruct(id=2, vector=[1.0, 0.0], payload={"workspace": "a"})], wait=True)
        with pytest.raises(DataUnsupportedOperationError, match="managed policy version"):
            store.upsert_vectors([{"id": "2", "vector": [1.0, 0.0], "payload": {"workspace": "a"}}])
        with pytest.raises(DataUnsupportedOperationError, match="maintenance"):
            store.adopt_existing("2", expected_fields={"workspace": "a"}, maintenance_confirmed=False)
        store.adopt_existing("2", expected_fields={"workspace": "a"}, maintenance_confirmed=True)
        assert store.upsert_vectors([{"id": "2", "vector": [1.0, 0.0], "payload": {"workspace": "a"}}]).written == 1
        with pytest.raises(DataUnsupportedOperationError, match="already managed"):
            store.adopt_existing("2", expected_fields={"workspace": "a"}, maintenance_confirmed=True)
        client.upsert(collection, points=[models.PointStruct(id=5, vector=[1.0, 0.0], payload={"workspace": "a"})], wait=True)
        confirm_adopt = adapter.policy_confirm_upsert
        adapter.policy_confirm_upsert = lambda *_args: None
        with pytest.raises(DataWriteOutcomeUnknownError) as adoption:
            store.adopt_existing("5", expected_fields={"workspace": "a"}, maintenance_confirmed=True)
        assert adoption.value.operation_kind == "adopt"
        adapter.policy_confirm_upsert = confirm_adopt
        assert next(entry for entry in store.reconcile_pending() if entry["operation_id"] == adoption.value.operation_id)["confirmed"] is True
        client.upsert(collection, points=[models.PointStruct(id=6, vector=[1.0, 0.0], payload={"workspace": "a"})], wait=True)
        set_payload = client.set_payload
        def concurrent_adoption(**kwargs):
            set_payload(collection_name=collection, payload={"workspace": "b"}, points=[6], wait=True)
            return set_payload(**kwargs)
        client.set_payload = concurrent_adoption
        with pytest.raises(DataWriteOutcomeUnknownError) as raced_adoption:
            store.adopt_existing("6", expected_fields={"workspace": "a"}, maintenance_confirmed=True)
        client.set_payload = set_payload
        assert raced_adoption.value.operation_kind == "adopt"
        raced_point = client.retrieve(collection, ids=[6], with_payload=True)[0].payload
        assert raced_point["workspace"] == "b" and "_muscles_policy_operation" not in raced_point
        client.upsert(collection, points=[models.PointStruct(id=4, vector=[1.0, 0.0], payload={"workspace": "a", "_muscles_policy_deleted": False})], wait=True)
        assert adapter.policy_get("4") is None
        assert all(hit.id != "4" for hit in store.search_vectors([1.0, 0.0], limit=10))
        store.upsert_vectors([{"id": "3", "vector": [1.0, 0.0], "payload": {"workspace": "a"}}])
        confirm_delete = adapter.policy_confirm_delete
        adapter.policy_confirm_delete = lambda *_args: None
        with pytest.raises(DataWriteOutcomeUnknownError) as deletion:
            store.delete_vectors(ids=["3"])
        assert deletion.value.operation_kind == "delete"
        assert adapter.policy_get("3") is None
        assert all(hit.id != "3" for hit in store.search_vectors([1.0, 0.0], limit=10))
        adapter.policy_confirm_delete = confirm_delete
        recovery = store.reconcile_pending()
        assert next(item for item in recovery if item["operation_id"] == deletion.value.operation_id)["confirmed"] is True
        assert all(item["operation_id"] != deletion.value.operation_id for item in journal.pending("v"))
        assert client.retrieve(collection, ids=[3], with_payload=True)[0].payload["_muscles_policy_deleted"] is True
        with pytest.raises(DataUnsupportedOperationError, match="tombstone"):
            store.upsert_vectors([{"id": "3", "vector": [1.0, 0.0], "payload": {"workspace": "a"}}])
    finally:
        if client is not None:
            try:
                client.delete_collection(collection_name=collection)
            except Exception:
                pass
        runtime.close()


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
        contracts = pytest.importorskip("muscles.data.contracts")
        contract = getattr(contracts, "assert_vector_search_contract", None)
        if contract is not None:
            if "dimension" in inspect.signature(contract).parameters:
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
