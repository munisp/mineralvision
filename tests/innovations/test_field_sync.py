"""Deterministic tests for the field_sync innovation (B5-18)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.innovations.field_sync import router
from api.innovations.field_sync.db import Base, get_db
from api.innovations.field_sync.logic import FieldSync
from api.innovations.field_sync.models import ConflictModel, SyncOpModel


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = factory()
    yield db
    db.close()
    engine.dispose()


@pytest.fixture()
def sync(session):
    return FieldSync(session)


class TestOptimisticVersioning:
    def test_create_update_delete_version_monotonic(self, sync):
        r1 = sync.apply_op("op-1", "sample-1", "create", 0, {"east": 500500, "commodity": "Au"})
        assert r1.status == "applied" and r1.version == 1
        r2 = sync.apply_op("op-2", "sample-1", "update", 1, {"depth": 45.5})
        assert r2.status == "applied" and r2.version == 2
        state = sync.get_state("sample-1")
        assert state.data == {"east": 500500, "commodity": "Au", "depth": 45.5}  # merge
        r3 = sync.apply_op("op-3", "sample-1", "delete", 2)
        assert r3.version == 3 and sync.get_state("sample-1").deleted is True

    def test_stale_base_version_conflicts_server_wins(self, sync, session):
        sync.apply_op("op-1", "sample-1", "create", 0, {"grade": 1.2})
        sync.apply_op("op-2", "sample-1", "update", 1, {"grade": 2.4})  # server now v2
        # offline client still at base_version=1 tries its own edit
        r = sync.apply_op("op-3", "sample-1", "update", 1, {"grade": 9.9})
        assert r.status == "conflict" and r.version == 2
        # server state untouched (server wins)
        assert sync.get_state("sample-1").data["grade"] == 2.4
        # conflict record retains BOTH versions
        conflicts = sync.list_conflicts("sample-1")
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.base_version == 1 and c.server_version == 2
        assert c.client_payload == {"grade": 9.9}
        assert c.server_payload["grade"] == 2.4
        assert c.resolution == "server_wins"

    def test_invalid_ops_and_states_rejected(self, sync):
        with pytest.raises(ValueError):
            sync.apply_op("op-x", "e", "frobnicate", 0, {})
        with pytest.raises(ValueError):
            sync.apply_op("op-y", "ghost", "update", 0, {})
        sync.apply_op("op-1", "dup", "create", 0, {})
        with pytest.raises(ValueError):
            sync.apply_op("op-2", "dup", "create", 1, {})  # already exists


class TestIdempotency:
    def test_retry_same_client_op_id_is_duplicate(self, sync, session):
        first = sync.apply_op("op-retry", "sample-1", "create", 0, {"v": 1})
        assert first.status == "applied" and first.version == 1
        # network retry — same client_op_id
        second = sync.apply_op("op-retry", "sample-1", "create", 0, {"v": 1})
        assert second.status == "duplicate"
        assert "applied" in second.detail
        # no extra version bump, no extra op row
        assert sync.get_state("sample-1").version == 1
        assert session.query(SyncOpModel).filter_by(client_op_id="op-retry").count() == 1

    def test_retry_of_conflict_is_duplicate_not_new_conflict(self, sync, session):
        sync.apply_op("op-1", "s", "create", 0, {})
        sync.apply_op("op-2", "s", "update", 1, {})
        assert sync.apply_op("op-3", "s", "update", 1, {"x": 1}).status == "conflict"
        assert sync.apply_op("op-3", "s", "update", 1, {"x": 1}).status == "duplicate"
        assert session.query(ConflictModel).count() == 1


class TestPullSince:
    def test_pull_returns_exactly_the_delta(self, sync):
        sync.apply_op("op-1", "a", "create", 0, {"n": 1})
        sync.apply_op("op-2", "a", "update", 1, {"n": 2})
        sync.apply_op("op-3", "b", "create", 0, {"n": 3})
        sync.apply_op("op-4", "a", "update", 2, {"n": 4})

        delta = sync.pull_since(since=2, entity_id="a")
        assert [(o.applied_version, o.payload) for o in delta] == [(3, {"n": 4})]
        # wait: a is at v3 after op-4 (create v1, update v2, update v3)
        full = sync.pull_since(since=0)
        assert [(o.entity_id, o.applied_version) for o in full] == [
            ("a", 1), ("a", 2), ("a", 3), ("b", 1),
        ]
        assert sync.pull_since(since=3, entity_id="a") == []

    def test_pull_excludes_conflicts(self, sync):
        sync.apply_op("op-1", "a", "create", 0, {})
        sync.apply_op("op-2", "a", "update", 1, {"x": 1})
        sync.apply_op("op-3", "a", "update", 1, {"x": 2})  # conflict
        versions = [o.applied_version for o in sync.pull_since(since=0, entity_id="a")]
        assert versions == [1, 2]


class TestAPI:
    @pytest.fixture()
    def client(self, session):
        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield session

        app.dependency_overrides[get_db] = override_db
        return TestClient(app)

    def test_batch_push_and_pull(self, client):
        resp = client.post("/innovations/field_sync/ops/batch", json={"ops": [
            {"client_op_id": "c1", "entity_id": "s1", "op": "create", "base_version": 0,
             "payload": {"hole": "RC001"}},
            {"client_op_id": "c2", "entity_id": "s1", "op": "update", "base_version": 1,
             "payload": {"depth": 88}},
        ]})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert [r["status"] for r in results] == ["applied", "applied"]
        assert [r["version"] for r in results] == [1, 2]

        state = client.get("/innovations/field_sync/state/s1").json()
        assert state["version"] == 2 and state["data"] == {"hole": "RC001", "depth": 88}

        pull = client.get("/innovations/field_sync/pull?since=1&entity_id=s1").json()
        assert pull["count"] == 1 and pull["ops"][0]["applied_version"] == 2

    def test_conflict_and_idempotent_retry_via_api(self, client):
        client.post("/innovations/field_sync/ops", json={
            "client_op_id": "k1", "entity_id": "s1", "op": "create", "base_version": 0, "payload": {}})
        client.post("/innovations/field_sync/ops", json={
            "client_op_id": "k2", "entity_id": "s1", "op": "update", "base_version": 1,
            "payload": {"grade": 1.0}})
        conflict = client.post("/innovations/field_sync/ops", json={
            "client_op_id": "k3", "entity_id": "s1", "op": "update", "base_version": 1,
            "payload": {"grade": 7.7}})
        assert conflict.json()["status"] == "conflict"
        retry = client.post("/innovations/field_sync/ops", json={
            "client_op_id": "k3", "entity_id": "s1", "op": "update", "base_version": 1,
            "payload": {"grade": 7.7}})
        assert retry.json()["status"] == "duplicate"

        conflicts = client.get("/innovations/field_sync/conflicts?entity_id=s1").json()
        assert conflicts["count"] == 1
        assert conflicts["conflicts"][0]["client_payload"] == {"grade": 7.7}
        assert conflicts["conflicts"][0]["server_payload"] == {"grade": 1.0}

    def test_unknown_entity_state_404_and_bad_op_422(self, client):
        assert client.get("/innovations/field_sync/state/nope").status_code == 404
        resp = client.post("/innovations/field_sync/ops", json={
            "client_op_id": "z", "entity_id": "s", "op": "bogus", "base_version": 0, "payload": {}})
        assert resp.status_code == 422
