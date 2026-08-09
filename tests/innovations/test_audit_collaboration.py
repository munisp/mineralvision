"""Deterministic tests for the audit_collaboration innovation (B5-19)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.innovations.audit_collaboration import router
from api.innovations.audit_collaboration.db import Base, get_db
from api.innovations.audit_collaboration.logic import AuditCollaboration, json_diff


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
def svc(session):
    return AuditCollaboration(session)


class TestJsonDiff:
    def test_scalars_added_removed_changed(self):
        diff = json_diff({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 20, "d": 4})
        assert diff == [
            {"path": "b", "op": "changed", "before": 2, "after": 20},
            {"path": "c", "op": "removed", "before": 3, "after": None},
            {"path": "d", "op": "added", "before": None, "after": 4},
        ]

    def test_nested_and_list_paths(self):
        diff = json_diff(
            {"site": {"grid": {"spacing": 50}}, "tags": ["a", "b"]},
            {"site": {"grid": {"spacing": 25}}, "tags": ["a", "b", "c"]},
        )
        assert diff == [
            {"path": "site.grid.spacing", "op": "changed", "before": 50, "after": 25},
            {"path": "tags[2]", "op": "added", "before": None, "after": "c"},
        ]

    def test_no_change_empty_diff(self):
        assert json_diff({"x": [1, {"y": 2}]}, {"x": [1, {"y": 2}]}) == []


class TestAuditEvents:
    def test_event_stores_computed_diff(self, svc):
        event = svc.record_event(
            actor="geo.a", action="update_drillhole", entity_type="drillhole",
            entity_id="RC001", before={"depth": 150}, after={"depth": 180, "logged": True},
        )
        assert event.diff == [
            {"path": "depth", "op": "changed", "before": 150, "after": 180},
            {"path": "logged", "op": "added", "before": None, "after": True},
        ]
        listed = svc.list_events(entity_type="drillhole", entity_id="RC001")
        assert len(listed) == 1 and listed[0].id == event.id


class TestComments:
    def test_thread_order_depth_first(self, svc):
        c1 = svc.add_comment("project", "P1", "alice", "first")
        c2 = svc.add_comment("project", "P1", "bob", "second")
        r1 = svc.add_comment("project", "P1", "carol", "reply to first", parent_id=c1.id)
        r1a = svc.add_comment("project", "P1", "dave", "nested reply", parent_id=r1.id)
        r2 = svc.add_comment("project", "P1", "erin", "reply to second", parent_id=c2.id)

        threaded = svc.get_threaded_comments("project", "P1")
        assert [c["id"] for c in threaded] == [c1.id, r1.id, r1a.id, c2.id, r2.id]
        assert [c["depth"] for c in threaded] == [0, 1, 2, 0, 1]

    def test_parent_validation(self, svc):
        svc.add_comment("project", "P1", "alice", "on P1")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            svc.add_comment("project", "P2", "bob", "wrong entity", parent_id=1)
        assert exc.value.status_code == 422
        with pytest.raises(HTTPException) as exc2:
            svc.add_comment("project", "P1", "bob", "missing parent", parent_id=999)
        assert exc2.value.status_code == 404


class TestSettingsVersioning:
    def test_monotonic_versions_and_current(self, svc):
        s1 = svc.set_settings("P1", {"grid": 50, "cutoff": 0.5}, "alice")
        s2 = svc.set_settings("P1", {"grid": 25, "cutoff": 0.5}, "bob")
        assert (s1.version, s2.version) == (1, 2)
        assert svc.get_current_settings("P1").settings == {"grid": 25, "cutoff": 0.5}
        other = svc.set_settings("P2", {"grid": 100}, "alice")
        assert other.version == 1  # independent per project

    def test_revert_creates_new_snapshot_with_old_content(self, svc):
        svc.set_settings("P1", {"grid": 50}, "alice")
        svc.set_settings("P1", {"grid": 25}, "bob")
        svc.set_settings("P1", {"grid": 10}, "carol")
        reverted = svc.revert_settings("P1", version=1, actor="dave")
        assert reverted.version == 4  # append-only: revert is a NEW version
        assert reverted.settings == {"grid": 50}
        assert "revert to version 1" in reverted.note
        history = svc.get_settings_history("P1")
        assert [s.version for s in history] == [1, 2, 3, 4]
        assert history[0].settings == {"grid": 50}  # original snapshots untouched

    def test_revert_unknown_version_404(self, svc):
        from fastapi import HTTPException
        svc.set_settings("P1", {"grid": 50}, "alice")
        with pytest.raises(HTTPException) as exc:
            svc.revert_settings("P1", version=9, actor="bob")
        assert exc.value.status_code == 404


class TestAPI:
    @pytest.fixture()
    def client(self, session):
        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield session

        app.dependency_overrides[get_db] = override_db
        return TestClient(app)

    def test_event_roundtrip(self, client):
        resp = client.post("/innovations/audit_collaboration/events", json={
            "actor": "geo", "action": "update", "entity_type": "target", "entity_id": "T1",
            "before": {"rank": 3}, "after": {"rank": 1},
        })
        assert resp.status_code == 201
        assert resp.json()["diff"] == [{"path": "rank", "op": "changed", "before": 3, "after": 1}]
        events = client.get("/innovations/audit_collaboration/events?entity_type=target&entity_id=T1")
        assert events.json()["count"] == 1

    def test_comment_threading_via_api(self, client):
        c1 = client.post("/innovations/audit_collaboration/comments", json={
            "entity_type": "drillhole", "entity_id": "RC001", "author": "a", "body": "top"}).json()
        client.post("/innovations/audit_collaboration/comments", json={
            "entity_type": "drillhole", "entity_id": "RC001", "author": "b",
            "body": "reply", "parent_id": c1["id"]})
        threaded = client.get(
            "/innovations/audit_collaboration/comments?entity_type=drillhole&entity_id=RC001").json()
        assert [c["depth"] for c in threaded["comments"]] == [0, 1]

    def test_settings_flow_via_api(self, client):
        client.put("/innovations/audit_collaboration/settings/P1",
                   json={"settings": {"grid": 50}, "actor": "a"})
        client.put("/innovations/audit_collaboration/settings/P1",
                   json={"settings": {"grid": 25}, "actor": "b"})
        current = client.get("/innovations/audit_collaboration/settings/P1").json()
        assert current["current"]["version"] == 2
        revert = client.post("/innovations/audit_collaboration/settings/P1/revert",
                             json={"version": 1, "actor": "c"})
        assert revert.status_code == 201
        assert revert.json()["version"] == 3 and revert.json()["settings"] == {"grid": 50}
        assert client.get("/innovations/audit_collaboration/settings/NOPE").status_code == 404
