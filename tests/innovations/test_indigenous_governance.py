"""Deterministic tests for the indigenous_governance innovation (B4-16)."""

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth_middleware import TokenPayload, require_auth
from api.innovations.indigenous_governance import router
from api.innovations.indigenous_governance.db import Base, get_db
from api.innovations.indigenous_governance.logic import (
    AccessTier,
    IndigenousGovernance,
    tier_permits_role,
)
from api.innovations.indigenous_governance.models import AccessAuditModel


def make_user(username: str, role: str) -> TokenPayload:
    return TokenPayload(
        user_id=f"u-{username}", username=username, email=f"{username}@mv.test",
        role=role, exp=datetime(2099, 1, 1),
    )


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
def gov(session):
    return IndigenousGovernance(session)


def _seed_records(gov: IndigenousGovernance):
    public = gov.create_record(
        title="Public ethnobotany note", community="Martu", tier=AccessTier.PUBLIC,
        content="Published survey excerpt", consent_reference="CONS-PUB-1",
        attribution="Martu Rangers", actor="custodian1",
    )
    restricted = gov.create_record(
        title="Restricted site buffer", community="Martu", tier=AccessTier.RESTRICTED,
        content="Buffer coordinates for avoidance", consent_reference="CONS-RES-1",
        attribution="Martu Traditional Owners", actor="custodian1",
    )
    sacred = gov.create_record(
        title="Sacred ceremony site", community="Martu", tier=AccessTier.SACRED,
        content="Sensitive location details", consent_reference="CONS-SAC-1",
        attribution="Elders Council", actor="custodian1",
    )
    return public, restricted, sacred


class TestTierPolicy:
    @pytest.mark.parametrize("tier,role,expected", [
        (AccessTier.PUBLIC, "viewer", True),
        (AccessTier.PUBLIC, "guest", True),
        (AccessTier.RESTRICTED, "researcher", True),
        (AccessTier.RESTRICTED, "custodian", True),
        (AccessTier.RESTRICTED, "viewer", False),
        (AccessTier.SACRED, "custodian", True),
        (AccessTier.SACRED, "admin", True),
        (AccessTier.SACRED, "researcher", False),
        (AccessTier.SACRED, "viewer", False),
    ])
    def test_tier_role_matrix(self, tier, role, expected):
        assert tier_permits_role(tier, role) is expected


class TestGovernanceLogic:
    def test_list_never_includes_sacred(self, gov):
        public, restricted, sacred = _seed_records(gov)
        listed = gov.list_records(actor="admin1", role="admin")
        ids = {r.id for r in listed}
        assert sacred.id not in ids
        assert ids == {public.id, restricted.id}

    def test_list_filters_restricted_for_plain_role(self, gov):
        public, restricted, sacred = _seed_records(gov)
        listed = gov.list_records(actor="viewer1", role="viewer")
        assert {r.id for r in listed} == {public.id}

    def test_export_excludes_sacred_even_for_admin(self, gov):
        public, restricted, sacred = _seed_records(gov)
        exported = gov.export_records(actor="admin1", role="admin")
        assert {r.id for r in exported} == {public.id, restricted.id}

    def test_sacred_direct_access_requires_sacred_role(self, gov):
        _, _, sacred = _seed_records(gov)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            gov.get_record(sacred.id, actor="researcher1", role="researcher")
        assert exc.value.status_code == 403
        # custodian can read it directly
        record = gov.get_record(sacred.id, actor="custodian1", role="custodian")
        assert record.id == sacred.id

    def test_every_read_writes_audit_row(self, gov, session):
        public, restricted, sacred = _seed_records(gov)
        gov.get_record(public.id, actor="viewer1", role="viewer")
        gov.get_record(restricted.id, actor="researcher1", role="researcher")
        gov.list_records(actor="admin1", role="admin")          # audits 2 rows (list)
        gov.export_records(actor="custodian1", role="custodian")  # audits 2 rows (export)

        rows = session.query(AccessAuditModel).order_by(AccessAuditModel.id).all()
        assert len(rows) == 6
        assert rows[0].record_id == public.id and rows[0].action == "read" and rows[0].actor == "viewer1"
        assert rows[0].actor_role == "viewer" and rows[0].tier == "public"
        assert rows[1].action == "read" and rows[1].tier == "restricted"
        assert [r.action for r in rows[2:4]] == ["list", "list"]
        assert [r.action for r in rows[4:6]] == ["export", "export"]
        # sacred never audited via list/export (only direct read would audit it)
        assert not any(r.record_id == sacred.id and r.action in ("list", "export") for r in rows)

    def test_failed_access_does_not_audit(self, gov, session):
        _, _, sacred = _seed_records(gov)
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            gov.get_record(sacred.id, actor="researcher1", role="researcher")
        assert session.query(AccessAuditModel).count() == 0


class TestAPI:
    @pytest.fixture()
    def make_client(self, session):
        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield session

        app.dependency_overrides[get_db] = override_db

        def _client(role: str, username: str = "tester"):
            # Override the REAL require_auth dependency — require_role's inner
            # checker Depends(require_auth), so the real role gate still runs.
            app.dependency_overrides[require_auth] = lambda: make_user(username, role)
            return TestClient(app)

        return _client

    @pytest.fixture()
    def seeded(self, gov):
        return _seed_records(gov)

    PAYLOAD = {
        "title": "New site", "community": "Martu", "tier": "restricted",
        "content": "Details", "consent_reference": "CONS-NEW", "attribution": "TO Group",
    }

    def test_create_forbidden_for_wrong_role(self, make_client):
        client = make_client("viewer")
        resp = client.post("/innovations/indigenous_governance/records", json=self.PAYLOAD)
        assert resp.status_code == 403

    def test_create_allowed_for_custodian(self, make_client):
        client = make_client("custodian", "custodian1")
        resp = client.post("/innovations/indigenous_governance/records", json=self.PAYLOAD)
        assert resp.status_code == 201
        assert resp.json()["tier"] == "restricted"

    def test_sacred_hidden_from_list_and_export(self, make_client, seeded):
        public, restricted, sacred = seeded
        client = make_client("admin")
        listed = client.get("/innovations/indigenous_governance/records").json()
        assert sacred.id not in {r["id"] for r in listed["records"]}
        assert {r["id"] for r in listed["records"]} == {public.id, restricted.id}

        exported = client.get("/innovations/indigenous_governance/records/export").json()
        assert {r["id"] for r in exported["records"]} == {public.id, restricted.id}

    def test_export_forbidden_for_plain_user(self, make_client, seeded):
        client = make_client("viewer")
        assert client.get("/innovations/indigenous_governance/records/export").status_code == 403

    def test_sacred_direct_get_403_then_200(self, make_client, seeded):
        _, _, sacred = seeded
        assert make_client("researcher").get(
            f"/innovations/indigenous_governance/records/{sacred.id}").status_code == 403
        assert make_client("viewer").get(
            f"/innovations/indigenous_governance/records/{sacred.id}").status_code == 403
        ok = make_client("custodian").get(f"/innovations/indigenous_governance/records/{sacred.id}")
        assert ok.status_code == 200
        assert ok.json()["tier"] == "sacred"

    def test_api_reads_write_audit_rows(self, make_client, seeded, session):
        public, restricted, sacred = seeded
        client = make_client("admin", "admin1")
        client.get(f"/innovations/indigenous_governance/records/{public.id}")
        client.get("/innovations/indigenous_governance/records")
        rows = session.query(AccessAuditModel).all()
        assert len(rows) == 3  # 1 read + 2 list entries (sacred excluded)
        assert rows[0].action == "read" and rows[0].actor == "admin1"

        audit = client.get("/innovations/indigenous_governance/audit")
        assert audit.status_code == 200
        assert audit.json()["count"] == 3

    def test_audit_endpoint_forbidden_for_researcher(self, make_client, seeded):
        client = make_client("researcher")
        assert client.get("/innovations/indigenous_governance/audit").status_code == 403

    def test_unknown_record_404(self, make_client):
        client = make_client("admin")
        assert client.get("/innovations/indigenous_governance/records/9999").status_code == 404
