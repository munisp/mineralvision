from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.innovations.integration_hub.db import Base
from src.api.innovations.integration_hub.governed import (
    approve_writeback,
    discover_arcgis_capabilities,
    register_evidence,
    stage_writeback,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _evidence(db_session, tenant_id="tenant-a"):
    return register_evidence(
        db_session,
        tenant_id=tenant_id,
        source_system="arcgis_enterprise",
        source_ref="https://arcgis.example/FeatureServer/7/feature/42",
        source_version="2026-08-22T00:00:00Z",
        observed_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        geometry={"type": "Point", "coordinates": [120.0, -31.0]},
        payload={"grade_ppm": 1.2},
        model_run={"model_id": "prospectivity-1", "artifact_hash": "a" * 64},
    )


def test_lineage_is_stable_and_tenant_bound(db_session):
    first = _evidence(db_session)
    second = _evidence(db_session)
    assert first.lineage_hash == second.lineage_hash
    assert first.evidence_id != second.evidence_id

    with pytest.raises(ValueError, match="requesting tenant"):
        stage_writeback(
            db_session,
            tenant_id="tenant-b",
            evidence_id=first.evidence_id,
            target_system="arcgis_enterprise",
            target_ref="https://arcgis.example/FeatureServer/9",
            candidate_payload={"attributes": {"status": "candidate"}},
            submitted_by="analyst-a",
            dry_run={"schema_valid": True, "crs_valid": True},
        )


def test_arcgis_capabilities_are_conservative():
    capabilities = discover_arcgis_capabilities(
        {
            "capabilities": "Query,Create,Update,Delete,Sync,Uploads",
            "allowGeometryUpdates": True,
            "advancedEditingCapabilities": {"supportsApplyEditsWithGlobalIds": True},
        }
    )
    assert capabilities == {
        "query": True,
        "create": True,
        "update": True,
        "delete": True,
        "sync": True,
        "uploads": True,
        "editing": True,
        "supports_apply_edits_with_global_ids": True,
    }
    assert discover_arcgis_capabilities({"capabilities": "Query"})["editing"] is False


def test_staged_writeback_requires_mfa_and_distinct_reviewer(db_session):
    evidence = _evidence(db_session)
    proposal = stage_writeback(
        db_session,
        tenant_id="tenant-a",
        evidence_id=evidence.evidence_id,
        target_system="arcgis_enterprise",
        target_ref="https://arcgis.example/FeatureServer/9",
        candidate_payload={"attributes": {"status": "candidate"}},
        submitted_by="analyst-a",
        dry_run={"schema_valid": True, "crs_valid": True, "diff": {}},
    )
    assert proposal.state == "staged"

    with pytest.raises(ValueError, match="MFA"):
        approve_writeback(
            db_session,
            tenant_id="tenant-a",
            proposal_id=proposal.proposal_id,
            reviewer_id="reviewer-b",
            mfa_verified=False,
            review_reason="reviewed",
        )
    with pytest.raises(ValueError, match="cannot approve"):
        approve_writeback(
            db_session,
            tenant_id="tenant-a",
            proposal_id=proposal.proposal_id,
            reviewer_id="analyst-a",
            mfa_verified=True,
            review_reason="reviewed",
        )

    approved = approve_writeback(
        db_session,
        tenant_id="tenant-a",
        proposal_id=proposal.proposal_id,
        reviewer_id="reviewer-b",
        mfa_verified=True,
        review_reason="schema and geometry reviewed",
    )
    assert approved.state == "approved"
    assert approved.reviewer_id == "reviewer-b"
    assert approved.mfa_verified is True


def test_integration_api_key_carries_tenant_binding(db_session):
    from src.api.innovations.integration_hub.logic import create_api_key
    from src.api.innovations.integration_hub.routes import _require_governed_tenant

    created = create_api_key(db_session, "tenant-a connector", ["read", "write"], tenant_id="tenant-a")
    assert created["tenant_id"] == "tenant-a"
    key_record = db_session.query(
        __import__("src.api.innovations.integration_hub.models", fromlist=["ApiKeyModel"]).ApiKeyModel
    ).filter_by(key_id=created["key_id"]).one()

    _require_governed_tenant(key_record, "tenant-a")
    with pytest.raises(Exception, match="not bound to this tenant"):
        _require_governed_tenant(key_record, "tenant-b")
