"""Regression tests for governed, manifest-only OctoSense integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.innovations.integration_hub.db import Base
from src.api.innovations.integration_hub.governed import evidence_to_dict, register_evidence
from src.api.innovations.octosense.manifest import (
    DOCUMENTED_REVISION,
    OctoSenseManifestError,
    admit_octosense_manifest,
    build_governed_evidence_request,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "octosense"


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


def _write_fixture_pair(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    samples = tmp_path / "samples.json"
    metadata = tmp_path / "metadata.json"
    samples.write_text((FIXTURE_DIR / "samples.json").read_text(encoding="utf-8"), encoding="utf-8")
    metadata.write_text((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"), encoding="utf-8")
    return samples, metadata


def _admit(tmp_path: Path, **overrides):
    samples, metadata = _write_fixture_pair(tmp_path)
    kwargs = {
        "samples_path": samples,
        "metadata_path": metadata,
        "dataset_revision": DOCUMENTED_REVISION,
        "purpose": "sensor_fusion_evaluation",
    }
    kwargs.update(overrides)
    return admit_octosense_manifest(**kwargs)


def test_admission_normalizes_multimodal_manifest_and_scope(tmp_path):
    admission = _admit(tmp_path)

    assert admission["admission_state"] == "manifest_validated_pending_data_governance_review"
    assert admission["dataset"]["revision"] == DOCUMENTED_REVISION
    assert admission["dataset"]["license_assertion"] == "MIT"
    assert admission["quality_report"]["episode_count"] == 3
    assert admission["quality_report"]["platform_distribution"] == {
        "boat": 1,
        "car": 1,
        "unitree": 1,
    }
    assert admission["quality_report"]["split_distribution"] == {
        "test": 1,
        "unassigned": 2,
    }
    assert admission["quality_report"]["segmentation_annotation_episode_count"] == 1
    assert admission["quality_report"]["declared_issue_count"] == 3
    assert admission["episodes"][0]["asset"]["integrity_status"] == "not_downloaded_or_verified_by_manifest_adapter"
    assert "oil_spill_segmentation_accuracy" in admission["governance"]["prohibited_validation_claims"]
    assert admission["governance"]["review_required"] is True


def test_lineage_is_deterministic_and_expected_hashes_are_enforced(tmp_path):
    first = _admit(tmp_path / "first")
    second = _admit(tmp_path / "second")
    assert first["lineage_hash"] == second["lineage_hash"]

    with pytest.raises(OctoSenseManifestError, match="SHA-256 mismatch"):
        _admit(tmp_path / "wrong-hash", expected_samples_sha256="0" * 64)


def test_prohibited_domain_purpose_fails_closed(tmp_path):
    with pytest.raises(OctoSenseManifestError, match="not admitted"):
        _admit(tmp_path, purpose="oil_spill_segmentation_evaluation")


def test_duplicate_path_unknown_platform_and_non_mcap_path_fail_closed(tmp_path):
    samples_path, metadata_path = _write_fixture_pair(tmp_path)
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    source["samples"][1]["filepath"] = source["samples"][0]["filepath"]
    samples_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(OctoSenseManifestError, match="duplicate filepath"):
        admit_octosense_manifest(
            samples_path=samples_path,
            metadata_path=metadata_path,
            dataset_revision=DOCUMENTED_REVISION,
            purpose="sensor_ingestion_validation",
        )

    samples_path, metadata_path = _write_fixture_pair(tmp_path / "unknown-platform")
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    source["samples"][0]["platform"] = "drone"
    samples_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(OctoSenseManifestError, match="platform"):
        admit_octosense_manifest(
            samples_path=samples_path,
            metadata_path=metadata_path,
            dataset_revision=DOCUMENTED_REVISION,
            purpose="sensor_ingestion_validation",
        )

    samples_path, metadata_path = _write_fixture_pair(tmp_path / "wrong-extension")
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    source["samples"][0]["filepath"] = "data/boat-demo.bag"
    samples_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(OctoSenseManifestError, match=".mcap"):
        admit_octosense_manifest(
            samples_path=samples_path,
            metadata_path=metadata_path,
            dataset_revision=DOCUMENTED_REVISION,
            purpose="sensor_ingestion_validation",
        )


def test_governed_evidence_request_is_tenant_bound_and_persists(db_session, tmp_path):
    admission = _admit(tmp_path)
    request = build_governed_evidence_request(
        admission,
        tenant_id="tenant-octosense",
        observed_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    assert request["source_system"] == "huggingface_dataset"
    assert request["geometry"] == {}
    assert request["model_run"]["performance_evidence"] is False
    assert request["model_run"]["raw_assets_loaded"] is False

    record = register_evidence(db_session, **request)
    rendered = evidence_to_dict(record)
    assert rendered["tenant_id"] == "tenant-octosense"
    assert rendered["source_system"] == "huggingface_dataset"
    assert rendered["payload"]["episode_count"] == 3
    assert rendered["payload"]["governance"]["review_required"] is True


def test_gps_count_inconsistency_is_rejected(tmp_path):
    samples_path, metadata_path = _write_fixture_pair(tmp_path)
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    source["samples"][0]["n_gps_valid"] = 273.0
    samples_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(OctoSenseManifestError, match="must not exceed"):
        admit_octosense_manifest(
            samples_path=samples_path,
            metadata_path=metadata_path,
            dataset_revision=DOCUMENTED_REVISION,
            purpose="sensor_ingestion_validation",
        )
