"""Validate PostgreSQL-backed oil-spill model governance end to end.

Use OIL_SPILL_TEST_DATABASE_URL (or DATABASE_URL) pointing at an isolated, migrated
PostgreSQL database. The script creates and removes only its own uniquely named records.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.environ.get("OIL_SPILL_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://")):
    raise RuntimeError("OIL_SPILL_TEST_DATABASE_URL (or DATABASE_URL) must point to migrated PostgreSQL")
os.environ["DATABASE_URL"] = DATABASE_URL
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))

from api.database import AuditLogModel, OilSpillEvaluationRunModel, OilSpillModelModel, SessionLocal, init_db  # noqa: E402
from api.endpoints.oil_spill import approve_model, get_model_promotion_status, record_evaluation, register_model  # noqa: E402
from api.oil_spill.schemas import EvaluationRunRequest, EvaluationSplit, ModelApprovalRequest, ModelRegistrationRequest, SegmentationMetrics  # noqa: E402


def main() -> None:
    init_db()
    suffix = uuid.uuid4().hex[:12]
    model_id = f"governance-validation-{suffix}"
    model_version = "1.0.0"
    db = SessionLocal()
    model_registration_id = None
    try:
        registration = asyncio.run(register_model(
            ModelRegistrationRequest(
                model_id=model_id,
                model_version=model_version,
                engine="onnx",
                artifact_sha256="a" * 64,
                intended_domains=["drone_rgb_daylight", "port_glare"],
            ),
            db,
        ))
        model_registration_id = registration.id
        metrics = SegmentationMetrics(oil_f1=0.975, oil_iou=0.955, oil_precision=0.978, oil_recall=0.972)
        for domain in ("drone_rgb_daylight", "port_glare"):
            asyncio.run(record_evaluation(
                model_id,
                model_version,
                EvaluationRunRequest(
                    dataset_fingerprint=f"sha256:sealed-{domain}",
                    split=EvaluationSplit.SEALED_HOLDOUT,
                    domain=domain,
                    sample_count=150,
                    metrics=metrics,
                    jepa_backbone="vjepa2.1-vit-large-384",
                    reviewer="governance_validation",
                ),
                db,
            ))
        status = asyncio.run(get_model_promotion_status(model_id, model_version, db))
        assert status.eligible and status.lifecycle_status.value == "candidate"
        approved = asyncio.run(approve_model(
            model_id,
            model_version,
            ModelApprovalRequest(reviewer="governance_validation", note="Automated validation only"),
            db,
        ))
        assert approved.eligible and approved.lifecycle_status.value == "approved"
        print(f"Oil-spill PostgreSQL governance verified: {model_id}:{model_version}")
    finally:
        if model_registration_id:
            db.query(OilSpillEvaluationRunModel).filter(
                OilSpillEvaluationRunModel.model_registration_id == model_registration_id
            ).delete(synchronize_session=False)
            db.query(AuditLogModel).filter(
                AuditLogModel.entity_type == "oil_spill_model", AuditLogModel.entity_id == model_registration_id
            ).delete(synchronize_session=False)
            db.query(OilSpillModelModel).filter(OilSpillModelModel.id == model_registration_id).delete()
            db.commit()
        db.close()


if __name__ == "__main__":
    main()
