"""Validate oil-spill persistence against a migrated PostgreSQL database.

Set ``OIL_SPILL_TEST_DATABASE_URL`` to an isolated database URL. If it is absent,
``DATABASE_URL`` is used. This script intentionally refuses SQLite so it exercises the
same JSONB and transaction behavior used by the production deployment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.environ.get("OIL_SPILL_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://")):
    raise RuntimeError(
        "Set OIL_SPILL_TEST_DATABASE_URL (or DATABASE_URL) to an isolated PostgreSQL database before running this check."
    )
os.environ["DATABASE_URL"] = DATABASE_URL
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))

from api.database import AuditLogModel, OilSpillIncidentModel, SessionLocal, init_db  # noqa: E402
from api.endpoints.oil_spill import _persist_assessment  # noqa: E402
from api.oil_spill.schemas import GeographicBounds, MaskAnalysisRequest, ObservationSource, ReviewStatus  # noqa: E402


def main() -> None:
    init_db()  # Connectivity only; Alembic must already have created the schema.
    probability_map = np.zeros((12, 12), dtype=np.float32)
    probability_map[3:9, 4:10] = 0.92
    request = MaskAnalysisRequest(
        mask_base64="AA==",  # Unused by the internal persistence path after inference/mask decoding.
        image_width_px=12,
        image_height_px=12,
        source=ObservationSource.DRONE_RGB,
        model_id="validation-unet",
        model_version="1.0.0",
        ground_sampling_distance_m=0.5,
        geographic_bounds=GeographicBounds(west=4.0, south=51.0, east=4.0012, north=51.0012),
    )
    db = SessionLocal()
    incident_id = None
    try:
        response = _persist_assessment(request, probability_map, db)
        incident_id = response.incident_id
        stored = db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first()
        assert stored is not None
        assert response.review_status == ReviewStatus.PENDING_REVIEW
        assert response.oil_pixel_count == 36
        assert response.oil_area_m2 == 9.0
        assert response.geometry_geojson is not None
        assert stored.source_metadata["analysis_parameters"]["ground_sampling_distance_m"] == 0.5
        print(f"Oil-spill PostgreSQL persistence verified: {incident_id}")
    finally:
        if incident_id:
            db.query(AuditLogModel).filter(
                AuditLogModel.entity_type == "oil_spill_incident", AuditLogModel.entity_id == incident_id
            ).delete(synchronize_session=False)
            db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).delete()
            db.commit()
        db.close()


if __name__ == "__main__":
    main()
