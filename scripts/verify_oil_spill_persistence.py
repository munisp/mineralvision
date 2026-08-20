"""Validate persistence of an oil-spill assessment against an isolated SQLite database."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMP_DB = Path(tempfile.gettempdir()) / "mineralvision_oil_spill_validation.db"
if TEMP_DB.exists():
    TEMP_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TEMP_DB}"
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))

from api.database import OilSpillIncidentModel, SessionLocal, init_db  # noqa: E402
from api.endpoints.oil_spill import _persist_assessment  # noqa: E402
from api.oil_spill.schemas import GeographicBounds, MaskAnalysisRequest, ObservationSource, ReviewStatus  # noqa: E402


def main() -> None:
    init_db()
    probability_map = np.zeros((12, 12), dtype=np.float32)
    probability_map[3:9, 4:10] = 0.92
    request = MaskAnalysisRequest(
        mask_base64="AA==",  # unused by the internal persistence path after inference/mask decoding
        image_width_px=12,
        image_height_px=12,
        source=ObservationSource.DRONE_RGB,
        model_id="validation-unet",
        model_version="1.0.0",
        ground_sampling_distance_m=0.5,
        geographic_bounds=GeographicBounds(west=4.0, south=51.0, east=4.0012, north=51.0012),
    )
    db = SessionLocal()
    try:
        response = _persist_assessment(request, probability_map, db)
        stored = db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == response.incident_id).first()
        assert stored is not None
        assert response.review_status == ReviewStatus.PENDING_REVIEW
        assert response.oil_pixel_count == 36
        assert response.oil_area_m2 == 9.0
        assert response.geometry_geojson is not None
        print(f"Oil-spill persistence verified: {response.incident_id}")
    finally:
        db.close()
        if TEMP_DB.exists():
            TEMP_DB.unlink()


if __name__ == "__main__":
    main()
