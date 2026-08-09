"""HTTP layer for the drill-core scan pipeline (thin; see logic.py).

Scans are held in an in-memory store keyed by ``scan_id`` (UUID); the image
and registration live with the scan so downstream endpoints only need the id.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(
    prefix="/innovations/core-scan",
    tags=["core_scan"])

# scan_id -> {"image": np.ndarray, "boxes": [...], "spectral_rows": [...]}
SCAN_STORE: Dict[str, Dict[str, Any]] = {}


class CoreBox(BaseModel):
    """One core box/tray: pixel-row span registered to a depth interval."""
    row_start: int = Field(ge=0, description="first pixel row (inclusive)")
    row_end: int = Field(gt=0, description="last pixel row (exclusive)")
    start_depth_m: float = Field(ge=0)
    end_depth_m: float = Field(gt=0)


class SpectralRow(BaseModel):
    row: int = Field(ge=0, description="pixel row this profile belongs to")
    bands: List[float] = Field(min_length=2,
                               description="band values, 1-based order")


class BandRatio(BaseModel):
    numerator: int = Field(ge=1)
    denominator: int = Field(ge=1)


class IngestRequest(BaseModel):
    """Core photo (nested list rows x cols [x 3] or base64 PNG) + registration."""
    image: Any
    boxes: List[CoreBox] = Field(min_length=1)
    spectral_rows: Optional[List[SpectralRow]] = None
    hole_id: Optional[str] = None


class MapRequest(BaseModel):
    scan_id: str
    segment_m: float = Field(default=1.0, gt=0)
    preset: str = "aster"
    band_map: Optional[Dict[str, BandRatio]] = None
    thresholds: Optional[Dict[str, float]] = None


class AlterationLogRequest(BaseModel):
    scan_id: str
    segment_m: float = Field(default=1.0, gt=0)
    preset: str = "aster"
    band_map: Optional[Dict[str, BandRatio]] = None
    thresholds: Optional[Dict[str, float]] = None
    zonation: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="alteration zone -> mineral classes, e.g. "
                    "{phyllic: [clay], propylitic: [carbonate]}")


class QualityRequest(BaseModel):
    scan_id: str
    present_threshold: Optional[float] = Field(
        default=None, description="row-brightness cut; Otsu when omitted")
    fracture_min_width_px: int = Field(default=1, ge=1)
    rq_piece_m: float = Field(default=0.10, gt=0,
                              description="RQD piece length cutoff (m)")


def _get_scan(scan_id: str) -> Dict[str, Any]:
    scan = SCAN_STORE.get(scan_id)
    if scan is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown scan_id: {scan_id}")
    return scan


def _band_map(req: Optional[Dict[str, BandRatio]]
              ) -> Optional[Dict[str, Tuple[int, int]]]:
    if req is None:
        return None
    return {k: (v.numerator, v.denominator) for k, v in req.items()}


@router.post("/scan/ingest")
def ingest_scan(req: IngestRequest) -> Dict[str, Any]:
    """Ingest a core scan and return its id + depth-registration QC."""
    try:
        image = logic.decode_image(req.image)
        boxes = logic.validate_boxes([b.model_dump() for b in req.boxes])
        qc = logic.registration_qc(boxes, image.shape[0])
        spectral_rows = ([s.model_dump() for s in req.spectral_rows]
                         if req.spectral_rows else None)
        if spectral_rows:
            for s in spectral_rows:
                if s["row"] >= image.shape[0]:
                    raise ValueError(
                        f"spectral row {s['row']} exceeds image rows")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scan_id = uuid.uuid4().hex
    SCAN_STORE[scan_id] = {"image": image, "boxes": boxes,
                           "spectral_rows": spectral_rows,
                           "hole_id": req.hole_id}
    return {"scan_id": scan_id, "hole_id": req.hole_id,
            "image_shape": list(image.shape),
            "n_spectral_rows": len(spectral_rows or []),
            "registration_qc": qc}


def _run_map(req: MapRequest) -> Dict[str, Any]:
    scan = _get_scan(req.scan_id)
    try:
        return logic.mineral_map(
            scan["image"], scan["boxes"],
            spectral_rows=scan["spectral_rows"],
            segment_m=req.segment_m, preset=req.preset,
            band_map=_band_map(req.band_map), thresholds=req.thresholds)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/scan/mineral-map")
def mineral_map(req: MapRequest) -> Dict[str, Any]:
    """Per-segment downhole mineral log (spectral indices or RGB proxies)."""
    result = _run_map(req)
    result["scan_id"] = req.scan_id
    return result


@router.post("/scan/alteration-log")
def alteration_log(req: AlterationLogRequest) -> Dict[str, Any]:
    """Alteration log: contiguous class runs, lengths in m, zonation + CSV."""
    mapped = _run_map(MapRequest(
        scan_id=req.scan_id, segment_m=req.segment_m, preset=req.preset,
        band_map=req.band_map, thresholds=req.thresholds))
    try:
        result = logic.alteration_log(mapped["log"], zonation=req.zonation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["scan_id"] = req.scan_id
    result["segment_m"] = req.segment_m
    return result


@router.post("/scan/quality")
def scan_quality(req: QualityRequest) -> Dict[str, Any]:
    """Photo-derived recovery % and RQD-style fracture estimate."""
    scan = _get_scan(req.scan_id)
    try:
        result = logic.core_quality(
            scan["image"], scan["boxes"],
            present_threshold=req.present_threshold,
            fracture_min_width_px=req.fracture_min_width_px,
            rq_piece_m=req.rq_piece_m)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["scan_id"] = req.scan_id
    return result
