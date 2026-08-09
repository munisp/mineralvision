"""geoai integration router — /innovations/geoai.

Honest degradation contract:
- Heavy optional backends (geoai, samgeo, torch) are imported lazily inside
  handlers; when unavailable the endpoint either returns 503 with explicit
  detail or falls back to a real CPU implementation and names it in the
  ``backend`` response field. Nothing is ever fabricated.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:  # dual-context import
    from src.api.innovations.geoai import core, models
except ImportError:  # pragma: no cover
    from api.innovations.geoai import core, models

router = APIRouter(prefix="/innovations/geoai", tags=["geoai-integration"])

_OPTIONAL_BACKENDS = ["geoai", "samgeo", "torch", "torchgeo", "rasterio", "shapely", "skimage"]


def _probe(modname: str) -> Dict[str, Any]:
    try:
        m = importlib.import_module(modname)
        return {"available": True, "version": getattr(m, "__version__", "unknown")}
    except Exception as exc:  # never fail the capabilities endpoint
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


@router.get("/capabilities")
def capabilities() -> Dict[str, Any]:
    """Report which backends are importable, with versions. Never fails."""
    backends = {name: _probe(name) for name in _OPTIONAL_BACKENDS}
    return {
        "module": "geoai-integration",
        "backends": backends,
        "cpu_fallbacks": {
            "segmentation": "skimage-slic or scipy-ndimage-otsu",
            "change_detection": "image-differencing + otsu",
            "indices": "numpy band math (always available)",
        },
    }


# ---------------------------------------------------------------------------
# raster indices
# ---------------------------------------------------------------------------


class IndicesRequest(BaseModel):
    bands: Dict[str, List[List[float]]] = Field(
        ..., description="named bands as nested row-major lists (red/green/blue/nir/swir...)"
    )
    thumbnail_index: Optional[str] = Field(None, description="index name to render as PNG thumbnail")


@router.post("/raster/indices")
def raster_indices(req: IndicesRequest) -> Dict[str, Any]:
    if not req.bands:
        raise HTTPException(422, "at least one band is required")
    try:
        bands = {k: np.asarray(v, dtype=np.float64) for k, v in req.bands.items()}
    except Exception as exc:
        raise HTTPException(422, f"invalid band array: {exc}")
    shapes = {v.shape for v in bands.values()}
    if len(shapes) != 1:
        raise HTTPException(422, f"band shape mismatch: {shapes}")
    indices = core.compute_indices(bands)
    if not indices:
        raise HTTPException(
            422,
            "no computable indices from supplied bands "
            "(need nir+red for ndvi, green+nir for ndwi, red+blue for iron_oxide, swir pair for clay)",
        )
    out: Dict[str, Any] = {
        "backend": "numpy",
        "indices": {name: core.index_stats(a) for name, a in indices.items()},
    }
    thumb = req.thumbnail_index or next(iter(indices))
    if thumb in indices:
        out["thumbnail_png_b64"] = core.array_to_png_b64(indices[thumb])
        out["thumbnail_index"] = thumb
    return out


# ---------------------------------------------------------------------------
# auto-segmentation
# ---------------------------------------------------------------------------


class SegmentRequest(BaseModel):
    array: List[List[float]]
    n_segments: int = 50
    min_area_px: int = 1
    prefer_backend: Optional[str] = Field(
        None, description="'geoai'/'samgeo' to require the ML backend; 503 if unavailable"
    )


def _segment_with_geoai(arr: np.ndarray, n_segments: int) -> tuple:
    """Lazy ML backend: geoai + samgeo + torch. Raises ImportError if absent."""
    import geoai  # noqa: F401
    import samgeo  # noqa: F401
    import torch  # noqa: F401

    raise NotImplementedError("geoai SAM pipeline not configured in this deployment")


@router.post("/raster/auto-segment")
def raster_auto_segment(req: SegmentRequest) -> Dict[str, Any]:
    arr = np.asarray(req.array, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise HTTPException(422, "array must be a non-empty 2-D nested list")

    if req.prefer_backend in ("geoai", "samgeo"):
        try:
            labels, backend = _segment_with_geoai(arr, req.n_segments)
        except ImportError as exc:
            raise HTTPException(
                503,
                detail=(
                    f"requested backend '{req.prefer_backend}' unavailable: {exc}. "
                    "Install geoai + samgeo + torch, or omit prefer_backend to use "
                    "the honest CPU fallback (skimage-slic / scipy-ndimage-otsu)."
                ),
            )
        except NotImplementedError as exc:
            raise HTTPException(503, detail=str(exc))
    else:
        try:
            labels, backend = core.segment_slic(arr, req.n_segments)
        except ImportError:
            labels, backend = core.segment_ndimage(arr)

    features, stats = core.labels_to_geojson(labels, min_area_px=req.min_area_px)
    return {
        "backend": backend,
        "n_regions": len(features),
        "total_labeled_px": int(np.count_nonzero(labels)),
        "regions_geojson": {"type": "FeatureCollection", "features": features},
        "area_stats": stats,
    }


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


class ChangeRequest(BaseModel):
    before: List[List[float]]
    after: List[List[float]]
    min_area_px: int = 1
    prefer_backend: Optional[str] = Field(None, description="'changestar' requires torch/torchgeo")


@router.post("/detect/change")
def detect_change(req: ChangeRequest) -> Dict[str, Any]:
    before = np.asarray(req.before, dtype=np.float64)
    after = np.asarray(req.after, dtype=np.float64)
    if before.shape != after.shape:
        raise HTTPException(422, f"shape mismatch: {before.shape} vs {after.shape}")
    if before.ndim != 2:
        raise HTTPException(422, "before/after must be 2-D nested lists")

    if req.prefer_backend == "changestar":
        try:
            import torch  # noqa: F401
            import torchgeo  # noqa: F401
        except ImportError as exc:
            raise HTTPException(
                503,
                detail=(
                    f"backend 'changestar' unavailable: {exc}. Install torch + torchgeo, "
                    "or omit prefer_backend for real CPU differencing + Otsu."
                ),
            )
        raise HTTPException(503, detail="changestar weights not configured in this deployment")

    mask, threshold, thr_backend = core.change_mask(before, after)
    labels = np.zeros_like(mask, dtype=np.int64)
    labels[mask] = 1
    features, stats = core.labels_to_geojson(labels, min_area_px=req.min_area_px)
    return {
        "backend": f"image-differencing+otsu-{thr_backend}",
        "threshold": threshold,
        "changed_px": int(mask.sum()),
        "change_fraction": float(mask.mean()),
        "change_geojson": {"type": "FeatureCollection", "features": features},
        "area_stats": stats,
    }


# ---------------------------------------------------------------------------
# training chips
# ---------------------------------------------------------------------------


class ChipsRequest(BaseModel):
    raster: List[List[float]]
    chip_size: int = 16
    stride: Optional[int] = None
    labels: Optional[List[List[float]]] = None
    drop_empty: bool = False


@router.post("/datasets/chips")
def datasets_chips(req: ChipsRequest) -> Dict[str, Any]:
    raster = np.asarray(req.raster, dtype=np.float64)
    if raster.ndim != 2:
        raise HTTPException(422, "raster must be a 2-D nested list")
    if req.chip_size < 1:
        raise HTTPException(422, "chip_size must be >= 1")
    labels = None
    if req.labels is not None:
        labels = np.asarray(req.labels)
        if labels.shape != raster.shape:
            raise HTTPException(422, "labels shape must match raster shape")
    try:
        manifest = core.extract_chips(
            raster, req.chip_size, req.stride, labels, drop_empty=req.drop_empty
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    manifest["backend"] = "numpy-slicing"
    return manifest


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------


class ModelRegisterRequest(BaseModel):
    name: str
    task: str
    backend: str
    version: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    training_chips: Optional[int] = None


@router.post("/models/register", status_code=201)
def register_model(req: ModelRegisterRequest) -> Dict[str, Any]:
    with models.get_session() as s:
        row = models.GeoAIModelRegistry(**req.model_dump())
        s.add(row)
        s.commit()
        s.refresh(row)
        return {
            "id": row.id,
            "name": row.name,
            "task": row.task,
            "backend": row.backend,
            "created_at": row.created_at.isoformat(),
        }


@router.get("/models")
def list_models() -> Dict[str, Any]:
    with models.get_session() as s:
        rows = s.query(models.GeoAIModelRegistry).order_by(models.GeoAIModelRegistry.id).all()
        return {
            "count": len(rows),
            "models": [
                {
                    "id": r.id,
                    "name": r.name,
                    "task": r.task,
                    "backend": r.backend,
                    "version": r.version,
                    "checkpoint_path": r.checkpoint_path,
                    "metrics": r.metrics,
                    "training_chips": r.training_chips,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
        }
