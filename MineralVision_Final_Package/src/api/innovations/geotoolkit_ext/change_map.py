"""Innovation 7 — change-map-service.

Bi-temporal change detection: image differencing + Otsu thresholding +
connected-region polygonization. Wraps the platform satellite_change_detection
core when importable; otherwise uses the built-in numpy/scipy/shapely
implementation (honest `backend` field in the response).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import ndimage
from shapely.geometry import Polygon

try:
    from src.api.innovations.geotoolkit_ext.geo_common import geom_to_feature
except ImportError:  # pragma: no cover
    from api.innovations.geotoolkit_ext.geo_common import geom_to_feature

# Dual-context import of the platform core (optional).
_CORE = None
for _modname in (
    "src.api.innovations.satellite_change_detection",
    "api.innovations.satellite_change_detection",
    "src.api.satellite_change_detection",
    "api.satellite_change_detection",
):
    try:
        import importlib
        _CORE = importlib.import_module(_modname)
        break
    except ImportError:
        continue

router = APIRouter()


class ChangeMapRequest(BaseModel):
    before: List[List[float]]
    after: List[List[float]]
    pixel_size: float = 1.0            # ground units per pixel
    origin: Optional[List[float]] = None  # [x0, y0] of pixel (0,0); default [0,0]
    threshold: Optional[float] = None  # override Otsu
    min_region_pixels: int = 1
    backend: str = "auto"              # auto | core | builtin


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Classic Otsu threshold via between-class variance maximization."""
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1e-12))[::-1]
    variance12 = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(variance12)) if variance12.size else 0
    return float(centers[idx])


def polygonize_mask(mask: np.ndarray, pixel_size: float,
                    origin: Tuple[float, float], min_pixels: int
                    ) -> List[Tuple[Polygon, int]]:
    """Connected-component labelling -> per-region polygons in ground coordinates.

    Region polygon traces pixel-corner boundaries of the labelled cells.
    """
    lab, n = ndimage.label(mask)
    regions: List[Tuple[Polygon, int]] = []
    x0, y0 = origin
    for region_id in range(1, n + 1):
        cells = np.argwhere(lab == region_id)  # (row, col) pairs
        if len(cells) < min_pixels:
            continue
        # Build union of unit squares per cell (row r, col c) -> square from
        # (x0 + c*ps, y0 - r*ps) to (x0 + (c+1)*ps, y0 - (r+1)*ps); y down.
        from shapely.geometry import box
        from shapely.ops import unary_union
        squares = []
        for r, c in cells:
            xa = x0 + c * pixel_size
            ya = y0 - r * pixel_size
            squares.append(box(xa, ya - pixel_size, xa + pixel_size, ya))
        poly = unary_union(squares)
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            regions.append((poly, len(cells)))
        else:  # MultiPolygon
            for g in poly.geoms:
                regions.append((g, len(cells)))
    return regions


def _builtin_change_map(req: ChangeMapRequest) -> Dict[str, Any]:
    before = np.asarray(req.before, dtype=float)
    after = np.asarray(req.after, dtype=float)
    if before.shape != after.shape:
        raise HTTPException(status_code=422, detail="before and after arrays must have the same shape")
    if before.ndim not in (2, 3):
        raise HTTPException(status_code=422, detail="arrays must be 2-D (or 3-D multi-band)")

    diff = np.abs(after - before)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    thr = req.threshold if req.threshold is not None else otsu_threshold(diff)
    if diff.max() <= 0:
        # identical rasters — degenerate Otsu binning; nothing changed
        mask = np.zeros_like(diff, dtype=bool)
        thr = 0.0
    else:
        mask = diff > thr
    origin = tuple(req.origin) if req.origin else (0.0, 0.0)

    regions = polygonize_mask(mask, req.pixel_size, origin, req.min_region_pixels)

    features = []
    stats = []
    px_area = req.pixel_size ** 2
    for i, (poly, npx) in enumerate(regions):
        props = {
            "class": "change",
            "pixel_count": npx,
            "area": npx * px_area,
            "mean_abs_change": float(diff[mask].mean()) if mask.any() else 0.0,
        }
        f = geom_to_feature(poly, props)
        f["id"] = i
        features.append(f)
        stats.append({
            "feature_index": i,
            "pixel_count": npx,
            "area": npx * px_area,
            "centroid": [poly.centroid.x, poly.centroid.y],
        })

    changed_px = int(mask.sum())
    unchanged_px = int(mask.size - changed_px)
    return {
        "backend": "builtin:numpy-otsu-shapely",
        "threshold": float(thr),
        "shape": list(diff.shape),
        "result": {"type": "FeatureCollection", "features": features},
        "regions": stats,
        "class_areas": {
            "change": {"pixels": changed_px, "area": changed_px * px_area},
            "no_change": {"pixels": unchanged_px, "area": unchanged_px * px_area},
        },
        "pixel_size": req.pixel_size,
    }


@router.post("/change/map")
def change_map_endpoint(req: ChangeMapRequest) -> Dict[str, Any]:
    use_core = req.backend == "core" or (req.backend == "auto" and _CORE is not None)
    if use_core and _CORE is not None and hasattr(_CORE, "detect_change"):
        try:
            return _CORE.detect_change(req)  # pragma: no cover - core absent in env
        except Exception:
            pass  # fall through to builtin implementation
    if req.backend == "core" and _CORE is None:
        raise HTTPException(status_code=503,
                            detail="satellite_change_detection core is not importable in this environment")
    return _builtin_change_map(req)
