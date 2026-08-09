"""
Bi-temporal change detection on co-registered rasters — pure logic.

Pipeline
--------
1. Index selection: ``band`` (single band), ``ndvi`` = (nir-red)/(nir+red) or
   ``ndmi`` = (nir-swir)/(nir+swir), computed per scene from a band map.
2. Differencing: delta = index(t2) - index(t1).
3. Thresholding (both required): |delta| >= ``abs_threshold`` AND
   |z| >= ``z_threshold`` where z = (delta - scene_mean) / scene_std — i.e.
   changes must be large in absolute terms *and* outliers vs the scene's own
   delta distribution.
4. Morphological clean-up: binary opening (speckle removal) then binary
   closing (hole filling), ``scipy.ndimage``.
5. Connected components (8-connectivity) -> change regions with area,
   centroid, mean delta; GeoJSON export as bbox polygons (no shapely/rasterio
   dependency by design).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

INDEX_BANDS = {
    "ndvi": ("nir", "red"),
    "ndmi": ("nir", "swir"),
}


def _band(cube: np.ndarray, band_1based: int) -> np.ndarray:
    if not 1 <= band_1based <= cube.shape[0]:
        raise ValueError(
            f"cube has {cube.shape[0]} bands; band {band_1based} unavailable")
    return cube[band_1based - 1].astype(float)


def compute_scene_index(cube: np.ndarray, index: str, band: int = 1,
                        band_map: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Compute per-pixel index for one scene cube (bands, rows, cols)."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("scene must be (bands, rows, cols)")
    if index == "band":
        return _band(cube, band)
    if index not in INDEX_BANDS:
        raise ValueError(f"unsupported index: {index}")
    if band_map is None:
        raise ValueError(f"band_map required for index '{index}'")
    a = _band(cube, band_map[INDEX_BANDS[index][0]])
    b = _band(cube, band_map[INDEX_BANDS[index][1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(a + b) > 1e-12, (a - b) / (a + b), 0.0)


def detect_changes(
    scene_t1: np.ndarray,
    scene_t2: np.ndarray,
    index: str = "band",
    band: int = 1,
    band_map: Optional[Dict[str, int]] = None,
    abs_threshold: float = 0.2,
    z_threshold: float = 2.0,
    morph_open: int = 1,
    morph_close: int = 1,
    min_pixels: int = 1,
    pixel_size: float = 30.0,
) -> Dict[str, Any]:
    """Detect change regions between two co-registered scene cubes.

    ``morph_open`` / ``morph_close`` are iteration counts for binary opening
    and closing (0 disables).  Returns delta stats plus a region list.
    """
    s1 = np.asarray(scene_t1, dtype=float)
    s2 = np.asarray(scene_t2, dtype=float)
    if s1.shape != s2.shape:
        raise ValueError("scenes must have identical shapes")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1")
    if abs_threshold < 0 or z_threshold < 0:
        raise ValueError("thresholds must be >= 0")

    i1 = compute_scene_index(s1, index, band, band_map)
    i2 = compute_scene_index(s2, index, band, band_map)
    delta = i2 - i1

    mean = float(delta.mean())
    std = float(delta.std())
    z = np.zeros_like(delta) if std < 1e-12 else (delta - mean) / std

    mask = (np.abs(delta) >= abs_threshold) & (np.abs(z) >= z_threshold)

    if morph_open > 0:
        mask = ndimage.binary_opening(mask, iterations=morph_open)
    if morph_close > 0:
        mask = ndimage.binary_closing(mask, iterations=morph_close)

    structure = np.ones((3, 3), dtype=int)
    labels, n = ndimage.label(mask, structure=structure)

    regions: List[Dict[str, Any]] = []
    for lab in range(1, n + 1):
        rows, cols = np.where(labels == lab)
        count = int(len(rows))
        if count < min_pixels:
            continue
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        regions.append({
            "label": int(lab),
            "n_pixels": count,
            "area": count * pixel_size ** 2,
            "centroid_rc": [float(rows.mean()), float(cols.mean())],
            "mean_delta": float(delta[rows, cols].mean()),
            "max_abs_delta": float(np.abs(delta[rows, cols]).max()),
            "bbox_pixels": [c0, r0, c1 + 1, r1 + 1],
            "bbox_map": [c0 * pixel_size, r0 * pixel_size,
                         (c1 + 1) * pixel_size, (r1 + 1) * pixel_size],
        })

    return {
        "index": index,
        "abs_threshold": float(abs_threshold),
        "z_threshold": float(z_threshold),
        "pixel_size": float(pixel_size),
        "shape": list(delta.shape),
        "delta_mean": mean,
        "delta_std": std,
        "n_changed_pixels": int(mask.sum()),
        "n_regions": len(regions),
        "regions": sorted(regions, key=lambda r: r["n_pixels"], reverse=True),
    }


def regions_to_geojson(regions: List[Dict[str, Any]],
                       origin_x: float = 0.0,
                       origin_y: float = 0.0,
                       crs: str = "EPSG:local") -> Dict[str, Any]:
    """Export change regions as a GeoJSON FeatureCollection of bbox polygons."""
    features = []
    for r in regions:
        x0, y0, x1, y1 = r["bbox_map"]
        x0 += origin_x; x1 += origin_x
        y0 += origin_y; y1 += origin_y
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
                ]],
            },
            "properties": {
                "label": r["label"],
                "n_pixels": r["n_pixels"],
                "area": r["area"],
                "mean_delta": r["mean_delta"],
                "max_abs_delta": r["max_abs_delta"],
            },
        })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }
