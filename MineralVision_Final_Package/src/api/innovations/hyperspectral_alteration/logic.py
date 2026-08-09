"""
Alteration mineral mapping from multi-band raster cubes — pure logic.

Band-ratio indices (ratios enhance mineral absorption features; ASTER-style
indices after e.g. Rowan & Mars 2003):

* **clay**      = b7 / b5   (ASTER SWIR: Al-OH absorption in band 7)
* **iron_oxide**= b4 / b2   (Fe3+ absorption in the blue region)
* **carbonate** = b8 / b7   (ASTER TIR CO3 feature)
* **NDVI**      = (nir - red) / (nir + red), used to *mask out* vegetation
  (NDVI > ``ndvi_mask_threshold`` pixels are excluded from alteration zones)

Band presets map logical indices to 1-based band numbers of the posted cube
(the cube must contain at least the referenced number of bands):

* ``aster``     (>=9 bands): clay=(7,5), iron=(4,2), carbonate=(8,7), ndvi=(3,2)
* ``landsat8``  (>=7 bands): clay=(7,6), iron=(4,2), ndvi=(5,4)  [no TIR -> no carbonate]
* ``sentinel2`` (>=12 bands): clay=(12,11), iron=(4,2), ndvi=(8,4) [no TIR -> no carbonate]

Zones: pixels passing the per-index threshold (after vegetation masking) are
grouped with 8-connectivity connected components (``scipy.ndimage.label``);
each zone reports pixel count, area (pixel_size^2 x count), mean index and a
bounding box.  GeoJSON export emits one rectangular Polygon feature per zone
(bbox polygon — no shapely/rasterio dependency by design).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

# preset -> index -> (numerator_band, denominator_band), 1-based
BAND_PRESETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "aster": {
        "clay": (7, 5),
        "iron_oxide": (4, 2),
        "carbonate": (8, 7),
        "ndvi": (3, 2),
    },
    "landsat8": {
        "clay": (7, 6),
        "iron_oxide": (4, 2),
        "ndvi": (5, 4),
    },
    "sentinel2": {
        "clay": (12, 11),
        "iron_oxide": (4, 2),
        "ndvi": (8, 4),
    },
}

DEFAULT_THRESHOLDS = {"clay": 1.4, "iron_oxide": 1.6, "carbonate": 1.1}


def _band(cube: np.ndarray, band_1based: int) -> np.ndarray:
    if not 1 <= band_1based <= cube.shape[0]:
        raise ValueError(
            f"cube has {cube.shape[0]} bands; band {band_1based} unavailable")
    return cube[band_1based - 1].astype(float)


def compute_index(cube: np.ndarray, index: str,
                  band_map: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Compute a band-ratio (or NDVI) index over a (bands, rows, cols) cube."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("cube must be (bands, rows, cols)")
    if index not in band_map:
        raise ValueError(f"index '{index}' not available for this band preset")
    num_b, den_b = band_map[index]
    num = _band(cube, num_b)
    den = _band(cube, den_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        if index == "ndvi":
            out = np.where(np.abs(num + den) > 1e-12,
                           (num - den) / (num + den), 0.0)
        else:
            out = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    return out


def map_alteration_zones(
    cube: np.ndarray,
    index: str,
    preset: str = "aster",
    threshold: Optional[float] = None,
    band_map: Optional[Dict[str, Tuple[int, int]]] = None,
    ndvi_mask_threshold: Optional[float] = 0.3,
    min_pixels: int = 1,
    pixel_size: float = 15.0,
) -> Dict[str, Any]:
    """Detect alteration zones for one index.

    Returns the index array stats plus a zone list (label, pixel count, area,
    mean index, bbox in pixel coords and map units).
    """
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("cube must be (bands, rows, cols)")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1")
    if band_map is None:
        if preset not in BAND_PRESETS:
            raise ValueError(f"unknown preset: {preset}")
        band_map = BAND_PRESETS[preset]
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS.get(index, 1.0)

    idx = compute_index(cube, index, band_map)

    valid = np.ones(idx.shape, dtype=bool)
    if ndvi_mask_threshold is not None and "ndvi" in band_map:
        ndvi = compute_index(cube, "ndvi", band_map)
        valid &= ndvi <= ndvi_mask_threshold

    mask = (idx >= threshold) & valid
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labels, n_zones = ndimage.label(mask, structure=structure)

    zones: List[Dict[str, Any]] = []
    for label in range(1, n_zones + 1):
        rows, cols = np.where(labels == label)
        count = int(len(rows))
        if count < min_pixels:
            continue
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        zones.append({
            "label": int(label),
            "n_pixels": count,
            "area": count * pixel_size ** 2,
            "mean_index": float(idx[rows, cols].mean()),
            "max_index": float(idx[rows, cols].max()),
            "centroid_rc": [float(rows.mean()), float(cols.mean())],
            "bbox_pixels": [c0, r0, c1 + 1, r1 + 1],          # [x0,y0,x1,y1)
            "bbox_map": [c0 * pixel_size, r0 * pixel_size,
                         (c1 + 1) * pixel_size, (r1 + 1) * pixel_size],
        })

    return {
        "index": index,
        "preset": preset,
        "threshold": float(threshold),
        "ndvi_mask_threshold": ndvi_mask_threshold,
        "pixel_size": float(pixel_size),
        "shape": list(idx.shape),
        "index_min": float(idx.min()),
        "index_max": float(idx.max()),
        "index_mean": float(idx.mean()),
        "n_pixels_above_threshold": int(mask.sum()),
        "n_zones": len(zones),
        "zones": sorted(zones, key=lambda z: z["n_pixels"], reverse=True),
    }


def zones_to_geojson(zones: List[Dict[str, Any]],
                     origin_x: float = 0.0,
                     origin_y: float = 0.0,
                     crs: str = "EPSG:local") -> Dict[str, Any]:
    """Export zones as a GeoJSON FeatureCollection of bbox polygons."""
    features = []
    for z in zones:
        x0, y0, x1, y1 = z["bbox_map"]
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
                "label": z["label"],
                "n_pixels": z["n_pixels"],
                "area": z["area"],
                "mean_index": z["mean_index"],
                "max_index": z["max_index"],
            },
        })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }
