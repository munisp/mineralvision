"""Core raster math for the geoai innovation module.

Everything here is real CPU math on numpy arrays -- no fabricated results.
Heavy optional backends (geoai / samgeo / torch) are imported lazily inside
handlers in ``router.py``; this module only depends on numpy / scipy / skimage
/ rasterio / PIL.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# band indices
# ---------------------------------------------------------------------------

_EPS = 1e-10


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + _EPS)


def compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute available spectral indices from named bands.

    Recognised band names (case-insensitive): red, green, blue, nir, swir,
    swir1, swir2. Only indices whose inputs are present are returned.

    - NDVI  = (nir - red) / (nir + red)
    - NDWI  = (green - nir) / (green + nir)   (McFeeters)
    - iron_oxide = red / blue                 (ratio for ferric iron)
    - clay  = swir1 / swir2  (or swir / nir if only one SWIR band given)
    """
    b = {k.lower(): np.asarray(v, dtype=np.float64) for k, v in bands.items()}
    out: Dict[str, np.ndarray] = {}
    if "nir" in b and "red" in b:
        out["ndvi"] = (b["nir"] - b["red"]) / (b["nir"] + b["red"] + _EPS)
    if "green" in b and "nir" in b:
        out["ndwi"] = (b["green"] - b["nir"]) / (b["green"] + b["nir"] + _EPS)
    if "red" in b and "blue" in b:
        out["iron_oxide"] = _safe_ratio(b["red"], b["blue"])
    if "swir1" in b and "swir2" in b:
        out["clay"] = _safe_ratio(b["swir1"], b["swir2"])
    elif "swir" in b and "nir" in b:
        out["clay"] = _safe_ratio(b["swir"], b["nir"])
    return out


def index_stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def array_to_png_b64(arr: np.ndarray) -> str:
    """Render a 2-D array to a grayscale PNG (min-max stretched), base64."""
    from PIL import Image

    a = np.asarray(arr, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < _EPS:
        scaled = np.zeros_like(a, dtype=np.uint8)
    else:
        scaled = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    img = Image.fromarray(scaled, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Otsu threshold (numpy fallback identical in spirit to skimage)
# ---------------------------------------------------------------------------


def otsu_threshold(arr: np.ndarray, nbins: int = 256) -> float:
    """Otsu's between-class-variance threshold, pure numpy."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < _EPS:
        return lo
    hist, edges = np.histogram(a, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(np.float64)
    p = w / w.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + _EPS)
    return float(centers[int(np.argmax(sigma_b2))])


def threshold_otsu(arr: np.ndarray) -> Tuple[float, str]:
    """Use skimage's Otsu when importable, else the numpy implementation."""
    try:
        from skimage.filters import threshold_otsu as _sk_otsu

        return float(_sk_otsu(np.asarray(arr, dtype=np.float64))), "skimage"
    except ImportError:
        return otsu_threshold(arr), "numpy"


# ---------------------------------------------------------------------------
# segmentation backends
# ---------------------------------------------------------------------------


def segment_slic(arr: np.ndarray, n_segments: int = 50) -> Tuple[np.ndarray, str]:
    """SLIC superpixels via skimage. Raises ImportError if unavailable."""
    from skimage.segmentation import slic

    a = np.asarray(arr, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    norm = (a - lo) / (hi - lo + _EPS)
    labels = slic(
        norm,
        n_segments=int(n_segments),
        compactness=10.0,
        start_label=1,
        channel_axis=None,
    )
    return labels.astype(np.int64), "skimage-slic"


def segment_ndimage(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fallback segmentation: Otsu threshold + scipy ndimage labeling."""
    from scipy import ndimage

    a = np.asarray(arr, dtype=np.float64)
    thr, _ = threshold_otsu(a)
    mask = a > thr
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, _ = ndimage.label(mask)
    return labels.astype(np.int64), "scipy-ndimage-otsu"


# ---------------------------------------------------------------------------
# mask / labels -> GeoJSON polygons (rasterio.features)
# ---------------------------------------------------------------------------


def labels_to_geojson(
    labels: np.ndarray, min_area_px: int = 1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Polygonize an integer label raster into GeoJSON features + area stats."""
    from rasterio.features import shapes

    lab = np.asarray(labels)
    features: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []
    for geom, value in shapes(
        lab.astype(np.int32), mask=lab > 0, connectivity=8
    ):
        region = int(value)
        area_px = int(np.count_nonzero(lab == region))
        if area_px < min_area_px:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"region_id": region, "area_px": area_px},
            }
        )
        stats.append({"region_id": region, "area_px": area_px})
    return features, stats


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


def change_mask(
    before: np.ndarray, after: np.ndarray
) -> Tuple[np.ndarray, float, str]:
    """Absolute image differencing + Otsu threshold -> boolean change mask."""
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch: {b.shape} vs {a.shape}")
    diff = np.abs(a - b)
    thr, thr_backend = threshold_otsu(diff)
    return diff > thr, thr, thr_backend


# ---------------------------------------------------------------------------
# chip extraction
# ---------------------------------------------------------------------------


def extract_chips(
    raster: np.ndarray,
    chip_size: int,
    stride: Optional[int] = None,
    labels: Optional[np.ndarray] = None,
    drop_empty: bool = False,
) -> Dict[str, Any]:
    """Tile a 2-D raster into chips with real numpy slicing; build a manifest."""
    r = np.asarray(raster)
    if r.ndim != 2:
        raise ValueError("raster must be 2-D")
    stride = stride or chip_size
    h, w = r.shape
    lab = np.asarray(labels) if labels is not None else None
    chips: List[Dict[str, Any]] = []
    idx = 0
    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            chip = r[y : y + chip_size, x : x + chip_size]
            entry: Dict[str, Any] = {
                "chip_id": idx,
                "row": y,
                "col": x,
                "height": chip_size,
                "width": chip_size,
                "mean": float(chip.mean()),
                "std": float(chip.std()),
            }
            if lab is not None:
                lchip = lab[y : y + chip_size, x : x + chip_size]
                pos = int(np.count_nonzero(lchip))
                entry["label_positive_px"] = pos
                entry["label_fraction"] = float(pos / lchip.size)
                if drop_empty and pos == 0:
                    idx += 1
                    continue
            chips.append(entry)
            idx += 1
    return {
        "chip_size": chip_size,
        "stride": stride,
        "raster_shape": [h, w],
        "n_chips": len(chips),
        "chips": chips,
    }
