"""Core algorithms for marine sonar / bathymetry processing.

Depth convention: depth is positive *down* from the sea surface (metres).
Internal terrain maths use elevation = -depth so that "high" features
(pinnacles, reefs) are positive relief.

All functions are pure numpy/scipy/sklearn — no I/O, no globals mutated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import griddata
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def as_rectangular_grid(grid: List[List[float]], name: str = "grid") -> np.ndarray:
    """Validate a nested list is a non-empty rectangular 2-D array."""
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError(f"{name} must be at least 3x3")
    return arr


def _disk_kernel(radius_cells: int) -> np.ndarray:
    r = int(max(radius_cells, 1))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return ((x * x + y * y) <= r * r).astype(float)


def _annulus_mean(grid: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Mean over an annulus (ring) around each cell via FFT convolution."""
    outer_k = _disk_kernel(outer)
    inner_k = _disk_kernel(inner)
    # pad inner kernel to outer size
    pad = outer_k.shape[0] - inner_k.shape[0]
    inner_k = np.pad(inner_k, pad // 2)
    ring = outer_k - inner_k
    ring /= ring.sum()
    return signal.fftconvolve(grid, ring, mode="same")


def _fill_nans(grid: np.ndarray) -> np.ndarray:
    """Fill NaNs with nearest valid value (constant-grid fallback)."""
    if not np.isnan(grid).any():
        return grid
    if np.isnan(grid).all():
        return np.zeros_like(grid)
    idx = ndimage.distance_transform_edt(
        np.isnan(grid), return_distances=False, return_indices=True
    )
    return grid[tuple(idx)]


# ---------------------------------------------------------------------------
# 1. bathymetry processing
# ---------------------------------------------------------------------------


def process_bathymetry(
    pings: np.ndarray,
    grid_shape: Tuple[int, int] = (100, 100),
    median_size: int = 5,
    spike_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Grid a raw ping cloud and filter depth spikes.

    pings: (N, 3|4) array of x, y, depth[, intensity].
    Spike filter: grid the cloud, apply a median filter, flag cells whose
    depth deviates from the local median by more than ``spike_threshold``
    metres as artifacts, and replace them with the median value.
    """
    pings = np.asarray(pings, dtype=float)
    if pings.ndim != 2 or pings.shape[1] not in (3, 4) or pings.shape[0] < 4:
        raise ValueError("pings must be an (N,3) or (N,4) array with N>=4")
    if not np.isfinite(pings[:, :3]).all():
        raise ValueError("pings contain non-finite values")

    ny, nx = grid_shape
    x, y, z = pings[:, 0], pings[:, 1], pings[:, 2]
    gx = np.linspace(x.min(), x.max(), nx)
    gy = np.linspace(y.min(), y.max(), ny)
    GX, GY = np.meshgrid(gx, gy)

    raw = griddata((x, y), z, (GX, GY), method="linear")
    # fill edges / holes with nearest ping so the grid has no NaN
    near = griddata((x, y), z, (GX, GY), method="nearest")
    raw = np.where(np.isnan(raw), near, raw)

    med = ndimage.median_filter(raw, size=int(median_size), mode="nearest")
    diff = raw - med
    artifact_mask = np.abs(diff) > float(spike_threshold)
    filtered = np.where(artifact_mask, med, raw)

    intensity_grid = None
    if pings.shape[1] == 4:
        ig = griddata((x, y), pings[:, 3], (GX, GY), method="linear")
        ig_near = griddata((x, y), pings[:, 3], (GX, GY), method="nearest")
        intensity_grid = np.where(np.isnan(ig), ig_near, ig)

    stats = {
        "n_pings": int(pings.shape[0]),
        "n_artifacts": int(artifact_mask.sum()),
        "min_depth": float(np.min(filtered)),
        "max_depth": float(np.max(filtered)),
        "mean_depth": float(np.mean(filtered)),
        "std_depth": float(np.std(filtered)),
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
        "max_spike_deviation": float(np.max(np.abs(diff))),
    }
    return {
        "grid": filtered,
        "artifact_mask": artifact_mask,
        "intensity_grid": intensity_grid,
        "x_coords": gx,
        "y_coords": gy,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# 2. terrain derivatives
# ---------------------------------------------------------------------------


def _hillshade(dz_dx: np.ndarray, dz_dy: np.ndarray,
               azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    hs = (
        np.sin(alt) * np.cos(slope)
        + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    )
    return np.clip(hs, 0.0, 1.0)


def terrain_derivatives(
    depth_grid: np.ndarray,
    cell_size: float = 1.0,
    bpi_inner: int = 2,
    bpi_outer: int = 8,
) -> Dict[str, np.ndarray]:
    """Slope, aspect, rugosity (VRM), BPI and hillshade from a depth grid."""
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if not (0 < bpi_inner < bpi_outer):
        raise ValueError("require 0 < bpi_inner < bpi_outer")
    elev = -_fill_nans(depth_grid)  # elevation, positive up

    dz_dy, dz_dx = np.gradient(elev, cell_size)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect_deg = np.mod(np.degrees(np.arctan2(-dz_dx, dz_dy)), 360.0)

    # Vector ruggedness measure: 1 - |sum of unit normals| / n over 3x3 window
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(elev)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    win = 3
    sx = ndimage.uniform_filter(nx, win, mode="nearest") * win * win
    sy = ndimage.uniform_filter(ny, win, mode="nearest") * win * win
    sz = ndimage.uniform_filter(nz, win, mode="nearest") * win * win
    resultant = np.sqrt(sx * sx + sy * sy + sz * sz)
    rugosity = np.clip(1.0 - resultant / (win * win), 0.0, 1.0)

    bpi = elev - _annulus_mean(elev, bpi_inner, bpi_outer)
    hillshade = _hillshade(dz_dx, dz_dy)

    return {
        "slope": slope_deg,
        "aspect": aspect_deg,
        "rugosity": rugosity,
        "bpi": bpi,
        "hillshade": hillshade,
    }


# ---------------------------------------------------------------------------
# 3. backscatter classification
# ---------------------------------------------------------------------------


def _local_entropy(img: np.ndarray, size: int, bins: int = 16) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    idx = ((img - lo) / (hi - lo) * (bins - 1)).astype(int)
    ent = np.zeros_like(img, dtype=float)
    for b in range(bins):
        p = ndimage.uniform_filter((idx == b).astype(float), size, mode="nearest")
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(p > 0, p * np.log2(p), 0.0)
        ent -= term
    return ent


_SEABED_LABELS = ["fine_sediment", "coarse_sediment", "rock", "very_hard_substrate"]


def classify_backscatter(
    mosaic: np.ndarray,
    n_classes: int = 3,
    window: int = 9,
    slope_grid: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Unsupervised seafloor classification on backscatter texture.

    Features: local mean, local std, local entropy (scipy.ndimage), plus
    optional terrain slope. K-means clusters are ordered by mean intensity
    and interpreted as fine sediment (lowest) -> rock (highest).
    """
    mosaic = np.asarray(mosaic, dtype=float)
    if mosaic.ndim != 2 or min(mosaic.shape) < window:
        raise ValueError("mosaic must be 2-D and larger than the texture window")
    n_classes = int(n_classes)
    if not 2 <= n_classes <= 8:
        raise ValueError("n_classes must be between 2 and 8")

    feats = [
        ndimage.uniform_filter(mosaic, window, mode="nearest"),
        np.sqrt(
            np.clip(
                ndimage.uniform_filter(mosaic * mosaic, window, mode="nearest")
                - ndimage.uniform_filter(mosaic, window, mode="nearest") ** 2,
                0.0,
                None,
            )
        ),
        _local_entropy(mosaic, window),
    ]
    if slope_grid is not None:
        slope_grid = np.asarray(slope_grid, dtype=float)
        if slope_grid.shape != mosaic.shape:
            raise ValueError("slope_grid shape must match mosaic")
        feats.append(slope_grid)

    X = np.stack([f.ravel() for f in feats], axis=1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    km = KMeans(n_clusters=n_classes, n_init=10, random_state=int(seed))
    labels = km.fit_predict(Xs).reshape(mosaic.shape)

    # order clusters by mean backscatter intensity
    mean_intensity = np.array(
        [mosaic[labels == k].mean() for k in range(n_classes)]
    )
    order = np.argsort(mean_intensity)  # low -> high intensity
    remap = np.empty(n_classes, dtype=int)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    class_map = remap[labels]

    stats = []
    for k in range(n_classes):
        mask = class_map == k
        frac = float(mask.mean())
        stats.append(
            {
                "class_id": k,
                "interpretation": _SEABED_LABELS[min(k, len(_SEABED_LABELS) - 1)]
                + (f"_{k}" if k >= len(_SEABED_LABELS) else ""),
                "fraction": frac,
                "mean_intensity": float(mosaic[mask].mean()),
                "std_intensity": float(mosaic[mask].std()),
                "mean_local_std": float(feats[1][mask].mean()),
                "mean_entropy": float(feats[2][mask].mean()),
            }
        )
    return {"class_map": class_map, "class_stats": stats}


# ---------------------------------------------------------------------------
# 4. feature detection
# ---------------------------------------------------------------------------


def _component_geometry(mask: np.ndarray, cell_size: float) -> Dict[str, float]:
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    # second-moment elongation
    yc = ys - cy
    xc = xs - cx
    cov = np.array(
        [[np.mean(xc * xc), np.mean(xc * yc)], [np.mean(xc * yc), np.mean(yc * yc)]]
    )
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = np.clip(eig, 1e-12, None)
    length = 4.0 * np.sqrt(eig[0]) * cell_size  # ~2 sigma either side
    width = 4.0 * np.sqrt(eig[1]) * cell_size
    elongation = float(np.sqrt(eig[0] / eig[1]))
    orientation = float(np.degrees(np.arctan2(cov[0, 1], cov[0, 0])))
    return {
        "centroid_row": cy,
        "centroid_col": cx,
        "length_m": float(length),
        "width_m": float(width),
        "elongation": elongation,
        "orientation_deg": orientation,
        "area_cells": int(mask.sum()),
    }


def detect_features(
    depth_grid: np.ndarray,
    cell_size: float = 1.0,
    relief_threshold: float = 1.0,
    smooth_window: int = 15,
    min_area_cells: int = 6,
    min_elongation: float = 3.0,
) -> Dict[str, Any]:
    """Detect local relief maxima/minima and elongated (linear) features.

    Residual relief = grid minus a large-window local mean; components above
    the threshold are labelled. Elongated components (second-moment aspect
    ratio >= min_elongation) are classed as linear features (channels,
    lineaments); compact positive ones as pinnacles/relief highs.
    """
    if cell_size <= 0 or relief_threshold <= 0:
        raise ValueError("cell_size and relief_threshold must be positive")
    elev = -_fill_nans(np.asarray(depth_grid, dtype=float))
    local_mean = ndimage.uniform_filter(elev, int(smooth_window), mode="nearest")
    residual = elev - local_mean

    # orientation histogram of strong gradients (Hough-style summary)
    gy, gx = np.gradient(elev, cell_size)
    mag = np.hypot(gx, gy)
    strong = mag > np.percentile(mag, 75)
    # lineament direction is perpendicular to the gradient
    orient = np.mod(np.degrees(np.arctan2(gy, gx)) + 90.0, 180.0)
    hist, edges = np.histogram(
        orient[strong], bins=18, range=(0.0, 180.0), weights=mag[strong]
    )
    dominant_orientation = float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2.0)

    features: List[Dict[str, Any]] = []
    for kind, mask_src in (
        ("relief_high", residual > relief_threshold),
        ("relief_low", residual < -relief_threshold),
    ):
        lab, n = ndimage.label(mask_src)
        for i in range(1, n + 1):
            comp = lab == i
            if comp.sum() < min_area_cells:
                continue
            geom = _component_geometry(comp, cell_size)
            amp = float(np.abs(residual[comp]).mean())
            linear = geom["elongation"] >= float(min_elongation)
            ftype = (
                ("channel" if kind == "relief_low" else "ridge")
                if linear
                else ("depression" if kind == "relief_low" else "pinnacle")
            )
            confidence = float(
                min(1.0, amp / (2.0 * relief_threshold))
                * min(1.0, comp.sum() / (3.0 * min_area_cells))
            )
            features.append(
                {
                    "type": ftype,
                    "centroid": [geom["centroid_col"] * cell_size,
                                 geom["centroid_row"] * cell_size],
                    "centroid_index": [geom["centroid_row"], geom["centroid_col"]],
                    "length_m": geom["length_m"],
                    "width_m": geom["width_m"],
                    "orientation_deg": geom["orientation_deg"],
                    "area_cells": geom["area_cells"],
                    "mean_relief_m": float(np.sign(residual[comp]).mean() * amp),
                    "confidence": round(confidence, 3),
                }
            )
    features.sort(key=lambda f: -f["confidence"])
    return {
        "features": features,
        "dominant_lineament_orientation_deg": dominant_orientation,
        "residual_grid": residual,
        "n_features": len(features),
    }


# ---------------------------------------------------------------------------
# 5 & 6. deposit models + prospectivity scoring
# ---------------------------------------------------------------------------

DEPOSIT_MODELS: Dict[str, Dict[str, Any]] = {
    "placer_gold": {
        "name": "Offshore placer gold",
        "depth_window_m": [5.0, 60.0],
        "slope_window_deg": [0.0, 12.0],
        "rugosity_favorable": True,
        "rugosity_weight": 0.35,
        "backscatter_preference": "coarse_sediment",
        "backscatter_weight": 0.20,
        "depth_weight": 0.45,
        "description": (
            "Trap sites in palaeochannels, lee sides of bedrock highs and "
            "gravel lags where dense particles settle from bedload."
        ),
        "diagnostic_backscatter": "Moderate-high, patchy coarse lag with scour texture",
        "indicative_terrain": "Palaeochannel margins, bedrock lows, moderate rugosity",
    },
    "marine_diamond": {
        "name": "Marine diamond placer",
        "depth_window_m": [20.0, 140.0],
        "slope_window_deg": [0.0, 8.0],
        "rugosity_favorable": True,
        "rugosity_weight": 0.30,
        "backscatter_preference": "coarse_sediment",
        "backscatter_weight": 0.25,
        "depth_weight": 0.45,
        "description": (
            "Gully and pothole traps in bedrock on drowned shelves; gravel "
            "lags and deflation surfaces concentrate diamonds."
        ),
        "diagnostic_backscatter": "High-relief gravel lag, gully-fill signatures",
        "indicative_terrain": "Bedrock gullies, wave-cut platforms, strong microrelief",
    },
    "tin_placer": {
        "name": "Offshore tin (cassiterite) placer",
        "depth_window_m": [5.0, 50.0],
        "slope_window_deg": [0.0, 10.0],
        "rugosity_favorable": True,
        "rugosity_weight": 0.30,
        "backscatter_preference": "coarse_sediment",
        "backscatter_weight": 0.25,
        "depth_weight": 0.45,
        "description": (
            "Drowned valley and alluvial-channel placers seaward of "
            "cassiterite-bearing granitoids (e.g. SE Asian tin belt)."
        ),
        "diagnostic_backscatter": "Channelised coarse sediment, moderate intensity",
        "indicative_terrain": "Drowned fluvial channels, gentle gradients",
    },
    "sms": {
        "name": "Seafloor massive sulfide (SMS)",
        "depth_window_m": [1000.0, 3500.0],
        "slope_window_deg": [5.0, 45.0],
        "rugosity_favorable": True,
        "rugosity_weight": 0.40,
        "backscatter_preference": "rock",
        "backscatter_weight": 0.25,
        "depth_weight": 0.35,
        "description": (
            "Hydrothermal vent fields at spreading ridges and arc volcanoes; "
            "chimneys and mounds on rugged volcanic seafloor."
        ),
        "diagnostic_backscatter": "Very high, hard rock/chimney returns",
        "indicative_terrain": "Rugged volcanic ridges, fault scarps, steep flanks",
    },
    "polymetallic_nodule": {
        "name": "Polymetallic nodule field",
        "depth_window_m": [3500.0, 6000.0],
        "slope_window_deg": [0.0, 5.0],
        "rugosity_favorable": False,
        "rugosity_weight": 0.35,
        "backscatter_preference": "fine_sediment",
        "backscatter_weight": 0.20,
        "depth_weight": 0.45,
        "description": (
            "Abyssal-plain nodule fields (Mn, Ni, Cu, Co) on slowly "
            "accumulating pelagic sediment below the CCD."
        ),
        "diagnostic_backscatter": "Low, uniform pelagic drape with speckled nodule texture",
        "indicative_terrain": "Flat abyssal plains, very low rugosity, gentle slopes",
    },
}


def deposit_models() -> Dict[str, Any]:
    """Return the marine deposit model presets."""
    return DEPOSIT_MODELS


def _window_score(values: np.ndarray, lo: float, hi: float, decay: float) -> np.ndarray:
    """1.0 inside [lo, hi], Gaussian decay outside."""
    d = np.where(values < lo, lo - values, np.where(values > hi, values - hi, 0.0))
    return np.exp(-((d / decay) ** 2))


def score_targets(
    depth_grid: np.ndarray,
    rugosity: np.ndarray,
    slope: np.ndarray,
    class_map: Optional[np.ndarray] = None,
    model: str = "placer_gold",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Composite marine prospectivity score for a deposit model preset."""
    if model not in DEPOSIT_MODELS:
        raise ValueError(f"unknown deposit model '{model}'")
    m = DEPOSIT_MODELS[model]
    depth = _fill_nans(np.asarray(depth_grid, dtype=float))
    rugosity = np.asarray(rugosity, dtype=float)
    slope = np.asarray(slope, dtype=float)
    if rugosity.shape != depth.shape or slope.shape != depth.shape:
        raise ValueError("terrain grids must match depth grid shape")

    dlo, dhi = m["depth_window_m"]
    depth_decay = max((dhi - dlo) / 2.0, 1.0)
    depth_score = _window_score(depth, dlo, dhi, depth_decay)

    slo, shi = m["slope_window_deg"]
    slope_score = _window_score(slope, slo, shi, max((shi - slo) / 2.0, 1.0))

    rspan = float(rugosity.max() - rugosity.min())
    rnorm = (rugosity - rugosity.min()) / rspan if rspan > 1e-12 else np.zeros_like(rugosity)
    rug_score = rnorm if m["rugosity_favorable"] else (1.0 - rnorm)

    total_w = m["depth_weight"] + m["rugosity_weight"] + m["backscatter_weight"]
    score = (
        m["depth_weight"] * depth_score
        + m["rugosity_weight"] * (0.5 * rug_score + 0.5 * slope_score)
    )
    bs_score = None
    if class_map is not None:
        class_map = np.asarray(class_map)
        if class_map.shape != depth.shape:
            raise ValueError("class_map must match depth grid shape")
        pref_idx = _SEABED_LABELS.index(m["backscatter_preference"])
        n_classes = int(class_map.max()) + 1
        # distance of each cell's class from the preferred class, normalised
        dist = np.abs(class_map.astype(float) - pref_idx) / max(n_classes - 1, 1)
        bs_score = 1.0 - dist
        score += m["backscatter_weight"] * bs_score
    else:
        score += m["backscatter_weight"] * 0.5  # neutral without backscatter
    score = score / total_w

    # top-k zones from high-score connected components
    thresh = np.percentile(score, 90.0)
    lab, n = ndimage.label(score >= thresh)
    zones: List[Dict[str, Any]] = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 3:
            continue
        ys, xs = np.nonzero(comp)
        zones.append(
            {
                "centroid_index": [float(ys.mean()), float(xs.mean())],
                "mean_score": float(score[comp].mean()),
                "max_score": float(score[comp].max()),
                "area_cells": int(comp.sum()),
                "mean_depth_m": float(depth[comp].mean()),
                "mean_rugosity": float(rugosity[comp].mean()),
            }
        )
    zones.sort(key=lambda z: -z["mean_score"])
    for rank, z in enumerate(zones[: max(int(top_k), 1)], start=1):
        z["rank"] = rank
        z["model"] = model
        z["explanation"] = (
            f"{m['name']}: depth {z['mean_depth_m']:.1f} m "
            f"(window {dlo:.0f}-{dhi:.0f} m), "
            f"{'high' if m['rugosity_favorable'] else 'low'} rugosity "
            f"{z['mean_rugosity']:.3f} favourable for "
            f"{m['indicative_terrain'].lower()}."
        )
    return {
        "score_grid": score,
        "component_grids": {
            "depth_score": depth_score,
            "slope_score": slope_score,
            "rugosity_score": rug_score,
            "backscatter_score": bs_score,
        },
        "top_zones": zones[: max(int(top_k), 1)],
        "model": m,
    }
