"""
Drill-core scan pipeline — pure logic (GeologicAI-style core photo + spectral
registration, mineral mapping, alteration logging and photo-derived QC).

Depth registration
------------------
A scanned core-tray photo is registered downhole with a list of *boxes*:
each box maps a pixel-row span ``[row_start, row_end)`` linearly onto a depth
interval ``[start_depth_m, end_depth_m)``.  Registration QC reports the
pixels-per-metre of each box and the *real* depth coverage gaps (missing or
short core) derived from the declared depth intervals.

Mineral indices
---------------
Spectral rows reuse the platform's hyperspectral band-ratio cores
(``hyperspectral_alteration.logic.compute_index`` — ASTER-style ratios after
Rowan & Mars 2003) via a dual-context import; no index math is reimplemented
here.  ``silica`` is added as a generic band ratio (default b9/b8) computed by
the same core.  RGB-only photos fall back to honest colour *proxies* (redness
index for oxidation etc.) and every RGB-derived result is flagged
``proxy: true``.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # dual-context import (repo-root vs package-root execution)
    from src.api.innovations.hyperspectral_alteration import (
        logic as hs_logic)
except ImportError:  # pragma: no cover
    from api.innovations.hyperspectral_alteration import logic as hs_logic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINERAL_CLASSES = ("iron_oxide", "clay", "carbonate", "silica")

# silica as a generic band ratio (computed by the platform core); None where
# the sensor lacks the bands.
SILICA_BANDS: Dict[str, Optional[Tuple[int, int]]] = {
    "aster": (9, 8),
    "landsat8": None,
    "sentinel2": None,
}

DEFAULT_SILICA_THRESHOLD = 1.05

# Default alteration zonation grouping (porphyry-style).
DEFAULT_ZONATION: Dict[str, List[str]] = {
    "potassic": ["silica"],
    "phyllic": ["clay"],
    "propylitic": ["carbonate"],
    "oxidized": ["iron_oxide"],
}

EPS = 1e-9


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def decode_image(image: Any) -> np.ndarray:
    """Decode an RGB core photo from a nested list or a base64 PNG.

    Returns float array of shape (rows, cols, 3) with values in [0, 255].
    """
    if isinstance(image, str):
        from PIL import Image
        raw = base64.b64decode(image)
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=float)
        return arr
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            "image must be a nested list (rows x cols [x 3]) or base64 PNG")
    return arr


def grayscale(img: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an (rows, cols, 3) image."""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's between-class-variance threshold for a 1-D sample."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("no values to threshold")
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < EPS:
        return lo
    hist, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(float)
    p = w / w.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > EPS,
                            (mu_t * omega - mu) ** 2 / denom, -1.0)
    return float(centres[int(np.argmax(sigma_b2))])


# ---------------------------------------------------------------------------
# Depth registration
# ---------------------------------------------------------------------------

def validate_boxes(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Validate and normalise box registration records."""
    out = []
    for i, b in enumerate(boxes):
        try:
            rs = float(b["row_start"]); re = float(b["row_end"])
            sd = float(b["start_depth_m"]); ed = float(b["end_depth_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"box {i}: requires numeric row_start, row_end, "
                f"start_depth_m, end_depth_m") from exc
        if re <= rs:
            raise ValueError(f"box {i}: row_end must exceed row_start")
        if ed <= sd:
            raise ValueError(f"box {i}: end_depth_m must exceed start_depth_m")
        out.append({"row_start": rs, "row_end": re,
                    "start_depth_m": sd, "end_depth_m": ed,
                    "pixels_per_meter": (re - rs) / (ed - sd)})
    return out


def registration_qc(boxes: List[Dict[str, float]],
                    image_rows: int) -> Dict[str, Any]:
    """Registration QC: pixels-per-metre, depth gaps, coverage fraction."""
    for i, b in enumerate(boxes):
        if b["row_end"] > image_rows:
            raise ValueError(
                f"box {i}: row_end {b['row_end']} exceeds image rows "
                f"{image_rows}")
    by_depth = sorted(boxes, key=lambda b: b["start_depth_m"])
    gaps: List[Dict[str, float]] = []
    for prev, nxt in zip(by_depth, by_depth[1:]):
        gap = nxt["start_depth_m"] - prev["end_depth_m"]
        if gap > EPS:
            gaps.append({"from_m": prev["end_depth_m"],
                         "to_m": nxt["start_depth_m"],
                         "length_m": gap})
    span_lo = by_depth[0]["start_depth_m"]
    span_hi = max(b["end_depth_m"] for b in by_depth)
    drilled = sum(b["end_depth_m"] - b["start_depth_m"] for b in by_depth)
    ppms = [b["pixels_per_meter"] for b in boxes]
    return {
        "n_boxes": len(boxes),
        "image_rows": int(image_rows),
        "pixels_per_meter": float(np.median(ppms)),
        "pixels_per_meter_per_box": [float(p) for p in ppms],
        "depth_from_m": float(span_lo),
        "depth_to_m": float(span_hi),
        "depth_span_m": float(span_hi - span_lo),
        "registered_length_m": float(drilled),
        "coverage_fraction": float(drilled / (span_hi - span_lo)),
        "depth_gaps": gaps,
        "n_gap_m": float(sum(g["length_m"] for g in gaps)),
    }


def depth_to_row(boxes: List[Dict[str, float]],
                 depth_m: float) -> Optional[int]:
    """Map a downhole depth to a pixel row, or None inside a coverage gap."""
    for b in sorted(boxes, key=lambda x: x["start_depth_m"]):
        if b["start_depth_m"] - EPS <= depth_m < b["end_depth_m"] - EPS:
            frac = ((depth_m - b["start_depth_m"]) /
                    (b["end_depth_m"] - b["start_depth_m"]))
            return int(b["row_start"] + frac * (b["row_end"] - b["row_start"]))
    return None


def row_to_depth(boxes: List[Dict[str, float]],
                 row: float) -> Optional[float]:
    """Map a pixel row to a downhole depth, or None outside any box."""
    for b in boxes:
        if b["row_start"] - EPS <= row < b["row_end"]:
            frac = (row - b["row_start"]) / (b["row_end"] - b["row_start"])
            return (b["start_depth_m"] +
                    frac * (b["end_depth_m"] - b["start_depth_m"]))
    return None


def depth_intervals(boxes: List[Dict[str, float]],
                    segment_m: float) -> List[Dict[str, float]]:
    """Split registered depth coverage into fixed-length segments.

    Segments never cross a depth gap or a box boundary: a gap truncates the
    current segment.  Returns [{from_m, to_m}] with to-from <= segment_m.
    """
    if segment_m <= 0:
        raise ValueError("segment_m must be positive")
    out: List[Dict[str, float]] = []
    for b in sorted(boxes, key=lambda x: x["start_depth_m"]):
        d = b["start_depth_m"]
        while d < b["end_depth_m"] - EPS:
            nxt = min(d + segment_m, b["end_depth_m"])
            out.append({"from_m": d, "to_m": nxt})
            d = nxt
    return out


# ---------------------------------------------------------------------------
# Spectral classification (reuses hyperspectral_alteration cores)
# ---------------------------------------------------------------------------

def _band_map_for(preset: str,
                  band_map: Optional[Dict[str, Tuple[int, int]]]
                  ) -> Dict[str, Tuple[int, int]]:
    if band_map is not None:
        return band_map
    if preset not in hs_logic.BAND_PRESETS:
        raise ValueError(f"unknown preset: {preset}")
    bm = dict(hs_logic.BAND_PRESETS[preset])
    bm.pop("ndvi", None)  # vegetation mask is meaningless on drill core
    sil = SILICA_BANDS.get(preset)
    if sil is not None:
        bm["silica"] = sil
    return bm


def spectral_indices(bands: Sequence[float],
                     preset: str = "aster",
                     band_map: Optional[Dict[str, Tuple[int, int]]] = None,
                     ) -> Dict[str, float]:
    """Band-ratio indices for one spectral profile row.

    Reuses ``hs_logic.compute_index`` on a (bands, 1, 1) cube — identical math
    to the platform's raster pipeline.
    """
    bands = np.asarray(bands, dtype=float).ravel()
    bm = _band_map_for(preset, band_map)
    cube = bands.reshape(-1, 1, 1)
    out: Dict[str, float] = {}
    for index in ("iron_oxide", "clay", "carbonate", "silica"):
        if index not in bm:
            continue
        num_b, den_b = bm[index]
        if num_b > cube.shape[0] or den_b > cube.shape[0]:
            continue  # sensor lacks the bands for this index
        out[index] = float(hs_logic.compute_index(cube, index, bm)[0, 0])
    return out


def classify_indices(indices: Dict[str, float],
                     thresholds: Optional[Dict[str, float]] = None
                     ) -> Tuple[str, float]:
    """Dominant mineral class = index with the largest threshold-normalised
    score, requiring score >= 1; otherwise ``barren``.

    Returns (class, confidence = winning score).
    """
    thr = dict(hs_logic.DEFAULT_THRESHOLDS)
    thr["silica"] = DEFAULT_SILICA_THRESHOLD
    if thresholds:
        thr.update(thresholds)
    best_cls, best_score = "barren", 0.0
    for cls, val in indices.items():
        t = thr.get(cls, 1.0)
        if t <= 0:
            continue
        score = val / t
        if score > best_score + EPS:
            best_cls, best_score = cls, score
    if best_score < 1.0:
        return "barren", float(best_score)
    return best_cls, float(best_score)


# ---------------------------------------------------------------------------
# RGB proxies (honest photo-derived estimates)
# ---------------------------------------------------------------------------

def rgb_proxy_indices(mean_rgb: np.ndarray) -> Dict[str, float]:
    """Colour proxies from mean RGB of a core segment.

    * ``iron_oxide`` — redness index R / (R + G + B) normalised by the 1/3
      neutral value (hematite/goethite oxidation stains the core red-brown).
    * ``clay``       — brightness proxy (luma / 255): bright, low-chroma
      alteration (white clay/sericite).
    * ``carbonate``  — (G + B) / (2R): carbonate veins are pale grey-green.
    * ``silica``     — 1 - saturation proxy: grey-white silica flooding.
    """
    r, g, b = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
    total = r + g + b
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = max(r, g, b), min(r, g, b)
    saturation = (mx - mn) / mx if mx > EPS else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "iron_oxide": (r / total) / (1.0 / 3.0) if total > EPS else 0.0,
            "clay": luma / 255.0,
            "carbonate": (g + b) / (2.0 * r) if r > EPS else 0.0,
            "silica": 1.0 - saturation,
        }


RGB_PROXY_THRESHOLDS = {
    "iron_oxide": 1.15,   # >15% redder than neutral
    "clay": 0.75,         # bright core
    "carbonate": 1.10,    # green-blue dominant
    "silica": 0.85,       # very low saturation
}


# ---------------------------------------------------------------------------
# Mineral map (downhole log)
# ---------------------------------------------------------------------------

def mineral_map(image: np.ndarray,
                boxes: List[Dict[str, float]],
                spectral_rows: Optional[List[Dict[str, Any]]] = None,
                segment_m: float = 1.0,
                preset: str = "aster",
                band_map: Optional[Dict[str, Tuple[int, int]]] = None,
                thresholds: Optional[Dict[str, float]] = None,
                ) -> Dict[str, Any]:
    """Per-segment mineral classification along the registered depth axis.

    Spectral rows take priority where present (matched by pixel row);
    otherwise RGB proxies are computed from the segment's pixels and flagged
    ``proxy: true``.
    """
    segments = depth_intervals(boxes, segment_m)
    spec_by_row: Dict[int, List[float]] = {}
    if spectral_rows:
        for i, s in enumerate(spectral_rows):
            try:
                spec_by_row[int(s["row"])] = list(s["bands"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"spectral_rows[{i}]: requires integer row and band "
                    f"list") from exc

    ppm = float(np.median([b["pixels_per_meter"] for b in boxes]))
    log: List[Dict[str, Any]] = []
    for seg in segments:
        mid = (seg["from_m"] + seg["to_m"]) / 2.0
        row = depth_to_row(boxes, mid)
        entry: Dict[str, Any] = {
            "depth_from_m": seg["from_m"], "depth_to_m": seg["to_m"],
            "length_m": seg["to_m"] - seg["from_m"],
        }
        if row is None:
            entry.update({"mineral_class": "no_core", "proxy": False,
                          "indices": {}, "confidence": 0.0})
            log.append(entry)
            continue
        entry["pixel_row"] = row
        if spec_by_row:
            tol = max(1, int(round(0.5 * segment_m * ppm)))
            near = min(spec_by_row, key=lambda r: abs(r - row), default=None)
            if near is not None and abs(near - row) <= tol:
                idx = spectral_indices(spec_by_row[near], preset=preset,
                                       band_map=band_map)
                cls, conf = classify_indices(idx, thresholds)
                entry.update({"mineral_class": cls, "proxy": False,
                              "indices": idx, "confidence": conf,
                              "spectral_row": near})
                log.append(entry)
                continue
        # RGB fallback over the segment's pixel rows
        r0 = depth_to_row(boxes, seg["from_m"])
        r1 = depth_to_row(boxes, seg["to_m"] - EPS)
        if r0 is None:
            r0 = row
        if r1 is None:
            r1 = row
        r1 = max(r1, r0) + 1
        mean_rgb = image[r0:r1].reshape(-1, 3).mean(axis=0)
        idx = rgb_proxy_indices(mean_rgb)
        thr = dict(RGB_PROXY_THRESHOLDS)
        if thresholds:
            thr.update(thresholds)
        best_cls, best_score = "barren", 0.0
        for cls, val in idx.items():
            score = val / thr.get(cls, 1.0)
            if score > best_score + EPS:
                best_cls, best_score = cls, score
        if best_score < 1.0:
            best_cls = "barren"
        entry.update({"mineral_class": best_cls, "proxy": True,
                      "indices": idx, "confidence": float(best_score),
                      "mean_rgb": [float(x) for x in mean_rgb]})
        log.append(entry)

    return {"segment_m": float(segment_m),
            "n_segments": len(log),
            "source": "spectral" if spectral_rows else "rgb_proxy",
            "log": log}


# ---------------------------------------------------------------------------
# Alteration log
# ---------------------------------------------------------------------------

def alteration_log(map_log: List[Dict[str, Any]],
                   zonation: Optional[Dict[str, List[str]]] = None
                   ) -> Dict[str, Any]:
    """Merge contiguous same-class segments into an alteration log.

    Runs are merged only across *adjacent* segments.  Lengths are exact
    metres from the registered depths.  A zonation summary groups classes
    into alteration zones (config-driven); CSV rows are emitted alongside
    the JSON structure.
    """
    zone = zonation or DEFAULT_ZONATION
    class_to_zone = {c: z for z, classes in zone.items() for c in classes}

    runs: List[Dict[str, Any]] = []
    for seg in map_log:
        cls = seg["mineral_class"]
        if (runs and runs[-1]["mineral_class"] == cls and
                abs(runs[-1]["depth_to_m"] - seg["depth_from_m"]) < 1e-6):
            runs[-1]["depth_to_m"] = seg["depth_to_m"]
            runs[-1]["length_m"] += seg["length_m"]
            for k, v in seg.get("indices", {}).items():
                runs[-1]["_idx_sum"][k] = \
                    runs[-1]["_idx_sum"].get(k, 0.0) + v
                runs[-1]["_idx_n"][k] = runs[-1]["_idx_n"].get(k, 0) + 1
        else:
            runs.append({
                "depth_from_m": seg["depth_from_m"],
                "depth_to_m": seg["depth_to_m"],
                "length_m": seg["length_m"],
                "mineral_class": cls,
                "alteration_zone": class_to_zone.get(cls, "unzoned"),
                "proxy": bool(seg.get("proxy", False)),
                "_idx_sum": dict(seg.get("indices", {})),
                "_idx_n": {k: 1 for k in seg.get("indices", {})},
            })

    for run in runs:
        run["mean_indices"] = {
            k: run["_idx_sum"][k] / run["_idx_n"][k]
            for k in run["_idx_sum"]}
        del run["_idx_sum"], run["_idx_n"]

    class_m: Dict[str, float] = {}
    zone_m: Dict[str, float] = {}
    for run in runs:
        class_m[run["mineral_class"]] = \
            class_m.get(run["mineral_class"], 0.0) + run["length_m"]
        zone_m[run["alteration_zone"]] = \
            zone_m.get(run["alteration_zone"], 0.0) + run["length_m"]

    csv_rows = [
        "depth_from_m,depth_to_m,length_m,mineral_class,alteration_zone,proxy"]
    for run in runs:
        csv_rows.append(
            f"{run['depth_from_m']:.3f},{run['depth_to_m']:.3f},"
            f"{run['length_m']:.3f},{run['mineral_class']},"
            f"{run['alteration_zone']},{run['proxy']}")

    return {"n_runs": len(runs),
            "runs": runs,
            "zonation": zone,
            "metres_by_class": class_m,
            "metres_by_zone": zone_m,
            "csv": "\n".join(csv_rows) + "\n"}


# ---------------------------------------------------------------------------
# Photo-derived core quality (recovery %, fracture/RQD-style estimate)
# ---------------------------------------------------------------------------

def core_quality(image: np.ndarray,
                 boxes: List[Dict[str, float]],
                 present_threshold: Optional[float] = None,
                 fracture_min_width_px: int = 1,
                 rq_piece_m: float = 0.10,
                 ) -> Dict[str, Any]:
    """Recovery % and an RQD-style fracture estimate from the core photo.

    * Core columns are located first: columns whose mean brightness sits
      clearly above the tray-background level (the tray dominates the dim
      end of the column-mean distribution).
    * Row brightness is then measured over the core columns only, per
      registered box span.  Background and core levels are taken from the
      brightness percentiles; core-present rows sit above
      ``bg + 0.25 * (core - bg)`` (overridable via ``present_threshold``),
      so missing-core gaps are detected as absent rows.
    * Fractures are transverse dark lines: rows *present* as core but darker
      than ``bg + 0.6 * (core - bg)`` — darker than sound core, lighter than
      empty tray; consecutive dark rows merge into one fracture event.
    * Pieces = runs of present, non-fracture rows; the RQD-style value is the
      fraction of recovered length in pieces longer than ``rq_piece_m``
      (default 0.1 m, per the ISRM RQD convention).
    """
    if rq_piece_m <= 0:
        raise ValueError("rq_piece_m must be positive")
    gray = grayscale(image)
    ppm = float(np.median([b["pixels_per_meter"] for b in boxes]))

    rows: List[int] = []
    for b in sorted(boxes, key=lambda x: x["row_start"]):
        rows.extend(range(int(b["row_start"]), int(b["row_end"])))
    if not rows:
        raise ValueError("boxes span no image rows")

    # locate core-strip columns (tray columns cluster at the dim end)
    col_mean = gray[rows].mean(axis=0)
    col_bg = float(np.percentile(col_mean, 25))
    core_cols = np.where(
        col_mean > col_bg + 0.2 * (float(col_mean.max()) - col_bg))[0]
    if core_cols.size == 0:
        core_cols = np.arange(gray.shape[1])

    brightness = gray[np.ix_(rows, core_cols)].mean(axis=1)
    bg_level = float(np.percentile(brightness, 10))
    hi_level = float(np.percentile(brightness, 90))
    if hi_level - bg_level < EPS:
        raise ValueError("image lacks core/background contrast")

    thr = (bg_level + 0.25 * (hi_level - bg_level)
           if present_threshold is None else float(present_threshold))
    present = brightness > thr

    core_level = (float(np.median(brightness[present])) if present.any()
                  else hi_level)
    dark_cut = bg_level + 0.6 * (core_level - bg_level)
    fracture = present & (brightness < dark_cut)

    # merge consecutive fracture rows into single fracture events
    fracture_events: List[Dict[str, Any]] = []
    fracture_rows = np.zeros(len(rows), dtype=bool)
    i = 0
    n = len(rows)
    while i < n:
        if fracture[i]:
            j = i
            while j + 1 < n and fracture[j + 1]:
                j += 1
            if j - i + 1 >= fracture_min_width_px:
                mid_row = (rows[i] + rows[j]) / 2.0
                fracture_events.append({
                    "row_start": rows[i], "row_end": rows[j] + 1,
                    "depth_m": row_to_depth(boxes, mid_row),
                    "width_px": j - i + 1,
                })
                fracture_rows[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    # pieces: runs of present, non-fracture rows
    solid = present & ~fracture_rows
    pieces: List[float] = []
    i = 0
    while i < n:
        if solid[i]:
            j = i
            while j + 1 < n and solid[j + 1]:
                j += 1
            pieces.append((j - i + 1) / ppm)
            i = j + 1
        else:
            i += 1

    present_rows = int(present.sum())
    recovered_m = present_rows / ppm
    good_m = float(sum(p for p in pieces if p >= rq_piece_m - EPS))
    registered_m = sum(b["end_depth_m"] - b["start_depth_m"] for b in boxes)

    return {
        "method": "photo-derived (brightness segmentation + dark-line "
                  "detection); estimate, not a geotech measurement",
        "pixels_per_meter": ppm,
        "present_threshold": float(thr),
        "core_brightness_level": core_level,
        "fracture_dark_cut": float(dark_cut),
        "total_rows": n,
        "core_present_rows": present_rows,
        "recovery_fraction": present_rows / n,
        "recovery_pct": 100.0 * present_rows / n,
        "registered_length_m": float(registered_m),
        "recovered_length_m": recovered_m,
        "n_fractures": len(fracture_events),
        "fractures": fracture_events,
        "fractures_per_meter": (len(fracture_events) / recovered_m
                                if recovered_m > EPS else 0.0),
        "rqd_piece_m": float(rq_piece_m),
        "n_pieces": len(pieces),
        "pieces_gt_rq_length": int(sum(p >= rq_piece_m - EPS
                                       for p in pieces)),
        "rqd_fraction": good_m / recovered_m if recovered_m > EPS else 0.0,
        "rqd_pct": (100.0 * good_m / recovered_m
                    if recovered_m > EPS else 0.0),
    }
