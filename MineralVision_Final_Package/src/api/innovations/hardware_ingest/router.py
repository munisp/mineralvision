"""hardware-ingest router — /innovations/hardware-ingest.

HTTP exposure of the REAL parsers in ``src/api/ingestion``:

- LiDAR LAS via the hand-rolled ``LASReader``; LAZ via laspy+lazrs only
  (honest 503 with remediation when the backend is missing — compressed
  point data is never fabricated or silently gunzipped).
- pXRF CSV via the vendor parsers (Olympus/Bruker/generic auto-detect).
- GNSS via NMEA/CSV/GPX parsers (RINEX/KML have no dedicated parser in
  the codebase; that is reported truthfully).
- Downhole well-log LAS 2.0 via the pure-python ``WellLogLASReader``.
- Sentinel-1 InSAR coherence/backscatter change metrics reusing the real
  bi-temporal logic from ``satellite_change_detection``.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

try:  # dual-context imports
    from src.api.ingestion.gnss_ingestion import GNSSFormat, GNSSIngestionPipeline
    from src.api.ingestion.lidar_ingestion import (
        LAZBackendUnavailableError,
        LASReader,
    )
    from src.api.ingestion.welllog_las import WellLogLASError, WellLogLASReader
    from src.api.ingestion.xrf_ingestion import XRFIngestionPipeline, XRFVendor
    from src.api.innovations.satellite_change_detection.logic import detect_changes
except ImportError:  # pragma: no cover
    from api.ingestion.gnss_ingestion import GNSSFormat, GNSSIngestionPipeline
    from api.ingestion.lidar_ingestion import (
        LAZBackendUnavailableError,
        LASReader,
    )
    from api.ingestion.welllog_las import WellLogLASError, WellLogLASReader
    from api.ingestion.xrf_ingestion import XRFIngestionPipeline, XRFVendor
    from api.innovations.satellite_change_detection.logic import detect_changes

router = APIRouter(prefix="/innovations/hardware-ingest", tags=["hardware-ingest"])

_MOCK_OK = os.environ.get("MV_ALLOW_MOCK_FALLBACK", "").lower() == "true"


def _probe(modname: str) -> Dict[str, Any]:
    try:
        m = importlib.import_module(modname)
        return {"available": True, "version": getattr(m, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "version": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _save_upload(file: UploadFile, data: bytes) -> str:
    suffix = os.path.splitext(file.filename or "upload.bin")[1] or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="mv_hw_ingest_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


@router.get("/capabilities")
def capabilities() -> Dict[str, Any]:
    """Truthful hardware/format/backend matrix. Never fails."""
    laspy = _probe("laspy")
    lazrs = _probe("lazrs")
    rasterio = _probe("rasterio")
    laz_ok = laspy["available"] and lazrs["available"]
    return {
        "module": "hardware-ingest",
        "mock_fallback_allowed": _MOCK_OK,
        "hardware": {
            "lidar": {
                "formats": {
                    "las": {"supported": True,
                            "backend": "built-in LASReader (pure python)"},
                    "laz": {"supported": laz_ok,
                            "backend": "laspy+lazrs" if laz_ok else None,
                            "remediation": None if laz_ok else
                                "pip install laspy lazrs"},
                },
            },
            "pxrf": {
                "formats": {"csv": {"supported": True}},
                "vendors": ["olympus_vanta", "olympus_delta",
                            "bruker_s1_titan", "bruker_tracer",
                            "thermo_niton", "sciaps", "generic"],
                "backend": "built-in vendor CSV parsers",
            },
            "gnss": {
                "formats": {
                    "nmea": {"supported": True, "parser": "NMEAParser"},
                    "csv": {"supported": True, "parser": "CSVParser"},
                    "gpx": {"supported": True, "parser": "GPXParser"},
                    "rinex": {"supported": False,
                              "note": "no dedicated RINEX parser in "
                                      "codebase; generic parser fallback "
                                      "only, results not guaranteed"},
                    "kml": {"supported": False,
                            "note": "no dedicated KML parser in codebase; "
                                    "generic parser fallback only, results "
                                    "not guaranteed"},
                },
            },
            "downhole_welllog": {
                "formats": {"las_2_0": {"supported": True,
                                        "backend": "built-in WellLogLASReader "
                                                   "(pure python, no lasio)"}},
            },
            "insar": {
                "inputs": {
                    "json_arrays": {"supported": True, "backend": "numpy"},
                    "geotiff": {"supported": rasterio["available"],
                                "backend": "rasterio" if rasterio["available"]
                                else None,
                                "remediation": None if rasterio["available"]
                                else "pip install rasterio"},
                },
            },
        },
        "backends": {
            "laspy": laspy,
            "lazrs": lazrs,
            "rasterio": rasterio,
            "numpy": _probe("numpy"),
            "scipy": _probe("scipy"),
        },
    }


# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------


@router.post("/lidar/las")
async def ingest_lidar(file: UploadFile = File(...),
                       max_sample_points: int = 500) -> Dict[str, Any]:
    """Parse an uploaded .las (built-in reader) or .laz (laspy+lazrs)."""
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty upload")
    path = _save_upload(file, data)
    try:
        reader = LASReader(path)
        try:
            meta = reader.read_header()
        except LAZBackendUnavailableError as exc:
            # Honest failure: never fabricate decompressed points.
            raise HTTPException(
                503,
                f"{exc} Set MV_ALLOW_MOCK_FALLBACK=true to acknowledge "
                f"synthetic fallbacks elsewhere in the platform, but LAZ "
                f"point data itself is never fabricated here.")
        except ValueError as exc:
            raise HTTPException(422, f"LAS parse error: {exc}")

        points = reader.read_points_array(max_points=100000)
        n = len(points)
        # deterministic decimation for preview
        step = max(1, n // max_sample_points) if n else 1
        sample = points[::step][:max_sample_points]

        return {
            "file_name": file.filename,
            "format": meta.format.value,
            "version": meta.version,
            "backend": "laspy+lazrs" if meta.format.value == "laz"
            else "built-in LASReader",
            "point_count": int(meta.point_count),
            "point_format": reader._header.get("point_format"),
            "crs": reader._header.get("crs"),
            "bounds": {
                "min_x": meta.min_x, "max_x": meta.max_x,
                "min_y": meta.min_y, "max_y": meta.max_y,
                "min_z": meta.min_z, "max_z": meta.max_z,
            },
            "scale": [meta.scale_x, meta.scale_y, meta.scale_z],
            "offset": [meta.offset_x, meta.offset_y, meta.offset_z],
            "point_density": meta.point_density,
            "has_rgb": meta.has_rgb,
            "has_gps_time": meta.has_gps_time,
            "sample_point_count": int(len(sample)),
            "sample_points": [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                 "intensity": float(p[3]), "classification": int(p[4]),
                 "return_number": int(p[5])}
                for p in sample
            ],
        }
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# pXRF
# ---------------------------------------------------------------------------


def _detect_xrf_vendor(text: str) -> str:
    """Content-based vendor detection for pXRF CSV exports."""
    head = text[:4096].lower()
    if "olympus" in head or "vanta" in head or "delta professional" in head:
        return "olympus_vanta"
    if "bruker" in head or "s1 titan" in head or "tracer" in head:
        return "bruker_s1_titan"
    if "niton" in head or "thermo" in head:
        return "thermo_niton"
    if "sciaps" in head or "x-550" in head:
        return "sciaps_x"
    first_line = head.splitlines()[0] if head.splitlines() else ""
    # Bruker-style header: Spectrum/Sample columns + element symbols
    if ("spectrum" in first_line or "sample" in first_line) and any(
            e in first_line for e in ("fe", "cu", "zn")):
        return "bruker_s1_titan"
    return "generic"


@router.post("/xrf/csv")
async def ingest_xrf(file: UploadFile = File(...),
                     vendor: Optional[str] = None) -> Dict[str, Any]:
    """Parse an uploaded pXRF CSV (vendor auto-detected unless given)."""
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty upload")
    text = data.decode("utf-8-sig", errors="replace")
    detected = vendor or _detect_xrf_vendor(text)
    try:
        vendor_enum = XRFVendor(detected)
    except ValueError:
        raise HTTPException(
            422, f"unknown vendor '{detected}'; use one of "
                 f"{[v.value for v in XRFVendor]} or omit for auto-detect")
    path = _save_upload(file, data)
    try:
        pipeline = XRFIngestionPipeline(vendor=vendor_enum)
        try:
            summary = pipeline.ingest(path)
        except Exception as exc:
            raise HTTPException(422, f"XRF parse error: {exc}")

        qc_summary: Dict[str, int] = {}
        for r in pipeline.readings:
            for _elem, flag in r.quality_flags.items():
                for f in str(flag).split(","):
                    if f:
                        qc_summary[f] = qc_summary.get(f, 0) + 1

        return {
            "file_name": file.filename,
            "vendor_detected": detected,
            "reading_count": summary["reading_count"],
            "valid_readings": summary["valid_readings"],
            "elements_detected": summary["elements_detected"],
            "element_statistics": summary["statistics"]["summary"],
            "qc_flags_summary": qc_summary,
            "invalid_readings": summary["reading_count"] - summary["valid_readings"],
        }
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# GNSS
# ---------------------------------------------------------------------------

_GNSS_FMT_MAP = {
    "rinex": GNSSFormat.RINEX_OBS,
    "nmea": GNSSFormat.NMEA,
    "csv": GNSSFormat.CSV,
    "gpx": GNSSFormat.GPX,
    "kml": GNSSFormat.KML,
}
_GNSS_DEDICATED = {"nmea": "NMEAParser", "csv": "CSVParser", "gpx": "GPXParser"}


@router.post("/gnss/{fmt}")
async def ingest_gnss(fmt: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parse uploaded GNSS data (nmea/csv/gpx have dedicated real parsers)."""
    fmt = fmt.lower()
    if fmt not in _GNSS_FMT_MAP:
        raise HTTPException(
            422, f"unsupported fmt '{fmt}'; use one of {sorted(_GNSS_FMT_MAP)}")
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty upload")
    path = _save_upload(file, data)
    try:
        pipeline = GNSSIngestionPipeline()
        try:
            summary = pipeline.ingest(path, format=_GNSS_FMT_MAP[fmt])
        except Exception as exc:
            raise HTTPException(422, f"GNSS parse error: {exc}")

        dedicated = fmt in _GNSS_DEDICATED
        body: Dict[str, Any] = {
            "file_name": file.filename,
            "format": fmt,
            "parser": _GNSS_DEDICATED.get(fmt, "CSVParser (generic fallback — "
                                              "no dedicated parser for this "
                                              "format; results not guaranteed)"),
            "dedicated_parser": dedicated,
            "observation_count": summary.get("observation_count", 0),
            "duration_seconds": summary.get("duration_seconds"),
            "total_distance_m": summary.get("total_distance_m"),
            "bounds": summary.get("bounds"),
            "quality": summary.get("quality"),
        }
        if summary.get("observation_count", 0) == 0 and summary.get("error"):
            body["error"] = summary["error"]
        if not dedicated:
            body["warning"] = (
                f"no dedicated {fmt} parser in codebase; parsed with generic "
                f"fallback. observation_count may be 0 for real {fmt} data.")
        return body
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# downhole well-log LAS 2.0
# ---------------------------------------------------------------------------


@router.post("/welllog/las")
async def ingest_welllog(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parse an uploaded LAS 2.0 downhole well log (pure python)."""
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty upload")
    path = _save_upload(file, data)
    try:
        try:
            log = WellLogLASReader().read(path)
        except WellLogLASError as exc:
            raise HTTPException(422, f"well-log LAS parse error: {exc}")

        curves_meta = []
        for c in log.curves:
            stats = log.curve_stats(c.mnemonic)
            curves_meta.append({
                "mnemonic": c.mnemonic,
                "unit": c.unit,
                "api_code": c.api_code,
                "description": c.description,
                **stats,
            })
        return {
            "file_name": file.filename,
            "standard": "LAS 2.0 (CWLS Log ASCII Standard)",
            "backend": "built-in WellLogLASReader (pure python)",
            "version": log.version,
            "well": log.well,
            "parameters": log.parameters,
            "null_value": log.null_value,
            "n_rows": int(log.data.shape[0]),
            "n_curves": len(log.curves),
            "depth_mnemonic": log.depth_mnemonic,
            "depth_range": log.depth_range,
            "curves": curves_meta,
        }
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Sentinel-1 InSAR coherence / backscatter change
# ---------------------------------------------------------------------------


class InsarChangeRequest(BaseModel):
    coherence_t1: List[List[float]] = Field(..., description="row-major coherence/backscatter at t1")
    coherence_t2: List[List[float]] = Field(..., description="row-major coherence/backscatter at t2")
    abs_threshold: float = Field(default=0.2, ge=0)
    z_threshold: float = Field(default=2.0, ge=0)
    pixel_size: float = Field(default=20.0, gt=0, description="Sentinel-1 GRD ~20 m")


def _change_metrics(a1: np.ndarray, a2: np.ndarray, abs_threshold: float,
                    z_threshold: float, pixel_size: float) -> Dict[str, Any]:
    if a1.shape != a2.shape:
        raise HTTPException(422, f"shape mismatch: {a1.shape} vs {a2.shape}")
    if a1.ndim != 2 or a1.size == 0:
        raise HTTPException(422, "arrays must be non-empty 2-D")
    if not (np.isfinite(a1).all() and np.isfinite(a2).all()):
        raise HTTPException(422, "arrays must contain only finite values")
    # Reuse the real bi-temporal detection logic with single-band cubes.
    res = detect_changes(a1[None, :, :], a2[None, :, :], index="band", band=1,
                         abs_threshold=abs_threshold, z_threshold=z_threshold,
                         morph_open=0, morph_close=0, min_pixels=1,
                         pixel_size=pixel_size)
    delta = a2 - a1
    n = int(delta.size)
    return {
        "shape": list(delta.shape),
        "delta_mean": res["delta_mean"],
        "delta_std": res["delta_std"],
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "abs_threshold": res["abs_threshold"],
        "z_threshold": res["z_threshold"],
        "n_cells": n,
        "n_changed_cells": res["n_changed_pixels"],
        "changed_fraction": res["n_changed_pixels"] / n,
        "n_regions": res["n_regions"],
        "regions": res["regions"],
        "backend": "satellite_change_detection.detect_changes (numpy/scipy)",
        "synthetic": False,
    }


@router.post("/insar/coherence-change")
async def insar_coherence_change(request: Request) -> Dict[str, Any]:
    """Real change metrics between two coherence/backscatter rasters.

    Accepts either:
    - ``application/json`` body: ``{"coherence_t1": [[...]], "coherence_t2":
      [[...]], "abs_threshold": 0.2, "z_threshold": 2.0, "pixel_size": 20.0}``
    - ``multipart/form-data`` with two GeoTIFF uploads (``file_t1``,
      ``file_t2``) — requires rasterio; honest 503 with remediation when
      rasterio is unavailable.
    """
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        try:
            payload = InsarChangeRequest(**(await request.json()))
        except Exception as exc:
            raise HTTPException(422, f"invalid JSON payload: {exc}")
        try:
            a1 = np.asarray(payload.coherence_t1, dtype=np.float64)
            a2 = np.asarray(payload.coherence_t2, dtype=np.float64)
        except Exception as exc:
            raise HTTPException(422, f"invalid arrays: {exc}")
        return _change_metrics(a1, a2, payload.abs_threshold,
                               payload.z_threshold, payload.pixel_size)

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        file_t1 = form.get("file_t1")
        file_t2 = form.get("file_t2")
        if file_t1 is None or file_t2 is None:
            raise HTTPException(422, "multipart form must include file_t1 and file_t2")
        abs_threshold = float(form.get("abs_threshold", 0.2))
        z_threshold = float(form.get("z_threshold", 2.0))
        pixel_size = float(form.get("pixel_size", 20.0))
        try:
            import rasterio  # type: ignore
        except ImportError:
            raise HTTPException(
                503, "rasterio is not installed; cannot read GeoTIFF uploads. "
                     "Remediation: pip install rasterio, or use the JSON "
                     "array form of this endpoint.")
        paths = []
        try:
            for up in (file_t1, file_t2):
                data = await up.read()
                if not data:
                    raise HTTPException(422, "empty GeoTIFF upload")
                paths.append(_save_upload(up, data))
            with rasterio.open(paths[0]) as ds:
                a1 = ds.read(1).astype(np.float64)
            with rasterio.open(paths[1]) as ds:
                a2 = ds.read(1).astype(np.float64)
        finally:
            for p in paths:
                os.unlink(p)
        return _change_metrics(a1, a2, abs_threshold, z_threshold, pixel_size)

    raise HTTPException(
        422, "content-type must be application/json (arrays) or "
             "multipart/form-data (two GeoTIFF uploads)")
