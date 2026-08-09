"""Innovation 5 — targeting-tiles: interpolated prospectivity surface -> PNG tiles.

Interpolation uses the platform ordinary-kriging core
(`src.api.geostatistics.kriging` / `api.geostatistics.kriging`, dual-context);
when the core is unavailable or underdetermined, falls back to a numpy IDW
implemented here. The fallback is reported honestly in `method`.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .core import (RASTER_REGISTRY, lonlat_to_merc, register_raster,
                   render_tile_png, validate_tile)

router = APIRouter(tags=["geotoolkit-targeting-tiles"])


class Sample(BaseModel):
    x: float
    y: float
    value: float


class HeatmapRequest(BaseModel):
    samples: List[Sample] = Field(..., min_length=3)
    crs: str = Field("EPSG:3857", description="EPSG:3857 or EPSG:4326")
    grid_size: int = Field(50, ge=8, le=400)
    bounds: Optional[List[float]] = Field(None, description="[minx,miny,maxx,maxy]; default = sample extent + 10%")
    colormap: str = Field("iron-oxide")
    method: str = Field("auto", description="auto|kriging|idw")
    idw_power: float = Field(2.0, gt=0)


def _idw(sx, sy, sv, gx, gy, power):
    """Inverse-distance-weighted interpolation; exact at sample points."""
    out = np.empty(gx.shape, dtype=float)
    for idx in np.ndindex(gx.shape):
        d = np.hypot(sx - gx[idx], sy - gy[idx])
        hit = d < 1e-12
        if hit.any():
            out[idx] = sv[hit][0]
            continue
        w = 1.0 / d ** power
        out[idx] = float(np.dot(w, sv) / w.sum())
    return out


def _kriging_grid(sx, sy, sv, gx_flat, gy_flat):
    """Estimate via platform ordinary kriging core. Returns values or None."""
    try:
        try:
            from src.api.geostatistics import kriging as kg  # type: ignore
        except ImportError:
            from api.geostatistics import kriging as kg  # type: ignore
    except ImportError:
        return None
    try:
        var = float(np.var(sv))
        variogram = kg.VariogramModel(
            nugget=0.05 * var,
            structures=[{"model": "spherical", "contribution": 0.95 * var,
                         "range": max(1.0, 0.5 * math.hypot(sx.max() - sx.min(),
                                                            sy.max() - sy.min()))}],
        )
        search = kg.SearchParameters(min_samples=min(3, len(sv)),
                                     max_samples=min(16, len(sv)),
                                     max_per_octant=min(16, len(sv)))
        est = kg.OrdinaryKriging(variogram, search)
        pts = [kg.Point3D(float(x), float(y), 0.0, float(v)) for x, y, v in zip(sx, sy, sv)]
        est.set_data(pts)
        vals = np.full(gx_flat.shape, np.nan)
        for i in range(len(gx_flat)):
            r = est.estimate(kg.Point3D(float(gx_flat[i]), float(gy_flat[i]), 0.0))
            if r is not None:
                vals[i] = r.estimate
        return vals
    except Exception:
        return None


@router.post("/targeting/heatmap")
def targeting_heatmap(req: HeatmapRequest):
    sx = np.array([s.x for s in req.samples], dtype=float)
    sy = np.array([s.y for s in req.samples], dtype=float)
    sv = np.array([s.value for s in req.samples], dtype=float)
    if req.crs.upper().replace(" ", "") in ("EPSG:4326", "4326", "WGS84"):
        conv = np.array([lonlat_to_merc(x, y) for x, y in zip(sx, sy)])
        sx, sy = conv[:, 0], conv[:, 1]

    if req.bounds:
        minx, miny, maxx, maxy = req.bounds
        if req.crs.upper().replace(" ", "") in ("EPSG:4326", "4326", "WGS84"):
            minx, miny = lonlat_to_merc(minx, miny)
            maxx, maxy = lonlat_to_merc(maxx, maxy)
    else:
        pad_x = 0.1 * (sx.max() - sx.min() or 1.0)
        pad_y = 0.1 * (sy.max() - sy.min() or 1.0)
        minx, maxx = sx.min() - pad_x, sx.max() + pad_x
        miny, maxy = sy.min() - pad_y, sy.max() + pad_y

    n = req.grid_size
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(maxy, miny, n)  # rows north->south
    gx, gy = np.meshgrid(xs, ys)

    method_used = "idw"
    grid = None
    if req.method in ("auto", "kriging"):
        vals = _kriging_grid(sx, sy, sv, gx.ravel(), gy.ravel())
        if vals is not None and np.isfinite(vals).all():
            grid = vals.reshape(gx.shape)
            method_used = "kriging"
        elif req.method == "kriging":
            raise HTTPException(503, "platform kriging core unavailable or failed; "
                                     "retry with method='idw'")
    if grid is None:
        grid = _idw(sx, sy, sv, gx, gy, req.idw_power)
        if req.method == "auto":
            method_used = "idw"

    # Honor sample values exactly at sample locations on the grid.
    fx = (sx - minx) / (maxx - minx) * (n - 1)
    fy = (maxy - sy) / (maxy - miny) * (n - 1)
    ix = np.clip(np.round(fx).astype(int), 0, n - 1)
    iy = np.clip(np.round(fy).astype(int), 0, n - 1)
    grid[iy, ix] = sv

    raster = register_raster(grid, [minx, miny, maxx, maxy],
                             colormap=req.colormap, name="targeting-heatmap")
    return {
        "raster_id": raster.id,
        "method": method_used,
        "grid_size": n,
        "bounds_epsg3857": list(raster.bounds),
        "stats": {
            "min": float(np.nanmin(grid)),
            "max": float(np.nanmax(grid)),
            "mean": float(np.nanmean(grid)),
            "std": float(np.nanstd(grid)),
            "n_samples": len(req.samples),
            "sample_value_min": float(sv.min()),
            "sample_value_max": float(sv.max()),
        },
        "tile_url_template": f"/innovations/geotoolkit/targeting/tiles/{raster.id}/{{z}}/{{x}}/{{y}}",
    }


@router.get("/targeting/tiles/{raster_id}/{z}/{x}/{y}")
def targeting_tile(raster_id: str, z: int, x: int, y: int,
                   colormap: Optional[str] = Query(None),
                   size: int = Query(256, ge=64, le=1024)):
    if not validate_tile(z, x, y):
        raise HTTPException(400, "invalid z/x/y for slippy map tile scheme")
    raster = RASTER_REGISTRY.get(raster_id)
    if raster is None:
        raise HTTPException(404, f"raster '{raster_id}' not found")
    png = render_tile_png(raster, z, x, y, size=size, colormap=colormap)
    return Response(content=png, media_type="image/png")
