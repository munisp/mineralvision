"""Innovation 1 — raster-tiles: XYZ PNG tiles from registered rasters."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .core import (COLORMAPS, RASTER_REGISTRY, lonlat_to_merc, register_raster,
                   render_tile_png, validate_tile)

router = APIRouter(tags=["geotoolkit-raster-tiles"])


class RasterRegisterRequest(BaseModel):
    grid: List[List[float]] = Field(..., description="2D grid, rows north->south")
    bounds: List[float] = Field(..., description="[minx, miny, maxx, maxy]")
    crs: str = Field("EPSG:3857", description="EPSG:3857 or EPSG:4326")
    colormap: str = Field("viridis")
    name: Optional[str] = None


@router.post("/tiles/raster/register")
def register_raster_endpoint(req: RasterRegisterRequest):
    if len(req.bounds) != 4:
        raise HTTPException(422, "bounds must be [minx, miny, maxx, maxy]")
    if req.colormap not in COLORMAPS:
        raise HTTPException(422, f"unknown colormap; available: {sorted(COLORMAPS)}")
    minx, miny, maxx, maxy = req.bounds
    crs = req.crs.upper().replace(" ", "")
    if crs in ("EPSG:4326", "4326", "WGS84"):
        minx, miny = lonlat_to_merc(minx, miny)
        maxx, maxy = lonlat_to_merc(maxx, maxy)
    elif crs not in ("EPSG:3857", "3857", "WEBMERCATOR"):
        raise HTTPException(422, "supported crs: EPSG:3857 or EPSG:4326")
    try:
        raster = register_raster(np.array(req.grid, dtype=float),
                                 [minx, miny, maxx, maxy],
                                 colormap=req.colormap, name=req.name or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {
        "raster_id": raster.id,
        "name": raster.name,
        "shape": list(raster.grid.shape),
        "bounds_epsg3857": list(raster.bounds),
        "colormap": raster.colormap,
        "vmin": raster.vmin,
        "vmax": raster.vmax,
    }


@router.get("/tiles/raster/{z}/{x}/{y}")
def get_raster_tile(z: int, x: int, y: int,
                    raster_id: str = Query(...),
                    colormap: Optional[str] = Query(None),
                    size: int = Query(256, ge=64, le=1024)):
    if not validate_tile(z, x, y):
        raise HTTPException(400, "invalid z/x/y for slippy map tile scheme")
    raster = RASTER_REGISTRY.get(raster_id)
    if raster is None:
        raise HTTPException(404, f"raster '{raster_id}' not registered")
    if colormap is not None and colormap not in COLORMAPS:
        raise HTTPException(422, f"unknown colormap; available: {sorted(COLORMAPS)}")
    png = render_tile_png(raster, z, x, y, size=size, colormap=colormap)
    return Response(content=png, media_type="image/png")
