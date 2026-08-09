"""1. Drill-grid pattern generation with real geodesy (pyproj local UTM)."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import shape, Point, box

try:
    from src.api.innovations.drill_planner.common import LocalProjector
except ImportError:  # pragma: no cover
    from api.innovations.drill_planner.common import LocalProjector

router = APIRouter()


class GridRequest(BaseModel):
    # target area: either lon/lat rectangle bounds or a GeoJSON polygon
    bounds: Optional[List[float]] = None      # [minlon, minlat, maxlon, maxlat]
    polygon: Optional[Dict[str, Any]] = None  # GeoJSON geometry
    spacing_along: float                      # metres, along strike
    spacing_across: float                     # metres, across strike
    strike_azimuth: float = 0.0               # degrees clockwise from north
    pattern: str = "square"                   # square | staggered
    clip_to_area: bool = True                 # keep only collars inside polygon/bounds


@router.post("/patterns/grid")
def patterns_grid(req: GridRequest) -> Dict[str, Any]:
    if req.polygon is not None:
        area = shape(req.polygon)
    elif req.bounds is not None:
        if len(req.bounds) != 4:
            raise HTTPException(status_code=422, detail="bounds must be [minlon,minlat,maxlon,maxlat]")
        area = box(*req.bounds)
    else:
        raise HTTPException(status_code=422, detail="provide polygon or bounds")
    if req.spacing_along <= 0 or req.spacing_across <= 0:
        raise HTTPException(status_code=422, detail="spacings must be positive")
    if req.pattern not in ("square", "staggered"):
        raise HTTPException(status_code=422, detail="pattern must be square|staggered")

    c = area.centroid
    proj = LocalProjector(c.x, c.y)
    minx, miny, maxx, maxy = area.bounds
    corners = [proj.to_m(minx, miny), proj.to_m(maxx, miny),
               proj.to_m(maxx, maxy), proj.to_m(minx, maxy)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

    # rotate bounding box into strike frame to size the grid
    az = math.radians(req.strike_azimuth)
    cos_a, sin_a = math.cos(az), math.sin(az)

    def to_strike(x, y):
        dx, dy = x - cx, y - cy
        along = dx * sin_a + dy * cos_a     # north-rotated strike axis
        across = dx * cos_a - dy * sin_a
        return along, across

    def from_strike(along, across):
        dx = along * sin_a + across * cos_a
        dy = along * cos_a - across * sin_a
        return cx + dx, cy + dy

    sa, sc = zip(*[to_strike(x, y) for x, y in corners])
    pad = max(req.spacing_along, req.spacing_across)
    along_min, along_max = min(sa) - pad, max(sa) + pad
    across_min, across_max = min(sc) - pad, max(sc) + pad

    n_along = int(math.floor((along_max - along_min) / req.spacing_along)) + 1
    n_across = int(math.floor((across_max - across_min) / req.spacing_across)) + 1

    collars = []
    idx = 0
    for r in range(n_across):
        across = across_min + r * req.spacing_across
        offset = (req.spacing_along / 2.0) if (req.pattern == "staggered" and r % 2 == 1) else 0.0
        for q in range(n_along):
            along = along_min + q * req.spacing_along + offset
            x, y = from_strike(along, across)
            p = Point(x, y)
            if req.clip_to_area:
                p_ll = Point(*proj.to_lonlat(x, y))
                if not area.covers(p_ll):
                    continue
            lon, lat = proj.to_lonlat(x, y)
            collars.append({
                "id": f"C{idx:04d}",
                "row": r, "col": q,
                "lon": lon, "lat": lat,
                "easting": float(x), "northing": float(y),
                "along_m": float(along), "across_m": float(across),
            })
            idx += 1

    return {
        "pattern": req.pattern,
        "strike_azimuth": req.strike_azimuth,
        "spacing_along": req.spacing_along,
        "spacing_across": req.spacing_across,
        "crs": f"EPSG:{proj.epsg}",
        "count": len(collars),
        "collars": collars,
    }
