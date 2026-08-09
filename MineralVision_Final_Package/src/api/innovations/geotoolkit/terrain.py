"""Innovation 4 — terrain-profile: DTM elevation profiles and cross-sections."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .core import bilinear_sample, lonlat_to_merc
from .drillhole3d import Collar, SurveyStation, desurvey_trace

router = APIRouter(tags=["geotoolkit-terrain-profile"])


class DTMGrid(BaseModel):
    grid: List[List[float]] = Field(..., description="DTM elevations, rows north->south")
    bounds: List[float] = Field(..., description="[minx, miny, maxx, maxy]")
    crs: str = Field("EPSG:3857")

    def merc_bounds(self) -> List[float]:
        minx, miny, maxx, maxy = self.bounds
        if self.crs.upper().replace(" ", "") in ("EPSG:4326", "4326", "WGS84"):
            minx, miny = lonlat_to_merc(minx, miny)
            maxx, maxy = lonlat_to_merc(maxx, maxy)
        return [minx, miny, maxx, maxy]


class ProfileRequest(BaseModel):
    dtm: DTMGrid
    polyline: List[List[float]] = Field(..., description="[[x, y], ...] vertices, EPSG:3857")
    n_samples: int = Field(200, ge=2, le=5000)


def _sample_profile(dtm: DTMGrid, polyline: List[List[float]], n: int):
    grid = np.asarray(dtm.grid, dtype=float)
    bounds = dtm.merc_bounds()
    pts = [tuple(map(float, p)) for p in polyline]
    # cumulative distance along polyline
    segs, cum = [], [0.0]
    for i in range(1, len(pts)):
        d = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        segs.append(d)
        cum.append(cum[-1] + d)
    total = cum[-1]
    if total <= 0:
        raise HTTPException(422, "polyline has zero length")
    dists = np.linspace(0.0, total, n)
    xs, ys = np.empty(n), np.empty(n)
    seg_idx = 0
    for k, d in enumerate(dists):
        while seg_idx < len(segs) - 1 and d > cum[seg_idx + 1]:
            seg_idx += 1
        s = segs[seg_idx]
        t = 0.0 if s <= 0 else (d - cum[seg_idx]) / s
        xs[k] = pts[seg_idx][0] + t * (pts[seg_idx + 1][0] - pts[seg_idx][0])
        ys[k] = pts[seg_idx][1] + t * (pts[seg_idx + 1][1] - pts[seg_idx][1])
    elev = bilinear_sample(grid, bounds, xs, ys)
    return dists, xs, ys, elev, total


@router.post("/terrain/profile")
def terrain_profile(req: ProfileRequest):
    dists, xs, ys, elev, total = _sample_profile(req.dtm, req.polyline, req.n_samples)
    finite = np.isfinite(elev)
    return {
        "distance": dists.tolist(),
        "elevation": np.where(finite, elev, None).tolist(),
        "x": xs.tolist(),
        "y": ys.tolist(),
        "total_length": total,
        "n_samples": req.n_samples,
        "min_elevation": float(np.nanmin(elev)) if finite.any() else None,
        "max_elevation": float(np.nanmax(elev)) if finite.any() else None,
    }


class SectionHole(BaseModel):
    hole_id: str
    collar: Collar
    survey: List[SurveyStation]
    assays: list = Field(default_factory=list)


class CrossSectionRequest(BaseModel):
    dtm: DTMGrid
    line: List[List[float]] = Field(..., description="section line [[x0,y0],[x1,y1]], EPSG:3857")
    n_samples: int = Field(200, ge=2, le=5000)
    corridor_width: float = Field(100.0, description="half-width (m) for hole inclusion")
    holes: List[SectionHole] = Field(default_factory=list)


@router.post("/terrain/cross-section")
def terrain_cross_section(req: CrossSectionRequest):
    if len(req.line) != 2:
        raise HTTPException(422, "line must be exactly two vertices")
    dists, xs, ys, elev, total = _sample_profile(req.dtm, req.line, req.n_samples)

    (x0, y0), (x1, y1) = req.line
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L  # unit vector along section

    intersections = []
    for hole in req.holes:
        try:
            trace = desurvey_trace(hole.collar, hole.survey)
        except ValueError as exc:
            raise HTTPException(422, f"{hole.hole_id}: {exc}")
        points = []
        for p in trace:
            # signed distance from section line (2D) and chainage along it
            rx, ry = p["easting"] - x0, p["northing"] - y0
            across = rx * (-uy) + ry * ux
            along = rx * ux + ry * uy
            if abs(across) <= req.corridor_width and -1e-6 <= along <= total + 1e-6:
                points.append({"distance": along,
                               "elevation": p["elevation"],
                               "depth": p["depth"],
                               "across_offset": across})
        if points:
            intersections.append({"hole_id": hole.hole_id,
                                  "collar_distance": ((hole.collar.easting - x0) * ux +
                                                      (hole.collar.northing - y0) * uy),
                                  "points": points})

    return {
        "profile": {
            "distance": dists.tolist(),
            "elevation": np.where(np.isfinite(elev), elev, None).tolist(),
        },
        "total_length": total,
        "corridor_width": req.corridor_width,
        "drill_intersections": intersections,
    }
