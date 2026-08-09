"""3. Trace design: dip/azimuth from collar->target geometry, true depth,
expected intercept interval, dogleg deviation model with positional
uncertainty ellipse at target depth (real error propagation)."""

import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from src.api.innovations.drill_planner.common import LocalProjector
except ImportError:  # pragma: no cover
    from api.innovations.drill_planner.common import LocalProjector

router = APIRouter()


class Target(BaseModel):
    depth: float                      # vertical depth below collar (m)
    offset_east: float = 0.0          # horizontal offset from collar (m)
    offset_north: float = 0.0
    thickness: float = 10.0           # true thickness of target (m)
    target_dip: Optional[float] = None  # target plane dip (deg, 90 = vertical)


class HoleIn(BaseModel):
    id: Optional[str] = None
    lon: float
    lat: float
    elevation: float = 0.0
    target: Target


class DesignRequest(BaseModel):
    holes: List[HoleIn]
    dogleg_deg_per_100m: float = 0.5   # random-walk dogleg sigma


def design_hole(hole: HoleIn, dogleg: float) -> Dict[str, Any]:
    t = hole.target
    h_off = math.hypot(t.offset_east, t.offset_north)
    depth = t.depth

    azimuth = (math.degrees(math.atan2(t.offset_east, t.offset_north)) + 360.0) % 360.0         if h_off > 1e-9 else 0.0
    # dip measured from horizontal, positive downward; vertical hole -> 90
    dip = 90.0 if h_off < 1e-9 else math.degrees(math.atan2(depth, h_off))
    true_depth = math.hypot(depth, h_off)

    # expected intercept: angle between hole axis and target plane normal
    target_dip = t.target_dip if t.target_dip is not None else 0.0  # default horizontal
    # angle between hole and target plane = |target_dip - (90 - hole dip from vert)|
    hole_from_vertical = 90.0 - dip
    beta = math.radians(abs(target_dip - hole_from_vertical))
    sinb = max(abs(math.sin(math.pi / 2 - beta)), 1e-6)
    intercept = t.thickness / sinb

    # deviation model: random-walk dogleg sigma_d per 100 m segment.
    # Angular std after n segments: sigma_theta = sigma_d * sqrt(n).
    # Lateral positional std (integrated random walk): sigma_d_rad * step * sqrt(n^3 / 3)
    n = max(true_depth / 100.0, 1e-9)
    sigma_d = math.radians(dogleg)
    step = 100.0
    lateral_std = sigma_d * step * math.sqrt(n ** 3 / 3.0)
    vertical_std = lateral_std * max(math.cos(math.radians(dip)), 0.05)
    # 95% confidence ellipse (2-sigma): major axis lateral, minor vertical
    semi_major = 2.0 * lateral_std
    semi_minor = 2.0 * vertical_std
    return {
        "azimuth_deg": round(azimuth, 4),
        "dip_deg": round(dip, 4),
        "true_depth_m": round(true_depth, 4),
        "horizontal_offset_m": round(h_off, 4),
        "expected_intercept_m": round(intercept, 4),
        "uncertainty": {
            "lateral_std_m": round(lateral_std, 4),
            "vertical_std_m": round(vertical_std, 4),
            "semi_major_m": round(semi_major, 4),
            "semi_minor_m": round(semi_minor, 4),
            "ellipse_area_m2": round(math.pi * semi_major * semi_minor, 4),
            "confidence": 0.95,
            "model": f"random-walk dogleg {dogleg}deg/100m",
        },
    }


@router.post("/traces/design")
def traces_design(req: DesignRequest) -> Dict[str, Any]:
    if not req.holes:
        raise HTTPException(status_code=422, detail="no holes supplied")
    out = []
    for i, h in enumerate(req.holes):
        if h.target.depth <= 0:
            raise HTTPException(status_code=422, detail=f"hole {i}: target depth must be > 0")
        d = design_hole(h, req.dogleg_deg_per_100m)
        d.update({
            "id": h.id or f"H{i:03d}",
            "collar": {"lon": h.lon, "lat": h.lat, "elevation": h.elevation},
        })
        out.append(d)
    return {
        "hole_count": len(out),
        "dogleg_deg_per_100m": req.dogleg_deg_per_100m,
        "holes": out,
    }
