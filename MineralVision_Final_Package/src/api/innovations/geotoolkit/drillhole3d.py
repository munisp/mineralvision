"""Innovation 3 — drillhole-3d: minimum-curvature desurvey + three.js scene.

Desurvey uses the industry-standard minimum curvature method
(radius-of-curvature balancing / ratio factor):

    cos(dogleg) = cos(I1)*cos(I2) + sin(I1)*sin(I2)*cos(A2 - A1)
    RF = (2 / dogleg) * tan(dogleg / 2)          (1.0 for a straight interval)
    dEast  = MD/2 * (sin I1 * sin A1 + sin I2 * sin A2) * RF
    dNorth = MD/2 * (sin I1 * cos A1 + sin I2 * cos A2) * RF
    dDown  = MD/2 * (cos I1 + cos I2) * RF

Conventions: azimuth 0 = north, clockwise; dip in degrees below horizontal
(negative = downward, e.g. -60); inclination I is measured from vertical
(I = 90 + dip). Coordinates are easting/northing/elevation (EPSG of caller,
typically a local grid or UTM).
"""

from __future__ import annotations

import math
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["geotoolkit-drillhole-3d"])


class SurveyStation(BaseModel):
    depth: float = Field(..., description="measured depth down hole (m)")
    azimuth: float = Field(..., description="degrees, 0=north clockwise")
    dip: float = Field(..., description="degrees below horizontal (negative = down)")


class Collar(BaseModel):
    easting: float
    northing: float
    elevation: float = 0.0


class DesurveyRequest(BaseModel):
    collar: Collar
    survey: List[SurveyStation]
    step: Optional[float] = Field(None, description="resample step along trace (m)")


def _ratio_factor(dogleg: float) -> float:
    if dogleg < 1e-12:
        return 1.0
    return (2.0 / dogleg) * math.tan(dogleg / 2.0)


def desurvey_trace(collar: Collar, survey: List[SurveyStation]) -> List[dict]:
    """Minimum-curvature desurvey -> list of {depth, easting, northing, elevation}."""
    if not survey:
        raise ValueError("survey must contain at least one station")
    stations = sorted(survey, key=lambda s: s.depth)
    for i in range(1, len(stations)):
        if stations[i].depth <= stations[i - 1].depth:
            raise ValueError("survey depths must be strictly increasing")

    trace = [{
        "depth": 0.0,
        "easting": collar.easting,
        "northing": collar.northing,
        "elevation": collar.elevation,
    }]
    # Surface station attitude = first station attitude (straight above).
    prev_i = math.radians(90.0 + stations[0].dip)
    prev_a = math.radians(stations[0].azimuth)
    prev_depth = 0.0

    for st in stations:
        inc = math.radians(90.0 + st.dip)
        az = math.radians(st.azimuth)
        md = st.depth - prev_depth

        cos_dl = (math.cos(prev_i) * math.cos(inc) +
                  math.sin(prev_i) * math.sin(inc) * math.cos(az - prev_a))
        cos_dl = max(-1.0, min(1.0, cos_dl))
        dogleg = math.acos(cos_dl)
        rf = _ratio_factor(dogleg)

        d_east = md / 2.0 * (math.sin(prev_i) * math.sin(prev_a) +
                             math.sin(inc) * math.sin(az)) * rf
        d_north = md / 2.0 * (math.sin(prev_i) * math.cos(prev_a) +
                              math.sin(inc) * math.cos(az)) * rf
        d_down = md / 2.0 * (math.cos(prev_i) + math.cos(inc)) * rf

        last = trace[-1]
        trace.append({
            "depth": st.depth,
            "easting": last["easting"] + d_east,
            "northing": last["northing"] + d_north,
            "elevation": last["elevation"] - d_down,
        })
        prev_i, prev_a, prev_depth = inc, az, st.depth
    return trace


def resample_trace(trace: List[dict], step: float) -> List[dict]:
    """Linearly interpolate the polyline at regular measured-depth spacing."""
    out = [dict(trace[0])]
    for i in range(1, len(trace)):
        a, b = trace[i - 1], trace[i]
        d = out[-1]["depth"]
        while d + step <= b["depth"] + 1e-9:
            d = d + step
            t = (d - a["depth"]) / (b["depth"] - a["depth"])
            out.append({
                "depth": d,
                "easting": a["easting"] + t * (b["easting"] - a["easting"]),
                "northing": a["northing"] + t * (b["northing"] - a["northing"]),
                "elevation": a["elevation"] + t * (b["elevation"] - a["elevation"]),
            })
    if out[-1]["depth"] < trace[-1]["depth"]:
        out.append(dict(trace[-1]))
    return out


@router.post("/drillholes/desurvey")
def desurvey_endpoint(req: DesurveyRequest):
    try:
        trace = desurvey_trace(req.collar, req.survey)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if req.step and req.step > 0:
        trace = resample_trace(trace, req.step)
    return {
        "hole": {"collar": req.collar.model_dump(),
                 "n_stations": len(req.survey),
                 "max_depth": max(s.depth for s in req.survey)},
        "trace": trace,
        "trace_vertices": [[p["easting"], p["northing"], p["elevation"]] for p in trace],
    }


# ---------------------------------------------------------------------------
# three.js-ready scene
# ---------------------------------------------------------------------------

class AssayInterval(BaseModel):
    from_depth: float
    to_depth: float
    value: float = Field(..., description="grade value used for segment coloring")
    element: str = "grade"


class SceneHole(BaseModel):
    hole_id: str
    collar: Collar
    survey: List[SurveyStation]
    assays: List[AssayInterval] = Field(default_factory=list)


class SceneRequest(BaseModel):
    holes: List[SceneHole]
    step: float = Field(10.0, description="trace resample step (m)")


def _grade_color(t: float) -> List[float]:
    """Blue -> green -> yellow -> red ramp, t in [0,1]. Returns RGB 0-1."""
    t = max(0.0, min(1.0, t))
    if t < 0.33:
        f = t / 0.33
        rgb = (0.0, f, 1.0 - 0.5 * f)
    elif t < 0.66:
        f = (t - 0.33) / 0.33
        rgb = (f, 1.0, 0.5 - 0.5 * f)
    else:
        f = (t - 0.66) / 0.34
        rgb = (1.0, 1.0 - f, 0.0)
    return [round(c, 4) for c in rgb]


def _position_on_trace(trace: List[dict], depth: float) -> List[float]:
    depth = max(trace[0]["depth"], min(depth, trace[-1]["depth"]))
    for i in range(1, len(trace)):
        if depth <= trace[i]["depth"]:
            a, b = trace[i - 1], trace[i]
            span = b["depth"] - a["depth"]
            t = 0.0 if span <= 0 else (depth - a["depth"]) / span
            return [a["easting"] + t * (b["easting"] - a["easting"]),
                    a["northing"] + t * (b["northing"] - a["northing"]),
                    a["elevation"] + t * (b["elevation"] - a["elevation"])]
    p = trace[-1]
    return [p["easting"], p["northing"], p["elevation"]]


@router.post("/drillholes/scene")
def scene_endpoint(req: SceneRequest):
    scene_holes = []
    for hole in req.holes:
        try:
            trace = desurvey_trace(hole.collar, hole.survey)
        except ValueError as exc:
            raise HTTPException(422, f"{hole.hole_id}: {exc}")
        if req.step > 0:
            trace = resample_trace(trace, req.step)

        vmax = max((a.value for a in hole.assays), default=0.0)
        vmin = min((a.value for a in hole.assays), default=0.0)
        span = (vmax - vmin) or 1.0
        segments = []
        for a in hole.assays:
            p0 = _position_on_trace(trace, a.from_depth)
            p1 = _position_on_trace(trace, a.to_depth)
            t = (a.value - vmin) / span
            segments.append({
                "from_depth": a.from_depth,
                "to_depth": a.to_depth,
                "value": a.value,
                "element": a.element,
                "start": p0,
                "end": p1,
                "color_rgb": _grade_color(t),
            })

        scene_holes.append({
            "hole_id": hole.hole_id,
            "collar": {"position": [hole.collar.easting, hole.collar.northing,
                                    hole.collar.elevation]},
            "trace_vertices": [[p["easting"], p["northing"], p["elevation"]]
                               for p in trace],
            "segments": segments,
            "grade_range": [vmin, vmax],
        })
    return {
        "type": "DrillholeScene",
        "coordinate_order": ["easting", "northing", "elevation"],
        "n_holes": len(scene_holes),
        "holes": scene_holes,
    }
