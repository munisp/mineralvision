"""4+5. Campaign optimization (budget-constrained greedy weighted set-cover)
and multi-rig scheduling (nearest-neighbour + 2-opt) with rig-software export."""

import csv
import io
import itertools
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

try:
    from src.api.innovations.drill_planner.common import LocalProjector, haversine_m
except ImportError:  # pragma: no cover
    from api.innovations.drill_planner.common import LocalProjector, haversine_m

router = APIRouter()

# In-memory plan store for export (keyed by plan_id).
_PLANS: Dict[str, Dict[str, Any]] = {}


# ------------------------------------------------------------- optimize
class TargetPoint(BaseModel):
    id: Optional[str] = None
    lon: float
    lat: float
    depth: float = 100.0
    score: float = 1.0


class CandidateHole(BaseModel):
    id: Optional[str] = None
    lon: float
    lat: float
    planned_m: float
    target_ids: Optional[List[str]] = None  # explicit coverage; else nearest-in-radius


class OptimizeRequest(BaseModel):
    candidates: List[CandidateHole]
    targets: List[TargetPoint]
    budget: float
    cost_per_m: float = 250.0
    mobilization_per_hole: float = 5000.0
    max_holes: int = 50
    coverage_radius_m: float = 100.0
    plan_id: Optional[str] = None


@router.post("/campaign/optimize")
def campaign_optimize(req: OptimizeRequest) -> Dict[str, Any]:
    if not req.candidates or not req.targets:
        raise HTTPException(status_code=422, detail="candidates and targets required")
    targets = []
    for i, t in enumerate(req.targets):
        targets.append({"id": t.id or f"T{i:03d}", "lon": t.lon, "lat": t.lat,
                        "score": t.score, "depth": t.depth})
    tby_id = {t["id"]: t for t in targets}
    total_score = sum(t["score"] for t in targets)

    ref = req.targets[0]
    proj = LocalProjector(ref.lon, ref.lat)
    tx, ty = proj.to_m_array([t["lon"] for t in targets], [t["lat"] for t in targets])

    cands = []
    for i, c in enumerate(req.candidates):
        cid = c.id or f"H{i:03d}"
        cost = req.mobilization_per_hole + req.cost_per_m * c.planned_m
        if c.target_ids is not None:
            covers = {tid for tid in c.target_ids if tid in tby_id}
        else:
            cx, cy = proj.to_m(c.lon, c.lat)
            covers = {targets[j]["id"] for j in range(len(targets))
                      if math.hypot(tx[j] - cx, ty[j] - cy) <= req.coverage_radius_m}
        cands.append({"id": cid, "lon": c.lon, "lat": c.lat, "planned_m": c.planned_m,
                      "cost": cost, "covers": covers})

    # greedy weighted set-cover: best marginal covered-score per dollar
    uncovered = set(tby_id)
    selected: List[Dict[str, Any]] = []
    spent = 0.0
    covered_score = 0.0
    remaining = list(cands)
    while remaining and len(selected) < req.max_holes:
        best, best_ratio, best_gain = None, 0.0, 0.0
        for c in remaining:
            if spent + c["cost"] > req.budget:
                continue
            gain = sum(tby_id[t]["score"] for t in (c["covers"] & uncovered))
            if gain <= 0:
                continue
            ratio = gain / c["cost"]
            # tie-break: absolute gain, then id for determinism
            if (ratio, gain, c["id"]) > (best_ratio, best_gain, best["id"] if best else ""):
                best, best_ratio, best_gain = c, ratio, gain
        if best is None:
            break
        selected.append({**best, "covers": sorted(best["covers"]),
                         "marginal_score": best_gain})
        uncovered -= best["covers"]
        covered_score += best_gain
        spent += best["cost"]
        remaining.remove(best)

    plan_id = req.plan_id or f"plan_{len(_PLANS):04d}"
    plan = {
        "plan_id": plan_id,
        "selected": selected,
        "hole_count": len(selected),
        "total_meters": sum(c["planned_m"] for c in selected),
        "total_cost": spent,
        "budget": req.budget,
        "within_budget": spent <= req.budget,
        "covered_targets": sorted(set(tby_id) - uncovered),
        "uncovered_targets": sorted(uncovered),
        "coverage_pct": (100.0 * covered_score / total_score) if total_score > 0 else 0.0,
        "covered_score": covered_score,
        "total_score": total_score,
        "cost_model": {"cost_per_m": req.cost_per_m,
                       "mobilization_per_hole": req.mobilization_per_hole},
    }
    _PLANS[plan_id] = plan
    return plan


# ------------------------------------------------------------- schedule
class SchedHole(BaseModel):
    id: Optional[str] = None
    lon: float
    lat: float
    planned_m: float


class Rig(BaseModel):
    id: Optional[str] = None
    daily_rate_m: float = 60.0
    travel_km_per_day: float = 20.0


class ScheduleRequest(BaseModel):
    holes: List[SchedHole]
    rigs: List[Rig]
    plan_id: Optional[str] = None
    two_opt_iterations: int = 100


def _route_distance(route: List[int], xs, ys) -> float:
    return sum(math.hypot(xs[route[k + 1]] - xs[route[k]],
                          ys[route[k + 1]] - ys[route[k]])
               for k in range(len(route) - 1))


def _nearest_neighbor(n: int, xs, ys) -> List[int]:
    if n == 0:
        return []
    unvisited = set(range(n))
    route = [unvisited.pop()]
    while unvisited:
        last = route[-1]
        nxt = min(unvisited, key=lambda j: (math.hypot(xs[j] - xs[last], ys[j] - ys[last]), j))
        unvisited.remove(nxt)
        route.append(nxt)
    return route


def _two_opt(route: List[int], xs, ys, max_iter: int = 100) -> List[int]:
    best = route[:]
    best_d = _route_distance(best, xs, ys)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                d = _route_distance(cand, xs, ys)
                if d < best_d - 1e-9:
                    best, best_d = cand, d
                    improved = True
    return best


@router.post("/campaign/schedule")
def campaign_schedule(req: ScheduleRequest) -> Dict[str, Any]:
    if not req.holes or not req.rigs:
        raise HTTPException(status_code=422, detail="holes and rigs required")
    holes = [{"id": h.id or f"H{i:03d}", "lon": h.lon, "lat": h.lat,
              "planned_m": h.planned_m} for i, h in enumerate(req.holes)]
    rigs = [{"id": r.id or f"RIG{ri + 1}", "daily_rate_m": r.daily_rate_m,
             "travel_km_per_day": r.travel_km_per_day} for ri, r in enumerate(req.rigs)]

    proj = LocalProjector(holes[0]["lon"], holes[0]["lat"])
    hx, hy = proj.to_m_array([h["lon"] for h in holes], [h["lat"] for h in holes])

    # assign holes to rigs: angle-sort around centroid, contiguous chunks (balanced)
    cx = sum(hx) / len(hx)
    cy = sum(hy) / len(hy)
    order = sorted(range(len(holes)), key=lambda i: math.atan2(hy[i] - cy, hx[i] - cx))
    n_rigs = len(rigs)
    chunks = [order[k::n_rigs] if False else [] for k in range(n_rigs)]
    # contiguous chunking keeps each rig's holes spatially coherent
    per = len(order) // n_rigs
    extra = len(order) % n_rigs
    pos = 0
    for k in range(n_rigs):
        take = per + (1 if k < extra else 0)
        chunks[k] = order[pos:pos + take]
        pos += take

    rig_schedules = []
    all_entries = []
    for rig, chunk in zip(rigs, chunks):
        xs = [hx[i] for i in chunk]
        ys = [hy[i] for i in chunk]
        nn_route = _nearest_neighbor(len(chunk), xs, ys)
        nn_dist = _route_distance(nn_route, xs, ys)
        opt_route = _two_opt(nn_route, xs, ys, req.two_opt_iterations)
        opt_dist = _route_distance(opt_route, xs, ys)

        day = 0.0
        entries = []
        drill_days = 0.0
        travel_days = 0.0
        prev = None
        for local_idx in opt_route:
            h = holes[chunk[local_idx]]
            if prev is not None:
                t_days = math.hypot(xs[local_idx] - xs[prev], ys[local_idx] - ys[prev])                     / 1000.0 / rig["travel_km_per_day"]
                day += t_days
                travel_days += t_days
            d_days = h["planned_m"] / rig["daily_rate_m"]
            entries.append({
                "hole_id": h["id"], "rig_id": rig["id"],
                "lon": h["lon"], "lat": h["lat"], "planned_m": h["planned_m"],
                "start_day": round(day, 4), "end_day": round(day + d_days, 4),
            })
            day += d_days
            drill_days += d_days
            prev = local_idx
        span = day
        rig_schedules.append({
            "rig_id": rig["id"],
            "holes": len(chunk),
            "route_order": [holes[chunk[i]]["id"] for i in opt_route],
            "nn_travel_m": round(nn_dist, 2),
            "optimized_travel_m": round(opt_dist, 2),
            "travel_days": round(travel_days, 4),
            "drill_days": round(drill_days, 4),
            "span_days": round(span, 4),
            "utilization": round(drill_days / span, 4) if span > 0 else 0.0,
        })
        all_entries.extend(entries)

    makespan = max((r["span_days"] for r in rig_schedules), default=0.0)
    plan_id = req.plan_id or f"sched_{len(_PLANS):04d}"
    plan = {
        "plan_id": plan_id,
        "rigs": rig_schedules,
        "schedule": sorted(all_entries, key=lambda e: (e["rig_id"], e["start_day"])),
        "hole_count": len(all_entries),
        "total_meters": sum(h["planned_m"] for h in holes),
        "makespan_days": round(makespan, 4),
        "holes_assigned": len(all_entries),
    }
    _PLANS[plan_id] = plan
    return plan


# ------------------------------------------------------------- export
@router.get("/campaign/export")
def campaign_export(plan_id: str = Query(...),
                    format: str = Query(default="json", pattern="^(json|csv)$")):
    if plan_id not in _PLANS:
        raise HTTPException(status_code=404, detail=f"plan '{plan_id}' not found")
    plan = _PLANS[plan_id]
    # normalize rows: rig-software compatible fields
    if "schedule" in plan:  # scheduled plan
        rows = plan["schedule"]
        extra = {}
    else:  # optimized plan
        rows = [{"hole_id": c["id"], "rig_id": "", "lon": c["lon"], "lat": c["lat"],
                 "planned_m": c["planned_m"], "start_day": "", "end_day": ""}
                for c in plan["selected"]]
        extra = {}
    fields = ["hole_id", "rig_id", "lon", "lat", "planned_m", "start_day", "end_day",
              "azimuth", "dip", "planned_depth"]
    norm_rows = []
    for r in rows:
        norm_rows.append({
            "hole_id": r.get("hole_id", r.get("id")),
            "rig_id": r.get("rig_id", ""),
            "lon": r.get("lon"), "lat": r.get("lat"),
            "azimuth": r.get("azimuth_deg", r.get("azimuth", "")),
            "dip": r.get("dip_deg", r.get("dip", "")),
            "planned_depth": r.get("planned_m", r.get("true_depth_m", "")),
            "start_day": r.get("start_day", ""), "end_day": r.get("end_day", ""),
        })
    if format == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["hole_id", "rig_id", "lon", "lat",
                                            "azimuth", "dip", "planned_depth",
                                            "start_day", "end_day"])
        w.writeheader()
        w.writerows(norm_rows)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {"plan_id": plan_id, "format": "json", "rows": norm_rows, "count": len(norm_rows)}
