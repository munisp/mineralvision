"""Tests for drill_planner (competitive gap #11). Seeded, real assertions."""

import math
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "MineralVision_Final_Package", "src"))

from api.innovations.drill_planner import router  # noqa: E402

app = FastAPI()
app.include_router(router)
client = TestClient(app)
BASE = "/innovations/drill-planner"

# reference area near Kalgoorlie (UTM zone 51 south)
MINLON, MINLAT, MAXLON, MAXLAT = 121.40, -30.80, 121.42, -30.79


def test_grid_spacing_exact_in_utm():
    r = client.post(f"{BASE}/patterns/grid", json={
        "bounds": [MINLON, MINLAT, MAXLON, MAXLAT],
        "spacing_along": 100.0, "spacing_across": 50.0,
        "strike_azimuth": 30.0, "pattern": "square"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] > 4
    collars = body["collars"]
    rows = {}
    for c in collars:
        rows.setdefault(c["row"], []).append(c)
    # same row, adjacent columns -> exact along-strike spacing
    row = next(sorted(rows[k], key=lambda c: c["col"]) for k in rows if len(rows[k]) >= 2)
    d = math.hypot(row[1]["easting"] - row[0]["easting"],
                   row[1]["northing"] - row[0]["northing"])
    assert d == pytest.approx(100.0, abs=1e-6)
    # same column, adjacent rows -> exact across-strike spacing
    by_key = {(c["row"], c["col"]): c for c in collars}
    pair = next((by_key[(r, q)], by_key[(r + 1, q)]) for (r, q) in by_key
                if (r + 1, q) in by_key)
    d2 = math.hypot(pair[1]["easting"] - pair[0]["easting"],
                    pair[1]["northing"] - pair[0]["northing"])
    assert d2 == pytest.approx(50.0, abs=1e-6)
    assert body["crs"] == "EPSG:32751"


def test_grid_staggered_offsets_half_spacing():
    r = client.post(f"{BASE}/patterns/grid", json={
        "bounds": [MINLON, MINLAT, MAXLON, MAXLAT],
        "spacing_along": 100.0, "spacing_across": 50.0,
        "strike_azimuth": 0.0, "pattern": "staggered"})
    collars = r.json()["collars"]
    rows = {}
    for c in collars:
        rows.setdefault(c["row"], []).append(c)
    keys = sorted(rows)
    even = sorted(rows[keys[0]], key=lambda c: c["col"])[0]
    odd = sorted(rows[keys[1]], key=lambda c: c["col"])[0]
    # odd row shifted ~50 m along strike relative to even row
    dx = abs(odd["easting"] - even["easting"])
    assert dx == pytest.approx(50.0, abs=1e-6)


def test_slope_rejection_on_steep_dtm():
    # 21x21 DTM, flat 1000 m with a spike at the centre
    n = 21
    dtm = [[1000.0] * n for _ in range(n)]
    dtm[10][10] = 1500.0  # 500 m spike
    b = [MINLON, MINLAT, MAXLON, MAXLAT]
    mid_lon = (MINLON + MAXLON) / 2
    mid_lat = (MINLAT + MAXLAT) / 2
    # collar adjacent to the spike cell sees huge slope; corner collar is flat
    step_lon = (MAXLON - MINLON) / (n - 1)
    step_lat = (MAXLAT - MINLAT) / (n - 1)
    collars = [
        {"id": "steep", "lon": mid_lon + step_lon, "lat": mid_lat},
        {"id": "flat", "lon": MINLON + step_lon, "lat": MINLAT + step_lat},
    ]
    r = client.post(f"{BASE}/collars/snap", json={
        "collars": collars, "dtm": dtm, "dtm_bounds": b, "max_slope_deg": 25.0})
    body = r.json()
    assert body["accepted_count"] == 1
    assert body["rejected_count"] == 1
    assert body["rejected"][0]["id"] == "steep"
    assert any("slope" in reason for reason in body["rejected"][0]["reasons"])
    assert body["accepted"][0]["id"] == "flat"
    assert body["accepted"][0]["slope_deg"] == pytest.approx(0.0, abs=1e-6)
    # elevation at flat collar interpolated from DTM
    assert body["accepted"][0]["elevation"] == pytest.approx(1000.0)


def test_keepout_exclusion_count_exact():
    keepout = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [121.405, -30.800], [121.410, -30.800],
                [121.410, -30.795], [121.405, -30.795],
                [121.405, -30.800]]]},
            "properties": {"name": "water"}}]}
    collars = [
        {"id": "in1", "lon": 121.407, "lat": -30.797},
        {"id": "in2", "lon": 121.409, "lat": -30.796},
        {"id": "out", "lon": 121.415, "lat": -30.797},
    ]
    r = client.post(f"{BASE}/collars/snap",
                    json={"collars": collars, "keepouts": keepout})
    body = r.json()
    assert body["rejected_count"] == 2
    assert body["accepted_count"] == 1
    assert {c["id"] for c in body["rejected"]} == {"in1", "in2"}
    assert all(any("keep-out" in x for x in c["reasons"]) for c in body["rejected"])


def test_trace_design_vertical_hole():
    r = client.post(f"{BASE}/traces/design", json={
        "holes": [{"id": "V1", "lon": 121.41, "lat": -30.795,
                   "target": {"depth": 250.0, "thickness": 20.0}}]})
    hole = r.json()["holes"][0]
    assert hole["dip_deg"] == pytest.approx(90.0)
    assert hole["true_depth_m"] == pytest.approx(250.0)
    assert hole["horizontal_offset_m"] == pytest.approx(0.0)
    # vertical hole through horizontal target: intercept == thickness
    assert hole["expected_intercept_m"] == pytest.approx(20.0)


def test_trace_design_angled_and_uncertainty_grows():
    r = client.post(f"{BASE}/traces/design", json={
        "holes": [
            {"id": "shallow", "lon": 121.41, "lat": -30.795,
             "target": {"depth": 100.0, "thickness": 10.0}},
            {"id": "deep", "lon": 121.41, "lat": -30.795,
             "target": {"depth": 400.0, "thickness": 10.0}},
            {"id": "offset", "lon": 121.41, "lat": -30.795,
             "target": {"depth": 300.0, "offset_east": 100.0,
                        "offset_north": 0.0, "thickness": 10.0}},
        ]})
    holes = {h["id"]: h for h in r.json()["holes"]}
    s, d, o = holes["shallow"], holes["deep"], holes["offset"]
    assert d["uncertainty"]["semi_major_m"] > s["uncertainty"]["semi_major_m"]
    assert d["uncertainty"]["ellipse_area_m2"] > s["uncertainty"]["ellipse_area_m2"]
    # offset hole: azimuth due east, true depth = hypot(300, 100)
    assert o["azimuth_deg"] == pytest.approx(90.0)
    assert o["true_depth_m"] == pytest.approx(math.hypot(300.0, 100.0))
    assert o["dip_deg"] == pytest.approx(math.degrees(math.atan2(300.0, 100.0)))


def test_optimize_respects_budget_picks_best_first():
    # 1:1 candidates covering their own target; budget fits only the two best
    targets = [
        {"id": "T-low", "lon": 121.405, "lat": -30.795, "score": 1.0, "depth": 100},
        {"id": "T-mid", "lon": 121.410, "lat": -30.795, "score": 5.0, "depth": 100},
        {"id": "T-high", "lon": 121.415, "lat": -30.795, "score": 9.0, "depth": 100},
    ]
    candidates = [
        {"id": "H-low", "lon": 121.405, "lat": -30.795, "planned_m": 100,
         "target_ids": ["T-low"]},
        {"id": "H-mid", "lon": 121.410, "lat": -30.795, "planned_m": 100,
         "target_ids": ["T-mid"]},
        {"id": "H-high", "lon": 121.415, "lat": -30.795, "planned_m": 100,
         "target_ids": ["T-high"]},
    ]
    # each hole costs 5000 + 250*100 = 30000 ; budget 65000 -> only 2 holes
    r = client.post(f"{BASE}/campaign/optimize", json={
        "candidates": candidates, "targets": targets, "budget": 65000.0,
        "cost_per_m": 250.0, "mobilization_per_hole": 5000.0, "max_holes": 10})
    body = r.json()
    assert body["hole_count"] == 2
    assert body["total_cost"] <= 65000.0
    assert body["within_budget"] is True
    picked = [c["id"] for c in body["selected"]]
    assert picked[0] == "H-high"   # highest score picked first
    assert picked[1] == "H-mid"
    assert body["coverage_pct"] == pytest.approx(100.0 * (9 + 5) / 15)
    assert body["total_meters"] == pytest.approx(200.0)


def test_schedule_assigns_all_two_opt_improves_and_export():
    import numpy as np
    rng = np.random.default_rng(7)
    holes = [{"id": f"H{i:02d}", "lon": 121.40 + 0.03 * float(v[0]),
              "lat": -30.80 + 0.015 * float(v[1]),
              "planned_m": float(120 + 60 * v[2])}
             for i, v in enumerate(rng.random((8, 3)))]
    r = client.post(f"{BASE}/campaign/schedule", json={
        "holes": holes,
        "rigs": [{"id": "R1", "daily_rate_m": 80.0},
                 {"id": "R2", "daily_rate_m": 60.0}],
        "plan_id": "test-sched"})
    body = r.json()
    assert body["holes_assigned"] == 8
    assigned = {e["hole_id"] for e in body["schedule"]}
    assert assigned == {h["id"] for h in holes}
    for rig in body["rigs"]:
        assert rig["optimized_travel_m"] <= rig["nn_travel_m"] + 1e-9
        assert 0.0 < rig["utilization"] <= 1.0
    # per-rig timing is non-overlapping and sequential
    for rig_id in ("R1", "R2"):
        entries = sorted([e for e in body["schedule"] if e["rig_id"] == rig_id],
                         key=lambda e: e["start_day"])
        for a, b in zip(entries, entries[1:], strict=False):
            assert b["start_day"] >= a["end_day"] - 1e-9

    # export CSV + JSON
    rc = client.get(f"{BASE}/campaign/export",
                    params={"plan_id": "test-sched", "format": "csv"})
    assert rc.status_code == 200
    lines = list(rc.text.strip().splitlines())
    assert lines[0].startswith("hole_id")
    assert len(lines) == 9  # header + 8 holes
    rj = client.get(f"{BASE}/campaign/export",
                    params={"plan_id": "test-sched", "format": "json"})
    assert rj.json()["count"] == 8
    assert all("azimuth" in row and "planned_depth" in row
               for row in rj.json()["rows"])
    r404 = client.get(f"{BASE}/campaign/export", params={"plan_id": "nope"})
    assert r404.status_code == 404
