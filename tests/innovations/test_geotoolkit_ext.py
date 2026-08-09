"""Tests for geotoolkit_ext innovations 6-10. Seeded, real assertions, no mocks."""

import base64
import io
import math
import os
import sys

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "MineralVision_Final_Package", "src"))

from api.innovations.geotoolkit_ext import router  # noqa: E402

app = FastAPI()
app.include_router(router)
client = TestClient(app)

BASE = "/innovations/geotoolkit-ext"


def square(minx, miny, maxx, maxy, props=None):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]},
        "properties": props or {},
    }


def fc(features):
    return {"type": "FeatureCollection", "features": features}


# ------------------------------------------------------------- 6. overlay
def test_overlay_intersect_known_squares():
    a = fc([square(0, 0, 4, 4, {"name": "target"})])
    b = fc([square(2, 2, 6, 6, {"score": 0.8})])
    r = client.post(f"{BASE}/overlay/intersect", json={"layer_a": a, "layer_b": b})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feature_count"] == 1
    assert body["total_area"] == pytest.approx(4.0)  # 2x2 overlap
    props = body["result"]["features"][0]["properties"]
    assert props["name"] == "target"
    # area-weighted transfer: overlap fraction of b = 4/16 = 0.25
    assert props["overlap_fraction_of_b"] == pytest.approx(0.25)
    assert props["score_weighted"] == pytest.approx(0.8 * 0.25)


def test_overlay_union_erase_clip():
    a = fc([square(0, 0, 4, 4)])
    b = fc([square(2, 0, 6, 4)])
    r = client.post(f"{BASE}/overlay/union", json={"layer_a": a, "layer_b": b})
    assert r.status_code == 200
    assert r.json()["total_area"] == pytest.approx(24.0)  # 16+16-8

    r = client.post(f"{BASE}/overlay/erase", json={"layer_a": a, "layer_b": b})
    assert r.json()["total_area"] == pytest.approx(8.0)  # left half of a

    r = client.post(f"{BASE}/overlay/clip", json={"layer_a": a, "layer_b": b})
    assert r.json()["total_area"] == pytest.approx(8.0)  # a ∩ b

    r = client.post(f"{BASE}/overlay/bogus", json={"layer_a": a, "layer_b": b})
    assert r.status_code == 422


def test_overlay_layer_refs():
    client.post(f"{BASE}/overlay/layers/register",
                json={"name": "tenements", "feature_collection": fc([square(0, 0, 10, 10, {"name": "E1"})])})
    r = client.post(f"{BASE}/overlay/intersect",
                    json={"layer_a_ref": "tenements", "layer_b": fc([square(5, 5, 8, 8, {"score": 1.0})])})
    assert r.status_code == 200
    assert r.json()["total_area"] == pytest.approx(9.0)


# ------------------------------------------------------------- 7. change map
def test_change_map_planted_polygon():
    rng = np.random.default_rng(42)
    before = rng.normal(0.5, 0.01, size=(20, 20))
    after = before.copy()
    # plant change in rows 5:10, cols 7:12 (5x5 pixels)
    after[5:10, 7:12] += 1.0
    r = client.post(f"{BASE}/change/map", json={
        "before": before.tolist(), "after": after.tolist(), "pixel_size": 30.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["class_areas"]["change"]["pixels"] == 25
    assert body["class_areas"]["change"]["area"] == pytest.approx(25 * 900.0)
    assert len(body["regions"]) == 1
    cx, cy = body["regions"][0]["centroid"]
    # region cols 7..11 -> x centre at (7+12)/2*30 = 285 ; rows 5..9 -> y = -(5+10)/2*30 = -225
    assert cx == pytest.approx(285.0)
    assert cy == pytest.approx(-225.0)
    feat = body["result"]["features"][0]
    assert feat["properties"]["pixel_count"] == 25
    assert feat["properties"]["area"] == pytest.approx(22500.0)


def test_change_map_no_change():
    arr = np.full((10, 10), 0.5)
    r = client.post(f"{BASE}/change/map",
                    json={"before": arr.tolist(), "after": arr.tolist()})
    body = r.json()
    assert body["class_areas"]["change"]["pixels"] == 0
    assert body["regions"] == []


# ------------------------------------------------------------- 8. terrain mesh
def test_terrain_mesh_3x3_grid():
    dtm = [[0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0]]
    r = client.post(f"{BASE}/terrain/mesh", json={
        "dtm": dtm,
        "bounds": {"minx": 0.0, "miny": 0.0, "maxx": 2.0, "maxy": 2.0},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["triangle_count"] == 8  # (3-1)*(3-1)*2
    assert body["vertex_count"] == 9
    # corner vertex (0,0,0) belongs to flat quads around the peak only partly;
    # check the peak vertex normal points up-ish and all normals are unit length
    normals = np.array(body["normals"])
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)
    # peak is vertex index 4 (row 1, col 1)
    peak_n = normals[4]
    assert peak_n[2] > 0.5  # tilts up
    # flat plane mesh -> all normals exactly (0,0,1) and 8 triangles
    flat = client.post(f"{BASE}/terrain/mesh", json={
        "dtm": [[1.0] * 3 for _ in range(3)],
        "bounds": {"minx": 0.0, "miny": 0.0, "maxx": 2.0, "maxy": 2.0},
    }).json()
    assert flat["triangle_count"] == 8
    assert np.allclose(np.array(flat["normals"]), np.array([0.0, 0.0, 1.0]), atol=1e-6)
    # geometry: vertex 0 at (minx, maxy), vertex 8 at (maxx, miny)
    assert body["vertices"][0][:2] == [0.0, 2.0]
    assert body["vertices"][8][:2] == [2.0, 0.0]


def test_terrain_mesh_colors_hillshade_decimation():
    dtm = (np.arange(36, dtype=float).reshape(6, 6) % 7).tolist()
    r = client.post(f"{BASE}/terrain/mesh", json={
        "dtm": dtm,
        "bounds": {"minx": 0.0, "miny": 0.0, "maxx": 5.0, "maxy": 5.0},
        "decimation": 2, "colors": True, "hillshade": True,
    })
    body = r.json()
    assert body["grid_shape"] == [3, 3]
    assert body["triangle_count"] == 8
    assert "colors" in body and len(body["colors"]) == body["vertex_count"]
    png = base64.b64decode(body["hillshade_png_base64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    assert img.size == (6, 6)


# ------------------------------------------------------------- 9. tenements
def test_tenement_containment():
    ten = fc([square(0, 0, 10, 10, {"name": "E123"})])
    points = fc([
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [5, 5]},
         "properties": {"name": "DH001"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [15, 5]},
         "properties": {"name": "DH002"}},
    ])
    r = client.post(f"{BASE}/tenements/check", json={"tenements": ten, "points": points})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["checked"] == 2
    assert body["violation_count"] == 1
    assert body["violations"][0]["name"] == "DH002"
    assert body["results"][0]["within_tenements"] == ["E123"]
    assert body["results"][1]["compliant"] is False


def test_tenement_expiry_and_alerts(tmp_path):
    from api.innovations.geotoolkit_ext import tenement as tmod
    tmod.reset_engine(str(tmp_path / "test_tenements.sqlite3"))

    soon = tmod.date.today().toordinal() + 10
    from datetime import date as _date
    soon_iso = _date.fromordinal(soon).isoformat()
    far_iso = _date.fromordinal(soon + 500).isoformat()

    r = client.post(f"{BASE}/tenements/expiry-watch", json={
        "name": "E-expiring", "expiry_date": soon_iso,
        "obligations": [{"description": "lodg1 expenditure report", "met": False},
                        {"description": "rent paid", "met": True}],
    })
    assert r.status_code == 201, r.text
    assert r.json()["days_until_expiry"] == 10

    client.post(f"{BASE}/tenements/expiry-watch", json={
        "name": "E-far", "expiry_date": far_iso, "obligations": []})

    r = client.get(f"{BASE}/tenements/alerts", params={"within_days": 30})
    body = r.json()
    names = {a["name"] for a in body["alerts"]}
    assert "E-expiring" in names
    assert "E-far" not in names
    alert = next(a for a in body["alerts"] if a["name"] == "E-expiring")
    assert alert["expiring_within_window"] is True
    assert alert["unmet_obligation_count"] == 1
    assert alert["unmet_obligations"][0]["description"] == "lodg1 expenditure report"

    # restore default engine for other tests
    tmod.reset_engine(os.path.join(tmod.tempfile.gettempdir(), "geotoolkit_ext_tenements_test.sqlite3"))


# ------------------------------------------------------------- 10. crs
def test_crs_transform_roundtrip():
    coords = [[141.0, -37.0]]
    r = client.post(f"{BASE}/crs/transform",
                    json={"coords": coords, "src_crs": "EPSG:4326", "dst_crs": "EPSG:32754"})
    assert r.status_code == 200, r.text
    utm = r.json()["coords"]
    assert 100000 < utm[0][0] < 900000  # easting
    assert 5000000 < utm[0][1] < 6500000  # southern-hemisphere northing
    back = client.post(f"{BASE}/crs/transform",
                       json={"coords": utm, "src_crs": "EPSG:32754", "dst_crs": "EPSG:4326"})
    lon, lat = back.json()["coords"][0]
    assert lon == pytest.approx(141.0, abs=1e-6)
    assert lat == pytest.approx(-37.0, abs=1e-6)


def test_utm_zone_known_coords():
    r = client.get(f"{BASE}/crs/utm-zone", params={"lon": 141.0, "lat": -37.0})
    body = r.json()
    assert body["utm_zone"] == 54
    assert body["hemisphere"] == "south"
    assert body["epsg"] == 32754
    r2 = client.get(f"{BASE}/crs/utm-zone", params={"lon": -74.0, "lat": 40.7})
    assert r2.json()["utm_zone"] == 18
    assert r2.json()["epsg"] == 32618


def test_crs_detect():
    r = client.post(f"{BASE}/crs/detect", json={"coords": [[141.0, -37.0], [144.9, -37.8]]})
    best = r.json()["best_guess"]
    assert best["epsg"] == 4326
    r = client.post(f"{BASE}/crs/detect", json={"coords": [[500000, 5900000], [510000, 5910000]]})
    cands = r.json()["candidates"]
    assert any("UTM" in c["name"] for c in cands)
    assert all("reasoning" in c for c in cands)


def test_grid_ref_geocoding():
    # Melbourne CBD approx: zone 55 south, easting ~318000, northing ~5813000
    r = client.post(f"{BASE}/geocode/grid-ref", json={"grid_ref": "55H 318000 5813000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hemisphere"] == "south"
    assert body["epsg"] == 32755
    assert body["lon"] == pytest.approx(144.96, abs=0.05)
    assert body["lat"] == pytest.approx(-37.81, abs=0.05)
    # bad ref
    r = client.post(f"{BASE}/geocode/grid-ref", json={"grid_ref": "garbage"})
    assert r.status_code == 422
