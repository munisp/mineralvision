"""
Tests for geotoolkit innovations 1-5.

Seeded synthetic data; asserts real numeric values:
- desurvey of a straight 60-degree-dip hole matches analytic geometry
- profile of a planar DTM matches the plane
- kriging/IDW surface honors sample values at sample points
- Web-Mercator tile math matches slippy-map formulas
- PNG outputs carry the PNG magic and decode via PIL
"""

import io
import math
import os
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL_PKG = os.path.join(REPO_ROOT, "MineralVision_Final_Package")
SRC = os.path.join(FINAL_PKG, "src")
for p in (REPO_ROOT, FINAL_PKG, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.innovations.geotoolkit import router as geotoolkit_router
from src.api.innovations.geotoolkit import core as gcore

rng = np.random.default_rng(42)

app = FastAPI()
app.include_router(geotoolkit_router)
client = TestClient(app)

# A raster spanning the whole world in Web Mercator, 16x16.
WORLD = [-gcore.WEBMERC_MAX, -gcore.WEBMERC_MAX,
         gcore.WEBMERC_MAX, gcore.WEBMERC_MAX]


def make_world_raster(colormap="viridis"):
    grid = rng.random((16, 16)).tolist()
    r = client.post("/innovations/geotoolkit/tiles/raster/register", json={
        "grid": grid, "bounds": WORLD, "crs": "EPSG:3857", "colormap": colormap})
    assert r.status_code == 200, r.text
    return r.json()["raster_id"], np.array(grid)


# ---------------------------------------------------------------------------
# Tile math (slippy-map formulas)
# ---------------------------------------------------------------------------

def test_tile_math_matches_slippy_map():
    # z=0: single tile covers the whole web-mercator world.
    assert gcore.tile_bounds_merc(0, 0, 0) == (
        -gcore.WEBMERC_MAX, -gcore.WEBMERC_MAX,
        gcore.WEBMERC_MAX, gcore.WEBMERC_MAX)

    # Known lon/lat -> tile at z=12 (standard slippy-map formula values):
    # lon=0, lat=0 -> x = 2^11 = 2048, y = 2048.
    xt, yt = gcore.lonlat_to_tile(0.0, 0.0, 12)
    assert (xt, yt) == (2048, 2048)

    z, x, y = 12, 2048, 2048
    minx, miny, maxx, maxy = gcore.tile_bounds_merc(z, x, y)
    # tile 2048 starts exactly at mercator origin and spans 40075016.6856/4096 m.
    size = 40075016.68557849 / 4096
    assert minx == pytest.approx(0.0, abs=1e-6)
    assert maxx == pytest.approx(size, rel=1e-9)
    assert maxy == pytest.approx(0.0, abs=1e-6)  # equator row
    assert miny == pytest.approx(-size, rel=1e-9)
    # y=0 row tops out at the mercator max.
    assert gcore.tile_bounds_merc(z, 2048, 0)[3] == pytest.approx(
        gcore.WEBMERC_MAX, rel=1e-12)

    # Round-trip lon/lat -> merc -> lon/lat.
    x0, y0 = gcore.lonlat_to_merc(149.0, -35.3)
    lon, lat = gcore.merc_to_lonlat(x0, y0)
    assert lon == pytest.approx(149.0, abs=1e-9)
    assert lat == pytest.approx(-35.3, abs=1e-9)

    # Mercator x for lon=90deg is a quarter of the world width.
    xm, _ = gcore.lonlat_to_merc(90.0, 0.0)
    assert xm == pytest.approx(gcore.WEBMERC_MAX / 2, rel=1e-9)


def test_bilinear_sampling_exact_at_nodes():
    grid = np.arange(12, dtype=float).reshape(3, 4)  # rows north->south
    bounds = [0.0, 0.0, 3.0, 2.0]  # node spacing 1 m
    # node (col 1, row 0) = north row, x=1, y=2 -> value grid[0,1] = 1.0
    v = gcore.bilinear_sample(grid, bounds, np.array([1.0]), np.array([2.0]))
    assert v[0] == pytest.approx(1.0)
    # midpoint between grid[1,1]=5 and grid[1,2]=6
    v = gcore.bilinear_sample(grid, bounds, np.array([1.5]), np.array([1.0]))
    assert v[0] == pytest.approx(5.5)
    # out of bounds -> NaN
    v = gcore.bilinear_sample(grid, bounds, np.array([5.0]), np.array([1.0]))
    assert np.isnan(v[0])


# ---------------------------------------------------------------------------
# Innovation 1 — raster tiles
# ---------------------------------------------------------------------------

def test_raster_register_and_tile_png():
    rid, grid = make_world_raster()
    r = client.get(f"/innovations/geotoolkit/tiles/raster/1/0/0?raster_id={rid}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (256, 256)
    assert img.mode == "RGBA"

    # Center pixel of tile z1/x0/y0 should map to the north-west quadrant of the
    # world raster; its normalized value must be a valid colormap color.
    px = np.array(img)[128, 128, :3]
    assert px.sum() > 0  # not transparent/black-empty


def test_raster_tile_colormaps_and_errors():
    rid, _ = make_world_raster(colormap="terrain")
    for cmap in ("viridis", "terrain", "iron-oxide"):
        r = client.get(f"/innovations/geotoolkit/tiles/raster/2/1/2"
                       f"?raster_id={rid}&colormap={cmap}")
        assert r.status_code == 200, cmap
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = client.get(f"/innovations/geotoolkit/tiles/raster/2/1/2?raster_id={rid}"
                   "&colormap=plasma")
    assert r.status_code == 422
    r = client.get("/innovations/geotoolkit/tiles/raster/2/1/2?raster_id=nope")
    assert r.status_code == 404
    r = client.get(f"/innovations/geotoolkit/tiles/raster/2/9/0?raster_id={rid}")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Innovation 2 — vector geojson tiles
# ---------------------------------------------------------------------------

def test_vector_tile_point_clipping():
    # Register points around Null-Island area; query the z=1 tile (0,0) which
    # spans lon [-180, 0].
    feats = [{"geometry": {"type": "Point", "coordinates": [-30.0, 10.0]},
              "properties": {"name": "in-tile"}},
             {"geometry": {"type": "Point", "coordinates": [50.0, 10.0]},
              "properties": {"name": "out-of-tile"}}]
    r = client.post("/innovations/geotoolkit/tiles/features/register",
                    json={"layer": "test-points", "features": feats})
    assert r.status_code == 200 and r.json()["registered"] == 2

    r = client.get("/innovations/geotoolkit/tiles/features/1/0/0?layer=test-points")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    lon, lat = body["features"][0]["geometry"]["coordinates"]
    assert lon == pytest.approx(-30.0, abs=1e-6)
    assert lat == pytest.approx(10.0, abs=1e-6)
    assert body["features"][0]["properties"]["name"] == "in-tile"


def test_vector_tile_polygon_clipping():
    # Polygon crossing the tile boundary: lon [-10, 10], lat [-10, 10]; tile
    # z1/x0/y0 covers lon [-180,0], so the clipped polygon must end at lon 0.
    poly = {"type": "Polygon", "coordinates": [[
        [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0]]]}
    client.post("/innovations/geotoolkit/tiles/features/register",
                json={"layer": "test-polys",
                      "features": [{"geometry": poly, "properties": {}}]})
    r = client.get("/innovations/geotoolkit/tiles/features/1/0/0?layer=test-polys")
    body = r.json()
    assert body["count"] == 1
    geom = body["features"][0]["geometry"]
    lons = [c[0] for ring in geom["coordinates"] for c in ring]
    lats = [c[1] for ring in geom["coordinates"] for c in ring]
    assert max(lons) <= 1e-6  # clipped at lon 0
    assert min(lons) == pytest.approx(-10.0, abs=1e-6)
    assert max(lats) == pytest.approx(10.0, abs=1e-6)
    assert body["features"][0]["properties"]["clipped"] is True


# ---------------------------------------------------------------------------
# Innovation 3 — drillhole desurvey + scene
# ---------------------------------------------------------------------------

def test_desurvey_straight_hole_matches_analytic():
    # Straight hole: azimuth 0 (north), dip -60 throughout, 100 m deep.
    # Inclination from vertical = 30 deg -> horizontal = MD*sin30 = 0.5*MD,
    # vertical drop = MD*cos30.
    req = {"collar": {"easting": 500000.0, "northing": 6000000.0, "elevation": 350.0},
           "survey": [{"depth": 50.0, "azimuth": 0.0, "dip": -60.0},
                      {"depth": 100.0, "azimuth": 0.0, "dip": -60.0}]}
    r = client.post("/innovations/geotoolkit/drillholes/desurvey", json=req)
    assert r.status_code == 200
    tr = r.json()["trace"]
    assert len(tr) == 3
    end = tr[-1]
    assert end["easting"] == pytest.approx(500000.0, abs=1e-6)
    assert end["northing"] == pytest.approx(6000000.0 + 100.0 * 0.5, abs=1e-6)
    assert end["elevation"] == pytest.approx(350.0 - 100.0 * math.cos(math.radians(30)),
                                             abs=1e-6)
    mid = tr[1]
    assert mid["northing"] == pytest.approx(6000000.0 + 25.0, abs=1e-6)


def test_desurvey_curved_hole_reduces_horizontal_vs_tangent():
    # Hole steepening from vertical (dip -90) to -45 over 100 m: horizontal
    # displacement must be less than tangential (45 deg from collar) estimate
    # and greater than the pure vertical case (0).
    req = {"collar": {"easting": 0.0, "northing": 0.0, "elevation": 0.0},
           "survey": [{"depth": 50.0, "azimuth": 90.0, "dip": -75.0},
                      {"depth": 100.0, "azimuth": 90.0, "dip": -45.0}]}
    r = client.post("/innovations/geotoolkit/drillholes/desurvey", json=req)
    end = r.json()["trace"][-1]
    # Analytic check via independent minimum-curvature recomputation.
    i1, i2 = math.radians(15), math.radians(45)
    md = 50.0
    cos_dl = (math.cos(i1) * math.cos(i2) +
              math.sin(i1) * math.sin(i2))  # same azimuth
    dl = math.acos(max(-1, min(1, cos_dl)))
    rf = 1.0 if dl < 1e-12 else (2 / dl) * math.tan(dl / 2)
    # first segment: dip -75 constant from surface assumption (i0 = i1)
    de1 = 50.0 * math.sin(i1)
    dd1 = 50.0 * math.cos(i1)
    de2 = md / 2 * (math.sin(i1) + math.sin(i2)) * rf
    dd2 = md / 2 * (math.cos(i1) + math.cos(i2)) * rf
    assert end["easting"] == pytest.approx(de1 + de2, abs=1e-6)
    assert end["elevation"] == pytest.approx(-(dd1 + dd2), abs=1e-6)
    # east positive because azimuth 90 = east
    assert end["easting"] > 0


def test_drillhole_scene_colors_by_grade():
    req = {
        "step": 25.0,
        "holes": [{
            "hole_id": "DH001",
            "collar": {"easting": 1000.0, "northing": 2000.0, "elevation": 100.0},
            "survey": [{"depth": 100.0, "azimuth": 0.0, "dip": -60.0}],
            "assays": [
                {"from_depth": 10.0, "to_depth": 40.0, "value": 0.2, "element": "Au"},
                {"from_depth": 40.0, "to_depth": 70.0, "value": 5.0, "element": "Au"},
            ],
        }],
    }
    r = client.post("/innovations/geotoolkit/drillholes/scene", json=req)
    assert r.status_code == 200
    body = r.json()
    hole = body["holes"][0]
    assert hole["collar"]["position"] == [1000.0, 2000.0, 100.0]
    # trace last vertex: 100 m at 30 deg from vertical
    last = hole["trace_vertices"][-1]
    assert last[1] == pytest.approx(2000.0 + 50.0, abs=1e-6)
    assert last[2] == pytest.approx(100.0 - 100.0 * math.cos(math.radians(30)), abs=1e-6)
    segs = hole["segments"]
    assert len(segs) == 2
    # high-grade segment is redder than low-grade segment
    assert segs[1]["color_rgb"][0] > segs[0]["color_rgb"][0]
    # segment start for 10 m depth sits at 5 m north, cos30*10 below collar
    s0 = segs[0]["start"]
    assert s0[1] == pytest.approx(2005.0, abs=1e-6)
    assert s0[2] == pytest.approx(100.0 - 10 * math.cos(math.radians(30)), abs=1e-6)
    assert hole["grade_range"] == [0.2, 5.0]


# ---------------------------------------------------------------------------
# Innovation 4 — terrain profile / cross-section
# ---------------------------------------------------------------------------

def planar_dtm():
    # z = 100 + 0.5*x - 0.25*y over [0,1000]x[0,1000], 11x11 nodes
    xs = np.linspace(0, 1000, 11)
    ys = np.linspace(1000, 0, 11)  # rows north->south
    grid = np.array([[100 + 0.5 * x - 0.25 * y for x in xs] for y in ys])
    return {"grid": grid.tolist(), "bounds": [0, 0, 1000, 1000], "crs": "EPSG:3857"}


def test_profile_planar_dtm_matches_plane():
    dtm = planar_dtm()
    line = [[100.0, 200.0], [900.0, 800.0]]
    r = client.post("/innovations/geotoolkit/terrain/profile",
                    json={"dtm": dtm, "polyline": line, "n_samples": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["total_length"] == pytest.approx(1000.0, abs=1e-9)
    for d, x, y, e in zip(body["distance"], body["x"], body["y"], body["elevation"]):
        assert e == pytest.approx(100 + 0.5 * x - 0.25 * y, abs=1e-9)
    # straight line: x advances monotonically with distance
    assert body["x"][-1] == pytest.approx(900.0, abs=1e-9)
    assert body["elevation"][-1] == pytest.approx(100 + 450 - 200, abs=1e-9)


def test_cross_section_drill_intersection():
    dtm = planar_dtm()
    line = [[0.0, 500.0], [1000.0, 500.0]]
    hole = {"hole_id": "DH-X",
            "collar": {"easting": 500.0, "northing": 520.0, "elevation": 280.0},
            "survey": [{"depth": 100.0, "azimuth": 180.0, "dip": -60.0}],
            "assays": []}
    r = client.post("/innovations/geotoolkit/terrain/cross-section",
                    json={"dtm": dtm, "line": line, "n_samples": 25,
                          "corridor_width": 60.0, "holes": [hole]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["drill_intersections"]) == 1
    inter = body["drill_intersections"][0]
    assert inter["hole_id"] == "DH-X"
    assert inter["collar_distance"] == pytest.approx(500.0, abs=1e-9)
    # hole end: 100 m at dip -60 az 180 -> 50 m south of collar, 86.6 m down
    end = inter["points"][-1]
    assert end["distance"] == pytest.approx(500.0, abs=1e-6)
    assert end["across_offset"] == pytest.approx(-30.0, abs=1e-6)  # 30 m south of line
    assert end["elevation"] == pytest.approx(280.0 - 100 * math.cos(math.radians(30)),
                                             abs=1e-6)


# ---------------------------------------------------------------------------
# Innovation 5 — targeting heatmap + tiles
# ---------------------------------------------------------------------------

def test_targeting_heatmap_honors_samples():
    n = 12
    pts = rng.uniform(0, 1000, size=(n, 2))
    vals = (np.sin(pts[:, 0] / 150.0) + np.cos(pts[:, 1] / 200.0)).tolist()
    samples = [{"x": float(x), "y": float(y), "value": float(v)}
               for (x, y), v in zip(pts, vals)]
    r = client.post("/innovations/geotoolkit/targeting/heatmap",
                    json={"samples": samples, "grid_size": 32,
                          "colormap": "iron-oxide", "method": "auto"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] in ("kriging", "idw")
    rid = body["raster_id"]
    grid = np.array(gcore.RASTER_REGISTRY[rid].grid)
    bounds = gcore.RASTER_REGISTRY[rid].bounds

    # Surface honors sample values at sample points (bilinear probe).
    for (x, y), v in zip(pts, vals):
        est = gcore.bilinear_sample(grid, bounds,
                                    np.array([x]), np.array([y]))[0]
        assert est == pytest.approx(v, abs=0.15 * (max(vals) - min(vals)) + 0.05)

    st = body["stats"]
    assert st["min"] <= min(vals) + 0.51  # interpolation overshoot bounded
    assert st["max"] >= max(vals) - 0.51
    assert st["n_samples"] == n

    # Tile endpoint serves decodable PNG of the surface.
    r = client.get(f"/innovations/geotoolkit/targeting/tiles/{rid}/4/8/10")
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (256, 256)


def test_targeting_heatmap_4326_input_and_bad_method():
    samples = [{"x": 149.10 + 0.01 * i, "y": -35.30 + 0.01 * i, "value": float(i)}
               for i in range(6)]
    r = client.post("/innovations/geotoolkit/targeting/heatmap",
                    json={"samples": samples, "crs": "EPSG:4326", "grid_size": 16,
                          "method": "idw"})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "idw"
    assert body["stats"]["sample_value_max"] == 5.0
    # bounds converted to mercator metres (non-degrees magnitude)
    assert abs(body["bounds_epsg3857"][0]) > 1000
