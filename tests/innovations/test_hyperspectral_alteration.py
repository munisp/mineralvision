"""Tests for hyperspectral alteration mapping (planted signatures)."""

import numpy as np
import pytest
from api.innovations.hyperspectral_alteration import logic, router
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _cube(seed=3, shape=(9, 20, 20)):
    rng = np.random.default_rng(seed)
    return rng.normal(1.0, 0.02, shape)


# ---------------------------------------------------------------- logic ----

def test_index_math_exact():
    cube = np.ones((9, 4, 4))
    cube[6] = 2.0   # b7
    cube[4] = 1.0   # b5
    clay = logic.compute_index(cube, "clay", logic.BAND_PRESETS["aster"])
    assert np.allclose(clay, 2.0)
    ndvi = logic.compute_index(
        np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.8)]), "ndvi",
        {"ndvi": (2, 1)})
    assert np.allclose(ndvi, 0.6)  # (0.8-0.2)/(0.8+0.2)


def test_planted_clay_zone_detected_at_location():
    cube = _cube()
    # Plant clay signature: b7 high over rows/cols 5..9 -> clay ratio ~2.
    cube[6, 5:10, 5:10] = 2.0
    res = logic.map_alteration_zones(
        cube, "clay", preset="aster", threshold=1.4,
        ndvi_mask_threshold=None, pixel_size=15.0)
    assert res["n_zones"] == 1
    z = res["zones"][0]
    assert z["n_pixels"] == 25
    assert z["area"] == pytest.approx(25 * 15.0 ** 2)
    assert z["bbox_pixels"] == [5, 5, 10, 10]
    assert z["mean_index"] == pytest.approx(2.0, abs=0.05)
    assert z["centroid_rc"] == [pytest.approx(7.0), pytest.approx(7.0)]


def test_ndvi_mask_excludes_vegetated_zone():
    cube = _cube()
    # Two clay patches; one vegetated (high NDVI via b3/b2).
    cube[6, 2:5, 2:5] = 2.0      # clay patch A
    cube[6, 12:15, 12:15] = 2.0  # clay patch B (vegetated)
    cube[2, 12:15, 12:15] = 4.0  # b3 (nir) high -> NDVI = (4-1)/(4+1) = 0.6
    res = logic.map_alteration_zones(
        cube, "clay", preset="aster", threshold=1.4,
        ndvi_mask_threshold=0.3, pixel_size=15.0)
    assert res["n_zones"] == 1
    assert res["zones"][0]["bbox_pixels"] == [2, 2, 5, 5]


def test_iron_oxide_and_carbonate_indices():
    cube = _cube()
    cube[3] = 1.0
    cube[1] = 1.0
    cube[3, 0:3, 0:3] = 2.0          # iron: b4/b2 = 2 in corner
    iron = logic.map_alteration_zones(
        cube, "iron_oxide", preset="aster", threshold=1.6,
        ndvi_mask_threshold=None)
    assert iron["n_zones"] == 1
    assert iron["zones"][0]["bbox_pixels"] == [0, 0, 3, 3]

    cube[7] = 1.0
    cube[7, 10:13, 1:4] = 1.5        # carbonate: b8/b7 = 1.5
    carb = logic.map_alteration_zones(
        cube, "carbonate", preset="aster", threshold=1.1,
        ndvi_mask_threshold=None)
    assert carb["n_zones"] == 1
    assert carb["zones"][0]["n_pixels"] == 9


def test_min_pixels_and_presets():
    cube = _cube()
    cube[6, 5, 5] = 5.0   # single hot pixel
    res = logic.map_alteration_zones(
        cube, "clay", preset="aster", threshold=1.4,
        ndvi_mask_threshold=None, min_pixels=2)
    assert res["n_zones"] == 0
    # landsat8 preset: clay = b7/b6.
    cube8 = np.ones((7, 5, 5))
    cube8[6] = 2.0
    cube8[5] = 1.0
    clay = logic.compute_index(cube8, "clay", logic.BAND_PRESETS["landsat8"])
    assert np.allclose(clay, 2.0)
    with pytest.raises(ValueError, match="not available"):
        logic.compute_index(cube8, "carbonate", logic.BAND_PRESETS["landsat8"])


def test_geojson_export_structure():
    zones = [{"label": 1, "n_pixels": 4, "area": 900.0, "mean_index": 2.0,
              "max_index": 2.1, "bbox_map": [15.0, 30.0, 45.0, 60.0]}]
    gj = logic.zones_to_geojson(zones, origin_x=100.0, origin_y=200.0)
    assert gj["type"] == "FeatureCollection"
    f = gj["features"][0]
    coords = f["geometry"]["coordinates"][0]
    assert coords[0] == [115.0, 230.0]
    assert coords[2] == [145.0, 260.0]
    assert coords[0] == coords[-1]            # closed ring
    assert f["properties"]["mean_index"] == 2.0


# ------------------------------------------------------------------ API ----

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_map_and_geojson(client):
    cube = _cube()
    cube[6, 5:10, 5:10] = 2.0
    payload = {"cube": cube.tolist(), "index": "clay", "preset": "aster",
               "threshold": 1.4, "ndvi_mask_threshold": None,
               "pixel_size": 15.0}
    r = client.post("/innovations/hyperspectral_alteration/map", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n_zones"] == 1
    assert body["zones"][0]["n_pixels"] == 25

    r = client.post("/innovations/hyperspectral_alteration/map/geojson",
                    json={**payload, "origin_x": 500000.0})
    assert r.status_code == 200
    gj = r.json()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    x0 = gj["features"][0]["geometry"]["coordinates"][0][0][0]
    assert x0 == pytest.approx(500000.0 + 5 * 15.0)


def test_api_presets_and_indices(client):
    r = client.get("/innovations/hyperspectral_alteration/presets")
    assert r.status_code == 200
    assert set(r.json()["presets"]) == {"aster", "landsat8", "sentinel2"}

    cube = np.ones((9, 3, 3)) * 2.0
    r = client.post("/innovations/hyperspectral_alteration/indices",
                    json={"cube": cube.tolist(), "index": "clay",
                          "preset": "aster"})
    assert r.status_code == 200
    assert r.json()["mean"] == pytest.approx(1.0)

    r = client.post("/innovations/hyperspectral_alteration/map",
                    json={"cube": [[[1.0]]], "index": "clay",
                          "preset": "aster"})
    assert r.status_code == 422   # band 7 unavailable in 1-band cube
