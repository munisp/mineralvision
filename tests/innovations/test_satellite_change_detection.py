"""Tests for bi-temporal change detection (planted disturbance)."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.innovations.satellite_change_detection import router
from api.innovations.satellite_change_detection import logic


def _scenes(seed=11, shape=(2, 30, 30)):
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0.5, 0.01, shape)
    return s1, s1.copy()


# ---------------------------------------------------------------- logic ----

def test_planted_disturbance_detected():
    s1, s2 = _scenes()
    s2[0, 10:15, 10:15] += 0.5   # 5x5 clearing in band 1
    res = logic.detect_changes(s1, s2, index="band", band=1,
                               abs_threshold=0.2, z_threshold=2.0,
                               pixel_size=30.0)
    assert res["n_regions"] == 1
    r = res["regions"][0]
    # morphology (open+close) may trim edge pixels; core block must remain
    assert 16 <= r["n_pixels"] <= 25
    assert r["centroid_rc"][0] == pytest.approx(12.0, abs=1.0)
    assert r["centroid_rc"][1] == pytest.approx(12.0, abs=1.0)
    assert r["mean_delta"] == pytest.approx(0.5, abs=0.02)
    assert r["area"] == pytest.approx(r["n_pixels"] * 900.0)


def test_identical_scenes_zero_regions():
    s1, _ = _scenes()
    res = logic.detect_changes(s1, s1.copy(), index="band", band=1,
                               abs_threshold=0.2, z_threshold=2.0)
    assert res["n_regions"] == 0
    assert res["n_changed_pixels"] == 0
    assert res["delta_std"] == pytest.approx(0.0)


def test_no_false_positives_on_noise_only(seed_check=True):
    # Two independent noise realisations: abs threshold dominates -> 0 regions.
    rng = np.random.default_rng(21)
    s1 = rng.normal(0.5, 0.01, (1, 40, 40))
    s2 = rng.normal(0.5, 0.01, (1, 40, 40))
    res = logic.detect_changes(s1, s2, index="band", band=1,
                               abs_threshold=0.1, z_threshold=2.0,
                               morph_open=1)
    assert res["n_regions"] == 0


def test_ndvi_index_change():
    rng = np.random.default_rng(5)
    # 2 bands per scene: red (b1), nir (b2).
    s1 = np.stack([rng.normal(0.3, 0.005, (20, 20)),
                   rng.normal(0.6, 0.005, (20, 20))])
    s2 = s1.copy()
    # vegetation loss in rows 5:10 -> nir drops, NDVI drops ~0.19
    s2[1, 5:10, 5:10] = 0.35
    res = logic.detect_changes(s1, s2, index="ndvi",
                               band_map={"nir": 2, "red": 1},
                               abs_threshold=0.1, z_threshold=2.0,
                               morph_open=0, morph_close=0)
    assert res["n_regions"] == 1
    assert res["regions"][0]["mean_delta"] == pytest.approx(
        (0.35 - 0.3) / (0.35 + 0.3) - (0.6 - 0.3) / (0.6 + 0.3), abs=0.03)


def test_z_threshold_requires_scene_outlier():
    # Uniform shift everywhere: large |delta| but z ~ 0 -> no regions.
    s1, s2 = _scenes()
    s2 = s2 + 0.3
    res = logic.detect_changes(s1, s2, index="band", band=1,
                               abs_threshold=0.2, z_threshold=2.0)
    assert res["n_regions"] == 0


def test_morph_open_removes_speckle():
    s1, s2 = _scenes()
    s2[0, 3, 3] += 1.0      # single hot pixel (speckle)
    res = logic.detect_changes(s1, s2, index="band", band=1,
                               abs_threshold=0.2, z_threshold=2.0,
                               morph_open=1, min_pixels=1)
    assert res["n_regions"] == 0
    res = logic.detect_changes(s1, s2, index="band", band=1,
                               abs_threshold=0.2, z_threshold=2.0,
                               morph_open=0, min_pixels=1)
    assert res["n_regions"] == 1   # without opening the speckle survives


def test_geojson_export():
    regions = [{"label": 1, "n_pixels": 9, "area": 8100.0, "mean_delta": -0.4,
                "max_abs_delta": 0.5, "bbox_map": [30.0, 60.0, 120.0, 150.0]}]
    gj = logic.regions_to_geojson(regions, origin_x=700000.0)
    f = gj["features"][0]
    assert gj["type"] == "FeatureCollection"
    assert f["geometry"]["coordinates"][0][0] == [700030.0, 60.0]
    assert f["properties"]["mean_delta"] == -0.4


# ------------------------------------------------------------------ API ----

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_detect_and_geojson(client):
    s1, s2 = _scenes()
    s2[0, 10:15, 10:15] += 0.5
    payload = {"scene_t1": s1.tolist(), "scene_t2": s2.tolist(),
               "index": "band", "band": 1, "abs_threshold": 0.2,
               "z_threshold": 2.0, "pixel_size": 30.0}
    r = client.post("/innovations/satellite_change_detection/detect",
                    json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n_regions"] == 1
    assert body["regions"][0]["mean_delta"] == pytest.approx(0.5, abs=0.02)

    r = client.post("/innovations/satellite_change_detection/detect/geojson",
                    json=payload)
    assert r.status_code == 200
    assert len(r.json()["features"]) == 1


def test_api_shape_mismatch_422(client):
    s1, _ = _scenes()
    r = client.post("/innovations/satellite_change_detection/detect",
                    json={"scene_t1": s1.tolist(),
                          "scene_t2": s1[:, :10, :10].tolist(),
                          "index": "band", "band": 1})
    assert r.status_code == 422
