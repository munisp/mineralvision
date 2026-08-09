"""Deterministic tests for the marine sonar / bathymetry innovation module."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.innovations.marine_sonar import logic, router

SEED = 7
GRID = 60  # 60x60 cells, 2 m cell size -> 120 m square patch
CELL = 2.0


def _synthetic_seafloor():
    """Depth grid with a planted channel (linear depression) and a pinnacle."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    depth = 30.0 + 0.05 * xx + 0.5 * rng.standard_normal((GRID, GRID))
    # channel along row ~20, 4 cells wide, 6 m deeper
    chan = np.exp(-((yy - 20) ** 2) / (2 * 2.0 ** 2)) * 6.0
    depth += chan
    # pinnacle at (row 42, col 40): 8 m high, 3-cell sigma
    pin = np.exp(-(((yy - 42) ** 2 + (xx - 40) ** 2)) / (2 * 3.0 ** 2)) * 8.0
    depth -= pin
    return depth


def _pings_from_grid(depth, n=4000, spikes=30):
    rng = np.random.default_rng(SEED + 1)
    xs = rng.uniform(0, (GRID - 1) * CELL, n)
    ys = rng.uniform(0, (GRID - 1) * CELL, n)
    ix = np.clip((xs / CELL).astype(int), 0, GRID - 1)
    iy = np.clip((ys / CELL).astype(int), 0, GRID - 1)
    z = depth[iy, ix] + 0.1 * rng.standard_normal(n)
    pings = np.column_stack([xs, ys, z])
    # planted spikes: +25 m outliers
    idx = rng.choice(n, spikes, replace=False)
    pings[idx, 2] += 25.0
    return pings, idx


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ------------------------------------------------------------- logic -------

def test_terrain_derivatives_sanity():
    depth = _synthetic_seafloor()
    out = logic.terrain_derivatives(depth, cell_size=CELL, bpi_inner=2, bpi_outer=8)
    for key in ("slope", "aspect", "rugosity", "bpi", "hillshade"):
        assert key in out and out[key].shape == depth.shape
    assert out["slope"].max() > 10.0  # pinnacle flanks are steep
    assert (out["aspect"] >= 0).all() and (out["aspect"] < 360).all()
    assert (out["hillshade"] >= 0).all() and (out["hillshade"] <= 1).all()
    # BPI positive over the pinnacle (high vs surroundings)
    assert out["bpi"][42, 40] > 0.5
    # rugosity higher at the pinnacle than on the flat plain
    assert out["rugosity"][42, 40] > out["rugosity"][50, 5]


def test_spike_filter_removes_outliers_without_shifting_grid():
    depth = _synthetic_seafloor()
    pings, spike_idx = _pings_from_grid(depth)
    res = logic.process_bathymetry(pings, grid_shape=(GRID, GRID),
                                   median_size=5, spike_threshold=5.0)
    assert res["stats"]["n_artifacts"] > 0
    clean = logic.process_bathymetry(np.delete(pings, spike_idx, axis=0),
                                     grid_shape=(GRID, GRID))["grid"]
    # filtered grid should be close to spike-free grid (spikes excised)
    diff = np.abs(res["grid"] - clean)
    assert np.median(diff) < 0.5
    assert res["stats"]["max_spike_deviation"] > 10.0


def test_feature_detection_finds_channel_and_pinnacle():
    depth = _synthetic_seafloor()
    res = logic.detect_features(depth, cell_size=CELL, relief_threshold=1.5,
                                smooth_window=15, min_area_cells=6,
                                min_elongation=3.0)
    types = {f["type"] for f in res["features"]}
    assert "channel" in types
    assert "pinnacle" in types
    chan = next(f for f in res["features"] if f["type"] == "channel")
    pin = next(f for f in res["features"] if f["type"] == "pinnacle")
    # channel centroid near row 20 (x=40m); planted horizontally
    assert abs(chan["centroid_index"][0] - 20) < 4
    assert chan["length_m"] > 4 * chan["width_m"]
    # pinnacle centroid near (row 42, col 40)
    assert abs(pin["centroid_index"][0] - 42) < 3
    assert abs(pin["centroid_index"][1] - 40) < 3
    assert all(0 < f["confidence"] <= 1 for f in res["features"])


def test_backscatter_classes_separate():
    rng = np.random.default_rng(SEED + 2)
    mos = np.zeros((GRID, GRID))
    mos[:, :20] = 30 + 3 * rng.standard_normal((GRID, 20))    # fine sediment
    mos[:, 20:40] = 60 + 6 * rng.standard_normal((GRID, 20))  # coarse
    mos[:, 40:] = 100 + 8 * rng.standard_normal((GRID, 20))   # rock
    res = logic.classify_backscatter(mos, n_classes=3, window=9, seed=0)
    cm = res["class_map"]
    # classes ordered by intensity: col 10 low, col 50 high
    assert cm[30, 10] == 0 and cm[30, 50] == 2
    # strip interiors (away from texture-window boundary blur) match their class
    for sl, k in ((slice(0, 15), 0), (slice(23, 37), 1), (slice(45, 60), 2)):
        assert (cm[:, sl] == k).mean() > 0.95
    labels = [s["interpretation"] for s in res["class_stats"]]
    assert labels == ["fine_sediment", "coarse_sediment", "rock"]


def test_placer_scores_highest_in_trap_geometry():
    rng = np.random.default_rng(SEED + 3)
    depth = 25.0 + 0.2 * rng.standard_normal((GRID, GRID))
    terr = logic.terrain_derivatives(depth, cell_size=CELL)
    flat_score = logic.score_targets(depth, terr["rugosity"], terr["slope"],
                                     model="placer_gold")["score_grid"]
    # plant a rugged bedrock trap (depression with sharp rims) at (30,30)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    trap = np.exp(-(((yy - 30) ** 2 + (xx - 30) ** 2)) / (2 * 3.0 ** 2)) * 5.0
    depth2 = depth + trap
    terr2 = logic.terrain_derivatives(depth2, cell_size=CELL)
    res = logic.score_targets(depth2, terr2["rugosity"], terr2["slope"],
                              model="placer_gold", top_k=3)
    assert res["score_grid"][30, 30] > np.median(flat_score) + 0.05
    top = res["top_zones"][0]
    assert abs(top["centroid_index"][0] - 30) < 6
    assert abs(top["centroid_index"][1] - 30) < 6
    assert "placer" in top["explanation"].lower()


def test_deposit_model_presets():
    models = logic.deposit_models()
    assert set(models) == {"placer_gold", "marine_diamond", "tin_placer",
                           "sms", "polymetallic_nodule"}
    for m in models.values():
        lo, hi = m["depth_window_m"]
        assert 0 < lo < hi
    assert models["sms"]["depth_window_m"][0] >= 1000.0
    assert models["polymetallic_nodule"]["rugosity_favorable"] is False


# ---------------------------------------------------------------- api ------

def test_api_process_and_terrain(client):
    depth = _synthetic_seafloor()
    pings, _ = _pings_from_grid(depth, n=1500, spikes=10)
    r = client.post("/innovations/marine-sonar/bathymetry/process",
                    json={"pings": pings.tolist(),
                          "grid_shape": [GRID, GRID],
                          "spike_threshold": 5.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["grid"]) == GRID and len(body["grid"][0]) == GRID
    assert body["stats"]["n_artifacts"] > 0

    r2 = client.post("/innovations/marine-sonar/bathymetry/terrain",
                     json={"grid": body["grid"], "cell_size": CELL})
    assert r2.status_code == 200, r2.text
    terr = r2.json()
    assert set(terr) == {"slope", "aspect", "rugosity", "bpi", "hillshade"}
    assert len(terr["slope"]) == GRID


def test_api_feature_detect_and_targets(client):
    depth = _synthetic_seafloor()
    r = client.post("/innovations/marine-sonar/features/detect",
                    json={"grid": depth.tolist(), "cell_size": CELL,
                          "relief_threshold": 1.5})
    assert r.status_code == 200, r.text
    feats = r.json()["features"]
    assert any(f["type"] == "pinnacle" for f in feats)
    assert any(f["type"] == "channel" for f in feats)

    terr = logic.terrain_derivatives(depth, cell_size=CELL)
    r2 = client.post("/innovations/marine-sonar/targets/score",
                     json={"grid": depth.tolist(),
                           "rugosity": terr["rugosity"].tolist(),
                           "slope": terr["slope"].tolist(),
                           "model": "placer_gold", "top_k": 3})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert len(body["score_grid"]) == GRID
    assert 1 <= len(body["top_zones"]) <= 3
    assert body["top_zones"][0]["mean_score"] >= body["top_zones"][-1]["mean_score"]


def test_api_backscatter_and_models(client):
    rng = np.random.default_rng(3)
    mos = np.concatenate(
        [20 + 2 * rng.standard_normal((30, 20)),
         70 + 5 * rng.standard_normal((30, 20))], axis=1)
    r = client.post("/innovations/marine-sonar/backscatter/classify",
                    json={"mosaic": mos.tolist(), "n_classes": 2, "window": 7})
    assert r.status_code == 200, r.text
    cm = np.array(r.json()["class_map"])
    assert cm.shape == (30, 40)
    assert cm[15, 5] == 0 and cm[15, 35] == 1

    r2 = client.get("/innovations/marine-sonar/deposit-models")
    assert r2.status_code == 200
    assert "sms" in r2.json()["models"]


def test_api_422_on_malformed(client):
    # ragged grid -> pydantic 422
    r = client.post("/innovations/marine-sonar/bathymetry/terrain",
                    json={"grid": [[1.0, 2.0, 3.0], [4.0]], "cell_size": 1.0})
    assert r.status_code == 422
    # wrong ping width -> logic 422 via HTTPException
    r2 = client.post("/innovations/marine-sonar/bathymetry/process",
                     json={"pings": [[1.0, 2.0]] * 10})
    assert r2.status_code == 422
    # unknown deposit model
    r3 = client.post("/innovations/marine-sonar/targets/score",
                     json={"grid": [[1.0] * 5] * 5,
                           "rugosity": [[0.1] * 5] * 5,
                           "slope": [[1.0] * 5] * 5,
                           "model": "not_a_model"})
    assert r3.status_code == 422


def test_router_imports_standalone():
    from api.innovations.marine_sonar import router as r
    assert r.prefix == "/innovations/marine-sonar"
    assert r.tags == ["marine-sonar"]
    paths = {route.path for route in r.routes}
    assert paths == {
        "/innovations/marine-sonar/bathymetry/process",
        "/innovations/marine-sonar/bathymetry/terrain",
        "/innovations/marine-sonar/backscatter/classify",
        "/innovations/marine-sonar/features/detect",
        "/innovations/marine-sonar/targets/score",
        "/innovations/marine-sonar/deposit-models",
    }
