"""Tests for /innovations/geoai — deterministic synthetic rasters, real math.

No mocks of computation, no skip-on-ImportError. Unavailability of optional
heavy backends (geoai/samgeo/torch) is exercised via a temporary
``sys.modules`` block (monkeypatch), so the 503 paths run for real.
"""

import base64
import io
import sys

import numpy as np
import pytest
from api.innovations.geoai import core
from api.innovations.geoai import router as geoai_router
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(geoai_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# synthetic rasters (deterministic, no RNG)
# ---------------------------------------------------------------------------

H, W = 20, 24
_yy, _xx = np.mgrid[0:H, 0:W]

# Vegetation block top-left (high NIR, low red) vs bare soil elsewhere.
NIR = np.full((H, W), 40.0)
NIR[(_yy < 8) & (_xx < 10)] = 200.0
RED = np.full((H, W), 120.0)
RED[(_yy < 8) & (_xx < 10)] = 20.0
GREEN = np.full((H, W), 60.0)
BLUE = np.full((H, W), 80.0)
SWIR1 = np.full((H, W), 100.0)
SWIR2 = np.full((H, W), 50.0)

BANDS = {
    "red": RED.tolist(),
    "green": GREEN.tolist(),
    "blue": BLUE.tolist(),
    "nir": NIR.tolist(),
    "swir1": SWIR1.tolist(),
    "swir2": SWIR2.tolist(),
}

# hand-computed expectations
VEG_NDVI = (200.0 - 20.0) / (200.0 + 20.0)      # 0.81818...
SOIL_NDVI = (40.0 - 120.0) / (40.0 + 120.0)     # -0.5
EXP_NDVI_MEAN = (80 * VEG_NDVI + (H * W - 80) * SOIL_NDVI) / (H * W)
EXP_CLAY = 100.0 / 50.0                          # 2.0 everywhere
EXP_IRON = 120.0 / 80.0                          # bare-soil value


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_never_fails(client):
    r = client.get("/innovations/geoai/capabilities")
    assert r.status_code == 200
    body = r.json()
    for name in ("geoai", "samgeo", "torch", "rasterio", "shapely", "skimage"):
        assert name in body["backends"]
        assert body["backends"][name]["available"] in (True, False)
    # rasterio and skimage are installed in this environment — must be real
    assert body["backends"]["rasterio"]["available"] is True
    assert body["backends"]["skimage"]["available"] is True


# ---------------------------------------------------------------------------
# raster indices
# ---------------------------------------------------------------------------


def test_indices_real_values(client):
    r = client.post("/innovations/geoai/raster/indices", json={"bands": BANDS})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "numpy"
    ndvi = body["indices"]["ndvi"]
    assert ndvi["max"] == pytest.approx(VEG_NDVI, abs=1e-9)
    assert ndvi["min"] == pytest.approx(SOIL_NDVI, abs=1e-9)
    assert ndvi["mean"] == pytest.approx(EXP_NDVI_MEAN, abs=1e-9)
    assert body["indices"]["clay"]["mean"] == pytest.approx(EXP_CLAY, abs=1e-6)
    assert body["indices"]["iron_oxide"]["max"] == pytest.approx(EXP_IRON, abs=1e-9)
    assert body["indices"]["ndwi"]["mean"] == pytest.approx(
        (60.0 - 40.0) / (60.0 + 40.0) * (400 / 480) + (60.0 - 200.0) / (60.0 + 200.0) * (80 / 480),
        abs=1e-9,
    )


def test_indices_thumbnail_is_real_png(client):
    r = client.post(
        "/innovations/geoai/raster/indices",
        json={"bands": BANDS, "thumbnail_index": "ndvi"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["thumbnail_index"] == "ndvi"
    raw = base64.b64decode(body["thumbnail_png_b64"])
    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    assert img.size == (W, H)
    px = np.asarray(img)
    # veg block (high NDVI) must render brighter than soil
    assert px[0, 0] > px[H - 1, W - 1]


def test_indices_rejects_missing_bands(client):
    r = client.post("/innovations/geoai/raster/indices", json={"bands": {"nir": NIR.tolist()}})
    assert r.status_code == 422


def test_indices_rejects_shape_mismatch(client):
    r = client.post(
        "/innovations/geoai/raster/indices",
        json={"bands": {"nir": NIR.tolist(), "red": RED[:5].tolist()}},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# auto-segment
# ---------------------------------------------------------------------------


def test_auto_segment_cpu_fallback_real(client):
    r = client.post(
        "/innovations/geoai/raster/auto-segment",
        json={"array": NIR.tolist(), "n_segments": 12, "min_area_px": 4},
    )
    assert r.status_code == 200
    body = r.json()
    # honest fallback naming — geoai/samgeo not installed here
    assert body["backend"] in ("skimage-slic", "scipy-ndimage-otsu")
    assert body["n_regions"] >= 2  # veg block vs background at minimum
    fc = body["regions_geojson"]
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == body["n_regions"]
    total = sum(f["properties"]["area_px"] for f in fc["features"])
    assert total == body["total_labeled_px"] >= H * W * 0.9
    for f in fc["features"]:
        assert f["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert f["properties"]["area_px"] >= 4


def test_auto_segment_503_when_geoai_missing(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "geoai", None)  # force ImportError
    r = client.post(
        "/innovations/geoai/raster/auto-segment",
        json={"array": NIR.tolist(), "prefer_backend": "geoai"},
    )
    assert r.status_code == 503
    assert "geoai" in r.json()["detail"]


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


def test_change_detection_finds_planted_region(client):
    before = np.full((H, W), 100.0)
    after = before.copy()
    after[5:12, 14:21] = 250.0  # planted change block: 7x7 = 49 px
    r = client.post(
        "/innovations/geoai/detect/change",
        json={"before": before.tolist(), "after": after.tolist()},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backend"].startswith("image-differencing+otsu-")
    assert body["changed_px"] == 49
    assert body["change_fraction"] == pytest.approx(49.0 / (H * W), abs=1e-12)
    feats = body["change_geojson"]["features"]
    assert len(feats) == 1
    assert feats[0]["properties"]["area_px"] == 49
    coords = np.array(feats[0]["geometry"]["coordinates"][0])
    assert coords[:, 0].min() == 14 and coords[:, 0].max() == 21
    assert coords[:, 1].min() == 5 and coords[:, 1].max() == 12


def test_change_detection_no_change(client):
    arr = np.full((H, W), 100.0)
    r = client.post(
        "/innovations/geoai/detect/change",
        json={"before": arr.tolist(), "after": arr.tolist()},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["changed_px"] == 0
    assert body["change_geojson"]["features"] == []


def test_change_detection_503_when_torch_missing(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    arr = np.zeros((4, 4)).tolist()
    r = client.post(
        "/innovations/geoai/detect/change",
        json={"before": arr, "after": arr, "prefer_backend": "changestar"},
    )
    assert r.status_code == 503
    assert "changestar" in r.json()["detail"]


def test_change_detection_shape_mismatch(client):
    r = client.post(
        "/innovations/geoai/detect/change",
        json={"before": np.zeros((4, 4)).tolist(), "after": np.zeros((5, 4)).tolist()},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# chips
# ---------------------------------------------------------------------------


def test_chips_manifest_counts(client):
    raster = (_yy * W + _xx).astype(float)  # deterministic gradient
    labels = np.zeros((H, W), dtype=int)
    labels[0:8, 0:8] = 1
    r = client.post(
        "/innovations/geoai/datasets/chips",
        json={
            "raster": raster.tolist(),
            "chip_size": 8,
            "labels": labels.tolist(),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "numpy-slicing"
    # floor((20-8)/8)+1 = 2 rows, floor((24-8)/8)+1 = 3 cols -> 6 chips
    assert body["n_chips"] == 6
    positions = {(c["row"], c["col"]) for c in body["chips"]}
    assert positions == {(0, 0), (0, 8), (0, 16), (8, 0), (8, 8), (8, 16)}
    c00 = next(c for c in body["chips"] if c["row"] == 0 and c["col"] == 0)
    assert c00["label_positive_px"] == 64
    assert c00["label_fraction"] == pytest.approx(1.0)
    expected_chip = raster[0:8, 0:8]
    assert c00["mean"] == pytest.approx(float(expected_chip.mean()))
    c08 = next(c for c in body["chips"] if c["row"] == 0 and c["col"] == 8)
    assert c08["label_positive_px"] == 0


def test_chips_drop_empty(client):
    labels = np.zeros((H, W), dtype=int)
    labels[0:8, 0:8] = 1
    r = client.post(
        "/innovations/geoai/datasets/chips",
        json={
            "raster": np.zeros((H, W)).tolist(),
            "chip_size": 8,
            "labels": labels.tolist(),
            "drop_empty": True,
        },
    )
    body = r.json()
    assert body["n_chips"] == 1
    assert body["chips"][0]["label_positive_px"] == 64


def test_chips_stride(client):
    r = client.post(
        "/innovations/geoai/datasets/chips",
        json={"raster": np.zeros((H, W)).tolist(), "chip_size": 8, "stride": 4},
    )
    body = r.json()
    # floor((20-8)/4)+1 = 4, floor((24-8)/4)+1 = 5 -> 20
    assert body["n_chips"] == 20


# ---------------------------------------------------------------------------
# model registry (real sqlite in tmp path)
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry_db(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAI_REGISTRY_DB", f"sqlite:///{tmp_path}/reg.sqlite")
    return tmp_path


def test_model_registry_roundtrip(client, registry_db):
    payload = {
        "name": "sam-disturbed-ground-v1",
        "task": "segment",
        "backend": "samgeo",
        "version": "1.0.0",
        "metrics": {"iou": 0.71},
        "training_chips": 128,
    }
    r = client.post("/innovations/geoai/models/register", json=payload)
    assert r.status_code == 201
    assert r.json()["id"] == 1
    r2 = client.get("/innovations/geoai/models")
    assert r2.status_code == 200
    body = r2.json()
    assert body["count"] == 1
    m = body["models"][0]
    assert m["name"] == "sam-disturbed-ground-v1"
    assert m["metrics"] == {"iou": 0.71}
    assert m["training_chips"] == 128


# ---------------------------------------------------------------------------
# pure-core checks (numpy Otsu path, no skimage)
# ---------------------------------------------------------------------------


def test_numpy_otsu_matches_skimage():
    from skimage.filters import threshold_otsu as sk_otsu

    rng_free = np.concatenate([np.full(500, 10.0), np.full(500, 90.0), np.linspace(0, 100, 256)])
    np_val = core.otsu_threshold(rng_free)
    sk_val = float(sk_otsu(rng_free))
    assert abs(np_val - sk_val) < 5.0  # same bimodal split, bin-quantisation aside


def test_core_extract_chips_no_label():
    manifest = core.extract_chips(np.arange(64.0).reshape(8, 8), chip_size=4)
    assert manifest["n_chips"] == 4
    assert manifest["chips"][0]["mean"] == pytest.approx(float(np.arange(64.0).reshape(8, 8)[0:4, 0:4].mean()))
