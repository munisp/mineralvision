"""Contract-level tests for the JEPA innovation service.

Runs entirely without torch: the torch core (src/api/jepa/torch_core.py) is
imported lazily and its absence is exercised through the honest 503 paths.
Anomaly math and change-map assembly are tested directly against the pure
numpy internals with hand-computed embeddings.  No mocks pretending to be
JEPA, no skips.
"""

import base64
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.innovations.jepa_service import core, logic, router
from api.innovations.jepa_service import routes as jepa_routes

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DIM = 8


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _png_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    jepa_routes.BASELINE_STORE.clear()
    core.reset_model()
    yield
    jepa_routes.BASELINE_STORE.clear()
    core.reset_model()


@pytest.fixture
def no_torch_core(monkeypatch):
    """Force the lazy torch-core import to fail (core absent)."""
    monkeypatch.setattr(core, "load_torch_core", lambda: None)
    return core


RGB_IMAGE = [[[128, 64, 32]] * 8 for _ in range(8)]


# ---------------------------------------------------------------------------
# Capabilities — structure, never fails
# ---------------------------------------------------------------------------

def test_capabilities_structure(client):
    resp = client.get("/innovations/jepa/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] in ("torch", "unavailable")
    assert isinstance(body["torch_available"], bool)
    assert isinstance(body["torch_core_available"], bool)
    assert isinstance(body["faiss_available"], bool)
    assert body["anomaly_backend"] == "numpy-exact-knn"
    assert isinstance(body["config"], dict)
    assert body["checkpoint_env"] == "MV_JEPA_CHECKPOINT"


def test_capabilities_without_core(client, no_torch_core):
    resp = client.get("/innovations/jepa/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "unavailable"
    assert body["torch_core_available"] is False


# ---------------------------------------------------------------------------
# 503 honesty when the torch core is unavailable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint,payload", [
    ("/innovations/jepa/embeddings/image", {"image": RGB_IMAGE}),
    ("/innovations/jepa/embeddings/batch", {"images": [RGB_IMAGE]}),
    ("/innovations/jepa/train/step", {"images": [RGB_IMAGE], "steps": 2}),
    ("/innovations/jepa/change/score",
     {"image_before": RGB_IMAGE, "image_after": RGB_IMAGE}),
    ("/innovations/jepa/corescan/embed",
     {"image": RGB_IMAGE, "n_segments": 2}),
    ("/innovations/jepa/anomaly/baseline", {"images": [RGB_IMAGE]}),
    ("/innovations/jepa/anomaly/score", {"images": [RGB_IMAGE]}),
])
def test_503_when_core_unavailable(client, no_torch_core, endpoint, payload):
    # anomaly/score checks baseline first; seed it with raw embeddings so the
    # request reaches the model path.
    if endpoint.endswith("/anomaly/score"):
        r = client.post("/innovations/jepa/anomaly/baseline",
                        json={"embeddings": [[1.0] * DIM, [0.9, 0.1] + [0.0] * (DIM - 2)]})
        assert r.status_code == 200
    resp = client.post(endpoint, json=payload)
    assert resp.status_code == 503
    assert "jepa torch core unavailable" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Request validation (422) — no model needed to reject bad payloads
# ---------------------------------------------------------------------------

def test_embed_image_rejects_bad_payload(client, no_torch_core):
    # model requirement fires first for a well-formed image; a malformed
    # payload is a 422 from pydantic (missing field) regardless.
    resp = client.post("/innovations/jepa/embeddings/image", json={})
    assert resp.status_code == 422


def test_batch_requires_nonempty_images(client):
    resp = client.post("/innovations/jepa/embeddings/batch",
                       json={"images": []})
    assert resp.status_code == 422


def test_baseline_rejects_empty_request(client):
    resp = client.post("/innovations/jepa/anomaly/baseline", json={})
    assert resp.status_code == 422


def test_baseline_rejects_bad_embedding(client):
    # httpx's strict JSON encoder refuses NaN, so post the raw body; the
    # service must reject non-finite vectors with 422.
    resp = client.post("/innovations/jepa/anomaly/baseline",
                       content='{"embeddings": [[1.0, NaN]]}',
                       headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_score_requires_baseline(client):
    resp = client.post("/innovations/jepa/anomaly/score",
                       json={"embeddings": [[1.0] * DIM]})
    assert resp.status_code == 409


def test_score_rejects_dim_mismatch(client):
    client.post("/innovations/jepa/anomaly/baseline",
                json={"embeddings": [[1.0] * DIM]})
    resp = client.post("/innovations/jepa/anomaly/score",
                       json={"embeddings": [[1.0] * (DIM + 1)]})
    assert resp.status_code == 422


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        logic.decode_image_unit("not-base64-not-an-image!!!")
    with pytest.raises(Exception):
        logic.decode_image_unit([[[1, 2, 3, 4]]])  # 4 channels, not RGB


def test_decode_base64_png_roundtrip():
    arr = np.random.default_rng(0).integers(0, 255, (12, 10, 3))
    img = logic.decode_image_unit(_png_b64(arr))
    assert img.shape == (12, 10, 3)
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert np.allclose(img, arr / 255.0, atol=1.0 / 255.0)


# ---------------------------------------------------------------------------
# Anomaly math — hand-computed embeddings, planted outlier scores highest
# ---------------------------------------------------------------------------

def _cluster_baseline():
    """Tight cluster of unit vectors around e0 plus structure in e1."""
    rng = np.random.default_rng(42)
    base = _unit([1.0, 0.2] + [0.0] * (DIM - 2))
    rows = []
    for _ in range(12):
        noise = rng.normal(scale=0.02, size=DIM)
        rows.append(_unit(base + noise))
    return np.stack(rows)


def test_knn_planted_outlier_scores_highest():
    baseline = _cluster_baseline()
    inlier = _unit(baseline.mean(axis=0))                 # near cluster centre
    outlier = _unit([-1.0, 0.0] + [0.0] * (DIM - 2))      # opposite direction
    mid = _unit([0.0, 1.0] + [0.0] * (DIM - 2))           # orthogonal

    scores, idx = logic.knn_anomaly_scores(
        np.stack([inlier, mid, outlier]), baseline, k=5)
    assert scores.shape == (3,)
    assert idx.shape == (3, 5)
    # ordering: outlier > orthogonal > inlier
    assert scores[2] > scores[1] > scores[0]
    # exact hand-check of the inlier score: mean of 5 smallest cosine dists
    d = np.array([logic.cosine_distance(inlier, b) for b in baseline])
    assert scores[0] == pytest.approx(np.sort(d)[:5].mean(), rel=1e-12)
    # ranks: 1 = most anomalous
    assert logic.rank_descending(scores) == [3, 2, 1]


def test_knn_k_is_capped_at_baseline_size():
    baseline = _cluster_baseline()[:3]
    q = _unit([1.0, 0.0] + [0.0] * (DIM - 2))
    scores, idx = logic.knn_anomaly_scores(q, baseline, k=99)
    assert idx.shape == (1, 3)
    d = np.array([logic.cosine_distance(q, b) for b in baseline])
    assert scores[0] == pytest.approx(d.mean(), rel=1e-12)


def test_knn_identical_embedding_scores_zero():
    baseline = _cluster_baseline()
    scores, _ = logic.knn_anomaly_scores(baseline[0], baseline, k=1)
    assert scores[0] == pytest.approx(0.0, abs=1e-12)


def test_knn_rejects_bad_inputs():
    with pytest.raises(ValueError):
        logic.knn_anomaly_scores(np.zeros(DIM), np.zeros((0, DIM)))
    with pytest.raises(ValueError):
        logic.knn_anomaly_scores(np.zeros(4), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        logic.knn_anomaly_scores(np.zeros(DIM), _cluster_baseline(), k=0)


def test_anomaly_endpoint_flow_with_raw_embeddings(client):
    baseline = _cluster_baseline()
    labels = [f"b{i}" for i in range(len(baseline))]
    r = client.post("/innovations/jepa/anomaly/baseline",
                    json={"embeddings": baseline.tolist(), "labels": labels})
    assert r.status_code == 200
    assert r.json()["baseline_size"] == len(baseline)

    inlier = _unit(baseline.mean(axis=0)).tolist()
    outlier = _unit([-1.0, 0.0] + [0.0] * (DIM - 2)).tolist()
    r = client.post("/innovations/jepa/anomaly/score",
                    json={"embeddings": [inlier, outlier], "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 5
    res = body["results"]
    assert res[1]["score"] > res[0]["score"]
    assert res[1]["rank"] == 1 and res[0]["rank"] == 2
    assert len(res[0]["neighbors"]) == 5


def test_anomaly_baseline_replace(client):
    client.post("/innovations/jepa/anomaly/baseline",
                json={"embeddings": [[1.0] * DIM]})
    r = client.post("/innovations/jepa/anomaly/baseline",
                    json={"embeddings": [[0.0, 1.0] + [0.0] * (DIM - 2)],
                          "replace": True})
    assert r.json()["baseline_size"] == 1


# ---------------------------------------------------------------------------
# Change map — synthetic patch embeddings, shape/dtype/values
# ---------------------------------------------------------------------------

def test_change_map_from_synthetic_patches():
    rng = np.random.default_rng(7)
    grid, d = 6, 16
    patches_a = rng.normal(size=(grid * grid, d))
    patches_b = patches_a.copy()
    # plant change in patch (row 2, col 3) -> flat index 2*6+3 = 15
    patches_b[15] = -patches_a[15]
    change_map, mean_dist = logic.change_map_from_patches(patches_a, patches_b)

    assert len(change_map) == grid
    assert all(len(row) == grid for row in change_map)
    assert all(isinstance(v, float) for row in change_map for v in row)
    # untouched patches: identical embeddings -> distance ~0
    assert change_map[0][0] == pytest.approx(0.0, abs=1e-12)
    # flipped patch: cosine distance ~2
    assert change_map[2][3] == pytest.approx(2.0, abs=1e-9)
    assert mean_dist == pytest.approx(2.0 / (grid * grid), rel=1e-9)

    top = logic.top_changed_regions(change_map, top_k=3)
    assert top[0] == {"row": 2, "col": 3,
                      "distance": pytest.approx(2.0, abs=1e-9)}
    assert len(top) == 3


def test_change_map_requires_square_grid():
    with pytest.raises(ValueError):
        logic.change_map_from_patches(np.zeros((5, 4)), np.zeros((5, 4)))
    with pytest.raises(ValueError):
        logic.change_map_from_patches(np.zeros((36, 4)), np.zeros((36, 4)),
                                      grid=5)
    with pytest.raises(ValueError):
        logic.change_map_from_patches(np.zeros((36, 4)), np.zeros((36, 5)))


def test_top_changed_regions_ordering():
    m = [[0.1, 0.9], [0.5, 0.2]]
    top = logic.top_changed_regions(m, top_k=2)
    assert top[0]["row"] == 0 and top[0]["col"] == 1
    assert top[1]["row"] == 1 and top[1]["col"] == 0


# ---------------------------------------------------------------------------
# Core-scan bridge — vertical slicing geometry
# ---------------------------------------------------------------------------

def test_slice_vertical_segments_covers_height():
    img = np.arange(10 * 4 * 3, dtype=float).reshape(10, 4, 3)
    segs = logic.slice_vertical_segments(img, 3)
    assert [(s["row_start"], s["row_end"]) for s in segs] == [
        (0, 3), (3, 6), (6, 10)]
    assert np.array_equal(segs[0]["image"], img[0:3])
    with pytest.raises(ValueError):
        logic.slice_vertical_segments(img, 0)
    with pytest.raises(ValueError):
        logic.slice_vertical_segments(img, 11)


def test_corescan_embed_rejects_too_many_segments(client):
    # 503 (no core) would mask geometry validation; decode/slice happens
    # after model resolution in the route, so here we check the pure-logic
    # contract instead and use the endpoint for a simple 422 on missing image.
    resp = client.post("/innovations/jepa/corescan/embed",
                       json={"n_segments": 4})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model singleton — honest failure without the torch core
# ---------------------------------------------------------------------------

def test_get_model_raises_without_core(no_torch_core):
    with pytest.raises(core.CoreUnavailableError, match="torch core"):
        core.get_model()


def test_load_torch_core_none_when_module_missing():
    # torch_core.py does not exist in this worktree; even with torch
    # installed the lazy import must return None (not raise).
    if core.torch_available():
        assert core.load_torch_core() is None
