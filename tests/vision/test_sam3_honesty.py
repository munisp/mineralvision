"""Honesty tests for the SAM3 decontamination.

Covers (with the native ``sam3`` package absent and backend flags
monkeypatched off — no mocks beyond that):

- /segment/* -> 503 with remediation text when no backend is available
- allow_empty_fallback=true -> 200 with metadata.mock == True (UI dev)
- real-inference exceptions surface as 500, not empty success
- /training/start -> 503 without MV_ALLOW_MOCK_FALLBACK; with the flag the
  labelled no-op result contains no fabricated metrics (no final_loss)
- /health reports availability truthfully
- router is importable from the package __init__ (mountable by orchestrator)
"""

import io
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.vision import sam3 as sam3_pkg
from api.vision.sam3 import api_endpoints as ep
from api.vision.sam3 import fine_tuning as ft
from api.vision.sam3 import sam3_segmenter as seg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_backend(monkeypatch):
    """Guarantee 'no backend' state and clean globals per test."""
    monkeypatch.setattr(seg, "SAM3_AVAILABLE", False)
    monkeypatch.setattr(ft, "TORCH_AVAILABLE", False)
    monkeypatch.delenv("MV_ALLOW_MOCK_FALLBACK", raising=False)
    monkeypatch.delenv("SAM3_SERVICE_URL", raising=False)
    monkeypatch.delenv("SAM3_CHECKPOINT_PATH", raising=False)
    ep._segmenter = None
    yield
    ep._segmenter = None


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(sam3_pkg.router)
    return TestClient(app)


def _png(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    img = (rng.normal(128, 10, (16, 16, 3))).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _post_text(client, query: str = ""):
    return client.post(
        f"/api/v1/sam3/segment/text{query}",
        files={"image": ("core.png", _png(), "image/png")},
        data={"request": json.dumps({"text_prompt": "quartz vein",
                                     "concept": "vein"})},
    )


# ---------------------------------------------------------------------------
# Router mounting readiness
# ---------------------------------------------------------------------------

def test_router_exported_from_package_init():
    assert sam3_pkg.router is ep.router
    assert sam3_pkg.router.prefix == "/api/v1/sam3"
    assert "router" in sam3_pkg.__all__
    paths = {r.path for r in sam3_pkg.router.routes}
    assert "/api/v1/sam3/segment/text" in paths
    assert "/api/v1/sam3/health" in paths


# ---------------------------------------------------------------------------
# /segment/* honesty
# ---------------------------------------------------------------------------

def test_segment_text_503_without_backend(client):
    resp = _post_text(client)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "SAM backend unavailable" in detail
    assert "SAM3_SERVICE_URL" in detail  # remediation text


def test_segment_point_503_without_backend(client):
    resp = client.post(
        "/api/v1/sam3/segment/point",
        files={"image": ("core.png", _png(), "image/png")},
        data={"request": json.dumps(
            {"points": [{"x": 3, "y": 4}], "labels": [1]})},
    )
    assert resp.status_code == 503


def test_segment_exemplar_503_without_backend(client):
    resp = client.post(
        "/api/v1/sam3/segment/exemplar",
        files={"image": ("core.png", _png(), "image/png"),
               "exemplar": ("ex.png", _png(1), "image/png")},
        data={"request": json.dumps({"exemplar_box": [0, 0, 8, 8]})},
    )
    assert resp.status_code == 503


def test_segment_allow_empty_fallback_returns_labeled_mock(client):
    resp = _post_text(client, "?allow_empty_fallback=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["mask_count"] == 0
    assert body["metadata"]["mock"] is True
    assert "reason" in body["metadata"]


def test_segmenter_initialize_honest_when_backend_missing():
    segmenter = seg.create_sam3_segmenter(modality="drillcore")
    assert segmenter._initialized is False
    with pytest.raises(seg.SAM3UnavailableError):
        segmenter.segment_by_text(np.zeros((8, 8, 3), dtype=np.uint8), "vein")


def test_real_inference_error_propagates_as_500(client, monkeypatch):
    """A failing real backend must surface 500, not an empty success."""
    segmenter = seg.create_sam3_segmenter()
    monkeypatch.setattr(seg, "SAM3_AVAILABLE", True)
    segmenter._initialized = True

    class _BoomPredictor:
        def set_image(self, image):
            pass

        def predict(self, **kwargs):
            raise RuntimeError("checkpoint corrupted")

    segmenter.predictor = _BoomPredictor()
    ep._segmenter = segmenter
    resp = _post_text(client)
    assert resp.status_code == 500
    assert "checkpoint corrupted" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /training/* honesty
# ---------------------------------------------------------------------------

def test_training_start_503_without_fallback_flag(client, tmp_path):
    resp = client.post("/api/v1/sam3/training/start", json={
        "job_name": "j1", "dataset_path": str(tmp_path)})
    assert resp.status_code == 503
    assert "Training backend unavailable" in resp.json()["detail"]


def test_training_start_with_fallback_flag_has_no_fabricated_metrics(
        client, tmp_path, monkeypatch):
    # real (tiny) dataset on disk so the job pipeline is genuine
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / "images" / "a.png")
    Image.fromarray(img[:, :, 0]).save(tmp_path / "masks" / "a_mask.png")

    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")
    resp = client.post("/api/v1/sam3/training/start", json={
        "job_name": "j2", "dataset_path": str(tmp_path)})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    # TestClient runs background tasks before returning; job must be done
    job = ep._training_jobs[job_id]
    assert job["status"] == "completed", job["message"]
    result = job["result"]
    assert result["status"] == "mock"
    assert "final_loss" not in result
    assert not any(isinstance(v, float) for v in result.values())


def test_train_raises_without_backend_or_flag(tmp_path):
    config = ft.TrainingConfig(checkpoint_dir=str(tmp_path))
    tuner = ft.SAM3FineTuner(config)
    with pytest.raises(RuntimeError, match="Training backend unavailable"):
        tuner.train(train_dataset=[])


def test_mock_train_only_status_and_message(tmp_path, monkeypatch):
    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")
    config = ft.TrainingConfig(checkpoint_dir=str(tmp_path))
    tuner = ft.SAM3FineTuner(config)
    result = tuner.train(train_dataset=[])
    assert set(result.keys()) == {"status", "message"}
    assert result["status"] == "mock"
    assert "final_loss" not in result


# ---------------------------------------------------------------------------
# /health truthfulness
# ---------------------------------------------------------------------------

def test_health_reports_unavailable_truthfully(client):
    resp = client.get("/api/v1/sam3/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sam3_available"] is False
    assert body["status"] == "degraded"
    assert body["native_sam3_package"] is False
    assert body["mock_fallback_enabled"] is False


def test_health_reports_fallback_flag(client, monkeypatch):
    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")
    body = client.get("/api/v1/sam3/health").json()
    assert body["mock_fallback_enabled"] is True
    # the flag must never masquerade as a real backend
    assert body["sam3_available"] is False
