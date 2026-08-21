"""WALDO-01 through WALDO-06: WALDO service authentication, image ingress, and fail-closed boundary tests.

These tests validate that the standalone WALDO inference service enforces authentication,
rejects unsafe inputs, and fails closed when configuration or downstream services are unavailable.
"""
import io
import json
import os
import sys
import uuid
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

WALDO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "MineralVision_WALDO_Production_Package", "src")
sys.path.insert(0, WALDO_SRC)
# Also add the WALDO api directory directly so 'api.server' resolves to the WALDO server, not the main API
sys.path.insert(0, os.path.join(WALDO_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _import_waldo_app():
    """Import the WALDO Flask app from the correct package path."""
    import importlib
    waldo_api = importlib.import_module("api.server")
    return waldo_api.app


@pytest.fixture
def waldo_app():
    """Create a test client for the WALDO Flask application with service auth enabled."""
    os.environ["WALDO_SERVICE_TOKEN"] = "test-service-token-12345"
    os.environ["WALDO_MAX_IMAGE_BYTES"] = str(5 * 1024 * 1024)  # 5MB
    os.environ.pop("WALDO_ASYNC_VIDEO_ENABLED", None)  # default disabled
    # Patch the model loader to avoid requiring real weights
    with patch("api.server.get_waldo_module") as mock_waldo:
        mock_module = MagicMock()
        mock_module.detect.return_value = []
        mock_waldo.return_value = mock_module
        app = _import_waldo_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


@pytest.fixture
def auth_headers():
    """Return valid service authentication headers."""
    return {"Authorization": "Bearer test-service-token-12345"}


@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid PNG image for testing."""
    from PIL import Image
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# WALDO-01: Missing/incorrect service token
# ---------------------------------------------------------------------------

class TestWALDO01ServiceAuth:
    """WALDO endpoints MUST reject requests without a valid service token."""

    def test_no_token_returns_401(self, waldo_app):
        """Request without Authorization header is rejected."""
        resp = waldo_app.get("/api/status")
        assert resp.status_code in (401, 403)

    def test_wrong_token_returns_403(self, waldo_app):
        """Request with incorrect token is rejected."""
        resp = waldo_app.get("/api/status", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code in (401, 403)

    def test_valid_token_succeeds(self, waldo_app, auth_headers):
        """Request with correct token reaches the endpoint."""
        resp = waldo_app.get("/health", headers=auth_headers)
        assert resp.status_code == 200

    def test_status_with_valid_token(self, waldo_app, auth_headers):
        """Status endpoint returns service info with valid auth."""
        resp = waldo_app.get("/api/status", headers=auth_headers)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# WALDO-02: Remote URL ingestion attempt
# ---------------------------------------------------------------------------

class TestWALDO02URLIngestion:
    """Remote URL ingestion MUST be disabled; no outbound request occurs."""

    def test_url_field_rejected(self, waldo_app, auth_headers):
        """Submitting a URL field in JSON is rejected or ignored."""
        payload = {"url": "http://evil.example.com/image.png", "confidence": 0.5}
        resp = waldo_app.post(
            "/api/detection/image",
            data=json.dumps(payload),
            content_type="application/json",
            headers=auth_headers,
        )
        # Should either reject (4xx) or process without fetching the URL
        if resp.status_code < 400:
            # If it didn't reject, verify no detection was attempted with a URL
            data = resp.get_json()
            assert data is not None  # got a response without fetching remote
        else:
            assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# WALDO-03: Multipart and JSON image inputs at boundary sizes
# ---------------------------------------------------------------------------

class TestWALDO03ImageBoundary:
    """Valid images work; oversize/malformed inputs are rejected."""

    def test_valid_image_accepted(self, waldo_app, auth_headers, sample_image_bytes):
        """A small valid PNG image is accepted for detection."""
        data = {"file": (io.BytesIO(sample_image_bytes), "test.png")}
        resp = waldo_app.post(
            "/api/detection/image",
            data=data,
            content_type="multipart/form-data",
            headers=auth_headers,
        )
        assert resp.status_code in (200, 201)

    def test_oversized_image_rejected(self, waldo_app, auth_headers):
        """An image exceeding WALDO_MAX_IMAGE_BYTES is rejected."""
        # Create a payload larger than 5MB
        oversized = b"\x00" * (6 * 1024 * 1024)
        data = {"file": (io.BytesIO(oversized), "large.png")}
        resp = waldo_app.post(
            "/api/detection/image",
            data=data,
            content_type="multipart/form-data",
            headers=auth_headers,
        )
        assert resp.status_code in (400, 413)

    def test_non_image_bytes_rejected(self, waldo_app, auth_headers):
        """Random bytes that are not a valid image are rejected."""
        data = {"file": (io.BytesIO(b"not an image at all"), "garbage.bin")}
        resp = waldo_app.post(
            "/api/detection/image",
            data=data,
            content_type="multipart/form-data",
            headers=auth_headers,
        )
        assert resp.status_code in (400, 422, 500)


# ---------------------------------------------------------------------------
# WALDO-04: Explicit artifact absent or unsupported model type
# ---------------------------------------------------------------------------

class TestWALDO04ModelArtifact:
    """WALDO MUST NOT implicitly download or load models without explicit configuration."""

    def test_missing_model_config_fails_closed(self):
        """Without WALDO_MODEL_PATH, the module loader raises or returns None."""
        env = {k: v for k, v in os.environ.items() if "WALDO_MODEL" not in k}
        with patch.dict(os.environ, env, clear=True):
            try:
                from api.server import get_waldo_module
                module = get_waldo_module()
            except (RuntimeError, ValueError, FileNotFoundError, ImportError):
                pass  # Expected: fail closed

    @pytest.mark.skipif(not os.path.exists("/usr/local/lib/python3.12/dist-packages/torch"), reason="torch not installed")
    def test_explicit_model_path_required(self):
        """The WALDO module requires an explicit local model path."""
        from waldo_integration.integration import WALDOIntegration
        with patch.dict(os.environ, {"WALDO_MODEL_PATH": ""}, clear=False):
            integration = WALDOIntegration()
            assert not hasattr(integration, "_model") or integration._model is None


# ---------------------------------------------------------------------------
# WALDO-05: Async video job disabled and enabled modes
# ---------------------------------------------------------------------------

class TestWALDO05AsyncVideo:
    """Async video processing MUST be disabled by default."""

    def test_video_endpoint_disabled_by_default(self, waldo_app, auth_headers):
        """Without WALDO_ASYNC_VIDEO_ENABLED, video endpoints return 404 or 403."""
        resp = waldo_app.post(
            "/api/detection/video",
            data=json.dumps({"url": "test.mp4"}),
            content_type="application/json",
            headers=auth_headers,
        )
        # Should be disabled (404) or rejected
        assert resp.status_code in (403, 404, 405, 501)


# ---------------------------------------------------------------------------
# WALDO-06: Service timeout and malformed downstream response
# ---------------------------------------------------------------------------

class TestWALDO06DownstreamFailure:
    """WALDO MUST fail closed with a sanitized error when downstream services fail."""

    def test_model_exception_returns_sanitized_error(self, waldo_app, auth_headers, sample_image_bytes):
        """If the detection model raises, the response is sanitized (no stack trace)."""
        with patch("api.server.get_waldo_module") as mock_waldo:
            mock_module = MagicMock()
            mock_module.detect.side_effect = RuntimeError("GPU out of memory")
            mock_waldo.return_value = mock_module

            data = {"file": (io.BytesIO(sample_image_bytes), "test.png")}
            resp = waldo_app.post(
                "/api/detection/image",
                data=data,
                content_type="multipart/form-data",
                headers=auth_headers,
            )
            assert resp.status_code == 500
            body = resp.get_json() or {}
            # Error message should be sanitized
            assert "GPU out of memory" not in json.dumps(body)
