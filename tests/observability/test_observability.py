"""
Tests for wave-2 observability: Prometheus /metrics, request-id middleware,
real /health checks, and Alembic migrations on a fresh SQLite database.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_PACKAGE = REPO_ROOT / "MineralVision_Final_Package"
sys.path.insert(0, str(FINAL_PACKAGE))

# Point the app at a throwaway SQLite database BEFORE importing src.api.*
# (src/api/database.py binds its engine at import time).
_TMP = tempfile.mkdtemp(prefix="mv_obs_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/app_test.db"
os.environ["DATA_DIR"] = os.path.join(_TMP, "data")

from fastapi.testclient import TestClient  # noqa: E402
from src.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200_with_real_checks(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["version"]
    assert body["checks"]["database"]["status"] == "ok"
    assert "latency_ms" in body["checks"]["database"]
    assert body["checks"]["data_dir"]["status"] == "ok"
    assert body["checks"]["data_dir"]["writable"] is True


def test_metrics_endpoint_exposes_request_counts(client):
    # Generate traffic that the middleware should record.
    for _ in range(3):
        client.get("/health")

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text_body = resp.text

    assert "mineralvision_http_requests_total" in text_body
    assert 'method="GET"' in text_body
    assert 'path="/health"' in text_body
    assert 'status="200"' in text_body
    assert "mineralvision_http_request_duration_seconds_bucket" in text_body
    assert "mineralvision_http_requests_in_progress" in text_body

    # The counter for /health must be > 0 after the requests above.
    count_lines = [
        line for line in text_body.splitlines()
        if line.startswith("mineralvision_http_requests_total")
        and 'path="/health"' in line and 'status="200"' in line
    ]
    assert count_lines, "no counter line for GET /health 200"
    assert float(count_lines[0].rsplit(" ", 1)[1]) >= 3


def test_request_id_roundtrip(client):
    # Server generates a uuid4 request id when none is supplied.
    resp = client.get("/health")
    generated = resp.headers.get("x-request-id")
    assert generated
    assert len(generated) == 36  # uuid4 string form

    # An inbound X-Request-ID is propagated to the response.
    resp2 = client.get("/health", headers={"X-Request-ID": "test-req-id-123"})
    assert resp2.headers.get("x-request-id") == "test-req-id-123"


def test_alembic_upgrade_and_downgrade_fresh_sqlite():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    db_path = os.path.join(_TMP, "alembic_test.db")
    url = f"sqlite:///{db_path}"

    cfg = Config(str(FINAL_PACKAGE / "alembic.ini"))
    cfg.set_main_option("script_location", str(FINAL_PACKAGE / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, "head")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    expected = {
        "users", "projects", "drillholes", "samples",
        "qaqc_records", "reports", "audit_logs", "alembic_version",
    }
    assert expected <= tables, f"missing tables: {expected - tables}"

    # Schema must match the SQLAlchemy models (spot-check key columns).
    user_cols = {c["name"] for c in inspect(engine).get_columns("users")}
    assert {"id", "username", "email", "password_hash", "role"} <= user_cols
    dh_cols = {c["name"] for c in inspect(engine).get_columns("drillholes")}
    assert {"hole_id", "project_id", "collar_x", "total_depth"} <= dh_cols

    # The upgraded schema must be identical to what the models would create
    # (alembic must report no pending changes).
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from src.api.database import Base

    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"schema drift between models and migration: {diff}"

    # Downgrade cleanly back to base.
    command.downgrade(cfg, "base")
    tables_after = set(inspect(engine).get_table_names())
    assert not ({"users", "projects", "drillholes"} & tables_after), \
        f"tables survived downgrade: {tables_after}"


def test_app_boots():
    """Boot check: the canonical app object imports cleanly."""
    from src.api.main import app as booted
    assert booted is not None
    assert any(r.path == "/metrics" for r in booted.routes if hasattr(r, "path"))
    assert any(r.path == "/health" for r in booted.routes if hasattr(r, "path"))
