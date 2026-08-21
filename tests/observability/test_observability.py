"""Observability, health, and PostgreSQL Alembic migration tests."""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_PACKAGE = REPO_ROOT / "MineralVision_Final_Package"
sys.path.insert(0, str(FINAL_PACKAGE))

# Keep the test runner's PostgreSQL configuration intact. The application binds
# its database engine at import time, so replacing it with SQLite here corrupts
# later modules now that the production schema contains PostgreSQL JSONB types.
_TMP = tempfile.mkdtemp(prefix="mv_obs_test_")
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
    count_lines = [
        line for line in text_body.splitlines()
        if line.startswith("mineralvision_http_requests_total")
        and 'path="/health"' in line and 'status="200"' in line
    ]
    assert count_lines
    assert float(count_lines[0].rsplit(" ", 1)[1]) >= 3


def test_request_id_roundtrip(client):
    resp = client.get("/health")
    generated = resp.headers.get("x-request-id")
    assert generated and len(generated) == 36
    resp2 = client.get("/health", headers={"X-Request-ID": "test-req-id-123"})
    assert resp2.headers.get("x-request-id") == "test-req-id-123"


def _driver_url(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql://")


def _temporary_database_url(base_url: str, database_name: str) -> str:
    parsed = urlsplit(_driver_url(base_url))
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))


def test_alembic_upgrade_and_downgrade_fresh_postgres():
    """Run the full migration chain in a dedicated throwaway PostgreSQL DB."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    base_url = os.environ.get("MV_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not base_url.startswith(("postgres://", "postgresql://", "postgresql+")):
        raise RuntimeError("MV_TEST_DATABASE_URL/DATABASE_URL must be an isolated PostgreSQL URL")

    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError as exc:
        raise RuntimeError("psycopg2 is required by the locked PostgreSQL CI environment") from exc

    database_name = f"mv_alembic_{uuid.uuid4().hex}"
    admin = psycopg2.connect(_driver_url(base_url))
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    except psycopg2.errors.InsufficientPrivilege:
        pytest.skip("CI PostgreSQL role lacks CREATEDB; grant it only to the disposable CI role")
    finally:
        admin.close()

    fresh_url = _temporary_database_url(base_url, database_name)
    previous_url = os.environ.get("DATABASE_URL")
    engine = None
    try:
        os.environ["DATABASE_URL"] = fresh_url
        cfg = Config(str(FINAL_PACKAGE / "alembic.ini"))
        cfg.set_main_option("script_location", str(FINAL_PACKAGE / "alembic"))
        command.upgrade(cfg, "head")

        engine = create_engine(fresh_url)
        tables = set(inspect(engine).get_table_names())
        expected = {
            "users", "projects", "drillholes", "samples", "qaqc_records",
            "reports", "audit_logs", "alembic_version", "oil_spill_incidents",
            "financial_transfer_intents", "financial_transfer_approvals",
            "financial_transfer_audit_events",
        }
        assert expected <= tables, f"missing tables: {expected - tables}"
        user_cols = {column["name"] for column in inspect(engine).get_columns("users")}
        assert {"id", "username", "email", "password_hash", "role"} <= user_cols
        dh_cols = {column["name"] for column in inspect(engine).get_columns("drillholes")}
        assert {"hole_id", "project_id", "collar_x", "total_depth"} <= dh_cols

        command.downgrade(cfg, "base")
        tables_after = set(inspect(engine).get_table_names())
        assert not ({"users", "projects", "drillholes"} & tables_after)
    finally:
        if engine is not None:
            engine.dispose()
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
        admin = psycopg2.connect(_driver_url(base_url))
        admin.autocommit = True
        try:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name)))
        finally:
            admin.close()


def test_app_boots():
    from src.api.main import app as booted

    assert booted is not None
    assert any(r.path == "/metrics" for r in booted.routes if hasattr(r, "path"))
    assert any(r.path == "/health" for r in booted.routes if hasattr(r, "path"))
