"""
Real dependency checks backing the /health endpoint.

Checks performed:
- database: ``SELECT 1`` against the configured SQLAlchemy engine (with latency)
- data_dir: configured data directory exists and is writable
- version: platform version (APP_VERSION env, else pyproject.toml, else fallback)

``run_health_checks`` returns ``(http_status_code, payload)`` — 200 when all
checks pass, 503 when any check fails, with per-check detail either way.
"""

import os
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from ..database import engine

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_VERSION = "1.0.0"


def get_version() -> str:
    """Platform version: APP_VERSION env > pyproject.toml > built-in default."""
    env_version = os.environ.get("APP_VERSION")
    if env_version:
        return env_version
    pyproject = _REPO_ROOT / "pyproject.toml"
    try:
        with pyproject.open("rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except Exception:
        return _DEFAULT_VERSION


def _check_database() -> dict:
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        return {
            "status": "fail",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "error": str(exc),
        }


def _check_data_dir() -> dict:
    data_dir = os.environ.get("DATA_DIR", "./data")
    path = Path(data_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".health_probe"
        probe.write_text("ok")
        probe.unlink()
        return {"status": "ok", "path": str(path.resolve()), "writable": True}
    except Exception as exc:
        return {
            "status": "fail",
            "path": str(path),
            "writable": False,
            "error": str(exc),
        }


def run_health_checks() -> tuple[int, dict]:
    """Run all health checks; return (http_status, detail payload)."""
    checks = {
        "database": _check_database(),
        "data_dir": _check_data_dir(),
    }
    healthy = all(c["status"] == "ok" for c in checks.values())
    payload = {
        "status": "healthy" if healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": get_version(),
        "checks": checks,
    }
    return (200 if healthy else 503), payload
