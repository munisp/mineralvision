"""Tests for rig telemetry ingestion + CUSUM auto-logging."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.innovations.drill_telemetry import router
from api.innovations.drill_telemetry.models import Base, get_session
from api.innovations.drill_telemetry import logic


def _three_regime_trace(seed=7):
    """Synthetic 3-regime drill trace: ROP 50 (0-100m), 20 (100-200m),
    80 (200-300m), 1 m sampling, distinct torque regimes too."""
    rng = np.random.default_rng(seed)
    rop = np.concatenate([rng.normal(50, 2, 100),
                          rng.normal(20, 2, 100),
                          rng.normal(80, 3, 100)])
    torque = np.concatenate([rng.normal(100, 3, 100),
                             rng.normal(200, 3, 100),
                             rng.normal(60, 3, 100)])
    depth = np.arange(300.0) + 2.5   # collar datum offset 2.5 m
    points = [
        {"timestamp": f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
         "depth": float(depth[i]), "rop": float(rop[i]),
         "torque": float(torque[i]), "rpm": 120.0, "vibration": 0.3}
        for i in range(300)]
    return points, depth


# ---------------------------------------------------------------- logic ----

def test_cusum_three_regimes_exact_depths():
    rng = np.random.default_rng(7)
    rop = np.concatenate([rng.normal(50, 2, 100),
                          rng.normal(20, 2, 100),
                          rng.normal(80, 3, 100)])
    depth = np.arange(300.0)
    intervals = logic.segment_intervals(depth, rop)
    assert len(intervals) == 3
    assert intervals[0]["start_depth"] == pytest.approx(0.0)
    assert intervals[0]["end_depth"] == pytest.approx(99.0, abs=3)
    assert intervals[1]["start_depth"] == pytest.approx(100.0, abs=3)
    assert intervals[1]["end_depth"] == pytest.approx(199.0, abs=3)
    assert intervals[2]["start_depth"] == pytest.approx(200.0, abs=3)
    assert intervals[2]["end_depth"] == pytest.approx(299.0)
    # mean ROP per regime
    assert intervals[0]["mean_rop"] == pytest.approx(50, abs=1)
    assert intervals[1]["mean_rop"] == pytest.approx(20, abs=1)
    assert intervals[2]["mean_rop"] == pytest.approx(80, abs=1)
    # regime labels by ROP rank
    assert [i["regime"] for i in intervals] == ["medium", "slow", "fast"]


def test_cusum_stationary_no_splits():
    rng = np.random.default_rng(42)
    rop = rng.normal(50, 2, 300)
    intervals = logic.segment_intervals(np.arange(300.0), rop)
    assert len(intervals) == 1


def test_collar_alignment_and_deviation():
    intervals = [{"start_depth": 12.5, "end_depth": 100.0, "mean_rop": 50.0}]
    out = logic.align_to_collar(
        intervals, collar_depth=2.5, planned_total_depth=97.5,
        deviation_tolerance=0.5, final_measured_depth=100.0)
    assert out["intervals"][0]["aligned_start_depth"] == pytest.approx(10.0)
    assert out["deviation"] == pytest.approx(0.0)
    assert out["deviation_flag"] is False

    out = logic.align_to_collar(
        intervals, collar_depth=2.5, planned_total_depth=90.0,
        deviation_tolerance=0.5, final_measured_depth=100.0)
    assert out["deviation"] == pytest.approx(7.5)
    assert out["deviation_flag"] is True


def test_depth_ordering_robust_to_unsorted_input():
    # segment_intervals sorts by depth internally.
    rng = np.random.default_rng(1)
    rop = np.concatenate([np.full(50, 50.0), np.full(50, 20.0)])
    depth = np.arange(100.0)
    perm = rng.permutation(100)
    intervals = logic.segment_intervals(depth[perm], rop[perm])
    assert len(intervals) == 2
    assert intervals[0]["end_depth"] == pytest.approx(49.0)


# ------------------------------------------------------------------ API ----

@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_session():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = override_session
    with TestClient(app) as c:
        yield c


def test_ingest_store_and_auto_log(client):
    points, depth = _three_regime_trace()
    r = client.post("/innovations/drill_telemetry/rigs/RIG-01/telemetry",
                    json={"points": points})
    assert r.status_code == 201
    assert r.json()["n_ingested"] == 300

    # Stored rows retrievable, depth-ordered.
    r = client.get("/innovations/drill_telemetry/rigs/RIG-01/telemetry")
    assert r.status_code == 200
    stored = r.json()
    assert stored["n_points"] == 300
    depths = [p["depth"] for p in stored["points"]]
    assert depths == sorted(depths)
    assert stored["points"][0]["torque"] == pytest.approx(points[0]["torque"])

    # Auto-log: 3 intervals at correct depths with collar alignment.
    r = client.post("/innovations/drill_telemetry/rigs/RIG-01/auto-log",
                    json={"collar_depth": 2.5, "planned_total_depth": 299.0,
                          "deviation_tolerance": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_intervals"] == 3
    iv = body["intervals"]
    # raw depths carry the +2.5 collar offset; aligned depths start at 0
    assert iv[0]["start_depth"] == pytest.approx(2.5)
    assert iv[0]["aligned_start_depth"] == pytest.approx(0.0)
    assert iv[1]["start_depth"] == pytest.approx(102.5, abs=3)
    assert iv[2]["start_depth"] == pytest.approx(202.5, abs=3)
    # torque regimes detected as well
    assert iv[1]["mean_torque"] == pytest.approx(200, abs=5)
    # deviation: aligned final depth 301.5-2.5=299 vs planned 299 -> 0
    assert body["deviation"] == pytest.approx(0.0, abs=1e-9)
    assert body["deviation_flag"] is False


def test_auto_log_deviation_flag(client):
    points, _ = _three_regime_trace()
    client.post("/innovations/drill_telemetry/rigs/RIG-02/telemetry",
                json={"points": points})
    r = client.post("/innovations/drill_telemetry/rigs/RIG-02/auto-log",
                    json={"collar_depth": 2.5, "planned_total_depth": 280.0,
                          "deviation_tolerance": 5.0})
    body = r.json()
    assert body["deviation"] == pytest.approx(19.0)   # 301.5 - 2.5 - 280
    assert body["deviation_flag"] is True


def test_auto_log_insufficient_data_422(client):
    r = client.post("/innovations/drill_telemetry/rigs/RIG-99/auto-log",
                    json={})
    assert r.status_code == 422


def test_ingest_empty_batch_422(client):
    r = client.post("/innovations/drill_telemetry/rigs/RIG-03/telemetry",
                    json={"points": []})
    assert r.status_code == 422
