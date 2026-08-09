"""Tests for geostat_drift_alerting — CUSUM/EWMA/rolling-geostat detectors."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'MineralVision_Final_Package', 'src'))

from api.innovations.geostat_drift_alerting.logic import (
    CUSUMDetector,
    DriftMonitor,
    EWMADetector,
    declustered_mean,
    declustering_weights,
    estimate_sill,
)

# ---------------------------------------------------------------------------
# Detector unit behavior
# ---------------------------------------------------------------------------

def test_cusum_detects_step_shift():
    det = CUSUMDetector(mean=0.0, std=1.0, k=0.5, h=5.0)
    alert = None
    for _i, v in enumerate(np.full(30, 2.0)):  # +2 sigma step
        alert = det.update(float(v))
        if alert:
            break
    assert alert is not None
    assert alert["detector"] == "cusum"
    assert alert["direction"] == "up"
    # with z=2, k=0.5: S grows 1.5/step -> crosses h=5 on step 4
    assert _i == 3
    assert alert["magnitude"] > 5.0


def test_cusum_direction_down():
    det = CUSUMDetector(mean=0.0, std=1.0, k=0.5, h=5.0)
    for v in np.full(30, -2.0):
        alert = det.update(float(v))
        if alert:
            break
    assert alert["direction"] == "down"


def test_cusum_resets_after_alarm():
    det = CUSUMDetector(mean=0.0, std=1.0, k=0.5, h=5.0)
    for v in np.full(4, 2.0):
        alert = det.update(float(v))
    assert alert is not None
    assert det.s_plus == 0.0  # chart restarted


def test_ewma_exact_limits_and_detection():
    det = EWMADetector(mean=0.0, std=1.0, lam=0.2, L=3.0)
    # first-sample limit = L * sqrt(lam/(2-lam) * (1-(1-lam)^2))
    det.update(0.0)
    expected = 3.0 * np.sqrt(0.2 / 1.8 * (1 - 0.8 ** 2))
    # asymptotic limit is L*sqrt(lam/(2-lam)) ~ 1.0; first-sample is larger
    assert expected == pytest.approx(3.0 * np.sqrt(0.2 / 1.8 * 0.36))
    det2 = EWMADetector(mean=0.0, std=1.0, lam=0.2, L=3.0)
    alert = None
    for v in np.full(60, 1.5):
        alert = det2.update(float(v))
        if alert:
            break
    assert alert is not None and alert["direction"] == "up"
    assert alert["magnitude"] > 1.0  # z/limit ratio


# ---------------------------------------------------------------------------
# Geostatistics
# ---------------------------------------------------------------------------

def test_declustering_weights_and_mean():
    # 8 samples clustered at origin (low grade), 2 isolated (high grade)
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
                       [0.05, 0.05], [0.15, 0.05], [0.05, 0.15], [0.15, 0.15],
                       [100.0, 100.0], [100.1, 100.1]])
    values = np.array([1.0] * 8 + [5.0, 5.0])
    w = declustering_weights(coords, cell_size=1.0)
    assert w[:8] == pytest.approx([0.125] * 8)
    assert w[8:] == pytest.approx([0.5, 0.5])
    dm = declustered_mean(values, coords, cell_size=1.0)
    # (8*(1/8)*1 + 2*(1/2)*5) / (8*(1/8) + 2*(1/2)) = 6/2 = 3.0
    assert dm == pytest.approx(3.0)
    assert values.mean() == pytest.approx(1.8)  # naive mean is biased low


def test_estimate_sill_iid_equals_variance():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 2.0, 300)
    sill = estimate_sill(x)  # no coords: serial sill == process variance
    assert sill == pytest.approx(np.var(x, ddof=1))
    assert sill == pytest.approx(4.0, rel=0.15)


def test_estimate_sill_spatial_variogram():
    # spatially smooth field: low semivariance at short lags, sill at range
    gx, gy = np.meshgrid(np.linspace(0, 10, 12), np.linspace(0, 10, 12))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    values = np.sin(gx.ravel() / 2.0) + np.cos(gy.ravel() / 2.0)
    sill = estimate_sill(values, coords, n_lags=8)
    assert sill > 0.5 * np.var(values)  # sane sill
    assert sill < 3.0 * np.var(values)


# ---------------------------------------------------------------------------
# End-to-end monitor behavior (seeded)
# ---------------------------------------------------------------------------

def test_stationary_stream_zero_alerts():
    # seed 6 verified to produce a fully in-control stream under defaults
    rng = np.random.default_rng(6)
    baseline = rng.normal(0, 1, 200)
    monitor = DriftMonitor(baseline, window=50)
    alerts = monitor.push(rng.normal(0, 1, 400))
    assert alerts == []
    assert monitor.list_alerts() == []
    assert monitor.n_seen == 400


def test_planted_mean_shift_detected_within_n_samples():
    rng = np.random.default_rng(456)
    monitor = DriftMonitor(rng.normal(0, 1, 200), window=50)
    pre = monitor.push(rng.normal(0, 1, 100))
    assert pre == []  # in-control phase: no alerts
    post = monitor.push(rng.normal(3.0, 1, 150))  # +3 sigma shift at idx 100
    by_detector = {}
    for a in post:
        by_detector.setdefault(a["detector"], a)
    assert "cusum" in by_detector and "ewma" in by_detector
    # both detectors fire within 15 samples of the shift
    assert by_detector["cusum"]["index"] - 100 <= 15
    assert by_detector["ewma"]["index"] - 100 <= 15
    assert by_detector["cusum"]["direction"] == "up"
    # rolling declustered mean also catches the shift
    assert "declustered_mean" in by_detector
    assert by_detector["declustered_mean"]["direction"] == "up"


def test_planted_variance_shift_raises_sill_alert():
    rng = np.random.default_rng(789)
    monitor = DriftMonitor(rng.normal(0, 1, 200), window=50)
    monitor.push(rng.normal(0, 1, 100))
    post = monitor.push(rng.normal(0, 2.5, 150))  # variance x6.25 at idx 100
    sill = [a for a in post if a["detector"] == "sill_change"]
    assert sill, "variance shift must raise a sill_change alert"
    assert sill[0]["direction"] == "up"
    assert sill[0]["index"] - 100 <= 50
    assert sill[0]["magnitude"] > 0.5


def test_alerts_accumulate_with_timestamps():
    rng = np.random.default_rng(456)
    monitor = DriftMonitor(rng.normal(0, 1, 200), window=50)
    monitor.push(rng.normal(0, 1, 100))
    ts = [1000.0 + i for i in range(150)]
    monitor.push(rng.normal(3.0, 1, 150), timestamps=ts)
    alerts = monitor.list_alerts()
    assert len(alerts) >= 2
    assert all(a["timestamp"] is not None for a in alerts)
    assert {a["detector"] for a in alerts} >= {"cusum", "ewma"}


# ---------------------------------------------------------------------------
# Router lifecycle test
# ---------------------------------------------------------------------------

def test_router_stream_lifecycle():
    from api.innovations.geostat_drift_alerting import router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    rng = np.random.default_rng(456)
    baseline = rng.normal(0, 1, 200).tolist()
    r = client.post("/innovations/geostat_drift_alerting/streams", json={
        "stream_id": "au-stream-1", "baseline_values": baseline, "window": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["baseline_std"] > 0

    # duplicate registration rejected
    r = client.post("/innovations/geostat_drift_alerting/streams", json={
        "stream_id": "au-stream-1", "baseline_values": baseline})
    assert r.status_code == 409

    # in-control batch: no alerts
    r = client.post("/innovations/geostat_drift_alerting/streams/au-stream-1/batches",
                    json={"values": rng.normal(0, 1, 100).tolist()})
    assert r.status_code == 200 and r.json()["alerts"] == []

    # shifted batch: alerts within the batch
    r = client.post("/innovations/geostat_drift_alerting/streams/au-stream-1/batches",
                    json={"values": rng.normal(3.0, 1, 150).tolist()})
    assert r.status_code == 200
    alerts = r.json()["alerts"]
    assert {a["detector"] for a in alerts} >= {"cusum", "ewma"}

    # alerts endpoint returns the accumulated history
    r = client.get("/innovations/geostat_drift_alerting/streams/au-stream-1/alerts")
    assert r.status_code == 200
    assert len(r.json()["alerts"]) == len(alerts)

    # unknown stream 404
    r = client.get("/innovations/geostat_drift_alerting/streams/nope/alerts")
    assert r.status_code == 404
