"""
Tests for the real plant digital twin:
- mass/metal conservation across dt steps (closed loop + external feed)
- anomaly hooks fire on planted spikes (z-score + threshold)
- determinism (same inputs => same outputs)
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MineralVision_Final_Package"))

from src.api.digital_twin.plant_simulation import (  # noqa: E402
    PlantTwin,
    TelemetryPoint,
    TelemetryStream,
    create_plant_twin,
)


def build_closed_loop_twin() -> PlantTwin:
    """ROM pad -> crusher -> stockpile (closed system, no external flux)."""
    twin = create_plant_twin()
    twin.add_zone("rom", mass=1000.0, grade=0.02)      # 1000 t @ 2% metal
    twin.add_zone("stockpile", mass=0.0, grade=0.0)
    twin.add_equipment("crusher", "rom", "stockpile",
                       nominal_flow=100.0, capacity=150.0)
    return twin


def test_mass_conservation_closed_loop():
    twin = build_closed_loop_twin()
    m0, q0 = twin.total_mass(), twin.total_metal()
    assert m0 == pytest.approx(1000.0)
    assert q0 == pytest.approx(20.0)

    twin.run(duration=10.0, dt=0.5)  # 20 steps; crusher moves 100 t/h * 10 h

    assert twin.mass_balance_error(m0) < 1e-9
    assert twin.metal_balance_error(q0) < 1e-9
    # ROM depleted by exactly 1000 t over 10 h at 100 t/h
    assert twin.zones["rom"].mass == pytest.approx(0.0, abs=1e-9)
    assert twin.zones["stockpile"].mass == pytest.approx(1000.0)
    # Grade is conserved through blending (no upgrading)
    assert twin.zones["stockpile"].grade == pytest.approx(0.02, abs=1e-9)


def test_mass_conservation_with_external_feed():
    twin = create_plant_twin()
    twin.add_zone("bin", mass=100.0, grade=0.01)
    twin.add_zone("product", mass=0.0)
    twin.add_equipment("conveyor", "bin", "product",
                       nominal_flow=50.0, capacity=50.0)
    twin.add_external_feed("feed", "bin", flow=60.0, grade=0.03)

    m0, q0 = twin.total_mass(), twin.total_metal()
    twin.run(duration=5.0, dt=0.25)

    assert twin.mass_balance_error(m0) < 1e-9
    assert twin.metal_balance_error(q0) < 1e-9
    # External feed accounting: 60 t/h * 5 h = 300 t in
    assert twin.external_mass_in == pytest.approx(300.0)


def test_wear_degrades_throughput():
    twin = create_plant_twin()
    twin.add_zone("a", mass=100000.0, grade=0.01)
    twin.add_zone("b", mass=0.0)
    eq = twin.add_equipment("mill", "a", "b",
                            nominal_flow=200.0, capacity=200.0,
                            wear_rate=0.01)  # 1%/h wear
    cap0 = eq.current_capacity()
    twin.run(duration=20.0, dt=1.0)
    assert eq.wear == pytest.approx(0.2)
    assert eq.current_capacity() == pytest.approx(cap0 * 0.8)
    # Throughput in the last hour < first hour
    series = twin.telemetry("mill")["flow"]
    assert series[-1] < series[0]


def test_anomaly_fires_on_planted_spike_zscore():
    stream = TelemetryStream("test", z_threshold=3.0)
    anomalies = []
    # Baseline: steady flow
    for t in range(30):
        anomalies += stream.record(TelemetryPoint(float(t), "2024-01-01", 100.0, 0.02))
    # Planted spike
    anomalies += stream.record(TelemetryPoint(30.0, "2024-01-01", 1000.0, 0.02))

    z_anomalies = [a for a in anomalies if a.kind == "zscore"]
    assert len(z_anomalies) == 1
    assert z_anomalies[0].value == pytest.approx(1000.0)


def test_anomaly_fires_on_threshold():
    stream = TelemetryStream("test", flow_limits=(0.0, 500.0))
    anomalies = stream.record(TelemetryPoint(0.0, "2024-01-01", 900.0, 0.02))
    assert any(a.kind == "threshold_high" for a in anomalies)


def test_twin_records_telemetry_and_anomalies():
    twin = build_closed_loop_twin()
    # Tight threshold so the nominal 100 t/h flow trips it
    twin.streams["crusher"].flow_limits = (0.0, 50.0)
    twin.run(duration=2.0, dt=0.5)

    assert len(twin.get_anomalies()) > 0
    series = twin.telemetry("crusher")
    assert len(series["flow"]) == 4  # 2 h / 0.5 h steps


def test_determinism():
    def run_once():
        twin = build_closed_loop_twin()
        twin.run(duration=5.0, dt=0.5)
        return twin.to_dict()

    first, second = run_once(), run_once()
    assert first == second


def test_terrain_generation_deterministic():
    """advanced_simulation terrain noise no longer uses the global RNG."""
    import numpy as np
    from src.api.digital_twin.advanced_simulation import TerrainModel

    t1 = TerrainModel(100.0, 100.0, resolution=16)
    t2 = TerrainModel(100.0, 100.0, resolution=16)
    np.random.seed(12345)
    t1.generate_from_noise(seed=None)
    np.random.seed(99999)  # different global state — must not matter
    t2.generate_from_noise(seed=None)
    assert np.array_equal(t1.heightmap, t2.heightmap)
