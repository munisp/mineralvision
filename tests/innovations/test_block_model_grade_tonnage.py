"""Deterministic tests for block model builder + grade-tonnage engine."""

import io
import csv as csvmod

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.innovations.block_model_grade_tonnage import router
from api.innovations.block_model_grade_tonnage import logic
from api.innovations.resource_monte_carlo.logic import VariogramSpec

SPEC = VariogramSpec(model="spherical", nugget=0.05, contribution=0.95,
                     range=60.0)
BLOCK_VOLUME = 10.0 * 10.0 * 10.0


def _samples():
    """25 samples on a 20 m grid covering 0..80 m; grade = 1 + x/80 + y/80."""
    coords, values = [], []
    for i in range(5):
        for j in range(5):
            x, y = i * 20.0, j * 20.0
            coords.append([x, y, 0.0])
            values.append(1.0 + x / 80.0 + y / 80.0)
    return np.array(coords), np.array(values)


def _posted_blocks():
    """10 blocks with grades 0.5, 1.0, ..., 5.0 and alternating densities."""
    return [
        logic.Block(x=float(i), y=0.0, z=0.0, grade=0.5 * (i + 1),
                    density=2.0 + 0.1 * (i % 2), kriging_variance=0.0,
                    n_samples=0)
        for i in range(10)
    ]


# ---------------------------------------------------------------- logic ----

def test_grid_centroids_geometry():
    c = logic.grid_centroids([0, 0, 0], [10, 10, 10], [2, 2, 1])
    assert c.shape == (4, 3)
    assert set(map(tuple, c)) == {
        (5.0, 5.0, 5.0), (5.0, 15.0, 5.0), (15.0, 5.0, 5.0), (15.0, 15.0, 5.0)}


def test_kriging_exact_at_sample_location():
    """OK honours the data: estimate at a sample point equals its value."""
    coords, values = _samples()
    est = logic.ordinary_kriging_estimate(coords, values, coords[7], SPEC)
    assert est["estimate"] == pytest.approx(values[7], abs=1e-6)
    assert est["variance"] < 1e-6 * SPEC.sill + 1e-9


def test_kriging_smooth_between_samples():
    coords, values = _samples()
    mid = np.array([10.0, 10.0, 0.0])  # midpoint of four corner samples
    est = logic.ordinary_kriging_estimate(coords, values, mid, SPEC)
    surrounding = [1.0, 1.25, 1.25, 1.5]  # grades of the 4 corner samples
    assert min(surrounding) - 0.2 <= est["estimate"] <= max(surrounding) + 0.2
    assert 0.0 <= est["variance"] <= SPEC.sill


def test_build_block_model_mass_and_fields():
    coords, values = _samples()
    blocks = logic.build_block_model(
        coords, values, origin=[0, 0, -5], block_size=[10, 10, 10],
        n_blocks=[9, 9, 1], spec=SPEC, density=2.5)
    assert len(blocks) == 81
    assert all(b.density == 2.5 for b in blocks)
    assert all(b.n_samples >= 2 for b in blocks)
    grades = np.array([b.grade for b in blocks])
    # Kriged grades bounded by data range (small tolerance for edge effects).
    assert grades.min() >= values.min() - 0.3
    assert grades.max() <= values.max() + 0.3

    # Per-block density field overrides the default.
    df = np.full(81, 3.1)
    blocks = logic.build_block_model(
        coords, values, origin=[0, 0, -5], block_size=[10, 10, 10],
        n_blocks=[9, 9, 1], spec=SPEC, density_field=df)
    assert all(b.density == 3.1 for b in blocks)


def test_grade_tonnage_exact_values():
    blocks = _posted_blocks()
    gt = logic.grade_tonnage(blocks, BLOCK_VOLUME,
                             np.array([0.0, 2.0, 5.0, 6.0]))
    masses = np.array([b.density for b in blocks]) * BLOCK_VOLUME
    grades = np.array([b.grade for b in blocks])
    for i, c in enumerate([0.0, 2.0, 5.0, 6.0]):
        mask = grades >= c
        assert gt["tonnage"][i] == pytest.approx(masses[mask].sum())
        assert gt["metal"][i] == pytest.approx((masses * grades)[mask].sum())
        exp_grade = ((masses * grades)[mask].sum() / masses[mask].sum()
                     if mask.any() else 0.0)
        assert gt["avg_grade"][i] == pytest.approx(exp_grade)
    # cutoff above max grade -> zero tonnage and zero grade
    assert gt["tonnage"][3] == 0.0
    assert gt["avg_grade"][3] == 0.0


def test_tonnage_monotonically_non_increasing():
    blocks = _posted_blocks()
    gt = logic.cutoff_sweep(blocks, BLOCK_VOLUME, n_steps=30)
    diffs = np.diff(gt["tonnage"])
    assert np.all(diffs <= 1e-12)
    # Metal is non-increasing too; avg grade is non-decreasing where defined.
    assert np.all(np.diff(gt["metal"]) <= 1e-12)
    nz = gt["tonnage"] > 0
    assert np.all(np.diff(gt["avg_grade"][nz]) >= -1e-12)


def test_mass_balance():
    """Total tonnage at cutoff 0 == sum(volume x density); metal consistent."""
    blocks = _posted_blocks()
    gt = logic.cutoff_sweep(blocks, BLOCK_VOLUME, n_steps=15)
    expected_tonnage = sum(b.density for b in blocks) * BLOCK_VOLUME
    expected_metal = sum(b.density * BLOCK_VOLUME * b.grade for b in blocks)
    assert gt["tonnage"][0] == pytest.approx(expected_tonnage)
    assert gt["metal"][0] == pytest.approx(expected_metal)
    assert gt["avg_grade"][0] == pytest.approx(expected_metal / expected_tonnage)


def test_csv_rendering_roundtrip():
    blocks = _posted_blocks()
    gt = logic.cutoff_sweep(blocks, BLOCK_VOLUME, n_steps=5)
    text = logic.grade_tonnage_csv(gt)
    rows = list(csvmod.reader(io.StringIO(text)))
    assert rows[0] == ["cutoff", "tonnage", "avg_grade", "metal"]
    assert len(rows) == 6
    assert float(rows[1][0]) == pytest.approx(gt["cutoff"][0])
    assert float(rows[1][1]) == pytest.approx(gt["tonnage"][0])


# ------------------------------------------------------------------ API ----

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _build_payload():
    coords, values = _samples()
    return {
        "samples": [{"x": c[0], "y": c[1], "z": c[2], "grade": v}
                    for c, v in zip(coords, values)],
        "geometry": {"origin": [0, 0, -5], "block_size": [10, 10, 10],
                     "n_blocks": [9, 9, 1]},
        "variogram": {"model": "spherical", "nugget": 0.05,
                      "contribution": 0.95, "range": 60.0},
        "density": 2.5,
    }


def test_api_build_and_monotonic_sweep(client):
    r = client.post("/innovations/block_model_grade_tonnage/build",
                    json=_build_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["n_blocks"] == 81
    assert body["block_volume"] == pytest.approx(1000.0)
    assert all(b["density"] == 2.5 for b in body["blocks"])

    r = client.post("/innovations/block_model_grade_tonnage/build-and-sweep",
                    json={"build": _build_payload(), "n_steps": 25})
    assert r.status_code == 200
    body = r.json()
    ton = np.array(body["tonnage"])
    assert np.all(np.diff(ton) <= 1e-9)
    # Mass balance at cutoff 0: 81 blocks x 1000 m3 x 2.5 t/m3.
    assert ton[0] == pytest.approx(81 * 1000.0 * 2.5)


def test_api_posted_blocks_and_csv(client):
    payload = {
        "blocks": [b.to_dict() for b in _posted_blocks()],
        "block_volume": BLOCK_VOLUME,
        "n_steps": 12,
    }
    r = client.post("/innovations/block_model_grade_tonnage/grade-tonnage",
                    json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n_steps"] == 12
    assert np.all(np.diff(body["tonnage"]) <= 1e-9)

    r = client.post("/innovations/block_model_grade_tonnage/grade-tonnage/csv",
                    json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    lines = r.text.strip().split("\n")
    assert lines[0] == "cutoff,tonnage,avg_grade,metal"
    assert len(lines) == 13
