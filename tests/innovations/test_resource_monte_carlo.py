"""Deterministic tests for conditional-simulation resource uncertainty."""

import numpy as np
import pytest
from api.innovations.resource_monte_carlo import logic, router
from fastapi import FastAPI
from fastapi.testclient import TestClient

SPEC = logic.VariogramSpec(model="spherical", nugget=0.1,
                           contribution=0.9, range=50.0)


def _data():
    """8 conditioning samples on a 20 m grid (mean exactly 2.0)."""
    coords, values = [], []
    vals = [1.2, 2.8, 1.6, 2.4, 2.6, 1.4, 2.2, 1.8]  # mean = 2.0
    k = 0
    for i in range(4):
        for j in range(2):
            coords.append([i * 20.0, j * 20.0, 0.0])
            values.append(vals[k])
            k += 1
    return np.array(coords), np.array(values)


def _grid(n=10):
    pts = np.linspace(0, 60, n)
    gx, gy = np.meshgrid(pts, pts, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])


# ---------------------------------------------------------------- logic ----

def test_variogram_matches_geostatistics_core():
    """VariogramSpec covariance must equal the core kriging.VariogramModel."""
    from api.geostatistics.kriging import VariogramModel as CoreVG
    core = CoreVG(nugget=0.1, structures=[{
        "model": "spherical", "contribution": 0.9, "range": 50.0}])
    for h in (0.0, 1.0, 25.0, 49.9, 50.0, 75.0, 200.0):
        assert SPEC.covariance(h) == pytest.approx(core.covariance(h))
    assert SPEC.sill == pytest.approx(core.sill)


def test_seed_reproducibility():
    dc, dv = _data()
    g = _grid()
    a = logic.conditional_simulation(dc, dv, g, SPEC, 25, seed=42)
    b = logic.conditional_simulation(dc, dv, g, SPEC, 25, seed=42)
    np.testing.assert_array_equal(a["realizations"], b["realizations"])
    c = logic.conditional_simulation(dc, dv, g, SPEC, 25, seed=43)
    assert not np.allclose(a["realizations"], c["realizations"])


def test_realization_mean_matches_data_mean():
    """Ensemble mean converges to the conditional mean (== data mean when the
    grid carries no trend); tight check vs conditional mean, loose vs 2.0."""
    dc, dv = _data()
    g = _grid()
    sim = logic.conditional_simulation(dc, dv, g, SPEC, 400, seed=7)
    reals = sim["realizations"]
    # By construction E[realization] == conditional_mean exactly.
    assert np.mean(reals) == pytest.approx(
        np.mean(sim["conditional_mean"]), abs=0.05)
    # Conditional mean interpolates the data: stays inside the data range
    # (small overshoot tolerance) and averages close to the stationary mean.
    assert sim["conditional_mean"].min() >= dv.min() - 0.5
    assert sim["conditional_mean"].max() <= dv.max() + 0.5
    assert np.mean(sim["conditional_mean"]) == pytest.approx(2.0, abs=0.35)


def test_conditional_variance_reduced_at_data_points():
    """Conditioning on data collapses the variance at sampled locations."""
    dc, dv = _data()
    # Grid nodes exactly at the data locations plus one far-away node.
    g = np.vstack([dc, [[500.0, 500.0, 0.0]]])
    sim = logic.conditional_simulation(dc, dv, g, SPEC, 5, seed=1)
    cond_var = np.diag(sim["conditional_cov"])
    sill = SPEC.sill
    # At data points: conditional variance << sill.
    assert np.all(cond_var[:len(dc)] < 0.05 * sill)
    # Far from data: conditional variance ~ unconditional sill.
    assert cond_var[-1] > 0.9 * sill
    # Realizations honour the data (interpolate at sampled nodes).
    np.testing.assert_allclose(sim["realizations"][:len(dc), :].mean(axis=1),
                               dv, atol=0.05)


def test_conditional_mean_interpolates_data():
    """Exactness property: at data locations the conditional mean equals the
    sample value (nugget small relative to sill)."""
    dc, dv = _data()
    sim = logic.conditional_simulation(dc, dv, dc, SPEC, 3, seed=5)
    np.testing.assert_allclose(sim["conditional_mean"], dv, atol=0.02)


def test_uncertainty_percentiles_ordered_and_mass_consistent():
    dc, dv = _data()
    g = _grid()
    block_tonnages = np.full(len(g), 1000.0)  # 1 kt per node
    sim = logic.conditional_simulation(dc, dv, g, SPEC, 300, seed=11)
    summary = logic.uncertainty_summary(sim, block_tonnages, cutoff=2.0)
    for key in ("tonnage", "grade", "metal"):
        p = summary["percentiles"][key]
        assert p["p10"] <= p["p50"] <= p["p90"]
    # Mass balance: tonnage <= total, metal == tonnage*grade where tonnage>0.
    assert summary["percentiles"]["tonnage"]["p90"] <= len(g) * 1000.0
    tg = logic.tonnage_grade_above_cutoff(sim["realizations"],
                                          block_tonnages, 2.0)
    nonzero = tg["tonnage"] > 0
    assert np.allclose(tg["metal"][nonzero],
                       tg["tonnage"][nonzero] * tg["grade"][nonzero])
    # Grade above cutoff must be >= cutoff.
    assert np.all(tg["grade"][nonzero] >= 2.0 - 1e-9)


def test_cutoff_extremes():
    dc, dv = _data()
    g = _grid(5)
    tons = np.full(len(g), 500.0)
    sim = logic.conditional_simulation(dc, dv, g, SPEC, 20, seed=3)
    tg0 = logic.tonnage_grade_above_cutoff(sim["realizations"], tons, -1.0)
    assert np.all(tg0["tonnage"] == len(g) * 500.0)
    tg_hi = logic.tonnage_grade_above_cutoff(sim["realizations"], tons, 1e9)
    assert np.all(tg_hi["tonnage"] == 0.0)
    assert np.all(tg_hi["grade"] == 0.0)


def test_grid_cap_enforced():
    dc, dv = _data()
    big = np.column_stack([np.arange(501), np.zeros(501), np.zeros(501)])
    with pytest.raises(ValueError, match="cap"):
        logic.conditional_simulation(dc, dv, big, SPEC, 2, seed=1)


def test_exponential_and_gaussian_models():
    dc, dv = _data()
    g = _grid(5)
    for model in ("exponential", "gaussian"):
        spec = logic.VariogramSpec(model=model, nugget=0.1,
                                   contribution=0.9, range=50.0)
        sim = logic.conditional_simulation(dc, dv, g, spec, 10, seed=2)
        assert sim["realizations"].shape == (len(g), 10)
        assert np.all(np.diag(sim["conditional_cov"]) >= 0)


# ------------------------------------------------------------------ API ----

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _payload(seed=42, n_real=100):
    dc, dv = _data()
    g = _grid()
    return {
        "data": [{"x": c[0], "y": c[1], "z": c[2], "value": v}
                 for c, v in zip(dc, dv, strict=False)],
        "grid": [{"x": c[0], "y": c[1], "z": c[2], "tonnage": 1000.0}
                 for c in g],
        "variogram": {"model": "spherical", "nugget": 0.1,
                      "contribution": 0.9, "range": 50.0},
        "n_realizations": n_real,
        "seed": seed,
        "cutoff": 2.0,
    }


def test_api_simulate(client):
    r = client.post("/innovations/resource_monte_carlo/simulate",
                    json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["n_realizations"] == 100
    assert body["seed"] == 42
    for key in ("tonnage", "grade", "metal"):
        p = body["percentiles"][key]
        assert p["p10"] <= p["p50"] <= p["p90"]
    assert body["mean_conditional_variance"] < body["unconditional_variance"]
    assert len(body["conditional_mean"]) == 100
    assert len(body["conditional_std"]) == 100


def test_api_reproducible_realizations(client):
    r1 = client.post("/innovations/resource_monte_carlo/realizations",
                     json=_payload(seed=99, n_real=10))
    r2 = client.post("/innovations/resource_monte_carlo/realizations",
                     json=_payload(seed=99, n_real=10))
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["realizations"] == r2.json()["realizations"]


def test_api_grid_cap_422(client):
    p = _payload()
    p["grid"] = [{"x": float(i), "y": 0.0, "z": 0.0, "tonnage": 1.0}
                 for i in range(501)]
    r = client.post("/innovations/resource_monte_carlo/simulate", json=p)
    assert r.status_code == 422
