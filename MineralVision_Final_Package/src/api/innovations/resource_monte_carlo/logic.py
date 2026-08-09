"""
Resource uncertainty via conditional Gaussian simulation — pure logic.

Method
------
"Simplified but correct" Sequential Gaussian Simulation equivalent: instead
of visiting grid nodes sequentially, the exact multivariate-Gaussian
conditioning identity is applied once over the whole simulation grid:

    mean_cond = m + C_gd C_dd^-1 (d - m)
    cov_cond  = C_gg - C_gd C_dd^-1 C_dg

where C is the covariance function derived from the variogram model
(spherical / exponential / gaussian with nugget), d the conditioning data and
m the (simple-kriging) stationary mean taken as the data mean.  Realisations
are drawn through the Cholesky factor of ``cov_cond`` (LU/Cholesky path) with
a seeded ``numpy.random.Generator``.  This is the exact conditional
distribution of a Gaussian random field; the classic sequential algorithm
produces the same law node-by-node.

Limits (documented cap): the dense Cholesky factorisation is O(n^3), so the
simulation grid is capped at ``MAX_GRID_NODES`` (500) and the conditioning
data at ``MAX_DATA_POINTS`` (500).  Requests above the cap are rejected with
a clear error; use sub-grids or moving neighbourhoods for larger models.

Code reuse: the variogram covariance itself is delegated to the existing
geostatistics core (``api.geostatistics.kriging.VariogramModel`` — spherical
1.5h/a - 0.5(h/a)^3, exponential 1 - exp(-3h/a), gaussian 1 - exp(-3(h/a)^2));
this module adds vectorized evaluation and the conditioning algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from api.geostatistics.kriging import VariogramModel as _CoreVariogramModel

MAX_GRID_NODES = 500
MAX_DATA_POINTS = 500


@dataclass
class VariogramSpec:
    """Single-structure variogram: model, nugget, contribution, range.

    Thin vectorized adapter over the geostatistics-core
    ``kriging.VariogramModel`` (identical covariance definitions).
    """
    model: str = "spherical"   # spherical | exponential | gaussian
    nugget: float = 0.0
    contribution: float = 1.0
    range: float = 100.0

    def __post_init__(self):
        if self.range <= 0:
            raise ValueError("variogram range must be positive")
        if self.model not in ("spherical", "exponential", "gaussian"):
            raise ValueError(f"unsupported variogram model: {self.model}")
        self._core = _CoreVariogramModel(
            nugget=self.nugget,
            structures=[{"model": self.model,
                         "contribution": self.contribution,
                         "range": self.range}],
        )

    @property
    def sill(self) -> float:
        return self.nugget + self.contribution

    def semivariance(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        flat = h.ravel()
        out = np.array([self._core.semivariance(float(v)) for v in flat])
        return out.reshape(h.shape)

    def covariance(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        flat = h.ravel()
        out = np.array([self._core.covariance(float(v)) for v in flat])
        return out.reshape(h.shape)


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between rows of a (n,3) and b (m,3)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def conditional_simulation(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    grid_coords: np.ndarray,
    spec: VariogramSpec,
    n_realizations: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Exact Gaussian conditional simulation on a grid.

    Returns dict with realizations (n_grid, n_realizations), conditional_mean,
    conditional_std, conditional_cov, stationary mean and diagnostics.
    """
    data_coords = np.asarray(data_coords, dtype=float)
    data_values = np.asarray(data_values, dtype=float)
    grid_coords = np.asarray(grid_coords, dtype=float)

    if data_coords.ndim != 2 or data_coords.shape[1] != 3:
        raise ValueError("data_coords must be (n, 3)")
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError("grid_coords must be (m, 3)")
    if len(data_values) != len(data_coords):
        raise ValueError("data_values length must match data_coords")
    if len(data_coords) < 2:
        raise ValueError("at least 2 conditioning data points are required")
    if len(grid_coords) > MAX_GRID_NODES:
        raise ValueError(
            f"grid has {len(grid_coords)} nodes; cap is {MAX_GRID_NODES} "
            "(dense Cholesky limit — sub-grid the model)")
    if len(data_coords) > MAX_DATA_POINTS:
        raise ValueError(
            f"{len(data_coords)} conditioning points; cap is {MAX_DATA_POINTS}")
    if n_realizations < 1:
        raise ValueError("n_realizations must be >= 1")

    m = float(np.mean(data_values))

    C_dd = spec.covariance(_pairwise_distances(data_coords, data_coords))
    C_gd = spec.covariance(_pairwise_distances(grid_coords, data_coords))
    C_gg = spec.covariance(_pairwise_distances(grid_coords, grid_coords))

    # Numerical nugget jitter for factorisation stability.
    jitter = 1e-10 * spec.sill
    C_dd = C_dd + jitter * np.eye(len(data_coords))

    A = np.linalg.solve(C_dd, C_gd.T)            # C_dd^-1 C_dg
    cond_mean = m + A.T @ (data_values - m)
    cond_cov = C_gg - C_gd @ A
    cond_cov = 0.5 * (cond_cov + cond_cov.T)     # enforce symmetry
    cond_cov += jitter * np.eye(len(grid_coords))

    L = np.linalg.cholesky(cond_cov)
    cond_std = np.sqrt(np.clip(np.diag(cond_cov), 0.0, None))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(grid_coords), n_realizations))
    realizations = cond_mean[:, None] + L @ z

    return {
        "realizations": realizations,
        "conditional_mean": cond_mean,
        "conditional_std": cond_std,
        "conditional_cov": cond_cov,
        "mean": m,
        "unconditional_variance": spec.sill,
        "seed": seed,
        "n_realizations": n_realizations,
    }


def tonnage_grade_above_cutoff(
    realizations: np.ndarray,
    block_tonnages: np.ndarray,
    cutoff: float,
) -> Dict[str, np.ndarray]:
    """Per-realization tonnage and average grade above a grade cutoff.

    ``block_tonnages`` is the tonnage represented by each grid node
    (volume x density).  Grade above cutoff is tonnage-weighted; when no block
    exceeds the cutoff, tonnage and grade are both 0.
    """
    realizations = np.asarray(realizations, dtype=float)
    block_tonnages = np.asarray(block_tonnages, dtype=float)
    if realizations.ndim != 2:
        raise ValueError("realizations must be (n_grid, n_realizations)")
    if len(block_tonnages) != realizations.shape[0]:
        raise ValueError("block_tonnages length must match grid")
    if np.any(block_tonnages <= 0):
        raise ValueError("block_tonnages must be positive")

    mask = realizations >= cutoff                      # (n_grid, n_real)
    ton = mask * block_tonnages[:, None]
    metal = ton * realizations
    tonnage = ton.sum(axis=0)
    metal_total = metal.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        grade = np.where(tonnage > 0, metal_total / np.maximum(tonnage, 1e-300), 0.0)
    return {"tonnage": tonnage, "grade": grade, "metal": metal_total}


def uncertainty_summary(
    sim_result: Dict[str, Any],
    block_tonnages: np.ndarray,
    cutoff: float,
) -> Dict[str, Any]:
    """P10/P50/P90 (percentile) tonnage, grade and metal above cutoff.

    Percentile convention: p10/p50/p90 are the 10th/50th/90th percentiles of
    the realization distribution (p10 = conservative, p90 = optimistic —
    equivalent to mining P90/P50/P10 exceedance nomenclature).
    """
    tg = tonnage_grade_above_cutoff(
        sim_result["realizations"], block_tonnages, cutoff)
    summary = {}
    for key in ("tonnage", "grade", "metal"):
        arr = tg[key]
        summary[key] = {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
        }
    return {
        "cutoff": float(cutoff),
        "n_realizations": int(sim_result["n_realizations"]),
        "seed": sim_result["seed"],
        "stationary_mean": float(sim_result["mean"]),
        "unconditional_variance": float(sim_result["unconditional_variance"]),
        "mean_conditional_variance": float(
            np.mean(np.diag(sim_result["conditional_cov"]))),
        "percentiles": summary,
    }
