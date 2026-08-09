"""
Regular block model builder + grade-tonnage engine — pure logic.

Block model construction uses Ordinary Kriging of sample grades onto a
regular 3-D grid (block centroids).  Grade-tonnage reporting sweeps a cutoff
range and returns tonnage / average grade / metal arrays honouring the
mass balance: tonnage = sum(block_volume x density) above cutoff,
metal = sum(tonnage x grade) above cutoff.

Code reuse: estimation is delegated to the existing geostatistics core
(``api.geostatistics.kriging.OrdinaryKriging`` — covariance-form OK solving
[K 1; 1 0] w = [k; 1], variance = sill - w.k - mu) with the core
``kriging.VariogramModel`` (via the sibling adapter
``resource_monte_carlo.logic.VariogramSpec``).  This module adds the regular
block-grid construction, density handling and grade-tonnage reporting.
"""

from __future__ import annotations

import io
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from api.geostatistics.kriging import (
    OrdinaryKriging,
    Point3D as _CorePoint3D,
    SearchEllipsoid,
    SearchParameters,
    SearchType,
)
from ..resource_monte_carlo.logic import VariogramSpec


@dataclass
class Block:
    x: float
    y: float
    z: float
    grade: float
    density: float
    kriging_variance: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "grade": self.grade, "density": self.density,
            "kriging_variance": self.kriging_variance,
            "n_samples": self.n_samples,
        }


def grid_centroids(
    origin: Sequence[float],
    block_size: Sequence[float],
    n_blocks: Sequence[int],
) -> np.ndarray:
    """Regular grid of block centroids (row-major x fastest)."""
    ox, oy, oz = (float(v) for v in origin)
    sx, sy, sz = (float(v) for v in block_size)
    nx, ny, nz = (int(v) for v in n_blocks)
    if min(nx, ny, nz) < 1 or min(sx, sy, sz) <= 0:
        raise ValueError("invalid block geometry")
    xs = ox + (np.arange(nx) + 0.5) * sx
    ys = oy + (np.arange(ny) + 0.5) * sy
    zs = oz + (np.arange(nz) + 0.5) * sz
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _make_engine(
    sample_coords: np.ndarray,
    sample_values: np.ndarray,
    spec: VariogramSpec,
    max_samples: int = 32,
    search_radius: Optional[float] = None,
) -> OrdinaryKriging:
    """Build a core OrdinaryKriging engine over the samples."""
    if search_radius is None:
        search_radius = 3.0 * spec.range
    ellipsoid = SearchEllipsoid(
        radius_major=search_radius,
        radius_minor=search_radius,
        radius_vertical=search_radius,
    )
    params = SearchParameters(
        ellipsoid=ellipsoid,
        search_type=SearchType.NEAREST,
        min_samples=2,
        max_samples=max_samples,
    )
    engine = OrdinaryKriging(spec._core, params)
    engine.set_data([
        _CorePoint3D(float(c[0]), float(c[1]), float(c[2]),
                     float(v), 1.0, str(i))
        for i, (c, v) in enumerate(zip(sample_coords, sample_values))
    ])
    return engine


def ordinary_kriging_estimate(
    sample_coords: np.ndarray,
    sample_values: np.ndarray,
    target: np.ndarray,
    spec: VariogramSpec,
    max_samples: int = 32,
) -> Optional[Dict[str, float]]:
    """Ordinary-kriging estimate at one target point (core OK engine).

    Returns estimate, kriging variance and sample count, or None when fewer
    than two samples fall inside the search ellipsoid.
    """
    engine = _make_engine(sample_coords, sample_values, spec, max_samples)
    res = engine.estimate(_CorePoint3D(*[float(v) for v in target]))
    if res is None:
        return None
    return {"estimate": float(res.estimate),
            "variance": float(res.variance),
            "n_samples": int(res.n_samples)}


def build_block_model(
    sample_coords: np.ndarray,
    sample_values: np.ndarray,
    origin: Sequence[float],
    block_size: Sequence[float],
    n_blocks: Sequence[int],
    spec: VariogramSpec,
    density: float = 2.7,
    density_field: Optional[np.ndarray] = None,
    max_samples: int = 32,
) -> List[Block]:
    """Krige sample grades onto a regular block grid.

    ``density`` is the default in-situ bulk density; ``density_field`` (one
    value per block, row-major order of :func:`grid_centroids`) overrides it
    per block.
    """
    sample_coords = np.asarray(sample_coords, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    if sample_coords.ndim != 2 or sample_coords.shape[1] != 3:
        raise ValueError("sample_coords must be (n, 3)")
    if len(sample_values) != len(sample_coords) or len(sample_coords) < 2:
        raise ValueError("need >=2 samples with matching values")
    if density <= 0:
        raise ValueError("density must be positive")

    centroids = grid_centroids(origin, block_size, n_blocks)
    n_total = len(centroids)
    if density_field is not None:
        density_field = np.asarray(density_field, dtype=float)
        if len(density_field) != n_total:
            raise ValueError("density_field must have one value per block")
        if np.any(density_field <= 0):
            raise ValueError("density_field values must be positive")

    engine = _make_engine(sample_coords, sample_values, spec, max_samples)

    blocks: List[Block] = []
    for i, c in enumerate(centroids):
        res = engine.estimate(_CorePoint3D(float(c[0]), float(c[1]), float(c[2])))
        if res is None:
            continue
        rho = float(density_field[i]) if density_field is not None else density
        blocks.append(Block(
            x=float(c[0]), y=float(c[1]), z=float(c[2]),
            grade=float(res.estimate), density=rho,
            kriging_variance=float(res.variance), n_samples=int(res.n_samples),
        ))
    return blocks


def grade_tonnage(
    blocks: Sequence[Block],
    block_volume: float,
    cutoffs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Grade-tonnage arrays for a cutoff sweep.

    For each cutoff c: tonnage = sum(volume x density | grade >= c),
    avg_grade = metal / tonnage (0 when tonnage is 0), metal = sum of
    tonnage x grade above cutoff.
    """
    if block_volume <= 0:
        raise ValueError("block_volume must be positive")
    grades = np.array([b.grade for b in blocks], dtype=float)
    masses = np.array([b.density for b in blocks], dtype=float) * block_volume
    cutoffs = np.asarray(cutoffs, dtype=float)

    tonnage = np.zeros(len(cutoffs))
    metal = np.zeros(len(cutoffs))
    for i, c in enumerate(cutoffs):
        mask = grades >= c
        tonnage[i] = masses[mask].sum()
        metal[i] = (masses[mask] * grades[mask]).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_grade = np.where(tonnage > 0, metal / np.maximum(tonnage, 1e-300), 0.0)
    return {
        "cutoff": cutoffs,
        "tonnage": tonnage,
        "avg_grade": avg_grade,
        "metal": metal,
    }


def cutoff_sweep(
    blocks: Sequence[Block],
    block_volume: float,
    n_steps: int = 20,
    cutoff_min: float = 0.0,
    cutoff_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Cutoff sweep from cutoff_min to cutoff_max (default: max block grade)."""
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if not blocks:
        raise ValueError("no blocks supplied")
    if cutoff_max is None:
        cutoff_max = max(b.grade for b in blocks)
    if cutoff_max < cutoff_min:
        raise ValueError("cutoff_max must be >= cutoff_min")
    cutoffs = np.linspace(cutoff_min, cutoff_max, n_steps)
    return grade_tonnage(blocks, block_volume, cutoffs)


def grade_tonnage_csv(gt: Dict[str, np.ndarray]) -> str:
    """Render grade-tonnage arrays as CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["cutoff", "tonnage", "avg_grade", "metal"])
    for i in range(len(gt["cutoff"])):
        writer.writerow([
            f"{gt['cutoff'][i]:.6g}",
            f"{gt['tonnage'][i]:.6g}",
            f"{gt['avg_grade'][i]:.6g}",
            f"{gt['metal'][i]:.6g}",
        ])
    return buf.getvalue()
