"""
Real-time geostatistical drift detection for assay/grade streams.

Detectors:
- CUSUM (two-sided, configurable reference value k and decision interval h)
  on standardized residuals; chart resets after each alarm.
- EWMA control chart with time-varying exact limits (lambda, L sigma).
- Rolling declustered mean vs baseline mean (cell declustering weights).
- Rolling variogram sill vs baseline sill (pairwise-lag semivariogram).

Alerts carry detector name, sample index, timestamp, magnitude and direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """Two-sided tabular CUSUM on standardized residuals.

    S+ = max(0, S+ + z - k)   detects upward shifts
    S- = max(0, S- - z - k)   detects downward shifts
    Alarm when S+ > h or S- > h; the chart restarts (S=0) after an alarm.
    """

    def __init__(self, mean: float, std: float, k: float = 0.5, h: float = 5.0):
        if std <= 0:
            raise ValueError("std must be positive")
        if k < 0 or h <= 0:
            raise ValueError("k must be >= 0 and h > 0")
        self.mean, self.std, self.k, self.h = mean, std, k, h
        self.s_plus = 0.0
        self.s_minus = 0.0

    def update(self, value: float) -> Optional[Dict[str, Any]]:
        z = (value - self.mean) / self.std
        self.s_plus = max(0.0, self.s_plus + z - self.k)
        self.s_minus = max(0.0, self.s_minus - z - self.k)
        if self.s_plus > self.h:
            mag = self.s_plus
            self.s_plus = 0.0
            return {"detector": "cusum", "direction": "up", "magnitude": mag}
        if self.s_minus > self.h:
            mag = self.s_minus
            self.s_minus = 0.0
            return {"detector": "cusum", "direction": "down", "magnitude": mag}
        return None


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

class EWMADetector:
    """EWMA control chart with exact time-varying limits.

    z_t = lambda * x_t + (1 - lambda) * z_{t-1}
    limit_t = L * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2t)))
    """

    def __init__(self, mean: float, std: float, lam: float = 0.2,
                 L: float = 3.0):
        if std <= 0:
            raise ValueError("std must be positive")
        if not (0 < lam <= 1) or L <= 0:
            raise ValueError("need 0 < lam <= 1 and L > 0")
        self.mean, self.std, self.lam, self.L = mean, std, lam, L
        self.z = 0.0
        self.t = 0

    def update(self, value: float) -> Optional[Dict[str, Any]]:
        x = (value - self.mean) / self.std
        self.t += 1
        self.z = self.lam * x + (1 - self.lam) * self.z
        var_factor = self.lam / (2 - self.lam) * (1 - (1 - self.lam) ** (2 * self.t))
        limit = self.L * math.sqrt(var_factor)
        if abs(self.z) > limit:
            direction = "up" if self.z > 0 else "down"
            return {"detector": "ewma", "direction": direction,
                    "magnitude": abs(self.z) / limit}
        return None


# ---------------------------------------------------------------------------
# Geostatistics: declustered mean + variogram sill
# ---------------------------------------------------------------------------

def declustering_weights(coords: np.ndarray,
                         cell_size: Optional[float] = None) -> np.ndarray:
    """Cell-declustering weights: 1 / (number of samples in the cell)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (n, 2)")
    n = len(coords)
    if n == 0:
        return np.array([])
    if cell_size is None:
        span = np.ptp(coords, axis=0).max()
        cell_size = span / max(int(round(math.sqrt(n))), 1)
        if cell_size <= 0:
            return np.ones(n)
    cells = np.floor((coords - coords.min(axis=0)) / cell_size).astype(np.int64)
    keys = cells[:, 0] * 1_000_003 + cells[:, 1]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def declustered_mean(values: np.ndarray,
                     coords: Optional[np.ndarray] = None,
                     cell_size: Optional[float] = None) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    if coords is None:
        return float(values.mean())
    w = declustering_weights(coords, cell_size)
    return float(np.dot(w, values) / w.sum())


def estimate_sill(values: np.ndarray,
                  coords: Optional[np.ndarray] = None,
                  n_lags: int = 8) -> float:
    """Variogram sill: mean semivariance over the longest third of lag bins.

    Uses index distance when coordinates are absent (gridded/stream data),
    pairwise Euclidean distance otherwise. Semivariance gamma(h) =
    0.5 * mean((v_i - v_j)^2) per lag bin.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 4:
        return float(np.var(values)) if n else 0.0
    if coords is None:
        # Serial stream: for an uncorrelated process the variogram sill IS
        # the process variance; gamma(h) ~ var for every lag h >= 1.
        return float(np.var(values, ddof=1))
    coords = np.asarray(coords, dtype=float)
    d = np.hypot(coords[:, 0:1] - coords[:, 0][None, :],
                 coords[:, 1:2] - coords[:, 1][None, :])
    iu = np.triu_indices(n, k=1)
    dists = d[iu]
    diffsq = (values[:, None] - values[None, :])[iu] ** 2
    max_d = dists.max()
    if max_d <= 0:
        return float(np.var(values))
    edges = np.linspace(0, max_d, n_lags + 1)[1:]
    gamma = np.full(n_lags, np.nan)
    for b in range(n_lags):
        lo = 0.0 if b == 0 else edges[b - 1]
        hi = edges[b]
        mask = (dists > lo) & (dists <= hi)
        if mask.any():
            gamma[b] = 0.5 * diffsq[mask].mean()
    tail = gamma[max(n_lags - n_lags // 3, 1):]
    tail = tail[np.isfinite(tail)]
    if not len(tail):
        return float(np.var(values))
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Stream monitor
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    detector: str
    index: int
    value: float
    magnitude: float
    direction: str
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "index": self.index,
            "value": self.value,
            "magnitude": self.magnitude,
            "direction": self.direction,
            "timestamp": self.timestamp,
        }


class DriftMonitor:
    """Combined CUSUM + EWMA + rolling geostat drift monitor for one stream."""

    def __init__(self,
                 baseline_values: Sequence[float],
                 baseline_coords: Optional[np.ndarray] = None,
                 cusum_k: float = 0.5, cusum_h: float = 5.0,
                 ewma_lam: float = 0.2, ewma_L: float = 3.0,
                 window: int = 50, mean_tol: float = 0.5,
                 sill_tol: float = 0.5):
        values = np.asarray(baseline_values, dtype=float)
        if len(values) < 8:
            raise ValueError("need at least 8 baseline samples")
        self.base_mean = float(values.mean())
        self.base_std = float(values.std(ddof=1))
        if self.base_std <= 0:
            raise ValueError("baseline std must be positive")
        self.base_sill = estimate_sill(values, baseline_coords)
        if self.base_sill <= 0:
            self.base_sill = float(np.var(values))
        self.cusum = CUSUMDetector(self.base_mean, self.base_std,
                                   cusum_k, cusum_h)
        self.ewma = EWMADetector(self.base_mean, self.base_std,
                                 ewma_lam, ewma_L)
        self.window = int(window)
        self.mean_tol = float(mean_tol)
        self.sill_tol = float(sill_tol)
        self.n_seen = 0
        self.buf_values: List[float] = []
        self.buf_coords: List[Optional[Tuple[float, float]]] = []
        self.alerts: List[Alert] = []
        self._mean_violated = False
        self._sill_violated = False

    def push(self, values: Sequence[float],
             coords: Optional[Sequence[Tuple[float, float]]] = None,
             timestamps: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
        """Push a batch; returns the alerts raised by this batch."""
        values = [float(v) for v in values]
        n = len(values)
        if coords is not None and len(coords) != n:
            raise ValueError("coords length must match values")
        if timestamps is not None and len(timestamps) != n:
            raise ValueError("timestamps length must match values")

        new_alerts: List[Alert] = []
        for i, v in enumerate(values):
            idx = self.n_seen
            ts = timestamps[i] if timestamps is not None else None
            for detector in (self.cusum, self.ewma):
                hit = detector.update(v)
                if hit is not None:
                    new_alerts.append(Alert(
                        detector=hit["detector"], index=idx, value=v,
                        magnitude=float(hit["magnitude"]),
                        direction=hit["direction"], timestamp=ts))
            c = coords[i] if coords is not None else None
            self.buf_values.append(v)
            self.buf_coords.append(c)
            if len(self.buf_values) > self.window:
                self.buf_values.pop(0)
                self.buf_coords.pop(0)
            self.n_seen += 1

            if len(self.buf_values) == self.window:
                new_alerts.extend(self._rolling_checks(idx, v, ts))

        self.alerts.extend(new_alerts)
        return [a.to_dict() for a in new_alerts]

    def _rolling_checks(self, idx: int, value: float,
                        ts: Optional[float]) -> List[Alert]:
        out: List[Alert] = []
        w = np.asarray(self.buf_values, dtype=float)
        coords_arr = None
        if all(c is not None for c in self.buf_coords):
            coords_arr = np.asarray(self.buf_coords, dtype=float)

        # rolling declustered mean vs baseline (in baseline-std units)
        dm = declustered_mean(w, coords_arr)
        shift = (dm - self.base_mean) / self.base_std
        if abs(shift) > self.mean_tol:
            if not self._mean_violated:
                out.append(Alert(
                    detector="declustered_mean", index=idx, value=value,
                    magnitude=float(abs(shift)),
                    direction="up" if shift > 0 else "down", timestamp=ts))
            self._mean_violated = True
        else:
            self._mean_violated = False

        # rolling sill vs baseline sill (relative change)
        sill = estimate_sill(w, coords_arr)
        rel = (sill - self.base_sill) / self.base_sill if self.base_sill > 0 else 0.0
        if abs(rel) > self.sill_tol:
            if not self._sill_violated:
                out.append(Alert(
                    detector="sill_change", index=idx, value=value,
                    magnitude=float(abs(rel)),
                    direction="up" if rel > 0 else "down", timestamp=ts))
            self._sill_violated = True
        else:
            self._sill_violated = False
        return out

    def list_alerts(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.alerts]


# ---------------------------------------------------------------------------
# Stream registry (in-memory; per-process like the rest of the dev API)
# ---------------------------------------------------------------------------

class StreamRegistry:
    def __init__(self):
        self._monitors: Dict[str, DriftMonitor] = {}

    def register(self, stream_id: str, baseline_values: Sequence[float],
                 baseline_coords=None, **kwargs) -> DriftMonitor:
        monitor = DriftMonitor(baseline_values, baseline_coords, **kwargs)
        self._monitors[stream_id] = monitor
        return monitor

    def get(self, stream_id: str) -> Optional[DriftMonitor]:
        return self._monitors.get(stream_id)

    def remove(self, stream_id: str) -> bool:
        return self._monitors.pop(stream_id, None) is not None


registry = StreamRegistry()
