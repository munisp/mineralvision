"""
Rig telemetry auto-logging — pure logic.

Regime segmentation
-------------------
Two-sided tabular CUSUM on the ROP series (ordered by measured depth),
after Page (1954):

    S+_i = max(0, S+_{i-1} + (x_i - mu - k))
    S-_i = max(0, S-_{i-1} - (x_i - mu + k))

with mu/sigma estimated over the submitted series, slack ``k =
slack_factor * sigma`` and decision threshold ``h = threshold_factor *
sigma``.  When either accumulator exceeds ``h`` a regime shift is declared;
the change boundary is backtracked to the index where the winning
accumulator last left zero (the classical CUSUM change-point estimate),
then the accumulators are reset.  Each resulting interval is summarised
(depth range, mean ROP/torque/RPM/vibration) and labelled slow/medium/fast
by mean-ROP rank.

Collar alignment & deviation
----------------------------
Depths are aligned to the collar by subtracting ``collar_depth`` (the
collar's down-hole datum, e.g. the drill-string stick-up / kelly bushing
offset).  The MWD-vs-collar deviation check compares the aligned final
measured depth against the planned total depth; |deviation| above
``deviation_tolerance`` is flagged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def cusum_change_points(
    values: np.ndarray,
    slack_factor: float = 0.5,
    threshold_factor: float = 4.0,
) -> List[int]:
    """Two-sided CUSUM change-point indices on a 1-D series.

    Returns sorted boundary indices (0 < i < n): index i starts a new
    regime (the change occurred between i-1 and i).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        return []
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < 1e-12:
        return []
    k = slack_factor * sigma
    h = threshold_factor * sigma

    boundaries: List[int] = []
    sp = sm = 0.0
    last_zero_p = last_zero_m = 0
    for i in range(n):
        sp = max(0.0, sp + (x[i] - mu - k))
        sm = max(0.0, sm - (x[i] - mu + k))
        if sp == 0.0:
            last_zero_p = i
        if sm == 0.0:
            last_zero_m = i
        if sp > h or sm > h:
            boundary = last_zero_p + 1 if sp > h else last_zero_m + 1
            boundary = min(max(boundary, 1), n - 1)
            if not boundaries or boundary > boundaries[-1]:
                boundaries.append(boundary)
            sp = sm = 0.0
            last_zero_p = last_zero_m = i
    return boundaries


def segment_intervals(
    depth: np.ndarray,
    rop: np.ndarray,
    torque: Optional[np.ndarray] = None,
    rpm: Optional[np.ndarray] = None,
    vibration: Optional[np.ndarray] = None,
    slack_factor: float = 0.5,
    threshold_factor: float = 4.0,
) -> List[Dict[str, Any]]:
    """Segment a drill trace into ROP regimes; returns the interval table."""
    depth = np.asarray(depth, dtype=float)
    rop = np.asarray(rop, dtype=float)
    n = len(depth)
    if n == 0:
        raise ValueError("empty series")
    if len(rop) != n:
        raise ValueError("depth and rop lengths differ")

    # Order by depth (drilling deepens monotonically).
    order = np.argsort(depth, kind="stable")
    depth = depth[order]
    rop = rop[order]

    def _opt(arr):
        return None if arr is None else np.asarray(arr, dtype=float)[order]

    torque = _opt(torque)
    rpm = _opt(rpm)
    vibration = _opt(vibration)

    bounds = [0] + cusum_change_points(
        rop, slack_factor, threshold_factor) + [n]

    intervals: List[Dict[str, Any]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        intervals.append({
            "i0": int(a),
            "i1": int(b - 1),
            "start_depth": float(depth[a]),
            "end_depth": float(depth[b - 1]),
            "n_points": int(b - a),
            "mean_rop": float(rop[a:b].mean()),
            "mean_torque": (float(torque[a:b].mean())
                            if torque is not None else None),
            "mean_rpm": float(rpm[a:b].mean()) if rpm is not None else None,
            "mean_vibration": (float(vibration[a:b].mean())
                               if vibration is not None else None),
        })

    # Regime labels by mean-ROP rank (fewest = slowest).
    labels = ["slow", "medium", "fast"]
    ranked = sorted(range(len(intervals)),
                    key=lambda i: intervals[i]["mean_rop"])
    for rank, idx in enumerate(ranked):
        # spread ranks over available labels
        lab_idx = round(rank * (len(labels) - 1) / max(1, len(ranked) - 1))
        intervals[idx]["regime"] = labels[lab_idx]
    return intervals


def align_to_collar(
    intervals: Sequence[Dict[str, Any]],
    collar_depth: float = 0.0,
    planned_total_depth: Optional[float] = None,
    deviation_tolerance: float = 0.5,
    final_measured_depth: Optional[float] = None,
) -> Dict[str, Any]:
    """Align interval depths to the collar datum; MWD-vs-collar deviation."""
    aligned = []
    for iv in intervals:
        iv = dict(iv)
        iv["aligned_start_depth"] = iv["start_depth"] - collar_depth
        iv["aligned_end_depth"] = iv["end_depth"] - collar_depth
        aligned.append(iv)

    deviation = None
    deviation_flag = False
    if planned_total_depth is not None and final_measured_depth is not None:
        aligned_td = final_measured_depth - collar_depth
        deviation = aligned_td - planned_total_depth
        deviation_flag = bool(abs(deviation) > deviation_tolerance)

    return {
        "collar_depth": float(collar_depth),
        "planned_total_depth": planned_total_depth,
        "deviation": deviation,
        "deviation_tolerance": float(deviation_tolerance),
        "deviation_flag": deviation_flag,
        "intervals": aligned,
    }
