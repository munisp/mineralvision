"""
Multi-criteria exploration portfolio optimization.

- AHP weighted scoring with Saaty consistency-ratio validation (CR < 0.1).
- Pareto frontier via non-dominated sorting (value vs risk).
- Budget-constrained selection: greedy by value/cost ratio + local swap
  improvement (remove one, greedy refill; 1-for-1 swaps are a special case).

All math is exact numpy; no external decision-library dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Saaty random consistency index (RI) for n = 1..15.
SAATY_RI: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
    8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57,
    15: 1.59,
}

CR_THRESHOLD = 0.1


class InconsistentMatrixError(ValueError):
    """Raised when a pairwise-comparison matrix fails the CR < 0.1 check."""


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------

def ahp_weights(matrix: Sequence[Sequence[float]],
                cr_threshold: float = CR_THRESHOLD) -> Dict[str, Any]:
    """AHP principal-eigenvector weights with Saaty consistency ratio.

    Raises :class:`InconsistentMatrixError` when CR >= ``cr_threshold``.
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("comparison matrix must be square")
    n = A.shape[0]
    if n < 2:
        raise ValueError("need at least 2 criteria")
    if not np.all(A > 0):
        raise ValueError("comparison matrix entries must be positive")
    if not np.allclose(A, 1.0 / A.T, rtol=1e-6, atol=1e-9):
        raise ValueError("matrix must be reciprocal: a_ij == 1 / a_ji")

    eigvals, eigvecs = np.linalg.eig(A)
    k = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals.real[k])
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()

    ci = (lambda_max - n) / (n - 1)
    ri = SAATY_RI.get(n, 1.59)
    cr = 0.0 if ri == 0.0 else ci / ri
    result = {
        "weights": w,
        "lambda_max": lambda_max,
        "ci": float(ci),
        "cr": float(cr),
        "consistent": bool(cr < cr_threshold),
    }
    if cr >= cr_threshold:
        raise InconsistentMatrixError(
            f"Saaty consistency ratio {cr:.4f} >= {cr_threshold}: "
            "the pairwise comparisons are too inconsistent to use")
    return result


def weighted_scores(criteria_scores: Sequence[Sequence[float]],
                    weights: Sequence[float]) -> np.ndarray:
    """Row-weighted AHP scores: projects x criteria matrix @ weights."""
    S = np.asarray(criteria_scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if S.ndim != 2 or S.shape[1] != len(w):
        raise ValueError("criteria_scores must be (n_projects, n_criteria) "
                         "matching weights length")
    return S @ w


# ---------------------------------------------------------------------------
# Pareto frontier (non-dominated sort)
# ---------------------------------------------------------------------------

def _dominates(p: np.ndarray, q: np.ndarray, maximize: np.ndarray) -> bool:
    """p dominates q: p no worse on every objective and strictly better on one.

    ``maximize`` is a boolean mask: True => higher is better.
    """
    better_or_equal = np.where(maximize, p >= q, p <= q)
    strictly_better = np.where(maximize, p > q, p < q)
    return bool(np.all(better_or_equal) and np.any(strictly_better))


def non_dominated_sort(points: Sequence[Sequence[float]],
                       senses: Sequence[str]) -> List[List[int]]:
    """Non-dominated sort; returns fronts as lists of row indices (front 0 =
    the Pareto frontier). ``senses`` entries are "max" or "min"."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2:
        raise ValueError("points must be a 2D array")
    if len(senses) != P.shape[1]:
        raise ValueError("senses length must match number of objectives")
    maximize = np.array([s == "max" for s in senses])
    if not np.all(np.isin(list(senses), ["max", "min"])):
        raise ValueError('senses must be "max" or "min"')

    n = len(P)
    dominates_list: List[set] = [set() for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(P[i], P[j], maximize):
                dominates_list[i].add(j)
            elif _dominates(P[j], P[i], maximize):
                domination_count[i] += 1

    fronts: List[List[int]] = []
    current = [i for i in range(n) if domination_count[i] == 0]
    assigned = np.zeros(n, dtype=bool)
    while current:
        fronts.append(sorted(current))
        nxt: set = set()
        for i in current:
            assigned[i] = True
            for j in dominates_list[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0 and not assigned[j]:
                    nxt.add(j)
        current = sorted(nxt)
    return fronts


def pareto_frontier(points: Sequence[Sequence[float]],
                    senses: Sequence[str]) -> List[int]:
    """Indices of the first (non-dominated) front."""
    return non_dominated_sort(points, senses)[0]


# ---------------------------------------------------------------------------
# Budget-constrained selection
# ---------------------------------------------------------------------------

def _total(values: np.ndarray, idxs: Sequence[int]) -> float:
    return float(values[list(idxs)].sum()) if idxs else 0.0


def _greedy(costs: np.ndarray, values: np.ndarray, budget: float,
            forced: Tuple[int, ...] = (),
            excluded: Tuple[int, ...] = ()) -> List[int]:
    """Ratio-greedy fill given already-forced picks and banned items."""
    selected = list(forced)
    spent = float(costs[list(forced)].sum()) if forced else 0.0
    remaining = [j for j in range(len(costs))
                 if j not in selected and j not in excluded]
    remaining.sort(key=lambda j: (-values[j] / costs[j], j))
    for j in remaining:
        if spent + costs[j] <= budget + 1e-12:
            selected.append(j)
            spent += float(costs[j])
    return sorted(selected)


def budget_select(costs: Sequence[float], values: Sequence[float],
                  budget: float) -> Dict[str, Any]:
    """Greedy by value/cost ratio + local swap improvement.

    Local swap: for each selected item, remove it and greedily refill the
    freed budget from unselected items (covers 1-for-1 and 1-for-many swaps);
    accept the best strict improvement; repeat until no improving move.
    """
    costs = np.asarray(costs, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(costs) != len(values):
        raise ValueError("costs and values length mismatch")
    if np.any(costs <= 0):
        raise ValueError("costs must be positive")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    selected = _greedy(costs, values, budget)
    greedy_value = _total(values, selected)

    improved = True
    while improved:
        improved = False
        best_sel, best_val = selected, _total(values, selected)
        for i in list(selected):
            rest = [x for x in selected if x != i]
            # the removed item is banned from the refill: a true swap
            cand = _greedy(costs, values, budget, forced=tuple(rest),
                           excluded=(i,))
            val = _total(values, cand)
            if val > best_val + 1e-12:
                best_sel, best_val = cand, val
        if best_val > _total(values, selected) + 1e-12:
            selected, improved = best_sel, True

    total_cost = float(costs[selected].sum()) if selected else 0.0
    return {
        "selected": [int(i) for i in selected],
        "total_value": _total(values, selected),
        "total_cost": total_cost,
        "budget": float(budget),
        "greedy_value": greedy_value,
        "swap_gain": _total(values, selected) - greedy_value,
    }
