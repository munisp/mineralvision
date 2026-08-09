"""Tests for portfolio_optimizer — AHP/CR, Pareto frontier, budget selection."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'MineralVision_Final_Package', 'src'))

from api.innovations.portfolio_optimizer.logic import (
    InconsistentMatrixError,
    ahp_weights,
    budget_select,
    non_dominated_sort,
    pareto_frontier,
    weighted_scores,
)

# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------

def test_ahp_perfectly_consistent_matrix():
    # w = (4/7, 2/7, 1/7) generates exactly this reciprocal matrix
    mat = [[1, 2, 4], [0.5, 1, 2], [0.25, 0.5, 1]]
    res = ahp_weights(mat)
    assert res["consistent"] is True
    assert res["cr"] == pytest.approx(0.0, abs=1e-9)
    assert res["lambda_max"] == pytest.approx(3.0, abs=1e-9)
    assert res["weights"] == pytest.approx([4 / 7, 2 / 7, 1 / 7], abs=1e-6)


def test_ahp_slightly_inconsistent_accepted():
    # classic Saaty example with small inconsistency
    mat = [[1, 5, 3], [1 / 5, 1, 1 / 3], [1 / 3, 3, 1]]
    res = ahp_weights(mat)
    assert res["consistent"] is True
    assert 0.0 < res["cr"] < 0.1
    assert res["weights"].sum() == pytest.approx(1.0)


def test_ahp_inconsistent_matrix_rejected():
    # cyclic comparisons: mat>B, B>C, C>mat — CR ~ 0.43
    mat = [[1, 2, 0.5], [0.5, 1, 2], [2, 0.5, 1]]
    with pytest.raises(InconsistentMatrixError):
        ahp_weights(mat)


def test_ahp_rejects_non_reciprocal():
    with pytest.raises(ValueError):
        ahp_weights([[1, 2], [0.6, 1]])


def test_ahp_weights_sum_to_one_5x5():
    w_true = np.array([0.4, 0.2, 0.15, 0.15, 0.1])
    mat = w_true[:, None] / w_true[None, :]
    res = ahp_weights(mat)
    assert res["weights"] == pytest.approx(w_true, abs=1e-6)


def test_weighted_scores():
    score_mat = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    w = [0.75, 0.25]
    out = weighted_scores(score_mat, w)
    assert out == pytest.approx([0.75, 0.25, 0.5])


# ---------------------------------------------------------------------------
# Pareto frontier (value max vs risk min)
# ---------------------------------------------------------------------------

def test_frontier_known_set():
    # (value, risk): maximize value, minimize risk
    points = [
        [10.0, 0.2],   # 0 frontier
        [8.0, 0.1],    # 1 frontier
        [6.0, 0.5],    # 2 dominated by 0 and 1
        [4.0, 0.05],   # 3 frontier (lowest risk)
        [5.0, 0.6],    # 4 dominated
    ]
    frontier = pareto_frontier(points, ["max", "min"])
    assert frontier == [0, 1, 3]


def test_non_dominated_sort_layers():
    points = [
        [10.0, 0.2],   # front 0
        [8.0, 0.1],    # front 0
        [6.0, 0.5],    # front 1
        [4.0, 0.05],   # front 0
        [5.0, 0.6],    # front 2
    ]
    fronts = non_dominated_sort(points, ["max", "min"])
    assert fronts[0] == [0, 1, 3]
    assert fronts[1] == [2]
    assert fronts[2] == [4]
    assert sorted(i for f in fronts for i in f) == [0, 1, 2, 3, 4]


def test_frontier_all_equal_points():
    # identical points: no strict domination, all in frontier
    frontier = pareto_frontier([[1.0, 1.0], [1.0, 1.0]], ["max", "min"])
    assert frontier == [0, 1]


def test_frontier_single_line_tradeoff():
    # value rises with risk: every point trades value for risk, none dominated
    pts = [[float(v), v / 10.0] for v in range(1, 10)]
    frontier = pareto_frontier(pts, ["max", "min"])
    assert frontier == list(range(9))


# ---------------------------------------------------------------------------
# Budget-constrained selection
# ---------------------------------------------------------------------------

def test_budget_never_exceeded():
    rng = np.random.default_rng(11)
    costs = rng.uniform(0.5, 5.0, size=40)
    values = rng.uniform(1.0, 10.0, size=40)
    for budget in (3.0, 7.5, 15.0, 50.0):
        res = budget_select(costs, values, budget)
        assert res["total_cost"] <= budget + 1e-9
        assert res["total_value"] >= res["greedy_value"] - 1e-9


def test_swap_improves_over_greedy():
    # greedy takes the ratio winner (6.8/6 = 1.133) and then cannot fit
    # the 5+5 pair; swap must discover the better combination
    costs = [6.0, 5.0, 5.0]
    values = [6.8, 5.6, 5.5]
    budget = 10.0
    res = budget_select(costs, values, budget)
    assert res["greedy_value"] == pytest.approx(6.8)
    assert res["selected"] == [1, 2]
    assert res["total_value"] == pytest.approx(11.1)
    assert res["total_cost"] == pytest.approx(10.0)
    assert res["swap_gain"] == pytest.approx(11.1 - 6.8)


def test_greedy_already_optimal_no_gain():
    costs = [1.0, 2.0, 3.0]
    values = [10.0, 9.0, 1.0]
    res = budget_select(costs, values, 3.0)
    assert res["selected"] == [0, 1]
    assert res["swap_gain"] == pytest.approx(0.0, abs=1e-12)


def test_budget_select_exact_knapsack_case():
    # known optimum: items 0+3 (cost 9, value 16) beats greedy 0+1 (value 15)
    costs = [4.0, 5.0, 6.0, 5.0]
    values = [8.0, 7.0, 6.0, 8.0]
    res = budget_select(costs, values, 9.0)
    assert res["selected"] == [0, 3]
    assert res["total_value"] == pytest.approx(16.0)
    assert res["total_cost"] <= 9.0


# ---------------------------------------------------------------------------
# Router smoke tests
# ---------------------------------------------------------------------------

def _client():
    from api.innovations.portfolio_optimizer import router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_router_ahp_rejects_inconsistent():
    client = _client()
    r = client.post("/innovations/portfolio_optimizer/ahp/weights",
                    json={"matrix": [[1, 2, 0.5], [0.5, 1, 2], [2, 0.5, 1]]})
    assert r.status_code == 422
    assert "consistency ratio" in r.json()["detail"]


def test_router_optimize_end_to_end():
    client = _client()
    w = [0.4, 0.2, 0.15, 0.15, 0.1]
    comp = (np.array(w)[:, None] / np.array(w)[None, :]).tolist()
    body = {
        "comparison_matrix": comp,
        "budget": 10.0,
        "projects": [
            {"id": "P1", "criteria_scores": {"prospectivity": 0.9, "cost": 0.7,
             "jurisdiction_risk": 0.8, "esg": 0.6, "logistics": 0.9},
             "cost": 6.0, "expected_value": 10.0, "risk": 0.3},
            {"id": "P2", "criteria_scores": {"prospectivity": 0.8, "cost": 0.8,
             "jurisdiction_risk": 0.7, "esg": 0.7, "logistics": 0.8},
             "cost": 5.0, "expected_value": 8.0, "risk": 0.2},
            {"id": "P3", "criteria_scores": {"prospectivity": 0.7, "cost": 0.6,
             "jurisdiction_risk": 0.6, "esg": 0.8, "logistics": 0.7},
             "cost": 5.0, "expected_value": 7.0, "risk": 0.4},
        ],
    }
    r = client.post("/innovations/portfolio_optimizer/optimize", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["weights"]["prospectivity"] == pytest.approx(0.4, abs=1e-6)
    assert data["selection"]["total_cost"] <= 10.0 + 1e-9
    assert set(data["pareto_frontier"]) == {"P1", "P2"}  # P3 dominated by P2
    # P2+P3 (cost 10) beats P1 alone on weighted score
    assert set(data["selection"]["selected_ids"]) == {"P2", "P3"}
