"""HTTP layer for the portfolio optimizer (thin — see logic.py)."""

from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .logic import (
    InconsistentMatrixError, ahp_weights, weighted_scores,
    non_dominated_sort, budget_select,
)

router = APIRouter(prefix="/innovations/portfolio_optimizer",
                   tags=["portfolio_optimizer"])

CRITERIA = ["prospectivity", "cost", "jurisdiction_risk", "esg", "logistics"]


class AHPRequest(BaseModel):
    matrix: List[List[float]]
    cr_threshold: float = 0.1


class ParetoRequest(BaseModel):
    points: List[List[float]]
    senses: List[str] = Field(..., description='"max" or "min" per objective')


class SelectItem(BaseModel):
    id: str
    cost: float
    value: float


class SelectRequest(BaseModel):
    items: List[SelectItem]
    budget: float


class OptimizeProject(BaseModel):
    id: str
    criteria_scores: Dict[str, float]   # keys from CRITERIA, normalized 0..1
    cost: float
    expected_value: float
    risk: float


class OptimizeRequest(BaseModel):
    projects: List[OptimizeProject]
    comparison_matrix: List[List[float]]  # over CRITERIA (5x5)
    budget: float


def _ahp_or_422(matrix, threshold):
    try:
        return ahp_weights(matrix, threshold)
    except InconsistentMatrixError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/ahp/weights")
def ahp(req: AHPRequest):
    res = _ahp_or_422(req.matrix, req.cr_threshold)
    return {"weights": [float(x) for x in res["weights"]],
            "lambda_max": res["lambda_max"], "ci": res["ci"],
            "cr": res["cr"], "consistent": res["consistent"]}


@router.post("/pareto/frontier")
def pareto(req: ParetoRequest):
    try:
        fronts = non_dominated_sort(req.points, req.senses)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"fronts": fronts, "frontier": fronts[0] if fronts else []}


@router.post("/select")
def select(req: SelectRequest):
    costs = [it.cost for it in req.items]
    values = [it.value for it in req.items]
    try:
        res = budget_select(costs, values, req.budget)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "selected_ids": [req.items[i].id for i in res["selected"]],
        "selected_indices": res["selected"],
        "total_value": res["total_value"],
        "total_cost": res["total_cost"],
        "budget": res["budget"],
        "greedy_value": res["greedy_value"],
        "swap_gain": res["swap_gain"],
    }


@router.post("/optimize")
def optimize(req: OptimizeRequest):
    """Full pipeline: AHP weights -> weighted scores -> Pareto -> budget."""
    res = _ahp_or_422(req.comparison_matrix, 0.1)
    weights = res["weights"]
    S = np.array([[p.criteria_scores.get(c, 0.0) for c in CRITERIA]
                  for p in req.projects])
    scores = weighted_scores(S, weights)
    points = [[float(p.expected_value), float(p.risk)] for p in req.projects]
    fronts = non_dominated_sort(points, ["max", "min"])
    sel = budget_select([p.cost for p in req.projects],
                        [float(s) for s in scores], req.budget)
    return {
        "criteria": CRITERIA,
        "weights": {c: float(w) for c, w in zip(CRITERIA, weights)},
        "cr": res["cr"],
        "weighted_scores": {p.id: float(s) for p, s in zip(req.projects, scores)},
        "pareto_frontier": [req.projects[i].id for i in fronts[0]],
        "fronts": [[req.projects[i].id for i in f] for f in fronts],
        "selection": {
            "selected_ids": [req.projects[i].id for i in sel["selected"]],
            "total_value": sel["total_value"],
            "total_cost": sel["total_cost"],
            "budget": sel["budget"],
            "swap_gain": sel["swap_gain"],
        },
    }
