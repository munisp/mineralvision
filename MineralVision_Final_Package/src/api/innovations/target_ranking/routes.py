"""HTTP layer for target ranking (thin — see logic.py)."""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database import get_db, ProjectModel, DrillholeModel
from .logic import (
    FEATURE_NAMES, rank_targets, permutation_importance, explain_target,
    features_from_drillholes, labels_from_grade_hits,
)

router = APIRouter(prefix="/innovations/target_ranking", tags=["target_ranking"])


class RankRequest(BaseModel):
    features: List[List[float]] = Field(..., description="feature matrix rows")
    labels: List[int]
    feature_names: Optional[List[str]] = None
    model_type: str = "gradient_boosting"
    scoring: str = "roc_auc"
    seed: int = 42
    n_repeats: int = 10


class ExplainRequest(RankRequest):
    target_index: int = 0
    top_k: int = 5


class RankProjectRequest(BaseModel):
    project_id: str
    commodity: str = "Au"
    cutoff_gpt: float = 1.0
    min_thickness: float = 1.0
    model_type: str = "gradient_boosting"
    seed: int = 42


def _fit(req: RankRequest):
    X = np.asarray(req.features, dtype=float)
    y = np.asarray(req.labels, dtype=int)
    names = req.feature_names or [f"f{i}" for i in range(X.shape[1])]
    try:
        result = rank_targets(X, y, seed=req.seed, model_type=req.model_type,
                              scoring=req.scoring, feature_names=names)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return X, y, names, result, result.model


@router.post("/rank")
def rank(req: RankRequest):
    X, y, names, result, model = _fit(req)
    imp = permutation_importance(model, X, y, seed=req.seed,
                                 n_repeats=req.n_repeats, scoring=req.scoring)
    order = np.argsort(-imp["importances"])
    return {
        "ranking": [int(i) for i in result.order],
        "scores": [float(result.scores[i]) for i in result.order],
        "baseline_score": result.baseline_score,
        "feature_importance": [
            {"feature": names[j], "importance": float(imp["importances"][j]),
             "std": float(imp["stds"][j])}
            for j in order
        ],
        "model_type": result.model_type,
        "seed": req.seed,
    }


@router.post("/explain")
def explain(req: ExplainRequest):
    X, y, names, result, model = _fit(req)
    if not (0 <= req.target_index < len(X)):
        raise HTTPException(status_code=422, detail="target_index out of range")
    out = explain_target(model, X, req.target_index, feature_names=names,
                         seed=req.seed, top_k=req.top_k)
    out["rank"] = int(np.where(result.order == req.target_index)[0][0]) + 1
    return out


@router.post("/rank-project")
def rank_project(req: RankProjectRequest, db: Session = Depends(get_db)):
    project = db.query(ProjectModel).filter(
        ProjectModel.id == req.project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    holes = db.query(DrillholeModel).filter(
        DrillholeModel.project_id == project.id).all()
    if len(holes) < 4:
        raise HTTPException(status_code=422,
                            detail="need at least 4 drillholes to rank")
    X, ids = features_from_drillholes(holes, req.commodity, req.cutoff_gpt)
    y = labels_from_grade_hits(X, req.min_thickness)
    if len(np.unique(y)) < 2:
        raise HTTPException(status_code=422,
                            detail="labels are single-class for this cutoff")
    result = rank_targets(X, y, seed=req.seed, model_type=req.model_type)
    ranked = []
    for pos, idx in enumerate(result.order):
        exp = explain_target(result.model, X, int(idx), seed=req.seed, top_k=3)
        ranked.append({
            "rank": pos + 1,
            "hole_id": ids[int(idx)],
            "score": float(result.scores[int(idx)]),
            "top_drivers": exp["drivers"],
        })
    return {"project_id": project.id, "commodity": req.commodity,
            "baseline_score": result.baseline_score, "ranked": ranked}
