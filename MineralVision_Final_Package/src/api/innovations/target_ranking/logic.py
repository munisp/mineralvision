"""
Drill target ranking with explainability.

Ranking: scikit-learn GradientBoosting / RandomForest classifiers (seeded).
Explanation: permutation importance implemented directly (no shap) —
seeded column shuffles, score drop measured with ROC-AUC (or accuracy),
plus per-target (local) importance with direction.

Feature extraction is self-contained: the shared
``src/api/ml/prospectivity_workflow.py`` core requires ``xarray``, which is
not part of the sanctioned dependency set for this batch, so drillhole/assay
feature extraction is implemented here from SQLAlchemy rows or plain dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURE_NAMES: List[str] = [
    "n_samples",
    "max_grade_gpt",
    "mean_grade_gpt",
    "grade_std",
    "grade_thickness",        # sum(grade * interval length)
    "thickness_above_cutoff",
    "depth_to_first_hit",     # from_depth of first sample above cutoff
    "total_depth",
    "collar_x",
    "collar_y",
    "collar_z",
]


# ---------------------------------------------------------------------------
# Feature extraction (self-contained — see module docstring)
# ---------------------------------------------------------------------------

def extract_target_features(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    total_depth: float,
    intervals: Sequence[Tuple[float, float, Optional[float]]],
    cutoff_gpt: float = 1.0,
) -> np.ndarray:
    """Build the feature vector for one target.

    ``intervals`` is a sequence of (from_depth, to_depth, grade_gpt-or-None).
    """
    grades = np.array([g for _, _, g in intervals if g is not None],
                      dtype=float)
    n = float(len(grades))
    max_g = float(grades.max()) if n else 0.0
    mean_g = float(grades.mean()) if n else 0.0
    std_g = float(grades.std()) if n else 0.0

    gt = 0.0
    thick = 0.0
    first_hit = float(total_depth)
    for frm, to, g in intervals:
        if g is None:
            continue
        length = max(float(to) - float(frm), 0.0)
        gt += float(g) * length
        if g >= cutoff_gpt:
            thick += length
            first_hit = min(first_hit, float(frm))
    if thick == 0.0:
        first_hit = float(total_depth)

    return np.array([
        n, max_g, mean_g, std_g, gt, thick, first_hit,
        float(total_depth), float(collar_x), float(collar_y), float(collar_z),
    ], dtype=float)


def features_from_drillholes(holes, commodity: str,
                             cutoff_gpt: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """Extract features from DrillholeModel-like rows (``samples`` relation)."""
    rows, ids = [], []
    from ...database import SampleModel  # noqa: F401  (documents the contract)
    for h in holes:
        intervals = []
        for s in h.samples:
            data = s.assay_data or {}
            g = None
            for key in (commodity, commodity.capitalize(),
                        commodity.upper()):
                if key in data and data[key] is not None:
                    g = float(data[key])
                    break
            intervals.append((s.from_depth, s.to_depth, g))
        rows.append(extract_target_features(h.collar_x, h.collar_y, h.collar_z,
                                            h.total_depth, intervals,
                                            cutoff_gpt))
        ids.append(h.hole_id)
    X = np.vstack(rows) if rows else np.zeros((0, len(FEATURE_NAMES)))
    return X, ids


def labels_from_grade_hits(X: np.ndarray, min_thickness: float = 1.0) -> np.ndarray:
    """Label = 1 when the hole carries at least ``min_thickness`` m above cutoff."""
    return (X[:, FEATURE_NAMES.index("thickness_above_cutoff")] >= min_thickness
            ).astype(int)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def build_model(model_type: str = "gradient_boosting", seed: int = 42):
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=1)
    raise ValueError(f"unknown model_type: {model_type!r}")


def _scorer(name: str) -> Callable[[np.ndarray, np.ndarray, Any], float]:
    def score(y_true: np.ndarray, X: np.ndarray, model) -> float:
        if name == "roc_auc":
            proba = model.predict_proba(X)[:, 1]
            return float(roc_auc_score(y_true, proba))
        if name == "accuracy":
            return float(accuracy_score(y_true, model.predict(X)))
        raise ValueError(f"unknown scoring: {name!r}")
    return score


@dataclass
class RankingResult:
    scores: np.ndarray            # predicted P(mineralized) per target
    order: np.ndarray             # indices sorted best-first
    baseline_score: float
    scoring: str
    model_type: str
    seed: int
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    model: Any = None             # fitted sklearn model


def rank_targets(X: np.ndarray, y: np.ndarray, seed: int = 42,
                 model_type: str = "gradient_boosting",
                 scoring: str = "roc_auc",
                 feature_names: Optional[List[str]] = None) -> RankingResult:
    """Fit the model and rank targets by predicted probability (desc)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y shape mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError("labels must contain both classes")
    model = build_model(model_type, seed)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    # deterministic tie-break on index
    order = np.lexsort((np.arange(len(scores)), -scores))
    base = _scorer(scoring)(y, X, model)
    return RankingResult(scores=scores, order=order, baseline_score=base,
                         scoring=scoring, model_type=model_type, seed=seed,
                         feature_names=feature_names or list(FEATURE_NAMES),
                         model=model)


# ---------------------------------------------------------------------------
# Permutation importance (implemented directly — no shap)
# ---------------------------------------------------------------------------

def _permute(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.permutation(n)


def permutation_importance(model, X: np.ndarray, y: np.ndarray,
                           seed: int = 42, n_repeats: int = 10,
                           scoring: str = "roc_auc") -> Dict[str, np.ndarray]:
    """Global permutation importance: seeded per-column shuffle, score drop.

    Returns dict with baseline, importances (mean drop per feature) and stds.
    """
    X = np.asarray(X, dtype=float)
    score_fn = _scorer(scoring)
    baseline = score_fn(y, X, model)
    n_features = X.shape[1]
    drops = np.zeros((n_features, n_repeats))
    for j in range(n_features):
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, j, r])
            Xp = X.copy()
            Xp[:, j] = Xp[_permute(rng, len(Xp)), j]
            drops[j, r] = baseline - score_fn(y, Xp, model)
    return {
        "baseline": baseline,
        "importances": drops.mean(axis=1),
        "stds": drops.std(axis=1),
    }


def sequential_permutation_drops(model, X: np.ndarray, y: np.ndarray,
                                 seed: int = 42,
                                 scoring: str = "roc_auc"
                                 ) -> Tuple[np.ndarray, float, float]:
    """Cumulative column shuffling: shuffle col j on top of cols 0..j-1.

    The per-column drops telescope, so ``sum(drops) == baseline - final``
    exactly (up to float round-off) — this is the consistency identity used
    to validate the importance bookkeeping.
    """
    X = np.asarray(X, dtype=float)
    score_fn = _scorer(scoring)
    baseline = score_fn(y, X, model)
    Xp = X.copy()
    prev = baseline
    drops = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        rng = np.random.default_rng([seed, j])
        Xp[:, j] = X[rng.permutation(len(X)), j]
        cur = score_fn(y, Xp, model)
        drops[j] = prev - cur
        prev = cur
    return drops, baseline, prev


def explain_target(model, X: np.ndarray, target_index: int,
                   feature_names: Optional[List[str]] = None,
                   seed: int = 42, n_repeats: int = 30,
                   top_k: int = 5) -> Dict[str, Any]:
    """Per-target (local) permutation explanation.

    For each feature column, shuffle that column across the dataset and
    measure the change in *this target's* predicted probability. Direction is
    the sign of the correlation between the shuffled values and the target's
    predicted score: positive => higher values push this target's score up.
    """
    X = np.asarray(X, dtype=float)
    names = feature_names or list(FEATURE_NAMES)
    row = X[target_index:target_index + 1]
    base = float(model.predict_proba(row)[0, 1])
    n, p = X.shape
    drivers = []
    for j in range(p):
        deltas = np.zeros(n_repeats)
        corr_vals = np.zeros(n_repeats)
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, target_index, j, r])
            perm = rng.permutation(n)
            new_val = X[perm[0], j]
            row_p = row.copy()
            row_p[0, j] = new_val
            new_score = float(model.predict_proba(row_p)[0, 1])
            deltas[r] = base - new_score
            corr_vals[r] = new_val - X[:, j].mean()
        # direction: does raising the feature raise this target's score?
        direction = float(np.corrcoef(
            corr_vals, base - deltas)[0, 1]) if n_repeats > 1 else 0.0
        if not np.isfinite(direction):
            direction = 0.0
        drivers.append({
            "feature": names[j],
            "importance": float(deltas.mean()),
            "abs_importance": float(abs(deltas.mean())),
            "std": float(deltas.std()),
            "direction": "positive" if direction >= 0 else "negative",
            "direction_strength": direction,
        })
    drivers.sort(key=lambda d: d["abs_importance"], reverse=True)
    return {
        "target_index": int(target_index),
        "base_score": base,
        "drivers": drivers[:top_k],
    }
