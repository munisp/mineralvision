"""Tests for target_ranking — seeded ranking + permutation explainability."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'MineralVision_Final_Package', 'src'))

from api.innovations.target_ranking.logic import (
    FEATURE_NAMES, extract_target_features, rank_targets,
    permutation_importance, sequential_permutation_drops, explain_target,
    labels_from_grade_hits,
)


def make_dataset(seed=7, n=200, n_noise=4):
    """Two informative features (max_grade_gpt, grade_thickness) + noise."""
    rng = np.random.default_rng(seed)
    p = len(FEATURE_NAMES) + n_noise
    X = rng.normal(0.0, 1.0, size=(n, p))
    # informative: columns 1 (max grade) and 4 (grade_thickness)
    logit = 2.5 * X[:, 1] + 2.0 * X[:, 4]
    y = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
    names = FEATURE_NAMES + [f"noise_{i}" for i in range(n_noise)]
    return X, y, names


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_target_features_real_numbers():
    feats = extract_target_features(
        collar_x=100.0, collar_y=200.0, collar_z=350.0, total_depth=150.0,
        intervals=[(0.0, 50.0, 0.2), (50.0, 60.0, 3.0), (60.0, 100.0, None),
                   (100.0, 120.0, 5.0)],
        cutoff_gpt=1.0,
    )
    d = dict(zip(FEATURE_NAMES, feats))
    assert d["n_samples"] == 3.0
    assert d["max_grade_gpt"] == pytest.approx(5.0)
    assert d["mean_grade_gpt"] == pytest.approx((0.2 + 3.0 + 5.0) / 3)
    assert d["grade_thickness"] == pytest.approx(0.2 * 50 + 3.0 * 10 + 5.0 * 20)
    assert d["thickness_above_cutoff"] == pytest.approx(30.0)
    assert d["depth_to_first_hit"] == pytest.approx(50.0)
    assert d["collar_x"] == 100.0 and d["collar_z"] == 350.0


def test_extract_features_no_hits():
    feats = extract_target_features(0, 0, 0, 100.0,
                                    [(0.0, 50.0, 0.1), (50.0, 100.0, 0.3)],
                                    cutoff_gpt=1.0)
    d = dict(zip(FEATURE_NAMES, feats))
    assert d["thickness_above_cutoff"] == 0.0
    assert d["depth_to_first_hit"] == pytest.approx(100.0)


def test_labels_from_grade_hits():
    X = np.zeros((3, len(FEATURE_NAMES)))
    X[0, FEATURE_NAMES.index("thickness_above_cutoff")] = 5.0
    X[1, FEATURE_NAMES.index("thickness_above_cutoff")] = 0.5
    y = labels_from_grade_hits(X, min_thickness=1.0)
    assert list(y) == [1, 0, 0]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_planted_signal_ranks_top():
    # separable dataset: label is a deterministic function of the two
    # informative features; noise columns constant so the learner cannot
    # memorize residual noise (which would make in-sample scores arbitrary).
    rng = np.random.default_rng(7)
    n, p = 200, len(FEATURE_NAMES) + 4
    X = np.zeros((n, p))
    X[:, 1] = rng.normal(0, 1, n)
    X[:, 4] = rng.normal(0, 1, n)
    y = (2.5 * X[:, 1] + 2.0 * X[:, 4] > 0).astype(int)
    # plant an unambiguous top target: extreme informative feature values
    X[0, 1] = 6.0
    X[0, 4] = 6.0
    y[0] = 1
    for model_type in ("gradient_boosting", "random_forest"):
        result = rank_targets(X, y, seed=42, model_type=model_type)
        assert result.order[0] == 0, model_type
        assert result.scores[0] == pytest.approx(result.scores.max())
        assert result.baseline_score > 0.99  # dataset is perfectly learnable


def test_ranking_stable_with_seed():
    X, y, _ = make_dataset()
    r1 = rank_targets(X, y, seed=123, model_type="random_forest")
    r2 = rank_targets(X, y, seed=123, model_type="random_forest")
    assert np.array_equal(r1.order, r2.order)
    assert np.allclose(r1.scores, r2.scores)


def test_ranking_rejects_single_class():
    X, y, _ = make_dataset()
    with pytest.raises(ValueError):
        rank_targets(X, np.zeros(len(y), dtype=int))


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def test_importance_identifies_informative_features():
    X, y, names = make_dataset()
    result = rank_targets(X, y, seed=42)
    imp = permutation_importance(result.model, X, y, seed=42, n_repeats=8)
    order = np.argsort(-imp["importances"])
    top2 = set(order[:2].tolist())
    assert top2 == {1, 4}  # the two informative columns
    assert imp["baseline"] == pytest.approx(result.baseline_score)
    assert imp["importances"][1] > 0.05


def test_importance_sums_match_total_score_drop():
    """Sequential permutation drops telescope: sum == baseline - final."""
    X, y, names = make_dataset()
    result = rank_targets(X, y, seed=42)
    drops, baseline, final = sequential_permutation_drops(result.model, X, y,
                                                          seed=42)
    assert drops.sum() == pytest.approx(baseline - final, abs=1e-9)
    # and the sequential drops are consistent with standalone importance:
    # informative features contribute the bulk of the total drop
    total = drops.sum()
    if abs(total) > 1e-9:
        informative_share = (drops[1] + drops[4]) / total
        assert informative_share > 0.5


def test_permutation_importance_deterministic():
    X, y, _ = make_dataset()
    result = rank_targets(X, y, seed=42)
    a = permutation_importance(result.model, X, y, seed=9, n_repeats=5)
    b = permutation_importance(result.model, X, y, seed=9, n_repeats=5)
    assert np.allclose(a["importances"], b["importances"])


# ---------------------------------------------------------------------------
# Per-target explanation
# ---------------------------------------------------------------------------

def test_explain_target_top_driver_and_direction():
    X, y, names = make_dataset()
    # plant target with extreme max_grade (feature 1)
    X[5, 1] = 5.0
    X[5, 4] = 0.0
    y[5] = 1
    result = rank_targets(X, y, seed=42)
    out = explain_target(result.model, X, 5, feature_names=names,
                         seed=42, top_k=3)
    assert out["target_index"] == 5
    assert 0.0 <= out["base_score"] <= 1.0
    assert len(out["drivers"]) == 3
    # for a target whose edge is a huge max_grade, shuffling that column
    # (which replaces 5.0 with a typical value) must DROP its score
    driver = out["drivers"][0]
    assert driver["feature"] == "max_grade_gpt"
    assert driver["importance"] > 0.0
    assert driver["direction"] == "positive"


def test_explain_target_deterministic():
    X, y, names = make_dataset()
    result = rank_targets(X, y, seed=42)
    a = explain_target(result.model, X, 3, feature_names=names, seed=1)
    b = explain_target(result.model, X, 3, feature_names=names, seed=1)
    assert a == b


# ---------------------------------------------------------------------------
# Router smoke test
# ---------------------------------------------------------------------------

def test_router_rank_endpoint():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.innovations.target_ranking import router

    X, y, names = make_dataset(n=200)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.post("/innovations/target_ranking/rank", json={
        "features": X.tolist(), "labels": y.tolist(),
        "feature_names": names, "seed": 42, "n_repeats": 6,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["ranking"]) == 200
    assert body["baseline_score"] > 0.8
    top_feats = {f["feature"] for f in body["feature_importance"][:2]}
    assert top_feats == {"max_grade_gpt", "grade_thickness"}
