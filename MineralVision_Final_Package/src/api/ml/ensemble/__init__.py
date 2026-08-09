"""
Ensemble Methods for MineralVision

Model averaging, stacking, boosting, and deep ensembles.
"""

from .ensemble_methods import (
    EnsembleMethod,
    EnsembleConfig,
    EnsembleFactory,
    ModelAveraging,
    StackingEnsemble,
    BaggingEnsemble,
    SnapshotEnsemble,
    DeepEnsemble,
    GradientBoostingEnsemble,
    create_model_averaging,
    create_stacking_ensemble,
    create_deep_ensemble,
)

__all__ = [
    "EnsembleMethod",
    "EnsembleConfig",
    "EnsembleFactory",
    "ModelAveraging",
    "StackingEnsemble",
    "BaggingEnsemble",
    "SnapshotEnsemble",
    "DeepEnsemble",
    "GradientBoostingEnsemble",
    "create_model_averaging",
    "create_stacking_ensemble",
    "create_deep_ensemble",
]
