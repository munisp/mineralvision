"""
Model Zoo for MineralVision

Pre-trained models for mineral exploration tasks.
"""

from .pretrained_models import (
    ModelZoo,
    ModelTask,
    ModelArchitecture,
    ModelInfo,
    MineralClassificationModel,
    DepositPredictionModel,
    GeologicalFeatureDetector,
    AnomalyDetectionModel,
    SpectralAnalysisTransformer,
    list_pretrained_models,
    load_pretrained_model,
    get_best_model_for_task,
)

__all__ = [
    "ModelZoo",
    "ModelTask",
    "ModelArchitecture",
    "ModelInfo",
    "MineralClassificationModel",
    "DepositPredictionModel",
    "GeologicalFeatureDetector",
    "AnomalyDetectionModel",
    "SpectralAnalysisTransformer",
    "list_pretrained_models",
    "load_pretrained_model",
    "get_best_model_for_task",
]
