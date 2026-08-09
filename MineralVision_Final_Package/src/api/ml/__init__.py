"""
MineralVision Machine Learning Module.

This module provides comprehensive ML capabilities for mineral exploration including:
- Prospectivity mapping with spatial cross-validation
- Feature generation from geospatial raster stacks
- NLP extraction from geological reports
- Dataset registry and benchmarking
- Spatially-aware evaluation metrics

Exports are loaded lazily (PEP 562) because most submodules require optional
heavy ML/geospatial dependencies (torch, xarray, ...). Importing this package
itself is always safe; accessing an export imports its submodule on demand.
"""

import importlib

# export name -> submodule
_EXPORTS = {
    # prospectivity_workflow
    "ValidationStrategy": "prospectivity_workflow",
    "ProspectivityModel": "prospectivity_workflow",
    "TrainingPoint": "prospectivity_workflow",
    "ProspectivityDataset": "prospectivity_workflow",
    "SpatialBufferCV": "prospectivity_workflow",
    "create_benchmark_dataset": "prospectivity_workflow",
    # gold_exploration
    "AlterationType": "gold_exploration",
    "RegolithType": "gold_exploration",
    "GoldOccurrence": "gold_exploration",
    "StructuralFeature": "gold_exploration",
    "AlterationIndices": "gold_exploration",
    "RegolithModel": "gold_exploration",
    "StructuralComplexity": "gold_exploration",
    "GoldDepositPriors": "gold_exploration",
    "create_synthetic_gold_dataset": "gold_exploration",
    # lithium_exploration
    "LithiumMineral": "lithium_exploration",
    "BrineType": "lithium_exploration",
    "WellData": "lithium_exploration",
    "PegmatiteSample": "lithium_exploration",
    "BrineChemistry": "lithium_exploration",
    "HydrogeologyModel": "lithium_exploration",
    "ClayLithiumAnalysis": "lithium_exploration",
    "LithiumDepositPriors": "lithium_exploration",
    "create_synthetic_lithium_dataset": "lithium_exploration",
    # uncover_ml
    "UncoverMLPipeline": "uncover_ml",
    "ProspectivityWorkflow": "uncover_ml",
    "FeatureEngineering": "uncover_ml",
    "ModelEnsemble": "uncover_ml",
    "UncertaintyQuantification": "uncover_ml",
    "RasterStack": "uncover_ml",
    "TrainingData": "uncover_ml",
    "PredictionResult": "uncover_ml",
    "FeatureScale": "uncover_ml",
    "AggregationType": "uncover_ml",
    "ModelType": "uncover_ml",
    # torchgeo_models
    "GeoFoundationModel": "torchgeo_models",
    "SatelliteImageProcessor": "torchgeo_models",
    "UAVImageProcessor": "torchgeo_models",
    "PretrainedBackbone": "torchgeo_models",
    "FoundationModelType": "torchgeo_models",
    "ImageryType": "torchgeo_models",
    "TaskType": "torchgeo_models",
    "ImageBatch": "torchgeo_models",
    "ModelOutput": "torchgeo_models",
    "create_geo_foundation_model": "torchgeo_models",
    # spatial_cv
    "SpatialCrossValidator": "spatial_cv",
    "BlockCV": "spatial_cv",
    "BufferedLeaveOneOut": "spatial_cv",
    "SpatialKFold": "spatial_cv",
    "CVStrategy": "spatial_cv",
    "CVFold": "spatial_cv",
    "CVResult": "spatial_cv",
    "validate_spatial_model": "spatial_cv",
    # uncertainty_quantification
    "UncertaintyType": "uncertainty_quantification",
    "ConfidenceLevel": "uncertainty_quantification",
    "SensitivityResult": "uncertainty_quantification",
    "CalibrationResult": "uncertainty_quantification",
    "MCDropoutEstimator": "uncertainty_quantification",
    "DeepEnsembleEstimator": "uncertainty_quantification",
    "QuantileRegressionEstimator": "uncertainty_quantification",
    "SobolSensitivityAnalyzer": "uncertainty_quantification",
    "CalibrationAssessor": "uncertainty_quantification",
    "UncertaintyPropagator": "uncertainty_quantification",
    "UncertaintyQuantificationPipeline": "uncertainty_quantification",
    "create_uq_pipeline": "uncertainty_quantification",
    "estimate_grid_uncertainty": "uncertainty_quantification",
    # mlops_hardening
    "ModelStage": "mlops_hardening",
    "DatasetType": "mlops_hardening",
    "DriftType": "mlops_hardening",
    "DatasetVersion": "mlops_hardening",
    "ExperimentRun": "mlops_hardening",
    "ModelCard": "mlops_hardening",
    "DriftAlert": "mlops_hardening",
    "DatasetVersionManager": "mlops_hardening",
    "ExperimentTracker": "mlops_hardening",
    "DriftMonitor": "mlops_hardening",
    "EvaluationSuite": "mlops_hardening",
    "MLOpsPipeline": "mlops_hardening",
    "create_mlops_pipeline": "mlops_hardening",
    "create_experiment_tracker": "mlops_hardening",
    "create_drift_monitor": "mlops_hardening",
    # foundation_models
    "DataModality": "foundation_models",
    "PretrainingTask": "foundation_models",
    "FineTuningStrategy": "foundation_models",
    "ModalityConfig": "foundation_models",
    "PretrainingConfig": "foundation_models",
    "FineTuningConfig": "foundation_models",
    "ModelCheckpoint": "foundation_models",
    "ModalityAdapter": "foundation_models",
    "MultispectralAdapter": "foundation_models",
    "GeophysicsAdapter": "foundation_models",
    "TextAdapter": "foundation_models",
    "PretrainingPipeline": "foundation_models",
    "FineTuningPipeline": "foundation_models",
    "GeoscienceFoundationModel": "foundation_models",
    "FoundationModelRegistry": "foundation_models",
    "create_foundation_model": "foundation_models",
    "create_pretraining_config": "foundation_models",
    "create_finetuning_config": "foundation_models",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Lazily import the submodule providing an export (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent accesses
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
