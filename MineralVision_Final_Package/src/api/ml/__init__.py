"""
MineralVision Machine Learning Module.

This module provides comprehensive ML capabilities for mineral exploration including:
- Prospectivity mapping with spatial cross-validation
- Feature generation from geospatial raster stacks
- NLP extraction from geological reports
- Dataset registry and benchmarking
- Spatially-aware evaluation metrics
"""

from .prospectivity_workflow import (
    # Enums
    FeatureType,
    ValidationStrategy,
    ProspectivityModel,
    
    # Data classes
    RasterLayer,
    TrainingPoint,
    ProspectivityDataset,
    
    # Cross-validation
    SpatialBlockCV,
    SpatialBufferCV,
    
    # Feature generation
    FeatureGenerator,
    
    # NLP
    GeologicalNLPExtractor,
    
    # Metrics
    SpatialMetrics,
    
    # Registry
    DatasetRegistry,
    
    # Pipeline
    ProspectivityPipeline,
    
    # Factory functions
    create_prospectivity_pipeline,
    create_benchmark_dataset
)

from .gold_exploration import (
    # Enums
    GoldDepositType,
    AlterationType,
    RegolithType,
    
    # Data classes
    GeochemSample,
    GoldOccurrence,
    StructuralFeature,
    
    # Analysis modules
    GoldPathfinderElements,
    AlterationIndices,
    RegolithModel,
    StructuralComplexity,
    GoldDepositPriors,
    
    # Pipeline
    GoldExplorationPipeline,
    
    # Factory functions
    create_gold_exploration_pipeline,
    create_synthetic_gold_dataset
)

from .lithium_exploration import (
    # Enums
    LithiumDepositType,
    LithiumMineral,
    BrineType,
    
    # Data classes
    BrineSample,
    WellData,
    PegmatiteSample,
    
    # Analysis modules
    LithiumPathfinderElements,
    BrineChemistry,
    HydrogeologyModel,
    ClayLithiumAnalysis,
    LithiumDepositPriors,
    
    # Pipeline
    LithiumExplorationPipeline,
    
    # Factory functions
    create_lithium_exploration_pipeline,
    create_synthetic_lithium_dataset
)

from .soil_suitability import (
    # Enums
    CropType,
    SuitabilityClass,
    SoilTextureClass,
    DrainageClass,
    
    # Data classes
    SoilSample,
    ClimateData,
    TopographyData,
    
    # Analysis modules
    CropRequirements,
    SoilSuitabilityScorer,
    RemediationRecommender,
    
    # Pipeline
    SoilSuitabilityPipeline,
    
    # Factory functions
    create_soil_suitability_pipeline,
    create_synthetic_soil_dataset
)

from .advanced_soil_assessment import (
    # Enums
    HazardSeverity,
    ToxicityType,
    DiseaseRiskType,
    PhosphorusMethod,
    
    # Data classes
    SoilPhysicalConstraints,
    ToxicityHazard,
    NutrientBudget,
    UncertaintyEstimate,
    WaterBalanceResult,
    DiseaseRiskAssessment,
    EconomicAnalysis,
    
    # Assessors
    ToxicityHazardAssessor,
    NutrientBudgetCalculator,
    PhysicalConstraintAssessor,
    UncertaintyQuantifier,
    WaterBalanceCalculator,
    DiseaseRiskAssessor,
    EconomicAnalyzer,
    SpatialInterpolator,
    
    # Advanced Pipeline
    AdvancedSoilSuitabilityPipeline,
    
    # Factory functions
    create_advanced_soil_pipeline
)

from .uncover_ml import (
    UncoverMLPipeline,
    ProspectivityWorkflow,
    FeatureEngineering,
    ModelEnsemble,
    UncertaintyQuantification,
    RasterStack,
    TrainingData,
    PredictionResult,
    FeatureScale,
    AggregationType,
    ModelType,
    create_prospectivity_pipeline as create_uncover_pipeline,
)

from .torchgeo_models import (
    GeoFoundationModel,
    SatelliteImageProcessor,
    UAVImageProcessor,
    PretrainedBackbone,
    FoundationModelType,
    ImageryType,
    TaskType,
    ImageBatch,
    ModelOutput,
    create_geo_foundation_model,
)

from .spatial_cv import (
    SpatialCrossValidator,
    BlockCV,
    BufferedLeaveOneOut,
    SpatialKFold,
    CVStrategy,
    CVFold,
    CVResult,
    validate_spatial_model,
)

from .uncertainty_quantification import (
    UncertaintyType,
    ConfidenceLevel,
    UncertaintyEstimate as UQEstimate,
    SensitivityResult,
    CalibrationResult,
    MCDropoutEstimator,
    DeepEnsembleEstimator,
    QuantileRegressionEstimator,
    SobolSensitivityAnalyzer,
    CalibrationAssessor,
    UncertaintyPropagator,
    UncertaintyQuantificationPipeline,
    create_uq_pipeline,
    estimate_grid_uncertainty,
)

from .mlops_hardening import (
    ModelStage,
    DatasetType,
    DriftType,
    DatasetVersion,
    ExperimentRun,
    ModelCard,
    DriftAlert,
    DatasetVersionManager,
    ExperimentTracker,
    DriftMonitor,
    EvaluationSuite,
    MLOpsPipeline,
    create_mlops_pipeline,
    create_experiment_tracker,
    create_drift_monitor,
)

from .foundation_models import (
    DataModality,
    PretrainingTask,
    FineTuningStrategy,
    ModalityConfig,
    PretrainingConfig,
    FineTuningConfig,
    ModelCheckpoint,
    ModalityAdapter,
    MultispectralAdapter,
    GeophysicsAdapter,
    TextAdapter,
    PretrainingPipeline,
    FineTuningPipeline,
    GeoscienceFoundationModel,
    FoundationModelRegistry,
    create_foundation_model,
    create_pretraining_config,
    create_finetuning_config,
)

__all__ = [
    # Enums
    'FeatureType',
    'ValidationStrategy',
    'ProspectivityModel',
    
    # Data classes
    'RasterLayer',
    'TrainingPoint',
    'ProspectivityDataset',
    
    # Cross-validation
    'SpatialBlockCV',
    'SpatialBufferCV',
    
    # Feature generation
    'FeatureGenerator',
    
    # NLP
    'GeologicalNLPExtractor',
    
    # Metrics
    'SpatialMetrics',
    
    # Registry
    'DatasetRegistry',
    
    # Pipeline
    'ProspectivityPipeline',
    
    # Factory functions
    'create_prospectivity_pipeline',
    'create_benchmark_dataset',
    
    # Gold Exploration
    'GoldDepositType',
    'AlterationType',
    'RegolithType',
    'GeochemSample',
    'GoldOccurrence',
    'StructuralFeature',
    'GoldPathfinderElements',
    'AlterationIndices',
    'RegolithModel',
    'StructuralComplexity',
    'GoldDepositPriors',
    'GoldExplorationPipeline',
    'create_gold_exploration_pipeline',
    'create_synthetic_gold_dataset',
    
    # Lithium Exploration
    'LithiumDepositType',
    'LithiumMineral',
    'BrineType',
    'BrineSample',
    'WellData',
    'PegmatiteSample',
    'LithiumPathfinderElements',
    'BrineChemistry',
    'HydrogeologyModel',
    'ClayLithiumAnalysis',
    'LithiumDepositPriors',
    'LithiumExplorationPipeline',
    'create_lithium_exploration_pipeline',
    'create_synthetic_lithium_dataset',
    
    # Soil Suitability
    'CropType',
    'SuitabilityClass',
    'SoilTextureClass',
    'DrainageClass',
    'SoilSample',
    'ClimateData',
    'TopographyData',
    'CropRequirements',
    'SoilSuitabilityScorer',
    'RemediationRecommender',
    'SoilSuitabilityPipeline',
    'create_soil_suitability_pipeline',
    'create_synthetic_soil_dataset',
    
    # Advanced Soil Assessment
    'HazardSeverity',
    'ToxicityType',
    'DiseaseRiskType',
    'PhosphorusMethod',
    'SoilPhysicalConstraints',
    'ToxicityHazard',
    'NutrientBudget',
    'UncertaintyEstimate',
    'WaterBalanceResult',
    'DiseaseRiskAssessment',
    'EconomicAnalysis',
    'ToxicityHazardAssessor',
    'NutrientBudgetCalculator',
    'PhysicalConstraintAssessor',
    'UncertaintyQuantifier',
    'WaterBalanceCalculator',
    'DiseaseRiskAssessor',
    'EconomicAnalyzer',
    'SpatialInterpolator',
    'AdvancedSoilSuitabilityPipeline',
    'create_advanced_soil_pipeline',
    
    # UNCOVER-ML Style Pipeline
    'UncoverMLPipeline',
    'ProspectivityWorkflow',
    'FeatureEngineering',
    'ModelEnsemble',
    'UncertaintyQuantification',
    'RasterStack',
    'TrainingData',
    'PredictionResult',
    'FeatureScale',
    'AggregationType',
    'ModelType',
    'create_uncover_pipeline',
    
    # TorchGeo Foundation Models
    'GeoFoundationModel',
    'SatelliteImageProcessor',
    'UAVImageProcessor',
    'PretrainedBackbone',
    'FoundationModelType',
    'ImageryType',
    'TaskType',
    'ImageBatch',
    'ModelOutput',
    'create_geo_foundation_model',
    
    # Spatial Cross-Validation
    'SpatialCrossValidator',
    'BlockCV',
    'BufferedLeaveOneOut',
    'SpatialKFold',
    'CVStrategy',
    'CVFold',
    'CVResult',
    'validate_spatial_model',
    
    # Uncertainty Quantification
    'UncertaintyType',
    'ConfidenceLevel',
    'UQEstimate',
    'SensitivityResult',
    'CalibrationResult',
    'MCDropoutEstimator',
    'DeepEnsembleEstimator',
    'QuantileRegressionEstimator',
    'SobolSensitivityAnalyzer',
    'CalibrationAssessor',
    'UncertaintyPropagator',
    'UncertaintyQuantificationPipeline',
    'create_uq_pipeline',
    'estimate_grid_uncertainty',
    
    # MLOps Hardening
    'ModelStage',
    'DatasetType',
    'DriftType',
    'DatasetVersion',
    'ExperimentRun',
    'ModelCard',
    'DriftAlert',
    'DatasetVersionManager',
    'ExperimentTracker',
    'DriftMonitor',
    'EvaluationSuite',
    'MLOpsPipeline',
    'create_mlops_pipeline',
    'create_experiment_tracker',
    'create_drift_monitor',
    
    # Foundation Models
    'DataModality',
    'PretrainingTask',
    'FineTuningStrategy',
    'ModalityConfig',
    'PretrainingConfig',
    'FineTuningConfig',
    'ModelCheckpoint',
    'ModalityAdapter',
    'MultispectralAdapter',
    'GeophysicsAdapter',
    'TextAdapter',
    'PretrainingPipeline',
    'FineTuningPipeline',
    'GeoscienceFoundationModel',
    'FoundationModelRegistry',
    'create_foundation_model',
    'create_pretraining_config',
    'create_finetuning_config',
]
