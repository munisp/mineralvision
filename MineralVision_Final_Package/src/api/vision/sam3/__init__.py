"""
SAM3 Integration for MineralVision

Meta's Segment Anything Model 3 integration for geology, mining,
geospatial, and geophysics applications.

Modules:
- sam3_segmenter: Core segmentation functionality
- fine_tuning: Domain-specific fine-tuning pipeline
- data_preparation: Training data preparation utilities
- model_registry: Model and adapter management
- api_endpoints: FastAPI endpoints
- automated_pipeline: Fully automated inference/training pipeline
- temporal_workflows: Workflow orchestration with Temporal
- weak_supervision: Pseudo-label generation from existing data

Usage:
    from mineralvision.vision.sam3 import create_sam3_segmenter
    
    segmenter = create_sam3_segmenter(modality="drillcore")
    result = segmenter.segment_by_text(image, "quartz vein")
    
    # Automated pipeline (no human in loop)
    from mineralvision.vision.sam3 import create_automated_system
    system = create_automated_system()
    system.start()
    result = system.process_image("core.jpg", DataSource.CORE_PHOTO, "gold_default")
"""

from .sam3_segmenter import (
    SAM3Segmenter,
    SAM3VideoTracker,
    create_sam3_segmenter,
    Modality,
    GeologyConcept,
    SegmentationResult,
    GEOLOGY_CONCEPTS,
)

from .fine_tuning import (
    SAM3FineTuner,
    TrainingConfig,
    TrainingStrategy,
    DataAugmentation,
    GeologyDatasetConfig,
    GeologySegmentationDataset,
    create_training_config,
)

from .data_preparation import (
    GeologyDataConverter,
    GeologyDataAugmenter,
    InteractiveLabelingSession,
    DatasetVersionManager,
    LabeledSample,
    DatasetManifest,
    prepare_training_data,
)

from .model_registry import (
    SAM3ModelRegistry,
    ABTestManager,
    AdapterMetadata,
    ModelStatus,
    AdapterType,
    create_model_registry,
)

from .automated_pipeline import (
    AutomatedSAM3System,
    AutomatedInferencePipeline,
    DriftDetector,
    SelfTrainingPipeline,
    QualityGate,
    PromptPackRegistry,
    PromptPack,
    InferenceEvent,
    InferenceResult,
    DataSource,
    create_automated_system,
)

from .temporal_workflows import (
    SAM3WorkflowOrchestrator,
    ScheduledWorkflowManager,
    InferenceWorkflowInput,
    TrainingWorkflowInput,
    PromotionWorkflowInput,
    WorkflowResult,
    WorkflowStatus,
    create_workflow_orchestrator,
    create_scheduled_manager,
)

from .weak_supervision import (
    WeakSupervisionPipeline,
    GeophysicsWeakSupervision,
    IntervalLogWeakSupervision,
    SoilHorizonWeakSupervision,
    SpectralIndexWeakSupervision,
    PseudoLabel,
    SupervisionSource,
    WeakSupervisionConfig,
    create_weak_supervision_pipeline,
)

__all__ = [
    # Segmenter
    "SAM3Segmenter",
    "SAM3VideoTracker",
    "create_sam3_segmenter",
    "Modality",
    "GeologyConcept",
    "SegmentationResult",
    "GEOLOGY_CONCEPTS",
    # Fine-tuning
    "SAM3FineTuner",
    "TrainingConfig",
    "TrainingStrategy",
    "DataAugmentation",
    "GeologyDatasetConfig",
    "GeologySegmentationDataset",
    "create_training_config",
    # Data preparation
    "GeologyDataConverter",
    "GeologyDataAugmenter",
    "InteractiveLabelingSession",
    "DatasetVersionManager",
    "LabeledSample",
    "DatasetManifest",
    "prepare_training_data",
    # Model registry
    "SAM3ModelRegistry",
    "ABTestManager",
    "AdapterMetadata",
    "ModelStatus",
    "AdapterType",
    "create_model_registry",
    # Automated pipeline
    "AutomatedSAM3System",
    "AutomatedInferencePipeline",
    "DriftDetector",
    "SelfTrainingPipeline",
    "QualityGate",
    "PromptPackRegistry",
    "PromptPack",
    "InferenceEvent",
    "InferenceResult",
    "DataSource",
    "create_automated_system",
    # Temporal workflows
    "SAM3WorkflowOrchestrator",
    "ScheduledWorkflowManager",
    "InferenceWorkflowInput",
    "TrainingWorkflowInput",
    "PromotionWorkflowInput",
    "WorkflowResult",
    "WorkflowStatus",
    "create_workflow_orchestrator",
    "create_scheduled_manager",
    # Weak supervision
    "WeakSupervisionPipeline",
    "GeophysicsWeakSupervision",
    "IntervalLogWeakSupervision",
    "SoilHorizonWeakSupervision",
    "SpectralIndexWeakSupervision",
    "PseudoLabel",
    "SupervisionSource",
    "WeakSupervisionConfig",
    "create_weak_supervision_pipeline",
]

__version__ = "1.1.0"
