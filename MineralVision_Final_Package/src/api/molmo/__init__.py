"""
Molmo2 Integration Module for MineralVision.

Provides video understanding, pixel-level grounding, multi-image reasoning,
and object tracking capabilities using Allen Institute's Molmo2-8B model.

Key Features:
- Video and multi-image understanding
- Pixel-level pointing and grounding
- Object tracking across video frames
- WALDO fusion for artisanal mining detection
- Drone video analysis for exploration
- V-JEPA integration for anomaly detection
"""

from .molmo_integration import (
    Molmo2Config,
    Molmo2Client,
    Molmo2Backend,
    AnalysisType,
    VideoAnalysisResult,
    PointingResult,
    TrackingResult,
    BoundingBox,
    create_molmo_client,
    analyze_mining_site_image,
    detect_artisanal_mining,
    compare_site_changes,
)
from .video_understanding import (
    VideoUnderstandingPipeline,
    VideoUnderstandingResult,
    FrameAnalysis,
    TemporalEvent,
    EventType,
    SceneType,
)
from .waldo_molmo_fusion import (
    WALDOMolmoFusion,
    FusedDetectionResult,
    FusionAnalysisResult,
    WALDODetection,
    MolmoUnderstanding,
    DetectionSource,
    ActivityType,
)
from .drone_video_analysis import (
    DroneVideoAnalyzer,
    FlightAnalysisResult,
    SiteChangeDetection,
    GeologicalObservation,
    GeologicalFeature,
    DroneMetadata,
    GeoLocation,
    FlightPattern,
    TerrainType,
)
from .vjepa_molmo_integration import (
    VJEPAMolmoFusion,
    FusedVideoAnalysis,
    VJEPAEmbedding,
    AnomalyDetection,
    FusionStrategy,
)
from .optimization import (
    # Structured Output
    OutputSchema,
    StructuredOutput,
    StructuredOutputParser,
    # Prompt Templates
    PromptTemplate,
    PromptLibrary,
    DOMAIN_PROMPTS,
    # Fine-tuning
    LoRAConfig,
    QLoRAConfig,
    TrainingConfig,
    TrainingExample,
    FineTuningDataset,
    Molmo2FineTuner,
    # Multi-Adapter
    AdapterType,
    AdapterInfo,
    MultiAdapterManager,
)
from .ensemble_pipeline import (
    # Configuration
    EnsembleConfig,
    DetectorType,
    SegmenterType,
    EmbedderType,
    ReasonerType,
    # Results
    Detection,
    FrameResult,
    VideoResult,
    # Components
    YOLO11Detector,
    RFDETRDetector,
    EnsembleDetector,
    SAM3Segmenter,
    VJEPAEmbedder,
    Molmo2Reasoner,
    # Pipeline
    EnsemblePipeline,
    create_ensemble_pipeline,
    analyze_mining_video,
    analyze_geological_image,
)
from .evaluation import (
    # Configuration
    EvaluationConfig,
    # Ground Truth
    GroundTruthBox,
    GroundTruthFrame,
    GroundTruthVideo,
    GroundTruthSite,
    GroundTruthChange,
    # Predictions
    PredictionBox,
    PredictionFrame,
    PredictionSite,
    PredictionChange,
    # Metrics
    DetectionMetrics,
    SiteLevelMetrics,
    TrackingMetrics,
    ChangeDetectionMetrics,
    GeologicalMetrics,
    ComprehensiveMetrics,
    # Evaluators
    DetectionEvaluator,
    SiteLevelEvaluator,
    TrackingEvaluator,
    ChangeDetectionEvaluator,
    GeologicalEvaluator,
    ComprehensiveEvaluator,
    # Convenience
    evaluate_artisanal_mining,
    evaluate_geological_features,
    evaluate_change_detection,
    create_evaluation_report,
)

__all__ = [
    # Core
    "Molmo2Config",
    "Molmo2Client",
    "Molmo2Backend",
    "AnalysisType",
    "VideoAnalysisResult",
    "PointingResult",
    "TrackingResult",
    "BoundingBox",
    "create_molmo_client",
    "analyze_mining_site_image",
    "detect_artisanal_mining",
    "compare_site_changes",
    # Video Understanding
    "VideoUnderstandingPipeline",
    "VideoUnderstandingResult",
    "FrameAnalysis",
    "TemporalEvent",
    "EventType",
    "SceneType",
    # WALDO Fusion
    "WALDOMolmoFusion",
    "FusedDetectionResult",
    "FusionAnalysisResult",
    "WALDODetection",
    "MolmoUnderstanding",
    "DetectionSource",
    "ActivityType",
    # Drone Analysis
    "DroneVideoAnalyzer",
    "FlightAnalysisResult",
    "SiteChangeDetection",
    "GeologicalObservation",
    "GeologicalFeature",
    "DroneMetadata",
    "GeoLocation",
    "FlightPattern",
    "TerrainType",
    # V-JEPA Integration
    "VJEPAMolmoFusion",
    "FusedVideoAnalysis",
    "VJEPAEmbedding",
    "AnomalyDetection",
    "FusionStrategy",
    # Optimization - Structured Output
    "OutputSchema",
    "StructuredOutput",
    "StructuredOutputParser",
    # Optimization - Prompts
    "PromptTemplate",
    "PromptLibrary",
    "DOMAIN_PROMPTS",
    # Optimization - Fine-tuning
    "LoRAConfig",
    "QLoRAConfig",
    "TrainingConfig",
    "TrainingExample",
    "FineTuningDataset",
    "Molmo2FineTuner",
    # Optimization - Multi-Adapter
    "AdapterType",
    "AdapterInfo",
    "MultiAdapterManager",
    # Ensemble Pipeline
    "EnsembleConfig",
    "DetectorType",
    "SegmenterType",
    "EmbedderType",
    "ReasonerType",
    "Detection",
    "FrameResult",
    "VideoResult",
    "YOLO11Detector",
    "RFDETRDetector",
    "EnsembleDetector",
    "SAM3Segmenter",
    "VJEPAEmbedder",
    "Molmo2Reasoner",
    "EnsemblePipeline",
    "create_ensemble_pipeline",
    "analyze_mining_video",
    "analyze_geological_image",
    # Evaluation
    "EvaluationConfig",
    "GroundTruthBox",
    "GroundTruthFrame",
    "GroundTruthVideo",
    "GroundTruthSite",
    "GroundTruthChange",
    "PredictionBox",
    "PredictionFrame",
    "PredictionSite",
    "PredictionChange",
    "DetectionMetrics",
    "SiteLevelMetrics",
    "TrackingMetrics",
    "ChangeDetectionMetrics",
    "GeologicalMetrics",
    "ComprehensiveMetrics",
    "DetectionEvaluator",
    "SiteLevelEvaluator",
    "TrackingEvaluator",
    "ChangeDetectionEvaluator",
    "GeologicalEvaluator",
    "ComprehensiveEvaluator",
    "evaluate_artisanal_mining",
    "evaluate_geological_features",
    "evaluate_change_detection",
    "create_evaluation_report",
]
