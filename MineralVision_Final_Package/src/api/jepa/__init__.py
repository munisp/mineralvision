"""
V-JEPA Integration Module for MineralVision.

Provides Video Joint-Embedding Predictive Architecture integration for:
- Domain-adaptive pretraining on mining imagery (drone, satellite, core photos)
- Feature extraction and embedding storage
- Anomaly detection and change detection
- Integration with WALDO detection and SAM3 segmentation

Backend honesty contract (post-decontamination):
- All model computations (encoding, prediction, pretraining) go through
  ``api.jepa.torch_core`` — a real PyTorch V-JEPA implementation.
- When torch_core or PyTorch is unavailable, model-facing entry points
  raise ``JEPAUnavailableError`` (or ``WaldoIntegrationUnavailable`` for
  WALDO/SAM3 paths). NO random embeddings, fake losses, fabricated
  detections, or synthetic masks are ever returned.
- WALDO detection / SAM3 segmentation additionally support real HTTP
  backends via the WALDO_SERVICE_URL / SAM3_SERVICE_URL environment
  variables (docker-compose services).
- The local lakehouse backend writes real Parquet when pyarrow is
  installed; otherwise it writes honestly labeled JSON files and reports
  ``backend == "json"``.

Usage:
    # Create feature extractor
    from api.jepa import create_feature_extractor
    extractor = create_feature_extractor(checkpoint_path="path/to/checkpoint")
    
    # Extract features
    embedding = extractor.extract_features(frames)
    
    # Create anomaly detector
    from api.jepa import create_anomaly_detector
    detector = create_anomaly_detector(extractor)
    detector.build_baseline(normal_samples)
    result = detector.detect(sample)
    
    # Create unified pipeline
    from api.jepa import create_unified_pipeline
    pipeline = create_unified_pipeline()
    results = pipeline.process_image(image, tasks=["embedding", "detection", "segmentation"])
    
    # Run pretraining
    from api.jepa import run_pretraining
    results = run_pretraining(
        job_name="mining_domain_pretraining",
        data_sources={
            "drone_video": "/path/to/drone/videos",
            "satellite_rgb": "/path/to/satellite/tiles",
            "core_photo": "/path/to/core/photos",
        },
        backbone="vit_large",
        num_epochs=100,
    )

Based on: https://github.com/facebookresearch/jepa
"""

from .vjepa_integration import (
    JEPAUnavailableError,
    VJEPAConfig,
    ImageryType,
    PretrainingMode,
    MaskingStrategy,
    BackboneSize,
    ImageryMetadata,
    Embedding,
    AnomalyResult,
    ChangeDetectionResult,
    VectorIndex,
    FaissIndex,
    MiningDataLoader,
    MultiScaleMasking,
    VJEPAEncoder,
    VJEPAPredictor,
    VJEPAPretrainer,
    VJEPAFeatureExtractor,
    AnomalyDetector,
    ChangeDetector,
    SimilaritySearch,
    create_vjepa_config,
    create_mining_data_loader,
    create_pretrainer,
    create_feature_extractor,
    create_anomaly_detector,
    create_change_detector,
    create_similarity_search,
)

from .pretraining_pipeline import (
    PretrainingJob,
    TilingConfig,
    ImageTiler,
    VideoChunker,
    CorePhotoProcessor,
    DatasetManifestBuilder,
    PretrainingRunner,
    prepare_mining_dataset,
    run_pretraining,
)

from .waldo_sam3_integration import (
    WaldoIntegrationUnavailable,
    IntegrationMode,
    DetectionTarget,
    SegmentationTarget,
    DetectionResult,
    SegmentationResult,
    JEPAPrompt,
    WALDOJEPAIntegration,
    SAM3JEPAIntegration,
    UnifiedMiningVisionPipeline,
    FeatureDistillation,
    create_waldo_jepa_integration,
    create_sam3_jepa_integration,
    create_unified_pipeline,
)

from .llm_integration import (
    LLMProvider,
    ExplanationType,
    DomainContext,
    RetrievedEvidence,
    StructuredExplanation,
    ChatMessage,
    ChatResponse,
    LLMClient,
    OllamaClient,
    MockLLMClient,
    EvidenceRetriever,
    ExplanationService,
    JEPAChat,
    JEPAOrchestrator,
    create_llm_client,
    create_explanation_service,
    create_jepa_chat,
    create_orchestrator,
    explain_jepa_finding,
)

from .lakehouse_integration import (
    TableFormat,
    StorageBackend,
    LakehouseConfig,
    EmbeddingRecord,
    FindingRecord,
    TrainingRunRecord,
    LakehouseBackend,
    LocalParquetBackend,
    LocalJSONBackend,
    DeltaLakeBackend,
    JEPALakehouseStore,
    TrainingDatasetManager,
    RAGRetriever,
    create_jepa_lakehouse_store,
    create_training_dataset_manager,
    create_rag_retriever,
)

from .continuous_training import (
    RetrainingTrigger,
    ModelStatus,
    GateResult,
    DriftMetrics,
    QualityGateConfig,
    ModelVersion,
    TrainingCycleResult,
    ContinuousTrainingConfig,
    ModelRegistry,
    DriftDetector,
    QualityGateEvaluator,
    ReplayBuffer,
    ContinuousTrainingOrchestrator,
    create_continuous_training_orchestrator,
    create_model_registry,
)

__all__ = [
    # Errors
    "JEPAUnavailableError",
    "WaldoIntegrationUnavailable",

    # Core configuration
    "VJEPAConfig",
    "ImageryType",
    "PretrainingMode",
    "MaskingStrategy",
    "BackboneSize",
    
    # Data structures
    "ImageryMetadata",
    "Embedding",
    "AnomalyResult",
    "ChangeDetectionResult",
    
    # Vector index
    "VectorIndex",
    "FaissIndex",
    
    # Data loading
    "MiningDataLoader",
    "MultiScaleMasking",
    
    # Models
    "VJEPAEncoder",
    "VJEPAPredictor",
    "VJEPAPretrainer",
    "VJEPAFeatureExtractor",
    
    # Applications
    "AnomalyDetector",
    "ChangeDetector",
    "SimilaritySearch",
    
    # Factory functions
    "create_vjepa_config",
    "create_mining_data_loader",
    "create_pretrainer",
    "create_feature_extractor",
    "create_anomaly_detector",
    "create_change_detector",
    "create_similarity_search",
    
    # Pretraining pipeline
    "PretrainingJob",
    "TilingConfig",
    "ImageTiler",
    "VideoChunker",
    "CorePhotoProcessor",
    "DatasetManifestBuilder",
    "PretrainingRunner",
    "prepare_mining_dataset",
    "run_pretraining",
    
    # WALDO/SAM3 integration
    "IntegrationMode",
    "DetectionTarget",
    "SegmentationTarget",
    "DetectionResult",
    "SegmentationResult",
    "JEPAPrompt",
    "WALDOJEPAIntegration",
    "SAM3JEPAIntegration",
    "UnifiedMiningVisionPipeline",
    "FeatureDistillation",
    "create_waldo_jepa_integration",
    "create_sam3_jepa_integration",
    "create_unified_pipeline",
    
    # LLM integration
    "LLMProvider",
    "ExplanationType",
    "DomainContext",
    "RetrievedEvidence",
    "StructuredExplanation",
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "OllamaClient",
    "MockLLMClient",
    "EvidenceRetriever",
    "ExplanationService",
    "JEPAChat",
    "JEPAOrchestrator",
    "create_llm_client",
    "create_explanation_service",
    "create_jepa_chat",
    "create_orchestrator",
    "explain_jepa_finding",
    
    # Lakehouse integration
    "TableFormat",
    "StorageBackend",
    "LakehouseConfig",
    "EmbeddingRecord",
    "FindingRecord",
    "TrainingRunRecord",
    "LakehouseBackend",
    "LocalParquetBackend",
    "LocalJSONBackend",
    "DeltaLakeBackend",
    "JEPALakehouseStore",
    "TrainingDatasetManager",
    "RAGRetriever",
    "create_jepa_lakehouse_store",
    "create_training_dataset_manager",
    "create_rag_retriever",
    
    # Continuous training
    "RetrainingTrigger",
    "ModelStatus",
    "GateResult",
    "DriftMetrics",
    "QualityGateConfig",
    "ModelVersion",
    "TrainingCycleResult",
    "ContinuousTrainingConfig",
    "ModelRegistry",
    "DriftDetector",
    "QualityGateEvaluator",
    "ReplayBuffer",
    "ContinuousTrainingOrchestrator",
    "create_continuous_training_orchestrator",
    "create_model_registry",
]

__version__ = "1.3.0"
