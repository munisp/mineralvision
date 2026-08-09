"""
MineralVision WALDO Integration Module
======================================

This module provides integration between MineralVision platform and WALDO (Wide Area
Large-scale Detection and Observation) for object detection and tracking.

Enhanced features include:
- Model fine-tuning pipeline for custom mineral/geological classes
- Active learning for continuous model improvement
- Multi-camera fusion and stereo vision support
- Thermal/multispectral camera integration
- Real-time video streaming optimization (RTSP, WebRTC)
"""

from .detection import WALDODetector
from .tracking import ObjectTracker
from .measurement import MeasurementEngine
from .integration import WALDOIntegrationModule
from .advanced_waldo import (
    CameraType,
    StreamProtocol,
    SamplingStrategy,
    Detection,
    CameraConfig,
    TrainingConfig,
    ActiveLearningConfig,
    MineralDetectionDataset,
    DataAugmentation,
    FineTuningPipeline,
    ActiveLearningManager,
    MultiCameraFusion,
    ThermalIntegration,
    VideoStreamManager,
    AdvancedWALDOManager,
    create_advanced_waldo_manager,
    create_fine_tuning_pipeline,
    create_active_learning_manager,
    create_multi_camera_fusion,
    create_thermal_integration,
    create_stream_manager
)
from .rfdetr_backbone import (
    RFDETRVariant,
    ExportFormat,
    RFDETRConfig,
    RFDETRDetection,
    RFDETRTrainingConfig,
    RFDETRDetector,
    RFDETRFineTuner,
    RFDETRExporter,
    RFDETRSegmentation,
    UnifiedWALDODetector,
    create_rfdetr_detector,
    create_rfdetr_finetuner,
    create_rfdetr_exporter,
    create_unified_detector,
    compare_backbones
)

__all__ = [
    # Core WALDO
    'WALDODetector',
    'ObjectTracker',
    'MeasurementEngine',
    'WALDOIntegrationModule',
    # Advanced WALDO
    'CameraType',
    'StreamProtocol',
    'SamplingStrategy',
    'Detection',
    'CameraConfig',
    'TrainingConfig',
    'ActiveLearningConfig',
    'MineralDetectionDataset',
    'DataAugmentation',
    'FineTuningPipeline',
    'ActiveLearningManager',
    'MultiCameraFusion',
    'ThermalIntegration',
    'VideoStreamManager',
    'AdvancedWALDOManager',
    'create_advanced_waldo_manager',
    'create_fine_tuning_pipeline',
    'create_active_learning_manager',
    'create_multi_camera_fusion',
    'create_thermal_integration',
    'create_stream_manager',
    # RF-DETR Backbone
    'RFDETRVariant',
    'ExportFormat',
    'RFDETRConfig',
    'RFDETRDetection',
    'RFDETRTrainingConfig',
    'RFDETRDetector',
    'RFDETRFineTuner',
    'RFDETRExporter',
    'RFDETRSegmentation',
    'UnifiedWALDODetector',
    'create_rfdetr_detector',
    'create_rfdetr_finetuner',
    'create_rfdetr_exporter',
    'create_unified_detector',
    'compare_backbones'
]
