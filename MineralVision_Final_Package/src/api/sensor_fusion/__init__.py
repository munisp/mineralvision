"""
MineralVision Sensor Fusion Module.

This module provides comprehensive sensor fusion capabilities including:
- Core sensor data structures and fusion algorithms
- Kalman filtering for temporal fusion
- Deep learning-based neural network fusion
- Real-time streaming fusion
- Adapters for various sensor types
"""

from .core import (
    SensorData,
    SensorType,
    DataDimension,
    SensorDataAdapter,
    SensorFusionAlgorithm,
    SpatialAlignmentMethod,
    UncertaintyQuantification,
    SensorFusionManager
)

from .fusion_algorithms import (
    WeightedAverageFusion,
    BayesianFusion
)

from .kalman_fusion import (
    KalmanFilterType,
    KalmanState,
    KalmanConfig,
    StandardKalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    EnsembleKalmanFilter,
    AdaptiveKalmanFilter,
    KalmanFusionAlgorithm,
    create_kalman_filter,
    fuse_time_series
)

from .deep_learning_fusion import (
    FusionArchitecture,
    DeepFusionConfig,
    DeepFusionModel,
    EarlyFusionNetwork,
    LateFusionNetwork,
    AttentionFusionNetwork,
    TransformerFusionNetwork,
    CrossModalAttentionNetwork,
    MultimodalAutoencoder,
    DeepLearningFusionAlgorithm,
    SensorFusionDataset,
    create_fusion_model,
    fuse_with_attention
)

from .streaming_fusion import (
    StreamingMode,
    WindowType,
    StreamConfig,
    StreamingMetrics,
    SensorMessage,
    StreamingWindow,
    StreamingBuffer,
    IncrementalFusion,
    StreamingFusionPipeline,
    AsyncStreamingFusion,
    StreamingFusionSource,
    StreamingFusionSink,
    create_streaming_pipeline,
    create_sensor_message
)

from .hyperspectral_adapter import HyperspectralAdapter
from .lidar_adapter import LiDARAdapter
from .magnetometry_adapter import MagnetometryAdapter

# New sensor processing pipelines
from .magnetometry_pipeline import (
    MagneticFieldComponent,
    LevelingMethod,
    DerivativeType,
    MagnetometerConfig,
    SurveyLine,
    IGRFModel,
    PlatformCompensation,
    DiurnalCorrection,
    IGRFRemoval,
    TieLineLeveling,
    MicroLeveling,
    MagneticDerivatives,
    MagnetometryPipeline,
    create_magnetometry_pipeline,
    create_survey_design
)

from .radiometrics_pipeline import (
    RadiometricWindow,
    RadiometricProduct,
    SpectrometerConfig,
    RadiometricMeasurement,
    CalibrationPad,
    EnergyCalibration,
    WindowExtraction,
    StrippingCorrection,
    AltitudeCorrection,
    DeadTimeCorrection,
    BackgroundCorrection,
    SensitivityCalibration,
    DoseRateCalculation,
    RadiometricsPipeline,
    create_radiometrics_pipeline,
    create_survey_constraints
)

from .gpr_pipeline import (
    GPRSurveyMethod,
    GPRAntennaFrequency,
    MigrationMethod,
    GainType,
    GPRConfig,
    GPRTrace,
    GPRLine,
    TimeZeroCorrection,
    DewowFilter,
    BackgroundRemoval,
    BandpassFilter,
    GainFunction,
    VelocityEstimation,
    Migration,
    DepthConversion,
    GPRPipeline,
    create_gpr_pipeline,
    create_survey_design_gpr
)

from .segy_ingestion import (
    SEGYRevision,
    DataSampleFormat,
    SortingCode,
    TextHeader,
    BinaryHeader,
    TraceHeader,
    Trace,
    HeaderSchema,
    HeaderSchemaRegistry,
    DataFormatConverter,
    SEGYReader,
    SEGYWriter,
    SEGYIngestionPipeline,
    create_segy_pipeline,
    validate_segy_file
)

from .segy_visualization import (
    ViewType,
    ColorMap,
    InterpolationMethod,
    LayoutType,
    ViewSettings,
    SlicePosition,
    ViewState,
    SeismicVolume,
    AmplitudeStatistics,
    ColorMapGenerator,
    SliceRenderer,
    SeismicViewer,
    SEGYViewerIntegration,
    create_segy_viewer,
    create_volume_from_array
)

from .medusa_radiometrics import (
    MedusaSensorType,
    DetectorType,
    DataFormat,
    FlightMode,
    MedusaSensorSpec,
    MedusaCalibration,
    MedusaMeasurement,
    DroneFlightConstraints,
    MedusaDataParser,
    MedusaSpectralFitting,
    MedusaProcessor,
    MedusaFlightPlanner,
    create_medusa_processor,
    create_medusa_flight_planner
)

from .drone_gpr import (
    ZondAeroModel,
    GPRApplication,
    TerrainFollowingMode,
    ZondAeroSpec,
    DroneGPRConfig,
    DroneGPRTrace,
    DroneGPRLine,
    GPR3DVolume,
    DroneGPRProcessor,
    HorizontalSliceGenerator,
    ThicknessGridGenerator,
    GPR3DVolumeBuilder,
    DroneGPRFlightPlanner,
    DroneGPRPipeline,
    create_drone_gpr_pipeline,
    create_gpr_flight_planner
)

from .tiledb_segy import (
    TileDBGeometry,
    StorageBackend,
    Endianness,
    TileDBConfig,
    SEGYMetadata,
    TileDBArrayInfo,
    TileDBArraySchema,
    SEGYToTileDBConverter,
    TileDBSeismicArray,
    TileDBSegyIntegration,
    open_tiledb_segy,
    convert_segy_to_tiledb,
    create_tiledb_integration
)

from .drone_telemetry import (
    FlightQuality,
    SensorHealth,
    LineStatus,
    GNSSPosition,
    IMUData,
    LeverArm,
    SensorReading,
    FlightLine,
    FlightQualityReport,
    TimeSynchronizer,
    TerrainFollowingAnalyzer,
    SpeedComplianceChecker,
    SensorHealthMonitor,
    BadLineDetector,
    DroneTelemetryNormalizer,
    create_telemetry_normalizer,
    create_time_synchronizer,
    create_bad_line_detector,
)

__all__ = [
    # Core
    'SensorData',
    'SensorType',
    'DataDimension',
    'SensorDataAdapter',
    'SensorFusionAlgorithm',
    'SpatialAlignmentMethod',
    'UncertaintyQuantification',
    'SensorFusionManager',
    
    # Fusion Algorithms
    'WeightedAverageFusion',
    'BayesianFusion',
    
    # Kalman Fusion
    'KalmanFilterType',
    'KalmanState',
    'KalmanConfig',
    'StandardKalmanFilter',
    'ExtendedKalmanFilter',
    'UnscentedKalmanFilter',
    'EnsembleKalmanFilter',
    'AdaptiveKalmanFilter',
    'KalmanFusionAlgorithm',
    'create_kalman_filter',
    'fuse_time_series',
    
    # Deep Learning Fusion
    'FusionArchitecture',
    'DeepFusionConfig',
    'DeepFusionModel',
    'EarlyFusionNetwork',
    'LateFusionNetwork',
    'AttentionFusionNetwork',
    'TransformerFusionNetwork',
    'CrossModalAttentionNetwork',
    'MultimodalAutoencoder',
    'DeepLearningFusionAlgorithm',
    'SensorFusionDataset',
    'create_fusion_model',
    'fuse_with_attention',
    
    # Streaming Fusion
    'StreamingMode',
    'WindowType',
    'StreamConfig',
    'StreamingMetrics',
    'SensorMessage',
    'StreamingWindow',
    'StreamingBuffer',
    'IncrementalFusion',
    'StreamingFusionPipeline',
    'AsyncStreamingFusion',
    'StreamingFusionSource',
    'StreamingFusionSink',
    'create_streaming_pipeline',
    'create_sensor_message',
    
    # Adapters
    'HyperspectralAdapter',
    'LiDARAdapter',
    'MagnetometryAdapter',
    
    # Magnetometry Pipeline
    'MagneticFieldComponent',
    'LevelingMethod',
    'DerivativeType',
    'MagnetometerConfig',
    'SurveyLine',
    'IGRFModel',
    'PlatformCompensation',
    'DiurnalCorrection',
    'IGRFRemoval',
    'TieLineLeveling',
    'MicroLeveling',
    'MagneticDerivatives',
    'MagnetometryPipeline',
    'create_magnetometry_pipeline',
    'create_survey_design',
    
    # Radiometrics Pipeline
    'RadiometricWindow',
    'RadiometricProduct',
    'SpectrometerConfig',
    'RadiometricMeasurement',
    'CalibrationPad',
    'EnergyCalibration',
    'WindowExtraction',
    'StrippingCorrection',
    'AltitudeCorrection',
    'DeadTimeCorrection',
    'BackgroundCorrection',
    'SensitivityCalibration',
    'DoseRateCalculation',
    'RadiometricsPipeline',
    'create_radiometrics_pipeline',
    'create_survey_constraints',
    
    # GPR Pipeline
    'GPRSurveyMethod',
    'GPRAntennaFrequency',
    'MigrationMethod',
    'GainType',
    'GPRConfig',
    'GPRTrace',
    'GPRLine',
    'TimeZeroCorrection',
    'DewowFilter',
    'BackgroundRemoval',
    'BandpassFilter',
    'GainFunction',
    'VelocityEstimation',
    'Migration',
    'DepthConversion',
    'GPRPipeline',
    'create_gpr_pipeline',
    'create_survey_design_gpr',
    
    # SEG-Y Ingestion
    'SEGYRevision',
    'DataSampleFormat',
    'SortingCode',
    'TextHeader',
    'BinaryHeader',
    'TraceHeader',
    'Trace',
    'HeaderSchema',
    'HeaderSchemaRegistry',
    'DataFormatConverter',
    'SEGYReader',
    'SEGYWriter',
    'SEGYIngestionPipeline',
    'create_segy_pipeline',
    'validate_segy_file',
    
    # SEG-Y Visualization
    'ViewType',
    'ColorMap',
    'InterpolationMethod',
    'LayoutType',
    'ViewSettings',
    'SlicePosition',
    'ViewState',
    'SeismicVolume',
    'AmplitudeStatistics',
    'ColorMapGenerator',
    'SliceRenderer',
    'SeismicViewer',
    'SEGYViewerIntegration',
    'create_segy_viewer',
    'create_volume_from_array',
    
    # Medusa Radiometrics
    'MedusaSensorType',
    'DetectorType',
    'DataFormat',
    'FlightMode',
    'MedusaSensorSpec',
    'MedusaCalibration',
    'MedusaMeasurement',
    'DroneFlightConstraints',
    'MedusaDataParser',
    'MedusaSpectralFitting',
    'MedusaProcessor',
    'MedusaFlightPlanner',
    'create_medusa_processor',
    'create_medusa_flight_planner',
    
    # Drone GPR
    'ZondAeroModel',
    'GPRApplication',
    'TerrainFollowingMode',
    'ZondAeroSpec',
    'DroneGPRConfig',
    'DroneGPRTrace',
    'DroneGPRLine',
    'GPR3DVolume',
    'DroneGPRProcessor',
    'HorizontalSliceGenerator',
    'ThicknessGridGenerator',
    'GPR3DVolumeBuilder',
    'DroneGPRFlightPlanner',
    'DroneGPRPipeline',
    'create_drone_gpr_pipeline',
    'create_gpr_flight_planner',
    
    # TileDB-Segy
    'TileDBGeometry',
    'StorageBackend',
    'Endianness',
    'TileDBConfig',
    'SEGYMetadata',
    'TileDBArrayInfo',
    'TileDBArraySchema',
    'SEGYToTileDBConverter',
    'TileDBSeismicArray',
    'TileDBSegyIntegration',
    'open_tiledb_segy',
    'convert_segy_to_tiledb',
    'create_tiledb_integration',
    
    # Drone Telemetry
    'FlightQuality',
    'SensorHealth',
    'LineStatus',
    'GNSSPosition',
    'IMUData',
    'LeverArm',
    'SensorReading',
    'FlightLine',
    'FlightQualityReport',
    'TimeSynchronizer',
    'TerrainFollowingAnalyzer',
    'SpeedComplianceChecker',
    'SensorHealthMonitor',
    'BadLineDetector',
    'DroneTelemetryNormalizer',
    'create_telemetry_normalizer',
    'create_time_synchronizer',
    'create_bad_line_detector',
]
