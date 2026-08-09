"""
MineralVision Crop Monitoring Module.

Comprehensive EOS-style crop monitoring features:
- Vegetation Index Processing (NDVI, NDRE, SAVI, EVI, GNDVI, MSAVI, NDWI, NDMI)
- Field Management System (boundaries, upload, drawing)
- Weather Integration (14-day forecast, historical data)
- VRA Map Generation (sowing, nitrogen, P&K, custom)
- Alert System (vegetation stress, weather, pest/disease)

This module integrates with MineralVision's existing soil assessment
and satellite imagery capabilities to provide end-to-end agricultural
monitoring for oil palm, cocoa, ginger, and other crops.
"""

from .vegetation_indices import (
    VegetationIndexType,
    CropHealthStatus,
    CloudMaskMethod,
    SpectralBands,
    VegetationIndexResult,
    VegetationTimeSeries,
    VegetationIndexCalculator,
    CloudMasker,
    AtmosphericCorrector,
    TimeSeriesAnalyzer,
    CropMonitoringService,
    create_crop_monitoring_service,
    create_synthetic_spectral_bands
)

from .field_management import (
    FieldStatus,
    CropStage,
    BoundaryFormat,
    Coordinate,
    FieldBoundary,
    CropInfo,
    SoilInfo,
    Field,
    Farm,
    GeometryCalculator,
    BoundaryParser,
    FieldManager,
    CropCalendar,
    create_field_manager,
    create_sample_fields
)

from .weather_integration import (
    WeatherCondition,
    WeatherRiskLevel,
    WeatherAlertType,
    HourlyWeather,
    DailyWeather,
    WeatherForecast,
    HistoricalWeather,
    WeatherAlert,
    GrowingDegreeDays,
    EvapotranspirationCalculator,
    WeatherRiskAssessor,
    GrowingDegreeDaysCalculator,
    WeatherService,
    create_weather_service
)

from .vra_maps import (
    VRAMapType,
    ZoneMethod,
    ExportFormat,
    ApplicationUnit,
    ManagementZone,
    VRAMap,
    SowingMapParameters,
    NitrogenMapParameters,
    PKMapParameters,
    ZoneCreator,
    VRAMapGenerator,
    VRAExporter,
    SavingsCalculator,
    VRAService,
    create_vra_service,
    create_sample_vra_map
)

from .alert_system import (
    AlertSeverity,
    AlertCategory,
    AlertStatus,
    NotificationChannel,
    TriggerType,
    AlertRule,
    Alert,
    NotificationTemplate,
    AlertSubscription,
    AlertRuleEngine,
    NotificationService,
    AlertManager,
    AlertService,
    create_alert_service,
    create_sample_alerts
)

from .data_persistence import (
    StorageBackend,
    TableType,
    CropDataRecord,
    SpatialBoundary,
    VegetationIndexRecord,
    WeatherRecord,
    LakehouseAdapter,
    PostGISAdapter,
    SedonaAdapter,
    CropMonitoringDataStore,
    create_crop_data_store
)

__all__ = [
    # Vegetation Indices
    'VegetationIndexType',
    'CropHealthStatus',
    'CloudMaskMethod',
    'SpectralBands',
    'VegetationIndexResult',
    'VegetationTimeSeries',
    'VegetationIndexCalculator',
    'CloudMasker',
    'AtmosphericCorrector',
    'TimeSeriesAnalyzer',
    'CropMonitoringService',
    'create_crop_monitoring_service',
    'create_synthetic_spectral_bands',
    
    # Field Management
    'FieldStatus',
    'CropStage',
    'BoundaryFormat',
    'Coordinate',
    'FieldBoundary',
    'CropInfo',
    'SoilInfo',
    'Field',
    'Farm',
    'GeometryCalculator',
    'BoundaryParser',
    'FieldManager',
    'CropCalendar',
    'create_field_manager',
    'create_sample_fields',
    
    # Weather Integration
    'WeatherCondition',
    'WeatherRiskLevel',
    'WeatherAlertType',
    'HourlyWeather',
    'DailyWeather',
    'WeatherForecast',
    'HistoricalWeather',
    'WeatherAlert',
    'GrowingDegreeDays',
    'EvapotranspirationCalculator',
    'WeatherRiskAssessor',
    'GrowingDegreeDaysCalculator',
    'WeatherService',
    'create_weather_service',
    
    # VRA Maps
    'VRAMapType',
    'ZoneMethod',
    'ExportFormat',
    'ApplicationUnit',
    'ManagementZone',
    'VRAMap',
    'SowingMapParameters',
    'NitrogenMapParameters',
    'PKMapParameters',
    'ZoneCreator',
    'VRAMapGenerator',
    'VRAExporter',
    'SavingsCalculator',
    'VRAService',
    'create_vra_service',
    'create_sample_vra_map',
    
    # Alert System
    'AlertSeverity',
    'AlertCategory',
    'AlertStatus',
    'NotificationChannel',
    'TriggerType',
    'AlertRule',
    'Alert',
    'NotificationTemplate',
    'AlertSubscription',
    'AlertRuleEngine',
    'NotificationService',
    'AlertManager',
    'AlertService',
    'create_alert_service',
    'create_sample_alerts',
    
    # Data Persistence (Lakehouse + PostGIS + Sedona)
    'StorageBackend',
    'TableType',
    'CropDataRecord',
    'SpatialBoundary',
    'VegetationIndexRecord',
    'WeatherRecord',
    'LakehouseAdapter',
    'PostGISAdapter',
    'SedonaAdapter',
    'CropMonitoringDataStore',
    'create_crop_data_store',
]

__version__ = '1.0.0'
