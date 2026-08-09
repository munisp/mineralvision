"""
Climate Resilience Analysis Module for MineralVision.

This module provides comprehensive climate resilience analysis including:
- Real API integration with ERA5, OpenWeatherMap, and other providers
- Real-time climate monitoring with alerting
- Ensemble climate models for CMIP6 projections
- Satellite climate observations (MODIS, Landsat, Sentinel, GOES)
"""

from .climate_resilience_analysis import ClimateResilienceAnalysis
from .advanced_climate import (
    ClimateDataSource,
    ClimateAlert,
    AlertSeverity,
    ClimateDataProvider,
    ERA5Provider,
    OpenWeatherMapProvider,
    EnsembleClimateModel,
    RealTimeClimateMonitor,
    SatelliteClimateObserver,
    create_climate_provider,
    create_ensemble_model,
    create_realtime_monitor,
    create_satellite_observer
)

__all__ = [
    'ClimateResilienceAnalysis',
    'ClimateDataSource',
    'ClimateAlert',
    'AlertSeverity',
    'ClimateDataProvider',
    'ERA5Provider',
    'OpenWeatherMapProvider',
    'EnsembleClimateModel',
    'RealTimeClimateMonitor',
    'SatelliteClimateObserver',
    'create_climate_provider',
    'create_ensemble_model',
    'create_realtime_monitor',
    'create_satellite_observer'
]
