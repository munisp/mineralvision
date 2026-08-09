"""
Forecasting module for MineralVision.

Provides time-series forecasting capabilities.
"""

from .time_series import (
    ForecastModel,
    SeasonalityType,
    TrendType,
    ForecastMetric,
    TimeSeriesData,
    ForecastResult,
    ForecastMetrics,
    SeasonalComponent,
    TrendComponent,
    Decomposition,
    Forecaster,
    ProphetForecaster,
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
    EnsembleForecaster,
    BrineChemistryForecaster,
    GroundwaterForecaster,
    SoilMoistureForecaster,
    ForecastingService,
    create_forecasting_service,
    create_prophet_forecaster,
    create_ensemble_forecaster,
)

__all__ = [
    'ForecastModel',
    'SeasonalityType',
    'TrendType',
    'ForecastMetric',
    'TimeSeriesData',
    'ForecastResult',
    'ForecastMetrics',
    'SeasonalComponent',
    'TrendComponent',
    'Decomposition',
    'Forecaster',
    'ProphetForecaster',
    'ARIMAForecaster',
    'ExponentialSmoothingForecaster',
    'EnsembleForecaster',
    'BrineChemistryForecaster',
    'GroundwaterForecaster',
    'SoilMoistureForecaster',
    'ForecastingService',
    'create_forecasting_service',
    'create_prophet_forecaster',
    'create_ensemble_forecaster',
]
