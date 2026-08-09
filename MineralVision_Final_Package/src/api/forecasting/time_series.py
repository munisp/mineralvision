"""
Time-Series Forecasting Module for MineralVision.

Provides forecasting capabilities for:
- Brine chemistry evolution
- Groundwater level prediction
- Seasonal soil moisture patterns
- Environmental monitoring trends
- Production forecasting
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import hashlib

logger = logging.getLogger(__name__)


class ForecastModel(Enum):
    """Forecasting model types."""
    PROPHET = "prophet"
    NEURAL_PROPHET = "neural_prophet"
    ARIMA = "arima"
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class SeasonalityType(Enum):
    """Types of seasonality."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class TrendType(Enum):
    """Types of trends."""
    LINEAR = "linear"
    LOGISTIC = "logistic"
    FLAT = "flat"
    PIECEWISE = "piecewise"


class ForecastMetric(Enum):
    """Forecast evaluation metrics."""
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    SMAPE = "smape"
    R2 = "r2"


@dataclass
class TimeSeriesData:
    """Time series data container."""
    timestamps: List[datetime]
    values: List[float]
    name: str = "series"
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamps': [t.isoformat() for t in self.timestamps],
            'values': self.values,
            'name': self.name,
            'unit': self.unit,
            'metadata': self.metadata
        }
        
    def __len__(self) -> int:
        return len(self.values)
        
    def get_frequency(self) -> Optional[str]:
        """Infer data frequency."""
        if len(self.timestamps) < 2:
            return None
            
        diffs = [(self.timestamps[i+1] - self.timestamps[i]).total_seconds() 
                 for i in range(min(10, len(self.timestamps)-1))]
        avg_diff = np.mean(diffs)
        
        if avg_diff < 3600:
            return "hourly"
        elif avg_diff < 86400:
            return "daily"
        elif avg_diff < 604800:
            return "weekly"
        elif avg_diff < 2592000:
            return "monthly"
        else:
            return "yearly"


@dataclass
class ForecastResult:
    """Forecast result."""
    forecast_id: str
    model_type: ForecastModel
    timestamps: List[datetime]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence_level: float = 0.95
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'forecast_id': self.forecast_id,
            'model_type': self.model_type.value,
            'timestamps': [t.isoformat() for t in self.timestamps],
            'predictions': self.predictions,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_level': self.confidence_level,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ForecastMetrics:
    """Forecast evaluation metrics."""
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'smape': self.smape,
            'r2': self.r2
        }


@dataclass
class SeasonalComponent:
    """Seasonal component of time series."""
    seasonality_type: SeasonalityType
    period: int
    amplitude: float
    phase: float
    fourier_order: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'seasonality_type': self.seasonality_type.value,
            'period': self.period,
            'amplitude': self.amplitude,
            'phase': self.phase,
            'fourier_order': self.fourier_order
        }


@dataclass
class TrendComponent:
    """Trend component of time series."""
    trend_type: TrendType
    slope: float
    intercept: float
    changepoints: List[datetime] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trend_type': self.trend_type.value,
            'slope': self.slope,
            'intercept': self.intercept,
            'changepoints': [c.isoformat() for c in self.changepoints]
        }


@dataclass
class Decomposition:
    """Time series decomposition."""
    trend: List[float]
    seasonal: List[float]
    residual: List[float]
    timestamps: List[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'residual': self.residual,
            'timestamps': [t.isoformat() for t in self.timestamps]
        }


class Forecaster(ABC):
    """Abstract base class for forecasters."""
    
    @abstractmethod
    def fit(self, data: TimeSeriesData) -> None:
        """Fit the model to data."""
        pass
        
    @abstractmethod
    def predict(self, periods: int) -> ForecastResult:
        """Generate forecast for future periods."""
        pass
        
    @abstractmethod
    def evaluate(self, actual: List[float], predicted: List[float]) -> ForecastMetrics:
        """Evaluate forecast accuracy."""
        pass


class ProphetForecaster(Forecaster):
    """Prophet-style forecaster implementation."""
    
    def __init__(self, seasonality_mode: str = "additive",
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05):
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        
        self._data: Optional[TimeSeriesData] = None
        self._trend: Optional[TrendComponent] = None
        self._seasonalities: List[SeasonalComponent] = []
        self._fitted = False
        
    def fit(self, data: TimeSeriesData) -> None:
        """Fit Prophet-style model."""
        self._data = data
        
        self._trend = self._fit_trend(data)
        
        if self.yearly_seasonality:
            yearly = self._fit_seasonality(data, SeasonalityType.YEARLY, 365.25)
            self._seasonalities.append(yearly)
            
        if self.weekly_seasonality:
            weekly = self._fit_seasonality(data, SeasonalityType.WEEKLY, 7)
            self._seasonalities.append(weekly)
            
        if self.daily_seasonality:
            daily = self._fit_seasonality(data, SeasonalityType.DAILY, 1)
            self._seasonalities.append(daily)
            
        self._fitted = True
        
    def predict(self, periods: int, freq: str = "D") -> ForecastResult:
        """Generate forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        last_timestamp = self._data.timestamps[-1]
        
        if freq == "D":
            delta = timedelta(days=1)
        elif freq == "H":
            delta = timedelta(hours=1)
        elif freq == "W":
            delta = timedelta(weeks=1)
        elif freq == "M":
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)
            
        future_timestamps = [last_timestamp + delta * (i + 1) for i in range(periods)]
        
        predictions = []
        for t in future_timestamps:
            pred = self._predict_point(t)
            predictions.append(pred)
            
        std_residual = self._estimate_residual_std()
        z_score = 1.96
        
        lower_bound = [p - z_score * std_residual for p in predictions]
        upper_bound = [p + z_score * std_residual for p in predictions]
        
        forecast_id = hashlib.md5(
            f"prophet:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return ForecastResult(
            forecast_id=forecast_id,
            model_type=ForecastModel.PROPHET,
            timestamps=future_timestamps,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        
    def evaluate(self, actual: List[float], predicted: List[float]) -> ForecastMetrics:
        """Evaluate forecast accuracy."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        non_zero = actual != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = 0.0
            
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return ForecastMetrics(
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            smape=float(smape),
            r2=float(r2)
        )
        
    def _fit_trend(self, data: TimeSeriesData) -> TrendComponent:
        """Fit trend component."""
        n = len(data.values)
        t = np.arange(n)
        y = np.array(data.values)
        
        slope = (n * np.sum(t * y) - np.sum(t) * np.sum(y)) / (n * np.sum(t**2) - np.sum(t)**2 + 1e-8)
        intercept = (np.sum(y) - slope * np.sum(t)) / n
        
        return TrendComponent(
            trend_type=TrendType.LINEAR,
            slope=float(slope),
            intercept=float(intercept)
        )
        
    def _fit_seasonality(self, data: TimeSeriesData, 
                        seasonality_type: SeasonalityType,
                        period: float) -> SeasonalComponent:
        """Fit seasonal component using Fourier series."""
        n = len(data.values)
        t = np.arange(n)
        y = np.array(data.values)
        
        trend_values = self._trend.intercept + self._trend.slope * t
        detrended = y - trend_values
        
        freq = 2 * np.pi / period
        sin_component = np.sin(freq * t)
        cos_component = np.cos(freq * t)
        
        a = np.sum(detrended * sin_component) / (n / 2)
        b = np.sum(detrended * cos_component) / (n / 2)
        
        amplitude = np.sqrt(a**2 + b**2)
        phase = np.arctan2(b, a)
        
        return SeasonalComponent(
            seasonality_type=seasonality_type,
            period=int(period),
            amplitude=float(amplitude),
            phase=float(phase)
        )
        
    def _predict_point(self, timestamp: datetime) -> float:
        """Predict value for a single timestamp."""
        days_since_start = (timestamp - self._data.timestamps[0]).total_seconds() / 86400
        
        trend = self._trend.intercept + self._trend.slope * days_since_start
        
        seasonal = 0.0
        for s in self._seasonalities:
            freq = 2 * np.pi / s.period
            seasonal += s.amplitude * np.sin(freq * days_since_start + s.phase)
            
        if self.seasonality_mode == "multiplicative":
            return trend * (1 + seasonal)
        else:
            return trend + seasonal
            
    def _estimate_residual_std(self) -> float:
        """Estimate standard deviation of residuals."""
        if not self._data:
            return 1.0
            
        residuals = []
        for i, (t, y) in enumerate(zip(self._data.timestamps, self._data.values)):
            pred = self._predict_point(t)
            residuals.append(y - pred)
            
        return float(np.std(residuals))
        
    def decompose(self) -> Decomposition:
        """Decompose time series into components."""
        if not self._fitted:
            raise ValueError("Model must be fitted before decomposition")
            
        n = len(self._data.values)
        t = np.arange(n)
        
        trend = [self._trend.intercept + self._trend.slope * i for i in t]
        
        seasonal = []
        for i, timestamp in enumerate(self._data.timestamps):
            days = (timestamp - self._data.timestamps[0]).total_seconds() / 86400
            s = 0.0
            for sc in self._seasonalities:
                freq = 2 * np.pi / sc.period
                s += sc.amplitude * np.sin(freq * days + sc.phase)
            seasonal.append(s)
            
        residual = [y - tr - se for y, tr, se in zip(self._data.values, trend, seasonal)]
        
        return Decomposition(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            timestamps=self._data.timestamps
        )


class ARIMAForecaster(Forecaster):
    """ARIMA forecaster implementation."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = None):
        self.order = order
        self.seasonal_order = seasonal_order
        
        self._data: Optional[TimeSeriesData] = None
        self._ar_coeffs: List[float] = []
        self._ma_coeffs: List[float] = []
        self._diff_order: int = order[1]
        self._fitted = False
        
    def fit(self, data: TimeSeriesData) -> None:
        """Fit ARIMA model."""
        self._data = data
        
        values = np.array(data.values)
        
        for _ in range(self._diff_order):
            values = np.diff(values)
            
        p, d, q = self.order
        
        self._ar_coeffs = self._estimate_ar_coeffs(values, p)
        self._ma_coeffs = self._estimate_ma_coeffs(values, q)
        
        self._fitted = True
        
    def predict(self, periods: int, freq: str = "D") -> ForecastResult:
        """Generate ARIMA forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        last_timestamp = self._data.timestamps[-1]
        
        if freq == "D":
            delta = timedelta(days=1)
        elif freq == "H":
            delta = timedelta(hours=1)
        elif freq == "W":
            delta = timedelta(weeks=1)
        else:
            delta = timedelta(days=1)
            
        future_timestamps = [last_timestamp + delta * (i + 1) for i in range(periods)]
        
        values = list(self._data.values)
        predictions = []
        
        for _ in range(periods):
            pred = self._predict_next(values)
            predictions.append(pred)
            values.append(pred)
            
        std_residual = self._estimate_residual_std()
        z_score = 1.96
        
        lower_bound = [p - z_score * std_residual * np.sqrt(i+1) for i, p in enumerate(predictions)]
        upper_bound = [p + z_score * std_residual * np.sqrt(i+1) for i, p in enumerate(predictions)]
        
        forecast_id = hashlib.md5(
            f"arima:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return ForecastResult(
            forecast_id=forecast_id,
            model_type=ForecastModel.ARIMA,
            timestamps=future_timestamps,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        
    def evaluate(self, actual: List[float], predicted: List[float]) -> ForecastMetrics:
        """Evaluate forecast accuracy."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        non_zero = actual != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = 0.0
            
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return ForecastMetrics(
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            smape=float(smape),
            r2=float(r2)
        )
        
    def _estimate_ar_coeffs(self, values: np.ndarray, p: int) -> List[float]:
        """Estimate AR coefficients using Yule-Walker equations."""
        if p == 0 or len(values) < p + 1:
            return []
            
        n = len(values)
        
        autocorr = []
        for k in range(p + 1):
            if k == 0:
                autocorr.append(1.0)
            else:
                c = np.sum(values[k:] * values[:-k]) / (n - k)
                c0 = np.sum(values ** 2) / n
                autocorr.append(c / (c0 + 1e-8))
                
        R = np.zeros((p, p))
        r = np.zeros(p)
        
        for i in range(p):
            r[i] = autocorr[i + 1]
            for j in range(p):
                R[i, j] = autocorr[abs(i - j)]
                
        try:
            coeffs = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(p)
            
        return coeffs.tolist()
        
    def _estimate_ma_coeffs(self, values: np.ndarray, q: int) -> List[float]:
        """Estimate MA coefficients."""
        if q == 0:
            return []
            
        return [0.5 / (i + 1) for i in range(q)]
        
    def _predict_next(self, values: List[float]) -> float:
        """Predict next value."""
        p = len(self._ar_coeffs)
        
        pred = 0.0
        for i, coeff in enumerate(self._ar_coeffs):
            if len(values) > i:
                pred += coeff * values[-(i + 1)]
                
        return pred
        
    def _estimate_residual_std(self) -> float:
        """Estimate residual standard deviation."""
        if not self._data or len(self._data.values) < 10:
            return 1.0
            
        values = list(self._data.values)
        residuals = []
        
        for i in range(len(self._ar_coeffs), len(values)):
            pred = self._predict_next(values[:i])
            residuals.append(values[i] - pred)
            
        return float(np.std(residuals)) if residuals else 1.0


class ExponentialSmoothingForecaster(Forecaster):
    """Exponential smoothing forecaster."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1,
                 seasonal_periods: int = 12):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_periods = seasonal_periods
        
        self._level: float = 0.0
        self._trend: float = 0.0
        self._seasonal: List[float] = []
        self._data: Optional[TimeSeriesData] = None
        self._fitted = False
        
    def fit(self, data: TimeSeriesData) -> None:
        """Fit Holt-Winters exponential smoothing."""
        self._data = data
        values = np.array(data.values)
        n = len(values)
        
        self._level = np.mean(values[:self.seasonal_periods])
        self._trend = (np.mean(values[self.seasonal_periods:2*self.seasonal_periods]) - 
                      np.mean(values[:self.seasonal_periods])) / self.seasonal_periods
        
        self._seasonal = []
        for i in range(self.seasonal_periods):
            indices = list(range(i, n, self.seasonal_periods))
            if indices:
                self._seasonal.append(np.mean([values[j] for j in indices]) - self._level)
            else:
                self._seasonal.append(0.0)
                
        for t in range(n):
            season_idx = t % self.seasonal_periods
            
            new_level = self.alpha * (values[t] - self._seasonal[season_idx]) + \
                       (1 - self.alpha) * (self._level + self._trend)
            new_trend = self.beta * (new_level - self._level) + (1 - self.beta) * self._trend
            new_seasonal = self.gamma * (values[t] - new_level) + \
                          (1 - self.gamma) * self._seasonal[season_idx]
                          
            self._level = new_level
            self._trend = new_trend
            self._seasonal[season_idx] = new_seasonal
            
        self._fitted = True
        
    def predict(self, periods: int, freq: str = "D") -> ForecastResult:
        """Generate forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        last_timestamp = self._data.timestamps[-1]
        
        if freq == "D":
            delta = timedelta(days=1)
        elif freq == "H":
            delta = timedelta(hours=1)
        elif freq == "W":
            delta = timedelta(weeks=1)
        elif freq == "M":
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)
            
        future_timestamps = [last_timestamp + delta * (i + 1) for i in range(periods)]
        
        predictions = []
        for h in range(1, periods + 1):
            season_idx = (len(self._data.values) + h - 1) % self.seasonal_periods
            pred = self._level + h * self._trend + self._seasonal[season_idx]
            predictions.append(pred)
            
        std_residual = self._estimate_residual_std()
        z_score = 1.96
        
        lower_bound = [p - z_score * std_residual for p in predictions]
        upper_bound = [p + z_score * std_residual for p in predictions]
        
        forecast_id = hashlib.md5(
            f"ets:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return ForecastResult(
            forecast_id=forecast_id,
            model_type=ForecastModel.EXPONENTIAL_SMOOTHING,
            timestamps=future_timestamps,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        
    def evaluate(self, actual: List[float], predicted: List[float]) -> ForecastMetrics:
        """Evaluate forecast accuracy."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        non_zero = actual != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = 0.0
            
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return ForecastMetrics(
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            smape=float(smape),
            r2=float(r2)
        )
        
    def _estimate_residual_std(self) -> float:
        """Estimate residual standard deviation."""
        if not self._data:
            return 1.0
            
        residuals = []
        level = self._level
        trend = self._trend
        
        for t, y in enumerate(self._data.values):
            season_idx = t % self.seasonal_periods
            pred = level + trend + self._seasonal[season_idx]
            residuals.append(y - pred)
            
        return float(np.std(residuals)) if residuals else 1.0


class EnsembleForecaster(Forecaster):
    """Ensemble of multiple forecasters."""
    
    def __init__(self, forecasters: List[Forecaster] = None,
                 weights: List[float] = None):
        if forecasters is None:
            forecasters = [
                ProphetForecaster(),
                ARIMAForecaster(),
                ExponentialSmoothingForecaster()
            ]
        self.forecasters = forecasters
        self.weights = weights or [1.0 / len(forecasters)] * len(forecasters)
        self._data: Optional[TimeSeriesData] = None
        self._fitted = False
        
    def fit(self, data: TimeSeriesData) -> None:
        """Fit all forecasters."""
        self._data = data
        
        for forecaster in self.forecasters:
            try:
                forecaster.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {type(forecaster).__name__}: {e}")
                
        self._fitted = True
        
    def predict(self, periods: int, freq: str = "D") -> ForecastResult:
        """Generate ensemble forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        all_predictions = []
        all_lower = []
        all_upper = []
        timestamps = None
        
        for forecaster, weight in zip(self.forecasters, self.weights):
            try:
                result = forecaster.predict(periods, freq)
                all_predictions.append([p * weight for p in result.predictions])
                all_lower.append([l * weight for l in result.lower_bound])
                all_upper.append([u * weight for u in result.upper_bound])
                if timestamps is None:
                    timestamps = result.timestamps
            except Exception as e:
                logger.warning(f"Failed to predict with {type(forecaster).__name__}: {e}")
                
        if not all_predictions:
            raise ValueError("All forecasters failed")
            
        predictions = [sum(p[i] for p in all_predictions) for i in range(periods)]
        lower_bound = [sum(l[i] for l in all_lower) for i in range(periods)]
        upper_bound = [sum(u[i] for u in all_upper) for i in range(periods)]
        
        forecast_id = hashlib.md5(
            f"ensemble:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return ForecastResult(
            forecast_id=forecast_id,
            model_type=ForecastModel.ENSEMBLE,
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        
    def evaluate(self, actual: List[float], predicted: List[float]) -> ForecastMetrics:
        """Evaluate forecast accuracy."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        non_zero = actual != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = 0.0
            
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return ForecastMetrics(
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            smape=float(smape),
            r2=float(r2)
        )


class BrineChemistryForecaster:
    """Specialized forecaster for brine chemistry evolution."""
    
    def __init__(self):
        self.forecasters: Dict[str, Forecaster] = {}
        self.elements = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Cl', 'SO4', 'B']
        
    def fit(self, chemistry_data: Dict[str, TimeSeriesData]) -> None:
        """Fit forecasters for each element."""
        for element in self.elements:
            if element in chemistry_data:
                forecaster = ProphetForecaster(
                    yearly_seasonality=True,
                    weekly_seasonality=False
                )
                forecaster.fit(chemistry_data[element])
                self.forecasters[element] = forecaster
                
    def predict(self, periods: int) -> Dict[str, ForecastResult]:
        """Predict future brine chemistry."""
        results = {}
        for element, forecaster in self.forecasters.items():
            results[element] = forecaster.predict(periods)
        return results
        
    def predict_lithium_grade(self, periods: int) -> ForecastResult:
        """Predict lithium grade evolution."""
        if 'Li' in self.forecasters:
            return self.forecasters['Li'].predict(periods)
        raise ValueError("Lithium data not fitted")


class GroundwaterForecaster:
    """Specialized forecaster for groundwater levels."""
    
    def __init__(self):
        self.forecaster = ProphetForecaster(
            yearly_seasonality=True,
            weekly_seasonality=False,
            seasonality_mode="additive"
        )
        self._fitted = False
        
    def fit(self, water_level_data: TimeSeriesData,
           precipitation_data: TimeSeriesData = None) -> None:
        """Fit groundwater level forecaster."""
        self.forecaster.fit(water_level_data)
        self._precipitation = precipitation_data
        self._fitted = True
        
    def predict(self, periods: int) -> ForecastResult:
        """Predict groundwater levels."""
        return self.forecaster.predict(periods)
        
    def get_seasonal_pattern(self) -> Dict[str, Any]:
        """Get seasonal groundwater pattern."""
        if not self._fitted:
            raise ValueError("Model must be fitted first")
            
        decomp = self.forecaster.decompose()
        
        return {
            'seasonal_amplitude': max(decomp.seasonal) - min(decomp.seasonal),
            'peak_month': np.argmax(decomp.seasonal[:12]) + 1 if len(decomp.seasonal) >= 12 else None,
            'trough_month': np.argmin(decomp.seasonal[:12]) + 1 if len(decomp.seasonal) >= 12 else None
        }


class SoilMoistureForecaster:
    """Specialized forecaster for soil moisture patterns."""
    
    def __init__(self):
        self.forecaster = ProphetForecaster(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=True
        )
        self._fitted = False
        
    def fit(self, moisture_data: TimeSeriesData) -> None:
        """Fit soil moisture forecaster."""
        self.forecaster.fit(moisture_data)
        self._fitted = True
        
    def predict(self, periods: int) -> ForecastResult:
        """Predict soil moisture."""
        return self.forecaster.predict(periods)
        
    def get_irrigation_recommendations(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get irrigation recommendations based on forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted first")
            
        forecast = self.predict(30)
        
        recommendations = []
        for i, (ts, pred, lower) in enumerate(zip(
            forecast.timestamps, forecast.predictions, forecast.lower_bound
        )):
            if lower < threshold:
                recommendations.append({
                    'date': ts,
                    'predicted_moisture': pred,
                    'lower_bound': lower,
                    'action': 'irrigate',
                    'urgency': 'high' if lower < threshold * 0.5 else 'medium'
                })
                
        return recommendations


class ForecastingService:
    """Main forecasting service for MineralVision."""
    
    def __init__(self):
        self.brine_forecaster = BrineChemistryForecaster()
        self.groundwater_forecaster = GroundwaterForecaster()
        self.soil_moisture_forecaster = SoilMoistureForecaster()
        
        self._cache: Dict[str, ForecastResult] = {}
        
    def forecast(self, data: TimeSeriesData, periods: int,
                model: ForecastModel = ForecastModel.ENSEMBLE) -> ForecastResult:
        """Generate forecast using specified model."""
        if model == ForecastModel.PROPHET:
            forecaster = ProphetForecaster()
        elif model == ForecastModel.ARIMA:
            forecaster = ARIMAForecaster()
        elif model == ForecastModel.EXPONENTIAL_SMOOTHING:
            forecaster = ExponentialSmoothingForecaster()
        elif model == ForecastModel.ENSEMBLE:
            forecaster = EnsembleForecaster()
        else:
            forecaster = ProphetForecaster()
            
        forecaster.fit(data)
        return forecaster.predict(periods)
        
    def cross_validate(self, data: TimeSeriesData, 
                      model: ForecastModel = ForecastModel.ENSEMBLE,
                      n_splits: int = 5,
                      horizon: int = 30) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        n = len(data.values)
        fold_size = (n - horizon) // n_splits
        
        all_metrics = []
        
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            test_end = min(train_end + horizon, n)
            
            train_data = TimeSeriesData(
                timestamps=data.timestamps[:train_end],
                values=data.values[:train_end],
                name=data.name
            )
            
            test_actual = data.values[train_end:test_end]
            
            result = self.forecast(train_data, len(test_actual), model)
            
            if model == ForecastModel.PROPHET:
                forecaster = ProphetForecaster()
            else:
                forecaster = EnsembleForecaster()
                
            metrics = forecaster.evaluate(test_actual, result.predictions[:len(test_actual)])
            all_metrics.append(metrics)
            
        avg_metrics = ForecastMetrics(
            mae=np.mean([m.mae for m in all_metrics]),
            rmse=np.mean([m.rmse for m in all_metrics]),
            mape=np.mean([m.mape for m in all_metrics]),
            smape=np.mean([m.smape for m in all_metrics]),
            r2=np.mean([m.r2 for m in all_metrics])
        )
        
        return {
            'average_metrics': avg_metrics.to_dict(),
            'fold_metrics': [m.to_dict() for m in all_metrics],
            'n_splits': n_splits,
            'horizon': horizon
        }


def create_forecasting_service() -> ForecastingService:
    """Factory function to create forecasting service."""
    return ForecastingService()


def create_prophet_forecaster(**kwargs) -> ProphetForecaster:
    """Factory function to create Prophet forecaster."""
    return ProphetForecaster(**kwargs)


def create_ensemble_forecaster(forecasters: List[Forecaster] = None) -> EnsembleForecaster:
    """Factory function to create ensemble forecaster."""
    return EnsembleForecaster(forecasters)
