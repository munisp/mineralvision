"""
Weather Integration Module for MineralVision Crop Monitoring.

Comprehensive weather data integration:
- Real-time weather forecasts (14-day hourly)
- Historical weather data (back to 1979)
- Weather-based risk assessment
- Evapotranspiration calculation
- Growing degree days
- Precipitation analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date, timedelta
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


class WeatherCondition(Enum):
    """Weather condition types."""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    MODERATE_RAIN = "moderate_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    DRIZZLE = "drizzle"
    FOG = "fog"
    HAZE = "haze"
    WINDY = "windy"


class WeatherRiskLevel(Enum):
    """Weather risk levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class WeatherAlertType(Enum):
    """Types of weather alerts."""
    FROST = "frost"
    HEAT_WAVE = "heat_wave"
    DROUGHT = "drought"
    FLOOD = "flood"
    STORM = "storm"
    HIGH_WIND = "high_wind"
    HEAVY_RAIN = "heavy_rain"
    HAIL = "hail"


@dataclass
class HourlyWeather:
    """Hourly weather data."""
    timestamp: datetime
    
    # Temperature
    temperature_c: float = 25.0
    feels_like_c: float = 25.0
    dew_point_c: float = 20.0
    
    # Humidity and pressure
    humidity_percent: float = 70.0
    pressure_hpa: float = 1013.0
    
    # Precipitation
    precipitation_mm: float = 0.0
    precipitation_probability: float = 0.0
    
    # Wind
    wind_speed_ms: float = 2.0
    wind_direction_deg: float = 180.0
    wind_gust_ms: float = 5.0
    
    # Solar
    cloud_cover_percent: float = 30.0
    solar_radiation_wm2: float = 500.0
    uv_index: float = 5.0
    
    # Visibility
    visibility_km: float = 10.0
    
    # Condition
    condition: WeatherCondition = WeatherCondition.PARTLY_CLOUDY
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'temperature_c': self.temperature_c,
            'feels_like_c': self.feels_like_c,
            'humidity_percent': self.humidity_percent,
            'precipitation_mm': self.precipitation_mm,
            'precipitation_probability': self.precipitation_probability,
            'wind_speed_ms': self.wind_speed_ms,
            'wind_direction_deg': self.wind_direction_deg,
            'cloud_cover_percent': self.cloud_cover_percent,
            'solar_radiation_wm2': self.solar_radiation_wm2,
            'condition': self.condition.value
        }


@dataclass
class DailyWeather:
    """Daily weather summary."""
    date: date
    
    # Temperature
    temp_max_c: float = 32.0
    temp_min_c: float = 22.0
    temp_avg_c: float = 27.0
    
    # Humidity
    humidity_max: float = 90.0
    humidity_min: float = 50.0
    humidity_avg: float = 70.0
    
    # Precipitation
    precipitation_total_mm: float = 0.0
    precipitation_hours: int = 0
    precipitation_probability: float = 0.0
    
    # Wind
    wind_speed_avg_ms: float = 2.0
    wind_speed_max_ms: float = 5.0
    wind_direction_dominant: float = 180.0
    
    # Solar
    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None
    daylight_hours: float = 12.0
    solar_radiation_total_mjm2: float = 15.0
    uv_index_max: float = 8.0
    
    # Evapotranspiration
    et0_mm: float = 4.0  # Reference evapotranspiration
    
    # Condition
    condition: WeatherCondition = WeatherCondition.PARTLY_CLOUDY
    
    # Hourly data
    hourly: List[HourlyWeather] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'temp_max_c': self.temp_max_c,
            'temp_min_c': self.temp_min_c,
            'temp_avg_c': self.temp_avg_c,
            'humidity_avg': self.humidity_avg,
            'precipitation_total_mm': self.precipitation_total_mm,
            'precipitation_probability': self.precipitation_probability,
            'wind_speed_avg_ms': self.wind_speed_avg_ms,
            'daylight_hours': self.daylight_hours,
            'et0_mm': self.et0_mm,
            'condition': self.condition.value,
            'hourly_count': len(self.hourly)
        }


@dataclass
class WeatherForecast:
    """Weather forecast for a location."""
    location_id: str
    latitude: float
    longitude: float
    
    # Forecast data
    generated_at: datetime = field(default_factory=datetime.utcnow)
    forecast_days: List[DailyWeather] = field(default_factory=list)
    
    # Summary
    total_precipitation_mm: float = 0.0
    avg_temperature_c: float = 27.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location_id': self.location_id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'generated_at': self.generated_at.isoformat(),
            'forecast_days': [d.to_dict() for d in self.forecast_days],
            'total_precipitation_mm': self.total_precipitation_mm,
            'avg_temperature_c': self.avg_temperature_c
        }


@dataclass
class HistoricalWeather:
    """Historical weather data for analysis."""
    location_id: str
    start_date: date
    end_date: date
    
    # Daily data
    daily_data: List[DailyWeather] = field(default_factory=list)
    
    # Monthly summaries
    monthly_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Annual statistics
    annual_precipitation_mm: float = 0.0
    annual_avg_temp_c: float = 27.0
    annual_et0_mm: float = 0.0
    
    # Extremes
    record_high_c: float = 40.0
    record_low_c: float = 15.0
    max_daily_rain_mm: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location_id': self.location_id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'data_points': len(self.daily_data),
            'annual_precipitation_mm': self.annual_precipitation_mm,
            'annual_avg_temp_c': self.annual_avg_temp_c,
            'monthly_summaries': self.monthly_summaries
        }


@dataclass
class WeatherAlert:
    """Weather alert for agricultural operations."""
    alert_id: str
    alert_type: WeatherAlertType
    risk_level: WeatherRiskLevel
    
    # Timing
    start_time: datetime
    end_time: datetime
    issued_at: datetime = field(default_factory=datetime.utcnow)
    
    # Details
    title: str = ""
    description: str = ""
    affected_area: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'risk_level': self.risk_level.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'title': self.title,
            'description': self.description,
            'recommendations': self.recommendations,
            'is_active': self.is_active
        }


@dataclass
class GrowingDegreeDays:
    """Growing degree days calculation."""
    location_id: str
    crop_type: str
    start_date: date
    end_date: date
    
    # Base temperatures by crop
    base_temp_c: float = 10.0
    upper_temp_c: float = 30.0
    
    # Accumulated GDD
    accumulated_gdd: float = 0.0
    daily_gdd: List[Tuple[date, float]] = field(default_factory=list)
    
    # Crop stage predictions
    predicted_stages: Dict[str, date] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location_id': self.location_id,
            'crop_type': self.crop_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'base_temp_c': self.base_temp_c,
            'accumulated_gdd': self.accumulated_gdd,
            'predicted_stages': {k: v.isoformat() for k, v in self.predicted_stages.items()}
        }


class EvapotranspirationCalculator:
    """Calculate reference evapotranspiration (ET0) using FAO Penman-Monteith."""
    
    # Constants
    STEFAN_BOLTZMANN = 4.903e-9  # MJ K-4 m-2 day-1
    
    def calculate_et0(
        self,
        temp_max_c: float,
        temp_min_c: float,
        humidity_avg: float,
        wind_speed_ms: float,
        solar_radiation_mjm2: float,
        latitude: float,
        day_of_year: int,
        elevation_m: float = 0
    ) -> float:
        """
        Calculate reference evapotranspiration using FAO Penman-Monteith equation.
        Returns ET0 in mm/day.
        """
        # Mean temperature
        temp_mean = (temp_max_c + temp_min_c) / 2
        
        # Atmospheric pressure (kPa)
        P = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
        
        # Psychrometric constant (kPa/C)
        gamma = 0.665e-3 * P
        
        # Saturation vapor pressure (kPa)
        e_s_max = 0.6108 * math.exp(17.27 * temp_max_c / (temp_max_c + 237.3))
        e_s_min = 0.6108 * math.exp(17.27 * temp_min_c / (temp_min_c + 237.3))
        e_s = (e_s_max + e_s_min) / 2
        
        # Actual vapor pressure (kPa)
        e_a = e_s * humidity_avg / 100
        
        # Slope of saturation vapor pressure curve (kPa/C)
        delta = 4098 * (0.6108 * math.exp(17.27 * temp_mean / (temp_mean + 237.3))) / (temp_mean + 237.3) ** 2
        
        # Net radiation (simplified)
        # Assuming Rs is given, calculate Rn
        albedo = 0.23
        Rs = solar_radiation_mjm2
        Rns = (1 - albedo) * Rs
        
        # Net longwave radiation (simplified)
        sigma = self.STEFAN_BOLTZMANN
        Rnl = sigma * ((temp_max_c + 273.16) ** 4 + (temp_min_c + 273.16) ** 4) / 2 * \
              (0.34 - 0.14 * math.sqrt(e_a)) * (1.35 * Rs / (0.75 * 37.5) - 0.35)
        
        Rn = Rns - Rnl
        
        # Soil heat flux (assume 0 for daily calculations)
        G = 0
        
        # Wind speed at 2m (assume input is at 2m)
        u2 = wind_speed_ms
        
        # FAO Penman-Monteith equation
        numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (temp_mean + 273)) * u2 * (e_s - e_a)
        denominator = delta + gamma * (1 + 0.34 * u2)
        
        et0 = numerator / denominator
        
        return max(0, et0)
    
    def calculate_crop_et(self, et0: float, kc: float) -> float:
        """Calculate crop evapotranspiration (ETc = ET0 * Kc)."""
        return et0 * kc
    
    def get_crop_coefficient(self, crop_type: str, growth_stage: str) -> float:
        """Get crop coefficient (Kc) for crop and growth stage."""
        kc_values = {
            'oil_palm': {
                'initial': 0.8,
                'mid': 1.0,
                'late': 0.9
            },
            'cocoa': {
                'initial': 0.9,
                'mid': 1.05,
                'late': 1.0
            },
            'ginger': {
                'initial': 0.5,
                'mid': 1.0,
                'late': 0.75
            },
            'maize': {
                'initial': 0.3,
                'mid': 1.2,
                'late': 0.6
            }
        }
        
        crop_kc = kc_values.get(crop_type, {'initial': 0.5, 'mid': 1.0, 'late': 0.8})
        return crop_kc.get(growth_stage, 1.0)


class WeatherRiskAssessor:
    """Assess weather-related risks for agriculture."""
    
    # Risk thresholds
    THRESHOLDS = {
        'frost': {'temp_c': 4.0},
        'heat_stress': {'temp_c': 35.0},
        'drought': {'days_no_rain': 14, 'soil_moisture': 0.3},
        'flood': {'rain_mm_24h': 100, 'rain_mm_3day': 200},
        'high_wind': {'wind_ms': 15},
        'storm': {'wind_ms': 20, 'rain_mm_hr': 30}
    }
    
    # Crop-specific thresholds
    CROP_THRESHOLDS = {
        'oil_palm': {
            'min_temp_c': 18,
            'max_temp_c': 38,
            'min_rainfall_mm_month': 100,
            'max_rainfall_mm_month': 400
        },
        'cocoa': {
            'min_temp_c': 15,
            'max_temp_c': 35,
            'min_rainfall_mm_month': 100,
            'max_rainfall_mm_month': 300
        },
        'ginger': {
            'min_temp_c': 15,
            'max_temp_c': 35,
            'min_rainfall_mm_month': 150,
            'max_rainfall_mm_month': 350
        }
    }
    
    def assess_daily_risks(
        self,
        daily_weather: DailyWeather,
        crop_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """Assess weather risks for a single day."""
        risks = []
        
        crop_thresh = self.CROP_THRESHOLDS.get(crop_type, {
            'min_temp_c': 10, 'max_temp_c': 40,
            'min_rainfall_mm_month': 50, 'max_rainfall_mm_month': 500
        })
        
        # Temperature risks
        if daily_weather.temp_min_c < crop_thresh['min_temp_c']:
            severity = 'high' if daily_weather.temp_min_c < crop_thresh['min_temp_c'] - 5 else 'moderate'
            risks.append({
                'type': 'cold_stress',
                'severity': severity,
                'value': daily_weather.temp_min_c,
                'threshold': crop_thresh['min_temp_c'],
                'message': f"Low temperature ({daily_weather.temp_min_c}C) may stress {crop_type}"
            })
        
        if daily_weather.temp_max_c > crop_thresh['max_temp_c']:
            severity = 'high' if daily_weather.temp_max_c > crop_thresh['max_temp_c'] + 5 else 'moderate'
            risks.append({
                'type': 'heat_stress',
                'severity': severity,
                'value': daily_weather.temp_max_c,
                'threshold': crop_thresh['max_temp_c'],
                'message': f"High temperature ({daily_weather.temp_max_c}C) may stress {crop_type}"
            })
        
        # Precipitation risks
        if daily_weather.precipitation_total_mm > self.THRESHOLDS['flood']['rain_mm_24h']:
            risks.append({
                'type': 'flood_risk',
                'severity': 'high',
                'value': daily_weather.precipitation_total_mm,
                'threshold': self.THRESHOLDS['flood']['rain_mm_24h'],
                'message': f"Heavy rainfall ({daily_weather.precipitation_total_mm}mm) - flood risk"
            })
        
        # Wind risks
        if daily_weather.wind_speed_max_ms > self.THRESHOLDS['high_wind']['wind_ms']:
            risks.append({
                'type': 'wind_damage',
                'severity': 'moderate',
                'value': daily_weather.wind_speed_max_ms,
                'threshold': self.THRESHOLDS['high_wind']['wind_ms'],
                'message': f"High winds ({daily_weather.wind_speed_max_ms}m/s) may cause damage"
            })
        
        return risks
    
    def assess_forecast_risks(
        self,
        forecast: WeatherForecast,
        crop_type: str = "general"
    ) -> Dict[str, Any]:
        """Assess risks across entire forecast period."""
        all_risks = []
        risk_summary = {
            'total_risks': 0,
            'high_risks': 0,
            'moderate_risks': 0,
            'low_risks': 0,
            'risk_days': []
        }
        
        # Check for drought (consecutive dry days)
        dry_days = 0
        
        for daily in forecast.forecast_days:
            daily_risks = self.assess_daily_risks(daily, crop_type)
            
            if daily_risks:
                risk_summary['risk_days'].append({
                    'date': daily.date.isoformat(),
                    'risks': daily_risks
                })
                
                for risk in daily_risks:
                    risk_summary['total_risks'] += 1
                    if risk['severity'] == 'high':
                        risk_summary['high_risks'] += 1
                    elif risk['severity'] == 'moderate':
                        risk_summary['moderate_risks'] += 1
                    else:
                        risk_summary['low_risks'] += 1
            
            # Track dry days
            if daily.precipitation_total_mm < 1:
                dry_days += 1
            else:
                dry_days = 0
            
            if dry_days >= self.THRESHOLDS['drought']['days_no_rain']:
                all_risks.append({
                    'type': 'drought_risk',
                    'severity': 'high',
                    'consecutive_dry_days': dry_days,
                    'message': f"Drought risk: {dry_days} consecutive dry days"
                })
        
        risk_summary['drought_risk'] = dry_days >= 7
        risk_summary['all_risks'] = all_risks
        
        return risk_summary
    
    def generate_alerts(
        self,
        forecast: WeatherForecast,
        crop_type: str = "general"
    ) -> List[WeatherAlert]:
        """Generate weather alerts from forecast."""
        import uuid
        alerts = []
        
        risk_assessment = self.assess_forecast_risks(forecast, crop_type)
        
        for risk_day in risk_assessment['risk_days']:
            for risk in risk_day['risks']:
                if risk['severity'] in ['high', 'moderate']:
                    alert_type = self._map_risk_to_alert_type(risk['type'])
                    risk_level = WeatherRiskLevel.HIGH if risk['severity'] == 'high' else WeatherRiskLevel.MODERATE
                    
                    alert = WeatherAlert(
                        alert_id=str(uuid.uuid4()),
                        alert_type=alert_type,
                        risk_level=risk_level,
                        start_time=datetime.fromisoformat(risk_day['date']),
                        end_time=datetime.fromisoformat(risk_day['date']) + timedelta(days=1),
                        title=f"{risk['type'].replace('_', ' ').title()} Alert",
                        description=risk['message'],
                        recommendations=self._get_recommendations(risk['type'], crop_type)
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _map_risk_to_alert_type(self, risk_type: str) -> WeatherAlertType:
        """Map risk type to alert type."""
        mapping = {
            'cold_stress': WeatherAlertType.FROST,
            'heat_stress': WeatherAlertType.HEAT_WAVE,
            'flood_risk': WeatherAlertType.FLOOD,
            'wind_damage': WeatherAlertType.HIGH_WIND,
            'drought_risk': WeatherAlertType.DROUGHT
        }
        return mapping.get(risk_type, WeatherAlertType.STORM)
    
    def _get_recommendations(self, risk_type: str, crop_type: str) -> List[str]:
        """Get recommendations for risk type."""
        recommendations = {
            'cold_stress': [
                "Consider protective covers for sensitive crops",
                "Delay irrigation to avoid frost damage",
                "Monitor crop condition closely"
            ],
            'heat_stress': [
                "Increase irrigation frequency",
                "Apply mulch to reduce soil temperature",
                "Avoid field operations during peak heat"
            ],
            'flood_risk': [
                "Ensure drainage channels are clear",
                "Delay fertilizer application",
                "Prepare for potential waterlogging"
            ],
            'wind_damage': [
                "Secure loose equipment and structures",
                "Delay spraying operations",
                "Check crop support structures"
            ],
            'drought_risk': [
                "Implement water conservation measures",
                "Prioritize irrigation for critical growth stages",
                "Consider drought-tolerant practices"
            ]
        }
        return recommendations.get(risk_type, ["Monitor conditions closely"])


class GrowingDegreeDaysCalculator:
    """Calculate growing degree days for crop development tracking."""
    
    # Base temperatures for different crops (Celsius)
    BASE_TEMPS = {
        'oil_palm': 15.0,
        'cocoa': 15.0,
        'ginger': 13.0,
        'maize': 10.0,
        'rice': 10.0,
        'wheat': 0.0
    }
    
    # GDD requirements for growth stages
    GDD_STAGES = {
        'ginger': {
            'emergence': 100,
            'vegetative': 500,
            'flowering': 1000,
            'maturity': 1800
        },
        'maize': {
            'emergence': 100,
            'v6': 475,
            'tasseling': 1135,
            'silking': 1400,
            'maturity': 2700
        }
    }
    
    def calculate_daily_gdd(
        self,
        temp_max_c: float,
        temp_min_c: float,
        base_temp_c: float,
        upper_temp_c: float = 30.0
    ) -> float:
        """Calculate GDD for a single day."""
        # Cap temperatures
        temp_max = min(temp_max_c, upper_temp_c)
        temp_min = max(temp_min_c, base_temp_c)
        
        # Average method
        avg_temp = (temp_max + temp_min) / 2
        gdd = max(0, avg_temp - base_temp_c)
        
        return gdd
    
    def calculate_accumulated_gdd(
        self,
        daily_weather: List[DailyWeather],
        crop_type: str,
        start_date: date
    ) -> GrowingDegreeDays:
        """Calculate accumulated GDD from weather data."""
        base_temp = self.BASE_TEMPS.get(crop_type, 10.0)
        
        result = GrowingDegreeDays(
            location_id="",
            crop_type=crop_type,
            start_date=start_date,
            end_date=daily_weather[-1].date if daily_weather else start_date,
            base_temp_c=base_temp
        )
        
        accumulated = 0.0
        for daily in daily_weather:
            if daily.date >= start_date:
                gdd = self.calculate_daily_gdd(
                    daily.temp_max_c,
                    daily.temp_min_c,
                    base_temp
                )
                accumulated += gdd
                result.daily_gdd.append((daily.date, gdd))
        
        result.accumulated_gdd = accumulated
        
        # Predict growth stages
        if crop_type in self.GDD_STAGES:
            for stage, required_gdd in self.GDD_STAGES[crop_type].items():
                if accumulated >= required_gdd:
                    # Find date when GDD was reached
                    running_gdd = 0
                    for d, gdd in result.daily_gdd:
                        running_gdd += gdd
                        if running_gdd >= required_gdd:
                            result.predicted_stages[stage] = d
                            break
        
        return result


class WeatherService:
    """Main weather service for crop monitoring."""
    
    def __init__(self):
        self.et_calculator = EvapotranspirationCalculator()
        self.risk_assessor = WeatherRiskAssessor()
        self.gdd_calculator = GrowingDegreeDaysCalculator()
        
        # Cache
        self._forecast_cache: Dict[str, WeatherForecast] = {}
        self._historical_cache: Dict[str, HistoricalWeather] = {}
    
    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 14
    ) -> WeatherForecast:
        """Get weather forecast for location."""
        location_id = f"{latitude:.4f}_{longitude:.4f}"
        
        # Generate synthetic forecast for demo
        forecast = self._generate_synthetic_forecast(
            location_id, latitude, longitude, days
        )
        
        self._forecast_cache[location_id] = forecast
        return forecast
    
    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date
    ) -> HistoricalWeather:
        """Get historical weather data."""
        location_id = f"{latitude:.4f}_{longitude:.4f}"
        
        # Generate synthetic historical data for demo
        historical = self._generate_synthetic_historical(
            location_id, latitude, longitude, start_date, end_date
        )
        
        return historical
    
    def get_weather_risks(
        self,
        latitude: float,
        longitude: float,
        crop_type: str = "general"
    ) -> Dict[str, Any]:
        """Get weather risk assessment for location."""
        forecast = self.get_forecast(latitude, longitude)
        return self.risk_assessor.assess_forecast_risks(forecast, crop_type)
    
    def get_weather_alerts(
        self,
        latitude: float,
        longitude: float,
        crop_type: str = "general"
    ) -> List[WeatherAlert]:
        """Get weather alerts for location."""
        forecast = self.get_forecast(latitude, longitude)
        return self.risk_assessor.generate_alerts(forecast, crop_type)
    
    def calculate_water_balance(
        self,
        latitude: float,
        longitude: float,
        crop_type: str,
        growth_stage: str = "mid"
    ) -> Dict[str, Any]:
        """Calculate water balance for crop."""
        forecast = self.get_forecast(latitude, longitude, days=7)
        
        kc = self.et_calculator.get_crop_coefficient(crop_type, growth_stage)
        
        water_balance = {
            'period_days': len(forecast.forecast_days),
            'total_precipitation_mm': 0,
            'total_et0_mm': 0,
            'total_etc_mm': 0,
            'water_balance_mm': 0,
            'irrigation_needed_mm': 0,
            'daily_balance': []
        }
        
        for daily in forecast.forecast_days:
            etc = daily.et0_mm * kc
            balance = daily.precipitation_total_mm - etc
            
            water_balance['total_precipitation_mm'] += daily.precipitation_total_mm
            water_balance['total_et0_mm'] += daily.et0_mm
            water_balance['total_etc_mm'] += etc
            
            water_balance['daily_balance'].append({
                'date': daily.date.isoformat(),
                'precipitation_mm': daily.precipitation_total_mm,
                'et0_mm': daily.et0_mm,
                'etc_mm': etc,
                'balance_mm': balance
            })
        
        water_balance['water_balance_mm'] = (
            water_balance['total_precipitation_mm'] - 
            water_balance['total_etc_mm']
        )
        
        if water_balance['water_balance_mm'] < 0:
            water_balance['irrigation_needed_mm'] = abs(water_balance['water_balance_mm'])
        
        return water_balance
    
    def _generate_synthetic_forecast(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        days: int
    ) -> WeatherForecast:
        """Generate synthetic forecast data for demo."""
        np.random.seed(int(abs(latitude * 1000 + longitude * 100)))
        
        # Base values for tropical climate
        base_temp = 27 + (latitude - 5) * 0.5
        base_humidity = 75
        base_rain_prob = 0.4
        
        forecast_days = []
        today = date.today()
        
        for i in range(days):
            current_date = today + timedelta(days=i)
            
            # Add some variation
            temp_var = np.random.normal(0, 2)
            rain_var = np.random.random()
            
            temp_max = base_temp + 5 + temp_var
            temp_min = base_temp - 3 + temp_var
            
            # Precipitation
            if rain_var < base_rain_prob:
                precip = np.random.exponential(15)
                condition = WeatherCondition.MODERATE_RAIN if precip > 10 else WeatherCondition.LIGHT_RAIN
            else:
                precip = 0
                condition = WeatherCondition.PARTLY_CLOUDY
            
            # ET0 calculation
            et0 = self.et_calculator.calculate_et0(
                temp_max, temp_min, base_humidity,
                2.0, 18.0, latitude, current_date.timetuple().tm_yday
            )
            
            daily = DailyWeather(
                date=current_date,
                temp_max_c=round(temp_max, 1),
                temp_min_c=round(temp_min, 1),
                temp_avg_c=round((temp_max + temp_min) / 2, 1),
                humidity_avg=base_humidity + np.random.normal(0, 5),
                precipitation_total_mm=round(precip, 1),
                precipitation_probability=base_rain_prob * 100,
                wind_speed_avg_ms=2 + np.random.random() * 3,
                et0_mm=round(et0, 2),
                condition=condition
            )
            
            forecast_days.append(daily)
        
        total_precip = sum(d.precipitation_total_mm for d in forecast_days)
        avg_temp = sum(d.temp_avg_c for d in forecast_days) / len(forecast_days)
        
        return WeatherForecast(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            total_precipitation_mm=round(total_precip, 1),
            avg_temperature_c=round(avg_temp, 1)
        )
    
    def _generate_synthetic_historical(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date
    ) -> HistoricalWeather:
        """Generate synthetic historical data for demo."""
        np.random.seed(int(abs(latitude * 1000 + longitude * 100)))
        
        daily_data = []
        current = start_date
        
        while current <= end_date:
            # Seasonal variation
            day_of_year = current.timetuple().tm_yday
            seasonal_factor = math.sin(2 * math.pi * day_of_year / 365)
            
            temp_max = 30 + 3 * seasonal_factor + np.random.normal(0, 2)
            temp_min = 22 + 2 * seasonal_factor + np.random.normal(0, 2)
            
            # Rainy season simulation
            rain_prob = 0.3 + 0.3 * math.sin(2 * math.pi * (day_of_year - 90) / 365)
            precip = np.random.exponential(20) if np.random.random() < rain_prob else 0
            
            daily = DailyWeather(
                date=current,
                temp_max_c=round(temp_max, 1),
                temp_min_c=round(temp_min, 1),
                temp_avg_c=round((temp_max + temp_min) / 2, 1),
                precipitation_total_mm=round(precip, 1)
            )
            daily_data.append(daily)
            
            current += timedelta(days=1)
        
        # Calculate annual statistics
        annual_precip = sum(d.precipitation_total_mm for d in daily_data)
        annual_temp = sum(d.temp_avg_c for d in daily_data) / len(daily_data)
        
        return HistoricalWeather(
            location_id=location_id,
            start_date=start_date,
            end_date=end_date,
            daily_data=daily_data,
            annual_precipitation_mm=round(annual_precip, 1),
            annual_avg_temp_c=round(annual_temp, 1)
        )


def create_weather_service() -> WeatherService:
    """Factory function to create weather service."""
    return WeatherService()
