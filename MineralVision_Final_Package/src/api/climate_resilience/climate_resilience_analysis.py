import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, shape
import rasterio
from rasterio.mask import mask
import requests
import json
from scipy.stats import percentileofscore
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClimateResilienceAnalysis:
    """
    A comprehensive system for analyzing and enhancing the climate resilience
    of mineral exploration and mining operations.
    """
    
    def __init__(self, data_dir: str = None):
        data_dir = data_dir or os.getenv(
            "MINERALVISION_DATA_DIR",
            os.path.join(os.path.expanduser("~"), ".mineralvision", "data", "climate"),
        )
        """
        Initialize the climate resilience analysis system.
        
        Args:
            data_dir: Directory for storing climate data and analysis results
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        self.climate_data_dir = os.path.join(data_dir, "climate_data")
        self.models_dir = os.path.join(data_dir, "models")
        self.results_dir = os.path.join(data_dir, "results")
        
        for directory in [self.climate_data_dir, self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize data sources
        self.data_sources = {
            "precipitation": None,
            "temperature": None,
            "extreme_events": None,
            "drought_indices": None,
            "sea_level": None,
            "water_resources": None
        }
        
        # Initialize models
        self.models = {
            "extreme_weather_prediction": None,
            "water_availability": None,
            "operational_impact": None,
            "carbon_footprint": None
        }
        
        logger.info("Climate Resilience Analysis system initialized")
    
    def load_climate_data(self, data_type: str, source: str, region: Dict[str, float], 
                         time_range: Tuple[str, str]) -> xr.Dataset:
        """
        Load climate data for a specific region and time range.
        
        Args:
            data_type: Type of climate data (precipitation, temperature, etc.)
            source: Source of the data (e.g., "CMIP6", "ERA5", "CHIRPS")
            region: Geographic region defined by {min_lon, max_lon, min_lat, max_lat}
            time_range: Start and end dates in ISO format
            
        Returns:
            xarray Dataset containing the requested climate data
        """
        logger.info(f"Loading {data_type} data from {source} for specified region")
        
        # Define file path for cached data
        cache_file = os.path.join(
            self.climate_data_dir, 
            f"{data_type}_{source}_{region['min_lon']}_{region['max_lon']}_{region['min_lat']}_{region['max_lat']}_{time_range[0]}_{time_range[1]}.nc"
        )
        
        # Check if data is already cached
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            return xr.open_dataset(cache_file)
        
        # If not cached, fetch from appropriate source
        if source == "ERA5":
            data = self._fetch_era5_data(data_type, region, time_range)
        elif source == "CMIP6":
            data = self._fetch_cmip6_data(data_type, region, time_range)
        elif source == "CHIRPS":
            data = self._fetch_chirps_data(region, time_range)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Cache the data for future use
        data.to_netcdf(cache_file)
        logger.info(f"Data cached to {cache_file}")
        
        # Update data sources dictionary
        self.data_sources[data_type] = data
        
        return data
    
    def _fetch_era5_data(self, data_type: str, region: Dict[str, float], 
                        time_range: Tuple[str, str]) -> xr.Dataset:
        """
        Fetch ERA5 reanalysis data.
        
        This is a placeholder for actual API calls to climate data services.
        In a production environment, this would use the CDS API or similar.
        """
        # Simulate data fetching
        logger.info(f"Fetching ERA5 {data_type} data")
        
        # Create time range
        start_date = pd.to_datetime(time_range[0])
        end_date = pd.to_datetime(time_range[1])
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create lat/lon grid
        lats = np.linspace(region['min_lat'], region['max_lat'], 20)
        lons = np.linspace(region['min_lon'], region['max_lon'], 20)
        
        # Create simulated data based on data_type
        if data_type == "precipitation":
            # Simulated daily precipitation in mm
            data_values = np.random.gamma(shape=2, scale=5, size=(len(dates), len(lats), len(lons)))
            data = xr.Dataset(
                data_vars={"precipitation": (("time", "latitude", "longitude"), data_values)},
                coords={"time": dates, "latitude": lats, "longitude": lons},
                attrs={"units": "mm/day", "source": "ERA5 (simulated)"}
            )
        elif data_type == "temperature":
            # Simulated daily temperature in Celsius
            base_temp = 15  # Base temperature
            annual_cycle = 10 * np.sin(2 * np.pi * (dates.dayofyear / 365))  # Annual cycle
            data_values = base_temp + annual_cycle[:, np.newaxis, np.newaxis] + np.random.normal(0, 2, size=(len(dates), len(lats), len(lons)))
            data = xr.Dataset(
                data_vars={"temperature": (("time", "latitude", "longitude"), data_values)},
                coords={"time": dates, "latitude": lats, "longitude": lons},
                attrs={"units": "°C", "source": "ERA5 (simulated)"}
            )
        else:
            raise ValueError(f"Unsupported data type for ERA5: {data_type}")
        
        return data
    
    def _fetch_cmip6_data(self, data_type: str, region: Dict[str, float], 
                         time_range: Tuple[str, str]) -> xr.Dataset:
        """
        Fetch CMIP6 climate model data.
        
        This is a placeholder for actual API calls to climate data services.
        """
        # Simulate data fetching
        logger.info(f"Fetching CMIP6 {data_type} data")
        
        # Create time range (monthly for climate projections)
        start_date = pd.to_datetime(time_range[0])
        end_date = pd.to_datetime(time_range[1])
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Create lat/lon grid
        lats = np.linspace(region['min_lat'], region['max_lat'], 10)
        lons = np.linspace(region['min_lon'], region['max_lon'], 10)
        
        # Create simulated data for multiple scenarios
        scenarios = ["ssp126", "ssp245", "ssp585"]  # Low, medium, and high emissions scenarios
        
        datasets = {}
        for scenario in scenarios:
            if data_type == "precipitation":
                # Simulated monthly precipitation with trend based on scenario
                trend_factor = 1.0
                if scenario == "ssp126":
                    trend_factor = 1.05  # 5% increase by end of period
                elif scenario == "ssp245":
                    trend_factor = 1.10  # 10% increase
                elif scenario == "ssp585":
                    trend_factor = 1.20  # 20% increase
                
                time_index = np.linspace(0, 1, len(dates))
                trend = 1 + (trend_factor - 1) * time_index
                
                base_values = np.random.gamma(shape=2, scale=50, size=(len(dates), len(lats), len(lons)))
                data_values = base_values * trend[:, np.newaxis, np.newaxis]
                
                datasets[scenario] = xr.Dataset(
                    data_vars={"precipitation": (("time", "latitude", "longitude"), data_values)},
                    coords={"time": dates, "latitude": lats, "longitude": lons},
                    attrs={"units": "mm/month", "source": f"CMIP6 {scenario} (simulated)"}
                )
            elif data_type == "temperature":
                # Simulated monthly temperature with warming trend based on scenario
                base_temp = 15
                if scenario == "ssp126":
                    warming = 1.0  # 1°C warming by end of period
                elif scenario == "ssp245":
                    warming = 2.0  # 2°C warming
                elif scenario == "ssp585":
                    warming = 4.0  # 4°C warming
                
                time_index = np.linspace(0, 1, len(dates))
                trend = warming * time_index
                
                annual_cycle = 10 * np.sin(2 * np.pi * (dates.month / 12))
                data_values = base_temp + annual_cycle[:, np.newaxis, np.newaxis] + trend[:, np.newaxis, np.newaxis] + np.random.normal(0, 1, size=(len(dates), len(lats), len(lons)))
                
                datasets[scenario] = xr.Dataset(
                    data_vars={"temperature": (("time", "latitude", "longitude"), data_values)},
                    coords={"time": dates, "latitude": lats, "longitude": lons},
                    attrs={"units": "°C", "source": f"CMIP6 {scenario} (simulated)"}
                )
        
        # Combine all scenarios into a single dataset with a scenario dimension
        combined_data = xr.concat([datasets[scenario] for scenario in scenarios], dim=pd.Index(scenarios, name="scenario"))
        
        return combined_data
    
    def _fetch_chirps_data(self, region: Dict[str, float], time_range: Tuple[str, str]) -> xr.Dataset:
        """
        Fetch CHIRPS precipitation data.
        
        This is a placeholder for actual API calls to the CHIRPS data service.
        """
        # Simulate data fetching
        logger.info("Fetching CHIRPS precipitation data")
        
        # Create time range
        start_date = pd.to_datetime(time_range[0])
        end_date = pd.to_datetime(time_range[1])
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create lat/lon grid (CHIRPS has 0.05° resolution)
        lats = np.linspace(region['min_lat'], region['max_lat'], 50)
        lons = np.linspace(region['min_lon'], region['max_lon'], 50)
        
        # Simulated daily precipitation in mm (CHIRPS is precipitation only)
        data_values = np.random.gamma(shape=1.5, scale=4, size=(len(dates), len(lats), len(lons)))
        data = xr.Dataset(
            data_vars={"precipitation": (("time", "latitude", "longitude"), data_values)},
            coords={"time": dates, "latitude": lats, "longitude": lons},
            attrs={"units": "mm/day", "source": "CHIRPS (simulated)"}
        )
        
        return data
    
    def analyze_extreme_weather_risk(self, exploration_area: Dict[str, Any], 
                                    climate_data: xr.Dataset, 
                                    thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze the risk of extreme weather events for a specific exploration area.
        
        Args:
            exploration_area: GeoJSON-like dictionary defining the exploration area
            climate_data: Climate dataset containing relevant variables
            thresholds: Dictionary of thresholds for different extreme events
            
        Returns:
            Dictionary containing risk assessment results
        """
        logger.info("Analyzing extreme weather risk")
        
        # Extract area geometry
        if 'geometry' in exploration_area:
            area_geom = shape(exploration_area['geometry'])
        else:
            # If no geometry provided, create a bounding box from coordinates
            area_geom = Polygon([
                (exploration_area['min_lon'], exploration_area['min_lat']),
                (exploration_area['max_lon'], exploration_area['min_lat']),
                (exploration_area['max_lon'], exploration_area['max_lat']),
                (exploration_area['min_lon'], exploration_area['max_lat']),
                (exploration_area['min_lon'], exploration_area['min_lat'])
            ])
        
        # Extract climate data for the area
        area_climate = self._extract_data_for_area(climate_data, area_geom)
        
        # Analyze different types of extreme events
        results = {}
        
        # Heavy precipitation events
        if 'precipitation' in area_climate and 'heavy_precipitation' in thresholds:
            threshold = thresholds['heavy_precipitation']
            precip = area_climate['precipitation'].values
            
            # Calculate frequency of heavy precipitation events
            heavy_precip_days = (precip > threshold).sum(axis=0)
            avg_heavy_precip_days = heavy_precip_days.mean()
            max_heavy_precip_days = heavy_precip_days.max()
            
            # Calculate return periods
            sorted_precip = np.sort(precip.flatten())
            return_periods = {}
            for return_period in [2, 5, 10, 25, 50, 100]:
                # Calculate the precipitation value for the given return period
                # This is a simplified approach; more sophisticated methods exist
                exceedance_prob = 1.0 / return_period
                index = int((1 - exceedance_prob) * len(sorted_precip))
                return_periods[return_period] = sorted_precip[index]
            
            results['heavy_precipitation'] = {
                'average_days_per_year': float(avg_heavy_precip_days),
                'maximum_days_per_year': float(max_heavy_precip_days),
                'return_periods': return_periods
            }
        
        # Extreme heat events
        if 'temperature' in area_climate and 'extreme_heat' in thresholds:
            threshold = thresholds['extreme_heat']
            temp = area_climate['temperature'].values
            
            # Calculate frequency of extreme heat events
            extreme_heat_days = (temp > threshold).sum(axis=0)
            avg_extreme_heat_days = extreme_heat_days.mean()
            max_extreme_heat_days = extreme_heat_days.max()
            
            # Calculate heat wave frequency (3+ consecutive days above threshold)
            heat_waves = 0
            consecutive_days = 0
            for day in range(temp.shape[0]):
                if (temp[day] > threshold).any():
                    consecutive_days += 1
                    if consecutive_days >= 3:
                        heat_waves += 1
                        consecutive_days = 0  # Reset to avoid double counting
                else:
                    consecutive_days = 0
            
            results['extreme_heat'] = {
                'average_days_per_year': float(avg_extreme_heat_days),
                'maximum_days_per_year': float(max_extreme_heat_days),
                'heat_waves_per_year': float(heat_waves / (temp.shape[0] / 365))
            }
        
        # Drought analysis
        if 'precipitation' in area_climate:
            # Calculate Standardized Precipitation Index (SPI)
            # This is a simplified version; real SPI calculation is more complex
            precip = area_climate['precipitation'].values
            monthly_precip = []
            
            # Aggregate to monthly
            dates = pd.DatetimeIndex(area_climate['precipitation'].time.values)
            for year in range(dates.year.min(), dates.year.max() + 1):
                for month in range(1, 13):
                    mask = (dates.year == year) & (dates.month == month)
                    if mask.any():
                        monthly_precip.append(precip[mask].mean())
            
            monthly_precip = np.array(monthly_precip)
            
            # Calculate 3-month SPI (simplified)
            spi_values = []
            for i in range(2, len(monthly_precip)):
                three_month = monthly_precip[i-2:i+1].mean()
                # Normalize based on historical distribution
                percentile = percentileofscore(monthly_precip, three_month)
                # Convert to standard normal distribution (simplified)
                spi = (percentile / 100 * 2) - 1
                spi_values.append(spi)
            
            spi_values = np.array(spi_values)
            
            # Calculate drought frequency
            moderate_drought = (spi_values <= -1.0).sum() / len(spi_values)
            severe_drought = (spi_values <= -1.5).sum() / len(spi_values)
            extreme_drought = (spi_values <= -2.0).sum() / len(spi_values)
            
            results['drought'] = {
                'moderate_drought_frequency': float(moderate_drought),
                'severe_drought_frequency': float(severe_drought),
                'extreme_drought_frequency': float(extreme_drought),
                'average_spi': float(spi_values.mean()),
                'min_spi': float(spi_values.min())
            }
        
        # Save results
        result_file = os.path.join(
            self.results_dir, 
            f"extreme_weather_risk_{exploration_area.get('name', 'area')}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Extreme weather risk analysis completed and saved to {result_file}")
        
        return results
    
    def _extract_data_for_area(self, climate_data: xr.Dataset, area_geometry: Polygon) -> xr.Dataset:
        """
        Extract climate data for a specific geographic area.
        
        Args:
            climate_data: xarray Dataset containing climate data
            area_geometry: Shapely Polygon defining the area of interest
            
        Returns:
            xarray Dataset with data extracted for the specified area
        """
        # Get bounds of the area
        minx, miny, maxx, maxy = area_geometry.bounds
        
        # Subset the data to the bounding box
        subset = climate_data.sel(
            latitude=slice(miny, maxy),
            longitude=slice(minx, maxx)
        )
        
        # For more precise masking, we would use rasterio's mask function
        # This is a simplified version that just uses the bounding box
        
        return subset
    
    def analyze_water_resource_impacts(self, exploration_area: Dict[str, Any], 
                                      climate_data: xr.Dataset,
                                      water_usage: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze the impacts of climate change on water resources for mining operations.
        
        Args:
            exploration_area: GeoJSON-like dictionary defining the exploration area
            climate_data: Climate dataset containing precipitation and temperature
            water_usage: Dictionary of water usage requirements for different operations
            
        Returns:
            Dictionary containing water resource impact assessment
        """
        logger.info("Analyzing water resource impacts")
        
        # Extract area geometry
        if 'geometry' in exploration_area:
            area_geom = shape(exploration_area['geometry'])
        else:
            # If no geometry provided, create a bounding box from coordinates
            area_geom = Polygon([
                (exploration_area['min_lon'], exploration_area['min_lat']),
                (exploration_area['max_lon'], exploration_area['min_lat']),
                (exploration_area['max_lon'], exploration_area['max_lat']),
                (exploration_area['min_lon'], exploration_area['max_lat']),
                (exploration_area['min_lon'], exploration_area['min_lat'])
            ])
        
        # Extract climate data for the area
        area_climate = self._extract_data_for_area(climate_data, area_geom)
        
        # Calculate water balance components
        results = {}
        
        # Precipitation input
        if 'precipitation' in area_climate:
            precip = area_climate['precipitation'].values
            annual_precip = precip.sum(axis=0) * 365 / precip.shape[0]  # Scale to annual
            avg_annual_precip = float(annual_precip.mean())
            
            results['precipitation'] = {
                'average_annual_mm': avg_annual_precip,
                'minimum_annual_mm': float(annual_precip.min()),
                'maximum_annual_mm': float(annual_precip.max())
            }
        
        # Potential evapotranspiration (simplified Thornthwaite method)
        if 'temperature' in area_climate:
            temp = area_climate['temperature'].values
            
            # Calculate monthly mean temperatures
            dates = pd.DatetimeIndex(area_climate['temperature'].time.values)
            monthly_temps = []
            
            for year in range(dates.year.min(), dates.year.max() + 1):
                for month in range(1, 13):
                    mask = (dates.year == year) & (dates.month == month)
                    if mask.any():
                        monthly_temps.append(temp[mask].mean())
            
            monthly_temps = np.array(monthly_temps)
            
            # Simplified PET calculation
            pet_values = []
            for t in monthly_temps:
                if t < 0:
                    pet = 0
                else:
                    pet = 16 * (10 * t / 5) ** 1.514  # Simplified Thornthwaite
                pet_values.append(pet)
            
            annual_pet = np.sum(pet_values) * 12 / len(pet_values)  # Scale to annual
            
            results['evapotranspiration'] = {
                'average_annual_mm': float(annual_pet)
            }
        
        # Water balance
        if 'precipitation' in results and 'evapotranspiration' in results:
            water_balance = avg_annual_precip - annual_pet
            
            results['water_balance'] = {
                'average_annual_mm': float(water_balance)
            }
        
        # Water stress analysis
        if 'water_balance' in results and water_usage:
            # Convert water usage from m³ to mm over the area
            area_m2 = area_geom.area * 111000 * 111000  # Approximate conversion from degrees to m²
            total_usage_m3 = sum(water_usage.values())
            usage_mm = total_usage_m3 / area_m2 * 1000  # Convert to mm
            
            water_stress_ratio = usage_mm / avg_annual_precip if avg_annual_precip > 0 else float('inf')
            
            stress_category = "Low"
            if water_stress_ratio > 0.2:
                stress_category = "Moderate"
            if water_stress_ratio > 0.4:
                stress_category = "High"
            if water_stress_ratio > 0.8:
                stress_category = "Extreme"
            
            results['water_stress'] = {
                'water_usage_mm': float(usage_mm),
                'water_stress_ratio': float(water_stress_ratio),
                'stress_category': stress_category
            }
        
        # Climate change impacts on water resources
        if 'scenario' in area_climate.dims:
            scenario_results = {}
            
            for scenario in area_climate.scenario.values:
                scenario_data = area_climate.sel(scenario=scenario)
                
                # Calculate changes in precipitation
                if 'precipitation' in scenario_data:
                    precip = scenario_data['precipitation'].values
                    
                    # Split into early and late periods to assess change
                    mid_point = precip.shape[0] // 2
                    early_period = precip[:mid_point]
                    late_period = precip[mid_point:]
                    
                    early_annual = early_period.sum(axis=0) * 365 / early_period.shape[0]
                    late_annual = late_period.sum(axis=0) * 365 / late_period.shape[0]
                    
                    percent_change = (late_annual.mean() - early_annual.mean()) / early_annual.mean() * 100
                    
                    scenario_results[str(scenario)] = {
                        'precipitation_change_percent': float(percent_change)
                    }
            
            results['climate_change_impacts'] = scenario_results
        
        # Save results
        result_file = os.path.join(
            self.results_dir, 
            f"water_resource_impacts_{exploration_area.get('name', 'area')}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Water resource impact analysis completed and saved to {result_file}")
        
        return results
    
    def analyze_operational_resilience(self, exploration_area: Dict[str, Any],
                                      climate_data: xr.Dataset,
                                      operational_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the resilience of mining operations to climate impacts.
        
        Args:
            exploration_area: GeoJSON-like dictionary defining the exploration area
            climate_data: Climate dataset containing relevant variables
            operational_params: Dictionary of operational parameters and thresholds
            
        Returns:
            Dictionary containing operational resilience assessment
        """
        logger.info("Analyzing operational resilience")
        
        # Extract climate data for the area
        if 'geometry' in exploration_area:
            area_geom = shape(exploration_area['geometry'])
        else:
            area_geom = Polygon([
                (exploration_area['min_lon'], exploration_area['min_lat']),
                (exploration_area['max_lon'], exploration_area['min_lat']),
                (exploration_area['max_lon'], exploration_area['max_lat']),
                (exploration_area['min_lon'], exploration_area['max_lat']),
                (exploration_area['min_lon'], exploration_area['min_lat'])
            ])
        
        area_climate = self._extract_data_for_area(climate_data, area_geom)
        
        results = {}
        
        # Analyze operational disruptions
        disruption_days = 0
        disruption_causes = {
            'heavy_precipitation': 0,
            'extreme_heat': 0,
            'high_wind': 0,
            'flooding': 0
        }
        
        # Check precipitation thresholds
        if 'precipitation' in area_climate and 'precipitation_threshold' in operational_params:
            threshold = operational_params['precipitation_threshold']
            precip = area_climate['precipitation'].values
            
            heavy_precip_days = (precip > threshold).any(axis=(1, 2)).sum()
            disruption_days += heavy_precip_days
            disruption_causes['heavy_precipitation'] = int(heavy_precip_days)
        
        # Check temperature thresholds
        if 'temperature' in area_climate and 'temperature_threshold' in operational_params:
            threshold = operational_params['temperature_threshold']
            temp = area_climate['temperature'].values
            
            extreme_heat_days = (temp > threshold).any(axis=(1, 2)).sum()
            disruption_days += extreme_heat_days
            disruption_causes['extreme_heat'] = int(extreme_heat_days)
        
        # Calculate operational efficiency
        total_days = area_climate.dims['time']
        operational_days = total_days - disruption_days
        operational_efficiency = operational_days / total_days
        
        results['operational_disruptions'] = {
            'total_disruption_days': int(disruption_days),
            'operational_days': int(operational_days),
            'operational_efficiency': float(operational_efficiency),
            'disruption_causes': disruption_causes
        }
        
        # Calculate financial impacts
        if 'daily_operation_cost' in operational_params and 'daily_revenue' in operational_params:
            daily_cost = operational_params['daily_operation_cost']
            daily_revenue = operational_params['daily_revenue']
            
            # Calculate baseline scenario (no disruptions)
            baseline_revenue = daily_revenue * total_days
            baseline_cost = daily_cost * total_days
            baseline_profit = baseline_revenue - baseline_cost
            
            # Calculate climate-impacted scenario
            climate_revenue = daily_revenue * operational_days
            climate_cost = daily_cost * total_days  # Costs continue during disruptions
            climate_profit = climate_revenue - climate_cost
            
            # Calculate impact
            revenue_loss = baseline_revenue - climate_revenue
            profit_loss = baseline_profit - climate_profit
            
            results['financial_impacts'] = {
                'baseline_profit': float(baseline_profit),
                'climate_impacted_profit': float(climate_profit),
                'profit_loss': float(profit_loss),
                'profit_loss_percent': float(profit_loss / baseline_profit * 100 if baseline_profit > 0 else 0)
            }
        
        # Analyze adaptation options
        if 'adaptation_options' in operational_params:
            adaptation_results = {}
            
            for option, params in operational_params['adaptation_options'].items():
                # Calculate effectiveness in reducing disruptions
                reduced_disruptions = disruption_days * params.get('disruption_reduction_factor', 0)
                new_disruption_days = disruption_days - reduced_disruptions
                new_operational_days = total_days - new_disruption_days
                new_efficiency = new_operational_days / total_days
                
                # Calculate financial impact with adaptation
                implementation_cost = params.get('implementation_cost', 0)
                annual_maintenance = params.get('annual_maintenance', 0)
                years = params.get('lifespan', 10)
                
                if 'daily_revenue' in operational_params:
                    new_revenue = daily_revenue * new_operational_days
                    new_cost = daily_cost * total_days + annual_maintenance
                    new_profit = new_revenue - new_cost
                    
                    profit_improvement = new_profit - climate_profit
                    roi = (profit_improvement * years - implementation_cost) / implementation_cost
                    
                    adaptation_results[option] = {
                        'reduced_disruption_days': float(reduced_disruptions),
                        'new_operational_efficiency': float(new_efficiency),
                        'implementation_cost': float(implementation_cost),
                        'annual_maintenance': float(annual_maintenance),
                        'profit_improvement': float(profit_improvement),
                        'roi': float(roi),
                        'payback_period_years': float(implementation_cost / profit_improvement if profit_improvement > 0 else float('inf'))
                    }
            
            results['adaptation_options'] = adaptation_results
        
        # Save results
        result_file = os.path.join(
            self.results_dir, 
            f"operational_resilience_{exploration_area.get('name', 'area')}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Operational resilience analysis completed and saved to {result_file}")
        
        return results
    
    def analyze_carbon_footprint(self, operational_data: Dict[str, Any],
                               reduction_scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the carbon footprint of mining operations and potential reduction strategies.
        
        Args:
            operational_data: Dictionary containing operational parameters and energy usage
            reduction_scenarios: List of carbon reduction scenarios to analyze
            
        Returns:
            Dictionary containing carbon footprint assessment and reduction potential
        """
        logger.info("Analyzing carbon footprint")
        
        results = {}
        
        # Calculate baseline emissions
        baseline_emissions = 0
        emissions_by_source = {}
        
        # Electricity usage
        if 'electricity_usage_kwh' in operational_data:
            electricity_kwh = operational_data['electricity_usage_kwh']
            grid_emission_factor = operational_data.get('grid_emission_factor', 0.5)  # tCO2e/MWh
            
            electricity_emissions = electricity_kwh / 1000 * grid_emission_factor
            baseline_emissions += electricity_emissions
            emissions_by_source['electricity'] = float(electricity_emissions)
        
        # Fuel usage
        if 'fuel_usage_liters' in operational_data:
            fuel_liters = operational_data['fuel_usage_liters']
            fuel_emission_factor = operational_data.get('fuel_emission_factor', 2.68)  # kgCO2e/L diesel
            
            fuel_emissions = fuel_liters * fuel_emission_factor / 1000  # Convert to tonnes
            baseline_emissions += fuel_emissions
            emissions_by_source['fuel'] = float(fuel_emissions)
        
        # Explosives
        if 'explosives_tonnes' in operational_data:
            explosives_tonnes = operational_data['explosives_tonnes']
            explosives_emission_factor = operational_data.get('explosives_emission_factor', 0.17)  # tCO2e/t
            
            explosives_emissions = explosives_tonnes * explosives_emission_factor
            baseline_emissions += explosives_emissions
            emissions_by_source['explosives'] = float(explosives_emissions)
        
        # Process emissions
        if 'process_emissions' in operational_data:
            process_emissions = operational_data['process_emissions']
            baseline_emissions += process_emissions
            emissions_by_source['process'] = float(process_emissions)
        
        # Calculate emissions intensity
        if 'production_tonnes' in operational_data:
            production_tonnes = operational_data['production_tonnes']
            emissions_intensity = baseline_emissions / production_tonnes
            
            results['emissions_intensity'] = {
                'tonnes_co2e_per_tonne_product': float(emissions_intensity)
            }
        
        results['baseline_emissions'] = {
            'total_tonnes_co2e': float(baseline_emissions),
            'emissions_by_source': emissions_by_source
        }
        
        # Analyze reduction scenarios
        if reduction_scenarios:
            scenario_results = {}
            
            for scenario in reduction_scenarios:
                scenario_name = scenario['name']
                scenario_emissions = baseline_emissions
                
                # Apply reduction measures
                for measure in scenario.get('measures', []):
                    source = measure['source']
                    reduction_percent = measure['reduction_percent']
                    
                    if source in emissions_by_source:
                        reduction = emissions_by_source[source] * reduction_percent / 100
                        scenario_emissions -= reduction
                
                # Calculate financial implications
                implementation_cost = scenario.get('implementation_cost', 0)
                annual_savings = scenario.get('annual_savings', 0)
                carbon_price = scenario.get('carbon_price', 0)
                
                carbon_savings = baseline_emissions - scenario_emissions
                carbon_cost_savings = carbon_savings * carbon_price
                total_annual_savings = annual_savings + carbon_cost_savings
                
                payback_period = implementation_cost / total_annual_savings if total_annual_savings > 0 else float('inf')
                
                scenario_results[scenario_name] = {
                    'total_emissions_tonnes_co2e': float(scenario_emissions),
                    'emissions_reduction_tonnes_co2e': float(carbon_savings),
                    'emissions_reduction_percent': float(carbon_savings / baseline_emissions * 100),
                    'implementation_cost': float(implementation_cost),
                    'annual_savings': float(annual_savings),
                    'carbon_cost_savings': float(carbon_cost_savings),
                    'total_annual_savings': float(total_annual_savings),
                    'payback_period_years': float(payback_period)
                }
            
            results['reduction_scenarios'] = scenario_results
        
        # Save results
        result_file = os.path.join(
            self.results_dir, 
            f"carbon_footprint_analysis_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Carbon footprint analysis completed and saved to {result_file}")
        
        return results
    
    def train_extreme_weather_prediction_model(self, climate_data: xr.Dataset, 
                                             target_variable: str,
                                             features: List[str],
                                             threshold: float) -> Dict[str, Any]:
        """
        Train a machine learning model to predict extreme weather events.
        
        Args:
            climate_data: Climate dataset containing historical data
            target_variable: Variable to predict (e.g., 'precipitation', 'temperature')
            features: List of features to use for prediction
            threshold: Threshold for defining extreme events
            
        Returns:
            Dictionary containing model performance metrics
        """
        logger.info(f"Training extreme weather prediction model for {target_variable}")
        
        # Prepare data
        X = []
        y = []
        
        # Extract target variable
        target = climate_data[target_variable].values
        
        # Define extreme events
        extreme_events = target > threshold
        
        # Create feature matrix
        for feature in features:
            if feature in climate_data:
                X.append(climate_data[feature].values.flatten())
        
        X = np.column_stack(X)
        y = extreme_events.flatten()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        model_file = os.path.join(
            self.models_dir,
            f"extreme_weather_{target_variable}_{datetime.now().strftime('%Y%m%d')}.joblib"
        )
        joblib.dump(model, model_file)
        
        # Store model in instance
        self.models['extreme_weather_prediction'] = model
        
        logger.info(f"Model trained and saved to {model_file}")
        
        return {
            'model_file': model_file,
            'mse': float(mse),
            'r2': float(r2),
            'feature_importance': {feature: float(imp) for feature, imp in zip(features, model.feature_importances_)}
        }
    
    def generate_climate_resilience_report(self, exploration_area: Dict[str, Any],
                                         analyses: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a comprehensive climate resilience report for a mining operation.
        
        Args:
            exploration_area: Information about the exploration area
            analyses: Dictionary containing results of various analyses
            
        Returns:
            Path to the generated report file
        """
        logger.info("Generating climate resilience report")
        
        area_name = exploration_area.get('name', 'Unnamed Area')
        
        # Create report directory
        report_dir = os.path.join(self.results_dir, f"report_{area_name}_{datetime.now().strftime('%Y%m%d')}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report content
        report_content = f"# Climate Resilience Report for {area_name}\n\n"
        report_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Executive Summary
        report_content += "## Executive Summary\n\n"
        
        if 'extreme_weather_risk' in analyses:
            risk_analysis = analyses['extreme_weather_risk']
            
            report_content += "### Extreme Weather Risk\n\n"
            
            if 'heavy_precipitation' in risk_analysis:
                heavy_precip = risk_analysis['heavy_precipitation']
                report_content += f"- Heavy Precipitation: {heavy_precip['average_days_per_year']:.1f} days/year\n"
            
            if 'extreme_heat' in risk_analysis:
                extreme_heat = risk_analysis['extreme_heat']
                report_content += f"- Extreme Heat: {extreme_heat['average_days_per_year']:.1f} days/year\n"
            
            if 'drought' in risk_analysis:
                drought = risk_analysis['drought']
                report_content += f"- Drought Risk: {drought['severe_drought_frequency']*100:.1f}% frequency\n"
            
            report_content += "\n"
        
        if 'water_resource_impacts' in analyses:
            water_analysis = analyses['water_resource_impacts']
            
            report_content += "### Water Resource Impacts\n\n"
            
            if 'water_stress' in water_analysis:
                water_stress = water_analysis['water_stress']
                report_content += f"- Water Stress Category: {water_stress['stress_category']}\n"
                report_content += f"- Water Stress Ratio: {water_stress['water_stress_ratio']:.2f}\n"
            
            if 'climate_change_impacts' in water_analysis:
                climate_impacts = water_analysis['climate_change_impacts']
                for scenario, impacts in climate_impacts.items():
                    if 'precipitation_change_percent' in impacts:
                        report_content += f"- Precipitation Change ({scenario}): {impacts['precipitation_change_percent']:.1f}%\n"
            
            report_content += "\n"
        
        if 'operational_resilience' in analyses:
            op_analysis = analyses['operational_resilience']
            
            report_content += "### Operational Resilience\n\n"
            
            if 'operational_disruptions' in op_analysis:
                disruptions = op_analysis['operational_disruptions']
                report_content += f"- Operational Efficiency: {disruptions['operational_efficiency']*100:.1f}%\n"
                report_content += f"- Disruption Days: {disruptions['total_disruption_days']} days/year\n"
            
            if 'financial_impacts' in op_analysis:
                financial = op_analysis['financial_impacts']
                report_content += f"- Profit Loss: {financial['profit_loss_percent']:.1f}%\n"
            
            report_content += "\n"
        
        if 'carbon_footprint' in analyses:
            carbon_analysis = analyses['carbon_footprint']
            
            report_content += "### Carbon Footprint\n\n"
            
            if 'baseline_emissions' in carbon_analysis:
                baseline = carbon_analysis['baseline_emissions']
                report_content += f"- Total Emissions: {baseline['total_tonnes_co2e']:.1f} tonnes CO2e\n"
            
            if 'emissions_intensity' in carbon_analysis:
                intensity = carbon_analysis['emissions_intensity']
                report_content += f"- Emissions Intensity: {intensity['tonnes_co2e_per_tonne_product']:.2f} tCO2e/t product\n"
            
            if 'reduction_scenarios' in carbon_analysis:
                best_scenario = None
                best_reduction = 0
                
                for name, scenario in carbon_analysis['reduction_scenarios'].items():
                    if scenario['emissions_reduction_percent'] > best_reduction:
                        best_reduction = scenario['emissions_reduction_percent']
                        best_scenario = name
                
                if best_scenario:
                    report_content += f"- Best Reduction Scenario: {best_scenario} ({best_reduction:.1f}%)\n"
            
            report_content += "\n"
        
        # Detailed Analysis
        report_content += "## Detailed Analysis\n\n"
        
        # Extreme Weather Risk
        if 'extreme_weather_risk' in analyses:
            risk_analysis = analyses['extreme_weather_risk']
            
            report_content += "### Extreme Weather Risk Analysis\n\n"
            
            if 'heavy_precipitation' in risk_analysis:
                heavy_precip = risk_analysis['heavy_precipitation']
                report_content += "#### Heavy Precipitation\n\n"
                report_content += f"- Average Days per Year: {heavy_precip['average_days_per_year']:.1f}\n"
                report_content += f"- Maximum Days per Year: {heavy_precip['maximum_days_per_year']:.1f}\n"
                
                if 'return_periods' in heavy_precip:
                    report_content += "- Return Periods:\n"
                    for period, value in heavy_precip['return_periods'].items():
                        report_content += f"  - {period}-year: {value:.1f} mm\n"
                
                report_content += "\n"
            
            if 'extreme_heat' in risk_analysis:
                extreme_heat = risk_analysis['extreme_heat']
                report_content += "#### Extreme Heat\n\n"
                report_content += f"- Average Days per Year: {extreme_heat['average_days_per_year']:.1f}\n"
                report_content += f"- Maximum Days per Year: {extreme_heat['maximum_days_per_year']:.1f}\n"
                report_content += f"- Heat Waves per Year: {extreme_heat['heat_waves_per_year']:.1f}\n"
                report_content += "\n"
            
            if 'drought' in risk_analysis:
                drought = risk_analysis['drought']
                report_content += "#### Drought Risk\n\n"
                report_content += f"- Moderate Drought Frequency: {drought['moderate_drought_frequency']*100:.1f}%\n"
                report_content += f"- Severe Drought Frequency: {drought['severe_drought_frequency']*100:.1f}%\n"
                report_content += f"- Extreme Drought Frequency: {drought['extreme_drought_frequency']*100:.1f}%\n"
                report_content += f"- Average SPI: {drought['average_spi']:.2f}\n"
                report_content += f"- Minimum SPI: {drought['min_spi']:.2f}\n"
                report_content += "\n"
        
        # Water Resource Impacts
        if 'water_resource_impacts' in analyses:
            water_analysis = analyses['water_resource_impacts']
            
            report_content += "### Water Resource Impact Analysis\n\n"
            
            if 'precipitation' in water_analysis:
                precip = water_analysis['precipitation']
                report_content += "#### Precipitation\n\n"
                report_content += f"- Average Annual: {precip['average_annual_mm']:.1f} mm\n"
                report_content += f"- Minimum Annual: {precip['minimum_annual_mm']:.1f} mm\n"
                report_content += f"- Maximum Annual: {precip['maximum_annual_mm']:.1f} mm\n"
                report_content += "\n"
            
            if 'evapotranspiration' in water_analysis:
                et = water_analysis['evapotranspiration']
                report_content += "#### Evapotranspiration\n\n"
                report_content += f"- Average Annual: {et['average_annual_mm']:.1f} mm\n"
                report_content += "\n"
            
            if 'water_balance' in water_analysis:
                balance = water_analysis['water_balance']
                report_content += "#### Water Balance\n\n"
                report_content += f"- Average Annual: {balance['average_annual_mm']:.1f} mm\n"
                report_content += "\n"
            
            if 'water_stress' in water_analysis:
                stress = water_analysis['water_stress']
                report_content += "#### Water Stress\n\n"
                report_content += f"- Water Usage: {stress['water_usage_mm']:.1f} mm\n"
                report_content += f"- Water Stress Ratio: {stress['water_stress_ratio']:.2f}\n"
                report_content += f"- Stress Category: {stress['stress_category']}\n"
                report_content += "\n"
            
            if 'climate_change_impacts' in water_analysis:
                impacts = water_analysis['climate_change_impacts']
                report_content += "#### Climate Change Impacts\n\n"
                
                for scenario, scenario_impacts in impacts.items():
                    report_content += f"**Scenario {scenario}:**\n"
                    if 'precipitation_change_percent' in scenario_impacts:
                        change = scenario_impacts['precipitation_change_percent']
                        direction = "increase" if change > 0 else "decrease"
                        report_content += f"- Precipitation: {abs(change):.1f}% {direction}\n"
                
                report_content += "\n"
        
        # Operational Resilience
        if 'operational_resilience' in analyses:
            op_analysis = analyses['operational_resilience']
            
            report_content += "### Operational Resilience Analysis\n\n"
            
            if 'operational_disruptions' in op_analysis:
                disruptions = op_analysis['operational_disruptions']
                report_content += "#### Operational Disruptions\n\n"
                report_content += f"- Total Disruption Days: {disruptions['total_disruption_days']} days\n"
                report_content += f"- Operational Days: {disruptions['operational_days']} days\n"
                report_content += f"- Operational Efficiency: {disruptions['operational_efficiency']*100:.1f}%\n"
                
                if 'disruption_causes' in disruptions:
                    causes = disruptions['disruption_causes']
                    report_content += "- Disruption Causes:\n"
                    for cause, days in causes.items():
                        report_content += f"  - {cause}: {days} days\n"
                
                report_content += "\n"
            
            if 'financial_impacts' in op_analysis:
                financial = op_analysis['financial_impacts']
                report_content += "#### Financial Impacts\n\n"
                report_content += f"- Baseline Profit: ${financial['baseline_profit']:,.2f}\n"
                report_content += f"- Climate-Impacted Profit: ${financial['climate_impacted_profit']:,.2f}\n"
                report_content += f"- Profit Loss: ${financial['profit_loss']:,.2f} ({financial['profit_loss_percent']:.1f}%)\n"
                report_content += "\n"
            
            if 'adaptation_options' in op_analysis:
                adaptations = op_analysis['adaptation_options']
                report_content += "#### Adaptation Options\n\n"
                
                for option, details in adaptations.items():
                    report_content += f"**{option}:**\n"
                    report_content += f"- Reduced Disruption Days: {details['reduced_disruption_days']:.1f}\n"
                    report_content += f"- New Operational Efficiency: {details['new_operational_efficiency']*100:.1f}%\n"
                    report_content += f"- Implementation Cost: ${details['implementation_cost']:,.2f}\n"
                    report_content += f"- Annual Maintenance: ${details['annual_maintenance']:,.2f}\n"
                    report_content += f"- Profit Improvement: ${details['profit_improvement']:,.2f}\n"
                    report_content += f"- ROI: {details['roi']*100:.1f}%\n"
                    report_content += f"- Payback Period: {details['payback_period_years']:.1f} years\n\n"
                
                report_content += "\n"
        
        # Carbon Footprint
        if 'carbon_footprint' in analyses:
            carbon_analysis = analyses['carbon_footprint']
            
            report_content += "### Carbon Footprint Analysis\n\n"
            
            if 'baseline_emissions' in carbon_analysis:
                baseline = carbon_analysis['baseline_emissions']
                report_content += "#### Baseline Emissions\n\n"
                report_content += f"- Total Emissions: {baseline['total_tonnes_co2e']:.1f} tonnes CO2e\n"
                
                if 'emissions_by_source' in baseline:
                    sources = baseline['emissions_by_source']
                    report_content += "- Emissions by Source:\n"
                    for source, emissions in sources.items():
                        report_content += f"  - {source}: {emissions:.1f} tonnes CO2e\n"
                
                report_content += "\n"
            
            if 'emissions_intensity' in carbon_analysis:
                intensity = carbon_analysis['emissions_intensity']
                report_content += "#### Emissions Intensity\n\n"
                report_content += f"- Tonnes CO2e per Tonne Product: {intensity['tonnes_co2e_per_tonne_product']:.2f}\n"
                report_content += "\n"
            
            if 'reduction_scenarios' in carbon_analysis:
                scenarios = carbon_analysis['reduction_scenarios']
                report_content += "#### Reduction Scenarios\n\n"
                
                for name, scenario in scenarios.items():
                    report_content += f"**{name}:**\n"
                    report_content += f"- Total Emissions: {scenario['total_emissions_tonnes_co2e']:.1f} tonnes CO2e\n"
                    report_content += f"- Emissions Reduction: {scenario['emissions_reduction_tonnes_co2e']:.1f} tonnes CO2e ({scenario['emissions_reduction_percent']:.1f}%)\n"
                    report_content += f"- Implementation Cost: ${scenario['implementation_cost']:,.2f}\n"
                    report_content += f"- Annual Savings: ${scenario['annual_savings']:,.2f}\n"
                    report_content += f"- Carbon Cost Savings: ${scenario['carbon_cost_savings']:,.2f}\n"
                    report_content += f"- Total Annual Savings: ${scenario['total_annual_savings']:,.2f}\n"
                    report_content += f"- Payback Period: {scenario['payback_period_years']:.1f} years\n\n"
                
                report_content += "\n"
        
        # Recommendations
        report_content += "## Recommendations\n\n"
        
        # Generate recommendations based on analyses
        recommendations = []
        
        # Extreme weather recommendations
        if 'extreme_weather_risk' in analyses:
            risk_analysis = analyses['extreme_weather_risk']
            
            if 'heavy_precipitation' in risk_analysis:
                heavy_precip = risk_analysis['heavy_precipitation']
                if heavy_precip['average_days_per_year'] > 10:
                    recommendations.append("Implement enhanced drainage systems to manage heavy precipitation events")
                    recommendations.append("Develop contingency plans for operations during heavy rainfall periods")
            
            if 'extreme_heat' in risk_analysis:
                extreme_heat = risk_analysis['extreme_heat']
                if extreme_heat['average_days_per_year'] > 10:
                    recommendations.append("Install cooling systems for sensitive equipment and worker areas")
                    recommendations.append("Adjust work schedules to avoid peak heat hours during extreme heat days")
            
            if 'drought' in risk_analysis:
                drought = risk_analysis['drought']
                if drought['severe_drought_frequency'] > 0.1:
                    recommendations.append("Develop water storage and recycling systems to enhance drought resilience")
                    recommendations.append("Implement water-efficient processing technologies")
        
        # Water resource recommendations
        if 'water_resource_impacts' in analyses:
            water_analysis = analyses['water_resource_impacts']
            
            if 'water_stress' in water_analysis:
                water_stress = water_analysis['water_stress']
                if water_stress['stress_category'] in ['High', 'Extreme']:
                    recommendations.append("Implement comprehensive water management plan with recycling and efficiency measures")
                    recommendations.append("Explore alternative water sources such as treated wastewater or desalination")
                elif water_stress['stress_category'] == 'Moderate':
                    recommendations.append("Implement water efficiency measures in processing operations")
                    recommendations.append("Monitor local water resources and develop early warning systems for shortages")
            
            if 'climate_change_impacts' in water_analysis:
                climate_impacts = water_analysis['climate_change_impacts']
                for scenario, impacts in climate_impacts.items():
                    if 'precipitation_change_percent' in impacts:
                        change = impacts['precipitation_change_percent']
                        if change < -5:
                            recommendations.append(f"Prepare for potential water scarcity under {scenario} scenario with {abs(change):.1f}% precipitation decrease")
        
        # Operational resilience recommendations
        if 'operational_resilience' in analyses:
            op_analysis = analyses['operational_resilience']
            
            if 'operational_disruptions' in op_analysis:
                disruptions = op_analysis['operational_disruptions']
                if disruptions['operational_efficiency'] < 0.9:
                    recommendations.append("Develop climate-resilient infrastructure to reduce operational disruptions")
                    recommendations.append("Implement flexible operational schedules to adapt to climate-related disruptions")
            
            if 'adaptation_options' in op_analysis:
                adaptations = op_analysis['adaptation_options']
                best_option = None
                best_roi = -float('inf')
                
                for option, details in adaptations.items():
                    if details['roi'] > best_roi and details['payback_period_years'] < 5:
                        best_roi = details['roi']
                        best_option = option
                
                if best_option:
                    recommendations.append(f"Prioritize implementation of '{best_option}' adaptation option with {best_roi*100:.1f}% ROI")
        
        # Carbon footprint recommendations
        if 'carbon_footprint' in analyses:
            carbon_analysis = analyses['carbon_footprint']
            
            if 'emissions_intensity' in carbon_analysis:
                intensity = carbon_analysis['emissions_intensity']
                if intensity['tonnes_co2e_per_tonne_product'] > 1.0:
                    recommendations.append("Implement energy efficiency measures to reduce carbon intensity")
                    recommendations.append("Explore renewable energy options for powering operations")
            
            if 'reduction_scenarios' in carbon_analysis:
                scenarios = carbon_analysis['reduction_scenarios']
                best_scenario = None
                best_payback = float('inf')
                
                for name, scenario in scenarios.items():
                    if scenario['payback_period_years'] < best_payback and scenario['emissions_reduction_percent'] > 20:
                        best_payback = scenario['payback_period_years']
                        best_scenario = name
                
                if best_scenario:
                    recommendations.append(f"Implement '{best_scenario}' carbon reduction strategy with {best_payback:.1f} year payback period")
        
        # Add recommendations to report
        for i, recommendation in enumerate(recommendations, 1):
            report_content += f"{i}. {recommendation}\n"
        
        report_content += "\n"
        
        # Conclusion
        report_content += "## Conclusion\n\n"
        report_content += "This climate resilience assessment has identified key risks and opportunities for enhancing the resilience of mining operations at this site. By implementing the recommended measures, the operation can reduce climate-related disruptions, minimize financial impacts, and contribute to sustainability goals.\n\n"
        report_content += "Regular monitoring and updating of this assessment is recommended as climate conditions evolve and new data becomes available.\n"
        
        # Write report to file
        report_file = os.path.join(report_dir, "climate_resilience_report.md")
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Generate visualizations
        self._generate_report_visualizations(analyses, report_dir)
        
        logger.info(f"Climate resilience report generated at {report_file}")
        
        return report_file
    
    def _generate_report_visualizations(self, analyses: Dict[str, Dict[str, Any]], report_dir: str):
        """
        Generate visualizations for the climate resilience report.
        
        Args:
            analyses: Dictionary containing results of various analyses
            report_dir: Directory to save visualizations
        """
        # Create visualizations directory
        viz_dir = os.path.join(report_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extreme Weather Risk visualization
        if 'extreme_weather_risk' in analyses:
            risk_analysis = analyses['extreme_weather_risk']
            
            # Create bar chart of extreme weather frequencies
            plt.figure(figsize=(10, 6))
            
            risk_types = []
            frequencies = []
            
            if 'heavy_precipitation' in risk_analysis:
                risk_types.append('Heavy Precipitation')
                frequencies.append(risk_analysis['heavy_precipitation']['average_days_per_year'])
            
            if 'extreme_heat' in risk_analysis:
                risk_types.append('Extreme Heat')
                frequencies.append(risk_analysis['extreme_heat']['average_days_per_year'])
            
            if 'drought' in risk_analysis and 'moderate_drought_frequency' in risk_analysis['drought']:
                risk_types.append('Moderate Drought')
                frequencies.append(risk_analysis['drought']['moderate_drought_frequency'] * 365)  # Convert to days per year
                
                risk_types.append('Severe Drought')
                frequencies.append(risk_analysis['drought']['severe_drought_frequency'] * 365)
                
                risk_types.append('Extreme Drought')
                frequencies.append(risk_analysis['drought']['extreme_drought_frequency'] * 365)
            
            if risk_types:
                plt.bar(risk_types, frequencies, color='#1f77b4')
                plt.ylabel('Days per Year')
                plt.title('Frequency of Extreme Weather Events')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                plt.savefig(os.path.join(viz_dir, 'extreme_weather_frequency.png'), dpi=300)
                plt.close()
        
        # Water Resource Impacts visualization
        if 'water_resource_impacts' in analyses:
            water_analysis = analyses['water_resource_impacts']
            
            # Create water balance visualization
            if all(k in water_analysis for k in ['precipitation', 'evapotranspiration', 'water_balance']):
                precip = water_analysis['precipitation']['average_annual_mm']
                et = water_analysis['evapotranspiration']['average_annual_mm']
                balance = water_analysis['water_balance']['average_annual_mm']
                
                plt.figure(figsize=(8, 6))
                
                components = ['Precipitation', 'Evapotranspiration', 'Water Balance']
                values = [precip, et, balance]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                plt.bar(components, values, color=colors)
                plt.ylabel('mm/year')
                plt.title('Annual Water Balance Components')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(viz_dir, 'water_balance.png'), dpi=300)
                plt.close()
            
            # Create climate change impact visualization
            if 'climate_change_impacts' in water_analysis:
                impacts = water_analysis['climate_change_impacts']
                
                scenarios = []
                changes = []
                
                for scenario, scenario_impacts in impacts.items():
                    if 'precipitation_change_percent' in scenario_impacts:
                        scenarios.append(scenario)
                        changes.append(scenario_impacts['precipitation_change_percent'])
                
                if scenarios:
                    plt.figure(figsize=(8, 6))
                    
                    bars = plt.bar(scenarios, changes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    
                    # Add colors based on positive/negative
                    for i, change in enumerate(changes):
                        if change < 0:
                            bars[i].set_color('#d62728')
                    
                    plt.ylabel('Precipitation Change (%)')
                    plt.title('Projected Precipitation Changes by Climate Scenario')
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, 'precipitation_change.png'), dpi=300)
                    plt.close()
        
        # Operational Resilience visualization
        if 'operational_resilience' in analyses:
            op_analysis = analyses['operational_resilience']
            
            # Create disruption causes pie chart
            if 'operational_disruptions' in op_analysis and 'disruption_causes' in op_analysis['operational_disruptions']:
                causes = op_analysis['operational_disruptions']['disruption_causes']
                
                # Filter out zero values
                labels = []
                sizes = []
                
                for cause, days in causes.items():
                    if days > 0:
                        labels.append(cause.replace('_', ' ').title())
                        sizes.append(days)
                
                if labels:
                    plt.figure(figsize=(8, 8))
                    
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
                    plt.axis('equal')
                    plt.title('Causes of Operational Disruptions')
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, 'disruption_causes.png'), dpi=300)
                    plt.close()
            
            # Create adaptation options comparison
            if 'adaptation_options' in op_analysis:
                adaptations = op_analysis['adaptation_options']
                
                options = []
                efficiency = []
                roi = []
                payback = []
                
                for option, details in adaptations.items():
                    options.append(option)
                    efficiency.append(details['new_operational_efficiency'] * 100)
                    roi.append(details['roi'] * 100)
                    payback.append(details['payback_period_years'])
                
                if options:
                    # Efficiency comparison
                    plt.figure(figsize=(10, 6))
                    
                    plt.bar(options, efficiency, color='#1f77b4')
                    if 'operational_disruptions' in op_analysis:
                        baseline = op_analysis['operational_disruptions']['operational_efficiency'] * 100
                        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.1f}%)')
                    
                    plt.ylabel('Operational Efficiency (%)')
                    plt.title('Operational Efficiency with Adaptation Options')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend()
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, 'adaptation_efficiency.png'), dpi=300)
                    plt.close()
                    
                    # ROI and payback comparison
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    color = 'tab:blue'
                    ax1.set_xlabel('Adaptation Option')
                    ax1.set_ylabel('ROI (%)', color=color)
                    ax1.bar(options, roi, color=color, alpha=0.7)
                    ax1.tick_params(axis='y', labelcolor=color)
                    
                    ax2 = ax1.twinx()
                    color = 'tab:red'
                    ax2.set_ylabel('Payback Period (years)', color=color)
                    ax2.plot(options, payback, 'o-', color=color)
                    ax2.tick_params(axis='y', labelcolor=color)
                    
                    plt.title('ROI and Payback Period of Adaptation Options')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, 'adaptation_roi_payback.png'), dpi=300)
                    plt.close()
        
        # Carbon Footprint visualization
        if 'carbon_footprint' in analyses:
            carbon_analysis = analyses['carbon_footprint']
            
            # Create emissions by source pie chart
            if 'baseline_emissions' in carbon_analysis and 'emissions_by_source' in carbon_analysis['baseline_emissions']:
                sources = carbon_analysis['baseline_emissions']['emissions_by_source']
                
                labels = []
                sizes = []
                
                for source, emissions in sources.items():
                    if emissions > 0:
                        labels.append(source.replace('_', ' ').title())
                        sizes.append(emissions)
                
                if labels:
                    plt.figure(figsize=(8, 8))
                    
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
                    plt.axis('equal')
                    plt.title('Carbon Emissions by Source')
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, 'emissions_by_source.png'), dpi=300)
                    plt.close()
            
            # Create reduction scenarios comparison
            if 'reduction_scenarios' in carbon_analysis:
                scenarios = carbon_analysis['reduction_scenarios']
                
                names = []
                emissions = []
                reductions = []
                
                # Add baseline
                if 'baseline_emissions' in carbon_analysis:
                    baseline = carbon_analysis['baseline_emissions']['total_tonnes_co2e']
                    names.append('Baseline')
                    emissions.append(baseline)
                    reductions.append(0)
                
                for name, scenario in scenarios.items():
                    names.append(name)
                    emissions.append(scenario['total_emissions_tonnes_co2e'])
                    reductions.append(scenario['emissions_reduction_percent'])
                
                if names:
                    # Emissions comparison
                    plt.figure(figsize=(10, 6))
                    
                    bars = plt.bar(names, emissions, color='#1f77b4')
                    if 'baseline_emissions' in carbon_analysis:
                        bars[0].set_color('#d62728')  # Highlight baseline in red
                    
                    plt.ylabel('Emissions (tonnes CO2e)')
                    plt.title('Carbon Emissions by Scenario')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, 'emissions_by_scenario.png'), dpi=300)
                    plt.close()
                    
                    # Reduction percentage comparison (excluding baseline)
                    if len(names) > 1:
                        plt.figure(figsize=(10, 6))
                        
                        plt.bar(names[1:], reductions[1:], color='#2ca02c')
                        
                        plt.ylabel('Emissions Reduction (%)')
                        plt.title('Carbon Emissions Reduction by Scenario')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        plt.savefig(os.path.join(viz_dir, 'emissions_reduction.png'), dpi=300)
                        plt.close()
        
        logger.info(f"Report visualizations generated in {viz_dir}")
