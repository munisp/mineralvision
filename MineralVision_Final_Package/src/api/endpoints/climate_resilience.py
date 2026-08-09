import os
import json
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from ..climate_resilience.climate_resilience_analysis import ClimateResilienceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/climate-resilience",
    tags=["climate-resilience"],
    responses={404: {"description": "Not found"}},
)

# Initialize climate resilience analysis system
climate_resilience = ClimateResilienceAnalysis()

# Models for request/response
class Region(BaseModel):
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

class ExplorationArea(BaseModel):
    name: str
    region: Region
    geometry: Optional[Dict[str, Any]] = None

class ClimateDataRequest(BaseModel):
    data_type: str
    source: str
    region: Region
    time_range: List[str]

class ExtremeWeatherRequest(BaseModel):
    exploration_area: ExplorationArea
    thresholds: Dict[str, float]

class WaterResourceRequest(BaseModel):
    exploration_area: ExplorationArea
    water_usage: Dict[str, float]

class OperationalParams(BaseModel):
    precipitation_threshold: Optional[float] = None
    temperature_threshold: Optional[float] = None
    daily_operation_cost: Optional[float] = None
    daily_revenue: Optional[float] = None
    adaptation_options: Optional[Dict[str, Dict[str, Any]]] = None

class OperationalResilienceRequest(BaseModel):
    exploration_area: ExplorationArea
    operational_params: OperationalParams

class CarbonFootprintRequest(BaseModel):
    operational_data: Dict[str, Any]
    reduction_scenarios: Optional[List[Dict[str, Any]]] = None

class ReportRequest(BaseModel):
    exploration_area: ExplorationArea
    analyses: Dict[str, Dict[str, Any]]

@router.post("/load-climate-data", response_model=Dict[str, Any])
async def load_climate_data(request: ClimateDataRequest):
    """
    Load climate data for a specific region and time range.
    """
    try:
        logger.info(f"Loading climate data: {request.data_type} from {request.source}")
        
        # Convert region to dictionary
        region_dict = request.region.dict()
        
        # Load climate data
        data = climate_resilience.load_climate_data(
            data_type=request.data_type,
            source=request.source,
            region=region_dict,
            time_range=tuple(request.time_range)
        )
        
        # Convert xarray dataset to dictionary for response
        # Note: This is a simplified representation
        response = {
            "data_type": request.data_type,
            "source": request.source,
            "dimensions": {dim: len(data[dim]) for dim in data.dims},
            "variables": list(data.data_vars),
            "time_range": request.time_range,
            "region": region_dict,
            "status": "success",
            "message": f"Successfully loaded {request.data_type} data from {request.source}"
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error loading climate data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading climate data: {str(e)}")

@router.post("/analyze-extreme-weather", response_model=Dict[str, Any])
async def analyze_extreme_weather(request: ExtremeWeatherRequest):
    """
    Analyze the risk of extreme weather events for a specific exploration area.
    """
    try:
        logger.info(f"Analyzing extreme weather risk for {request.exploration_area.name}")
        
        # Check if climate data is available
        if not climate_resilience.data_sources.get("precipitation") and not climate_resilience.data_sources.get("temperature"):
            raise HTTPException(status_code=400, detail="Climate data not loaded. Please load climate data first.")
        
        # Use available climate data
        climate_data = None
        for data_type in ["precipitation", "temperature"]:
            if climate_resilience.data_sources.get(data_type):
                climate_data = climate_resilience.data_sources[data_type]
                break
        
        if not climate_data:
            raise HTTPException(status_code=400, detail="No suitable climate data available.")
        
        # Convert exploration area to dictionary
        exploration_area_dict = request.exploration_area.dict()
        
        # Analyze extreme weather risk
        results = climate_resilience.analyze_extreme_weather_risk(
            exploration_area=exploration_area_dict,
            climate_data=climate_data,
            thresholds=request.thresholds
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing extreme weather risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing extreme weather risk: {str(e)}")

@router.post("/analyze-water-resources", response_model=Dict[str, Any])
async def analyze_water_resources(request: WaterResourceRequest):
    """
    Analyze the impacts of climate change on water resources for mining operations.
    """
    try:
        logger.info(f"Analyzing water resource impacts for {request.exploration_area.name}")
        
        # Check if climate data is available
        if not climate_resilience.data_sources.get("precipitation") and not climate_resilience.data_sources.get("temperature"):
            raise HTTPException(status_code=400, detail="Climate data not loaded. Please load climate data first.")
        
        # Use available climate data
        climate_data = None
        for data_type in ["precipitation", "temperature"]:
            if climate_resilience.data_sources.get(data_type):
                climate_data = climate_resilience.data_sources[data_type]
                break
        
        if not climate_data:
            raise HTTPException(status_code=400, detail="No suitable climate data available.")
        
        # Convert exploration area to dictionary
        exploration_area_dict = request.exploration_area.dict()
        
        # Analyze water resource impacts
        results = climate_resilience.analyze_water_resource_impacts(
            exploration_area=exploration_area_dict,
            climate_data=climate_data,
            water_usage=request.water_usage
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing water resource impacts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing water resource impacts: {str(e)}")

@router.post("/analyze-operational-resilience", response_model=Dict[str, Any])
async def analyze_operational_resilience(request: OperationalResilienceRequest):
    """
    Analyze the resilience of mining operations to climate impacts.
    """
    try:
        logger.info(f"Analyzing operational resilience for {request.exploration_area.name}")
        
        # Check if climate data is available
        if not climate_resilience.data_sources.get("precipitation") and not climate_resilience.data_sources.get("temperature"):
            raise HTTPException(status_code=400, detail="Climate data not loaded. Please load climate data first.")
        
        # Use available climate data
        climate_data = None
        for data_type in ["precipitation", "temperature"]:
            if climate_resilience.data_sources.get(data_type):
                climate_data = climate_resilience.data_sources[data_type]
                break
        
        if not climate_data:
            raise HTTPException(status_code=400, detail="No suitable climate data available.")
        
        # Convert exploration area to dictionary
        exploration_area_dict = request.exploration_area.dict()
        
        # Convert operational params to dictionary
        operational_params_dict = request.operational_params.dict(exclude_none=True)
        
        # Analyze operational resilience
        results = climate_resilience.analyze_operational_resilience(
            exploration_area=exploration_area_dict,
            climate_data=climate_data,
            operational_params=operational_params_dict
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing operational resilience: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing operational resilience: {str(e)}")

@router.post("/analyze-carbon-footprint", response_model=Dict[str, Any])
async def analyze_carbon_footprint(request: CarbonFootprintRequest):
    """
    Analyze the carbon footprint of mining operations and potential reduction strategies.
    """
    try:
        logger.info("Analyzing carbon footprint")
        
        # Analyze carbon footprint
        results = climate_resilience.analyze_carbon_footprint(
            operational_data=request.operational_data,
            reduction_scenarios=request.reduction_scenarios
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing carbon footprint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing carbon footprint: {str(e)}")

@router.post("/generate-report", response_model=Dict[str, Any])
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive climate resilience report for a mining operation.
    """
    try:
        logger.info(f"Generating climate resilience report for {request.exploration_area.name}")
        
        # Convert exploration area to dictionary
        exploration_area_dict = request.exploration_area.dict()
        
        # Generate report
        report_file = climate_resilience.generate_climate_resilience_report(
            exploration_area=exploration_area_dict,
            analyses=request.analyses
        )
        
        # Get report directory
        report_dir = os.path.dirname(report_file)
        
        # Get visualization files
        viz_dir = os.path.join(report_dir, "visualizations")
        viz_files = []
        if os.path.exists(viz_dir):
            viz_files = [os.path.join(viz_dir, f) for f in os.listdir(viz_dir) if f.endswith('.png')]
        
        return {
            "report_file": report_file,
            "visualization_files": viz_files,
            "status": "success",
            "message": f"Successfully generated climate resilience report for {request.exploration_area.name}"
        }
    
    except Exception as e:
        logger.error(f"Error generating climate resilience report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating climate resilience report: {str(e)}")

@router.post("/upload-climate-data", response_model=Dict[str, Any])
async def upload_climate_data(
    background_tasks: BackgroundTasks,
    data_type: str = Form(...),
    file: UploadFile = File(...),
    region_min_lon: float = Form(...),
    region_max_lon: float = Form(...),
    region_min_lat: float = Form(...),
    region_max_lat: float = Form(...),
):
    """
    Upload custom climate data file for analysis.
    """
    try:
        logger.info(f"Uploading climate data file: {file.filename}")
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(climate_resilience.data_dir, "uploads")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Define region
        region = {
            "min_lon": region_min_lon,
            "max_lon": region_max_lon,
            "min_lat": region_min_lat,
            "max_lat": region_max_lat
        }
        
        # Process the uploaded file based on file extension
        import xarray as xr
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        try:
            if file_ext in ['.nc', '.nc4', '.netcdf']:
                data = xr.open_dataset(file_path)
                climate_resilience.data_sources[data_type] = data
                logger.info(f"Loaded NetCDF climate data: {data_type}")
            elif file_ext == '.csv':
                import pandas as pd
                import numpy as np
                df = pd.read_csv(file_path)
                if 'time' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    value_cols = [c for c in df.columns if c not in ['time', 'latitude', 'longitude']]
                    if value_cols:
                        data = xr.Dataset.from_dataframe(df.set_index(['time', 'latitude', 'longitude']))
                        climate_resilience.data_sources[data_type] = data
                        logger.info(f"Loaded CSV climate data: {data_type}")
            elif file_ext in ['.tif', '.tiff', '.geotiff']:
                try:
                    import rioxarray
                    data = rioxarray.open_rasterio(file_path)
                    climate_resilience.data_sources[data_type] = data.to_dataset(name=data_type)
                    logger.info(f"Loaded GeoTIFF climate data: {data_type}")
                except ImportError:
                    logger.warning("rioxarray not available for GeoTIFF processing")
        except Exception as process_error:
            logger.warning(f"Could not process file as climate data: {process_error}")
        
        return {
            "file_path": file_path,
            "data_type": data_type,
            "region": region,
            "status": "success",
            "message": f"Successfully uploaded climate data file: {file.filename}"
        }
    
    except Exception as e:
        logger.error(f"Error uploading climate data file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading climate data file: {str(e)}")
