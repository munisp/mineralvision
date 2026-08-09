"""
API endpoints for sensor fusion functionality.

This module provides FastAPI endpoints for the sensor fusion framework,
allowing clients to upload, process, and retrieve fused sensor data.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Path, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional, Union
import os
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import shutil
from pydantic import BaseModel, Field

# Heavy geospatial dependencies (xarray, rasterio, pyproj) are optional.
# The API boots without them; sensor-fusion endpoints degrade with HTTP 503.
try:
    import xarray as xr
    from ..sensor_fusion.core import SensorData, SensorType, DataDimension
    from ..sensor_fusion.hyperspectral_adapter import HyperspectralDataAdapter
    from ..sensor_fusion.lidar_adapter import LidarDataAdapter
    from ..sensor_fusion.magnetometry_adapter import MagnetometryDataAdapter
    from ..sensor_fusion.fusion_algorithms import WeightedAverageFusion, BayesianFusion
    SENSOR_FUSION_AVAILABLE = True
    _SENSOR_FUSION_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - depends on optional deps
    xr = None
    SensorData = SensorType = DataDimension = None
    HyperspectralDataAdapter = LidarDataAdapter = MagnetometryDataAdapter = None
    WeightedAverageFusion = BayesianFusion = None
    SENSOR_FUSION_AVAILABLE = False
    _SENSOR_FUSION_ERROR = str(exc)

# Create router
router = APIRouter(
    prefix="/sensor-fusion",
    tags=["sensor-fusion"],
    responses={404: {"description": "Not found"}},
)

# Initialize adapters (None when heavy geospatial deps are not installed)
hyperspectral_adapter = HyperspectralDataAdapter() if SENSOR_FUSION_AVAILABLE else None
lidar_adapter = LidarDataAdapter() if SENSOR_FUSION_AVAILABLE else None
magnetometry_adapter = MagnetometryDataAdapter() if SENSOR_FUSION_AVAILABLE else None

# Initialize fusion algorithms
weighted_fusion = WeightedAverageFusion() if SENSOR_FUSION_AVAILABLE else None
bayesian_fusion = BayesianFusion() if SENSOR_FUSION_AVAILABLE else None


def _require_sensor_fusion():
    """Raise HTTP 503 when the sensor-fusion stack is not installed."""
    if not SENSOR_FUSION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sensor fusion is unavailable: optional geospatial dependencies "
                f"are not installed ({_SENSOR_FUSION_ERROR}). "
                "Install the optional geospatial requirements to enable this feature."
            )
        )

# Runtime storage for API operations
# In a production environment, this would be replaced with a database
sensor_data_storage = {}
fusion_results_storage = {}

# Data models
class SensorDataInfo(BaseModel):
    """Information about a sensor data object."""
    data_id: str
    sensor_type: str
    dimensions: List[str]
    metadata: Dict[str, Any]
    timestamp: str
    file_name: str

class FusionRequest(BaseModel):
    """Request for sensor fusion."""
    sensor_data_ids: List[str] = Field(..., description="List of sensor data IDs to fuse")
    algorithm: str = Field(..., description="Fusion algorithm to use (weighted_average or bayesian)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")
    name: Optional[str] = Field(None, description="Optional name for the fusion result")

class FusionResult(BaseModel):
    """Information about a fusion result."""
    result_id: str
    name: Optional[str]
    algorithm: str
    source_data_ids: List[str]
    metadata: Dict[str, Any]
    timestamp: str

@router.post("/upload/{sensor_type}", response_model=SensorDataInfo)
async def upload_sensor_data(
    sensor_type: str = Path(..., description="Type of sensor data (hyperspectral, lidar, magnetometry, etc.)"),
    file: UploadFile = File(...),
    metadata: str = Form(None),
    coordinate_system: str = Form(None),
    _: None = Depends(_require_sensor_fusion)
):
    """
    Upload sensor data file.
    
    Args:
        sensor_type: Type of sensor data
        file: Data file
        metadata: Optional JSON metadata
        coordinate_system: Optional coordinate system
        
    Returns:
        Information about the uploaded sensor data
    """
    # Generate unique ID for the data
    data_id = str(uuid.uuid4())
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Add coordinate system to metadata if provided
        if coordinate_system:
            metadata_dict["crs"] = coordinate_system
        
        # Load data using appropriate adapter
        if sensor_type.lower() == "hyperspectral":
            adapter = hyperspectral_adapter
            sensor_type_enum = SensorType.HYPERSPECTRAL
        elif sensor_type.lower() == "lidar":
            adapter = lidar_adapter
            sensor_type_enum = SensorType.LIDAR
        elif sensor_type.lower() == "magnetometry":
            adapter = magnetometry_adapter
            sensor_type_enum = SensorType.MAGNETOMETRY
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported sensor type: {sensor_type}")
        
        # Load data
        try:
            sensor_data = adapter.load(temp_file_path, **metadata_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")
        
        # Set data ID
        sensor_data.data_id = data_id
        
        # Store data
        sensor_data_storage[data_id] = sensor_data
        
        # Create response
        response = SensorDataInfo(
            data_id=data_id,
            sensor_type=sensor_data.sensor_type.value,
            dimensions=[dim.value for dim in sensor_data.dimensions],
            metadata=sensor_data.metadata,
            timestamp=sensor_data.timestamp.isoformat() if sensor_data.timestamp else datetime.now().isoformat(),
            file_name=file.filename
        )
        
        return response
        
    finally:
        # Clean up temporary file
        shutil.rmtree(temp_dir)

@router.get("/data", response_model=List[SensorDataInfo])
async def list_sensor_data():
    """
    List all available sensor data.
    
    Returns:
        List of sensor data information
    """
    result = []
    for data_id, sensor_data in sensor_data_storage.items():
        result.append(SensorDataInfo(
            data_id=data_id,
            sensor_type=sensor_data.sensor_type.value,
            dimensions=[dim.value for dim in sensor_data.dimensions],
            metadata=sensor_data.metadata,
            timestamp=sensor_data.timestamp.isoformat() if sensor_data.timestamp else datetime.now().isoformat(),
            file_name=sensor_data.metadata.get("file_name", "unknown")
        ))
    
    return result

@router.get("/data/{data_id}", response_model=SensorDataInfo)
async def get_sensor_data_info(data_id: str):
    """
    Get information about a specific sensor data.
    
    Args:
        data_id: ID of the sensor data
        
    Returns:
        Sensor data information
    """
    if data_id not in sensor_data_storage:
        raise HTTPException(status_code=404, detail=f"Sensor data not found: {data_id}")
    
    sensor_data = sensor_data_storage[data_id]
    
    return SensorDataInfo(
        data_id=data_id,
        sensor_type=sensor_data.sensor_type.value,
        dimensions=[dim.value for dim in sensor_data.dimensions],
        metadata=sensor_data.metadata,
        timestamp=sensor_data.timestamp.isoformat() if sensor_data.timestamp else datetime.now().isoformat(),
        file_name=sensor_data.metadata.get("file_name", "unknown")
    )

@router.post("/preprocess/{data_id}", response_model=SensorDataInfo)
async def preprocess_sensor_data(
    data_id: str,
    parameters: Dict[str, Any],
    _: None = Depends(_require_sensor_fusion)
):
    """
    Preprocess sensor data.
    
    Args:
        data_id: ID of the sensor data
        parameters: Preprocessing parameters
        
    Returns:
        Information about the preprocessed sensor data
    """
    if data_id not in sensor_data_storage:
        raise HTTPException(status_code=404, detail=f"Sensor data not found: {data_id}")
    
    sensor_data = sensor_data_storage[data_id]
    
    # Get appropriate adapter
    if sensor_data.sensor_type == SensorType.HYPERSPECTRAL:
        adapter = hyperspectral_adapter
    elif sensor_data.sensor_type == SensorType.LIDAR:
        adapter = lidar_adapter
    elif sensor_data.sensor_type == SensorType.MAGNETOMETRY:
        adapter = magnetometry_adapter
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported sensor type: {sensor_data.sensor_type}")
    
    # Preprocess data
    try:
        preprocessed_data = adapter.preprocess(sensor_data, **parameters)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")
    
    # Generate new ID for preprocessed data
    new_data_id = str(uuid.uuid4())
    preprocessed_data.data_id = new_data_id
    
    # Store preprocessed data
    sensor_data_storage[new_data_id] = preprocessed_data
    
    # Create response
    response = SensorDataInfo(
        data_id=new_data_id,
        sensor_type=preprocessed_data.sensor_type.value,
        dimensions=[dim.value for dim in preprocessed_data.dimensions],
        metadata=preprocessed_data.metadata,
        timestamp=preprocessed_data.timestamp.isoformat() if preprocessed_data.timestamp else datetime.now().isoformat(),
        file_name=sensor_data.metadata.get("file_name", "unknown") + "_preprocessed"
    )
    
    return response

@router.post("/fuse", response_model=FusionResult)
async def fuse_sensor_data(
    fusion_request: FusionRequest,
    _: None = Depends(_require_sensor_fusion)
):
    """
    Fuse multiple sensor data.
    
    Args:
        fusion_request: Fusion request
        
    Returns:
        Information about the fusion result
    """
    # Check if all sensor data IDs exist
    for data_id in fusion_request.sensor_data_ids:
        if data_id not in sensor_data_storage:
            raise HTTPException(status_code=404, detail=f"Sensor data not found: {data_id}")
    
    # Get sensor data
    sensor_data_list = [sensor_data_storage[data_id] for data_id in fusion_request.sensor_data_ids]
    
    # Get fusion algorithm
    if fusion_request.algorithm.lower() == "weighted_average":
        algorithm = weighted_fusion
    elif fusion_request.algorithm.lower() == "bayesian":
        algorithm = bayesian_fusion
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported fusion algorithm: {fusion_request.algorithm}")
    
    # Perform fusion
    try:
        fused_data = algorithm.fuse(sensor_data_list, **fusion_request.parameters)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing fusion: {str(e)}")
    
    # Generate ID for fusion result
    result_id = str(uuid.uuid4())
    
    # Store fusion result
    fusion_results_storage[result_id] = fused_data
    
    # Create response
    response = FusionResult(
        result_id=result_id,
        name=fusion_request.name,
        algorithm=fusion_request.algorithm,
        source_data_ids=fusion_request.sensor_data_ids,
        metadata=fused_data.metadata,
        timestamp=fused_data.timestamp.isoformat() if fused_data.timestamp else datetime.now().isoformat()
    )
    
    return response

@router.get("/fusion", response_model=List[FusionResult])
async def list_fusion_results():
    """
    List all available fusion results.
    
    Returns:
        List of fusion result information
    """
    result = []
    for result_id, fused_data in fusion_results_storage.items():
        result.append(FusionResult(
            result_id=result_id,
            name=fused_data.metadata.get("name"),
            algorithm=fused_data.metadata.get("fusion_algorithm", "unknown"),
            source_data_ids=fused_data.metadata.get("source_data_ids", []),
            metadata=fused_data.metadata,
            timestamp=fused_data.timestamp.isoformat() if fused_data.timestamp else datetime.now().isoformat()
        ))
    
    return result

@router.get("/fusion/{result_id}", response_model=FusionResult)
async def get_fusion_result_info(result_id: str):
    """
    Get information about a specific fusion result.
    
    Args:
        result_id: ID of the fusion result
        
    Returns:
        Fusion result information
    """
    if result_id not in fusion_results_storage:
        raise HTTPException(status_code=404, detail=f"Fusion result not found: {result_id}")
    
    fused_data = fusion_results_storage[result_id]
    
    return FusionResult(
        result_id=result_id,
        name=fused_data.metadata.get("name"),
        algorithm=fused_data.metadata.get("fusion_algorithm", "unknown"),
        source_data_ids=fused_data.metadata.get("source_data_ids", []),
        metadata=fused_data.metadata,
        timestamp=fused_data.timestamp.isoformat() if fused_data.timestamp else datetime.now().isoformat()
    )

@router.get("/export/{data_id}")
async def export_data(
    data_id: str,
    format: str = Query("geotiff", description="Export format (geotiff, csv, json)"),
    _: None = Depends(_require_sensor_fusion)
):
    """
    Export sensor data or fusion result.
    
    Args:
        data_id: ID of the sensor data or fusion result
        format: Export format
        
    Returns:
        Exported data file
    """
    # Check if data exists
    if data_id in sensor_data_storage:
        data = sensor_data_storage[data_id]
    elif data_id in fusion_results_storage:
        data = fusion_results_storage[data_id]
    else:
        raise HTTPException(status_code=404, detail=f"Data not found: {data_id}")
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    
    try:
        if format.lower() == "geotiff":
            # Export as GeoTIFF
            temp_file_path = os.path.join(temp_dir, f"{data_id}.tif")
            
            # Check if data is compatible with GeoTIFF
            if not isinstance(data.data, xr.DataArray):
                raise HTTPException(status_code=400, detail="Data is not in grid format, cannot export as GeoTIFF")
            
            # Export
            try:
                # This is a simplified example, in a real implementation
                # we would use rasterio to properly export with CRS information
                data.data.to_netcdf(temp_file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error exporting as GeoTIFF: {str(e)}")
            
            return FileResponse(
                temp_file_path,
                media_type="image/tiff",
                filename=f"{data_id}.tif"
            )
            
        elif format.lower() == "csv":
            # Export as CSV
            temp_file_path = os.path.join(temp_dir, f"{data_id}.csv")
            
            # Convert to DataFrame if needed
            if isinstance(data.data, pd.DataFrame):
                df = data.data
            elif isinstance(data.data, xr.DataArray):
                # Convert grid to points
                if len(data.data.shape) == 2:
                    # 2D grid
                    x_coords = data.data.coords['x'].values
                    y_coords = data.data.coords['y'].values
                    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                    values = data.data.values
                    
                    df = pd.DataFrame({
                        'x': x_grid.flatten(),
                        'y': y_grid.flatten(),
                        'value': values.flatten()
                    })
                else:
                    raise HTTPException(status_code=400, detail="Cannot export 3D+ data as CSV")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported data type for CSV export: {type(data.data)}")
            
            # Export
            try:
                df.to_csv(temp_file_path, index=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error exporting as CSV: {str(e)}")
            
            return FileResponse(
                temp_file_path,
                media_type="text/csv",
                filename=f"{data_id}.csv"
            )
            
        elif format.lower() == "json":
            # Export as JSON
            temp_file_path = os.path.join(temp_dir, f"{data_id}.json")
            
            # Create JSON representation
            json_data = {
                "data_id": data_id,
                "sensor_type": data.sensor_type.value,
                "dimensions": [dim.value for dim in data.dimensions],
                "metadata": data.metadata,
                "timestamp": data.timestamp.isoformat() if data.timestamp else datetime.now().isoformat()
            }
            
            # Add data values
            if isinstance(data.data, pd.DataFrame):
                json_data["data"] = data.data.to_dict(orient="records")
            elif isinstance(data.data, xr.DataArray):
                # Convert to nested list
                json_data["data"] = data.data.values.tolist()
                json_data["coords"] = {
                    dim: data.data.coords[dim].values.tolist()
                    for dim in data.data.coords
                }
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported data type for JSON export: {type(data.data)}")
            
            # Export
            try:
                with open(temp_file_path, "w") as f:
                    json.dump(json_data, f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error exporting as JSON: {str(e)}")
            
            return FileResponse(
                temp_file_path,
                media_type="application/json",
                filename=f"{data_id}.json"
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")
            
    finally:
        # Clean up will be handled by FastAPI's background tasks
        pass  # We'll let FileResponse handle the cleanup
