"""
Core sensor fusion framework for MineralVision platform.

This module provides the base classes and interfaces for the advanced sensor fusion framework,
enabling integration of multiple data sources including hyperspectral imagery, LiDAR,
magnetometry, gravity gradiometry, seismic data, ground-penetrating radar, and InSAR.
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import pyproj
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from datetime import datetime
from scipy import interpolate, ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SensorType(Enum):
    """Enumeration of supported sensor types."""
    HYPERSPECTRAL = "hyperspectral"
    LIDAR = "lidar"
    MAGNETOMETRY = "magnetometry"
    GRAVITY = "gravity"
    SEISMIC = "seismic"
    GPR = "ground_penetrating_radar"
    INSAR = "insar"
    MULTISPECTRAL = "multispectral"
    THERMAL = "thermal"
    RADIOMETRIC = "radiometric"
    ELECTROMAGNETIC = "electromagnetic"
    CUSTOM = "custom"

class DataDimension(Enum):
    """Enumeration of data dimensions."""
    SPATIAL_2D = "2d"
    SPATIAL_3D = "3d"
    SPECTRAL = "spectral"
    TEMPORAL = "temporal"
    CUSTOM = "custom"

class SensorData:
    """
    Base class for sensor data representation.
    
    This class provides a standardized container for sensor data with metadata,
    coordinate reference system information, and quality metrics.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, xr.DataArray, pd.DataFrame],
        sensor_type: SensorType,
        dimensions: List[DataDimension],
        metadata: Dict[str, Any],
        crs: Optional[str] = None,
        quality_metrics: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        data_id: Optional[str] = None
    ):
        """
        Initialize sensor data object.
        
        Args:
            data: The actual sensor data
            sensor_type: Type of sensor that produced the data
            dimensions: List of data dimensions
            metadata: Additional metadata about the data
            crs: Coordinate reference system (EPSG code or WKT)
            quality_metrics: Dictionary of quality metrics
            timestamp: Acquisition timestamp
            data_id: Unique identifier for the data
        """
        self.data = data
        self.sensor_type = sensor_type
        self.dimensions = dimensions
        self.metadata = metadata
        self.crs = crs
        self.quality_metrics = quality_metrics or {}
        self.timestamp = timestamp or datetime.now()
        self.data_id = data_id or str(uuid.uuid4())
        
        # Validate data
        self._validate()
    
    def _validate(self):
        """Validate the sensor data."""
        # Basic validation
        if self.data is None:
            raise ValueError("Data cannot be None")
        
        # Validate based on sensor type and dimensions
        if SensorType.HYPERSPECTRAL in [self.sensor_type]:
            if DataDimension.SPECTRAL not in self.dimensions:
                raise ValueError(f"Spectral dimension required for {self.sensor_type}")
        
        if SensorType.LIDAR in [self.sensor_type]:
            if DataDimension.SPATIAL_3D not in self.dimensions:
                raise ValueError(f"3D spatial dimension required for {self.sensor_type}")
    
    def to_xarray(self) -> xr.DataArray:
        """Convert data to xarray DataArray for standardized processing."""
        if isinstance(self.data, xr.DataArray):
            return self.data
        
        if isinstance(self.data, np.ndarray):
            # Create appropriate dimensions based on data shape and type
            dims = []
            coords = {}
            
            if DataDimension.SPATIAL_2D in self.dimensions:
                if len(self.data.shape) >= 2:
                    dims.extend(['y', 'x'])
                    coords['y'] = np.arange(self.data.shape[0])
                    coords['x'] = np.arange(self.data.shape[1])
            
            if DataDimension.SPECTRAL in self.dimensions:
                spectral_dim_idx = 2 if DataDimension.SPATIAL_2D in self.dimensions else 0
                if len(self.data.shape) > spectral_dim_idx:
                    dims.append('band')
                    if 'wavelengths' in self.metadata:
                        coords['band'] = self.metadata['wavelengths']
                    else:
                        coords['band'] = np.arange(self.data.shape[spectral_dim_idx])
            
            if DataDimension.TEMPORAL in self.dimensions:
                temp_dim_idx = len(dims)
                if len(self.data.shape) > temp_dim_idx:
                    dims.append('time')
                    if 'timestamps' in self.metadata:
                        coords['time'] = self.metadata['timestamps']
                    else:
                        coords['time'] = np.arange(self.data.shape[temp_dim_idx])
            
            # Fill in any remaining dimensions
            while len(dims) < len(self.data.shape):
                dims.append(f'dim_{len(dims)}')
                coords[dims[-1]] = np.arange(self.data.shape[len(dims)-1])
            
            return xr.DataArray(
                data=self.data,
                dims=dims,
                coords=coords,
                attrs={
                    'sensor_type': self.sensor_type.value,
                    'crs': self.crs,
                    'timestamp': str(self.timestamp),
                    'data_id': self.data_id,
                    **self.metadata
                }
            )
        
        if isinstance(self.data, pd.DataFrame):
            # Convert DataFrame to xarray Dataset then extract the DataArray
            ds = self.data.to_xarray()
            # If there's only one data variable, return it as a DataArray
            if len(ds.data_vars) == 1:
                var_name = list(ds.data_vars.keys())[0]
                return ds[var_name]
            # Otherwise, convert the entire dataset to a DataArray with a new dimension
            return ds.to_array(dim='variable')
        
        raise TypeError(f"Cannot convert data of type {type(self.data)} to xarray DataArray")
    
    def to_numpy(self) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(self.data, np.ndarray):
            return self.data
        
        if isinstance(self.data, xr.DataArray):
            return self.data.values
        
        if isinstance(self.data, pd.DataFrame):
            return self.data.values
        
        raise TypeError(f"Cannot convert data of type {type(self.data)} to numpy array")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get complete metadata including quality metrics and basic properties."""
        return {
            'sensor_type': self.sensor_type.value,
            'dimensions': [dim.value for dim in self.dimensions],
            'crs': self.crs,
            'timestamp': str(self.timestamp),
            'data_id': self.data_id,
            'quality_metrics': self.quality_metrics,
            'data_shape': self.get_shape(),
            **self.metadata
        }
    
    def get_shape(self) -> Tuple:
        """Get the shape of the data."""
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        
        if isinstance(self.data, xr.DataArray):
            return self.data.shape
        
        if isinstance(self.data, pd.DataFrame):
            return self.data.shape
        
        raise TypeError(f"Cannot determine shape of data of type {type(self.data)}")
    
    def save(self, directory: str) -> str:
        """
        Save sensor data to disk.
        
        Args:
            directory: Directory to save the data
            
        Returns:
            Path to the saved data
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save data based on type
        data_path = os.path.join(directory, f"{self.data_id}_data")
        
        if isinstance(self.data, np.ndarray):
            np.save(data_path, self.data)
            data_path += ".npy"
        elif isinstance(self.data, xr.DataArray):
            self.data.to_netcdf(data_path + ".nc")
            data_path += ".nc"
        elif isinstance(self.data, pd.DataFrame):
            self.data.to_csv(data_path + ".csv")
            data_path += ".csv"
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{self.data_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert non-serializable types
            metadata = self.get_metadata()
            metadata['timestamp'] = str(self.timestamp)
            metadata['sensor_type'] = self.sensor_type.value
            metadata['dimensions'] = [dim.value for dim in self.dimensions]
            
            json.dump(metadata, f, indent=2)
        
        return data_path
    
    @classmethod
    def load(cls, directory: str, data_id: str) -> 'SensorData':
        """
        Load sensor data from disk.
        
        Args:
            directory: Directory containing the data
            data_id: ID of the data to load
            
        Returns:
            Loaded SensorData object
        """
        # Load metadata
        metadata_path = os.path.join(directory, f"{data_id}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Determine data file type and load accordingly
        data_path_base = os.path.join(directory, f"{data_id}_data")
        
        if os.path.exists(data_path_base + ".npy"):
            data = np.load(data_path_base + ".npy")
        elif os.path.exists(data_path_base + ".nc"):
            data = xr.open_dataarray(data_path_base + ".nc")
        elif os.path.exists(data_path_base + ".csv"):
            data = pd.read_csv(data_path_base + ".csv")
        else:
            raise FileNotFoundError(f"No data file found for {data_id} in {directory}")
        
        # Extract required parameters from metadata
        sensor_type = SensorType(metadata.pop('sensor_type'))
        dimensions = [DataDimension(dim) for dim in metadata.pop('dimensions')]
        crs = metadata.pop('crs', None)
        timestamp_str = metadata.pop('timestamp', None)
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else None
        quality_metrics = metadata.pop('quality_metrics', {})
        
        # Remove other auto-generated metadata fields
        metadata.pop('data_shape', None)
        metadata.pop('data_id', None)
        
        return cls(
            data=data,
            sensor_type=sensor_type,
            dimensions=dimensions,
            metadata=metadata,
            crs=crs,
            quality_metrics=quality_metrics,
            timestamp=timestamp,
            data_id=data_id
        )


class SensorDataAdapter(ABC):
    """
    Abstract base class for sensor data adapters.
    
    Adapters are responsible for loading, preprocessing, and standardizing
    data from specific sensor types.
    """
    
    @abstractmethod
    def load(self, file_path: str, **kwargs) -> SensorData:
        """
        Load sensor data from a file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded and preprocessed SensorData object
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: SensorData, **kwargs) -> SensorData:
        """
        Preprocess sensor data.
        
        Args:
            data: SensorData object to preprocess
            **kwargs: Additional parameters for preprocessing
            
        Returns:
            Preprocessed SensorData object
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        pass


class SensorFusionAlgorithm(ABC):
    """
    Abstract base class for sensor fusion algorithms.
    
    Fusion algorithms combine data from multiple sensors into a unified representation.
    """
    
    @abstractmethod
    def fuse(self, sensor_data_list: List[SensorData], **kwargs) -> SensorData:
        """
        Fuse multiple sensor data objects.
        
        Args:
            sensor_data_list: List of SensorData objects to fuse
            **kwargs: Additional parameters for fusion
            
        Returns:
            Fused SensorData object
        """
        pass
    
    @abstractmethod
    def get_compatibility_matrix(self) -> Dict[Tuple[SensorType, SensorType], float]:
        """
        Get compatibility matrix for sensor types.
        
        Returns:
            Dictionary mapping pairs of sensor types to compatibility scores (0-1)
        """
        pass


class SpatialAlignmentMethod(ABC):
    """
    Abstract base class for spatial alignment methods.
    
    Spatial alignment methods ensure that data from different sensors
    are properly aligned in the same coordinate system.
    """
    
    @abstractmethod
    def align(self, reference: SensorData, target: SensorData, **kwargs) -> SensorData:
        """
        Align target data to reference data.
        
        Args:
            reference: Reference SensorData object
            target: Target SensorData object to align
            **kwargs: Additional parameters for alignment
            
        Returns:
            Aligned SensorData object
        """
        pass


class UncertaintyQuantification(ABC):
    """
    Abstract base class for uncertainty quantification methods.
    
    Uncertainty quantification methods estimate the uncertainty in
    sensor data and fusion results.
    """
    
    @abstractmethod
    def quantify(self, sensor_data: SensorData, **kwargs) -> Dict[str, Any]:
        """
        Quantify uncertainty in sensor data.
        
        Args:
            sensor_data: SensorData object to analyze
            **kwargs: Additional parameters for uncertainty quantification
            
        Returns:
            Dictionary of uncertainty metrics
        """
        pass
    
    @abstractmethod
    def propagate(self, sensor_data_list: List[SensorData], fusion_result: SensorData, **kwargs) -> SensorData:
        """
        Propagate uncertainty through fusion process.
        
        Args:
            sensor_data_list: List of input SensorData objects
            fusion_result: Fused SensorData object
            **kwargs: Additional parameters for uncertainty propagation
            
        Returns:
            SensorData object with uncertainty information
        """
        pass


class SensorFusionManager:
    """
    Manager class for the sensor fusion framework.
    
    This class orchestrates the sensor fusion process, including data loading,
    preprocessing, alignment, fusion, and uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize the sensor fusion manager."""
        self.adapters: Dict[SensorType, SensorDataAdapter] = {}
        self.fusion_algorithms: Dict[str, SensorFusionAlgorithm] = {}
        self.alignment_methods: Dict[str, SpatialAlignmentMethod] = {}
        self.uncertainty_methods: Dict[str, UncertaintyQuantification] = {}
    
    def register_adapter(self, sensor_type: SensorType, adapter: SensorDataAdapter):
        """
        Register a sensor data adapter.
        
        Args:
            sensor_type: Type of sensor the adapter handles
            adapter: SensorDataAdapter instance
        """
        self.adapters[sensor_type] = adapter
    
    def register_fusion_algorithm(self, name: str, algorithm: SensorFusionAlgorithm):
        """
        Register a fusion algorithm.
        
        Args:
            name: Name of the algorithm
            algorithm: SensorFusionAlgorithm instance
        """
        self.fusion_algorithms[name] = algorithm
    
    def register_alignment_method(self, name: str, method: SpatialAlignmentMethod):
        """
        Register a spatial alignment method.
        
        Args:
            name: Name of the method
            method: SpatialAlignmentMethod instance
        """
        self.alignment_methods[name] = method
    
    def register_uncertainty_method(self, name: str, method: UncertaintyQuantification):
        """
        Register an uncertainty quantification method.
        
        Args:
            name: Name of the method
            method: UncertaintyQuantification instance
        """
        self.uncertainty_methods[name] = method
    
    def load_data(self, file_path: str, sensor_type: SensorType, **kwargs) -> SensorData:
        """
        Load sensor data from a file.
        
        Args:
            file_path: Path to the data file
            sensor_type: Type of sensor that produced the data
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded SensorData object
        """
        if sensor_type not in self.adapters:
            raise ValueError(f"No adapter registered for sensor type {sensor_type}")
        
        return self.adapters[sensor_type].load(file_path, **kwargs)
    
    def preprocess_data(self, data: SensorData, **kwargs) -> SensorData:
        """
        Preprocess sensor data.
        
        Args:
            data: SensorData object to preprocess
            **kwargs: Additional parameters for preprocessing
            
        Returns:
            Preprocessed SensorData object
        """
        if data.sensor_type not in self.adapters:
            raise ValueError(f"No adapter registered for sensor type {data.sensor_type}")
        
        return self.adapters[data.sensor_type].preprocess(data, **kwargs)
    
    def align_data(self, reference: SensorData, target: SensorData, method_name: str = None, **kwargs) -> SensorData:
        """
        Align target data to reference data.
        
        Args:
            reference: Reference SensorData object
            target: Target SensorData object to align
            method_name: Name of alignment method to use (uses default if None)
            **kwargs: Additional parameters for alignment
            
        Returns:
            Aligned SensorData object
        """
        if not self.alignment_methods:
            raise ValueError("No alignment methods registered")
        
        if method_name is None:
            # Use first registered method as default
            method_name = next(iter(self.alignment_methods))
        
        if method_name not in self.alignment_methods:
            raise ValueError(f"Alignment method {method_name} not found")
        
        return self.alignment_methods[method_name].align(reference, target, **kwargs)
    
    def fuse_data(self, sensor_data_list: List[SensorData], algorithm_name: str = None, **kwargs) -> SensorData:
        """
        Fuse multiple sensor data objects.
        
        Args:
            sensor_data_list: List of SensorData objects to fuse
            algorithm_name: Name of fusion algorithm to use (uses default if None)
            **kwargs: Additional parameters for fusion
            
        Returns:
            Fused SensorData object
        """
        if not self.fusion_algorithms:
            raise ValueError("No fusion algorithms registered")
        
        if algorithm_name is None:
            # Use first registered algorithm as default
            algorithm_name = next(iter(self.fusion_algorithms))
        
        if algorithm_name not in self.fusion_algorithms:
            raise ValueError(f"Fusion algorithm {algorithm_name} not found")
        
        return self.fusion_algorithms[algorithm_name].fuse(sensor_data_list, **kwargs)
    
    def quantify_uncertainty(self, sensor_data: SensorData, method_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Quantify uncertainty in sensor data.
        
        Args:
            sensor_data: SensorData object to analyze
            method_name: Name of uncertainty method to use (uses default if None)
            **kwargs: Additional parameters for uncertainty quantification
            
        Returns:
            Dictionary of uncertainty metrics
        """
        if not self.uncertainty_methods:
            raise ValueError("No uncertainty methods registered")
        
        if method_name is None:
            # Use first registered method as default
            method_name = next(iter(self.uncertainty_methods))
        
        if method_name not in self.uncertainty_methods:
            raise ValueError(f"Uncertainty method {method_name} not found")
        
        return self.uncertainty_methods[method_name].quantify(sensor_data, **kwargs)
    
    def propagate_uncertainty(self, sensor_data_list: List[SensorData], fusion_result: SensorData, 
                             method_name: str = None, **kwargs) -> SensorData:
        """
        Propagate uncertainty through fusion process.
        
        Args:
            sensor_data_list: List of input SensorData objects
            fusion_result: Fused SensorData object
            method_name: Name of uncertainty method to use (uses default if None)
            **kwargs: Additional parameters for uncertainty propagation
            
        Returns:
            SensorData object with uncertainty information
        """
        if not self.uncertainty_methods:
            raise ValueError("No uncertainty methods registered")
        
        if method_name is None:
            # Use first registered method as default
            method_name = next(iter(self.uncertainty_methods))
        
        if method_name not in self.uncertainty_methods:
            raise ValueError(f"Uncertainty method {method_name} not found")
        
        return self.uncertainty_methods[method_name].propagate(sensor_data_list, fusion_result, **kwargs)
    
    def get_supported_formats(self, sensor_type: SensorType) -> List[str]:
        """
        Get list of supported file formats for a sensor type.
        
        Args:
            sensor_type: Type of sensor
            
        Returns:
            List of supported file extensions
        """
        if sensor_type not in self.adapters:
            raise ValueError(f"No adapter registered for sensor type {sensor_type}")
        
        return self.adapters[sensor_type].get_supported_formats()
    
    def get_compatibility_matrix(self, algorithm_name: str = None) -> Dict[Tuple[SensorType, SensorType], float]:
        """
        Get compatibility matrix for sensor types.
        
        Args:
            algorithm_name: Name of fusion algorithm (uses default if None)
            
        Returns:
            Dictionary mapping pairs of sensor types to compatibility scores (0-1)
        """
        if not self.fusion_algorithms:
            raise ValueError("No fusion algorithms registered")
        
        if algorithm_name is None:
            # Use first registered algorithm as default
            algorithm_name = next(iter(self.fusion_algorithms))
        
        if algorithm_name not in self.fusion_algorithms:
            raise ValueError(f"Fusion algorithm {algorithm_name} not found")
        
        return self.fusion_algorithms[algorithm_name].get_compatibility_matrix()
    
    def execute_fusion_pipeline(self, 
                               file_paths: Dict[SensorType, str],
                               alignment_method: str = None,
                               fusion_algorithm: str = None,
                               uncertainty_method: str = None,
                               preprocess_params: Dict[SensorType, Dict] = None,
                               alignment_params: Dict = None,
                               fusion_params: Dict = None,
                               uncertainty_params: Dict = None) -> Tuple[SensorData, Dict[str, Any]]:
        """
        Execute complete fusion pipeline from data loading to uncertainty quantification.
        
        Args:
            file_paths: Dictionary mapping sensor types to file paths
            alignment_method: Name of alignment method to use
            fusion_algorithm: Name of fusion algorithm to use
            uncertainty_method: Name of uncertainty method to use
            preprocess_params: Dictionary mapping sensor types to preprocessing parameters
            alignment_params: Parameters for alignment
            fusion_params: Parameters for fusion
            uncertainty_params: Parameters for uncertainty quantification
            
        Returns:
            Tuple of (fused SensorData object, dictionary of pipeline metrics)
        """
        # Initialize parameters if not provided
        preprocess_params = preprocess_params or {}
        alignment_params = alignment_params or {}
        fusion_params = fusion_params or {}
        uncertainty_params = uncertainty_params or {}
        
        # Load and preprocess data
        sensor_data_list = []
        for sensor_type, file_path in file_paths.items():
            # Load data
            data = self.load_data(file_path, sensor_type)
            
            # Preprocess data
            params = preprocess_params.get(sensor_type, {})
            data = self.preprocess_data(data, **params)
            
            sensor_data_list.append(data)
        
        # Align data to reference (first sensor in list)
        reference = sensor_data_list[0]
        aligned_data_list = [reference]
        
        for target in sensor_data_list[1:]:
            aligned = self.align_data(reference, target, method_name=alignment_method, **alignment_params)
            aligned_data_list.append(aligned)
        
        # Fuse aligned data
        fused_data = self.fuse_data(aligned_data_list, algorithm_name=fusion_algorithm, **fusion_params)
        
        # Quantify uncertainty
        uncertainty_metrics = self.quantify_uncertainty(fused_data, method_name=uncertainty_method, **uncertainty_params)
        
        # Propagate uncertainty
        fused_data_with_uncertainty = self.propagate_uncertainty(
            aligned_data_list, fused_data, method_name=uncertainty_method, **uncertainty_params
        )
        
        # Compile pipeline metrics
        pipeline_metrics = {
            'input_data_count': len(sensor_data_list),
            'sensor_types': [data.sensor_type.value for data in sensor_data_list],
            'alignment_method': alignment_method,
            'fusion_algorithm': fusion_algorithm,
            'uncertainty_method': uncertainty_method,
            'uncertainty_metrics': uncertainty_metrics,
            'execution_timestamp': datetime.now().isoformat()
        }
        
        return fused_data_with_uncertainty, pipeline_metrics
