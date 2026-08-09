"""
Magnetometry data adapter for sensor fusion framework.

This module provides adapters for loading, preprocessing, and standardizing
magnetometry data for the sensor fusion framework.
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from scipy import interpolate, ndimage
from sklearn.preprocessing import StandardScaler

from ..sensor_fusion.core import SensorData, SensorDataAdapter, SensorType, DataDimension

class MagnetometryDataAdapter(SensorDataAdapter):
    """
    Adapter for magnetometry data.
    
    This adapter handles loading and preprocessing of magnetometry data
    from various formats including CSV, GeoTIFF, and binary formats.
    """
    
    def __init__(self):
        """Initialize the magnetometry data adapter."""
        self.supported_formats = ['.csv', '.txt', '.xyz', '.tif', '.tiff', '.mag', '.bin']
    
    def load(self, file_path: str, **kwargs) -> SensorData:
        """
        Load magnetometry data from a file.
        
        Args:
            file_path: Path to the magnetometry data file
            **kwargs: Additional parameters for loading
                header_rows: Number of header rows to skip (for CSV/TXT)
                columns: Column mapping for CSV/TXT files
                coordinate_system: Coordinate system information
                
        Returns:
            Loaded SensorData object
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extract optional parameters
        header_rows = kwargs.get('header_rows', 0)
        columns = kwargs.get('columns', {})
        coordinate_system = kwargs.get('coordinate_system', None)
        
        # Load data based on file format
        if file_ext in ['.csv', '.txt', '.xyz']:
            # CSV/TXT format
            data, metadata = self._load_csv(file_path, header_rows, columns)
        elif file_ext in ['.tif', '.tiff']:
            # GeoTIFF format
            data, metadata = self._load_geotiff(file_path)
        elif file_ext in ['.mag', '.bin']:
            # Binary format
            data, metadata = self._load_binary(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Add coordinate system information if provided
        if coordinate_system:
            metadata['crs'] = coordinate_system
        
        # Create SensorData object
        dimensions = [DataDimension.SPATIAL_2D]
        
        return SensorData(
            data=data,
            sensor_type=SensorType.MAGNETOMETRY,
            dimensions=dimensions,
            metadata=metadata,
            crs=metadata.get('crs', None),
            timestamp=metadata.get('acquisition_time', None)
        )
    
    def _load_csv(self, file_path: str, header_rows: int = 0, 
                 columns: Dict[str, str] = {}) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load magnetometry data from CSV/TXT format.
        
        Args:
            file_path: Path to the CSV/TXT file
            header_rows: Number of header rows to skip
            columns: Column mapping (e.g., {'x': 'Easting', 'y': 'Northing', 'value': 'MagneticField'})
            
        Returns:
            Tuple of (data DataFrame, metadata dictionary)
        """
        # Default column mapping
        default_columns = {
            'x': 0,
            'y': 1,
            'value': 2,
            'error': 3
        }
        
        # Merge with provided column mapping
        column_mapping = {**default_columns, **columns}
        
        # Load data
        df = pd.read_csv(file_path, skiprows=header_rows)
        
        # Rename columns if needed
        if isinstance(column_mapping['x'], str) and column_mapping['x'] in df.columns:
            df.rename(columns={column_mapping['x']: 'x'}, inplace=True)
        elif isinstance(column_mapping['x'], int) and column_mapping['x'] < len(df.columns):
            df.rename(columns={df.columns[column_mapping['x']]: 'x'}, inplace=True)
        
        if isinstance(column_mapping['y'], str) and column_mapping['y'] in df.columns:
            df.rename(columns={column_mapping['y']: 'y'}, inplace=True)
        elif isinstance(column_mapping['y'], int) and column_mapping['y'] < len(df.columns):
            df.rename(columns={df.columns[column_mapping['y']]: 'y'}, inplace=True)
        
        if isinstance(column_mapping['value'], str) and column_mapping['value'] in df.columns:
            df.rename(columns={column_mapping['value']: 'value'}, inplace=True)
        elif isinstance(column_mapping['value'], int) and column_mapping['value'] < len(df.columns):
            df.rename(columns={df.columns[column_mapping['value']]: 'value'}, inplace=True)
        
        if 'error' in column_mapping:
            if isinstance(column_mapping['error'], str) and column_mapping['error'] in df.columns:
                df.rename(columns={column_mapping['error']: 'error'}, inplace=True)
            elif isinstance(column_mapping['error'], int) and column_mapping['error'] < len(df.columns):
                df.rename(columns={df.columns[column_mapping['error']]: 'error'}, inplace=True)
        
        # Ensure required columns exist
        required_columns = ['x', 'y', 'value']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Extract metadata
        metadata = {
            'data_type': 'magnetometry',
            'format': 'CSV/TXT',
            'point_count': len(df),
            'min_x': df['x'].min(),
            'min_y': df['y'].min(),
            'max_x': df['x'].max(),
            'max_y': df['y'].max(),
            'min_value': df['value'].min(),
            'max_value': df['value'].max(),
            'mean_value': df['value'].mean(),
            'std_value': df['value'].std()
        }
        
        # Add error statistics if available
        if 'error' in df.columns:
            metadata['min_error'] = df['error'].min()
            metadata['max_error'] = df['error'].max()
            metadata['mean_error'] = df['error'].mean()
        
        return df, metadata
    
    def _load_geotiff(self, file_path: str) -> Tuple[xr.DataArray, Dict[str, Any]]:
        """
        Load magnetometry data from GeoTIFF format.
        
        Args:
            file_path: Path to the GeoTIFF file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        with rasterio.open(file_path) as src:
            # Read data
            data = src.read(1)
            
            # Create coordinates
            height, width = data.shape
            x_coords = np.linspace(src.bounds.left, src.bounds.right, width)
            y_coords = np.linspace(src.bounds.bottom, src.bounds.top, height)
            
            # Create xarray DataArray
            da = xr.DataArray(
                data=data,
                dims=['y', 'x'],
                coords={'y': y_coords, 'x': x_coords},
                attrs={
                    'crs': src.crs.to_string() if src.crs else None,
                    'transform': src.transform.to_gdal() if src.transform else None
                }
            )
            
            # Extract metadata
            metadata = {
                'data_type': 'magnetometry',
                'format': 'GeoTIFF',
                'width': width,
                'height': height,
                'crs': src.crs.to_string() if src.crs else None,
                'transform': src.transform.to_gdal() if src.transform else None,
                'min_x': src.bounds.left,
                'min_y': src.bounds.bottom,
                'max_x': src.bounds.right,
                'max_y': src.bounds.top,
                'min_value': float(data.min()),
                'max_value': float(data.max()),
                'mean_value': float(data.mean()),
                'std_value': float(data.std())
            }
            
            return da, metadata
    
    def _load_binary(self, file_path: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load magnetometry data from binary format.
        
        Args:
            file_path: Path to the binary file
            **kwargs: Additional parameters for loading
                width: Width of the grid (required for binary)
                height: Height of the grid (required for binary)
                data_type: Data type (e.g., 'float32', 'float64')
                header_size: Size of header in bytes
                byte_order: Byte order ('little' or 'big')
                x_origin: X coordinate of origin
                y_origin: Y coordinate of origin
                cell_size: Cell size
                
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Extract required parameters
        width = kwargs.get('width')
        height = kwargs.get('height')
        data_type = kwargs.get('data_type', 'float32')
        header_size = kwargs.get('header_size', 0)
        byte_order = kwargs.get('byte_order', 'little')
        x_origin = kwargs.get('x_origin', 0)
        y_origin = kwargs.get('y_origin', 0)
        cell_size = kwargs.get('cell_size', 1)
        
        if width is None or height is None:
            raise ValueError("Width and height must be provided for binary files")
        
        # Determine numpy data type
        np_dtype = np.dtype(data_type)
        if byte_order == 'big':
            np_dtype = np_dtype.newbyteorder('>')
        
        # Read binary data
        with open(file_path, 'rb') as f:
            # Skip header if needed
            if header_size > 0:
                f.seek(header_size)
            
            # Read data
            data = np.fromfile(f, dtype=np_dtype)
        
        # Reshape data
        try:
            data = data.reshape((height, width))
        except ValueError:
            # Try transposing
            try:
                data = data.reshape((width, height)).T
            except ValueError:
                raise ValueError(f"Could not reshape data to {height}x{width}")
        
        # Create coordinates
        x_coords = np.linspace(x_origin, x_origin + cell_size * width, width)
        y_coords = np.linspace(y_origin, y_origin + cell_size * height, height)
        
        # Create xarray DataArray
        da = xr.DataArray(
            data=data,
            dims=['y', 'x'],
            coords={'y': y_coords, 'x': x_coords}
        )
        
        # Extract metadata
        metadata = {
            'data_type': 'magnetometry',
            'format': 'Binary',
            'width': width,
            'height': height,
            'data_type': str(np_dtype),
            'header_size': header_size,
            'byte_order': byte_order,
            'x_origin': x_origin,
            'y_origin': y_origin,
            'cell_size': cell_size,
            'min_x': x_origin,
            'min_y': y_origin,
            'max_x': x_origin + cell_size * width,
            'max_y': y_origin + cell_size * height,
            'min_value': float(data.min()),
            'max_value': float(data.max()),
            'mean_value': float(data.mean()),
            'std_value': float(data.std())
        }
        
        return da, metadata
    
    def preprocess(self, data: SensorData, **kwargs) -> SensorData:
        """
        Preprocess magnetometry data.
        
        Args:
            data: SensorData object to preprocess
            **kwargs: Additional parameters for preprocessing
                remove_outliers: Whether to remove outliers (default: True)
                outlier_std: Standard deviation threshold for outlier removal (default: 3.0)
                detrend: Whether to remove regional trend (default: False)
                detrend_order: Polynomial order for detrending (default: 1)
                interpolate_grid: Whether to interpolate to regular grid (default: False)
                grid_resolution: Resolution for grid interpolation (default: None)
                smooth: Whether to apply smoothing (default: False)
                smooth_sigma: Sigma for Gaussian smoothing (default: 1.0)
                
        Returns:
            Preprocessed SensorData object
        """
        # Extract parameters
        remove_outliers = kwargs.get('remove_outliers', True)
        outlier_std = kwargs.get('outlier_std', 3.0)
        detrend = kwargs.get('detrend', False)
        detrend_order = kwargs.get('detrend_order', 1)
        interpolate_grid = kwargs.get('interpolate_grid', False)
        grid_resolution = kwargs.get('grid_resolution', None)
        smooth = kwargs.get('smooth', False)
        smooth_sigma = kwargs.get('smooth_sigma', 1.0)
        
        # Convert to appropriate format for processing
        if isinstance(data.data, pd.DataFrame):
            # Point data
            df = data.data.copy()
            
            # Remove outliers
            if remove_outliers:
                mean = df['value'].mean()
                std = df['value'].std()
                df = df[(df['value'] > mean - outlier_std * std) & 
                        (df['value'] < mean + outlier_std * std)]
            
            # Detrend
            if detrend:
                # Fit polynomial surface
                x = df['x'].values
                y = df['y'].values
                z = df['value'].values
                
                # Normalize coordinates for numerical stability
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                x_norm = (x - x_min) / (x_max - x_min)
                y_norm = (y - y_min) / (y_max - y_min)
                
                # Create polynomial terms
                A = np.ones((len(x), 1))
                
                for i in range(1, detrend_order + 1):
                    for j in range(i + 1):
                        A = np.column_stack((A, x_norm**(i-j) * y_norm**j))
                
                # Solve for coefficients
                coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
                
                # Compute trend
                trend = A @ coeffs
                
                # Subtract trend
                df['value'] = z - trend
            
            # Interpolate to regular grid
            if interpolate_grid:
                x = df['x'].values
                y = df['y'].values
                z = df['value'].values
                
                # Determine grid resolution if not provided
                if grid_resolution is None:
                    # Estimate reasonable resolution based on data density
                    x_range = x.max() - x.min()
                    y_range = y.max() - y.min()
                    point_density = len(x) / (x_range * y_range)
                    grid_resolution = 1.0 / np.sqrt(point_density)
                
                # Create regular grid
                xi = np.arange(x.min(), x.max() + grid_resolution, grid_resolution)
                yi = np.arange(y.min(), y.max() + grid_resolution, grid_resolution)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                
                # Interpolate
                zi_grid = interpolate.griddata(
                    (x, y), z, (xi_grid, yi_grid), method='linear'
                )
                
                # Create xarray DataArray
                da = xr.DataArray(
                    data=zi_grid,
                    dims=['y', 'x'],
                    coords={'y': yi, 'x': xi}
                )
                
                # Apply smoothing if requested
                if smooth:
                    da.values = ndimage.gaussian_filter(da.values, sigma=smooth_sigma)
                
                # Create new SensorData object
                quality_metrics = {
                    'preprocessing_steps': []
                }
                
                if remove_outliers:
                    quality_metrics['preprocessing_steps'].append('remove_outliers')
                    quality_metrics['points_removed'] = len(data.data) - len(df)
                
                if detrend:
                    quality_metrics['preprocessing_steps'].append('detrend')
                    quality_metrics['detrend_order'] = detrend_order
                
                quality_metrics['preprocessing_steps'].append('interpolate_grid')
                quality_metrics['grid_resolution'] = grid_resolution
                
                if smooth:
                    quality_metrics['preprocessing_steps'].append('smooth')
                    quality_metrics['smooth_sigma'] = smooth_sigma
                
                # Update metadata
                updated_metadata = data.metadata.copy()
                updated_metadata['preprocessing'] = {
                    'remove_outliers': remove_outliers,
                    'outlier_std': outlier_std,
                    'detrend': detrend,
                    'detrend_order': detrend_order,
                    'interpolate_grid': interpolate_grid,
                    'grid_resolution': grid_resolution,
                    'smooth': smooth,
                    'smooth_sigma': smooth_sigma
                }
                updated_metadata['min_value'] = float(da.values.min())
                updated_metadata['max_value'] = float(da.values.max())
                updated_metadata['mean_value'] = float(da.values.mean())
                updated_metadata['std_value'] = float(da.values.std())
                
                return SensorData(
                    data=da,
                    sensor_type=data.sensor_type,
                    dimensions=data.dimensions,
                    metadata=updated_metadata,
                    crs=data.crs,
                    quality_metrics=quality_metrics,
                    timestamp=data.timestamp,
                    data_id=data.data_id
                )
            
            else:
                # Return preprocessed point data
                quality_metrics = {
                    'preprocessing_steps': []
                }
                
                if remove_outliers:
                    quality_metrics['preprocessing_steps'].append('remove_outliers')
                    quality_metrics['points_removed'] = len(data.data) - len(df)
                
                if detrend:
                    quality_metrics['preprocessing_steps'].append('detrend')
                    quality_metrics['detrend_order'] = detrend_order
                
                # Update metadata
                updated_metadata = data.metadata.copy()
                updated_metadata['preprocessing'] = {
                    'remove_outliers': remove_outliers,
                    'outlier_std': outlier_std,
                    'detrend': detrend,
                    'detrend_order': detrend_order
                }
                updated_metadata['min_value'] = float(df['value'].min())
                updated_metadata['max_value'] = float(df['value'].max())
                updated_metadata['mean_value'] = float(df['value'].mean())
                updated_metadata['std_value'] = float(df['value'].std())
                
                return SensorData(
                    data=df,
                    sensor_type=data.sensor_type,
                    dimensions=data.dimensions,
                    metadata=updated_metadata,
                    crs=data.crs,
                    quality_metrics=quality_metrics,
                    timestamp=data.timestamp,
                    data_id=data.data_id
                )
        
        elif isinstance(data.data, xr.DataArray):
            # Grid data
            da = data.data.copy()
            
            # Remove outliers
            if remove_outliers:
                mean = da.values.mean()
                std = da.values.std()
                mask = (da.values < mean - outlier_std * std) | (da.values > mean + outlier_std * std)
                da.values[mask] = np.nan
            
            # Detrend
            if detrend:
                # Create coordinate meshgrid
                x_coords = da.coords['x'].values
                y_coords = da.coords['y'].values
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                
                # Normalize coordinates for numerical stability
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                x_norm = (x_grid - x_min) / (x_max - x_min)
                y_norm = (y_grid - y_min) / (y_max - y_min)
                
                # Flatten for fitting
                x_flat = x_norm.flatten()
                y_flat = y_norm.flatten()
                z_flat = da.values.flatten()
                
                # Remove NaN values
                mask = ~np.isnan(z_flat)
                x_valid = x_flat[mask]
                y_valid = y_flat[mask]
                z_valid = z_flat[mask]
                
                # Create polynomial terms
                A = np.ones((len(x_valid), 1))
                
                for i in range(1, detrend_order + 1):
                    for j in range(i + 1):
                        A = np.column_stack((A, x_valid**(i-j) * y_valid**j))
                
                # Solve for coefficients
                coeffs, residuals, rank, s = np.linalg.lstsq(A, z_valid, rcond=None)
                
                # Create polynomial terms for full grid
                A_full = np.ones((len(x_flat), 1))
                
                for i in range(1, detrend_order + 1):
                    for j in range(i + 1):
                        A_full = np.column_stack((A_full, x_flat**(i-j) * y_flat**j))
                
                # Compute trend
                trend = A_full @ coeffs
                trend = trend.reshape(da.shape)
                
                # Subtract trend
                da.values = da.values - trend
            
            # Apply smoothing
            if smooth:
                # Handle NaN values
                mask = np.isnan(da.values)
                if mask.any():
                    # Interpolate NaN values for smoothing
                    values = da.values.copy()
                    values[mask] = 0
                    
                    # Apply smoothing
                    smoothed = ndimage.gaussian_filter(values, sigma=smooth_sigma)
                    
                    # Create weight array (0 for NaN, 1 for valid)
                    weights = np.ones_like(values)
                    weights[mask] = 0
                    
                    # Smooth weights
                    smoothed_weights = ndimage.gaussian_filter(weights, sigma=smooth_sigma)
                    
                    # Normalize
                    result = smoothed / (smoothed_weights + 1e-10)
                    
                    # Restore NaN values
                    result[mask] = np.nan
                    
                    da.values = result
                else:
                    # No NaN values, simple smoothing
                    da.values = ndimage.gaussian_filter(da.values, sigma=smooth_sigma)
            
            # Create new SensorData object
            quality_metrics = {
                'preprocessing_steps': []
            }
            
            if remove_outliers:
                quality_metrics['preprocessing_steps'].append('remove_outliers')
                quality_metrics['nan_percentage'] = np.isnan(da.values).sum() / da.size * 100
            
            if detrend:
                quality_metrics['preprocessing_steps'].append('detrend')
                quality_metrics['detrend_order'] = detrend_order
            
            if smooth:
                quality_metrics['preprocessing_steps'].append('smooth')
                quality_metrics['smooth_sigma'] = smooth_sigma
            
            # Update metadata
            updated_metadata = data.metadata.copy()
            updated_metadata['preprocessing'] = {
                'remove_outliers': remove_outliers,
                'outlier_std': outlier_std,
                'detrend': detrend,
                'detrend_order': detrend_order,
                'smooth': smooth,
                'smooth_sigma': smooth_sigma
            }
            
            # Update statistics in metadata
            valid_values = da.values[~np.isnan(da.values)]
            if len(valid_values) > 0:
                updated_metadata['min_value'] = float(valid_values.min())
                updated_metadata['max_value'] = float(valid_values.max())
                updated_metadata['mean_value'] = float(valid_values.mean())
                updated_metadata['std_value'] = float(valid_values.std())
            
            return SensorData(
                data=da,
                sensor_type=data.sensor_type,
                dimensions=data.dimensions,
                metadata=updated_metadata,
                crs=data.crs,
                quality_metrics=quality_metrics,
                timestamp=data.timestamp,
                data_id=data.data_id
            )
        
        else:
            raise ValueError(f"Unsupported data type: {type(data.data)}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return self.supported_formats
