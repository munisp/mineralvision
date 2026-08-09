"""
Fusion algorithms for sensor fusion framework.

This module provides implementations of various fusion algorithms for
combining data from multiple sensors into a unified representation.
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import interpolate, ndimage
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Any, Optional
import pyproj
from pyproj import Transformer
import rasterio
from rasterio.warp import reproject, Resampling
import os
import uuid
from datetime import datetime

from ..sensor_fusion.core import (
    SensorData, SensorFusionAlgorithm, SensorType, DataDimension
)

class WeightedAverageFusion(SensorFusionAlgorithm):
    """
    Weighted average fusion algorithm.
    
    This algorithm combines multiple sensor data sources by computing a weighted
    average based on data quality, uncertainty, and compatibility.
    """
    
    def __init__(self):
        """Initialize the weighted average fusion algorithm."""
        # Define compatibility matrix between sensor types
        self._compatibility_matrix = {
            (SensorType.HYPERSPECTRAL, SensorType.LIDAR): 0.9,
            (SensorType.HYPERSPECTRAL, SensorType.MAGNETOMETRY): 0.7,
            (SensorType.HYPERSPECTRAL, SensorType.GRAVITY): 0.6,
            (SensorType.HYPERSPECTRAL, SensorType.SEISMIC): 0.5,
            (SensorType.HYPERSPECTRAL, SensorType.GPR): 0.6,
            (SensorType.HYPERSPECTRAL, SensorType.INSAR): 0.7,
            
            (SensorType.LIDAR, SensorType.MAGNETOMETRY): 0.6,
            (SensorType.LIDAR, SensorType.GRAVITY): 0.5,
            (SensorType.LIDAR, SensorType.SEISMIC): 0.4,
            (SensorType.LIDAR, SensorType.GPR): 0.7,
            (SensorType.LIDAR, SensorType.INSAR): 0.8,
            
            (SensorType.MAGNETOMETRY, SensorType.GRAVITY): 0.9,
            (SensorType.MAGNETOMETRY, SensorType.SEISMIC): 0.7,
            (SensorType.MAGNETOMETRY, SensorType.GPR): 0.6,
            (SensorType.MAGNETOMETRY, SensorType.INSAR): 0.5,
            
            (SensorType.GRAVITY, SensorType.SEISMIC): 0.8,
            (SensorType.GRAVITY, SensorType.GPR): 0.6,
            (SensorType.GRAVITY, SensorType.INSAR): 0.5,
            
            (SensorType.SEISMIC, SensorType.GPR): 0.7,
            (SensorType.SEISMIC, SensorType.INSAR): 0.4,
            
            (SensorType.GPR, SensorType.INSAR): 0.6,
        }
        
        # Add self-compatibility (same sensor type)
        for sensor_type in SensorType:
            self._compatibility_matrix[(sensor_type, sensor_type)] = 1.0
        
        # Add reverse pairs
        for (type1, type2), score in list(self._compatibility_matrix.items()):
            self._compatibility_matrix[(type2, type1)] = score
    
    def fuse(self, sensor_data_list: List[SensorData], **kwargs) -> SensorData:
        """
        Fuse multiple sensor data objects using weighted average.
        
        Args:
            sensor_data_list: List of SensorData objects to fuse
            **kwargs: Additional parameters for fusion
                output_resolution: Resolution of output grid (default: auto)
                weights: Optional manual weights for each sensor (default: auto)
                normalize_values: Whether to normalize values before fusion (default: True)
                fill_missing: Whether to fill missing values (default: True)
                smooth_result: Whether to smooth the result (default: False)
                smooth_sigma: Sigma for Gaussian smoothing (default: 1.0)
                
        Returns:
            Fused SensorData object
        """
        if len(sensor_data_list) < 2:
            raise ValueError("At least two sensor data objects are required for fusion")
        
        # Extract parameters
        output_resolution = kwargs.get('output_resolution', None)
        manual_weights = kwargs.get('weights', None)
        normalize_values = kwargs.get('normalize_values', True)
        fill_missing = kwargs.get('fill_missing', True)
        smooth_result = kwargs.get('smooth_result', False)
        smooth_sigma = kwargs.get('smooth_sigma', 1.0)
        
        # Determine common CRS
        common_crs = self._determine_common_crs(sensor_data_list)
        
        # Convert all data to common grid
        grid_data_list = []
        for sensor_data in sensor_data_list:
            grid_data = self._convert_to_grid(sensor_data, common_crs, output_resolution)
            grid_data_list.append(grid_data)
        
        # Determine common grid extent and resolution
        common_grid = self._create_common_grid(grid_data_list, output_resolution)
        
        # Resample all data to common grid
        resampled_data_list = []
        for grid_data in grid_data_list:
            resampled = self._resample_to_common_grid(grid_data, common_grid)
            resampled_data_list.append(resampled)
        
        # Normalize values if requested
        if normalize_values:
            normalized_data_list = []
            for resampled in resampled_data_list:
                normalized = self._normalize_values(resampled)
                normalized_data_list.append(normalized)
            resampled_data_list = normalized_data_list
        
        # Calculate weights
        if manual_weights is not None:
            if len(manual_weights) != len(sensor_data_list):
                raise ValueError("Number of weights must match number of sensor data objects")
            weights = manual_weights
        else:
            weights = self._calculate_weights(sensor_data_list, resampled_data_list)
        
        # Perform weighted average fusion
        fused_data = self._weighted_average(resampled_data_list, weights, fill_missing)
        
        # Apply smoothing if requested
        if smooth_result:
            fused_data = self._smooth_result(fused_data, smooth_sigma)
        
        # Create metadata for fused data
        fused_metadata = self._create_fused_metadata(sensor_data_list, weights, common_crs, fused_data)
        
        # Create SensorData object for fused data
        dimensions = [DataDimension.SPATIAL_2D]
        if DataDimension.SPECTRAL in sensor_data_list[0].dimensions:
            dimensions.append(DataDimension.SPECTRAL)
        
        return SensorData(
            data=fused_data,
            sensor_type=SensorType.CUSTOM,  # Fused data is a custom type
            dimensions=dimensions,
            metadata=fused_metadata,
            crs=common_crs,
            timestamp=datetime.now()  # Fusion timestamp is current time
        )
    
    def _determine_common_crs(self, sensor_data_list: List[SensorData]) -> str:
        """
        Determine common CRS for all sensor data.
        
        Args:
            sensor_data_list: List of SensorData objects
            
        Returns:
            Common CRS string
        """
        # Collect all CRS
        crs_list = [data.crs for data in sensor_data_list if data.crs is not None]
        
        if not crs_list:
            # No CRS found, use default
            return "EPSG:4326"  # WGS84
        
        # Count occurrences of each CRS
        crs_counts = {}
        for crs in crs_list:
            crs_counts[crs] = crs_counts.get(crs, 0) + 1
        
        # Use most common CRS
        common_crs = max(crs_counts.items(), key=lambda x: x[1])[0]
        
        return common_crs
    
    def _convert_to_grid(self, sensor_data: SensorData, target_crs: str, 
                        output_resolution: Optional[float] = None) -> xr.DataArray:
        """
        Convert sensor data to grid format with target CRS.
        
        Args:
            sensor_data: SensorData object to convert
            target_crs: Target CRS
            output_resolution: Optional output resolution
            
        Returns:
            Grid data as xarray DataArray
        """
        data = sensor_data.data
        
        if isinstance(data, xr.DataArray):
            # Already a grid, just reproject if needed
            if sensor_data.crs != target_crs and sensor_data.crs is not None:
                # Reproject grid
                return self._reproject_grid(data, sensor_data.crs, target_crs)
            return data
        
        elif isinstance(data, pd.DataFrame):
            # Point data, convert to grid
            return self._points_to_grid(data, sensor_data.crs, target_crs, output_resolution)
        
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            # Structured array, convert to DataFrame then to grid
            df = pd.DataFrame({name: data[name] for name in data.dtype.names})
            return self._points_to_grid(df, sensor_data.crs, target_crs, output_resolution)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _reproject_grid(self, data: xr.DataArray, source_crs: str, target_crs: str) -> xr.DataArray:
        """
        Reproject grid data to target CRS.
        
        Args:
            data: Grid data as xarray DataArray
            source_crs: Source CRS
            target_crs: Target CRS
            
        Returns:
            Reprojected grid data
        """
        # Create transformer
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        
        # Get source coordinates
        x_coords = data.coords['x'].values
        y_coords = data.coords['y'].values
        
        # Create meshgrid
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Transform coordinates
        x_transformed, y_transformed = transformer.transform(x_grid.flatten(), y_grid.flatten())
        x_transformed = x_transformed.reshape(x_grid.shape)
        y_transformed = y_transformed.reshape(y_grid.shape)
        
        # Get min/max of transformed coordinates
        x_min, x_max = x_transformed.min(), x_transformed.max()
        y_min, y_max = y_transformed.min(), y_transformed.max()
        
        # Calculate new grid resolution
        x_res = (x_max - x_min) / (len(x_coords) - 1)
        y_res = (y_max - y_min) / (len(y_coords) - 1)
        
        # Create new coordinate arrays
        new_x_coords = np.linspace(x_min, x_max, len(x_coords))
        new_y_coords = np.linspace(y_min, y_max, len(y_coords))
        
        # Create new grid
        new_x_grid, new_y_grid = np.meshgrid(new_x_coords, new_y_coords)
        
        # Interpolate data to new grid
        points = np.column_stack((x_transformed.flatten(), y_transformed.flatten()))
        values = data.values.flatten()
        
        # Remove NaN values
        mask = ~np.isnan(values)
        points = points[mask]
        values = values[mask]
        
        # Interpolate
        grid_values = interpolate.griddata(
            points, values, (new_x_grid, new_y_grid), method='linear'
        )
        
        # Create new DataArray
        reprojected = xr.DataArray(
            data=grid_values,
            dims=['y', 'x'],
            coords={'y': new_y_coords, 'x': new_x_coords},
            attrs=data.attrs.copy()
        )
        
        return reprojected
    
    def _points_to_grid(self, df: pd.DataFrame, source_crs: Optional[str], target_crs: str,
                       output_resolution: Optional[float] = None) -> xr.DataArray:
        """
        Convert point data to grid format with target CRS.
        
        Args:
            df: Point data as DataFrame
            source_crs: Source CRS (can be None)
            target_crs: Target CRS
            output_resolution: Optional output resolution
            
        Returns:
            Grid data as xarray DataArray
        """
        # Ensure required columns exist
        required_columns = ['x', 'y']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Transform coordinates if needed
        x = df['x'].values
        y = df['y'].values
        
        if source_crs is not None and source_crs != target_crs:
            # Create transformer
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            
            # Transform coordinates
            x, y = transformer.transform(x, y)
        
        # Determine value column
        value_column = 'value' if 'value' in df.columns else df.columns[2]
        z = df[value_column].values
        
        # Determine grid resolution if not provided
        if output_resolution is None:
            # Estimate reasonable resolution based on data density
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            point_density = len(x) / (x_range * y_range)
            output_resolution = 1.0 / np.sqrt(point_density)
        
        # Create regular grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x_grid = np.arange(x_min, x_max + output_resolution, output_resolution)
        y_grid = np.arange(y_min, y_max + output_resolution, output_resolution)
        
        # Create meshgrid
        xi_grid, yi_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate
        zi_grid = interpolate.griddata(
            (x, y), z, (xi_grid, yi_grid), method='linear'
        )
        
        # Create DataArray
        grid_data = xr.DataArray(
            data=zi_grid,
            dims=['y', 'x'],
            coords={'y': y_grid, 'x': x_grid}
        )
        
        return grid_data
    
    def _create_common_grid(self, grid_data_list: List[xr.DataArray], 
                           output_resolution: Optional[float] = None) -> Dict[str, Any]:
        """
        Create common grid for all data.
        
        Args:
            grid_data_list: List of grid data
            output_resolution: Optional output resolution
            
        Returns:
            Dictionary with common grid parameters
        """
        # Determine common extent
        x_min = max(data.coords['x'].values.min() for data in grid_data_list)
        x_max = min(data.coords['x'].values.max() for data in grid_data_list)
        y_min = max(data.coords['y'].values.min() for data in grid_data_list)
        y_max = min(data.coords['y'].values.max() for data in grid_data_list)
        
        # Determine common resolution
        if output_resolution is None:
            # Use minimum resolution from all grids
            resolutions = []
            for data in grid_data_list:
                x_coords = data.coords['x'].values
                y_coords = data.coords['y'].values
                if len(x_coords) > 1:
                    resolutions.append(abs(x_coords[1] - x_coords[0]))
                if len(y_coords) > 1:
                    resolutions.append(abs(y_coords[1] - y_coords[0]))
            
            output_resolution = min(resolutions) if resolutions else 1.0
        
        # Create common grid
        x_grid = np.arange(x_min, x_max + output_resolution, output_resolution)
        y_grid = np.arange(y_min, y_max + output_resolution, output_resolution)
        
        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': output_resolution,
            'x_grid': x_grid,
            'y_grid': y_grid
        }
    
    def _resample_to_common_grid(self, data: xr.DataArray, common_grid: Dict[str, Any]) -> xr.DataArray:
        """
        Resample data to common grid.
        
        Args:
            data: Grid data as xarray DataArray
            common_grid: Common grid parameters
            
        Returns:
            Resampled data
        """
        # Create target grid
        x_grid = common_grid['x_grid']
        y_grid = common_grid['y_grid']
        
        # Create meshgrid
        xi_grid, yi_grid = np.meshgrid(x_grid, y_grid)
        
        # Get source coordinates
        x_coords = data.coords['x'].values
        y_coords = data.coords['y'].values
        
        # Check if resampling is needed
        if (len(x_coords) == len(x_grid) and len(y_coords) == len(y_grid) and
            np.allclose(x_coords, x_grid) and np.allclose(y_coords, y_grid)):
            # No resampling needed
            return data
        
        # Interpolate data to new grid
        if len(data.shape) == 2:
            # 2D data (single band)
            zi_grid = interpolate.griddata(
                (np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))),
                data.values.flatten(),
                (xi_grid, yi_grid),
                method='linear'
            )
            
            # Create new DataArray
            resampled = xr.DataArray(
                data=zi_grid,
                dims=['y', 'x'],
                coords={'y': y_grid, 'x': x_grid},
                attrs=data.attrs.copy()
            )
            
        elif len(data.shape) == 3:
            # 3D data (multiple bands)
            num_bands = data.shape[2]
            zi_grid = np.zeros((len(y_grid), len(x_grid), num_bands))
            
            for i in range(num_bands):
                zi_grid[:, :, i] = interpolate.griddata(
                    (np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))),
                    data.values[:, :, i].flatten(),
                    (xi_grid, yi_grid),
                    method='linear'
                )
            
            # Create new DataArray
            resampled = xr.DataArray(
                data=zi_grid,
                dims=['y', 'x', 'band'],
                coords={'y': y_grid, 'x': x_grid, 'band': np.arange(num_bands)},
                attrs=data.attrs.copy()
            )
            
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        return resampled
    
    def _normalize_values(self, data: xr.DataArray) -> xr.DataArray:
        """
        Normalize values to [0, 1] range.
        
        Args:
            data: Grid data as xarray DataArray
            
        Returns:
            Normalized data
        """
        # Create copy of data
        normalized = data.copy()
        
        if len(data.shape) == 2:
            # 2D data (single band)
            values = normalized.values
            valid_mask = ~np.isnan(values)
            if valid_mask.any():
                min_val = np.min(values[valid_mask])
                max_val = np.max(values[valid_mask])
                if max_val > min_val:
                    values[valid_mask] = (values[valid_mask] - min_val) / (max_val - min_val)
                    normalized.values = values
        
        elif len(data.shape) == 3:
            # 3D data (multiple bands)
            values = normalized.values
            for i in range(values.shape[2]):
                band_values = values[:, :, i]
                valid_mask = ~np.isnan(band_values)
                if valid_mask.any():
                    min_val = np.min(band_values[valid_mask])
                    max_val = np.max(band_values[valid_mask])
                    if max_val > min_val:
                        band_values[valid_mask] = (band_values[valid_mask] - min_val) / (max_val - min_val)
                        values[:, :, i] = band_values
            normalized.values = values
        
        return normalized
    
    def _calculate_weights(self, sensor_data_list: List[SensorData], 
                          resampled_data_list: List[xr.DataArray]) -> List[float]:
        """
        Calculate weights for each sensor based on data quality and compatibility.
        
        Args:
            sensor_data_list: List of SensorData objects
            resampled_data_list: List of resampled grid data
            
        Returns:
            List of weights
        """
        # Initialize weights
        weights = np.ones(len(sensor_data_list))
        
        # Adjust weights based on data quality
        for i, sensor_data in enumerate(sensor_data_list):
            # Check if quality metrics are available
            if sensor_data.quality_metrics:
                # Adjust weight based on quality metrics
                if 'nan_percentage' in sensor_data.quality_metrics:
                    nan_percentage = sensor_data.quality_metrics['nan_percentage']
                    weights[i] *= max(0.1, 1.0 - nan_percentage / 100.0)
            
            # Check for NaN values in resampled data
            resampled = resampled_data_list[i]
            nan_percentage = np.isnan(resampled.values).sum() / resampled.size * 100
            weights[i] *= max(0.1, 1.0 - nan_percentage / 100.0)
        
        # Adjust weights based on compatibility
        for i, sensor_data_i in enumerate(sensor_data_list):
            for j, sensor_data_j in enumerate(sensor_data_list):
                if i != j:
                    # Get compatibility score
                    compatibility = self.get_compatibility_matrix().get(
                        (sensor_data_i.sensor_type, sensor_data_j.sensor_type), 0.5
                    )
                    
                    # Adjust weight based on compatibility
                    weights[i] *= compatibility
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Equal weights if all weights are zero
            weights = np.ones(len(sensor_data_list)) / len(sensor_data_list)
        
        return weights.tolist()
    
    def _weighted_average(self, resampled_data_list: List[xr.DataArray], 
                         weights: List[float], fill_missing: bool = True) -> xr.DataArray:
        """
        Perform weighted average fusion.
        
        Args:
            resampled_data_list: List of resampled grid data
            weights: List of weights
            fill_missing: Whether to fill missing values
            
        Returns:
            Fused data
        """
        # Check if all data have the same shape
        shapes = [data.shape for data in resampled_data_list]
        if len(set(tuple(shape) for shape in shapes)) > 1:
            raise ValueError("All data must have the same shape after resampling")
        
        # Get common shape
        shape = shapes[0]
        
        if len(shape) == 2:
            # 2D data (single band)
            # Initialize result with zeros
            result = np.zeros(shape)
            weight_sum = np.zeros(shape)
            
            # Compute weighted sum
            for i, data in enumerate(resampled_data_list):
                values = data.values
                valid_mask = ~np.isnan(values)
                result[valid_mask] += values[valid_mask] * weights[i]
                weight_sum[valid_mask] += weights[i]
            
            # Normalize by weight sum
            valid_mask = weight_sum > 0
            result[valid_mask] /= weight_sum[valid_mask]
            
            # Set NaN for invalid pixels
            result[~valid_mask] = np.nan
            
            # Fill missing values if requested
            if fill_missing and np.any(np.isnan(result)):
                result = self._fill_missing_values(result)
            
            # Create DataArray
            fused = xr.DataArray(
                data=result,
                dims=['y', 'x'],
                coords={'y': resampled_data_list[0].coords['y'].values,
                        'x': resampled_data_list[0].coords['x'].values}
            )
            
        elif len(shape) == 3:
            # 3D data (multiple bands)
            # Initialize result with zeros
            result = np.zeros(shape)
            weight_sum = np.zeros(shape)
            
            # Compute weighted sum
            for i, data in enumerate(resampled_data_list):
                values = data.values
                valid_mask = ~np.isnan(values)
                result[valid_mask] += values[valid_mask] * weights[i]
                weight_sum[valid_mask] += weights[i]
            
            # Normalize by weight sum
            valid_mask = weight_sum > 0
            result[valid_mask] /= weight_sum[valid_mask]
            
            # Set NaN for invalid pixels
            result[~valid_mask] = np.nan
            
            # Fill missing values if requested
            if fill_missing:
                for i in range(shape[2]):
                    band = result[:, :, i]
                    if np.any(np.isnan(band)):
                        result[:, :, i] = self._fill_missing_values(band)
            
            # Create DataArray
            fused = xr.DataArray(
                data=result,
                dims=['y', 'x', 'band'],
                coords={'y': resampled_data_list[0].coords['y'].values,
                        'x': resampled_data_list[0].coords['x'].values,
                        'band': np.arange(shape[2])}
            )
            
        else:
            raise ValueError(f"Unsupported data shape: {shape}")
        
        return fused
    
    def _fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Fill missing values in data.
        
        Args:
            data: Data array with NaN values
            
        Returns:
            Data array with filled values
        """
        # Create mask of valid values
        mask = ~np.isnan(data)
        
        # Check if there are any valid values
        if not np.any(mask):
            return data
        
        # Check if there are any invalid values
        if np.all(mask):
            return data
        
        # Get coordinates of valid values
        coords = np.array(np.where(mask)).T
        
        # Get values of valid pixels
        values = data[mask]
        
        # Create grid for interpolation
        grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        
        # Interpolate
        filled = interpolate.griddata(
            coords, values, (grid_x, grid_y), method='nearest'
        )
        
        return filled
    
    def _smooth_result(self, data: xr.DataArray, sigma: float) -> xr.DataArray:
        """
        Apply Gaussian smoothing to result.
        
        Args:
            data: Data array
            sigma: Sigma for Gaussian smoothing
            
        Returns:
            Smoothed data array
        """
        # Create copy of data
        smoothed = data.copy()
        
        if len(data.shape) == 2:
            # 2D data (single band)
            values = smoothed.values
            
            # Create mask of valid values
            mask = np.isnan(values)
            
            # Fill NaN values temporarily for smoothing
            temp_values = values.copy()
            if np.any(mask):
                temp_values[mask] = np.nanmean(values)
            
            # Apply smoothing
            smoothed_values = ndimage.gaussian_filter(temp_values, sigma=sigma)
            
            # Restore NaN values
            smoothed_values[mask] = np.nan
            
            smoothed.values = smoothed_values
            
        elif len(data.shape) == 3:
            # 3D data (multiple bands)
            values = smoothed.values
            
            for i in range(values.shape[2]):
                band_values = values[:, :, i]
                
                # Create mask of valid values
                mask = np.isnan(band_values)
                
                # Fill NaN values temporarily for smoothing
                temp_values = band_values.copy()
                if np.any(mask):
                    temp_values[mask] = np.nanmean(band_values)
                
                # Apply smoothing
                smoothed_values = ndimage.gaussian_filter(temp_values, sigma=sigma)
                
                # Restore NaN values
                smoothed_values[mask] = np.nan
                
                values[:, :, i] = smoothed_values
            
            smoothed.values = values
            
        return smoothed
    
    def _create_fused_metadata(self, sensor_data_list: List[SensorData], weights: List[float],
                              common_crs: str, fused_data: xr.DataArray) -> Dict[str, Any]:
        """
        Create metadata for fused data.
        
        Args:
            sensor_data_list: List of SensorData objects
            weights: List of weights
            common_crs: Common CRS
            fused_data: Fused data
            
        Returns:
            Metadata dictionary
        """
        # Create basic metadata
        metadata = {
            'data_type': 'fused',
            'fusion_algorithm': 'weighted_average',
            'source_data_types': [data.sensor_type.value for data in sensor_data_list],
            'source_data_ids': [data.data_id for data in sensor_data_list],
            'weights': weights,
            'crs': common_crs,
            'fusion_timestamp': datetime.now().isoformat()
        }
        
        # Add statistics
        values = fused_data.values
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            if len(values.shape) == 2:
                metadata.update({
                    'min_value': float(np.min(values[valid_mask])),
                    'max_value': float(np.max(values[valid_mask])),
                    'mean_value': float(np.mean(values[valid_mask])),
                    'std_value': float(np.std(values[valid_mask]))
                })
            elif len(values.shape) == 3:
                metadata['band_stats'] = []
                for i in range(values.shape[2]):
                    band_values = values[:, :, i]
                    band_mask = ~np.isnan(band_values)
                    if np.any(band_mask):
                        metadata['band_stats'].append({
                            'band': i,
                            'min_value': float(np.min(band_values[band_mask])),
                            'max_value': float(np.max(band_values[band_mask])),
                            'mean_value': float(np.mean(band_values[band_mask])),
                            'std_value': float(np.std(band_values[band_mask]))
                        })
        
        return metadata
    
    def get_compatibility_matrix(self) -> Dict[Tuple[SensorType, SensorType], float]:
        """
        Get compatibility matrix for sensor types.
        
        Returns:
            Dictionary mapping pairs of sensor types to compatibility scores (0-1)
        """
        return self._compatibility_matrix


class BayesianFusion(SensorFusionAlgorithm):
    """
    Bayesian fusion algorithm.
    
    This algorithm combines multiple sensor data sources using Bayesian inference,
    taking into account the uncertainty in each data source.
    """
    
    def __init__(self):
        """Initialize the Bayesian fusion algorithm."""
        # Define compatibility matrix between sensor types
        self._compatibility_matrix = {
            (SensorType.HYPERSPECTRAL, SensorType.LIDAR): 0.9,
            (SensorType.HYPERSPECTRAL, SensorType.MAGNETOMETRY): 0.7,
            (SensorType.HYPERSPECTRAL, SensorType.GRAVITY): 0.6,
            (SensorType.HYPERSPECTRAL, SensorType.SEISMIC): 0.5,
            (SensorType.HYPERSPECTRAL, SensorType.GPR): 0.6,
            (SensorType.HYPERSPECTRAL, SensorType.INSAR): 0.7,
            
            (SensorType.LIDAR, SensorType.MAGNETOMETRY): 0.6,
            (SensorType.LIDAR, SensorType.GRAVITY): 0.5,
            (SensorType.LIDAR, SensorType.SEISMIC): 0.4,
            (SensorType.LIDAR, SensorType.GPR): 0.7,
            (SensorType.LIDAR, SensorType.INSAR): 0.8,
            
            (SensorType.MAGNETOMETRY, SensorType.GRAVITY): 0.9,
            (SensorType.MAGNETOMETRY, SensorType.SEISMIC): 0.7,
            (SensorType.MAGNETOMETRY, SensorType.GPR): 0.6,
            (SensorType.MAGNETOMETRY, SensorType.INSAR): 0.5,
            
            (SensorType.GRAVITY, SensorType.SEISMIC): 0.8,
            (SensorType.GRAVITY, SensorType.GPR): 0.6,
            (SensorType.GRAVITY, SensorType.INSAR): 0.5,
            
            (SensorType.SEISMIC, SensorType.GPR): 0.7,
            (SensorType.SEISMIC, SensorType.INSAR): 0.4,
            
            (SensorType.GPR, SensorType.INSAR): 0.6,
        }
        
        # Add self-compatibility (same sensor type)
        for sensor_type in SensorType:
            self._compatibility_matrix[(sensor_type, sensor_type)] = 1.0
        
        # Add reverse pairs
        for (type1, type2), score in list(self._compatibility_matrix.items()):
            self._compatibility_matrix[(type2, type1)] = score
    
    def fuse(self, sensor_data_list: List[SensorData], **kwargs) -> SensorData:
        """
        Fuse multiple sensor data objects using Bayesian inference.
        
        Args:
            sensor_data_list: List of SensorData objects to fuse
            **kwargs: Additional parameters for fusion
                output_resolution: Resolution of output grid (default: auto)
                prior_mean: Prior mean (default: 0.0)
                prior_variance: Prior variance (default: 1.0)
                sensor_variances: Variances for each sensor (default: auto)
                normalize_values: Whether to normalize values before fusion (default: True)
                
        Returns:
            Fused SensorData object
        """
        if len(sensor_data_list) < 2:
            raise ValueError("At least two sensor data objects are required for fusion")
        
        # Extract parameters
        output_resolution = kwargs.get('output_resolution', None)
        prior_mean = kwargs.get('prior_mean', 0.0)
        prior_variance = kwargs.get('prior_variance', 1.0)
        sensor_variances = kwargs.get('sensor_variances', None)
        normalize_values = kwargs.get('normalize_values', True)
        
        # Determine common CRS
        common_crs = self._determine_common_crs(sensor_data_list)
        
        # Convert all data to common grid
        grid_data_list = []
        for sensor_data in sensor_data_list:
            grid_data = self._convert_to_grid(sensor_data, common_crs, output_resolution)
            grid_data_list.append(grid_data)
        
        # Determine common grid extent and resolution
        common_grid = self._create_common_grid(grid_data_list, output_resolution)
        
        # Resample all data to common grid
        resampled_data_list = []
        for grid_data in grid_data_list:
            resampled = self._resample_to_common_grid(grid_data, common_grid)
            resampled_data_list.append(resampled)
        
        # Normalize values if requested
        if normalize_values:
            normalized_data_list = []
            for resampled in resampled_data_list:
                normalized = self._normalize_values(resampled)
                normalized_data_list.append(normalized)
            resampled_data_list = normalized_data_list
        
        # Determine sensor variances if not provided
        if sensor_variances is None:
            sensor_variances = self._estimate_sensor_variances(sensor_data_list, resampled_data_list)
        
        # Perform Bayesian fusion
        fused_data, fused_variance = self._bayesian_fusion(
            resampled_data_list, sensor_variances, prior_mean, prior_variance
        )
        
        # Create metadata for fused data
        fused_metadata = self._create_fused_metadata(
            sensor_data_list, sensor_variances, common_crs, fused_data, fused_variance
        )
        
        # Create SensorData object for fused data
        dimensions = [DataDimension.SPATIAL_2D]
        if DataDimension.SPECTRAL in sensor_data_list[0].dimensions:
            dimensions.append(DataDimension.SPECTRAL)
        
        return SensorData(
            data=fused_data,
            sensor_type=SensorType.CUSTOM,  # Fused data is a custom type
            dimensions=dimensions,
            metadata=fused_metadata,
            crs=common_crs,
            timestamp=datetime.now()  # Fusion timestamp is current time
        )
    
    def _determine_common_crs(self, sensor_data_list: List[SensorData]) -> str:
        """
        Determine common CRS for all sensor data.
        
        Args:
            sensor_data_list: List of SensorData objects
            
        Returns:
            Common CRS string
        """
        # Collect all CRS
        crs_list = [data.crs for data in sensor_data_list if data.crs is not None]
        
        if not crs_list:
            # No CRS found, use default
            return "EPSG:4326"  # WGS84
        
        # Count occurrences of each CRS
        crs_counts = {}
        for crs in crs_list:
            crs_counts[crs] = crs_counts.get(crs, 0) + 1
        
        # Use most common CRS
        common_crs = max(crs_counts.items(), key=lambda x: x[1])[0]
        
        return common_crs
    
    def _convert_to_grid(self, sensor_data: SensorData, target_crs: str, 
                        output_resolution: Optional[float] = None) -> xr.DataArray:
        """
        Convert sensor data to grid format with target CRS.
        
        Args:
            sensor_data: SensorData object to convert
            target_crs: Target CRS
            output_resolution: Optional output resolution
            
        Returns:
            Grid data as xarray DataArray
        """
        data = sensor_data.data
        
        if isinstance(data, xr.DataArray):
            if sensor_data.crs != target_crs and sensor_data.crs is not None:
                transformer = Transformer.from_crs(sensor_data.crs, target_crs, always_xy=True)
                x_coords = data.coords['x'].values
                y_coords = data.coords['y'].values
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                x_transformed, y_transformed = transformer.transform(x_grid.flatten(), y_grid.flatten())
                x_transformed = x_transformed.reshape(x_grid.shape)
                y_transformed = y_transformed.reshape(y_grid.shape)
                
                x_min, x_max = x_transformed.min(), x_transformed.max()
                y_min, y_max = y_transformed.min(), y_transformed.max()
                new_x_coords = np.linspace(x_min, x_max, len(x_coords))
                new_y_coords = np.linspace(y_min, y_max, len(y_coords))
                new_x_grid, new_y_grid = np.meshgrid(new_x_coords, new_y_coords)
                
                points = np.column_stack((x_transformed.flatten(), y_transformed.flatten()))
                values = data.values.flatten()
                mask = ~np.isnan(values)
                grid_values = interpolate.griddata(points[mask], values[mask], (new_x_grid, new_y_grid), method='linear')
                
                return xr.DataArray(data=grid_values, dims=['y', 'x'], coords={'y': new_y_coords, 'x': new_x_coords}, attrs=data.attrs.copy())
            return data
        
        elif isinstance(data, pd.DataFrame):
            required_columns = ['x', 'y']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            x = data['x'].values
            y = data['y'].values
            
            if sensor_data.crs is not None and sensor_data.crs != target_crs:
                transformer = Transformer.from_crs(sensor_data.crs, target_crs, always_xy=True)
                x, y = transformer.transform(x, y)
            
            value_column = 'value' if 'value' in data.columns else data.columns[2]
            z = data[value_column].values
            
            if output_resolution is None:
                x_range = x.max() - x.min()
                y_range = y.max() - y.min()
                point_density = len(x) / max(x_range * y_range, 1e-10)
                output_resolution = 1.0 / np.sqrt(max(point_density, 1e-10))
            
            x_grid = np.arange(x.min(), x.max() + output_resolution, output_resolution)
            y_grid = np.arange(y.min(), y.max() + output_resolution, output_resolution)
            xi_grid, yi_grid = np.meshgrid(x_grid, y_grid)
            zi_grid = interpolate.griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            
            return xr.DataArray(data=zi_grid, dims=['y', 'x'], coords={'y': y_grid, 'x': x_grid})
        
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            df = pd.DataFrame({name: data[name] for name in data.dtype.names})
            sensor_data_copy = SensorData(sensor_type=sensor_data.sensor_type, data=df, metadata=sensor_data.metadata, 
                                          quality_metrics=sensor_data.quality_metrics, crs=sensor_data.crs, timestamp=sensor_data.timestamp)
            return self._convert_to_grid(sensor_data_copy, target_crs, output_resolution)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _create_common_grid(self, grid_data_list: List[xr.DataArray], 
                           output_resolution: Optional[float] = None) -> Dict[str, Any]:
        """
        Create common grid for all data.
        
        Args:
            grid_data_list: List of grid data
            output_resolution: Optional output resolution
            
        Returns:
            Dictionary with common grid parameters
        """
        x_min = max(data.coords['x'].values.min() for data in grid_data_list)
        x_max = min(data.coords['x'].values.max() for data in grid_data_list)
        y_min = max(data.coords['y'].values.min() for data in grid_data_list)
        y_max = min(data.coords['y'].values.max() for data in grid_data_list)
        
        if output_resolution is None:
            resolutions = []
            for data in grid_data_list:
                x_coords = data.coords['x'].values
                y_coords = data.coords['y'].values
                if len(x_coords) > 1:
                    resolutions.append(abs(x_coords[1] - x_coords[0]))
                if len(y_coords) > 1:
                    resolutions.append(abs(y_coords[1] - y_coords[0]))
            output_resolution = min(resolutions) if resolutions else 1.0
        
        x_grid = np.arange(x_min, x_max + output_resolution, output_resolution)
        y_grid = np.arange(y_min, y_max + output_resolution, output_resolution)
        
        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': output_resolution,
            'x_grid': x_grid,
            'y_grid': y_grid
        }
    
    def _resample_to_common_grid(self, data: xr.DataArray, common_grid: Dict[str, Any]) -> xr.DataArray:
        """
        Resample data to common grid.
        
        Args:
            data: Grid data as xarray DataArray
            common_grid: Common grid parameters
            
        Returns:
            Resampled data
        """
        x_grid = common_grid['x_grid']
        y_grid = common_grid['y_grid']
        xi_grid, yi_grid = np.meshgrid(x_grid, y_grid)
        
        x_coords = data.coords['x'].values
        y_coords = data.coords['y'].values
        
        if (len(x_coords) == len(x_grid) and len(y_coords) == len(y_grid) and
            np.allclose(x_coords, x_grid) and np.allclose(y_coords, y_grid)):
            return data
        
        if len(data.shape) == 2:
            zi_grid = interpolate.griddata(
                (np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))),
                data.values.flatten(),
                (xi_grid, yi_grid),
                method='linear'
            )
            return xr.DataArray(data=zi_grid, dims=['y', 'x'], coords={'y': y_grid, 'x': x_grid}, attrs=data.attrs.copy())
        
        elif len(data.shape) == 3:
            num_bands = data.shape[2]
            zi_grid = np.zeros((len(y_grid), len(x_grid), num_bands))
            for i in range(num_bands):
                zi_grid[:, :, i] = interpolate.griddata(
                    (np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))),
                    data.values[:, :, i].flatten(),
                    (xi_grid, yi_grid),
                    method='linear'
                )
            return xr.DataArray(data=zi_grid, dims=['y', 'x', 'band'], coords={'y': y_grid, 'x': x_grid, 'band': np.arange(num_bands)}, attrs=data.attrs.copy())
        
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def _normalize_values(self, data: xr.DataArray) -> xr.DataArray:
        """
        Normalize values to [0, 1] range.
        
        Args:
            data: Grid data as xarray DataArray
            
        Returns:
            Normalized data
        """
        normalized = data.copy()
        
        if len(data.shape) == 2:
            values = normalized.values
            valid_mask = ~np.isnan(values)
            if valid_mask.any():
                min_val = np.min(values[valid_mask])
                max_val = np.max(values[valid_mask])
                if max_val > min_val:
                    values[valid_mask] = (values[valid_mask] - min_val) / (max_val - min_val)
                    normalized.values = values
        
        elif len(data.shape) == 3:
            values = normalized.values
            for i in range(values.shape[2]):
                band_values = values[:, :, i]
                valid_mask = ~np.isnan(band_values)
                if valid_mask.any():
                    min_val = np.min(band_values[valid_mask])
                    max_val = np.max(band_values[valid_mask])
                    if max_val > min_val:
                        band_values[valid_mask] = (band_values[valid_mask] - min_val) / (max_val - min_val)
                        values[:, :, i] = band_values
            normalized.values = values
        
        return normalized
    
    def _estimate_sensor_variances(self, sensor_data_list: List[SensorData], 
                                  resampled_data_list: List[xr.DataArray]) -> List[float]:
        """
        Estimate variances for each sensor.
        
        Args:
            sensor_data_list: List of SensorData objects
            resampled_data_list: List of resampled grid data
            
        Returns:
            List of variances
        """
        # Initialize variances
        variances = []
        
        for i, (sensor_data, resampled) in enumerate(zip(sensor_data_list, resampled_data_list)):
            # Check if quality metrics are available
            if sensor_data.quality_metrics and 'std_value' in sensor_data.metadata:
                # Use standard deviation from metadata
                variance = sensor_data.metadata['std_value'] ** 2
            else:
                # Estimate variance from data
                values = resampled.values
                valid_mask = ~np.isnan(values)
                if np.any(valid_mask):
                    variance = np.var(values[valid_mask])
                else:
                    variance = 1.0
            
            variances.append(variance)
        
        return variances
    
    def _bayesian_fusion(self, resampled_data_list: List[xr.DataArray], 
                        sensor_variances: List[float], prior_mean: float, 
                        prior_variance: float) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Perform Bayesian fusion.
        
        Args:
            resampled_data_list: List of resampled grid data
            sensor_variances: List of sensor variances
            prior_mean: Prior mean
            prior_variance: Prior variance
            
        Returns:
            Tuple of (fused data, fused variance)
        """
        # Check if all data have the same shape
        shapes = [data.shape for data in resampled_data_list]
        if len(set(tuple(shape) for shape in shapes)) > 1:
            raise ValueError("All data must have the same shape after resampling")
        
        # Get common shape
        shape = shapes[0]
        
        if len(shape) == 2:
            # 2D data (single band)
            # Initialize with prior
            posterior_precision = 1.0 / prior_variance
            posterior_mean_times_precision = prior_mean / prior_variance
            
            # Update with each sensor
            for i, data in enumerate(resampled_data_list):
                values = data.values
                valid_mask = ~np.isnan(values)
                
                # Update precision
                sensor_precision = 1.0 / sensor_variances[i]
                posterior_precision_update = np.zeros(shape)
                posterior_precision_update[valid_mask] = sensor_precision
                posterior_precision += posterior_precision_update
                
                # Update mean times precision
                posterior_mean_times_precision_update = np.zeros(shape)
                posterior_mean_times_precision_update[valid_mask] = values[valid_mask] * sensor_precision
                posterior_mean_times_precision += posterior_mean_times_precision_update
            
            # Calculate posterior mean
            posterior_mean = np.zeros(shape)
            valid_mask = posterior_precision > 0
            posterior_mean[valid_mask] = posterior_mean_times_precision[valid_mask] / posterior_precision[valid_mask]
            
            # Calculate posterior variance
            posterior_variance = np.zeros(shape)
            posterior_variance[valid_mask] = 1.0 / posterior_precision[valid_mask]
            
            # Set NaN for invalid pixels
            posterior_mean[~valid_mask] = np.nan
            posterior_variance[~valid_mask] = np.nan
            
            # Create DataArrays
            fused_data = xr.DataArray(
                data=posterior_mean,
                dims=['y', 'x'],
                coords={'y': resampled_data_list[0].coords['y'].values,
                        'x': resampled_data_list[0].coords['x'].values}
            )
            
            fused_variance = xr.DataArray(
                data=posterior_variance,
                dims=['y', 'x'],
                coords={'y': resampled_data_list[0].coords['y'].values,
                        'x': resampled_data_list[0].coords['x'].values}
            )
            
        elif len(shape) == 3:
            # 3D data (multiple bands)
            # Initialize with prior
            posterior_precision = np.ones(shape) / prior_variance
            posterior_mean_times_precision = np.ones(shape) * prior_mean / prior_variance
            
            # Update with each sensor
            for i, data in enumerate(resampled_data_list):
                values = data.values
                valid_mask = ~np.isnan(values)
                
                # Update precision
                sensor_precision = 1.0 / sensor_variances[i]
                posterior_precision_update = np.zeros(shape)
                posterior_precision_update[valid_mask] = sensor_precision
                posterior_precision += posterior_precision_update
                
                # Update mean times precision
                posterior_mean_times_precision_update = np.zeros(shape)
                posterior_mean_times_precision_update[valid_mask] = values[valid_mask] * sensor_precision
                posterior_mean_times_precision += posterior_mean_times_precision_update
            
            # Calculate posterior mean
            posterior_mean = np.zeros(shape)
            valid_mask = posterior_precision > 0
            posterior_mean[valid_mask] = posterior_mean_times_precision[valid_mask] / posterior_precision[valid_mask]
            
            # Calculate posterior variance
            posterior_variance = np.zeros(shape)
            posterior_variance[valid_mask] = 1.0 / posterior_precision[valid_mask]
            
            # Set NaN for invalid pixels
            posterior_mean[~valid_mask] = np.nan
            posterior_variance[~valid_mask] = np.nan
            
            # Create DataArrays
            fused_data = xr.DataArray(
                data=posterior_mean,
                dims=['y', 'x', 'band'],
                coords={'y': resampled_data_list[0].coords['y'].values,
                        'x': resampled_data_list[0].coords['x'].values,
                        'band': np.arange(shape[2])}
            )
            
            fused_variance = xr.DataArray(
                data=posterior_variance,
                dims=['y', 'x', 'band'],
                coords={'y': resampled_data_list[0].coords['y'].values,
                        'x': resampled_data_list[0].coords['x'].values,
                        'band': np.arange(shape[2])}
            )
            
        else:
            raise ValueError(f"Unsupported data shape: {shape}")
        
        return fused_data, fused_variance
    
    def _create_fused_metadata(self, sensor_data_list: List[SensorData], 
                              sensor_variances: List[float], common_crs: str,
                              fused_data: xr.DataArray, 
                              fused_variance: xr.DataArray) -> Dict[str, Any]:
        """
        Create metadata for fused data.
        
        Args:
            sensor_data_list: List of SensorData objects
            sensor_variances: List of sensor variances
            common_crs: Common CRS
            fused_data: Fused data
            fused_variance: Fused variance
            
        Returns:
            Metadata dictionary
        """
        # Create basic metadata
        metadata = {
            'data_type': 'fused',
            'fusion_algorithm': 'bayesian',
            'source_data_types': [data.sensor_type.value for data in sensor_data_list],
            'source_data_ids': [data.data_id for data in sensor_data_list],
            'sensor_variances': sensor_variances,
            'crs': common_crs,
            'fusion_timestamp': datetime.now().isoformat(),
            'has_uncertainty': True
        }
        
        # Add statistics
        values = fused_data.values
        variance_values = fused_variance.values
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            if len(values.shape) == 2:
                metadata.update({
                    'min_value': float(np.min(values[valid_mask])),
                    'max_value': float(np.max(values[valid_mask])),
                    'mean_value': float(np.mean(values[valid_mask])),
                    'std_value': float(np.std(values[valid_mask])),
                    'min_variance': float(np.min(variance_values[valid_mask])),
                    'max_variance': float(np.max(variance_values[valid_mask])),
                    'mean_variance': float(np.mean(variance_values[valid_mask]))
                })
            elif len(values.shape) == 3:
                metadata['band_stats'] = []
                for i in range(values.shape[2]):
                    band_values = values[:, :, i]
                    band_variance = variance_values[:, :, i]
                    band_mask = ~np.isnan(band_values)
                    if np.any(band_mask):
                        metadata['band_stats'].append({
                            'band': i,
                            'min_value': float(np.min(band_values[band_mask])),
                            'max_value': float(np.max(band_values[band_mask])),
                            'mean_value': float(np.mean(band_values[band_mask])),
                            'std_value': float(np.std(band_values[band_mask])),
                            'min_variance': float(np.min(band_variance[band_mask])),
                            'max_variance': float(np.max(band_variance[band_mask])),
                            'mean_variance': float(np.mean(band_variance[band_mask]))
                        })
        
        return metadata
    
    def get_compatibility_matrix(self) -> Dict[Tuple[SensorType, SensorType], float]:
        """
        Get compatibility matrix for sensor types.
        
        Returns:
            Dictionary mapping pairs of sensor types to compatibility scores (0-1)
        """
        return self._compatibility_matrix
