"""
LiDAR data adapter for sensor fusion framework.

This module provides adapters for loading, preprocessing, and standardizing
LiDAR point cloud data for the sensor fusion framework.
"""

import numpy as np
import laspy
import pdal
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import pyproj
from datetime import datetime
from scipy.spatial import cKDTree

from ..sensor_fusion.core import SensorData, SensorDataAdapter, SensorType, DataDimension

class LiDARDataAdapter(SensorDataAdapter):
    """
    Adapter for LiDAR point cloud data.
    
    This adapter handles loading and preprocessing of LiDAR data
    from various formats including LAS/LAZ, XYZ, and PLY.
    """
    
    def __init__(self):
        """Initialize the LiDAR data adapter."""
        self.supported_formats = ['.las', '.laz', '.xyz', '.txt', '.ply', '.pcd']
    
    def load(self, file_path: str, **kwargs) -> SensorData:
        """
        Load LiDAR data from a file.
        
        Args:
            file_path: Path to the LiDAR data file
            **kwargs: Additional parameters for loading
                max_points: Optional maximum number of points to load
                region: Optional bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
                classification: Optional list of classification values to include
                
        Returns:
            Loaded SensorData object
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extract optional parameters
        max_points = kwargs.get('max_points', None)
        region = kwargs.get('region', None)
        classification = kwargs.get('classification', None)
        
        # Load data based on file format
        if file_ext in ['.las', '.laz']:
            # LAS/LAZ format
            data, metadata = self._load_las(file_path, max_points, region, classification)
        elif file_ext in ['.xyz', '.txt']:
            # XYZ format
            data, metadata = self._load_xyz(file_path, max_points, region)
        elif file_ext in ['.ply']:
            # PLY format
            data, metadata = self._load_ply(file_path, max_points, region)
        elif file_ext in ['.pcd']:
            # PCD format
            data, metadata = self._load_pcd(file_path, max_points, region)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Create SensorData object
        dimensions = [DataDimension.SPATIAL_3D]
        
        return SensorData(
            data=data,
            sensor_type=SensorType.LIDAR,
            dimensions=dimensions,
            metadata=metadata,
            crs=metadata.get('crs', None),
            timestamp=metadata.get('acquisition_time', None)
        )
    
    def _load_las(self, file_path: str, max_points: Optional[int] = None,
                 region: Optional[Tuple[float, float, float, float, float, float]] = None,
                 classification: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load LiDAR data from LAS/LAZ format.
        
        Args:
            file_path: Path to the LAS/LAZ file
            max_points: Optional maximum number of points to load
            region: Optional bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
            classification: Optional list of classification values to include
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Use PDAL for flexible loading with filters
        pipeline = []
        
        # Reader
        reader = {
            "type": "readers.las",
            "filename": file_path
        }
        pipeline.append(reader)
        
        # Apply filters if specified
        if region:
            bounds_filter = {
                "type": "filters.crop",
                "bounds": f"([{region[0]}:{region[3]}],[{region[1]}:{region[4]}],[{region[2]}:{region[5]}])"
            }
            pipeline.append(bounds_filter)
        
        if classification:
            class_filter = {
                "type": "filters.range",
                "limits": "Classification[" + ",".join(str(c) for c in classification) + "]"
            }
            pipeline.append(class_filter)
        
        if max_points:
            # Use decimation filter to reduce point count
            decimation_factor = 1
            try:
                # First get point count
                with laspy.open(file_path) as f:
                    point_count = f.header.point_count
                
                if point_count > max_points:
                    decimation_factor = int(point_count / max_points)
                    if decimation_factor < 1:
                        decimation_factor = 1
            except:
                # If we can't get point count, use a reasonable decimation factor
                decimation_factor = 10
            
            if decimation_factor > 1:
                decimation_filter = {
                    "type": "filters.decimation",
                    "step": decimation_factor
                }
                pipeline.append(decimation_filter)
        
        # Execute pipeline
        pipeline_json = json.dumps(pipeline)
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        
        # Get point cloud data
        arrays = pipeline.arrays
        if len(arrays) == 0:
            raise ValueError("No points loaded from LAS/LAZ file")
        
        point_data = arrays[0]
        
        # Extract coordinates and other attributes
        coords = np.vstack((point_data['X'], point_data['Y'], point_data['Z'])).T
        
        # Extract intensity if available
        if 'Intensity' in point_data.dtype.names:
            intensity = point_data['Intensity']
            # Normalize intensity to 0-1 range
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
        else:
            intensity = np.ones(coords.shape[0])
        
        # Extract classification if available
        if 'Classification' in point_data.dtype.names:
            classification_values = point_data['Classification']
        else:
            classification_values = np.zeros(coords.shape[0])
        
        # Extract return number if available
        if 'ReturnNumber' in point_data.dtype.names:
            return_number = point_data['ReturnNumber']
        else:
            return_number = np.ones(coords.shape[0])
        
        # Extract number of returns if available
        if 'NumberOfReturns' in point_data.dtype.names:
            number_of_returns = point_data['NumberOfReturns']
        else:
            number_of_returns = np.ones(coords.shape[0])
        
        # Combine into a structured array
        dtype = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('classification', np.uint8),
            ('return_number', np.uint8),
            ('number_of_returns', np.uint8)
        ]
        
        data = np.zeros(coords.shape[0], dtype=dtype)
        data['x'] = coords[:, 0]
        data['y'] = coords[:, 1]
        data['z'] = coords[:, 2]
        data['intensity'] = intensity
        data['classification'] = classification_values
        data['return_number'] = return_number
        data['number_of_returns'] = number_of_returns
        
        # Extract metadata
        metadata = {
            'data_type': 'lidar',
            'format': 'LAS/LAZ',
            'point_count': coords.shape[0],
            'min_x': coords[:, 0].min(),
            'min_y': coords[:, 1].min(),
            'min_z': coords[:, 2].min(),
            'max_x': coords[:, 0].max(),
            'max_y': coords[:, 1].max(),
            'max_z': coords[:, 2].max()
        }
        
        # Extract CRS information if available
        try:
            with laspy.open(file_path) as f:
                header = f.header
                if hasattr(header, 'srs') and header.srs is not None:
                    metadata['crs'] = str(header.srs)
        except:
            pass
        
        return data, metadata
    
    def _load_xyz(self, file_path: str, max_points: Optional[int] = None,
                 region: Optional[Tuple[float, float, float, float, float, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load LiDAR data from XYZ format.
        
        Args:
            file_path: Path to the XYZ file
            max_points: Optional maximum number of points to load
            region: Optional bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Determine if file has header
        has_header = False
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            try:
                # Try to parse first line as numbers
                values = [float(v) for v in first_line.split()]
                has_header = False
            except:
                has_header = True
        
        # Load data
        if has_header:
            data = np.loadtxt(file_path, skiprows=1)
        else:
            data = np.loadtxt(file_path)
        
        # Apply region filter if specified
        if region:
            xmin, ymin, zmin, xmax, ymax, zmax = region
            mask = (
                (data[:, 0] >= xmin) & (data[:, 0] <= xmax) &
                (data[:, 1] >= ymin) & (data[:, 1] <= ymax) &
                (data[:, 2] >= zmin) & (data[:, 2] <= zmax)
            )
            data = data[mask]
        
        # Apply max_points filter if specified
        if max_points and data.shape[0] > max_points:
            # Randomly sample points
            indices = np.random.choice(data.shape[0], max_points, replace=False)
            data = data[indices]
        
        # Determine number of columns
        num_cols = data.shape[1]
        
        # Create structured array
        dtype = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ]
        
        if num_cols > 3:
            dtype.append(('intensity', np.float32))
        
        if num_cols > 4:
            dtype.append(('classification', np.uint8))
        
        if num_cols > 5:
            dtype.append(('return_number', np.uint8))
        
        if num_cols > 6:
            dtype.append(('number_of_returns', np.uint8))
        
        structured_data = np.zeros(data.shape[0], dtype=dtype)
        structured_data['x'] = data[:, 0]
        structured_data['y'] = data[:, 1]
        structured_data['z'] = data[:, 2]
        
        if num_cols > 3:
            intensity = data[:, 3]
            # Normalize intensity to 0-1 range
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
            structured_data['intensity'] = intensity
        
        if num_cols > 4:
            structured_data['classification'] = data[:, 4].astype(np.uint8)
        
        if num_cols > 5:
            structured_data['return_number'] = data[:, 5].astype(np.uint8)
        
        if num_cols > 6:
            structured_data['number_of_returns'] = data[:, 6].astype(np.uint8)
        
        # Extract metadata
        metadata = {
            'data_type': 'lidar',
            'format': 'XYZ',
            'point_count': data.shape[0],
            'min_x': data[:, 0].min(),
            'min_y': data[:, 1].min(),
            'min_z': data[:, 2].min(),
            'max_x': data[:, 0].max(),
            'max_y': data[:, 1].max(),
            'max_z': data[:, 2].max(),
            'columns': num_cols
        }
        
        return structured_data, metadata
    
    def _load_ply(self, file_path: str, max_points: Optional[int] = None,
                 region: Optional[Tuple[float, float, float, float, float, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load LiDAR data from PLY format.
        
        Args:
            file_path: Path to the PLY file
            max_points: Optional maximum number of points to load
            region: Optional bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Use PDAL for PLY loading
        pipeline = []
        
        # Reader
        reader = {
            "type": "readers.ply",
            "filename": file_path
        }
        pipeline.append(reader)
        
        # Apply filters if specified
        if region:
            bounds_filter = {
                "type": "filters.crop",
                "bounds": f"([{region[0]}:{region[3]}],[{region[1]}:{region[4]}],[{region[2]}:{region[5]}])"
            }
            pipeline.append(bounds_filter)
        
        if max_points:
            # Use decimation filter to reduce point count
            decimation_filter = {
                "type": "filters.decimation",
                "step": max(1, int(1000000 / max_points))  # Estimate based on typical PLY sizes
            }
            pipeline.append(decimation_filter)
        
        # Execute pipeline
        pipeline_json = json.dumps(pipeline)
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        
        # Get point cloud data
        arrays = pipeline.arrays
        if len(arrays) == 0:
            raise ValueError("No points loaded from PLY file")
        
        point_data = arrays[0]
        
        # Extract coordinates
        coords = np.vstack((point_data['X'], point_data['Y'], point_data['Z'])).T
        
        # Extract intensity if available
        if 'Intensity' in point_data.dtype.names:
            intensity = point_data['Intensity']
            # Normalize intensity to 0-1 range
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
        elif 'Red' in point_data.dtype.names and 'Green' in point_data.dtype.names and 'Blue' in point_data.dtype.names:
            # Use RGB average as intensity
            intensity = (point_data['Red'] + point_data['Green'] + point_data['Blue']) / 3.0
            # Normalize to 0-1
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
        else:
            intensity = np.ones(coords.shape[0])
        
        # Create structured array
        dtype = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ]
        
        # Add RGB if available
        if 'Red' in point_data.dtype.names and 'Green' in point_data.dtype.names and 'Blue' in point_data.dtype.names:
            dtype.extend([
                ('red', np.uint8),
                ('green', np.uint8),
                ('blue', np.uint8)
            ])
        
        structured_data = np.zeros(coords.shape[0], dtype=dtype)
        structured_data['x'] = coords[:, 0]
        structured_data['y'] = coords[:, 1]
        structured_data['z'] = coords[:, 2]
        structured_data['intensity'] = intensity
        
        if 'Red' in point_data.dtype.names and 'Green' in point_data.dtype.names and 'Blue' in point_data.dtype.names:
            structured_data['red'] = point_data['Red']
            structured_data['green'] = point_data['Green']
            structured_data['blue'] = point_data['Blue']
        
        # Extract metadata
        metadata = {
            'data_type': 'lidar',
            'format': 'PLY',
            'point_count': coords.shape[0],
            'min_x': coords[:, 0].min(),
            'min_y': coords[:, 1].min(),
            'min_z': coords[:, 2].min(),
            'max_x': coords[:, 0].max(),
            'max_y': coords[:, 1].max(),
            'max_z': coords[:, 2].max(),
            'has_rgb': 'Red' in point_data.dtype.names
        }
        
        return structured_data, metadata
    
    def _load_pcd(self, file_path: str, max_points: Optional[int] = None,
                 region: Optional[Tuple[float, float, float, float, float, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load LiDAR data from PCD format.
        
        Args:
            file_path: Path to the PCD file
            max_points: Optional maximum number of points to load
            region: Optional bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Use PDAL for PCD loading
        pipeline = []
        
        # Reader
        reader = {
            "type": "readers.pcd",
            "filename": file_path
        }
        pipeline.append(reader)
        
        # Apply filters if specified
        if region:
            bounds_filter = {
                "type": "filters.crop",
                "bounds": f"([{region[0]}:{region[3]}],[{region[1]}:{region[4]}],[{region[2]}:{region[5]}])"
            }
            pipeline.append(bounds_filter)
        
        if max_points:
            # Use decimation filter to reduce point count
            decimation_filter = {
                "type": "filters.decimation",
                "step": max(1, int(1000000 / max_points))  # Estimate based on typical PCD sizes
            }
            pipeline.append(decimation_filter)
        
        # Execute pipeline
        pipeline_json = json.dumps(pipeline)
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        
        # Get point cloud data
        arrays = pipeline.arrays
        if len(arrays) == 0:
            raise ValueError("No points loaded from PCD file")
        
        point_data = arrays[0]
        
        # Extract coordinates
        coords = np.vstack((point_data['X'], point_data['Y'], point_data['Z'])).T
        
        # Extract intensity if available
        if 'Intensity' in point_data.dtype.names:
            intensity = point_data['Intensity']
            # Normalize intensity to 0-1 range
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
        else:
            intensity = np.ones(coords.shape[0])
        
        # Create structured array
        dtype = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ]
        
        # Add RGB if available
        if 'Red' in point_data.dtype.names and 'Green' in point_data.dtype.names and 'Blue' in point_data.dtype.names:
            dtype.extend([
                ('red', np.uint8),
                ('green', np.uint8),
                ('blue', np.uint8)
            ])
        
        structured_data = np.zeros(coords.shape[0], dtype=dtype)
        structured_data['x'] = coords[:, 0]
        structured_data['y'] = coords[:, 1]
        structured_data['z'] = coords[:, 2]
        structured_data['intensity'] = intensity
        
        if 'Red' in point_data.dtype.names and 'Green' in point_data.dtype.names and 'Blue' in point_data.dtype.names:
            structured_data['red'] = point_data['Red']
            structured_data['green'] = point_data['Green']
            structured_data['blue'] = point_data['Blue']
        
        # Extract metadata
        metadata = {
            'data_type': 'lidar',
            'format': 'PCD',
            'point_count': coords.shape[0],
            'min_x': coords[:, 0].min(),
            'min_y': coords[:, 1].min(),
            'min_z': coords[:, 2].min(),
            'max_x': coords[:, 0].max(),
            'max_y': coords[:, 1].max(),
            'max_z': coords[:, 2].max(),
            'has_rgb': 'Red' in point_data.dtype.names
        }
        
        return structured_data, metadata
    
    def preprocess(self, data: SensorData, **kwargs) -> SensorData:
        """
        Preprocess LiDAR data.
        
        Args:
            data: SensorData object to preprocess
            **kwargs: Additional parameters for preprocessing
                remove_outliers: Whether to remove outliers (default: True)
                outlier_method: Method for outlier removal ('statistical', 'radius', 'none')
                outlier_k: Number of neighbors for statistical outlier removal (default: 8)
                outlier_std: Standard deviation multiplier for statistical outlier removal (default: 2.0)
                outlier_radius: Radius for radius outlier removal (default: 1.0)
                outlier_min_neighbors: Minimum neighbors for radius outlier removal (default: 2)
                ground_classification: Whether to classify ground points (default: False)
                noise_classification: Whether to classify noise points (default: False)
                voxel_downsample: Whether to downsample using voxel grid (default: False)
                voxel_size: Voxel size for downsampling (default: 0.1)
                
        Returns:
            Preprocessed SensorData object
        """
        # Extract parameters
        remove_outliers = kwargs.get('remove_outliers', True)
        outlier_method = kwargs.get('outlier_method', 'statistical')
        outlier_k = kwargs.get('outlier_k', 8)
        outlier_std = kwargs.get('outlier_std', 2.0)
        outlier_radius = kwargs.get('outlier_radius', 1.0)
        outlier_min_neighbors = kwargs.get('outlier_min_neighbors', 2)
        ground_classification = kwargs.get('ground_classification', False)
        noise_classification = kwargs.get('noise_classification', False)
        voxel_downsample = kwargs.get('voxel_downsample', False)
        voxel_size = kwargs.get('voxel_size', 0.1)
        
        # Convert to numpy structured array for processing
        array_data = data.to_numpy()
        
        # Create a copy of the data to avoid modifying the original
        processed_data = array_data.copy()
        
        # Extract coordinates
        coords = np.vstack((processed_data['x'], processed_data['y'], processed_data['z'])).T
        
        # Remove outliers
        if remove_outliers:
            if outlier_method == 'statistical':
                # Statistical outlier removal
                tree = cKDTree(coords)
                distances, _ = tree.query(coords, k=outlier_k+1)  # +1 because the first point is the point itself
                distances = distances[:, 1:]  # Remove the first column (distance to self)
                
                # Calculate mean distance for each point
                mean_distances = np.mean(distances, axis=1)
                
                # Calculate global mean and standard deviation
                global_mean = np.mean(mean_distances)
                global_std = np.std(mean_distances)
                
                # Identify inliers
                inlier_mask = mean_distances < global_mean + outlier_std * global_std
                
                # Apply mask
                processed_data = processed_data[inlier_mask]
                coords = coords[inlier_mask]
                
            elif outlier_method == 'radius':
                # Radius outlier removal
                tree = cKDTree(coords)
                
                # Count neighbors within radius
                indices = tree.query_ball_point(coords, outlier_radius)
                neighbor_counts = np.array([len(idx) for idx in indices]) - 1  # -1 to exclude self
                
                # Identify inliers
                inlier_mask = neighbor_counts >= outlier_min_neighbors
                
                # Apply mask
                processed_data = processed_data[inlier_mask]
                coords = coords[inlier_mask]
        
        # Classify ground points
        if ground_classification:
            # Use PDAL to classify ground points
            # First save to temporary LAS file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Create LAS file
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.offsets = [np.min(coords[:, 0]), np.min(coords[:, 1]), np.min(coords[:, 2])]
            header.scales = [0.001, 0.001, 0.001]
            
            las = laspy.LasData(header)
            las.x = coords[:, 0]
            las.y = coords[:, 1]
            las.z = coords[:, 2]
            
            if 'intensity' in processed_data.dtype.names:
                las.intensity = processed_data['intensity'] * 65535  # Scale to 16-bit
            
            las.write(temp_path)
            
            # Run PDAL ground classification
            pipeline = [
                {
                    "type": "readers.las",
                    "filename": temp_path
                },
                {
                    "type": "filters.pmf"  # Progressive morphological filter for ground classification
                },
                {
                    "type": "writers.las",
                    "filename": temp_path + ".classified.las"
                }
            ]
            
            pipeline_json = json.dumps(pipeline)
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()
            
            # Read back classified points
            classified_las = laspy.read(temp_path + ".classified.las")
            
            # Update classification in processed data
            if 'classification' in processed_data.dtype.names:
                processed_data['classification'] = classified_las.classification
            else:
                # Add classification field if it doesn't exist
                new_dtype = np.dtype(processed_data.dtype.descr + [('classification', np.uint8)])
                new_data = np.zeros(processed_data.shape, dtype=new_dtype)
                
                # Copy existing fields
                for name in processed_data.dtype.names:
                    new_data[name] = processed_data[name]
                
                # Add classification
                new_data['classification'] = classified_las.classification
                
                processed_data = new_data
            
            # Clean up temporary files
            try:
                os.remove(temp_path)
                os.remove(temp_path + ".classified.las")
            except:
                pass
        
        # Classify noise points
        if noise_classification:
            # Simple noise classification based on local density
            tree = cKDTree(coords)
            
            # Count neighbors within small radius
            noise_radius = voxel_size * 2 if voxel_downsample else 0.5
            indices = tree.query_ball_point(coords, noise_radius)
            neighbor_counts = np.array([len(idx) for idx in indices]) - 1  # -1 to exclude self
            
            # Identify noise points (very few neighbors)
            noise_mask = neighbor_counts <= 1
            
            # Update classification
            if 'classification' in processed_data.dtype.names:
                # Set noise points to classification 7 (noise)
                processed_data['classification'][noise_mask] = 7
            else:
                # Add classification field if it doesn't exist
                new_dtype = np.dtype(processed_data.dtype.descr + [('classification', np.uint8)])
                new_data = np.zeros(processed_data.shape, dtype=new_dtype)
                
                # Copy existing fields
                for name in processed_data.dtype.names:
                    new_data[name] = processed_data[name]
                
                # Add classification (7 for noise, 1 for unclassified)
                new_data['classification'] = np.where(noise_mask, 7, 1)
                
                processed_data = new_data
        
        # Voxel grid downsampling
        if voxel_downsample:
            # Create voxel grid
            min_bounds = np.min(coords, axis=0)
            max_bounds = np.max(coords, axis=0)
            
            # Calculate voxel indices for each point
            voxel_indices = np.floor((coords - min_bounds) / voxel_size).astype(int)
            
            # Create unique voxel ID for each point
            voxel_ids = voxel_indices[:, 0] + (voxel_indices[:, 1] << 10) + (voxel_indices[:, 2] << 20)
            
            # Find unique voxels and their first occurrence
            _, unique_indices = np.unique(voxel_ids, return_index=True)
            
            # Keep only one point per voxel
            processed_data = processed_data[unique_indices]
        
        # Create new SensorData object with processed data
        quality_metrics = {
            'point_count': processed_data.shape[0],
            'preprocessing_steps': []
        }
        
        if remove_outliers:
            quality_metrics['preprocessing_steps'].append(f'remove_outliers_{outlier_method}')
        if ground_classification:
            quality_metrics['preprocessing_steps'].append('ground_classification')
        if noise_classification:
            quality_metrics['preprocessing_steps'].append('noise_classification')
        if voxel_downsample:
            quality_metrics['preprocessing_steps'].append('voxel_downsample')
            quality_metrics['voxel_size'] = voxel_size
        
        # Update metadata
        updated_metadata = data.metadata.copy()
        updated_metadata['preprocessing'] = {
            'remove_outliers': remove_outliers,
            'outlier_method': outlier_method,
            'ground_classification': ground_classification,
            'noise_classification': noise_classification,
            'voxel_downsample': voxel_downsample,
            'voxel_size': voxel_size
        }
        
        # Update bounds in metadata
        coords = np.vstack((processed_data['x'], processed_data['y'], processed_data['z'])).T
        updated_metadata['min_x'] = coords[:, 0].min()
        updated_metadata['min_y'] = coords[:, 1].min()
        updated_metadata['min_z'] = coords[:, 2].min()
        updated_metadata['max_x'] = coords[:, 0].max()
        updated_metadata['max_y'] = coords[:, 1].max()
        updated_metadata['max_z'] = coords[:, 2].max()
        updated_metadata['point_count'] = processed_data.shape[0]
        
        return SensorData(
            data=processed_data,
            sensor_type=data.sensor_type,
            dimensions=data.dimensions,
            metadata=updated_metadata,
            crs=data.crs,
            quality_metrics=quality_metrics,
            timestamp=data.timestamp,
            data_id=data.data_id
        )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return self.supported_formats
