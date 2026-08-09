"""
Hyperspectral data adapter for sensor fusion framework.

This module provides adapters for loading, preprocessing, and standardizing
hyperspectral imagery data for the sensor fusion framework.
"""

import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import Resampling
import spectral.io.envi as envi
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import scipy.ndimage as ndimage
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from ..sensor_fusion.core import SensorData, SensorDataAdapter, SensorType, DataDimension

class HyperspectralDataAdapter(SensorDataAdapter):
    """
    Adapter for hyperspectral imagery data.
    
    This adapter handles loading and preprocessing of hyperspectral imagery
    from various formats including ENVI, GeoTIFF, and HDF.
    """
    
    def __init__(self):
        """Initialize the hyperspectral data adapter."""
        self.supported_formats = ['.hdr', '.tif', '.tiff', '.h5', '.hdf', '.hdf5']
    
    def load(self, file_path: str, **kwargs) -> SensorData:
        """
        Load hyperspectral data from a file.
        
        Args:
            file_path: Path to the hyperspectral data file
            **kwargs: Additional parameters for loading
                wavelength_file: Optional path to wavelength metadata file
                bands: Optional list of bands to load (indices)
                subset_region: Optional tuple of (row_start, row_end, col_start, col_end)
                
        Returns:
            Loaded SensorData object
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extract optional parameters
        wavelength_file = kwargs.get('wavelength_file', None)
        bands = kwargs.get('bands', None)
        subset_region = kwargs.get('subset_region', None)
        
        # Load data based on file format
        if file_ext in ['.hdr']:
            # ENVI format
            data, metadata = self._load_envi(file_path, bands, subset_region)
        elif file_ext in ['.tif', '.tiff']:
            # GeoTIFF format
            data, metadata = self._load_geotiff(file_path, bands, subset_region)
        elif file_ext in ['.h5', '.hdf', '.hdf5']:
            # HDF format
            data, metadata = self._load_hdf(file_path, bands, subset_region)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Load wavelength information if provided
        if wavelength_file:
            wavelengths = self._load_wavelengths(wavelength_file)
            metadata['wavelengths'] = wavelengths
        
        # Create SensorData object
        dimensions = [DataDimension.SPATIAL_2D, DataDimension.SPECTRAL]
        
        return SensorData(
            data=data,
            sensor_type=SensorType.HYPERSPECTRAL,
            dimensions=dimensions,
            metadata=metadata,
            crs=metadata.get('crs', None),
            timestamp=metadata.get('acquisition_time', None)
        )
    
    def _load_envi(self, file_path: str, bands: Optional[List[int]] = None, 
                  subset_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load hyperspectral data from ENVI format.
        
        Args:
            file_path: Path to the ENVI header file
            bands: Optional list of bands to load
            subset_region: Optional tuple of (row_start, row_end, col_start, col_end)
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Load ENVI data
        envi_data = envi.open(file_path)
        
        # Extract metadata
        metadata = {
            'data_type': 'hyperspectral',
            'format': 'ENVI',
            'bands': envi_data.nbands,
            'width': envi_data.ncols,
            'height': envi_data.nrows,
            'data_type': envi_data.dtype.name
        }
        
        # Extract wavelength information if available
        if hasattr(envi_data, 'metadata') and 'wavelength' in envi_data.metadata:
            wavelengths = [float(w) for w in envi_data.metadata['wavelength']]
            metadata['wavelengths'] = wavelengths
        
        # Extract acquisition time if available
        if hasattr(envi_data, 'metadata') and 'acquisition time' in envi_data.metadata:
            try:
                acq_time = datetime.strptime(envi_data.metadata['acquisition time'], '%Y-%m-%d %H:%M:%S')
                metadata['acquisition_time'] = acq_time
            except:
                pass
        
        # Load data
        if subset_region:
            row_start, row_end, col_start, col_end = subset_region
            if bands:
                data = envi_data.read_bands(bands, rows=range(row_start, row_end), cols=range(col_start, col_end))
            else:
                data = envi_data.read_bands(range(envi_data.nbands), rows=range(row_start, row_end), cols=range(col_start, col_end))
        else:
            if bands:
                data = envi_data.read_bands(bands)
            else:
                data = envi_data.load()
        
        # Ensure data is in (height, width, bands) format
        if data.shape[0] == envi_data.nbands:
            data = np.moveaxis(data, 0, 2)
        
        return data, metadata
    
    def _load_geotiff(self, file_path: str, bands: Optional[List[int]] = None,
                     subset_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load hyperspectral data from GeoTIFF format.
        
        Args:
            file_path: Path to the GeoTIFF file
            bands: Optional list of bands to load
            subset_region: Optional tuple of (row_start, row_end, col_start, col_end)
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        with rasterio.open(file_path) as src:
            # Extract metadata
            metadata = {
                'data_type': 'hyperspectral',
                'format': 'GeoTIFF',
                'bands': src.count,
                'width': src.width,
                'height': src.height,
                'crs': src.crs.to_string() if src.crs else None,
                'transform': src.transform.to_gdal() if src.transform else None,
                'data_type': src.dtypes[0]
            }
            
            # Load data
            if subset_region:
                row_start, row_end, col_start, col_end = subset_region
                window = ((row_start, row_end), (col_start, col_end))
                
                if bands:
                    data = src.read(bands, window=window)
                else:
                    data = src.read(window=window)
            else:
                if bands:
                    data = src.read(bands)
                else:
                    data = src.read()
            
            # Ensure data is in (height, width, bands) format
            if data.shape[0] != src.height:
                data = np.moveaxis(data, 0, 2)
            
            return data, metadata
    
    def _load_hdf(self, file_path: str, bands: Optional[List[int]] = None,
                 subset_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load hyperspectral data from HDF format.
        
        Args:
            file_path: Path to the HDF file
            bands: Optional list of bands to load
            subset_region: Optional tuple of (row_start, row_end, col_start, col_end)
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Find the dataset containing the hyperspectral data
            # This varies by HDF file structure, so we need to handle different cases
            
            # Common dataset names for hyperspectral data
            dataset_names = ['Reflectance', 'Radiance', 'data', 'hyperspectral_data', 'image']
            
            # Find the first matching dataset
            data_key = None
            for key in dataset_names:
                if key in f:
                    data_key = key
                    break
            
            # If no matching dataset found, use the first dataset
            if data_key is None:
                data_key = list(f.keys())[0]
            
            # Load the dataset
            dataset = f[data_key]
            
            # Extract metadata
            metadata = {
                'data_type': 'hyperspectral',
                'format': 'HDF',
                'data_key': data_key
            }
            
            # Add attributes as metadata
            for key, value in dataset.attrs.items():
                metadata[key] = value
            
            # Load data
            if subset_region:
                row_start, row_end, col_start, col_end = subset_region
                
                if bands:
                    if len(dataset.shape) == 3 and dataset.shape[2] > 1:  # (height, width, bands)
                        data = dataset[row_start:row_end, col_start:col_end, bands]
                    else:  # (bands, height, width)
                        data = dataset[bands, row_start:row_end, col_start:col_end]
                else:
                    data = dataset[row_start:row_end, col_start:col_end, :]
            else:
                if bands:
                    if len(dataset.shape) == 3 and dataset.shape[2] > 1:  # (height, width, bands)
                        data = dataset[:, :, bands]
                    else:  # (bands, height, width)
                        data = dataset[bands, :, :]
                else:
                    data = dataset[:]
            
            # Ensure data is in (height, width, bands) format
            if len(data.shape) == 3:
                if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                    # Likely (bands, height, width) format
                    data = np.moveaxis(data, 0, 2)
            
            # Update metadata with dimensions
            metadata['bands'] = data.shape[2] if len(data.shape) == 3 else 1
            metadata['height'] = data.shape[0]
            metadata['width'] = data.shape[1]
            
            return data, metadata
    
    def _load_wavelengths(self, wavelength_file: str) -> List[float]:
        """
        Load wavelength information from a file.
        
        Args:
            wavelength_file: Path to the wavelength file
            
        Returns:
            List of wavelengths
        """
        file_ext = os.path.splitext(wavelength_file)[1].lower()
        
        if file_ext == '.json':
            with open(wavelength_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'wavelengths' in data:
                    return data['wavelengths']
                else:
                    # Try to find a list in the JSON
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                            return value
        elif file_ext in ['.txt', '.csv']:
            wavelengths = []
            with open(wavelength_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            wavelengths.append(float(line.split(',')[0]))
                        except:
                            pass
            return wavelengths
        
        raise ValueError(f"Unsupported wavelength file format: {file_ext}")
    
    def preprocess(self, data: SensorData, **kwargs) -> SensorData:
        """
        Preprocess hyperspectral data.
        
        Args:
            data: SensorData object to preprocess
            **kwargs: Additional parameters for preprocessing
                remove_nodata: Whether to remove nodata values (default: True)
                nodata_value: Value to use for nodata (default: 0)
                normalize: Whether to normalize the data (default: True)
                correct_bad_bands: Whether to correct bad bands (default: True)
                bad_bands: List of bad band indices to correct
                smooth: Whether to apply smoothing (default: False)
                smooth_sigma: Sigma for Gaussian smoothing (default: 1.0)
                
        Returns:
            Preprocessed SensorData object
        """
        # Extract parameters
        remove_nodata = kwargs.get('remove_nodata', True)
        nodata_value = kwargs.get('nodata_value', 0)
        normalize = kwargs.get('normalize', True)
        correct_bad_bands = kwargs.get('correct_bad_bands', True)
        bad_bands = kwargs.get('bad_bands', [])
        smooth = kwargs.get('smooth', False)
        smooth_sigma = kwargs.get('smooth_sigma', 1.0)
        
        # Convert to numpy array for processing
        array_data = data.to_numpy()
        
        # Create a copy of the data to avoid modifying the original
        processed_data = array_data.copy()
        
        # Remove nodata values
        if remove_nodata:
            mask = processed_data == nodata_value
            processed_data[mask] = np.nan
        
        # Correct bad bands
        if correct_bad_bands and bad_bands:
            for band in bad_bands:
                if band < processed_data.shape[2]:
                    # Replace bad band with interpolation from adjacent bands
                    if band > 0 and band < processed_data.shape[2] - 1:
                        processed_data[:, :, band] = (processed_data[:, :, band-1] + processed_data[:, :, band+1]) / 2
                    elif band == 0:
                        processed_data[:, :, band] = processed_data[:, :, band+1]
                    else:
                        processed_data[:, :, band] = processed_data[:, :, band-1]
        
        # Apply smoothing
        if smooth:
            for i in range(processed_data.shape[2]):
                processed_data[:, :, i] = ndimage.gaussian_filter(processed_data[:, :, i], sigma=smooth_sigma)
        
        # Normalize data
        if normalize:
            # Reshape to 2D for normalization
            shape = processed_data.shape
            reshaped = processed_data.reshape(-1, shape[2])
            
            # Handle NaN values
            mask = ~np.isnan(reshaped).any(axis=1)
            valid_data = reshaped[mask]
            
            if len(valid_data) > 0:
                # Normalize valid data
                scaler = StandardScaler()
                normalized = scaler.fit_transform(valid_data)
                
                # Put normalized data back
                reshaped_normalized = np.full_like(reshaped, np.nan)
                reshaped_normalized[mask] = normalized
                
                # Reshape back to original shape
                processed_data = reshaped_normalized.reshape(shape)
        
        # Create new SensorData object with processed data
        quality_metrics = {
            'nan_percentage': np.isnan(processed_data).sum() / processed_data.size * 100,
            'preprocessing_steps': []
        }
        
        if remove_nodata:
            quality_metrics['preprocessing_steps'].append('remove_nodata')
        if correct_bad_bands:
            quality_metrics['preprocessing_steps'].append('correct_bad_bands')
        if smooth:
            quality_metrics['preprocessing_steps'].append('smooth')
        if normalize:
            quality_metrics['preprocessing_steps'].append('normalize')
        
        # Update metadata
        updated_metadata = data.metadata.copy()
        updated_metadata['preprocessing'] = {
            'remove_nodata': remove_nodata,
            'nodata_value': nodata_value,
            'normalize': normalize,
            'correct_bad_bands': correct_bad_bands,
            'bad_bands': bad_bands,
            'smooth': smooth,
            'smooth_sigma': smooth_sigma
        }
        
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
