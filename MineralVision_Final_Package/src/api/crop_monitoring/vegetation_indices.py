"""
Vegetation Index Processing Module for MineralVision Crop Monitoring.

Comprehensive vegetation index calculation from multispectral satellite imagery:
- NDVI (Normalized Difference Vegetation Index)
- NDRE (Normalized Difference Red Edge)
- SAVI (Soil Adjusted Vegetation Index)
- EVI (Enhanced Vegetation Index)
- GNDVI (Green NDVI)
- MSAVI (Modified SAVI)
- NDWI (Normalized Difference Water Index)
- LAI (Leaf Area Index estimation)

Supports time-series analysis, cloud masking, and atmospheric correction.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import numpy as np
import logging
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VegetationIndexType(Enum):
    """Supported vegetation indices."""
    NDVI = "ndvi"  # Normalized Difference Vegetation Index
    NDRE = "ndre"  # Normalized Difference Red Edge
    SAVI = "savi"  # Soil Adjusted Vegetation Index
    EVI = "evi"   # Enhanced Vegetation Index
    GNDVI = "gndvi"  # Green NDVI
    MSAVI = "msavi"  # Modified SAVI
    NDWI = "ndwi"  # Normalized Difference Water Index
    NDMI = "ndmi"  # Normalized Difference Moisture Index
    CVI = "cvi"   # Chlorophyll Vegetation Index
    TVI = "tvi"   # Triangular Vegetation Index


class CropHealthStatus(Enum):
    """Crop health classification based on vegetation indices."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    STRESSED = "stressed"
    CRITICAL = "critical"
    BARE_SOIL = "bare_soil"
    WATER = "water"


class CloudMaskMethod(Enum):
    """Cloud masking methods."""
    THRESHOLD = "threshold"
    FMASK = "fmask"
    SEN2COR = "sen2cor"
    ML_BASED = "ml_based"


@dataclass
class SpectralBands:
    """Multispectral band data from satellite imagery."""
    scene_id: str
    acquisition_date: datetime
    
    # Core bands (reflectance values 0-1)
    blue: Optional[np.ndarray] = None  # ~450-520nm
    green: Optional[np.ndarray] = None  # ~520-600nm
    red: Optional[np.ndarray] = None  # ~630-690nm
    red_edge: Optional[np.ndarray] = None  # ~705-745nm
    nir: Optional[np.ndarray] = None  # ~760-900nm
    swir1: Optional[np.ndarray] = None  # ~1550-1750nm
    swir2: Optional[np.ndarray] = None  # ~2080-2350nm
    
    # Additional bands
    coastal: Optional[np.ndarray] = None  # ~400-450nm
    yellow: Optional[np.ndarray] = None  # ~585-625nm
    nir2: Optional[np.ndarray] = None  # ~860-1040nm
    
    # Metadata
    cloud_mask: Optional[np.ndarray] = None
    quality_mask: Optional[np.ndarray] = None
    sun_elevation: float = 45.0
    sun_azimuth: float = 135.0
    sensor_zenith: float = 0.0
    
    # Geospatial info
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)  # minx, miny, maxx, maxy
    crs: str = "EPSG:4326"
    resolution_m: float = 10.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VegetationIndexResult:
    """Result of vegetation index calculation."""
    index_type: VegetationIndexType
    scene_id: str
    acquisition_date: datetime
    
    # Index values
    values: np.ndarray  # 2D array of index values
    min_value: float = -1.0
    max_value: float = 1.0
    mean_value: float = 0.0
    std_value: float = 0.0
    
    # Statistics by zone
    zone_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Health classification
    health_classification: Optional[np.ndarray] = None
    health_distribution: Dict[CropHealthStatus, float] = field(default_factory=dict)
    
    # Quality
    valid_pixel_percent: float = 100.0
    cloud_cover_percent: float = 0.0
    
    # Geospatial
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
    crs: str = "EPSG:4326"
    resolution_m: float = 10.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index_type': self.index_type.value,
            'scene_id': self.scene_id,
            'acquisition_date': self.acquisition_date.isoformat(),
            'min_value': float(self.min_value),
            'max_value': float(self.max_value),
            'mean_value': float(self.mean_value),
            'std_value': float(self.std_value),
            'valid_pixel_percent': self.valid_pixel_percent,
            'cloud_cover_percent': self.cloud_cover_percent,
            'health_distribution': {k.value: v for k, v in self.health_distribution.items()},
            'bounds': self.bounds,
            'crs': self.crs,
            'resolution_m': self.resolution_m
        }


@dataclass
class TimeSeriesPoint:
    """Single point in vegetation index time series."""
    date: datetime
    value: float
    quality: float  # 0-1, confidence in measurement
    cloud_free: bool = True
    interpolated: bool = False


@dataclass
class VegetationTimeSeries:
    """Time series of vegetation index values."""
    field_id: str
    index_type: VegetationIndexType
    start_date: datetime
    end_date: datetime
    
    # Time series data
    points: List[TimeSeriesPoint] = field(default_factory=list)
    
    # Trend analysis
    trend_slope: float = 0.0  # Change per day
    trend_direction: str = "stable"  # increasing, decreasing, stable
    seasonality_detected: bool = False
    
    # Anomaly detection
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    mean_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    std_value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_id': self.field_id,
            'index_type': self.index_type.value,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'points': [
                {'date': p.date.isoformat(), 'value': p.value, 'quality': p.quality}
                for p in self.points
            ],
            'trend_slope': self.trend_slope,
            'trend_direction': self.trend_direction,
            'anomalies': self.anomalies,
            'mean_value': self.mean_value,
            'min_value': self.min_value,
            'max_value': self.max_value
        }


class VegetationIndexCalculator:
    """Calculate vegetation indices from multispectral imagery."""
    
    # NDVI thresholds for health classification
    NDVI_THRESHOLDS = {
        CropHealthStatus.WATER: (-1.0, 0.0),
        CropHealthStatus.BARE_SOIL: (0.0, 0.15),
        CropHealthStatus.CRITICAL: (0.15, 0.25),
        CropHealthStatus.STRESSED: (0.25, 0.40),
        CropHealthStatus.MODERATE: (0.40, 0.55),
        CropHealthStatus.GOOD: (0.55, 0.70),
        CropHealthStatus.EXCELLENT: (0.70, 1.0)
    }
    
    # Crop-specific NDVI thresholds
    CROP_THRESHOLDS = {
        'oil_palm': {
            CropHealthStatus.CRITICAL: (0.15, 0.30),
            CropHealthStatus.STRESSED: (0.30, 0.45),
            CropHealthStatus.MODERATE: (0.45, 0.60),
            CropHealthStatus.GOOD: (0.60, 0.75),
            CropHealthStatus.EXCELLENT: (0.75, 1.0)
        },
        'cocoa': {
            CropHealthStatus.CRITICAL: (0.15, 0.28),
            CropHealthStatus.STRESSED: (0.28, 0.42),
            CropHealthStatus.MODERATE: (0.42, 0.55),
            CropHealthStatus.GOOD: (0.55, 0.68),
            CropHealthStatus.EXCELLENT: (0.68, 1.0)
        },
        'ginger': {
            CropHealthStatus.CRITICAL: (0.10, 0.22),
            CropHealthStatus.STRESSED: (0.22, 0.35),
            CropHealthStatus.MODERATE: (0.35, 0.48),
            CropHealthStatus.GOOD: (0.48, 0.62),
            CropHealthStatus.EXCELLENT: (0.62, 1.0)
        }
    }
    
    def __init__(self, crop_type: str = "general"):
        self.crop_type = crop_type
        self.thresholds = self.CROP_THRESHOLDS.get(crop_type, self.NDVI_THRESHOLDS)
    
    def calculate_ndvi(self, bands: SpectralBands) -> VegetationIndexResult:
        """
        Calculate NDVI (Normalized Difference Vegetation Index).
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1
        """
        if bands.nir is None or bands.red is None:
            raise ValueError("NIR and Red bands required for NDVI")
        
        # Avoid division by zero
        denominator = bands.nir + bands.red
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        ndvi = (bands.nir - bands.red) / denominator
        ndvi = np.clip(ndvi, -1, 1)
        
        # Apply cloud mask if available
        if bands.cloud_mask is not None:
            ndvi = np.where(bands.cloud_mask, np.nan, ndvi)
        
        return self._create_result(
            VegetationIndexType.NDVI, bands, ndvi
        )
    
    def calculate_ndre(self, bands: SpectralBands) -> VegetationIndexResult:
        """
        Calculate NDRE (Normalized Difference Red Edge).
        NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        Better for detecting chlorophyll content in mature vegetation.
        """
        if bands.nir is None or bands.red_edge is None:
            raise ValueError("NIR and Red Edge bands required for NDRE")
        
        denominator = bands.nir + bands.red_edge
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        ndre = (bands.nir - bands.red_edge) / denominator
        ndre = np.clip(ndre, -1, 1)
        
        if bands.cloud_mask is not None:
            ndre = np.where(bands.cloud_mask, np.nan, ndre)
        
        return self._create_result(
            VegetationIndexType.NDRE, bands, ndre
        )
    
    def calculate_savi(self, bands: SpectralBands, L: float = 0.5) -> VegetationIndexResult:
        """
        Calculate SAVI (Soil Adjusted Vegetation Index).
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        L = soil brightness correction factor (0.5 for intermediate vegetation)
        Better for areas with exposed soil.
        """
        if bands.nir is None or bands.red is None:
            raise ValueError("NIR and Red bands required for SAVI")
        
        denominator = bands.nir + bands.red + L
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        savi = ((bands.nir - bands.red) / denominator) * (1 + L)
        savi = np.clip(savi, -1, 1)
        
        if bands.cloud_mask is not None:
            savi = np.where(bands.cloud_mask, np.nan, savi)
        
        return self._create_result(
            VegetationIndexType.SAVI, bands, savi
        )
    
    def calculate_evi(self, bands: SpectralBands, 
                      G: float = 2.5, C1: float = 6.0, 
                      C2: float = 7.5, L: float = 1.0) -> VegetationIndexResult:
        """
        Calculate EVI (Enhanced Vegetation Index).
        EVI = G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))
        Better for high biomass regions, reduces atmospheric influences.
        """
        if bands.nir is None or bands.red is None or bands.blue is None:
            raise ValueError("NIR, Red, and Blue bands required for EVI")
        
        denominator = bands.nir + C1 * bands.red - C2 * bands.blue + L
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        evi = G * ((bands.nir - bands.red) / denominator)
        evi = np.clip(evi, -1, 1)
        
        if bands.cloud_mask is not None:
            evi = np.where(bands.cloud_mask, np.nan, evi)
        
        return self._create_result(
            VegetationIndexType.EVI, bands, evi
        )
    
    def calculate_gndvi(self, bands: SpectralBands) -> VegetationIndexResult:
        """
        Calculate GNDVI (Green NDVI).
        GNDVI = (NIR - Green) / (NIR + Green)
        More sensitive to chlorophyll concentration.
        """
        if bands.nir is None or bands.green is None:
            raise ValueError("NIR and Green bands required for GNDVI")
        
        denominator = bands.nir + bands.green
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        gndvi = (bands.nir - bands.green) / denominator
        gndvi = np.clip(gndvi, -1, 1)
        
        if bands.cloud_mask is not None:
            gndvi = np.where(bands.cloud_mask, np.nan, gndvi)
        
        return self._create_result(
            VegetationIndexType.GNDVI, bands, gndvi
        )
    
    def calculate_msavi(self, bands: SpectralBands) -> VegetationIndexResult:
        """
        Calculate MSAVI (Modified Soil Adjusted Vegetation Index).
        MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2
        Self-adjusting L factor, better for sparse vegetation.
        """
        if bands.nir is None or bands.red is None:
            raise ValueError("NIR and Red bands required for MSAVI")
        
        term = (2 * bands.nir + 1) ** 2 - 8 * (bands.nir - bands.red)
        term = np.maximum(term, 0)  # Avoid negative sqrt
        
        msavi = (2 * bands.nir + 1 - np.sqrt(term)) / 2
        msavi = np.clip(msavi, -1, 1)
        
        if bands.cloud_mask is not None:
            msavi = np.where(bands.cloud_mask, np.nan, msavi)
        
        return self._create_result(
            VegetationIndexType.MSAVI, bands, msavi
        )
    
    def calculate_ndwi(self, bands: SpectralBands) -> VegetationIndexResult:
        """
        Calculate NDWI (Normalized Difference Water Index).
        NDWI = (Green - NIR) / (Green + NIR)
        Detects water content in vegetation.
        """
        if bands.nir is None or bands.green is None:
            raise ValueError("NIR and Green bands required for NDWI")
        
        denominator = bands.green + bands.nir
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        ndwi = (bands.green - bands.nir) / denominator
        ndwi = np.clip(ndwi, -1, 1)
        
        if bands.cloud_mask is not None:
            ndwi = np.where(bands.cloud_mask, np.nan, ndwi)
        
        return self._create_result(
            VegetationIndexType.NDWI, bands, ndwi
        )
    
    def calculate_ndmi(self, bands: SpectralBands) -> VegetationIndexResult:
        """
        Calculate NDMI (Normalized Difference Moisture Index).
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        Detects vegetation water stress.
        """
        if bands.nir is None or bands.swir1 is None:
            raise ValueError("NIR and SWIR1 bands required for NDMI")
        
        denominator = bands.nir + bands.swir1
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        ndmi = (bands.nir - bands.swir1) / denominator
        ndmi = np.clip(ndmi, -1, 1)
        
        if bands.cloud_mask is not None:
            ndmi = np.where(bands.cloud_mask, np.nan, ndmi)
        
        return self._create_result(
            VegetationIndexType.NDMI, bands, ndmi
        )
    
    def calculate_all_indices(self, bands: SpectralBands) -> Dict[VegetationIndexType, VegetationIndexResult]:
        """Calculate all available vegetation indices."""
        results = {}
        
        # Always calculate NDVI if possible
        if bands.nir is not None and bands.red is not None:
            results[VegetationIndexType.NDVI] = self.calculate_ndvi(bands)
            results[VegetationIndexType.SAVI] = self.calculate_savi(bands)
            results[VegetationIndexType.MSAVI] = self.calculate_msavi(bands)
        
        # NDRE requires red edge
        if bands.nir is not None and bands.red_edge is not None:
            results[VegetationIndexType.NDRE] = self.calculate_ndre(bands)
        
        # EVI requires blue
        if bands.nir is not None and bands.red is not None and bands.blue is not None:
            results[VegetationIndexType.EVI] = self.calculate_evi(bands)
        
        # GNDVI and NDWI require green
        if bands.nir is not None and bands.green is not None:
            results[VegetationIndexType.GNDVI] = self.calculate_gndvi(bands)
            results[VegetationIndexType.NDWI] = self.calculate_ndwi(bands)
        
        # NDMI requires SWIR
        if bands.nir is not None and bands.swir1 is not None:
            results[VegetationIndexType.NDMI] = self.calculate_ndmi(bands)
        
        return results
    
    def _create_result(self, index_type: VegetationIndexType, 
                       bands: SpectralBands, values: np.ndarray) -> VegetationIndexResult:
        """Create vegetation index result with statistics."""
        # Calculate statistics (ignoring NaN)
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            min_val = float(np.min(valid_values))
            max_val = float(np.max(valid_values))
            mean_val = float(np.mean(valid_values))
            std_val = float(np.std(valid_values))
        else:
            min_val = max_val = mean_val = std_val = 0.0
        
        # Calculate valid pixel percentage
        total_pixels = values.size
        valid_pixels = len(valid_values)
        valid_percent = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Cloud cover percentage
        cloud_percent = 0.0
        if bands.cloud_mask is not None:
            cloud_percent = (np.sum(bands.cloud_mask) / bands.cloud_mask.size) * 100
        
        # Health classification
        health_class, health_dist = self._classify_health(values, index_type)
        
        return VegetationIndexResult(
            index_type=index_type,
            scene_id=bands.scene_id,
            acquisition_date=bands.acquisition_date,
            values=values,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            health_classification=health_class,
            health_distribution=health_dist,
            valid_pixel_percent=valid_percent,
            cloud_cover_percent=cloud_percent,
            bounds=bands.bounds,
            crs=bands.crs,
            resolution_m=bands.resolution_m
        )
    
    def _classify_health(self, values: np.ndarray, 
                        index_type: VegetationIndexType) -> Tuple[np.ndarray, Dict[CropHealthStatus, float]]:
        """Classify vegetation health based on index values."""
        health_class = np.full(values.shape, -1, dtype=np.int8)
        health_dist = {}
        
        # Use NDVI thresholds for classification
        thresholds = self.thresholds if index_type == VegetationIndexType.NDVI else self.NDVI_THRESHOLDS
        
        total_valid = np.sum(~np.isnan(values))
        
        for status, (low, high) in thresholds.items():
            mask = (values >= low) & (values < high) & (~np.isnan(values))
            health_class[mask] = list(CropHealthStatus).index(status)
            
            count = np.sum(mask)
            health_dist[status] = (count / total_valid * 100) if total_valid > 0 else 0.0
        
        return health_class, health_dist


class CloudMasker:
    """Cloud detection and masking for satellite imagery."""
    
    def __init__(self, method: CloudMaskMethod = CloudMaskMethod.THRESHOLD):
        self.method = method
    
    def create_cloud_mask(self, bands: SpectralBands) -> np.ndarray:
        """Create cloud mask from spectral bands."""
        if self.method == CloudMaskMethod.THRESHOLD:
            return self._threshold_mask(bands)
        elif self.method == CloudMaskMethod.FMASK:
            return self._fmask(bands)
        else:
            return self._threshold_mask(bands)
    
    def _threshold_mask(self, bands: SpectralBands) -> np.ndarray:
        """Simple threshold-based cloud detection."""
        if bands.blue is None:
            return np.zeros(bands.nir.shape, dtype=bool) if bands.nir is not None else np.array([])
        
        # High blue reflectance indicates clouds
        cloud_mask = bands.blue > 0.25
        
        # High brightness in all visible bands
        if bands.green is not None and bands.red is not None:
            brightness = (bands.blue + bands.green + bands.red) / 3
            cloud_mask = cloud_mask | (brightness > 0.3)
        
        return cloud_mask
    
    def _fmask(self, bands: SpectralBands) -> np.ndarray:
        """Simplified Fmask algorithm for cloud detection."""
        if bands.blue is None or bands.nir is None:
            return np.zeros(bands.nir.shape if bands.nir is not None else (1,), dtype=bool)
        
        # Basic cloud test
        cloud_mask = bands.blue > 0.2
        
        # Whiteness test
        if bands.green is not None and bands.red is not None:
            mean_vis = (bands.blue + bands.green + bands.red) / 3
            whiteness = (
                np.abs(bands.blue - mean_vis) +
                np.abs(bands.green - mean_vis) +
                np.abs(bands.red - mean_vis)
            ) / mean_vis
            cloud_mask = cloud_mask & (whiteness < 0.7)
        
        # HOT (Haze Optimized Transform)
        if bands.red is not None:
            hot = bands.blue - 0.5 * bands.red - 0.08
            cloud_mask = cloud_mask | (hot > 0)
        
        return cloud_mask


class AtmosphericCorrector:
    """Atmospheric correction for satellite imagery."""
    
    def __init__(self):
        self.dos_offset = 0.01  # Dark Object Subtraction offset
    
    def apply_dos(self, bands: SpectralBands) -> SpectralBands:
        """Apply Dark Object Subtraction (DOS) correction."""
        corrected = SpectralBands(
            scene_id=bands.scene_id,
            acquisition_date=bands.acquisition_date,
            bounds=bands.bounds,
            crs=bands.crs,
            resolution_m=bands.resolution_m,
            sun_elevation=bands.sun_elevation,
            sun_azimuth=bands.sun_azimuth,
            cloud_mask=bands.cloud_mask,
            quality_mask=bands.quality_mask,
            metadata=bands.metadata
        )
        
        # Apply DOS to each band
        for band_name in ['blue', 'green', 'red', 'red_edge', 'nir', 'swir1', 'swir2']:
            band_data = getattr(bands, band_name)
            if band_data is not None:
                # Find dark object value (1st percentile)
                dark_value = np.percentile(band_data[band_data > 0], 1) if np.any(band_data > 0) else 0
                corrected_data = band_data - dark_value + self.dos_offset
                corrected_data = np.clip(corrected_data, 0, 1)
                setattr(corrected, band_name, corrected_data)
        
        return corrected
    
    def apply_sun_angle_correction(self, bands: SpectralBands) -> SpectralBands:
        """Apply sun angle correction."""
        if bands.sun_elevation <= 0:
            return bands
        
        # Calculate correction factor
        sun_zenith_rad = np.radians(90 - bands.sun_elevation)
        correction_factor = 1 / np.cos(sun_zenith_rad)
        
        corrected = SpectralBands(
            scene_id=bands.scene_id,
            acquisition_date=bands.acquisition_date,
            bounds=bands.bounds,
            crs=bands.crs,
            resolution_m=bands.resolution_m,
            sun_elevation=bands.sun_elevation,
            sun_azimuth=bands.sun_azimuth,
            cloud_mask=bands.cloud_mask,
            quality_mask=bands.quality_mask,
            metadata=bands.metadata
        )
        
        for band_name in ['blue', 'green', 'red', 'red_edge', 'nir', 'swir1', 'swir2']:
            band_data = getattr(bands, band_name)
            if band_data is not None:
                corrected_data = band_data * correction_factor
                corrected_data = np.clip(corrected_data, 0, 1)
                setattr(corrected, band_name, corrected_data)
        
        return corrected


class TimeSeriesAnalyzer:
    """Analyze vegetation index time series."""
    
    def __init__(self):
        self.min_points_for_trend = 5
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def create_time_series(
        self,
        field_id: str,
        index_results: List[VegetationIndexResult],
        index_type: VegetationIndexType
    ) -> VegetationTimeSeries:
        """Create time series from multiple index results."""
        # Filter results for the specified index type
        filtered = [r for r in index_results if r.index_type == index_type]
        
        if not filtered:
            return VegetationTimeSeries(
                field_id=field_id,
                index_type=index_type,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow()
            )
        
        # Sort by date
        filtered.sort(key=lambda x: x.acquisition_date)
        
        # Create time series points
        points = []
        for result in filtered:
            quality = result.valid_pixel_percent / 100
            points.append(TimeSeriesPoint(
                date=result.acquisition_date,
                value=result.mean_value,
                quality=quality,
                cloud_free=result.cloud_cover_percent < 20
            ))
        
        # Calculate statistics
        values = [p.value for p in points]
        
        ts = VegetationTimeSeries(
            field_id=field_id,
            index_type=index_type,
            start_date=filtered[0].acquisition_date,
            end_date=filtered[-1].acquisition_date,
            points=points,
            mean_value=float(np.mean(values)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            std_value=float(np.std(values))
        )
        
        # Analyze trend
        ts = self._analyze_trend(ts)
        
        # Detect anomalies
        ts = self._detect_anomalies(ts)
        
        return ts
    
    def _analyze_trend(self, ts: VegetationTimeSeries) -> VegetationTimeSeries:
        """Analyze trend in time series."""
        if len(ts.points) < self.min_points_for_trend:
            ts.trend_direction = "insufficient_data"
            return ts
        
        # Simple linear regression
        dates = [(p.date - ts.start_date).days for p in ts.points]
        values = [p.value for p in ts.points]
        
        if len(dates) > 1:
            # Calculate slope using least squares
            n = len(dates)
            sum_x = sum(dates)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(dates, values))
            sum_x2 = sum(x ** 2 for x in dates)
            
            denominator = n * sum_x2 - sum_x ** 2
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                ts.trend_slope = slope
                
                # Classify trend direction
                if slope > 0.001:
                    ts.trend_direction = "increasing"
                elif slope < -0.001:
                    ts.trend_direction = "decreasing"
                else:
                    ts.trend_direction = "stable"
        
        return ts
    
    def _detect_anomalies(self, ts: VegetationTimeSeries) -> VegetationTimeSeries:
        """Detect anomalies in time series."""
        if len(ts.points) < 3:
            return ts
        
        values = [p.value for p in ts.points]
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return ts
        
        anomalies = []
        for i, point in enumerate(ts.points):
            z_score = abs(point.value - mean) / std
            if z_score > self.anomaly_threshold:
                anomalies.append({
                    'date': point.date.isoformat(),
                    'value': point.value,
                    'z_score': float(z_score),
                    'type': 'high' if point.value > mean else 'low'
                })
        
        ts.anomalies = anomalies
        return ts
    
    def interpolate_gaps(self, ts: VegetationTimeSeries, 
                        target_interval_days: int = 5) -> VegetationTimeSeries:
        """Interpolate gaps in time series."""
        if len(ts.points) < 2:
            return ts
        
        new_points = []
        
        for i in range(len(ts.points) - 1):
            current = ts.points[i]
            next_point = ts.points[i + 1]
            
            new_points.append(current)
            
            # Calculate gap
            gap_days = (next_point.date - current.date).days
            
            if gap_days > target_interval_days:
                # Interpolate
                num_interpolated = gap_days // target_interval_days - 1
                for j in range(1, num_interpolated + 1):
                    interp_date = current.date + timedelta(days=j * target_interval_days)
                    # Linear interpolation
                    t = j * target_interval_days / gap_days
                    interp_value = current.value + t * (next_point.value - current.value)
                    
                    new_points.append(TimeSeriesPoint(
                        date=interp_date,
                        value=interp_value,
                        quality=0.5,  # Lower quality for interpolated
                        cloud_free=True,
                        interpolated=True
                    ))
        
        new_points.append(ts.points[-1])
        ts.points = new_points
        
        return ts


class CropMonitoringService:
    """Main service for crop monitoring using vegetation indices."""
    
    def __init__(self, crop_type: str = "oil_palm"):
        self.crop_type = crop_type
        self.calculator = VegetationIndexCalculator(crop_type)
        self.cloud_masker = CloudMasker()
        self.atm_corrector = AtmosphericCorrector()
        self.ts_analyzer = TimeSeriesAnalyzer()
        
        # Cache for results
        self._results_cache: Dict[str, List[VegetationIndexResult]] = {}
    
    def process_imagery(
        self,
        bands: SpectralBands,
        apply_cloud_mask: bool = True,
        apply_atm_correction: bool = True
    ) -> Dict[VegetationIndexType, VegetationIndexResult]:
        """Process satellite imagery and calculate vegetation indices."""
        # Apply cloud masking
        if apply_cloud_mask and bands.cloud_mask is None:
            bands.cloud_mask = self.cloud_masker.create_cloud_mask(bands)
        
        # Apply atmospheric correction
        if apply_atm_correction:
            bands = self.atm_corrector.apply_dos(bands)
            bands = self.atm_corrector.apply_sun_angle_correction(bands)
        
        # Calculate all indices
        results = self.calculator.calculate_all_indices(bands)
        
        # Cache results
        for index_type, result in results.items():
            cache_key = f"{bands.scene_id}_{index_type.value}"
            if cache_key not in self._results_cache:
                self._results_cache[cache_key] = []
            self._results_cache[cache_key].append(result)
        
        return results
    
    def get_field_health_summary(
        self,
        field_id: str,
        ndvi_result: VegetationIndexResult
    ) -> Dict[str, Any]:
        """Get health summary for a field."""
        return {
            'field_id': field_id,
            'date': ndvi_result.acquisition_date.isoformat(),
            'mean_ndvi': ndvi_result.mean_value,
            'health_status': self._get_overall_health(ndvi_result),
            'health_distribution': {
                k.value: v for k, v in ndvi_result.health_distribution.items()
            },
            'problem_areas_percent': sum(
                v for k, v in ndvi_result.health_distribution.items()
                if k in [CropHealthStatus.CRITICAL, CropHealthStatus.STRESSED]
            ),
            'recommendations': self._get_recommendations(ndvi_result)
        }
    
    def _get_overall_health(self, result: VegetationIndexResult) -> str:
        """Determine overall health status."""
        # Find dominant health class
        max_percent = 0
        dominant_status = CropHealthStatus.MODERATE
        
        for status, percent in result.health_distribution.items():
            if percent > max_percent and status not in [CropHealthStatus.WATER, CropHealthStatus.BARE_SOIL]:
                max_percent = percent
                dominant_status = status
        
        return dominant_status.value
    
    def _get_recommendations(self, result: VegetationIndexResult) -> List[str]:
        """Generate recommendations based on vegetation index results."""
        recommendations = []
        
        critical_percent = result.health_distribution.get(CropHealthStatus.CRITICAL, 0)
        stressed_percent = result.health_distribution.get(CropHealthStatus.STRESSED, 0)
        
        if critical_percent > 10:
            recommendations.append(
                f"URGENT: {critical_percent:.1f}% of field shows critical vegetation stress. "
                "Immediate field inspection recommended."
            )
        
        if stressed_percent > 20:
            recommendations.append(
                f"WARNING: {stressed_percent:.1f}% of field shows vegetation stress. "
                "Check for water stress, nutrient deficiency, or pest/disease issues."
            )
        
        if result.mean_value < 0.3:
            recommendations.append(
                "Low overall vegetation vigor detected. Consider soil testing and "
                "fertilization program review."
            )
        
        if result.cloud_cover_percent > 30:
            recommendations.append(
                f"High cloud cover ({result.cloud_cover_percent:.1f}%) may affect accuracy. "
                "Consider requesting new imagery."
            )
        
        if not recommendations:
            recommendations.append(
                "Vegetation health appears normal. Continue regular monitoring."
            )
        
        return recommendations
    
    def analyze_time_series(
        self,
        field_id: str,
        results: List[VegetationIndexResult],
        index_type: VegetationIndexType = VegetationIndexType.NDVI
    ) -> VegetationTimeSeries:
        """Analyze vegetation index time series for a field."""
        ts = self.ts_analyzer.create_time_series(field_id, results, index_type)
        return self.ts_analyzer.interpolate_gaps(ts)


def create_crop_monitoring_service(crop_type: str = "oil_palm") -> CropMonitoringService:
    """Factory function to create crop monitoring service."""
    return CropMonitoringService(crop_type)


def create_synthetic_spectral_bands(
    width: int = 100,
    height: int = 100,
    scene_id: str = "synthetic_001",
    vegetation_coverage: float = 0.7
) -> SpectralBands:
    """Create synthetic spectral bands for testing."""
    np.random.seed(42)
    
    # Create base vegetation pattern
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    
    # Vegetation pattern with some variation
    veg_pattern = 0.5 + 0.3 * np.sin(xx) * np.cos(yy)
    veg_pattern += np.random.normal(0, 0.05, veg_pattern.shape)
    veg_pattern = np.clip(veg_pattern, 0, 1)
    
    # Create spectral bands based on vegetation pattern
    # Healthy vegetation: high NIR, low red
    nir = 0.3 + 0.5 * veg_pattern + np.random.normal(0, 0.02, veg_pattern.shape)
    red = 0.1 + 0.1 * (1 - veg_pattern) + np.random.normal(0, 0.02, veg_pattern.shape)
    green = 0.1 + 0.15 * veg_pattern + np.random.normal(0, 0.02, veg_pattern.shape)
    blue = 0.08 + 0.05 * (1 - veg_pattern) + np.random.normal(0, 0.02, veg_pattern.shape)
    red_edge = 0.2 + 0.3 * veg_pattern + np.random.normal(0, 0.02, veg_pattern.shape)
    
    # Clip to valid range
    nir = np.clip(nir, 0, 1)
    red = np.clip(red, 0, 1)
    green = np.clip(green, 0, 1)
    blue = np.clip(blue, 0, 1)
    red_edge = np.clip(red_edge, 0, 1)
    
    return SpectralBands(
        scene_id=scene_id,
        acquisition_date=datetime.utcnow(),
        blue=blue,
        green=green,
        red=red,
        red_edge=red_edge,
        nir=nir,
        bounds=(0, 0, 0.01, 0.01),
        crs="EPSG:4326",
        resolution_m=10.0
    )
