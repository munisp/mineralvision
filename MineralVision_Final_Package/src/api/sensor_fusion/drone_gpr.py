"""
Drone-Mounted GPR Processing Module for MineralVision.

This module provides enhanced GPR processing for drone-mounted systems:
- Zond Aero series integration (500MHz, 600MHz, 1000MHz, LF)
- Terrain-following flight support
- 3D subsurface visualization
- Horizontal slices and thickness grids
- Ice thickness and bedrock depth mapping
- Integration with drone telemetry

Based on SPH Engineering Zond Aero GPR systems and Radar Systems software.
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate, ndimage
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class ZondAeroModel(Enum):
    """Zond Aero GPR models."""
    ZOND_AERO_500 = "Zond Aero 500"      # 500 MHz, universal
    ZOND_AERO_600 = "Zond Aero 600"      # 600 MHz, compact
    ZOND_AERO_1000 = "Zond Aero 1000"    # 1000 MHz, high resolution
    ZOND_AERO_LF_50 = "Zond Aero LF 50"  # 50 MHz, deep penetration
    ZOND_AERO_LF_75 = "Zond Aero LF 75"  # 75 MHz
    ZOND_AERO_LF_100 = "Zond Aero LF 100"  # 100 MHz
    ZOND_AERO_LF_200 = "Zond Aero LF 200"  # 200 MHz
    MALA_GEODRONE_600 = "MALA GeoDrone 600"  # Standalone sensor


class GPRApplication(Enum):
    """GPR application types."""
    UTILITY_DETECTION = "utility_detection"
    BEDROCK_MAPPING = "bedrock_mapping"
    ICE_THICKNESS = "ice_thickness"
    SOIL_LAYERS = "soil_layers"
    ARCHAEOLOGY = "archaeology"
    MINERAL_EXPLORATION = "mineral_exploration"
    VOID_DETECTION = "void_detection"
    WATER_TABLE = "water_table"


class TerrainFollowingMode(Enum):
    """Terrain following modes."""
    CONSTANT_AGL = "constant_agl"  # Constant altitude above ground
    TERRAIN_FOLLOWING = "terrain_following"  # Follow terrain
    MANUAL = "manual"  # Manual altitude control


@dataclass
class ZondAeroSpec:
    """Zond Aero GPR specifications."""
    model: ZondAeroModel
    center_frequency: float  # MHz
    bandwidth: float  # MHz
    time_window: float  # ns
    samples_per_trace: int
    max_penetration_dry: float  # meters in dry sand
    max_penetration_wet: float  # meters in wet soil
    resolution_vertical: float  # meters
    resolution_horizontal: float  # meters
    weight: float  # kg
    power_consumption: float  # W
    antenna_dimensions: Tuple[float, float, float]  # L x W x H in cm
    shielded: bool
    
    @classmethod
    def get_spec(cls, model: ZondAeroModel) -> 'ZondAeroSpec':
        """Get specifications for a GPR model."""
        specs = {
            ZondAeroModel.ZOND_AERO_500: cls(
                model=ZondAeroModel.ZOND_AERO_500,
                center_frequency=500,
                bandwidth=400,
                time_window=100,
                samples_per_trace=512,
                max_penetration_dry=8.0,
                max_penetration_wet=2.0,
                resolution_vertical=0.05,
                resolution_horizontal=0.10,
                weight=1.8,
                power_consumption=8.0,
                antenna_dimensions=(30, 20, 10),
                shielded=True
            ),
            ZondAeroModel.ZOND_AERO_600: cls(
                model=ZondAeroModel.ZOND_AERO_600,
                center_frequency=600,
                bandwidth=500,
                time_window=80,
                samples_per_trace=512,
                max_penetration_dry=6.0,
                max_penetration_wet=1.5,
                resolution_vertical=0.04,
                resolution_horizontal=0.08,
                weight=1.2,
                power_consumption=6.0,
                antenna_dimensions=(25, 18, 8),
                shielded=True
            ),
            ZondAeroModel.ZOND_AERO_1000: cls(
                model=ZondAeroModel.ZOND_AERO_1000,
                center_frequency=1000,
                bandwidth=800,
                time_window=50,
                samples_per_trace=512,
                max_penetration_dry=3.0,
                max_penetration_wet=0.5,
                resolution_vertical=0.02,
                resolution_horizontal=0.04,
                weight=1.0,
                power_consumption=5.0,
                antenna_dimensions=(20, 15, 6),
                shielded=True
            ),
            ZondAeroModel.ZOND_AERO_LF_50: cls(
                model=ZondAeroModel.ZOND_AERO_LF_50,
                center_frequency=50,
                bandwidth=40,
                time_window=1000,
                samples_per_trace=1024,
                max_penetration_dry=50.0,
                max_penetration_wet=15.0,
                resolution_vertical=0.5,
                resolution_horizontal=1.0,
                weight=8.0,
                power_consumption=15.0,
                antenna_dimensions=(300, 50, 20),
                shielded=False
            ),
            ZondAeroModel.ZOND_AERO_LF_100: cls(
                model=ZondAeroModel.ZOND_AERO_LF_100,
                center_frequency=100,
                bandwidth=80,
                time_window=500,
                samples_per_trace=1024,
                max_penetration_dry=25.0,
                max_penetration_wet=8.0,
                resolution_vertical=0.25,
                resolution_horizontal=0.5,
                weight=5.0,
                power_consumption=12.0,
                antenna_dimensions=(150, 40, 15),
                shielded=False
            ),
            ZondAeroModel.MALA_GEODRONE_600: cls(
                model=ZondAeroModel.MALA_GEODRONE_600,
                center_frequency=600,
                bandwidth=500,
                time_window=80,
                samples_per_trace=512,
                max_penetration_dry=6.0,
                max_penetration_wet=1.5,
                resolution_vertical=0.04,
                resolution_horizontal=0.08,
                weight=1.5,
                power_consumption=7.0,
                antenna_dimensions=(28, 20, 10),
                shielded=True
            )
        }
        return specs.get(model, specs[ZondAeroModel.ZOND_AERO_500])


@dataclass
class DroneGPRConfig:
    """Configuration for drone-mounted GPR survey."""
    gpr_model: ZondAeroModel
    flight_altitude: float  # meters AGL
    flight_speed: float  # m/s
    line_spacing: float  # meters
    terrain_following: TerrainFollowingMode
    sample_rate: float  # traces per second
    stacking: int  # number of stacks
    velocity_model: float  # m/ns for depth conversion
    application: GPRApplication


@dataclass
class DroneGPRTrace:
    """Single GPR trace with drone telemetry."""
    trace_id: int
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_msl: float  # meters above sea level
    altitude_agl: float  # meters above ground level
    heading: float  # degrees
    pitch: float  # degrees
    roll: float  # degrees
    ground_speed: float  # m/s
    data: np.ndarray
    time_axis: np.ndarray  # ns
    quality_flag: int = 0


@dataclass
class DroneGPRLine:
    """GPR survey line with drone telemetry."""
    line_id: str
    traces: List[DroneGPRTrace]
    config: DroneGPRConfig
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert traces to 2D array."""
        return np.column_stack([t.data for t in self.traces])
    
    def get_positions(self) -> np.ndarray:
        """Get trace positions as (lat, lon) array."""
        return np.array([[t.latitude, t.longitude] for t in self.traces])
    
    def get_altitudes(self) -> np.ndarray:
        """Get trace altitudes."""
        return np.array([t.altitude_agl for t in self.traces])
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis."""
        if self.traces:
            return self.traces[0].time_axis
        return np.array([])


@dataclass
class GPR3DVolume:
    """3D GPR volume from multiple survey lines."""
    data: np.ndarray  # 3D array (x, y, z)
    x_coords: np.ndarray  # Easting or longitude
    y_coords: np.ndarray  # Northing or latitude
    z_coords: np.ndarray  # Depth in meters
    velocity_model: float  # m/ns used for depth conversion
    crs: str = "EPSG:4326"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DroneGPRProcessor:
    """
    Process drone-mounted GPR data.
    
    Handles terrain-following corrections, altitude normalization,
    and integration with drone telemetry.
    """
    
    def __init__(self, config: DroneGPRConfig):
        self.config = config
        self.spec = ZondAeroSpec.get_spec(config.gpr_model)
        
    def process_line(self, line: DroneGPRLine) -> DroneGPRLine:
        """
        Process a single survey line.
        
        Applies all standard GPR processing steps plus
        drone-specific corrections.
        """
        data = line.to_array()
        
        # 1. Time-zero correction
        data = self._time_zero_correction(data)
        
        # 2. Dewow filter
        data = self._dewow_filter(data)
        
        # 3. Altitude normalization (for terrain-following)
        if self.config.terrain_following == TerrainFollowingMode.TERRAIN_FOLLOWING:
            data = self._altitude_normalization(data, line.get_altitudes())
            
        # 4. Background removal
        data = self._background_removal(data)
        
        # 5. Bandpass filter
        data = self._bandpass_filter(data)
        
        # 6. Gain application
        data = self._apply_gain(data)
        
        # 7. Migration (optional)
        # data = self._migrate(data)
        
        # Update traces with processed data
        for i, trace in enumerate(line.traces):
            trace.data = data[:, i]
            
        return line
        
    def _time_zero_correction(self, data: np.ndarray) -> np.ndarray:
        """Apply time-zero correction."""
        num_samples, num_traces = data.shape
        corrected = np.zeros_like(data)
        
        # Detect time-zero for each trace
        time_zeros = np.zeros(num_traces, dtype=int)
        threshold = 0.1
        
        for i in range(num_traces):
            trace = data[:, i]
            max_amp = np.max(np.abs(trace))
            threshold_value = threshold * max_amp
            
            above_threshold = np.where(np.abs(trace) > threshold_value)[0]
            if len(above_threshold) > 0:
                time_zeros[i] = above_threshold[0]
                
        # Use median as reference
        ref_t0 = int(np.median(time_zeros))
        
        for i in range(num_traces):
            shift = time_zeros[i] - ref_t0
            if shift > 0:
                corrected[:-shift, i] = data[shift:, i]
            elif shift < 0:
                corrected[-shift:, i] = data[:shift, i]
            else:
                corrected[:, i] = data[:, i]
                
        return corrected
        
    def _dewow_filter(self, data: np.ndarray, window_size: int = 50) -> np.ndarray:
        """Remove low-frequency wow."""
        num_samples, num_traces = data.shape
        dewowed = np.zeros_like(data)
        
        kernel = np.ones(window_size) / window_size
        
        for i in range(num_traces):
            wow = np.convolve(data[:, i], kernel, mode='same')
            dewowed[:, i] = data[:, i] - wow
            
        return dewowed
        
    def _altitude_normalization(self, data: np.ndarray, 
                               altitudes: np.ndarray) -> np.ndarray:
        """
        Normalize data for altitude variations.
        
        Shifts traces to account for varying flight altitude.
        """
        num_samples, num_traces = data.shape
        normalized = np.zeros_like(data)
        
        # Reference altitude
        ref_altitude = np.median(altitudes)
        
        # Calculate sample shift for each trace
        sample_interval_ns = self.spec.time_window / self.spec.samples_per_trace
        velocity = self.config.velocity_model  # m/ns
        
        for i in range(num_traces):
            # Altitude difference in meters
            alt_diff = altitudes[i] - ref_altitude
            
            # Convert to time difference (two-way travel)
            time_diff_ns = 2 * alt_diff / velocity
            
            # Convert to samples
            sample_shift = int(time_diff_ns / sample_interval_ns)
            
            # Apply shift
            if sample_shift > 0:
                normalized[sample_shift:, i] = data[:-sample_shift, i]
            elif sample_shift < 0:
                normalized[:sample_shift, i] = data[-sample_shift:, i]
            else:
                normalized[:, i] = data[:, i]
                
        return normalized
        
    def _background_removal(self, data: np.ndarray) -> np.ndarray:
        """Remove horizontal banding."""
        background = np.mean(data, axis=1)
        return data - background[:, np.newaxis]
        
    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter."""
        num_samples, num_traces = data.shape
        
        # Calculate filter parameters
        sample_rate_ns = self.spec.time_window / self.spec.samples_per_trace
        fs = 1.0 / (sample_rate_ns * 1e-9)  # Hz
        nyquist = fs / 2
        
        # Filter bounds based on antenna frequency
        low_freq = self.spec.center_frequency * 0.3 * 1e6  # Hz
        high_freq = self.spec.center_frequency * 1.5 * 1e6  # Hz
        
        low_norm = min(0.99, max(0.01, low_freq / nyquist))
        high_norm = min(0.99, max(0.01, high_freq / nyquist))
        
        if low_norm >= high_norm:
            return data
            
        # Design filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter
        filtered = np.zeros_like(data)
        for i in range(num_traces):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
            
        return filtered
        
    def _apply_gain(self, data: np.ndarray) -> np.ndarray:
        """Apply time-varying gain."""
        num_samples, num_traces = data.shape
        
        # SEC gain
        time = np.arange(num_samples)
        gain = (time + 1) ** 1.0 * np.exp(0.01 * time)
        gain = gain / gain[0]
        
        return data * gain[:, np.newaxis]


class HorizontalSliceGenerator:
    """
    Generate horizontal slices from GPR data.
    
    Creates depth slices showing plan view of subsurface.
    """
    
    def __init__(self, velocity: float = 0.1):
        self.velocity = velocity  # m/ns
        
    def generate_slice(self, lines: List[DroneGPRLine],
                      depth: float,
                      cell_size: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Generate horizontal slice at specified depth.
        
        Args:
            lines: List of processed GPR lines
            depth: Depth in meters
            cell_size: Grid cell size in meters
            
        Returns:
            Dictionary with gridded slice data
        """
        # Collect all data points
        points = []
        values = []
        
        for line in lines:
            # Convert depth to sample index
            sample_interval_ns = line.traces[0].time_axis[1] - line.traces[0].time_axis[0]
            time_ns = 2 * depth / self.velocity  # Two-way travel time
            sample_idx = int(time_ns / sample_interval_ns)
            
            if sample_idx >= len(line.traces[0].data):
                continue
                
            for trace in line.traces:
                points.append([trace.longitude, trace.latitude])
                values.append(trace.data[sample_idx])
                
        if not points:
            return {}
            
        points = np.array(points)
        values = np.array(values)
        
        # Create grid
        lon_min, lon_max = points[:, 0].min(), points[:, 0].max()
        lat_min, lat_max = points[:, 1].min(), points[:, 1].max()
        
        # Convert cell size to degrees
        lat_center = (lat_min + lat_max) / 2
        lon_step = cell_size / (111000 * np.cos(np.radians(lat_center)))
        lat_step = cell_size / 111000
        
        lon_grid = np.arange(lon_min, lon_max, lon_step)
        lat_grid = np.arange(lat_min, lat_max, lat_step)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate
        grid = self._interpolate_idw(points[:, 0], points[:, 1], values,
                                    lon_mesh, lat_mesh)
        
        return {
            'data': grid,
            'longitude': lon_mesh,
            'latitude': lat_mesh,
            'depth': depth,
            'cell_size': cell_size
        }
        
    def _interpolate_idw(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        xi: np.ndarray, yi: np.ndarray, power: float = 2.0) -> np.ndarray:
        """Inverse distance weighted interpolation."""
        zi = np.zeros_like(xi)
        
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                distances = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
                
                if np.any(distances == 0):
                    zi[i, j] = z[distances == 0][0]
                else:
                    weights = 1.0 / (distances ** power)
                    zi[i, j] = np.sum(weights * z) / np.sum(weights)
                    
        return zi


class ThicknessGridGenerator:
    """
    Generate thickness grids from GPR data.
    
    Maps layer thickness (e.g., ice, soil, overburden).
    """
    
    def __init__(self, velocity: float = 0.1):
        self.velocity = velocity  # m/ns
        
    def detect_interface(self, trace: np.ndarray, 
                        time_axis: np.ndarray,
                        method: str = 'peak') -> Optional[float]:
        """
        Detect interface depth in a single trace.
        
        Args:
            trace: GPR trace data
            time_axis: Time axis in ns
            method: Detection method ('peak', 'threshold', 'gradient')
            
        Returns:
            Interface depth in meters, or None if not detected
        """
        if method == 'peak':
            # Find strongest reflection
            peak_idx = np.argmax(np.abs(trace))
            time_ns = time_axis[peak_idx]
            
        elif method == 'threshold':
            # Find first sample above threshold
            threshold = 0.3 * np.max(np.abs(trace))
            above = np.where(np.abs(trace) > threshold)[0]
            if len(above) == 0:
                return None
            time_ns = time_axis[above[0]]
            
        elif method == 'gradient':
            # Find maximum gradient
            gradient = np.abs(np.diff(trace))
            peak_idx = np.argmax(gradient)
            time_ns = time_axis[peak_idx]
            
        else:
            return None
            
        # Convert to depth (one-way travel)
        depth = time_ns * self.velocity / 2
        
        return depth
        
    def generate_thickness_grid(self, lines: List[DroneGPRLine],
                               cell_size: float = 1.0,
                               method: str = 'peak') -> Dict[str, np.ndarray]:
        """
        Generate thickness grid from GPR lines.
        
        Args:
            lines: List of processed GPR lines
            cell_size: Grid cell size in meters
            method: Interface detection method
            
        Returns:
            Dictionary with thickness grid data
        """
        # Collect thickness measurements
        points = []
        thicknesses = []
        
        for line in lines:
            for trace in line.traces:
                thickness = self.detect_interface(
                    trace.data, trace.time_axis, method
                )
                
                if thickness is not None:
                    points.append([trace.longitude, trace.latitude])
                    thicknesses.append(thickness)
                    
        if not points:
            return {}
            
        points = np.array(points)
        thicknesses = np.array(thicknesses)
        
        # Create grid
        lon_min, lon_max = points[:, 0].min(), points[:, 0].max()
        lat_min, lat_max = points[:, 1].min(), points[:, 1].max()
        
        lat_center = (lat_min + lat_max) / 2
        lon_step = cell_size / (111000 * np.cos(np.radians(lat_center)))
        lat_step = cell_size / 111000
        
        lon_grid = np.arange(lon_min, lon_max, lon_step)
        lat_grid = np.arange(lat_min, lat_max, lat_step)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate
        thickness_grid = self._interpolate_idw(
            points[:, 0], points[:, 1], thicknesses,
            lon_mesh, lat_mesh
        )
        
        return {
            'thickness': thickness_grid,
            'longitude': lon_mesh,
            'latitude': lat_mesh,
            'cell_size': cell_size,
            'method': method,
            'statistics': {
                'min': float(np.nanmin(thickness_grid)),
                'max': float(np.nanmax(thickness_grid)),
                'mean': float(np.nanmean(thickness_grid)),
                'std': float(np.nanstd(thickness_grid))
            }
        }
        
    def _interpolate_idw(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        xi: np.ndarray, yi: np.ndarray, power: float = 2.0) -> np.ndarray:
        """Inverse distance weighted interpolation."""
        zi = np.zeros_like(xi)
        
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                distances = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
                
                if np.any(distances == 0):
                    zi[i, j] = z[distances == 0][0]
                else:
                    weights = 1.0 / (distances ** power)
                    zi[i, j] = np.sum(weights * z) / np.sum(weights)
                    
        return zi


class GPR3DVolumeBuilder:
    """
    Build 3D GPR volumes from multiple survey lines.
    """
    
    def __init__(self, velocity: float = 0.1):
        self.velocity = velocity  # m/ns
        
    def build_volume(self, lines: List[DroneGPRLine],
                    cell_size_xy: float = 0.5,
                    cell_size_z: float = 0.05) -> GPR3DVolume:
        """
        Build 3D volume from GPR lines.
        
        Args:
            lines: List of processed GPR lines
            cell_size_xy: Horizontal cell size in meters
            cell_size_z: Vertical cell size in meters
            
        Returns:
            GPR3DVolume object
        """
        # Collect all data
        all_points = []
        all_data = []
        
        for line in lines:
            for trace in line.traces:
                all_points.append([trace.longitude, trace.latitude])
                all_data.append(trace.data)
                
        all_points = np.array(all_points)
        all_data = np.array(all_data)
        
        # Determine bounds
        lon_min, lon_max = all_points[:, 0].min(), all_points[:, 0].max()
        lat_min, lat_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Get depth range from first trace
        time_axis = lines[0].traces[0].time_axis
        max_depth = time_axis[-1] * self.velocity / 2
        
        # Create grid coordinates
        lat_center = (lat_min + lat_max) / 2
        lon_step = cell_size_xy / (111000 * np.cos(np.radians(lat_center)))
        lat_step = cell_size_xy / 111000
        
        x_coords = np.arange(lon_min, lon_max, lon_step)
        y_coords = np.arange(lat_min, lat_max, lat_step)
        z_coords = np.arange(0, max_depth, cell_size_z)
        
        # Initialize volume
        volume = np.zeros((len(x_coords), len(y_coords), len(z_coords)))
        
        # Interpolate each depth slice
        for k, depth in enumerate(z_coords):
            # Get sample index for this depth
            time_ns = 2 * depth / self.velocity
            sample_idx = int(time_ns / (time_axis[1] - time_axis[0]))
            
            if sample_idx >= all_data.shape[1]:
                continue
                
            # Extract values at this depth
            values = all_data[:, sample_idx]
            
            # Interpolate to grid
            lon_mesh, lat_mesh = np.meshgrid(x_coords, y_coords)
            
            slice_data = self._interpolate_idw(
                all_points[:, 0], all_points[:, 1], values,
                lon_mesh.T, lat_mesh.T
            )
            
            volume[:, :, k] = slice_data
            
        return GPR3DVolume(
            data=volume,
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            velocity_model=self.velocity,
            metadata={
                'n_lines': len(lines),
                'n_traces': len(all_points),
                'cell_size_xy': cell_size_xy,
                'cell_size_z': cell_size_z
            }
        )
        
    def _interpolate_idw(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        xi: np.ndarray, yi: np.ndarray, power: float = 2.0) -> np.ndarray:
        """Inverse distance weighted interpolation."""
        zi = np.zeros_like(xi)
        
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                distances = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
                
                if np.any(distances == 0):
                    zi[i, j] = z[distances == 0][0]
                else:
                    weights = 1.0 / (distances ** power)
                    zi[i, j] = np.sum(weights * z) / np.sum(weights)
                    
        return zi


class DroneGPRFlightPlanner:
    """
    Plan drone flights for GPR surveys.
    """
    
    def __init__(self, gpr_model: ZondAeroModel):
        self.spec = ZondAeroSpec.get_spec(gpr_model)
        
    def plan_survey(self, bounds: Tuple[float, float, float, float],
                   line_spacing: float = None,
                   altitude: float = 2.0,
                   speed: float = 3.0,
                   application: GPRApplication = GPRApplication.UTILITY_DETECTION) -> Dict[str, Any]:
        """
        Plan GPR survey flight.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            line_spacing: Line spacing in meters
            altitude: Flight altitude in meters AGL
            speed: Flight speed in m/s
            application: Survey application type
            
        Returns:
            Flight plan dictionary
        """
        # Default line spacing based on application
        if line_spacing is None:
            line_spacing = self._get_default_line_spacing(application)
            
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Calculate survey dimensions
        lat_center = (min_lat + max_lat) / 2
        width_m = (max_lon - min_lon) * 111000 * np.cos(np.radians(lat_center))
        height_m = (max_lat - min_lat) * 111000
        
        # Calculate number of lines
        n_lines = int(np.ceil(width_m / line_spacing))
        
        # Generate flight lines
        lines = []
        lon_step = (max_lon - min_lon) / n_lines
        
        for i in range(n_lines):
            lon = min_lon + (i + 0.5) * lon_step
            
            if i % 2 == 0:
                start = (lon, min_lat)
                end = (lon, max_lat)
            else:
                start = (lon, max_lat)
                end = (lon, min_lat)
                
            lines.append({
                'line_id': i + 1,
                'start': start,
                'end': end,
                'length_m': height_m
            })
            
        # Calculate flight time
        total_line_length = n_lines * height_m
        turn_time = (n_lines - 1) * 20  # 20 seconds per turn
        flight_time_s = total_line_length / speed + turn_time
        
        # Calculate expected penetration
        expected_penetration = self._estimate_penetration(application)
        
        # Calculate trace spacing
        trace_spacing = speed * 0.1  # Assuming 10 traces/second
        
        return {
            'bounds': bounds,
            'n_lines': n_lines,
            'line_spacing_m': line_spacing,
            'altitude_m': altitude,
            'speed_m_s': speed,
            'lines': lines,
            'total_line_length_m': total_line_length,
            'estimated_flight_time_s': flight_time_s,
            'estimated_flight_time_min': flight_time_s / 60,
            'gpr_model': self.spec.model.value,
            'application': application.value,
            'expected_penetration_m': expected_penetration,
            'trace_spacing_m': trace_spacing,
            'expected_traces': int(total_line_length / trace_spacing)
        }
        
    def _get_default_line_spacing(self, application: GPRApplication) -> float:
        """Get default line spacing for application."""
        spacings = {
            GPRApplication.UTILITY_DETECTION: 1.0,
            GPRApplication.BEDROCK_MAPPING: 5.0,
            GPRApplication.ICE_THICKNESS: 10.0,
            GPRApplication.SOIL_LAYERS: 2.0,
            GPRApplication.ARCHAEOLOGY: 0.5,
            GPRApplication.MINERAL_EXPLORATION: 5.0,
            GPRApplication.VOID_DETECTION: 1.0,
            GPRApplication.WATER_TABLE: 5.0
        }
        return spacings.get(application, 2.0)
        
    def _estimate_penetration(self, application: GPRApplication) -> float:
        """Estimate expected penetration for application."""
        # Based on typical soil conditions
        if application == GPRApplication.ICE_THICKNESS:
            # Ice has very low attenuation
            return self.spec.max_penetration_dry * 3
        elif application in [GPRApplication.BEDROCK_MAPPING, GPRApplication.MINERAL_EXPLORATION]:
            # Assume moderately dry conditions
            return (self.spec.max_penetration_dry + self.spec.max_penetration_wet) / 2
        else:
            # Assume typical soil
            return self.spec.max_penetration_wet * 1.5


class DroneGPRPipeline:
    """
    Complete drone GPR processing pipeline.
    """
    
    def __init__(self, config: DroneGPRConfig):
        self.config = config
        self.processor = DroneGPRProcessor(config)
        self.slice_generator = HorizontalSliceGenerator(config.velocity_model)
        self.thickness_generator = ThicknessGridGenerator(config.velocity_model)
        self.volume_builder = GPR3DVolumeBuilder(config.velocity_model)
        
    def process_survey(self, lines: List[DroneGPRLine]) -> Dict[str, Any]:
        """
        Process complete GPR survey.
        
        Args:
            lines: List of raw GPR lines
            
        Returns:
            Dictionary with all processed products
        """
        # Process each line
        processed_lines = []
        for line in lines:
            processed_line = self.processor.process_line(line)
            processed_lines.append(processed_line)
            
        # Generate products based on application
        products = {
            'processed_lines': processed_lines,
            'n_lines': len(processed_lines),
            'n_traces': sum(len(line.traces) for line in processed_lines)
        }
        
        # Generate horizontal slices at multiple depths
        depths = [0.5, 1.0, 2.0, 3.0, 5.0]
        slices = {}
        for depth in depths:
            slice_data = self.slice_generator.generate_slice(processed_lines, depth)
            if slice_data:
                slices[f'depth_{depth}m'] = slice_data
        products['horizontal_slices'] = slices
        
        # Generate thickness grid
        thickness = self.thickness_generator.generate_thickness_grid(processed_lines)
        if thickness:
            products['thickness_grid'] = thickness
            
        # Build 3D volume (optional, can be memory intensive)
        # volume = self.volume_builder.build_volume(processed_lines)
        # products['volume'] = volume
        
        return products


def create_drone_gpr_pipeline(gpr_model: str = "Zond Aero 500",
                             application: str = "utility_detection",
                             velocity: float = 0.1) -> DroneGPRPipeline:
    """
    Factory function to create drone GPR pipeline.
    
    Args:
        gpr_model: GPR model name
        application: Application type
        velocity: Velocity model in m/ns
        
    Returns:
        DroneGPRPipeline instance
    """
    model_map = {
        "Zond Aero 500": ZondAeroModel.ZOND_AERO_500,
        "Zond Aero 600": ZondAeroModel.ZOND_AERO_600,
        "Zond Aero 1000": ZondAeroModel.ZOND_AERO_1000,
        "Zond Aero LF 50": ZondAeroModel.ZOND_AERO_LF_50,
        "Zond Aero LF 100": ZondAeroModel.ZOND_AERO_LF_100,
        "MALA GeoDrone 600": ZondAeroModel.MALA_GEODRONE_600
    }
    
    app_map = {
        "utility_detection": GPRApplication.UTILITY_DETECTION,
        "bedrock_mapping": GPRApplication.BEDROCK_MAPPING,
        "ice_thickness": GPRApplication.ICE_THICKNESS,
        "soil_layers": GPRApplication.SOIL_LAYERS,
        "archaeology": GPRApplication.ARCHAEOLOGY,
        "mineral_exploration": GPRApplication.MINERAL_EXPLORATION,
        "void_detection": GPRApplication.VOID_DETECTION,
        "water_table": GPRApplication.WATER_TABLE
    }
    
    gpr = model_map.get(gpr_model, ZondAeroModel.ZOND_AERO_500)
    app = app_map.get(application, GPRApplication.UTILITY_DETECTION)
    
    config = DroneGPRConfig(
        gpr_model=gpr,
        flight_altitude=2.0,
        flight_speed=3.0,
        line_spacing=2.0,
        terrain_following=TerrainFollowingMode.TERRAIN_FOLLOWING,
        sample_rate=10.0,
        stacking=4,
        velocity_model=velocity,
        application=app
    )
    
    return DroneGPRPipeline(config)


def create_gpr_flight_planner(gpr_model: str = "Zond Aero 500") -> DroneGPRFlightPlanner:
    """Factory function to create GPR flight planner."""
    model_map = {
        "Zond Aero 500": ZondAeroModel.ZOND_AERO_500,
        "Zond Aero 600": ZondAeroModel.ZOND_AERO_600,
        "Zond Aero 1000": ZondAeroModel.ZOND_AERO_1000,
        "Zond Aero LF 50": ZondAeroModel.ZOND_AERO_LF_50,
        "Zond Aero LF 100": ZondAeroModel.ZOND_AERO_LF_100,
        "MALA GeoDrone 600": ZondAeroModel.MALA_GEODRONE_600
    }
    
    gpr = model_map.get(gpr_model, ZondAeroModel.ZOND_AERO_500)
    return DroneGPRFlightPlanner(gpr)
