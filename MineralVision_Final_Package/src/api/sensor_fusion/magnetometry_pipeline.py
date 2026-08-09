"""
Airborne Magnetometry Processing Pipeline for MineralVision.

This module provides comprehensive magnetometry data processing including:
- High-rate sampling support (up to 1kHz time-series)
- Platform magnetic compensation
- Diurnal correction with base-station tie-in
- IGRF (International Geomagnetic Reference Field) removal
- Tie-line leveling and micro-leveling
- Derived products (TMI, analytic signal, derivatives, RTP)
- Survey design templates for autonomous exploration

Based on SPH Engineering drone magnetometry best practices.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate, signal, ndimage
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class MagneticFieldComponent(Enum):
    """Magnetic field components."""
    TOTAL = "total"
    X = "x"
    Y = "y"
    Z = "z"
    HORIZONTAL = "horizontal"
    INCLINATION = "inclination"
    DECLINATION = "declination"


class LevelingMethod(Enum):
    """Tie-line leveling methods."""
    POLYNOMIAL = "polynomial"
    SPLINE = "spline"
    MINIMUM_CURVATURE = "minimum_curvature"
    MICROLEVELING = "microleveling"


class DerivativeType(Enum):
    """Types of magnetic derivatives."""
    FIRST_VERTICAL = "first_vertical_derivative"
    SECOND_VERTICAL = "second_vertical_derivative"
    FIRST_HORIZONTAL_X = "first_horizontal_x"
    FIRST_HORIZONTAL_Y = "first_horizontal_y"
    TOTAL_HORIZONTAL = "total_horizontal_gradient"
    ANALYTIC_SIGNAL = "analytic_signal"
    TILT_DERIVATIVE = "tilt_derivative"
    THETA_MAP = "theta_map"


@dataclass
class MagnetometerConfig:
    """Magnetometer sensor configuration."""
    sensor_id: str
    sensor_type: str  # fluxgate, optically_pumped, proton_precession
    sample_rate: float  # Hz
    sensitivity: float  # nT
    noise_level: float  # nT/sqrt(Hz)
    bandwidth: float  # Hz
    position_offset: Tuple[float, float, float] = (0, 0, 0)  # x, y, z offset from GPS
    calibration_date: Optional[datetime] = None
    calibration_coefficients: Dict[str, float] = field(default_factory=dict)


@dataclass
class SurveyLine:
    """Survey line data."""
    line_id: str
    line_type: str  # traverse, tie
    timestamps: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    magnetic_field: np.ndarray
    heading: Optional[np.ndarray] = None
    pitch: Optional[np.ndarray] = None
    roll: Optional[np.ndarray] = None
    quality_flags: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IGRFModel:
    """IGRF model coefficients."""
    epoch: float
    coefficients: Dict[str, float]
    
    def compute_field(self, lat: float, lon: float, alt: float, 
                     date: datetime) -> Dict[str, float]:
        """
        Compute IGRF field at a location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in km above sea level
            date: Date for field computation
            
        Returns:
            Dictionary with field components (total, x, y, z, inclination, declination)
        """
        # Simplified IGRF computation
        # In production, use pyIGRF or similar library
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Earth's mean radius
        R = 6371.2  # km
        
        # Simplified dipole model
        g10 = -29404.8  # nT (IGRF-13 epoch 2020)
        g11 = -1450.9
        h11 = 4652.5
        
        # Compute field components (simplified)
        r = R + alt
        theta = np.pi/2 - lat_rad
        phi = lon_rad
        
        # Radial component
        Br = 2 * (R/r)**3 * (g10 * np.cos(theta) + 
                            (g11 * np.cos(phi) + h11 * np.sin(phi)) * np.sin(theta))
        
        # Theta component
        Bt = (R/r)**3 * (g10 * np.sin(theta) - 
                        (g11 * np.cos(phi) + h11 * np.sin(phi)) * np.cos(theta))
        
        # Phi component
        Bp = (R/r)**3 * (g11 * np.sin(phi) - h11 * np.cos(phi))
        
        # Convert to X, Y, Z (North, East, Down)
        X = -Bt
        Y = Bp
        Z = -Br
        
        # Total field
        F = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Horizontal component
        H = np.sqrt(X**2 + Y**2)
        
        # Inclination and declination
        I = np.degrees(np.arctan2(Z, H))
        D = np.degrees(np.arctan2(Y, X))
        
        return {
            'total': F,
            'x': X,
            'y': Y,
            'z': Z,
            'horizontal': H,
            'inclination': I,
            'declination': D
        }


class PlatformCompensation:
    """
    Platform magnetic compensation for drone-mounted magnetometers.
    
    Removes magnetic interference from the drone platform using
    attitude-dependent compensation model.
    """
    
    def __init__(self):
        self.compensation_coefficients: Optional[np.ndarray] = None
        self.is_calibrated = False
        
    def calibrate(self, magnetic_data: np.ndarray, 
                 heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray,
                 method: str = 'tolles_lawson') -> Dict[str, Any]:
        """
        Calibrate compensation model from calibration flight data.
        
        Args:
            magnetic_data: Raw magnetic field measurements
            heading: Aircraft heading in degrees
            pitch: Aircraft pitch in degrees
            roll: Aircraft roll in degrees
            method: Compensation method ('tolles_lawson', 'simple')
            
        Returns:
            Calibration results
        """
        if method == 'tolles_lawson':
            return self._tolles_lawson_calibration(magnetic_data, heading, pitch, roll)
        else:
            return self._simple_calibration(magnetic_data, heading, pitch, roll)
            
    def _tolles_lawson_calibration(self, mag: np.ndarray, 
                                   hdg: np.ndarray, pitch: np.ndarray, 
                                   roll: np.ndarray) -> Dict[str, Any]:
        """
        Tolles-Lawson compensation model calibration.
        
        The model accounts for:
        - Permanent magnetization (3 terms)
        - Induced magnetization (9 terms)
        - Eddy currents (6 terms)
        """
        # Convert angles to radians
        h = np.radians(hdg)
        p = np.radians(pitch)
        r = np.radians(roll)
        
        # Direction cosines
        cos_h, sin_h = np.cos(h), np.sin(h)
        cos_p, sin_p = np.cos(p), np.sin(p)
        cos_r, sin_r = np.cos(r), np.sin(r)
        
        # Build design matrix (18 terms)
        # Permanent terms (3)
        T1 = cos_h * cos_p
        T2 = sin_h * cos_p
        T3 = sin_p
        
        # Induced terms (9) - simplified
        T4 = cos_h**2 * cos_p**2
        T5 = sin_h**2 * cos_p**2
        T6 = sin_p**2
        T7 = cos_h * sin_h * cos_p**2
        T8 = cos_h * cos_p * sin_p
        T9 = sin_h * cos_p * sin_p
        T10 = cos_r * cos_h * cos_p
        T11 = cos_r * sin_h * cos_p
        T12 = cos_r * sin_p
        
        # Eddy current terms (6) - simplified using rate of change
        dh = np.gradient(h)
        dp = np.gradient(p)
        dr = np.gradient(r)
        
        T13 = dh * cos_h * cos_p
        T14 = dh * sin_h * cos_p
        T15 = dp * cos_p
        T16 = dr * cos_r
        T17 = dh * sin_p
        T18 = dp * sin_p
        
        # Build design matrix
        A = np.column_stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, 
                            T10, T11, T12, T13, T14, T15, T16, T17, T18])
        
        # Solve least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A, mag, rcond=None)
        
        self.compensation_coefficients = coeffs
        self.is_calibrated = True
        
        # Compute compensated field
        compensated = mag - A @ coeffs
        
        # Compute improvement
        raw_std = np.std(mag)
        comp_std = np.std(compensated)
        improvement = (raw_std - comp_std) / raw_std * 100
        
        return {
            'coefficients': coeffs,
            'raw_std': raw_std,
            'compensated_std': comp_std,
            'improvement_percent': improvement,
            'residuals': residuals if len(residuals) > 0 else None
        }
        
    def _simple_calibration(self, mag: np.ndarray, 
                           hdg: np.ndarray, pitch: np.ndarray, 
                           roll: np.ndarray) -> Dict[str, Any]:
        """Simple heading-only compensation."""
        h = np.radians(hdg)
        
        # Simple sinusoidal model
        A = np.column_stack([np.ones_like(h), np.cos(h), np.sin(h), 
                            np.cos(2*h), np.sin(2*h)])
        
        coeffs, residuals, rank, s = np.linalg.lstsq(A, mag, rcond=None)
        
        self.compensation_coefficients = coeffs
        self.is_calibrated = True
        
        compensated = mag - A @ coeffs
        
        return {
            'coefficients': coeffs,
            'raw_std': np.std(mag),
            'compensated_std': np.std(compensated),
            'improvement_percent': (np.std(mag) - np.std(compensated)) / np.std(mag) * 100
        }
        
    def apply(self, magnetic_data: np.ndarray, 
             heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
        """
        Apply compensation to magnetic data.
        
        Args:
            magnetic_data: Raw magnetic field measurements
            heading: Aircraft heading in degrees
            pitch: Aircraft pitch in degrees
            roll: Aircraft roll in degrees
            
        Returns:
            Compensated magnetic field
        """
        if not self.is_calibrated:
            raise ValueError("Compensation model not calibrated")
            
        h = np.radians(heading)
        p = np.radians(pitch)
        r = np.radians(roll)
        
        cos_h, sin_h = np.cos(h), np.sin(h)
        cos_p, sin_p = np.cos(p), np.sin(p)
        cos_r, sin_r = np.cos(r), np.sin(r)
        
        # Build terms (matching calibration)
        T1 = cos_h * cos_p
        T2 = sin_h * cos_p
        T3 = sin_p
        T4 = cos_h**2 * cos_p**2
        T5 = sin_h**2 * cos_p**2
        T6 = sin_p**2
        T7 = cos_h * sin_h * cos_p**2
        T8 = cos_h * cos_p * sin_p
        T9 = sin_h * cos_p * sin_p
        T10 = cos_r * cos_h * cos_p
        T11 = cos_r * sin_h * cos_p
        T12 = cos_r * sin_p
        
        dh = np.gradient(h)
        dp = np.gradient(p)
        dr = np.gradient(r)
        
        T13 = dh * cos_h * cos_p
        T14 = dh * sin_h * cos_p
        T15 = dp * cos_p
        T16 = dr * cos_r
        T17 = dh * sin_p
        T18 = dp * sin_p
        
        A = np.column_stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, 
                            T10, T11, T12, T13, T14, T15, T16, T17, T18])
        
        # Handle coefficient length mismatch
        if len(self.compensation_coefficients) == 5:
            # Simple model
            A = np.column_stack([np.ones_like(h), np.cos(h), np.sin(h), 
                                np.cos(2*h), np.sin(2*h)])
        
        compensation = A @ self.compensation_coefficients
        return magnetic_data - compensation


class DiurnalCorrection:
    """
    Diurnal variation correction using base station data.
    
    Removes temporal variations in Earth's magnetic field caused by
    solar activity and ionospheric currents.
    """
    
    def __init__(self):
        self.base_station_data: Optional[pd.DataFrame] = None
        self.reference_value: Optional[float] = None
        
    def set_base_station(self, timestamps: np.ndarray, 
                        magnetic_field: np.ndarray,
                        reference_value: Optional[float] = None) -> None:
        """
        Set base station data for diurnal correction.
        
        Args:
            timestamps: Base station timestamps
            magnetic_field: Base station magnetic field readings
            reference_value: Reference value to normalize to (default: mean)
        """
        self.base_station_data = pd.DataFrame({
            'timestamp': timestamps,
            'field': magnetic_field
        })
        self.base_station_data.set_index('timestamp', inplace=True)
        
        self.reference_value = reference_value or np.mean(magnetic_field)
        
    def apply(self, timestamps: np.ndarray, 
             magnetic_field: np.ndarray) -> np.ndarray:
        """
        Apply diurnal correction to survey data.
        
        Args:
            timestamps: Survey timestamps
            magnetic_field: Survey magnetic field readings
            
        Returns:
            Diurnally corrected magnetic field
        """
        if self.base_station_data is None:
            raise ValueError("Base station data not set")
            
        # Interpolate base station to survey times
        base_interp = np.interp(
            timestamps.astype(np.float64),
            self.base_station_data.index.values.astype(np.float64),
            self.base_station_data['field'].values
        )
        
        # Compute correction
        correction = base_interp - self.reference_value
        
        return magnetic_field - correction
        
    def compute_diurnal_variation(self) -> Dict[str, Any]:
        """Compute statistics of diurnal variation."""
        if self.base_station_data is None:
            return {}
            
        field = self.base_station_data['field'].values
        
        return {
            'mean': np.mean(field),
            'std': np.std(field),
            'min': np.min(field),
            'max': np.max(field),
            'range': np.max(field) - np.min(field),
            'reference': self.reference_value
        }


class IGRFRemoval:
    """
    IGRF field removal to compute residual magnetic anomaly.
    """
    
    def __init__(self, epoch: float = 2020.0):
        self.igrf = IGRFModel(epoch=epoch, coefficients={})
        
    def compute_igrf(self, latitude: np.ndarray, longitude: np.ndarray,
                    altitude: np.ndarray, date: datetime) -> np.ndarray:
        """
        Compute IGRF field at survey locations.
        
        Args:
            latitude: Latitudes in degrees
            longitude: Longitudes in degrees
            altitude: Altitudes in meters
            date: Survey date
            
        Returns:
            IGRF total field values
        """
        igrf_field = np.zeros_like(latitude)
        
        for i in range(len(latitude)):
            result = self.igrf.compute_field(
                latitude[i], longitude[i], altitude[i] / 1000, date
            )
            igrf_field[i] = result['total']
            
        return igrf_field
        
    def remove(self, magnetic_field: np.ndarray, 
              latitude: np.ndarray, longitude: np.ndarray,
              altitude: np.ndarray, date: datetime) -> np.ndarray:
        """
        Remove IGRF from magnetic field to get residual anomaly.
        
        Args:
            magnetic_field: Total magnetic field
            latitude: Latitudes in degrees
            longitude: Longitudes in degrees
            altitude: Altitudes in meters
            date: Survey date
            
        Returns:
            Residual magnetic anomaly
        """
        igrf = self.compute_igrf(latitude, longitude, altitude, date)
        return magnetic_field - igrf


class TieLineLeveling:
    """
    Tie-line leveling for removing line-to-line striping artifacts.
    """
    
    def __init__(self, method: LevelingMethod = LevelingMethod.POLYNOMIAL):
        self.method = method
        self.intersection_points: List[Dict] = []
        self.corrections: Dict[str, np.ndarray] = {}
        
    def find_intersections(self, traverse_lines: List[SurveyLine],
                          tie_lines: List[SurveyLine],
                          tolerance: float = 10.0) -> List[Dict]:
        """
        Find intersection points between traverse and tie lines.
        
        Args:
            traverse_lines: List of traverse survey lines
            tie_lines: List of tie survey lines
            tolerance: Distance tolerance in meters
            
        Returns:
            List of intersection point dictionaries
        """
        intersections = []
        
        for trav in traverse_lines:
            trav_coords = np.column_stack([trav.longitude, trav.latitude])
            
            for tie in tie_lines:
                tie_coords = np.column_stack([tie.longitude, tie.latitude])
                
                # Build KD-tree for tie line
                tree = cKDTree(tie_coords)
                
                # Find closest points
                distances, indices = tree.query(trav_coords)
                
                # Find points within tolerance
                close_mask = distances < tolerance / 111000  # Approximate degrees
                
                for i, (dist, idx) in enumerate(zip(distances, indices)):
                    if dist < tolerance / 111000:
                        intersections.append({
                            'traverse_line': trav.line_id,
                            'tie_line': tie.line_id,
                            'traverse_idx': i,
                            'tie_idx': idx,
                            'traverse_value': trav.magnetic_field[i],
                            'tie_value': tie.magnetic_field[idx],
                            'difference': trav.magnetic_field[i] - tie.magnetic_field[idx],
                            'latitude': trav.latitude[i],
                            'longitude': trav.longitude[i]
                        })
                        
        self.intersection_points = intersections
        return intersections
        
    def compute_corrections(self, traverse_lines: List[SurveyLine]) -> Dict[str, np.ndarray]:
        """
        Compute leveling corrections for each traverse line.
        
        Args:
            traverse_lines: List of traverse survey lines
            
        Returns:
            Dictionary mapping line IDs to correction arrays
        """
        corrections = {}
        
        for line in traverse_lines:
            # Get intersections for this line
            line_intersections = [
                p for p in self.intersection_points 
                if p['traverse_line'] == line.line_id
            ]
            
            if len(line_intersections) < 2:
                # Not enough intersections, no correction
                corrections[line.line_id] = np.zeros(len(line.magnetic_field))
                continue
                
            # Extract intersection data
            indices = np.array([p['traverse_idx'] for p in line_intersections])
            differences = np.array([p['difference'] for p in line_intersections])
            
            if self.method == LevelingMethod.POLYNOMIAL:
                # Fit polynomial through differences
                degree = min(3, len(indices) - 1)
                coeffs = np.polyfit(indices, differences, degree)
                correction = np.polyval(coeffs, np.arange(len(line.magnetic_field)))
                
            elif self.method == LevelingMethod.SPLINE:
                # Spline interpolation
                if len(indices) >= 4:
                    spline = interpolate.UnivariateSpline(
                        indices, differences, k=3, s=len(indices)
                    )
                    correction = spline(np.arange(len(line.magnetic_field)))
                else:
                    correction = np.interp(
                        np.arange(len(line.magnetic_field)), indices, differences
                    )
                    
            else:
                # Linear interpolation as fallback
                correction = np.interp(
                    np.arange(len(line.magnetic_field)), indices, differences
                )
                
            corrections[line.line_id] = correction
            
        self.corrections = corrections
        return corrections
        
    def apply(self, line: SurveyLine) -> np.ndarray:
        """
        Apply leveling correction to a survey line.
        
        Args:
            line: Survey line to correct
            
        Returns:
            Leveled magnetic field
        """
        if line.line_id not in self.corrections:
            return line.magnetic_field
            
        return line.magnetic_field - self.corrections[line.line_id]


class MicroLeveling:
    """
    Micro-leveling for removing residual line-to-line noise.
    
    Uses directional filtering to remove along-line artifacts.
    """
    
    def __init__(self, filter_width: int = 5, direction: str = 'along_line'):
        self.filter_width = filter_width
        self.direction = direction
        
    def apply_to_grid(self, grid: np.ndarray, 
                     line_direction: float = 0.0) -> np.ndarray:
        """
        Apply micro-leveling to gridded data.
        
        Args:
            grid: 2D gridded magnetic data
            line_direction: Direction of survey lines in degrees from north
            
        Returns:
            Micro-leveled grid
        """
        # Create directional filter
        angle_rad = np.radians(line_direction)
        
        # Build 1D filter along line direction
        kernel_size = self.filter_width * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = self.filter_width
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.sin(angle_rad))
            y = int(center + offset * np.cos(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
                
        kernel = kernel / kernel.sum()
        
        # Apply directional filter
        along_line = ndimage.convolve(grid, kernel, mode='reflect')
        
        # Subtract to get across-line signal (the leveling artifact)
        across_line = grid - along_line
        
        # High-pass filter the across-line component
        # to remove long-wavelength geological signal
        across_line_hp = across_line - ndimage.gaussian_filter(across_line, sigma=10)
        
        # Subtract the high-frequency across-line component (the artifact)
        leveled = grid - across_line_hp
        
        return leveled


class MagneticDerivatives:
    """
    Compute magnetic field derivatives and transforms.
    """
    
    def __init__(self, cell_size: float = 10.0):
        self.cell_size = cell_size
        
    def first_vertical_derivative(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute first vertical derivative using FFT.
        
        Args:
            grid: 2D magnetic anomaly grid
            
        Returns:
            First vertical derivative grid
        """
        # FFT
        fft_grid = np.fft.fft2(grid)
        
        # Frequency arrays
        ny, nx = grid.shape
        kx = np.fft.fftfreq(nx, self.cell_size)
        ky = np.fft.fftfreq(ny, self.cell_size)
        KX, KY = np.meshgrid(kx, ky)
        
        # Wavenumber magnitude
        K = np.sqrt(KX**2 + KY**2)
        K[0, 0] = 1e-10  # Avoid division by zero
        
        # First vertical derivative filter
        filter_1vd = 2 * np.pi * K
        
        # Apply filter
        fft_1vd = fft_grid * filter_1vd
        
        # Inverse FFT
        return np.real(np.fft.ifft2(fft_1vd))
        
    def analytic_signal(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute analytic signal amplitude.
        
        Args:
            grid: 2D magnetic anomaly grid
            
        Returns:
            Analytic signal amplitude grid
        """
        # Compute derivatives
        dx = np.gradient(grid, self.cell_size, axis=1)
        dy = np.gradient(grid, self.cell_size, axis=0)
        dz = self.first_vertical_derivative(grid)
        
        # Analytic signal amplitude
        return np.sqrt(dx**2 + dy**2 + dz**2)
        
    def tilt_derivative(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute tilt derivative (TDR).
        
        Args:
            grid: 2D magnetic anomaly grid
            
        Returns:
            Tilt derivative grid in degrees
        """
        # Compute derivatives
        dx = np.gradient(grid, self.cell_size, axis=1)
        dy = np.gradient(grid, self.cell_size, axis=0)
        dz = self.first_vertical_derivative(grid)
        
        # Horizontal gradient
        dh = np.sqrt(dx**2 + dy**2)
        
        # Tilt derivative
        tdr = np.degrees(np.arctan2(dz, dh))
        
        return tdr
        
    def reduction_to_pole(self, grid: np.ndarray, 
                         inclination: float, declination: float) -> np.ndarray:
        """
        Reduce magnetic anomaly to the pole.
        
        Args:
            grid: 2D magnetic anomaly grid
            inclination: Magnetic inclination in degrees
            declination: Magnetic declination in degrees
            
        Returns:
            Reduced-to-pole grid
        """
        # Convert to radians
        I = np.radians(inclination)
        D = np.radians(declination)
        
        # FFT
        fft_grid = np.fft.fft2(grid)
        
        # Frequency arrays
        ny, nx = grid.shape
        kx = np.fft.fftfreq(nx, self.cell_size)
        ky = np.fft.fftfreq(ny, self.cell_size)
        KX, KY = np.meshgrid(kx, ky)
        
        # Wavenumber magnitude
        K = np.sqrt(KX**2 + KY**2)
        K[0, 0] = 1e-10
        
        # Direction cosines
        theta = np.arctan2(KY, KX)
        
        # RTP filter
        # Phase factor for magnetization direction
        sin_I = np.sin(I)
        cos_I = np.cos(I)
        
        # Avoid instability at low inclinations
        if abs(inclination) < 20:
            # Use pseudo-inclination
            I_pseudo = np.radians(np.sign(inclination) * 20)
            sin_I = np.sin(I_pseudo)
            cos_I = np.cos(I_pseudo)
            
        # RTP operator
        rtp_filter = 1.0 / (sin_I + 1j * cos_I * np.cos(theta - D))**2
        
        # Apply filter
        fft_rtp = fft_grid * rtp_filter
        
        # Inverse FFT
        return np.real(np.fft.ifft2(fft_rtp))
        
    def upward_continuation(self, grid: np.ndarray, height: float) -> np.ndarray:
        """
        Upward continue magnetic field.
        
        Args:
            grid: 2D magnetic anomaly grid
            height: Continuation height in same units as cell_size
            
        Returns:
            Upward continued grid
        """
        # FFT
        fft_grid = np.fft.fft2(grid)
        
        # Frequency arrays
        ny, nx = grid.shape
        kx = np.fft.fftfreq(nx, self.cell_size)
        ky = np.fft.fftfreq(ny, self.cell_size)
        KX, KY = np.meshgrid(kx, ky)
        
        # Wavenumber magnitude
        K = np.sqrt(KX**2 + KY**2)
        
        # Upward continuation filter
        uc_filter = np.exp(-2 * np.pi * K * height)
        
        # Apply filter
        fft_uc = fft_grid * uc_filter
        
        # Inverse FFT
        return np.real(np.fft.ifft2(fft_uc))


class MagnetometryPipeline:
    """
    Complete magnetometry processing pipeline.
    """
    
    def __init__(self, config: MagnetometerConfig = None):
        self.config = config
        self.platform_compensation = PlatformCompensation()
        self.diurnal_correction = DiurnalCorrection()
        self.igrf_removal = IGRFRemoval()
        self.tie_line_leveling = TieLineLeveling()
        self.micro_leveling = MicroLeveling()
        self.derivatives = MagneticDerivatives()
        
        self.processing_history: List[Dict] = []
        
    def process_survey(self, survey_lines: List[SurveyLine],
                      base_station: Optional[Dict] = None,
                      survey_date: datetime = None,
                      apply_compensation: bool = True,
                      apply_diurnal: bool = True,
                      apply_igrf: bool = True,
                      apply_leveling: bool = True) -> Dict[str, Any]:
        """
        Process complete magnetometry survey.
        
        Args:
            survey_lines: List of survey lines
            base_station: Base station data dictionary
            survey_date: Survey date
            apply_compensation: Whether to apply platform compensation
            apply_diurnal: Whether to apply diurnal correction
            apply_igrf: Whether to remove IGRF
            apply_leveling: Whether to apply tie-line leveling
            
        Returns:
            Processing results
        """
        survey_date = survey_date or datetime.now()
        results = {'lines': {}, 'statistics': {}}
        
        # Separate traverse and tie lines
        traverse_lines = [l for l in survey_lines if l.line_type == 'traverse']
        tie_lines = [l for l in survey_lines if l.line_type == 'tie']
        
        # Process each line
        for line in survey_lines:
            processed_field = line.magnetic_field.copy()
            
            # Platform compensation
            if apply_compensation and line.heading is not None:
                if self.platform_compensation.is_calibrated:
                    processed_field = self.platform_compensation.apply(
                        processed_field, line.heading, 
                        line.pitch or np.zeros_like(line.heading),
                        line.roll or np.zeros_like(line.heading)
                    )
                    
            # Diurnal correction
            if apply_diurnal and base_station is not None:
                if self.diurnal_correction.base_station_data is None:
                    self.diurnal_correction.set_base_station(
                        base_station['timestamps'],
                        base_station['field']
                    )
                processed_field = self.diurnal_correction.apply(
                    line.timestamps, processed_field
                )
                
            # IGRF removal
            if apply_igrf:
                processed_field = self.igrf_removal.remove(
                    processed_field, line.latitude, line.longitude,
                    line.altitude, survey_date
                )
                
            results['lines'][line.line_id] = {
                'raw': line.magnetic_field,
                'processed': processed_field,
                'latitude': line.latitude,
                'longitude': line.longitude
            }
            
        # Tie-line leveling
        if apply_leveling and len(tie_lines) > 0:
            self.tie_line_leveling.find_intersections(traverse_lines, tie_lines)
            self.tie_line_leveling.compute_corrections(traverse_lines)
            
            for line in traverse_lines:
                if line.line_id in results['lines']:
                    results['lines'][line.line_id]['processed'] = \
                        self.tie_line_leveling.apply(
                            SurveyLine(
                                line_id=line.line_id,
                                line_type=line.line_type,
                                timestamps=line.timestamps,
                                latitude=line.latitude,
                                longitude=line.longitude,
                                altitude=line.altitude,
                                magnetic_field=results['lines'][line.line_id]['processed']
                            )
                        )
                        
        # Compute statistics
        all_raw = np.concatenate([r['raw'] for r in results['lines'].values()])
        all_processed = np.concatenate([r['processed'] for r in results['lines'].values()])
        
        results['statistics'] = {
            'raw_mean': np.mean(all_raw),
            'raw_std': np.std(all_raw),
            'processed_mean': np.mean(all_processed),
            'processed_std': np.std(all_processed),
            'num_lines': len(survey_lines),
            'num_traverse': len(traverse_lines),
            'num_tie': len(tie_lines)
        }
        
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'num_lines': len(survey_lines),
            'apply_compensation': apply_compensation,
            'apply_diurnal': apply_diurnal,
            'apply_igrf': apply_igrf,
            'apply_leveling': apply_leveling
        })
        
        return results
        
    def grid_data(self, line_data: Dict[str, Dict], 
                 cell_size: float = 10.0,
                 method: str = 'minimum_curvature') -> xr.DataArray:
        """
        Grid line data to regular grid.
        
        Args:
            line_data: Dictionary of line data from process_survey
            cell_size: Grid cell size in meters
            method: Gridding method
            
        Returns:
            Gridded data as xarray DataArray
        """
        # Collect all points
        all_x = []
        all_y = []
        all_z = []
        
        for line_id, data in line_data.items():
            all_x.extend(data['longitude'])
            all_y.extend(data['latitude'])
            all_z.extend(data['processed'])
            
        x = np.array(all_x)
        y = np.array(all_y)
        z = np.array(all_z)
        
        # Create grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Convert cell size from meters to degrees (approximate)
        cell_deg = cell_size / 111000
        
        xi = np.arange(x_min, x_max + cell_deg, cell_deg)
        yi = np.arange(y_min, y_max + cell_deg, cell_deg)
        XI, YI = np.meshgrid(xi, yi)
        
        # Grid using scipy
        ZI = interpolate.griddata((x, y), z, (XI, YI), method='cubic')
        
        # Fill NaN with nearest neighbor
        mask = np.isnan(ZI)
        if mask.any():
            ZI_nearest = interpolate.griddata((x, y), z, (XI, YI), method='nearest')
            ZI[mask] = ZI_nearest[mask]
            
        return xr.DataArray(
            data=ZI,
            dims=['y', 'x'],
            coords={'y': yi, 'x': xi},
            attrs={'cell_size': cell_size, 'units': 'nT'}
        )
        
    def compute_products(self, grid: xr.DataArray,
                        inclination: float = 60.0,
                        declination: float = 0.0) -> Dict[str, xr.DataArray]:
        """
        Compute derived magnetic products.
        
        Args:
            grid: Gridded magnetic anomaly
            inclination: Magnetic inclination
            declination: Magnetic declination
            
        Returns:
            Dictionary of derived products
        """
        cell_size = grid.attrs.get('cell_size', 10.0)
        self.derivatives.cell_size = cell_size / 111000  # Convert to degrees
        
        data = grid.values
        
        products = {
            'tmi_residual': grid,
            'first_vertical_derivative': xr.DataArray(
                data=self.derivatives.first_vertical_derivative(data),
                dims=grid.dims,
                coords=grid.coords,
                attrs={'units': 'nT/m'}
            ),
            'analytic_signal': xr.DataArray(
                data=self.derivatives.analytic_signal(data),
                dims=grid.dims,
                coords=grid.coords,
                attrs={'units': 'nT/m'}
            ),
            'tilt_derivative': xr.DataArray(
                data=self.derivatives.tilt_derivative(data),
                dims=grid.dims,
                coords=grid.coords,
                attrs={'units': 'degrees'}
            ),
            'reduction_to_pole': xr.DataArray(
                data=self.derivatives.reduction_to_pole(data, inclination, declination),
                dims=grid.dims,
                coords=grid.coords,
                attrs={'units': 'nT'}
            )
        }
        
        return products


def create_magnetometry_pipeline(config: MagnetometerConfig = None) -> MagnetometryPipeline:
    """Factory function to create magnetometry pipeline."""
    return MagnetometryPipeline(config)


def create_survey_design(area_bounds: Tuple[float, float, float, float],
                        line_spacing: float = 100.0,
                        tie_line_spacing: float = 1000.0,
                        line_direction: float = 0.0,
                        altitude: float = 50.0,
                        speed: float = 10.0) -> Dict[str, Any]:
    """
    Create magnetometry survey design for autonomous exploration.
    
    Args:
        area_bounds: (min_lon, min_lat, max_lon, max_lat)
        line_spacing: Traverse line spacing in meters
        tie_line_spacing: Tie line spacing in meters
        line_direction: Line direction in degrees from north
        altitude: Survey altitude in meters
        speed: Survey speed in m/s
        
    Returns:
        Survey design dictionary
    """
    min_lon, min_lat, max_lon, max_lat = area_bounds
    
    # Convert spacings to degrees
    line_spacing_deg = line_spacing / 111000
    tie_spacing_deg = tie_line_spacing / 111000
    
    # Generate traverse lines
    traverse_lines = []
    if line_direction == 0 or line_direction == 180:
        # North-South lines
        x = min_lon
        line_num = 1
        while x <= max_lon:
            traverse_lines.append({
                'line_id': f'T{line_num:04d}',
                'start': (x, min_lat),
                'end': (x, max_lat),
                'direction': line_direction
            })
            x += line_spacing_deg
            line_num += 1
    else:
        # East-West lines
        y = min_lat
        line_num = 1
        while y <= max_lat:
            traverse_lines.append({
                'line_id': f'T{line_num:04d}',
                'start': (min_lon, y),
                'end': (max_lon, y),
                'direction': line_direction
            })
            y += line_spacing_deg
            line_num += 1
            
    # Generate tie lines (perpendicular to traverse)
    tie_lines = []
    tie_direction = (line_direction + 90) % 360
    
    if tie_direction == 0 or tie_direction == 180:
        x = min_lon
        line_num = 1
        while x <= max_lon:
            tie_lines.append({
                'line_id': f'TIE{line_num:04d}',
                'start': (x, min_lat),
                'end': (x, max_lat),
                'direction': tie_direction
            })
            x += tie_spacing_deg
            line_num += 1
    else:
        y = min_lat
        line_num = 1
        while y <= max_lat:
            tie_lines.append({
                'line_id': f'TIE{line_num:04d}',
                'start': (min_lon, y),
                'end': (max_lon, y),
                'direction': tie_direction
            })
            y += tie_spacing_deg
            line_num += 1
            
    # Calculate survey statistics
    total_traverse_km = sum(
        np.sqrt((l['end'][0] - l['start'][0])**2 + (l['end'][1] - l['start'][1])**2) * 111
        for l in traverse_lines
    )
    total_tie_km = sum(
        np.sqrt((l['end'][0] - l['start'][0])**2 + (l['end'][1] - l['start'][1])**2) * 111
        for l in tie_lines
    )
    
    estimated_time_hours = (total_traverse_km + total_tie_km) * 1000 / speed / 3600
    
    return {
        'traverse_lines': traverse_lines,
        'tie_lines': tie_lines,
        'parameters': {
            'line_spacing': line_spacing,
            'tie_line_spacing': tie_line_spacing,
            'line_direction': line_direction,
            'altitude': altitude,
            'speed': speed
        },
        'statistics': {
            'num_traverse_lines': len(traverse_lines),
            'num_tie_lines': len(tie_lines),
            'total_traverse_km': total_traverse_km,
            'total_tie_km': total_tie_km,
            'total_km': total_traverse_km + total_tie_km,
            'estimated_time_hours': estimated_time_hours
        }
    }
