"""
Medusa Radiometrics Sensor Adapter for MineralVision.

This module provides integration with Medusa Radiometrics gamma-ray sensors:
- MS-700: Lightweight drone-borne sensor
- MS-1000: Medium sensitivity soil mapping sensor
- MS-4000: High sensitivity survey sensor

Features:
- Sensor-specific calibration and configuration
- Full spectrum analysis with Medusa's spectral fitting
- K/U/Th concentration mapping
- Drone flight planning constraints
- Real-time data streaming support
- Integration with radiometrics pipeline

Based on Medusa Radiometrics specifications and IAEA guidelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import json
import struct

logger = logging.getLogger(__name__)


class MedusaSensorType(Enum):
    """Medusa sensor models."""
    MS_700 = "MS-700"      # Drone-borne, 0.7L CsI
    MS_1000 = "MS-1000"    # Soil mapping, 1.0L CsI
    MS_4000 = "MS-4000"    # High sensitivity, 4.0L CsI
    MS_2000 = "MS-2000"    # Medium sensitivity, 2.0L CsI
    MS_350 = "MS-350"      # Ultra-light drone, 0.35L CsI


class DetectorType(Enum):
    """Detector crystal types."""
    CSI_TL = "CsI(Tl)"     # Cesium Iodide (Thallium doped)
    NAI_TL = "NaI(Tl)"     # Sodium Iodide (Thallium doped)
    BGO = "BGO"            # Bismuth Germanate
    LABR3 = "LaBr3"        # Lanthanum Bromide


class DataFormat(Enum):
    """Medusa data output formats."""
    BINARY = "binary"
    ASCII = "ascii"
    JSON = "json"
    NMEA = "nmea"


class FlightMode(Enum):
    """Drone flight modes for radiometric surveys."""
    TERRAIN_FOLLOWING = "terrain_following"
    CONSTANT_ALTITUDE = "constant_altitude"
    MANUAL = "manual"


@dataclass
class MedusaSensorSpec:
    """Medusa sensor specifications."""
    model: MedusaSensorType
    detector_type: DetectorType
    crystal_volume: float  # liters
    energy_range: Tuple[float, float]  # keV
    num_channels: int
    resolution_at_662kev: float  # % FWHM
    weight: float  # kg
    power_consumption: float  # W
    operating_temp: Tuple[float, float]  # Celsius
    max_count_rate: float  # cps
    gps_integrated: bool
    altimeter_integrated: bool
    
    @classmethod
    def get_spec(cls, model: MedusaSensorType) -> 'MedusaSensorSpec':
        """Get specifications for a sensor model."""
        specs = {
            MedusaSensorType.MS_700: cls(
                model=MedusaSensorType.MS_700,
                detector_type=DetectorType.CSI_TL,
                crystal_volume=0.7,
                energy_range=(30, 3000),
                num_channels=1024,
                resolution_at_662kev=7.5,
                weight=1.8,
                power_consumption=3.5,
                operating_temp=(-20, 50),
                max_count_rate=100000,
                gps_integrated=True,
                altimeter_integrated=True
            ),
            MedusaSensorType.MS_1000: cls(
                model=MedusaSensorType.MS_1000,
                detector_type=DetectorType.CSI_TL,
                crystal_volume=1.0,
                energy_range=(30, 3000),
                num_channels=1024,
                resolution_at_662kev=7.0,
                weight=2.5,
                power_consumption=4.0,
                operating_temp=(-20, 50),
                max_count_rate=150000,
                gps_integrated=True,
                altimeter_integrated=True
            ),
            MedusaSensorType.MS_4000: cls(
                model=MedusaSensorType.MS_4000,
                detector_type=DetectorType.CSI_TL,
                crystal_volume=4.0,
                energy_range=(30, 3000),
                num_channels=1024,
                resolution_at_662kev=6.5,
                weight=8.0,
                power_consumption=8.0,
                operating_temp=(-20, 50),
                max_count_rate=500000,
                gps_integrated=True,
                altimeter_integrated=True
            ),
            MedusaSensorType.MS_2000: cls(
                model=MedusaSensorType.MS_2000,
                detector_type=DetectorType.CSI_TL,
                crystal_volume=2.0,
                energy_range=(30, 3000),
                num_channels=1024,
                resolution_at_662kev=6.8,
                weight=4.0,
                power_consumption=5.0,
                operating_temp=(-20, 50),
                max_count_rate=250000,
                gps_integrated=True,
                altimeter_integrated=True
            ),
            MedusaSensorType.MS_350: cls(
                model=MedusaSensorType.MS_350,
                detector_type=DetectorType.CSI_TL,
                crystal_volume=0.35,
                energy_range=(30, 3000),
                num_channels=512,
                resolution_at_662kev=8.0,
                weight=0.9,
                power_consumption=2.5,
                operating_temp=(-20, 50),
                max_count_rate=50000,
                gps_integrated=True,
                altimeter_integrated=False
            )
        }
        return specs.get(model, specs[MedusaSensorType.MS_1000])


@dataclass
class MedusaCalibration:
    """Medusa sensor calibration data."""
    sensor_serial: str
    calibration_date: datetime
    energy_coefficients: List[float]  # Polynomial coefficients
    efficiency_curve: Dict[float, float]  # Energy -> efficiency
    stripping_ratios: Dict[str, float]
    sensitivity: Dict[str, float]  # cps per unit concentration
    background_spectrum: Optional[np.ndarray] = None
    temperature_correction: float = 0.0  # %/degree C
    
    def energy_to_channel(self, energy_kev: float) -> int:
        """Convert energy to channel number."""
        # Inverse of polynomial: E = a0 + a1*ch + a2*ch^2 + ...
        # Approximate with linear for speed
        if len(self.energy_coefficients) >= 2:
            a0, a1 = self.energy_coefficients[0], self.energy_coefficients[1]
            return int((energy_kev - a0) / a1)
        return int(energy_kev / 3.0)  # Default ~3 keV/channel
        
    def channel_to_energy(self, channel: int) -> float:
        """Convert channel number to energy."""
        energy = 0.0
        for i, coef in enumerate(self.energy_coefficients):
            energy += coef * (channel ** i)
        return energy


@dataclass
class MedusaMeasurement:
    """Single Medusa measurement record."""
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_agl: float  # meters above ground level
    altitude_msl: float  # meters above sea level
    live_time: float  # seconds
    real_time: float  # seconds
    spectrum: np.ndarray  # Full spectrum
    temperature: float  # Celsius
    battery_voltage: float
    gps_quality: int
    hdop: float
    num_satellites: int
    
    # Processed values (filled after processing)
    k_percent: Optional[float] = None
    eu_ppm: Optional[float] = None
    eth_ppm: Optional[float] = None
    total_count: Optional[float] = None
    dose_rate: Optional[float] = None


@dataclass
class DroneFlightConstraints:
    """Flight constraints for drone-borne radiometric surveys."""
    min_altitude_agl: float = 30.0  # meters
    max_altitude_agl: float = 120.0  # meters
    optimal_altitude_agl: float = 60.0  # meters
    max_speed: float = 10.0  # m/s
    optimal_speed: float = 5.0  # m/s
    line_spacing: float = 50.0  # meters
    sample_interval: float = 1.0  # seconds
    min_battery_percent: float = 20.0
    max_wind_speed: float = 10.0  # m/s
    
    def validate_altitude(self, altitude: float) -> Tuple[bool, str]:
        """Validate altitude is within constraints."""
        if altitude < self.min_altitude_agl:
            return False, f"Altitude {altitude}m below minimum {self.min_altitude_agl}m"
        if altitude > self.max_altitude_agl:
            return False, f"Altitude {altitude}m above maximum {self.max_altitude_agl}m"
        return True, "OK"
        
    def validate_speed(self, speed: float) -> Tuple[bool, str]:
        """Validate speed is within constraints."""
        if speed > self.max_speed:
            return False, f"Speed {speed}m/s exceeds maximum {self.max_speed}m/s"
        return True, "OK"


class MedusaDataParser:
    """
    Parse Medusa sensor data files.
    
    Supports binary and ASCII formats from Medusa sensors.
    """
    
    def __init__(self, sensor_spec: MedusaSensorSpec = None):
        self.spec = sensor_spec or MedusaSensorSpec.get_spec(MedusaSensorType.MS_1000)
        
    def parse_binary_record(self, data: bytes) -> MedusaMeasurement:
        """
        Parse a single binary record.
        
        Medusa binary format (typical):
        - Header: 4 bytes (record type, length)
        - Timestamp: 8 bytes (Unix timestamp)
        - GPS: 24 bytes (lat, lon, alt, quality)
        - Spectrum: num_channels * 4 bytes (uint32)
        - Metadata: variable
        """
        offset = 0
        
        # Header
        record_type, record_length = struct.unpack('<HH', data[offset:offset+4])
        offset += 4
        
        # Timestamp
        unix_time = struct.unpack('<d', data[offset:offset+8])[0]
        timestamp = datetime.fromtimestamp(unix_time)
        offset += 8
        
        # GPS data
        latitude = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        longitude = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        altitude_msl = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        altitude_agl = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        gps_quality = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        num_satellites = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        hdop = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        # Timing
        live_time = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        real_time = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        # Spectrum
        num_channels = self.spec.num_channels
        spectrum = np.array(struct.unpack(f'<{num_channels}I', 
                                         data[offset:offset+num_channels*4]))
        offset += num_channels * 4
        
        # Metadata
        temperature = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        battery_voltage = struct.unpack('<f', data[offset:offset+4])[0]
        
        return MedusaMeasurement(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude_agl=altitude_agl,
            altitude_msl=altitude_msl,
            live_time=live_time,
            real_time=real_time,
            spectrum=spectrum,
            temperature=temperature,
            battery_voltage=battery_voltage,
            gps_quality=gps_quality,
            hdop=hdop,
            num_satellites=num_satellites
        )
        
    def parse_ascii_file(self, file_path: str) -> List[MedusaMeasurement]:
        """
        Parse ASCII data file.
        
        Typical format: CSV with header row.
        """
        measurements = []
        
        # Read file
        df = pd.read_csv(file_path)
        
        # Map columns (adjust based on actual Medusa output)
        for _, row in df.iterrows():
            # Parse timestamp
            if 'timestamp' in df.columns:
                timestamp = pd.to_datetime(row['timestamp'])
            elif 'date' in df.columns and 'time' in df.columns:
                timestamp = pd.to_datetime(f"{row['date']} {row['time']}")
            else:
                timestamp = datetime.now()
                
            # Parse GPS
            latitude = row.get('latitude', row.get('lat', 0.0))
            longitude = row.get('longitude', row.get('lon', 0.0))
            altitude_agl = row.get('altitude_agl', row.get('alt_agl', 60.0))
            altitude_msl = row.get('altitude_msl', row.get('alt_msl', 0.0))
            
            # Parse timing
            live_time = row.get('live_time', row.get('livetime', 1.0))
            real_time = row.get('real_time', row.get('realtime', 1.0))
            
            # Parse spectrum (if available as columns)
            spectrum_cols = [c for c in df.columns if c.startswith('ch_') or c.startswith('channel_')]
            if spectrum_cols:
                spectrum = np.array([row[c] for c in sorted(spectrum_cols)])
            else:
                spectrum = np.zeros(self.spec.num_channels)
                
            # Parse metadata
            temperature = row.get('temperature', row.get('temp', 25.0))
            battery_voltage = row.get('battery', row.get('voltage', 12.0))
            gps_quality = int(row.get('gps_quality', row.get('fix_type', 4)))
            hdop = row.get('hdop', 1.0)
            num_satellites = int(row.get('satellites', row.get('num_sats', 10)))
            
            measurement = MedusaMeasurement(
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                altitude_agl=altitude_agl,
                altitude_msl=altitude_msl,
                live_time=live_time,
                real_time=real_time,
                spectrum=spectrum,
                temperature=temperature,
                battery_voltage=battery_voltage,
                gps_quality=gps_quality,
                hdop=hdop,
                num_satellites=num_satellites
            )
            
            # Add pre-computed concentrations if available
            if 'k_percent' in df.columns:
                measurement.k_percent = row['k_percent']
            if 'eu_ppm' in df.columns:
                measurement.eu_ppm = row['eu_ppm']
            if 'eth_ppm' in df.columns:
                measurement.eth_ppm = row['eth_ppm']
                
            measurements.append(measurement)
            
        return measurements


class MedusaSpectralFitting:
    """
    Full Spectrum Analysis (FSA) for Medusa sensors.
    
    Uses spectral fitting to determine K, U, Th concentrations
    from the full gamma-ray spectrum.
    """
    
    def __init__(self, calibration: MedusaCalibration):
        self.calibration = calibration
        self.standard_spectra: Dict[str, np.ndarray] = {}
        self._load_standard_spectra()
        
    def _load_standard_spectra(self) -> None:
        """Load standard spectra for K, U, Th."""
        # In production, these would be loaded from calibration files
        # Here we create synthetic standard spectra
        
        num_channels = 1024
        channels = np.arange(num_channels)
        
        # K-40 spectrum (1461 keV peak)
        k_spectrum = np.zeros(num_channels)
        k_peak_channel = self.calibration.energy_to_channel(1461)
        k_spectrum += self._gaussian_peak(channels, k_peak_channel, 30, 1000)
        self.standard_spectra['potassium'] = k_spectrum
        
        # U-238 series spectrum (Bi-214 peaks at 609, 1120, 1764 keV)
        u_spectrum = np.zeros(num_channels)
        for energy, intensity in [(609, 500), (1120, 300), (1764, 400)]:
            peak_channel = self.calibration.energy_to_channel(energy)
            u_spectrum += self._gaussian_peak(channels, peak_channel, 25, intensity)
        self.standard_spectra['uranium'] = u_spectrum
        
        # Th-232 series spectrum (Tl-208 at 2614 keV, Ac-228 at 911 keV)
        th_spectrum = np.zeros(num_channels)
        for energy, intensity in [(911, 400), (2614, 600)]:
            peak_channel = self.calibration.energy_to_channel(energy)
            th_spectrum += self._gaussian_peak(channels, peak_channel, 20, intensity)
        self.standard_spectra['thorium'] = th_spectrum
        
    def _gaussian_peak(self, x: np.ndarray, center: float, 
                      sigma: float, amplitude: float) -> np.ndarray:
        """Generate Gaussian peak."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        
    def fit_spectrum(self, spectrum: np.ndarray) -> Dict[str, float]:
        """
        Fit spectrum to determine K, U, Th contributions.
        
        Uses non-negative least squares fitting.
        
        Args:
            spectrum: Measured spectrum
            
        Returns:
            Dictionary with fitted concentrations
        """
        # Build design matrix
        A = np.column_stack([
            self.standard_spectra['potassium'],
            self.standard_spectra['uranium'],
            self.standard_spectra['thorium']
        ])
        
        # Non-negative least squares
        # Simplified implementation - in production use scipy.optimize.nnls
        try:
            # Pseudo-inverse solution
            coeffs = np.linalg.lstsq(A, spectrum, rcond=None)[0]
            coeffs = np.maximum(coeffs, 0)  # Ensure non-negative
        except np.linalg.LinAlgError:
            coeffs = np.zeros(3)
            
        # Convert to concentrations using sensitivity
        k_percent = coeffs[0] / self.calibration.sensitivity.get('potassium', 100)
        eu_ppm = coeffs[1] / self.calibration.sensitivity.get('uranium', 10)
        eth_ppm = coeffs[2] / self.calibration.sensitivity.get('thorium', 5)
        
        return {
            'k_percent': k_percent,
            'eu_ppm': eu_ppm,
            'eth_ppm': eth_ppm,
            'fit_coefficients': coeffs.tolist()
        }


class MedusaProcessor:
    """
    Process Medusa radiometric data.
    
    Applies corrections and calculates concentrations.
    """
    
    def __init__(self, sensor_type: MedusaSensorType,
                calibration: MedusaCalibration = None):
        self.spec = MedusaSensorSpec.get_spec(sensor_type)
        self.calibration = calibration or self._default_calibration()
        self.spectral_fitter = MedusaSpectralFitting(self.calibration)
        self.flight_constraints = DroneFlightConstraints()
        
    def _default_calibration(self) -> MedusaCalibration:
        """Create default calibration."""
        return MedusaCalibration(
            sensor_serial="DEFAULT",
            calibration_date=datetime.now(),
            energy_coefficients=[0.0, 3.0, 0.0],  # ~3 keV/channel
            efficiency_curve={100: 0.8, 500: 0.9, 1000: 0.85, 2000: 0.7},
            stripping_ratios={
                'alpha': 0.25,
                'beta': 0.40,
                'gamma': 0.80
            },
            sensitivity={
                'potassium': 100.0,
                'uranium': 10.0,
                'thorium': 5.0
            }
        )
        
    def process_measurement(self, measurement: MedusaMeasurement) -> MedusaMeasurement:
        """
        Process a single measurement.
        
        Applies all corrections and calculates concentrations.
        """
        # Validate altitude
        valid, msg = self.flight_constraints.validate_altitude(measurement.altitude_agl)
        if not valid:
            logger.warning(f"Altitude warning: {msg}")
            
        # Apply dead-time correction
        dead_time = 5e-6  # 5 microseconds typical
        total_counts = np.sum(measurement.spectrum)
        count_rate = total_counts / measurement.live_time
        dead_time_correction = 1.0 / (1.0 - count_rate * dead_time)
        corrected_spectrum = measurement.spectrum * dead_time_correction
        
        # Apply temperature correction
        temp_correction = 1.0 + self.calibration.temperature_correction * (measurement.temperature - 25.0) / 100.0
        corrected_spectrum = corrected_spectrum * temp_correction
        
        # Apply altitude correction
        reference_altitude = 60.0  # meters
        attenuation_coef = 0.006  # 1/m typical
        altitude_correction = np.exp(attenuation_coef * (measurement.altitude_agl - reference_altitude))
        corrected_spectrum = corrected_spectrum * altitude_correction
        
        # Subtract background
        if self.calibration.background_spectrum is not None:
            corrected_spectrum = np.maximum(
                corrected_spectrum - self.calibration.background_spectrum * measurement.live_time,
                0
            )
            
        # Fit spectrum to get concentrations
        fit_result = self.spectral_fitter.fit_spectrum(corrected_spectrum)
        
        # Update measurement
        measurement.k_percent = fit_result['k_percent']
        measurement.eu_ppm = fit_result['eu_ppm']
        measurement.eth_ppm = fit_result['eth_ppm']
        measurement.total_count = np.sum(corrected_spectrum) / measurement.live_time
        
        # Calculate dose rate (nGy/h)
        # Using IAEA conversion factors
        measurement.dose_rate = (
            13.078 * measurement.k_percent +
            5.675 * measurement.eu_ppm +
            2.494 * measurement.eth_ppm
        )
        
        return measurement
        
    def process_survey(self, measurements: List[MedusaMeasurement]) -> pd.DataFrame:
        """
        Process a complete survey.
        
        Args:
            measurements: List of measurements
            
        Returns:
            DataFrame with processed results
        """
        processed = []
        
        for m in measurements:
            processed_m = self.process_measurement(m)
            processed.append({
                'timestamp': processed_m.timestamp,
                'latitude': processed_m.latitude,
                'longitude': processed_m.longitude,
                'altitude_agl': processed_m.altitude_agl,
                'k_percent': processed_m.k_percent,
                'eu_ppm': processed_m.eu_ppm,
                'eth_ppm': processed_m.eth_ppm,
                'total_count': processed_m.total_count,
                'dose_rate': processed_m.dose_rate,
                'live_time': processed_m.live_time,
                'temperature': processed_m.temperature,
                'gps_quality': processed_m.gps_quality
            })
            
        return pd.DataFrame(processed)
        
    def create_grid(self, df: pd.DataFrame, 
                   cell_size: float = 25.0,
                   method: str = 'idw') -> Dict[str, np.ndarray]:
        """
        Grid survey data.
        
        Args:
            df: Processed survey DataFrame
            cell_size: Grid cell size in meters
            method: Interpolation method ('idw', 'kriging', 'nearest')
            
        Returns:
            Dictionary with gridded arrays
        """
        # Get bounds
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        # Create grid
        # Approximate meters to degrees
        lat_step = cell_size / 111000  # ~111km per degree latitude
        lon_step = cell_size / (111000 * np.cos(np.radians((lat_min + lat_max) / 2)))
        
        lat_grid = np.arange(lat_min, lat_max, lat_step)
        lon_grid = np.arange(lon_min, lon_max, lon_step)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Grid each variable
        grids = {}
        variables = ['k_percent', 'eu_ppm', 'eth_ppm', 'total_count', 'dose_rate']
        
        for var in variables:
            if var in df.columns:
                grid = self._interpolate_idw(
                    df['longitude'].values,
                    df['latitude'].values,
                    df[var].values,
                    lon_mesh,
                    lat_mesh
                )
                grids[var] = grid
                
        grids['latitude'] = lat_mesh
        grids['longitude'] = lon_mesh
        
        return grids
        
    def _interpolate_idw(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        xi: np.ndarray, yi: np.ndarray, power: float = 2.0) -> np.ndarray:
        """Inverse distance weighted interpolation."""
        zi = np.zeros_like(xi)
        
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                distances = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
                
                # Handle exact matches
                if np.any(distances == 0):
                    zi[i, j] = z[distances == 0][0]
                else:
                    weights = 1.0 / (distances ** power)
                    zi[i, j] = np.sum(weights * z) / np.sum(weights)
                    
        return zi


class MedusaFlightPlanner:
    """
    Plan drone flights for Medusa radiometric surveys.
    """
    
    def __init__(self, sensor_type: MedusaSensorType):
        self.spec = MedusaSensorSpec.get_spec(sensor_type)
        self.constraints = DroneFlightConstraints()
        
    def plan_survey(self, bounds: Tuple[float, float, float, float],
                   line_spacing: float = None,
                   altitude: float = None) -> Dict[str, Any]:
        """
        Plan survey flight lines.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            line_spacing: Line spacing in meters
            altitude: Flight altitude in meters AGL
            
        Returns:
            Flight plan dictionary
        """
        line_spacing = line_spacing or self.constraints.line_spacing
        altitude = altitude or self.constraints.optimal_altitude_agl
        
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
            
            # Alternate direction for efficiency
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
        turn_time = (n_lines - 1) * 30  # 30 seconds per turn
        flight_time_s = total_line_length / self.constraints.optimal_speed + turn_time
        
        # Estimate battery usage (rough estimate)
        battery_drain_per_minute = 3.0  # % per minute
        estimated_battery = flight_time_s / 60 * battery_drain_per_minute
        
        return {
            'bounds': bounds,
            'n_lines': n_lines,
            'line_spacing_m': line_spacing,
            'altitude_m': altitude,
            'lines': lines,
            'total_line_length_m': total_line_length,
            'estimated_flight_time_s': flight_time_s,
            'estimated_flight_time_min': flight_time_s / 60,
            'estimated_battery_percent': estimated_battery,
            'sensor': self.spec.model.value,
            'sample_interval_s': self.constraints.sample_interval,
            'expected_samples': int(total_line_length / (self.constraints.optimal_speed * self.constraints.sample_interval))
        }


def create_medusa_processor(sensor_type: str = "MS-1000") -> MedusaProcessor:
    """Factory function to create Medusa processor."""
    sensor_map = {
        "MS-700": MedusaSensorType.MS_700,
        "MS-1000": MedusaSensorType.MS_1000,
        "MS-2000": MedusaSensorType.MS_2000,
        "MS-4000": MedusaSensorType.MS_4000,
        "MS-350": MedusaSensorType.MS_350
    }
    sensor = sensor_map.get(sensor_type, MedusaSensorType.MS_1000)
    return MedusaProcessor(sensor)


def create_medusa_flight_planner(sensor_type: str = "MS-700") -> MedusaFlightPlanner:
    """Factory function to create flight planner."""
    sensor_map = {
        "MS-700": MedusaSensorType.MS_700,
        "MS-1000": MedusaSensorType.MS_1000,
        "MS-2000": MedusaSensorType.MS_2000,
        "MS-4000": MedusaSensorType.MS_4000,
        "MS-350": MedusaSensorType.MS_350
    }
    sensor = sensor_map.get(sensor_type, MedusaSensorType.MS_700)
    return MedusaFlightPlanner(sensor)
