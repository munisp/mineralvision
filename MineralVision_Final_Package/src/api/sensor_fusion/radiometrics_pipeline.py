"""
Gamma-Ray Spectrometry Processing Pipeline for MineralVision.

This module provides comprehensive radiometric data processing including:
- K/U/Th window calibration and stripping corrections
- Dead-time and altitude correction
- Background subtraction
- Standard products: K%, eU ppm, eTh ppm, total count, dose rate
- Derived ratios: K/Th, U/Th, U/K for geological interpretation
- Mission constraints for stable altitude/speed requirements

Based on Medusa Radiometrics sensor specifications and IAEA guidelines.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate, signal, ndimage
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class RadiometricWindow(Enum):
    """Standard radiometric energy windows."""
    TOTAL_COUNT = "total_count"
    POTASSIUM = "potassium"  # K-40: 1.37-1.57 MeV
    URANIUM = "uranium"  # Bi-214: 1.66-1.86 MeV
    THORIUM = "thorium"  # Tl-208: 2.41-2.81 MeV
    CESIUM = "cesium"  # Cs-137: 0.60-0.72 MeV
    COSMIC = "cosmic"  # >3.0 MeV


class RadiometricProduct(Enum):
    """Radiometric data products."""
    POTASSIUM_PERCENT = "k_percent"
    EQUIVALENT_URANIUM = "eu_ppm"
    EQUIVALENT_THORIUM = "eth_ppm"
    TOTAL_COUNT = "total_count"
    DOSE_RATE = "dose_rate"
    URANIUM_THORIUM_RATIO = "u_th_ratio"
    URANIUM_POTASSIUM_RATIO = "u_k_ratio"
    POTASSIUM_THORIUM_RATIO = "k_th_ratio"
    TERNARY_IMAGE = "ternary_image"


@dataclass
class SpectrometerConfig:
    """Gamma-ray spectrometer configuration."""
    sensor_id: str
    sensor_type: str  # NaI, CsI, BGO, HPGe
    crystal_volume: float  # liters
    sample_rate: float  # Hz
    energy_range: Tuple[float, float]  # keV
    num_channels: int
    calibration_date: Optional[datetime] = None
    sensitivity: Dict[str, float] = field(default_factory=dict)  # cps per unit concentration
    stripping_ratios: Dict[str, float] = field(default_factory=dict)
    background_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class RadiometricMeasurement:
    """Single radiometric measurement."""
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float  # meters above ground
    live_time: float  # seconds
    spectrum: Optional[np.ndarray] = None  # full spectrum
    window_counts: Dict[str, float] = field(default_factory=dict)
    quality_flags: int = 0


@dataclass
class CalibrationPad:
    """Calibration pad data for sensitivity determination."""
    pad_id: str
    k_concentration: float  # percent
    u_concentration: float  # ppm
    th_concentration: float  # ppm
    measured_counts: Dict[str, float] = field(default_factory=dict)


class EnergyCalibration:
    """
    Energy calibration for gamma-ray spectrometer.
    
    Converts channel numbers to energy values.
    """
    
    def __init__(self, coefficients: List[float] = None):
        # Default linear calibration: E = a + b*channel
        self.coefficients = coefficients or [0.0, 3.0]  # keV
        
    def channel_to_energy(self, channel: np.ndarray) -> np.ndarray:
        """Convert channel number to energy in keV."""
        energy = np.zeros_like(channel, dtype=float)
        for i, coef in enumerate(self.coefficients):
            energy += coef * (channel ** i)
        return energy
        
    def energy_to_channel(self, energy: float) -> int:
        """Convert energy to channel number (approximate inverse)."""
        if len(self.coefficients) == 2:
            return int((energy - self.coefficients[0]) / self.coefficients[1])
        else:
            # Numerical inverse for higher-order polynomials
            channels = np.arange(4096)
            energies = self.channel_to_energy(channels)
            idx = np.argmin(np.abs(energies - energy))
            return int(channels[idx])
            
    def calibrate(self, known_peaks: List[Tuple[int, float]]) -> None:
        """
        Calibrate using known peak positions.
        
        Args:
            known_peaks: List of (channel, energy_keV) tuples
        """
        channels = np.array([p[0] for p in known_peaks])
        energies = np.array([p[1] for p in known_peaks])
        
        # Fit polynomial
        degree = min(2, len(known_peaks) - 1)
        self.coefficients = list(np.polyfit(channels, energies, degree)[::-1])


class WindowExtraction:
    """
    Extract counts from energy windows in gamma-ray spectra.
    """
    
    # Standard IAEA window definitions (keV)
    STANDARD_WINDOWS = {
        RadiometricWindow.TOTAL_COUNT: (410, 2810),
        RadiometricWindow.POTASSIUM: (1370, 1570),
        RadiometricWindow.URANIUM: (1660, 1860),
        RadiometricWindow.THORIUM: (2410, 2810),
        RadiometricWindow.CESIUM: (600, 720),
        RadiometricWindow.COSMIC: (3000, 10000)
    }
    
    def __init__(self, energy_calibration: EnergyCalibration = None,
                 custom_windows: Dict[str, Tuple[float, float]] = None):
        self.energy_cal = energy_calibration or EnergyCalibration()
        self.windows = dict(self.STANDARD_WINDOWS)
        if custom_windows:
            for name, bounds in custom_windows.items():
                self.windows[name] = bounds
                
    def extract_window(self, spectrum: np.ndarray, 
                      window: RadiometricWindow) -> float:
        """
        Extract counts from a single window.
        
        Args:
            spectrum: Full gamma-ray spectrum
            window: Window to extract
            
        Returns:
            Total counts in window
        """
        if window not in self.windows:
            raise ValueError(f"Unknown window: {window}")
            
        e_min, e_max = self.windows[window]
        ch_min = self.energy_cal.energy_to_channel(e_min)
        ch_max = self.energy_cal.energy_to_channel(e_max)
        
        ch_min = max(0, ch_min)
        ch_max = min(len(spectrum) - 1, ch_max)
        
        return np.sum(spectrum[ch_min:ch_max+1])
        
    def extract_all_windows(self, spectrum: np.ndarray) -> Dict[str, float]:
        """
        Extract counts from all standard windows.
        
        Args:
            spectrum: Full gamma-ray spectrum
            
        Returns:
            Dictionary of window counts
        """
        return {
            window.value: self.extract_window(spectrum, window)
            for window in RadiometricWindow
            if window in self.windows
        }


class StrippingCorrection:
    """
    Compton stripping correction for radiometric data.
    
    Removes spectral interference between K, U, and Th windows.
    """
    
    # Default stripping ratios (IAEA recommended starting values)
    DEFAULT_RATIOS = {
        'alpha': 0.25,   # Th into U window
        'beta': 0.40,    # Th into K window
        'gamma': 0.80,   # U into K window
        'a': 0.06,       # Reverse: U into Th (usually small)
        'b': 0.001,      # Reverse: K into Th (usually negligible)
        'g': 0.003       # Reverse: K into U (usually negligible)
    }
    
    def __init__(self, stripping_ratios: Dict[str, float] = None):
        self.ratios = stripping_ratios or dict(self.DEFAULT_RATIOS)
        
    def apply(self, k_counts: float, u_counts: float, 
             th_counts: float) -> Tuple[float, float, float]:
        """
        Apply stripping correction to window counts.
        
        Args:
            k_counts: Raw potassium window counts
            u_counts: Raw uranium window counts
            th_counts: Raw thorium window counts
            
        Returns:
            Tuple of (stripped_k, stripped_u, stripped_th)
        """
        alpha = self.ratios.get('alpha', 0.25)
        beta = self.ratios.get('beta', 0.40)
        gamma = self.ratios.get('gamma', 0.80)
        a = self.ratios.get('a', 0.06)
        b = self.ratios.get('b', 0.001)
        g = self.ratios.get('g', 0.003)
        
        # Build stripping matrix
        # [K_stripped]   [1    -gamma  -beta ] [K_raw]
        # [U_stripped] = [-g    1      -alpha] [U_raw]
        # [Th_stripped]  [-b   -a       1    ] [Th_raw]
        
        A = np.array([
            [1, -gamma, -beta],
            [-g, 1, -alpha],
            [-b, -a, 1]
        ])
        
        raw = np.array([k_counts, u_counts, th_counts])
        
        # Solve for stripped counts
        try:
            stripped = np.linalg.solve(A, raw)
        except np.linalg.LinAlgError:
            # Matrix singular, return raw counts
            stripped = raw
            
        # Ensure non-negative
        stripped = np.maximum(stripped, 0)
        
        return tuple(stripped)
        
    def calibrate_from_pads(self, pad_data: List[CalibrationPad]) -> Dict[str, float]:
        """
        Calibrate stripping ratios from calibration pad measurements.
        
        Args:
            pad_data: List of calibration pad measurements
            
        Returns:
            Calibrated stripping ratios
        """
        # Need at least 3 pads with different compositions
        if len(pad_data) < 3:
            logger.warning("Need at least 3 calibration pads for stripping calibration")
            return self.ratios
            
        # Build system of equations from pad measurements
        # This is a simplified approach - full calibration is more complex
        
        # Find pads with dominant K, U, Th
        k_pad = max(pad_data, key=lambda p: p.k_concentration)
        u_pad = max(pad_data, key=lambda p: p.u_concentration)
        th_pad = max(pad_data, key=lambda p: p.th_concentration)
        
        # Estimate alpha from Th pad (Th contribution to U window)
        if th_pad.th_concentration > 0:
            th_u_counts = th_pad.measured_counts.get('uranium', 0)
            th_th_counts = th_pad.measured_counts.get('thorium', 1)
            self.ratios['alpha'] = th_u_counts / th_th_counts
            
        # Estimate beta from Th pad (Th contribution to K window)
        if th_pad.th_concentration > 0:
            th_k_counts = th_pad.measured_counts.get('potassium', 0)
            th_th_counts = th_pad.measured_counts.get('thorium', 1)
            self.ratios['beta'] = th_k_counts / th_th_counts
            
        # Estimate gamma from U pad (U contribution to K window)
        if u_pad.u_concentration > 0:
            u_k_counts = u_pad.measured_counts.get('potassium', 0)
            u_u_counts = u_pad.measured_counts.get('uranium', 1)
            self.ratios['gamma'] = u_k_counts / u_u_counts
            
        return self.ratios


class AltitudeCorrection:
    """
    Altitude correction for airborne radiometric data.
    
    Accounts for atmospheric attenuation of gamma rays.
    """
    
    # Attenuation coefficients (1/m) for different windows
    # Based on IAEA TECDOC-1363
    DEFAULT_ATTENUATION = {
        RadiometricWindow.TOTAL_COUNT: 0.0066,
        RadiometricWindow.POTASSIUM: 0.0058,
        RadiometricWindow.URANIUM: 0.0052,
        RadiometricWindow.THORIUM: 0.0046,
        RadiometricWindow.CESIUM: 0.0072
    }
    
    def __init__(self, reference_altitude: float = 60.0,
                 attenuation_coefficients: Dict[RadiometricWindow, float] = None):
        self.reference_altitude = reference_altitude  # meters
        self.attenuation = attenuation_coefficients or dict(self.DEFAULT_ATTENUATION)
        
    def correct(self, counts: float, altitude: float, 
               window: RadiometricWindow) -> float:
        """
        Correct counts to reference altitude.
        
        Args:
            counts: Measured counts
            altitude: Measurement altitude in meters
            window: Radiometric window
            
        Returns:
            Altitude-corrected counts
        """
        if window not in self.attenuation:
            return counts
            
        mu = self.attenuation[window]
        
        # Exponential correction
        correction_factor = np.exp(mu * (altitude - self.reference_altitude))
        
        return counts * correction_factor
        
    def calibrate_attenuation(self, altitude_profile: np.ndarray,
                             count_profile: np.ndarray,
                             window: RadiometricWindow) -> float:
        """
        Calibrate attenuation coefficient from altitude profile.
        
        Args:
            altitude_profile: Array of altitudes
            count_profile: Array of counts at each altitude
            window: Window being calibrated
            
        Returns:
            Calibrated attenuation coefficient
        """
        # Fit exponential: counts = A * exp(-mu * altitude)
        # ln(counts) = ln(A) - mu * altitude
        
        valid = count_profile > 0
        ln_counts = np.log(count_profile[valid])
        altitudes = altitude_profile[valid]
        
        # Linear fit
        coeffs = np.polyfit(altitudes, ln_counts, 1)
        mu = -coeffs[0]
        
        self.attenuation[window] = mu
        return mu


class DeadTimeCorrection:
    """
    Dead-time correction for gamma-ray spectrometers.
    
    Accounts for count losses at high count rates.
    """
    
    def __init__(self, dead_time: float = 5e-6):
        self.dead_time = dead_time  # seconds
        
    def correct(self, measured_counts: float, live_time: float) -> float:
        """
        Apply dead-time correction.
        
        Args:
            measured_counts: Measured counts
            live_time: Live time in seconds
            
        Returns:
            Dead-time corrected counts
        """
        if live_time <= 0:
            return measured_counts
            
        count_rate = measured_counts / live_time
        
        # Non-paralyzable model
        correction_factor = 1.0 / (1.0 - count_rate * self.dead_time)
        
        # Limit correction to reasonable values
        correction_factor = min(correction_factor, 2.0)
        
        return measured_counts * correction_factor


class BackgroundCorrection:
    """
    Background correction for radiometric data.
    
    Removes aircraft background, cosmic ray background, and radon.
    """
    
    def __init__(self, aircraft_background: Dict[str, float] = None,
                 cosmic_coefficients: Dict[str, float] = None):
        self.aircraft_background = aircraft_background or {}
        self.cosmic_coefficients = cosmic_coefficients or {
            'potassium': 0.0,
            'uranium': 0.0,
            'thorium': 0.0
        }
        
    def set_aircraft_background(self, background_flight: Dict[str, np.ndarray]) -> None:
        """
        Set aircraft background from high-altitude flight.
        
        Args:
            background_flight: Dictionary with window counts from background flight
        """
        for window, counts in background_flight.items():
            self.aircraft_background[window] = np.mean(counts)
            
    def correct(self, counts: Dict[str, float], 
               cosmic_counts: float = 0) -> Dict[str, float]:
        """
        Apply background correction.
        
        Args:
            counts: Window counts dictionary
            cosmic_counts: Cosmic window counts for cosmic correction
            
        Returns:
            Background-corrected counts
        """
        corrected = {}
        
        for window, count in counts.items():
            # Subtract aircraft background
            bg = self.aircraft_background.get(window, 0)
            
            # Subtract cosmic contribution
            cosmic_coef = self.cosmic_coefficients.get(window, 0)
            cosmic_bg = cosmic_coef * cosmic_counts
            
            corrected[window] = max(0, count - bg - cosmic_bg)
            
        return corrected


class SensitivityCalibration:
    """
    Sensitivity calibration to convert counts to concentrations.
    """
    
    def __init__(self, sensitivities: Dict[str, float] = None):
        # Default sensitivities (cps per unit concentration at reference altitude)
        # These are typical values and should be calibrated for each system
        self.sensitivities = sensitivities or {
            'potassium': 100.0,  # cps per %K
            'uranium': 10.0,     # cps per ppm eU
            'thorium': 5.0       # cps per ppm eTh
        }
        
    def counts_to_concentration(self, counts: float, 
                               element: str, live_time: float) -> float:
        """
        Convert counts to concentration.
        
        Args:
            counts: Corrected counts
            element: Element name (potassium, uranium, thorium)
            live_time: Live time in seconds
            
        Returns:
            Concentration (% for K, ppm for U and Th)
        """
        if element not in self.sensitivities:
            return 0.0
            
        count_rate = counts / live_time if live_time > 0 else 0
        sensitivity = self.sensitivities[element]
        
        return count_rate / sensitivity if sensitivity > 0 else 0
        
    def calibrate_from_pads(self, pad_data: List[CalibrationPad],
                           live_time: float) -> Dict[str, float]:
        """
        Calibrate sensitivities from calibration pad measurements.
        
        Args:
            pad_data: List of calibration pad measurements
            live_time: Live time for pad measurements
            
        Returns:
            Calibrated sensitivities
        """
        # Collect data for regression
        k_conc, k_counts = [], []
        u_conc, u_counts = [], []
        th_conc, th_counts = [], []
        
        for pad in pad_data:
            if pad.k_concentration > 0:
                k_conc.append(pad.k_concentration)
                k_counts.append(pad.measured_counts.get('potassium', 0) / live_time)
            if pad.u_concentration > 0:
                u_conc.append(pad.u_concentration)
                u_counts.append(pad.measured_counts.get('uranium', 0) / live_time)
            if pad.th_concentration > 0:
                th_conc.append(pad.th_concentration)
                th_counts.append(pad.measured_counts.get('thorium', 0) / live_time)
                
        # Linear regression through origin
        if len(k_conc) > 0:
            self.sensitivities['potassium'] = np.sum(np.array(k_counts) * np.array(k_conc)) / np.sum(np.array(k_conc)**2)
        if len(u_conc) > 0:
            self.sensitivities['uranium'] = np.sum(np.array(u_counts) * np.array(u_conc)) / np.sum(np.array(u_conc)**2)
        if len(th_conc) > 0:
            self.sensitivities['thorium'] = np.sum(np.array(th_counts) * np.array(th_conc)) / np.sum(np.array(th_conc)**2)
            
        return self.sensitivities


class DoseRateCalculation:
    """
    Calculate dose rate from radiometric concentrations.
    """
    
    # Dose rate conversion factors (nGy/h per unit concentration)
    # Based on IAEA TECDOC-1363
    CONVERSION_FACTORS = {
        'potassium': 13.078,  # nGy/h per %K
        'uranium': 5.675,     # nGy/h per ppm eU
        'thorium': 2.494      # nGy/h per ppm eTh
    }
    
    def calculate(self, k_percent: float, eu_ppm: float, eth_ppm: float) -> float:
        """
        Calculate dose rate from concentrations.
        
        Args:
            k_percent: Potassium concentration in %
            eu_ppm: Equivalent uranium in ppm
            eth_ppm: Equivalent thorium in ppm
            
        Returns:
            Dose rate in nGy/h
        """
        dose = (
            self.CONVERSION_FACTORS['potassium'] * k_percent +
            self.CONVERSION_FACTORS['uranium'] * eu_ppm +
            self.CONVERSION_FACTORS['thorium'] * eth_ppm
        )
        return dose


class RadiometricsPipeline:
    """
    Complete radiometric data processing pipeline.
    """
    
    def __init__(self, config: SpectrometerConfig = None):
        self.config = config
        self.energy_calibration = EnergyCalibration()
        self.window_extraction = WindowExtraction(self.energy_calibration)
        self.stripping = StrippingCorrection()
        self.altitude_correction = AltitudeCorrection()
        self.dead_time_correction = DeadTimeCorrection()
        self.background_correction = BackgroundCorrection()
        self.sensitivity = SensitivityCalibration()
        self.dose_rate = DoseRateCalculation()
        
        self.processing_history: List[Dict] = []
        
    def process_measurement(self, measurement: RadiometricMeasurement,
                           apply_dead_time: bool = True,
                           apply_background: bool = True,
                           apply_stripping: bool = True,
                           apply_altitude: bool = True) -> Dict[str, float]:
        """
        Process a single radiometric measurement.
        
        Args:
            measurement: RadiometricMeasurement object
            apply_dead_time: Whether to apply dead-time correction
            apply_background: Whether to apply background correction
            apply_stripping: Whether to apply stripping correction
            apply_altitude: Whether to apply altitude correction
            
        Returns:
            Dictionary of processed products
        """
        # Extract window counts
        if measurement.spectrum is not None:
            window_counts = self.window_extraction.extract_all_windows(measurement.spectrum)
        else:
            window_counts = dict(measurement.window_counts)
            
        # Dead-time correction
        if apply_dead_time:
            for window in window_counts:
                window_counts[window] = self.dead_time_correction.correct(
                    window_counts[window], measurement.live_time
                )
                
        # Background correction
        if apply_background:
            cosmic = window_counts.get(RadiometricWindow.COSMIC.value, 0)
            window_counts = self.background_correction.correct(window_counts, cosmic)
            
        # Altitude correction
        if apply_altitude:
            for window_name, window_enum in [
                ('potassium', RadiometricWindow.POTASSIUM),
                ('uranium', RadiometricWindow.URANIUM),
                ('thorium', RadiometricWindow.THORIUM),
                ('total_count', RadiometricWindow.TOTAL_COUNT)
            ]:
                if window_name in window_counts:
                    window_counts[window_name] = self.altitude_correction.correct(
                        window_counts[window_name], measurement.altitude, window_enum
                    )
                    
        # Stripping correction
        k_counts = window_counts.get('potassium', 0)
        u_counts = window_counts.get('uranium', 0)
        th_counts = window_counts.get('thorium', 0)
        
        if apply_stripping:
            k_stripped, u_stripped, th_stripped = self.stripping.apply(
                k_counts, u_counts, th_counts
            )
        else:
            k_stripped, u_stripped, th_stripped = k_counts, u_counts, th_counts
            
        # Convert to concentrations
        k_percent = self.sensitivity.counts_to_concentration(
            k_stripped, 'potassium', measurement.live_time
        )
        eu_ppm = self.sensitivity.counts_to_concentration(
            u_stripped, 'uranium', measurement.live_time
        )
        eth_ppm = self.sensitivity.counts_to_concentration(
            th_stripped, 'thorium', measurement.live_time
        )
        
        # Calculate dose rate
        dose = self.dose_rate.calculate(k_percent, eu_ppm, eth_ppm)
        
        # Calculate ratios
        u_th_ratio = eu_ppm / eth_ppm if eth_ppm > 0 else 0
        u_k_ratio = eu_ppm / k_percent if k_percent > 0 else 0
        k_th_ratio = k_percent / eth_ppm if eth_ppm > 0 else 0
        
        return {
            RadiometricProduct.POTASSIUM_PERCENT.value: k_percent,
            RadiometricProduct.EQUIVALENT_URANIUM.value: eu_ppm,
            RadiometricProduct.EQUIVALENT_THORIUM.value: eth_ppm,
            RadiometricProduct.TOTAL_COUNT.value: window_counts.get('total_count', 0),
            RadiometricProduct.DOSE_RATE.value: dose,
            RadiometricProduct.URANIUM_THORIUM_RATIO.value: u_th_ratio,
            RadiometricProduct.URANIUM_POTASSIUM_RATIO.value: u_k_ratio,
            RadiometricProduct.POTASSIUM_THORIUM_RATIO.value: k_th_ratio,
            'latitude': measurement.latitude,
            'longitude': measurement.longitude,
            'altitude': measurement.altitude,
            'timestamp': measurement.timestamp
        }
        
    def process_survey(self, measurements: List[RadiometricMeasurement],
                      **kwargs) -> pd.DataFrame:
        """
        Process complete radiometric survey.
        
        Args:
            measurements: List of measurements
            **kwargs: Processing options
            
        Returns:
            DataFrame with processed results
        """
        results = []
        
        for measurement in measurements:
            result = self.process_measurement(measurement, **kwargs)
            results.append(result)
            
        df = pd.DataFrame(results)
        
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'num_measurements': len(measurements),
            'options': kwargs
        })
        
        return df
        
    def grid_data(self, df: pd.DataFrame, 
                 product: RadiometricProduct,
                 cell_size: float = 50.0) -> xr.DataArray:
        """
        Grid radiometric data to regular grid.
        
        Args:
            df: Processed data DataFrame
            product: Product to grid
            cell_size: Grid cell size in meters
            
        Returns:
            Gridded data as xarray DataArray
        """
        x = df['longitude'].values
        y = df['latitude'].values
        z = df[product.value].values
        
        # Create grid
        cell_deg = cell_size / 111000
        xi = np.arange(x.min(), x.max() + cell_deg, cell_deg)
        yi = np.arange(y.min(), y.max() + cell_deg, cell_deg)
        XI, YI = np.meshgrid(xi, yi)
        
        # Grid using scipy
        ZI = interpolate.griddata((x, y), z, (XI, YI), method='cubic')
        
        # Fill NaN
        mask = np.isnan(ZI)
        if mask.any():
            ZI_nearest = interpolate.griddata((x, y), z, (XI, YI), method='nearest')
            ZI[mask] = ZI_nearest[mask]
            
        return xr.DataArray(
            data=ZI,
            dims=['y', 'x'],
            coords={'y': yi, 'x': xi},
            attrs={'product': product.value, 'cell_size': cell_size}
        )
        
    def create_ternary_image(self, df: pd.DataFrame,
                            cell_size: float = 50.0) -> xr.DataArray:
        """
        Create RGB ternary image from K, U, Th data.
        
        Args:
            df: Processed data DataFrame
            cell_size: Grid cell size in meters
            
        Returns:
            RGB ternary image as xarray DataArray
        """
        # Grid each element
        k_grid = self.grid_data(df, RadiometricProduct.POTASSIUM_PERCENT, cell_size)
        u_grid = self.grid_data(df, RadiometricProduct.EQUIVALENT_URANIUM, cell_size)
        th_grid = self.grid_data(df, RadiometricProduct.EQUIVALENT_THORIUM, cell_size)
        
        # Normalize to 0-255
        def normalize(data):
            data = np.nan_to_num(data, nan=0)
            if data.max() > data.min():
                return ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            return np.zeros_like(data, dtype=np.uint8)
            
        # Standard ternary: R=K, G=Th, B=U
        r = normalize(k_grid.values)
        g = normalize(th_grid.values)
        b = normalize(u_grid.values)
        
        rgb = np.stack([r, g, b], axis=-1)
        
        return xr.DataArray(
            data=rgb,
            dims=['y', 'x', 'channel'],
            coords={'y': k_grid.coords['y'], 'x': k_grid.coords['x'], 
                   'channel': ['R', 'G', 'B']},
            attrs={'product': 'ternary_image', 'mapping': 'R=K, G=Th, B=U'}
        )


def create_radiometrics_pipeline(config: SpectrometerConfig = None) -> RadiometricsPipeline:
    """Factory function to create radiometrics pipeline."""
    return RadiometricsPipeline(config)


def create_survey_constraints(sensor_type: str = 'MS-350') -> Dict[str, Any]:
    """
    Create survey constraints for radiometric missions.
    
    Args:
        sensor_type: Medusa sensor type (MS-350, MS-700, MS-1000)
        
    Returns:
        Survey constraint dictionary
    """
    # Sensor-specific constraints
    sensor_specs = {
        'MS-350': {
            'weight_kg': 2.7,
            'min_count_rate': 100,  # cps for statistical validity
            'optimal_altitude': 30,  # meters
            'max_altitude': 60,
            'optimal_speed': 5,  # m/s
            'max_speed': 10
        },
        'MS-700': {
            'weight_kg': 4.7,
            'min_count_rate': 200,
            'optimal_altitude': 40,
            'max_altitude': 80,
            'optimal_speed': 7,
            'max_speed': 12
        },
        'MS-1000': {
            'weight_kg': 6.7,
            'min_count_rate': 300,
            'optimal_altitude': 50,
            'max_altitude': 100,
            'optimal_speed': 8,
            'max_speed': 15
        }
    }
    
    specs = sensor_specs.get(sensor_type, sensor_specs['MS-350'])
    
    return {
        'sensor_type': sensor_type,
        'weight_kg': specs['weight_kg'],
        'altitude': {
            'optimal': specs['optimal_altitude'],
            'max': specs['max_altitude'],
            'tolerance': 5  # meters
        },
        'speed': {
            'optimal': specs['optimal_speed'],
            'max': specs['max_speed'],
            'tolerance': 2  # m/s
        },
        'weather': {
            'max_wind_speed': 10,  # m/s
            'min_visibility': 3000,  # meters
            'no_precipitation': True
        },
        'line_spacing': {
            'recommended': 50,  # meters for detailed survey
            'max': 100
        },
        'statistics': {
            'min_count_rate': specs['min_count_rate'],
            'min_live_time': 1.0  # seconds per sample
        }
    }
