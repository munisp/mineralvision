"""
Ground Penetrating Radar (GPR) Processing Pipeline for MineralVision.

This module provides comprehensive GPR data processing including:
- Time-zero correction and dewow filtering
- Background removal and bandpass filtering
- Gain functions (SEC, AGC)
- Velocity estimation and migration
- Depth conversion
- Deliverables: profiles, horizontal slices, thickness grids, 3D volumes
- Survey method metadata handling

Based on SPH Engineering Zond Aero GPR systems and GSSI/Sensors & Software standards.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, interpolate, ndimage
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class GPRSurveyMethod(Enum):
    """GPR survey methods."""
    CART = "cart"
    HANDHELD = "handheld"
    DRAGGED = "dragged"
    DRONE_MOUNTED = "drone_mounted"
    VEHICLE_MOUNTED = "vehicle_mounted"


class GPRAntennaFrequency(Enum):
    """Standard GPR antenna frequencies."""
    LF_50MHZ = 50
    LF_75MHZ = 75
    LF_100MHZ = 100
    LF_200MHZ = 200
    LF_400MHZ = 400
    MF_500MHZ = 500
    MF_600MHZ = 600
    HF_1000MHZ = 1000
    HF_1600MHZ = 1600
    HF_2000MHZ = 2000


class MigrationMethod(Enum):
    """Migration algorithms."""
    KIRCHHOFF = "kirchhoff"
    STOLT_FK = "stolt_fk"
    PHASE_SHIFT = "phase_shift"
    FINITE_DIFFERENCE = "finite_difference"
    REVERSE_TIME = "reverse_time"


class GainType(Enum):
    """Gain function types."""
    SEC = "spherical_exponential_compensation"
    AGC = "automatic_gain_control"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


@dataclass
class GPRConfig:
    """GPR system configuration."""
    system_id: str
    antenna_frequency: float  # MHz
    sample_rate: float  # ns per sample
    num_samples: int
    time_window: float  # ns
    antenna_separation: float  # meters
    survey_method: GPRSurveyMethod
    position_mode: str  # wheel, gps, manual
    stacking: int = 1
    calibration_date: Optional[datetime] = None


@dataclass
class GPRTrace:
    """Single GPR trace."""
    trace_id: int
    position: float  # meters along line
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    time_axis: Optional[np.ndarray] = None
    quality_flag: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPRLine:
    """GPR survey line (B-scan)."""
    line_id: str
    traces: List[GPRTrace]
    config: GPRConfig
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert traces to 2D array."""
        return np.column_stack([t.data for t in self.traces])
        
    def get_positions(self) -> np.ndarray:
        """Get trace positions."""
        return np.array([t.position for t in self.traces])
        
    def get_time_axis(self) -> np.ndarray:
        """Get time axis."""
        if self.traces and self.traces[0].time_axis is not None:
            return self.traces[0].time_axis
        return np.arange(self.config.num_samples) * self.config.sample_rate


class TimeZeroCorrection:
    """
    Time-zero correction for GPR data.
    
    Aligns all traces to a common time-zero based on first arrival
    or direct wave detection.
    """
    
    def __init__(self, method: str = 'first_break'):
        self.method = method
        self.time_zero_samples: Optional[np.ndarray] = None
        
    def detect_time_zero(self, data: np.ndarray, 
                        threshold: float = 0.1) -> np.ndarray:
        """
        Detect time-zero for each trace.
        
        Args:
            data: 2D GPR data array (samples x traces)
            threshold: Detection threshold (fraction of max amplitude)
            
        Returns:
            Array of time-zero sample indices
        """
        num_samples, num_traces = data.shape
        time_zeros = np.zeros(num_traces, dtype=int)
        
        for i in range(num_traces):
            trace = data[:, i]
            
            if self.method == 'first_break':
                # Find first sample exceeding threshold
                max_amp = np.max(np.abs(trace))
                threshold_value = threshold * max_amp
                
                above_threshold = np.where(np.abs(trace) > threshold_value)[0]
                if len(above_threshold) > 0:
                    time_zeros[i] = above_threshold[0]
                    
            elif self.method == 'max_amplitude':
                # Find maximum amplitude in early part of trace
                search_window = min(100, num_samples // 4)
                time_zeros[i] = np.argmax(np.abs(trace[:search_window]))
                
            elif self.method == 'correlation':
                # Cross-correlate with reference trace
                if i == 0:
                    time_zeros[i] = 0
                else:
                    ref_trace = data[:, 0]
                    correlation = np.correlate(trace, ref_trace, mode='full')
                    lag = np.argmax(correlation) - (num_samples - 1)
                    time_zeros[i] = max(0, -lag)
                    
        self.time_zero_samples = time_zeros
        return time_zeros
        
    def apply(self, data: np.ndarray, 
             time_zeros: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply time-zero correction.
        
        Args:
            data: 2D GPR data array
            time_zeros: Time-zero samples (uses detected if None)
            
        Returns:
            Time-zero corrected data
        """
        if time_zeros is None:
            if self.time_zero_samples is None:
                time_zeros = self.detect_time_zero(data)
            else:
                time_zeros = self.time_zero_samples
                
        num_samples, num_traces = data.shape
        corrected = np.zeros_like(data)
        
        # Use median time-zero as reference
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


class DewowFilter:
    """
    Dewow filter to remove low-frequency wow from GPR data.
    
    Removes DC drift and very low frequency components caused by
    inductive coupling between transmitter and receiver.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply dewow filter.
        
        Args:
            data: 2D GPR data array
            
        Returns:
            Dewowed data
        """
        num_samples, num_traces = data.shape
        dewowed = np.zeros_like(data)
        
        for i in range(num_traces):
            trace = data[:, i]
            
            # Running mean filter
            kernel = np.ones(self.window_size) / self.window_size
            wow = np.convolve(trace, kernel, mode='same')
            
            dewowed[:, i] = trace - wow
            
        return dewowed


class BackgroundRemoval:
    """
    Background removal for GPR data.
    
    Removes horizontal banding caused by system ringing,
    antenna coupling, and other systematic noise.
    """
    
    def __init__(self, method: str = 'mean'):
        self.method = method
        self.background: Optional[np.ndarray] = None
        
    def compute_background(self, data: np.ndarray) -> np.ndarray:
        """
        Compute background trace.
        
        Args:
            data: 2D GPR data array
            
        Returns:
            Background trace
        """
        if self.method == 'mean':
            self.background = np.mean(data, axis=1)
        elif self.method == 'median':
            self.background = np.median(data, axis=1)
        elif self.method == 'moving_average':
            # Moving average along traces
            kernel_size = min(50, data.shape[1] // 4)
            kernel = np.ones((1, kernel_size)) / kernel_size
            smoothed = ndimage.convolve(data, kernel, mode='reflect')
            self.background = np.mean(smoothed, axis=1)
            
        return self.background
        
    def apply(self, data: np.ndarray, 
             background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply background removal.
        
        Args:
            data: 2D GPR data array
            background: Background trace (computes if None)
            
        Returns:
            Background-removed data
        """
        if background is None:
            if self.background is None:
                background = self.compute_background(data)
            else:
                background = self.background
                
        return data - background[:, np.newaxis]


class BandpassFilter:
    """
    Bandpass filter for GPR data.
    """
    
    def __init__(self, low_freq: float = None, high_freq: float = None,
                 sample_rate: float = 1.0):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sample_rate = sample_rate  # ns
        
    def design_filter(self, num_samples: int) -> np.ndarray:
        """
        Design bandpass filter.
        
        Args:
            num_samples: Number of samples per trace
            
        Returns:
            Filter coefficients
        """
        # Convert sample rate to frequency
        fs = 1.0 / (self.sample_rate * 1e-9)  # Hz
        nyquist = fs / 2
        
        # Normalize frequencies
        if self.low_freq is not None:
            low_norm = self.low_freq * 1e6 / nyquist
        else:
            low_norm = 0.01
            
        if self.high_freq is not None:
            high_norm = self.high_freq * 1e6 / nyquist
        else:
            high_norm = 0.99
            
        # Ensure valid range
        low_norm = max(0.01, min(0.99, low_norm))
        high_norm = max(0.01, min(0.99, high_norm))
        
        if low_norm >= high_norm:
            low_norm = 0.01
            high_norm = 0.99
            
        # Design Butterworth filter
        order = 4
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        return b, a
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter.
        
        Args:
            data: 2D GPR data array
            
        Returns:
            Filtered data
        """
        num_samples, num_traces = data.shape
        b, a = self.design_filter(num_samples)
        
        filtered = np.zeros_like(data)
        
        for i in range(num_traces):
            # Apply zero-phase filter
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
            
        return filtered


class GainFunction:
    """
    Gain functions for GPR data.
    
    Compensates for signal attenuation with depth.
    """
    
    def __init__(self, gain_type: GainType = GainType.SEC):
        self.gain_type = gain_type
        self.parameters: Dict[str, float] = {}
        
    def set_parameters(self, **kwargs) -> None:
        """Set gain parameters."""
        self.parameters.update(kwargs)
        
    def compute_gain_curve(self, num_samples: int, 
                          sample_rate: float) -> np.ndarray:
        """
        Compute gain curve.
        
        Args:
            num_samples: Number of samples
            sample_rate: Sample rate in ns
            
        Returns:
            Gain curve array
        """
        time = np.arange(num_samples) * sample_rate  # ns
        
        if self.gain_type == GainType.SEC:
            # Spherical and exponential compensation
            # g(t) = t^a * exp(b*t)
            a = self.parameters.get('power', 1.0)
            b = self.parameters.get('attenuation', 0.01)
            
            # Avoid division by zero at t=0
            time_safe = np.maximum(time, 1.0)
            gain = (time_safe ** a) * np.exp(b * time)
            
        elif self.gain_type == GainType.LINEAR:
            # Linear gain
            slope = self.parameters.get('slope', 0.01)
            gain = 1.0 + slope * time
            
        elif self.gain_type == GainType.EXPONENTIAL:
            # Exponential gain
            rate = self.parameters.get('rate', 0.01)
            gain = np.exp(rate * time)
            
        else:
            gain = np.ones(num_samples)
            
        # Normalize
        gain = gain / gain[0] if gain[0] != 0 else gain
        
        return gain
        
    def apply(self, data: np.ndarray, sample_rate: float = 1.0) -> np.ndarray:
        """
        Apply gain function.
        
        Args:
            data: 2D GPR data array
            sample_rate: Sample rate in ns
            
        Returns:
            Gained data
        """
        num_samples, num_traces = data.shape
        
        if self.gain_type == GainType.AGC:
            # Automatic gain control - trace by trace
            return self._apply_agc(data)
        else:
            gain_curve = self.compute_gain_curve(num_samples, sample_rate)
            return data * gain_curve[:, np.newaxis]
            
    def _apply_agc(self, data: np.ndarray, 
                  window_size: int = 50) -> np.ndarray:
        """Apply AGC."""
        num_samples, num_traces = data.shape
        gained = np.zeros_like(data)
        
        for i in range(num_traces):
            trace = data[:, i]
            
            # Compute envelope using Hilbert transform
            analytic = signal.hilbert(trace)
            envelope = np.abs(analytic)
            
            # Smooth envelope
            kernel = np.ones(window_size) / window_size
            smooth_envelope = np.convolve(envelope, kernel, mode='same')
            
            # Avoid division by zero
            smooth_envelope = np.maximum(smooth_envelope, 1e-10)
            
            # Normalize
            target_amplitude = self.parameters.get('target_amplitude', 1.0)
            gained[:, i] = trace * target_amplitude / smooth_envelope
            
        return gained


class VelocityEstimation:
    """
    Velocity estimation for GPR data.
    
    Estimates electromagnetic wave velocity for depth conversion.
    """
    
    # Typical velocities for different materials (m/ns)
    MATERIAL_VELOCITIES = {
        'air': 0.30,
        'dry_sand': 0.15,
        'wet_sand': 0.06,
        'dry_soil': 0.13,
        'wet_soil': 0.06,
        'clay': 0.06,
        'limestone': 0.12,
        'granite': 0.13,
        'ice': 0.17,
        'fresh_water': 0.033,
        'sea_water': 0.01,
        'concrete': 0.10,
        'asphalt': 0.12
    }
    
    def __init__(self, method: str = 'hyperbola'):
        self.method = method
        self.estimated_velocity: Optional[float] = None
        
    def estimate_from_hyperbola(self, data: np.ndarray,
                               positions: np.ndarray,
                               time_axis: np.ndarray,
                               hyperbola_apex: Tuple[int, int]) -> float:
        """
        Estimate velocity from hyperbola fitting.
        
        Args:
            data: 2D GPR data array
            positions: Trace positions in meters
            time_axis: Time axis in ns
            hyperbola_apex: (sample, trace) of hyperbola apex
            
        Returns:
            Estimated velocity in m/ns
        """
        apex_sample, apex_trace = hyperbola_apex
        apex_time = time_axis[apex_sample]
        apex_position = positions[apex_trace]
        
        # Find hyperbola points by tracking maximum amplitude
        num_samples, num_traces = data.shape
        hyperbola_times = []
        hyperbola_positions = []
        
        # Search window around apex
        search_window = 20  # samples
        
        for i in range(num_traces):
            # Search for local maximum near expected hyperbola
            dx = positions[i] - apex_position
            
            # Initial estimate assuming velocity
            v_init = 0.1  # m/ns
            expected_time = np.sqrt(apex_time**2 + (2*dx/v_init)**2)
            expected_sample = int(expected_time / (time_axis[1] - time_axis[0]))
            
            if 0 <= expected_sample < num_samples:
                start = max(0, expected_sample - search_window)
                end = min(num_samples, expected_sample + search_window)
                
                local_max = start + np.argmax(np.abs(data[start:end, i]))
                
                hyperbola_times.append(time_axis[local_max])
                hyperbola_positions.append(positions[i])
                
        # Fit hyperbola: t^2 = t0^2 + (2x/v)^2
        if len(hyperbola_times) > 3:
            t = np.array(hyperbola_times)
            x = np.array(hyperbola_positions) - apex_position
            
            # Least squares fit
            t_squared = t**2
            x_squared = x**2
            
            # t^2 = t0^2 + 4*x^2/v^2
            # Fit: t^2 = a + b*x^2, where b = 4/v^2
            A = np.column_stack([np.ones_like(x_squared), x_squared])
            coeffs, _, _, _ = np.linalg.lstsq(A, t_squared, rcond=None)
            
            if coeffs[1] > 0:
                v = 2.0 / np.sqrt(coeffs[1])
                self.estimated_velocity = v
                return v
                
        # Default velocity
        self.estimated_velocity = 0.1
        return 0.1
        
    def estimate_from_cmp(self, cmp_gather: np.ndarray,
                         offsets: np.ndarray,
                         time_axis: np.ndarray) -> float:
        """
        Estimate velocity from CMP gather (if available).
        
        Args:
            cmp_gather: CMP gather data
            offsets: Antenna offsets
            time_axis: Time axis
            
        Returns:
            Estimated velocity
        """
        # Semblance analysis
        velocities = np.linspace(0.05, 0.20, 50)  # m/ns
        semblance = np.zeros((len(time_axis), len(velocities)))
        
        for iv, v in enumerate(velocities):
            for it, t0 in enumerate(time_axis):
                if t0 > 0:
                    # NMO correction
                    t_nmo = np.sqrt(t0**2 + (offsets / v)**2)
                    
                    # Interpolate and stack
                    stack = 0
                    energy = 0
                    
                    for io, offset in enumerate(offsets):
                        t_idx = np.interp(t_nmo[io], time_axis, 
                                         np.arange(len(time_axis)))
                        if 0 <= t_idx < len(time_axis) - 1:
                            idx = int(t_idx)
                            frac = t_idx - idx
                            val = (1-frac) * cmp_gather[idx, io] + frac * cmp_gather[idx+1, io]
                            stack += val
                            energy += val**2
                            
                    if energy > 0:
                        semblance[it, iv] = stack**2 / (len(offsets) * energy)
                        
        # Find maximum semblance
        max_idx = np.unravel_index(np.argmax(semblance), semblance.shape)
        self.estimated_velocity = velocities[max_idx[1]]
        
        return self.estimated_velocity


class Migration:
    """
    Migration algorithms for GPR data.
    
    Collapses diffraction hyperbolas and positions reflectors correctly.
    """
    
    def __init__(self, method: MigrationMethod = MigrationMethod.KIRCHHOFF):
        self.method = method
        self.velocity: float = 0.1  # m/ns
        
    def set_velocity(self, velocity: float) -> None:
        """Set migration velocity."""
        self.velocity = velocity
        
    def migrate(self, data: np.ndarray, positions: np.ndarray,
               time_axis: np.ndarray) -> np.ndarray:
        """
        Apply migration.
        
        Args:
            data: 2D GPR data array
            positions: Trace positions in meters
            time_axis: Time axis in ns
            
        Returns:
            Migrated data
        """
        if self.method == MigrationMethod.KIRCHHOFF:
            return self._kirchhoff_migration(data, positions, time_axis)
        elif self.method == MigrationMethod.STOLT_FK:
            return self._stolt_fk_migration(data, positions, time_axis)
        elif self.method == MigrationMethod.PHASE_SHIFT:
            return self._phase_shift_migration(data, positions, time_axis)
        else:
            return data
            
    def _kirchhoff_migration(self, data: np.ndarray, 
                            positions: np.ndarray,
                            time_axis: np.ndarray) -> np.ndarray:
        """Kirchhoff diffraction summation migration."""
        num_samples, num_traces = data.shape
        migrated = np.zeros_like(data)
        
        dt = time_axis[1] - time_axis[0]
        dx = positions[1] - positions[0] if len(positions) > 1 else 1.0
        
        # Aperture (number of traces to sum)
        aperture = min(50, num_traces // 4)
        
        for iz in range(num_samples):
            z_time = time_axis[iz]
            z_depth = z_time * self.velocity / 2  # Two-way time
            
            for ix in range(num_traces):
                x_pos = positions[ix]
                
                # Sum over aperture
                stack = 0
                count = 0
                
                for jx in range(max(0, ix - aperture), 
                               min(num_traces, ix + aperture + 1)):
                    dx_dist = positions[jx] - x_pos
                    
                    # Travel time from (x_pos, z_depth) to surface at jx
                    travel_time = 2 * np.sqrt(z_depth**2 + dx_dist**2) / self.velocity
                    
                    # Find sample index
                    sample_idx = travel_time / dt
                    
                    if 0 <= sample_idx < num_samples - 1:
                        # Linear interpolation
                        idx = int(sample_idx)
                        frac = sample_idx - idx
                        val = (1 - frac) * data[idx, jx] + frac * data[idx + 1, jx]
                        
                        # Obliquity factor
                        obliquity = z_depth / np.sqrt(z_depth**2 + dx_dist**2 + 1e-10)
                        
                        stack += val * obliquity
                        count += 1
                        
                if count > 0:
                    migrated[iz, ix] = stack / np.sqrt(count)
                    
        return migrated
        
    def _stolt_fk_migration(self, data: np.ndarray,
                           positions: np.ndarray,
                           time_axis: np.ndarray) -> np.ndarray:
        """Stolt F-K migration."""
        num_samples, num_traces = data.shape
        
        dt = time_axis[1] - time_axis[0]
        dx = positions[1] - positions[0] if len(positions) > 1 else 1.0
        
        # 2D FFT
        data_fk = fft2(data)
        
        # Frequency axes
        freq_t = fftfreq(num_samples, dt)
        freq_x = fftfreq(num_traces, dx)
        
        FREQ_T, FREQ_X = np.meshgrid(freq_t, freq_x, indexing='ij')
        
        # Stolt stretch: kz = sqrt((2*f/v)^2 - kx^2)
        # Map from (f, kx) to (kz, kx)
        
        kz = np.sqrt(np.maximum((2 * FREQ_T / self.velocity)**2 - FREQ_X**2, 0))
        
        # Phase shift
        phase = np.exp(-1j * np.pi * kz * self.velocity * dt * num_samples / 2)
        
        # Apply phase shift
        data_fk_migrated = data_fk * phase
        
        # Inverse FFT
        migrated = np.real(ifft2(data_fk_migrated))
        
        return migrated
        
    def _phase_shift_migration(self, data: np.ndarray,
                              positions: np.ndarray,
                              time_axis: np.ndarray) -> np.ndarray:
        """Phase-shift migration."""
        num_samples, num_traces = data.shape
        
        dt = time_axis[1] - time_axis[0]
        dx = positions[1] - positions[0] if len(positions) > 1 else 1.0
        
        # FFT in x direction
        data_fx = fft(data, axis=1)
        
        # Frequency axes
        freq_x = fftfreq(num_traces, dx)
        
        migrated = np.zeros_like(data, dtype=complex)
        
        # Downward continuation
        for iz in range(num_samples):
            dz = iz * dt * self.velocity / 2
            
            for ikx, kx in enumerate(freq_x):
                # Phase shift operator
                kz = np.sqrt(np.maximum((2 / (self.velocity * dt))**2 - kx**2, 0))
                phase = np.exp(-1j * 2 * np.pi * kz * dz)
                
                migrated[iz, :] += data_fx[iz, ikx] * phase
                
        return np.real(ifft(migrated, axis=1))


class DepthConversion:
    """
    Convert GPR data from time to depth domain.
    """
    
    def __init__(self, velocity: float = 0.1):
        self.velocity = velocity  # m/ns
        self.velocity_model: Optional[np.ndarray] = None
        
    def set_velocity_model(self, velocity_model: np.ndarray) -> None:
        """Set depth-varying velocity model."""
        self.velocity_model = velocity_model
        
    def time_to_depth(self, time_axis: np.ndarray) -> np.ndarray:
        """
        Convert time axis to depth.
        
        Args:
            time_axis: Time axis in ns
            
        Returns:
            Depth axis in meters
        """
        if self.velocity_model is not None:
            # Integrate velocity model
            depth = np.zeros_like(time_axis)
            dt = time_axis[1] - time_axis[0]
            
            for i in range(1, len(time_axis)):
                v = self.velocity_model[min(i, len(self.velocity_model)-1)]
                depth[i] = depth[i-1] + v * dt / 2  # Two-way time
                
            return depth
        else:
            # Constant velocity
            return time_axis * self.velocity / 2
            
    def convert_data(self, data: np.ndarray, time_axis: np.ndarray,
                    depth_axis: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert data to depth domain.
        
        Args:
            data: 2D GPR data in time domain
            time_axis: Time axis in ns
            depth_axis: Target depth axis (auto if None)
            
        Returns:
            Tuple of (depth_data, depth_axis)
        """
        # Compute depth from time
        time_depths = self.time_to_depth(time_axis)
        
        if depth_axis is None:
            # Create regular depth axis
            depth_axis = np.linspace(0, time_depths[-1], len(time_axis))
            
        # Interpolate data to regular depth
        num_samples, num_traces = data.shape
        depth_data = np.zeros((len(depth_axis), num_traces))
        
        for i in range(num_traces):
            depth_data[:, i] = np.interp(depth_axis, time_depths, data[:, i])
            
        return depth_data, depth_axis


class GPRPipeline:
    """
    Complete GPR processing pipeline.
    """
    
    def __init__(self, config: GPRConfig = None):
        self.config = config
        self.time_zero = TimeZeroCorrection()
        self.dewow = DewowFilter()
        self.background = BackgroundRemoval()
        self.bandpass = BandpassFilter()
        self.gain = GainFunction()
        self.velocity_estimation = VelocityEstimation()
        self.migration = Migration()
        self.depth_conversion = DepthConversion()
        
        self.processing_history: List[Dict] = []
        
    def process_line(self, line: GPRLine,
                    apply_time_zero: bool = True,
                    apply_dewow: bool = True,
                    apply_background: bool = True,
                    apply_bandpass: bool = True,
                    apply_gain: bool = True,
                    apply_migration: bool = False,
                    convert_to_depth: bool = False,
                    velocity: float = None) -> Dict[str, Any]:
        """
        Process a GPR line.
        
        Args:
            line: GPRLine object
            apply_time_zero: Apply time-zero correction
            apply_dewow: Apply dewow filter
            apply_background: Apply background removal
            apply_bandpass: Apply bandpass filter
            apply_gain: Apply gain function
            apply_migration: Apply migration
            convert_to_depth: Convert to depth domain
            velocity: Velocity for migration/depth conversion
            
        Returns:
            Processing results dictionary
        """
        data = line.to_array()
        positions = line.get_positions()
        time_axis = line.get_time_axis()
        
        results = {
            'raw': data.copy(),
            'time_axis': time_axis,
            'positions': positions
        }
        
        # Time-zero correction
        if apply_time_zero:
            data = self.time_zero.apply(data)
            results['time_zero_corrected'] = data.copy()
            
        # Dewow
        if apply_dewow:
            data = self.dewow.apply(data)
            results['dewowed'] = data.copy()
            
        # Background removal
        if apply_background:
            data = self.background.apply(data)
            results['background_removed'] = data.copy()
            
        # Bandpass filter
        if apply_bandpass:
            if self.config:
                # Set filter based on antenna frequency
                center_freq = self.config.antenna_frequency
                self.bandpass.low_freq = center_freq * 0.3
                self.bandpass.high_freq = center_freq * 1.5
                self.bandpass.sample_rate = self.config.sample_rate
            data = self.bandpass.apply(data)
            results['filtered'] = data.copy()
            
        # Gain
        if apply_gain:
            sample_rate = self.config.sample_rate if self.config else 1.0
            data = self.gain.apply(data, sample_rate)
            results['gained'] = data.copy()
            
        # Migration
        if apply_migration:
            if velocity is None:
                velocity = self.velocity_estimation.estimated_velocity or 0.1
            self.migration.set_velocity(velocity)
            data = self.migration.migrate(data, positions, time_axis)
            results['migrated'] = data.copy()
            
        # Depth conversion
        if convert_to_depth:
            if velocity is None:
                velocity = self.velocity_estimation.estimated_velocity or 0.1
            self.depth_conversion.velocity = velocity
            depth_data, depth_axis = self.depth_conversion.convert_data(
                data, time_axis
            )
            results['depth_data'] = depth_data
            results['depth_axis'] = depth_axis
            
        results['processed'] = data
        
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'line_id': line.line_id,
            'num_traces': len(line.traces),
            'processing_steps': {
                'time_zero': apply_time_zero,
                'dewow': apply_dewow,
                'background': apply_background,
                'bandpass': apply_bandpass,
                'gain': apply_gain,
                'migration': apply_migration,
                'depth_conversion': convert_to_depth
            }
        })
        
        return results
        
    def create_horizontal_slice(self, lines: List[GPRLine],
                               time_sample: int,
                               cell_size: float = 0.5) -> xr.DataArray:
        """
        Create horizontal time/depth slice from multiple lines.
        
        Args:
            lines: List of GPR lines
            time_sample: Sample index for slice
            cell_size: Grid cell size in meters
            
        Returns:
            Horizontal slice as xarray DataArray
        """
        # Collect all points
        all_x = []
        all_y = []
        all_z = []
        
        for line in lines:
            data = line.to_array()
            
            for i, trace in enumerate(line.traces):
                if trace.longitude is not None and trace.latitude is not None:
                    all_x.append(trace.longitude)
                    all_y.append(trace.latitude)
                    if time_sample < data.shape[0]:
                        all_z.append(data[time_sample, i])
                        
        if not all_x:
            return None
            
        x = np.array(all_x)
        y = np.array(all_y)
        z = np.array(all_z)
        
        # Create grid
        cell_deg = cell_size / 111000
        xi = np.arange(x.min(), x.max() + cell_deg, cell_deg)
        yi = np.arange(y.min(), y.max() + cell_deg, cell_deg)
        XI, YI = np.meshgrid(xi, yi)
        
        # Grid
        ZI = interpolate.griddata((x, y), z, (XI, YI), method='cubic')
        
        return xr.DataArray(
            data=ZI,
            dims=['y', 'x'],
            coords={'y': yi, 'x': xi},
            attrs={'time_sample': time_sample, 'cell_size': cell_size}
        )
        
    def create_thickness_grid(self, lines: List[GPRLine],
                             interface_times: Dict[str, np.ndarray],
                             velocity: float = 0.1,
                             cell_size: float = 0.5) -> xr.DataArray:
        """
        Create thickness grid from picked interfaces.
        
        Args:
            lines: List of GPR lines
            interface_times: Dictionary mapping line_id to interface times
            velocity: Velocity for time-to-depth conversion
            cell_size: Grid cell size in meters
            
        Returns:
            Thickness grid as xarray DataArray
        """
        all_x = []
        all_y = []
        all_thickness = []
        
        for line in lines:
            if line.line_id not in interface_times:
                continue
                
            times = interface_times[line.line_id]
            
            for i, trace in enumerate(line.traces):
                if trace.longitude is not None and trace.latitude is not None:
                    if i < len(times):
                        thickness = times[i] * velocity / 2  # Two-way time
                        all_x.append(trace.longitude)
                        all_y.append(trace.latitude)
                        all_thickness.append(thickness)
                        
        if not all_x:
            return None
            
        x = np.array(all_x)
        y = np.array(all_y)
        z = np.array(all_thickness)
        
        # Create grid
        cell_deg = cell_size / 111000
        xi = np.arange(x.min(), x.max() + cell_deg, cell_deg)
        yi = np.arange(y.min(), y.max() + cell_deg, cell_deg)
        XI, YI = np.meshgrid(xi, yi)
        
        # Grid
        ZI = interpolate.griddata((x, y), z, (XI, YI), method='cubic')
        
        return xr.DataArray(
            data=ZI,
            dims=['y', 'x'],
            coords={'y': yi, 'x': xi},
            attrs={'velocity': velocity, 'cell_size': cell_size, 'units': 'meters'}
        )
        
    def create_3d_volume(self, lines: List[GPRLine],
                        cell_size: float = 0.5) -> xr.DataArray:
        """
        Create 3D volume from multiple GPR lines.
        
        Args:
            lines: List of GPR lines
            cell_size: Grid cell size in meters
            
        Returns:
            3D volume as xarray DataArray
        """
        if not lines:
            return None
            
        # Get dimensions
        num_samples = lines[0].config.num_samples
        time_axis = lines[0].get_time_axis()
        
        # Collect all trace locations
        all_x = []
        all_y = []
        
        for line in lines:
            for trace in line.traces:
                if trace.longitude is not None and trace.latitude is not None:
                    all_x.append(trace.longitude)
                    all_y.append(trace.latitude)
                    
        if not all_x:
            return None
            
        # Create grid
        cell_deg = cell_size / 111000
        xi = np.arange(min(all_x), max(all_x) + cell_deg, cell_deg)
        yi = np.arange(min(all_y), max(all_y) + cell_deg, cell_deg)
        
        # Initialize volume
        volume = np.full((num_samples, len(yi), len(xi)), np.nan)
        
        # Populate volume
        for line in lines:
            data = line.to_array()
            
            for i, trace in enumerate(line.traces):
                if trace.longitude is not None and trace.latitude is not None:
                    # Find nearest grid cell
                    ix = np.argmin(np.abs(xi - trace.longitude))
                    iy = np.argmin(np.abs(yi - trace.latitude))
                    
                    # Insert trace
                    volume[:, iy, ix] = data[:, i]
                    
        # Interpolate missing values (simple nearest neighbor)
        for iz in range(num_samples):
            slice_data = volume[iz, :, :]
            mask = np.isnan(slice_data)
            
            if mask.any() and not mask.all():
                # Get valid points
                valid_y, valid_x = np.where(~mask)
                valid_z = slice_data[~mask]
                
                # Interpolate
                XI, YI = np.meshgrid(np.arange(len(xi)), np.arange(len(yi)))
                interp = interpolate.griddata(
                    (valid_x, valid_y), valid_z, (XI, YI), method='nearest'
                )
                volume[iz, :, :] = interp
                
        return xr.DataArray(
            data=volume,
            dims=['time', 'y', 'x'],
            coords={'time': time_axis, 'y': yi, 'x': xi},
            attrs={'cell_size': cell_size}
        )


def create_gpr_pipeline(config: GPRConfig = None) -> GPRPipeline:
    """Factory function to create GPR pipeline."""
    return GPRPipeline(config)


def create_survey_design_gpr(area_bounds: Tuple[float, float, float, float],
                            line_spacing: float = 1.0,
                            line_direction: float = 0.0,
                            survey_method: GPRSurveyMethod = GPRSurveyMethod.DRONE_MOUNTED,
                            antenna_frequency: float = 500.0) -> Dict[str, Any]:
    """
    Create GPR survey design.
    
    Args:
        area_bounds: (min_lon, min_lat, max_lon, max_lat)
        line_spacing: Line spacing in meters
        line_direction: Line direction in degrees from north
        survey_method: Survey method
        antenna_frequency: Antenna frequency in MHz
        
    Returns:
        Survey design dictionary
    """
    min_lon, min_lat, max_lon, max_lat = area_bounds
    
    # Convert spacing to degrees
    line_spacing_deg = line_spacing / 111000
    
    # Generate lines
    lines = []
    if line_direction == 0 or line_direction == 180:
        x = min_lon
        line_num = 1
        while x <= max_lon:
            lines.append({
                'line_id': f'GPR{line_num:04d}',
                'start': (x, min_lat),
                'end': (x, max_lat),
                'direction': line_direction
            })
            x += line_spacing_deg
            line_num += 1
    else:
        y = min_lat
        line_num = 1
        while y <= max_lat:
            lines.append({
                'line_id': f'GPR{line_num:04d}',
                'start': (min_lon, y),
                'end': (max_lon, y),
                'direction': line_direction
            })
            y += line_spacing_deg
            line_num += 1
            
    # Estimate penetration depth based on frequency
    penetration_estimates = {
        50: 30.0,    # meters in dry conditions
        100: 15.0,
        200: 8.0,
        400: 4.0,
        500: 3.0,
        600: 2.5,
        1000: 0.5
    }
    
    freq_key = min(penetration_estimates.keys(), 
                  key=lambda k: abs(k - antenna_frequency))
    estimated_penetration = penetration_estimates[freq_key]
    
    # Survey constraints based on method
    if survey_method == GPRSurveyMethod.DRONE_MOUNTED:
        constraints = {
            'altitude': {'min': 0.5, 'max': 2.0, 'optimal': 1.0},
            'speed': {'min': 1.0, 'max': 5.0, 'optimal': 3.0},
            'terrain_following': True
        }
    elif survey_method == GPRSurveyMethod.CART:
        constraints = {
            'speed': {'min': 0.5, 'max': 2.0, 'optimal': 1.0},
            'surface': 'hard_surface_required'
        }
    else:
        constraints = {
            'speed': {'min': 0.3, 'max': 1.0, 'optimal': 0.5}
        }
        
    # Calculate statistics
    total_km = sum(
        np.sqrt((l['end'][0] - l['start'][0])**2 + 
               (l['end'][1] - l['start'][1])**2) * 111
        for l in lines
    )
    
    optimal_speed = constraints.get('speed', {}).get('optimal', 1.0)
    estimated_time_hours = total_km * 1000 / optimal_speed / 3600
    
    return {
        'lines': lines,
        'parameters': {
            'line_spacing': line_spacing,
            'line_direction': line_direction,
            'survey_method': survey_method.value,
            'antenna_frequency': antenna_frequency
        },
        'constraints': constraints,
        'estimates': {
            'penetration_depth': estimated_penetration,
            'total_km': total_km,
            'estimated_time_hours': estimated_time_hours,
            'num_lines': len(lines)
        }
    }
