"""
Drone Telemetry Normalization for MineralVision.

This module provides:
- Time synchronization between sensors and GNSS
- Lever-arm corrections for sensor offsets
- Terrain-following quality metrics
- Speed/altitude compliance checking
- Sensor health monitoring
- Automated "bad line" detection

Makes drone-acquired geophysical data usable at scale.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class FlightQuality(Enum):
    """Flight quality classification."""
    EXCELLENT = "excellent"   # All metrics within spec
    GOOD = "good"            # Minor deviations
    ACCEPTABLE = "acceptable" # Some issues but usable
    POOR = "poor"            # Significant issues
    REJECTED = "rejected"    # Not usable


class SensorHealth(Enum):
    """Sensor health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class LineStatus(Enum):
    """Survey line status."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass
class GNSSPosition:
    """GNSS position with accuracy."""
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_msl: float  # meters above mean sea level
    altitude_agl: float  # meters above ground level
    horizontal_accuracy: float  # meters
    vertical_accuracy: float  # meters
    n_satellites: int
    fix_type: str  # 'rtk_fixed', 'rtk_float', 'dgps', 'autonomous'
    
    @property
    def is_rtk(self) -> bool:
        return self.fix_type in ['rtk_fixed', 'rtk_float']
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude_msl': self.altitude_msl,
            'altitude_agl': self.altitude_agl,
            'horizontal_accuracy': self.horizontal_accuracy,
            'vertical_accuracy': self.vertical_accuracy,
            'n_satellites': self.n_satellites,
            'fix_type': self.fix_type
        }


@dataclass
class IMUData:
    """IMU orientation data."""
    timestamp: datetime
    roll: float      # degrees
    pitch: float     # degrees
    yaw: float       # degrees (heading)
    roll_rate: float   # deg/s
    pitch_rate: float  # deg/s
    yaw_rate: float    # deg/s
    accel_x: float   # m/s^2
    accel_y: float   # m/s^2
    accel_z: float   # m/s^2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'roll_rate': self.roll_rate,
            'pitch_rate': self.pitch_rate,
            'yaw_rate': self.yaw_rate,
            'accel_x': self.accel_x,
            'accel_y': self.accel_y,
            'accel_z': self.accel_z
        }


@dataclass
class LeverArm:
    """Lever arm offset from GNSS antenna to sensor."""
    sensor_name: str
    dx: float  # meters, forward positive
    dy: float  # meters, right positive
    dz: float  # meters, down positive
    
    def apply_correction(self, gnss: GNSSPosition, imu: IMUData) -> Tuple[float, float, float]:
        """
        Apply lever arm correction.
        
        Args:
            gnss: GNSS position
            imu: IMU orientation
            
        Returns:
            Corrected (lat, lon, alt) for sensor position
        """
        # Convert angles to radians
        roll_rad = np.radians(imu.roll)
        pitch_rad = np.radians(imu.pitch)
        yaw_rad = np.radians(imu.yaw)
        
        # Rotation matrix (simplified)
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Body to navigation frame rotation
        R = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])
        
        # Lever arm in body frame
        lever_body = np.array([self.dx, self.dy, self.dz])
        
        # Transform to navigation frame
        lever_nav = R @ lever_body
        
        # Convert to lat/lon offset (approximate)
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(gnss.latitude))
        
        lat_offset = lever_nav[0] / meters_per_deg_lat
        lon_offset = lever_nav[1] / meters_per_deg_lon
        alt_offset = -lever_nav[2]  # Negative because dz is down positive
        
        return (
            gnss.latitude + lat_offset,
            gnss.longitude + lon_offset,
            gnss.altitude_msl + alt_offset
        )


@dataclass
class SensorReading:
    """Generic sensor reading with metadata."""
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: float  # 0-1 quality indicator
    health: SensorHealth
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlightLine:
    """Survey flight line."""
    line_id: str
    line_number: int
    start_time: datetime
    end_time: datetime
    n_points: int
    mean_altitude_agl: float
    std_altitude_agl: float
    mean_speed: float
    std_speed: float
    mean_heading: float
    heading_deviation: float
    status: LineStatus
    quality_score: float
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_id': self.line_id,
            'line_number': self.line_number,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'n_points': self.n_points,
            'mean_altitude_agl': self.mean_altitude_agl,
            'std_altitude_agl': self.std_altitude_agl,
            'mean_speed': self.mean_speed,
            'std_speed': self.std_speed,
            'mean_heading': self.mean_heading,
            'heading_deviation': self.heading_deviation,
            'status': self.status.value,
            'quality_score': self.quality_score,
            'issues': self.issues
        }


@dataclass
class FlightQualityReport:
    """Complete flight quality report."""
    flight_id: str
    aircraft_id: str
    pilot: str
    date: datetime
    total_lines: int
    valid_lines: int
    rejected_lines: int
    total_distance_km: float
    flight_duration_minutes: float
    overall_quality: FlightQuality
    quality_score: float
    lines: List[FlightLine]
    sensor_health: Dict[str, SensorHealth]
    compliance: Dict[str, bool]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'flight_id': self.flight_id,
            'aircraft_id': self.aircraft_id,
            'pilot': self.pilot,
            'date': self.date.isoformat(),
            'total_lines': self.total_lines,
            'valid_lines': self.valid_lines,
            'rejected_lines': self.rejected_lines,
            'total_distance_km': self.total_distance_km,
            'flight_duration_minutes': self.flight_duration_minutes,
            'overall_quality': self.overall_quality.value,
            'quality_score': self.quality_score,
            'lines': [l.to_dict() for l in self.lines],
            'sensor_health': {k: v.value for k, v in self.sensor_health.items()},
            'compliance': self.compliance,
            'recommendations': self.recommendations
        }


class TimeSynchronizer:
    """
    Synchronize timestamps between sensors.
    
    Handles clock drift and latency compensation.
    """
    
    def __init__(self, reference_sensor: str = 'gnss'):
        self.reference_sensor = reference_sensor
        self.offsets: Dict[str, float] = {}  # sensor -> offset in seconds
        self.drift_rates: Dict[str, float] = {}  # sensor -> drift rate in ppm
        
    def calibrate(self, reference_times: np.ndarray,
                 sensor_times: np.ndarray,
                 sensor_id: str) -> Dict[str, float]:
        """
        Calibrate time offset for a sensor.
        
        Args:
            reference_times: Reference timestamps (seconds)
            sensor_times: Sensor timestamps (seconds)
            sensor_id: Sensor identifier
            
        Returns:
            Calibration results
        """
        if len(reference_times) != len(sensor_times):
            raise ValueError("Time arrays must have same length")
            
        # Calculate offsets
        offsets = sensor_times - reference_times
        
        # Fit linear model for drift
        if len(offsets) > 1:
            coeffs = np.polyfit(reference_times, offsets, 1)
            drift_rate = coeffs[0] * 1e6  # ppm
            initial_offset = coeffs[1]
        else:
            drift_rate = 0
            initial_offset = offsets[0] if len(offsets) > 0 else 0
            
        self.offsets[sensor_id] = initial_offset
        self.drift_rates[sensor_id] = drift_rate
        
        return {
            'sensor_id': sensor_id,
            'initial_offset_ms': initial_offset * 1000,
            'drift_rate_ppm': drift_rate,
            'residual_std_ms': np.std(offsets - np.polyval(coeffs, reference_times)) * 1000 if len(offsets) > 1 else 0
        }
    
    def synchronize(self, timestamp: float, sensor_id: str,
                   reference_time: float = None) -> float:
        """
        Synchronize a timestamp to reference time.
        
        Args:
            timestamp: Sensor timestamp (seconds)
            sensor_id: Sensor identifier
            reference_time: Current reference time for drift correction
            
        Returns:
            Synchronized timestamp
        """
        offset = self.offsets.get(sensor_id, 0)
        drift = self.drift_rates.get(sensor_id, 0)
        
        # Apply drift correction if reference time provided
        if reference_time is not None and drift != 0:
            drift_correction = drift * reference_time * 1e-6
            offset += drift_correction
            
        return timestamp - offset


class TerrainFollowingAnalyzer:
    """
    Analyze terrain-following performance.
    
    Evaluates altitude maintenance relative to ground.
    """
    
    def __init__(self, target_altitude_agl: float = 50.0,
                tolerance: float = 5.0):
        self.target_altitude = target_altitude_agl
        self.tolerance = tolerance
        
    def analyze(self, altitudes_agl: np.ndarray,
               timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Analyze terrain-following quality.
        
        Args:
            altitudes_agl: Above ground level altitudes (meters)
            timestamps: Timestamps (seconds)
            
        Returns:
            Analysis results
        """
        # Basic statistics
        mean_alt = np.mean(altitudes_agl)
        std_alt = np.std(altitudes_agl)
        min_alt = np.min(altitudes_agl)
        max_alt = np.max(altitudes_agl)
        
        # Deviation from target
        deviations = altitudes_agl - self.target_altitude
        mean_deviation = np.mean(deviations)
        max_deviation = np.max(np.abs(deviations))
        
        # Compliance
        within_tolerance = np.sum(np.abs(deviations) <= self.tolerance)
        compliance_percent = (within_tolerance / len(altitudes_agl)) * 100
        
        # Rate of change (terrain following responsiveness)
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            dalt = np.diff(altitudes_agl)
            climb_rates = dalt / dt
            max_climb_rate = np.max(np.abs(climb_rates))
        else:
            max_climb_rate = 0
            
        # Quality score
        quality_score = min(1.0, compliance_percent / 100)
        if max_deviation > self.tolerance * 2:
            quality_score *= 0.8
        if std_alt > self.tolerance:
            quality_score *= 0.9
            
        return {
            'target_altitude_m': self.target_altitude,
            'tolerance_m': self.tolerance,
            'mean_altitude_m': mean_alt,
            'std_altitude_m': std_alt,
            'min_altitude_m': min_alt,
            'max_altitude_m': max_alt,
            'mean_deviation_m': mean_deviation,
            'max_deviation_m': max_deviation,
            'compliance_percent': compliance_percent,
            'max_climb_rate_mps': max_climb_rate,
            'quality_score': quality_score
        }


class SpeedComplianceChecker:
    """
    Check flight speed compliance.
    
    Ensures consistent ground speed for data quality.
    """
    
    def __init__(self, target_speed: float = 10.0,
                tolerance_percent: float = 20.0):
        self.target_speed = target_speed
        self.tolerance_percent = tolerance_percent
        
    def check(self, speeds: np.ndarray) -> Dict[str, Any]:
        """
        Check speed compliance.
        
        Args:
            speeds: Ground speeds (m/s)
            
        Returns:
            Compliance results
        """
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        min_speed = np.min(speeds)
        max_speed = np.max(speeds)
        
        # Tolerance bounds
        lower_bound = self.target_speed * (1 - self.tolerance_percent / 100)
        upper_bound = self.target_speed * (1 + self.tolerance_percent / 100)
        
        # Compliance
        within_tolerance = np.sum((speeds >= lower_bound) & (speeds <= upper_bound))
        compliance_percent = (within_tolerance / len(speeds)) * 100
        
        # Quality score
        cv = std_speed / mean_speed if mean_speed > 0 else 1
        quality_score = max(0, 1 - cv) * (compliance_percent / 100)
        
        return {
            'target_speed_mps': self.target_speed,
            'tolerance_percent': self.tolerance_percent,
            'mean_speed_mps': mean_speed,
            'std_speed_mps': std_speed,
            'min_speed_mps': min_speed,
            'max_speed_mps': max_speed,
            'compliance_percent': compliance_percent,
            'quality_score': quality_score
        }


class SensorHealthMonitor:
    """
    Monitor sensor health during flight.
    
    Detects anomalies and failures.
    """
    
    def __init__(self):
        self.thresholds: Dict[str, Dict[str, float]] = {
            'magnetometer': {
                'noise_nt': 1.0,
                'spike_threshold': 100.0,
                'dropout_max_seconds': 1.0
            },
            'spectrometer': {
                'dead_time_percent': 10.0,
                'count_rate_min': 100,
                'count_rate_max': 100000
            },
            'gpr': {
                'signal_strength_min': -60,
                'noise_floor_max': -80
            }
        }
        
    def assess(self, sensor_type: str,
              readings: np.ndarray,
              timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Assess sensor health.
        
        Args:
            sensor_type: Type of sensor
            readings: Sensor readings
            timestamps: Timestamps
            
        Returns:
            Health assessment
        """
        thresholds = self.thresholds.get(sensor_type, {})
        issues = []
        health = SensorHealth.HEALTHY
        
        # Check for dropouts
        if len(timestamps) > 1:
            gaps = np.diff(timestamps)
            max_gap = np.max(gaps)
            dropout_threshold = thresholds.get('dropout_max_seconds', 1.0)
            if max_gap > dropout_threshold:
                issues.append(f"Data dropout detected: {max_gap:.2f}s gap")
                health = SensorHealth.DEGRADED
                
        # Check for noise
        if len(readings) > 10:
            noise = np.std(np.diff(readings))
            noise_threshold = thresholds.get('noise_nt', float('inf'))
            if noise > noise_threshold:
                issues.append(f"High noise level: {noise:.2f}")
                health = SensorHealth.DEGRADED
                
        # Check for spikes
        if len(readings) > 0:
            mean_val = np.mean(readings)
            std_val = np.std(readings)
            spikes = np.sum(np.abs(readings - mean_val) > 5 * std_val)
            if spikes > len(readings) * 0.01:
                issues.append(f"Spike detected: {spikes} anomalous readings")
                health = SensorHealth.DEGRADED
                
        # Check for complete failure
        if len(readings) == 0 or np.all(np.isnan(readings)):
            health = SensorHealth.FAILED
            issues.append("No valid readings")
            
        return {
            'sensor_type': sensor_type,
            'health': health.value,
            'n_readings': len(readings),
            'issues': issues,
            'statistics': {
                'mean': float(np.nanmean(readings)) if len(readings) > 0 else None,
                'std': float(np.nanstd(readings)) if len(readings) > 0 else None,
                'min': float(np.nanmin(readings)) if len(readings) > 0 else None,
                'max': float(np.nanmax(readings)) if len(readings) > 0 else None
            }
        }


class BadLineDetector:
    """
    Automated detection of bad survey lines.
    
    Identifies lines that should be reflown.
    """
    
    def __init__(self):
        self.criteria = {
            'min_points': 100,
            'max_altitude_std': 10.0,
            'max_speed_cv': 0.3,
            'max_heading_deviation': 10.0,
            'min_data_coverage': 0.9
        }
        
    def evaluate_line(self, line_data: Dict[str, np.ndarray]) -> FlightLine:
        """
        Evaluate a single survey line.
        
        Args:
            line_data: Dict with 'timestamps', 'altitudes', 'speeds', 'headings', 'readings'
            
        Returns:
            FlightLine with quality assessment
        """
        timestamps = line_data.get('timestamps', np.array([]))
        altitudes = line_data.get('altitudes', np.array([]))
        speeds = line_data.get('speeds', np.array([]))
        headings = line_data.get('headings', np.array([]))
        readings = line_data.get('readings', np.array([]))
        
        issues = []
        quality_score = 1.0
        
        n_points = len(timestamps)
        
        # Check minimum points
        if n_points < self.criteria['min_points']:
            issues.append(f"Insufficient points: {n_points}")
            quality_score *= 0.5
            
        # Check altitude stability
        if len(altitudes) > 0:
            alt_std = np.std(altitudes)
            if alt_std > self.criteria['max_altitude_std']:
                issues.append(f"High altitude variation: {alt_std:.1f}m")
                quality_score *= 0.8
                
        # Check speed consistency
        if len(speeds) > 0:
            speed_mean = np.mean(speeds)
            speed_std = np.std(speeds)
            speed_cv = speed_std / speed_mean if speed_mean > 0 else 1
            if speed_cv > self.criteria['max_speed_cv']:
                issues.append(f"Inconsistent speed: CV={speed_cv:.2f}")
                quality_score *= 0.8
                
        # Check heading deviation
        if len(headings) > 0:
            # Circular mean for headings
            mean_heading = np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(headings))),
                np.mean(np.cos(np.radians(headings)))
            ))
            heading_dev = np.std(np.abs(headings - mean_heading))
            if heading_dev > self.criteria['max_heading_deviation']:
                issues.append(f"High heading deviation: {heading_dev:.1f}°")
                quality_score *= 0.9
                
        # Check data coverage
        if len(readings) > 0:
            valid_readings = np.sum(~np.isnan(readings))
            coverage = valid_readings / len(readings)
            if coverage < self.criteria['min_data_coverage']:
                issues.append(f"Low data coverage: {coverage*100:.1f}%")
                quality_score *= coverage
                
        # Determine status
        if quality_score >= 0.8:
            status = LineStatus.VALID
        elif quality_score >= 0.5:
            status = LineStatus.WARNING
        else:
            status = LineStatus.INVALID
            
        return FlightLine(
            line_id=line_data.get('line_id', 'unknown'),
            line_number=line_data.get('line_number', 0),
            start_time=datetime.fromtimestamp(timestamps[0]) if len(timestamps) > 0 else datetime.now(),
            end_time=datetime.fromtimestamp(timestamps[-1]) if len(timestamps) > 0 else datetime.now(),
            n_points=n_points,
            mean_altitude_agl=float(np.mean(altitudes)) if len(altitudes) > 0 else 0,
            std_altitude_agl=float(np.std(altitudes)) if len(altitudes) > 0 else 0,
            mean_speed=float(np.mean(speeds)) if len(speeds) > 0 else 0,
            std_speed=float(np.std(speeds)) if len(speeds) > 0 else 0,
            mean_heading=float(np.mean(headings)) if len(headings) > 0 else 0,
            heading_deviation=float(np.std(headings)) if len(headings) > 0 else 0,
            status=status,
            quality_score=quality_score,
            issues=issues
        )


class DroneTelemetryNormalizer:
    """
    Complete drone telemetry normalization pipeline.
    
    Integrates all telemetry processing components.
    """
    
    def __init__(self, target_altitude: float = 50.0,
                target_speed: float = 10.0):
        self.time_sync = TimeSynchronizer()
        self.terrain_analyzer = TerrainFollowingAnalyzer(target_altitude)
        self.speed_checker = SpeedComplianceChecker(target_speed)
        self.health_monitor = SensorHealthMonitor()
        self.line_detector = BadLineDetector()
        self.lever_arms: Dict[str, LeverArm] = {}
        
    def add_lever_arm(self, sensor_name: str, dx: float, dy: float, dz: float):
        """Add lever arm configuration for a sensor."""
        self.lever_arms[sensor_name] = LeverArm(sensor_name, dx, dy, dz)
        
    def normalize_flight(self, flight_data: Dict[str, Any]) -> FlightQualityReport:
        """
        Normalize and assess a complete flight.
        
        Args:
            flight_data: Dict with flight telemetry
            
        Returns:
            FlightQualityReport
        """
        flight_id = flight_data.get('flight_id', 'unknown')
        aircraft_id = flight_data.get('aircraft_id', 'unknown')
        pilot = flight_data.get('pilot', 'unknown')
        
        lines_data = flight_data.get('lines', [])
        sensor_readings = flight_data.get('sensors', {})
        
        # Process each line
        lines = []
        for line_data in lines_data:
            line = self.line_detector.evaluate_line(line_data)
            lines.append(line)
            
        # Assess sensor health
        sensor_health = {}
        for sensor_name, readings in sensor_readings.items():
            health_result = self.health_monitor.assess(
                sensor_name,
                readings.get('values', np.array([])),
                readings.get('timestamps', np.array([]))
            )
            sensor_health[sensor_name] = SensorHealth(health_result['health'])
            
        # Calculate overall metrics
        total_lines = len(lines)
        valid_lines = len([l for l in lines if l.status == LineStatus.VALID])
        rejected_lines = len([l for l in lines if l.status == LineStatus.INVALID])
        
        # Calculate total distance
        total_distance = sum(l.mean_speed * (l.end_time - l.start_time).total_seconds() / 1000 
                           for l in lines)
        
        # Calculate flight duration
        if lines:
            flight_start = min(l.start_time for l in lines)
            flight_end = max(l.end_time for l in lines)
            flight_duration = (flight_end - flight_start).total_seconds() / 60
        else:
            flight_duration = 0
            
        # Overall quality score
        if total_lines > 0:
            quality_score = sum(l.quality_score for l in lines) / total_lines
        else:
            quality_score = 0
            
        # Determine overall quality
        if quality_score >= 0.9 and rejected_lines == 0:
            overall_quality = FlightQuality.EXCELLENT
        elif quality_score >= 0.8:
            overall_quality = FlightQuality.GOOD
        elif quality_score >= 0.6:
            overall_quality = FlightQuality.ACCEPTABLE
        elif quality_score >= 0.4:
            overall_quality = FlightQuality.POOR
        else:
            overall_quality = FlightQuality.REJECTED
            
        # Compliance checks
        compliance = {
            'altitude_compliance': all(l.std_altitude_agl < 10 for l in lines),
            'speed_compliance': all(l.std_speed < l.mean_speed * 0.3 for l in lines if l.mean_speed > 0),
            'coverage_compliance': valid_lines / total_lines >= 0.8 if total_lines > 0 else False,
            'sensor_health': all(h == SensorHealth.HEALTHY for h in sensor_health.values())
        }
        
        # Generate recommendations
        recommendations = []
        if rejected_lines > 0:
            recommendations.append(f"Reflying {rejected_lines} rejected lines recommended")
        if not compliance['altitude_compliance']:
            recommendations.append("Improve terrain-following performance")
        if not compliance['speed_compliance']:
            recommendations.append("Maintain more consistent flight speed")
        if not compliance['sensor_health']:
            unhealthy = [k for k, v in sensor_health.items() if v != SensorHealth.HEALTHY]
            recommendations.append(f"Check sensors: {', '.join(unhealthy)}")
            
        return FlightQualityReport(
            flight_id=flight_id,
            aircraft_id=aircraft_id,
            pilot=pilot,
            date=datetime.now(),
            total_lines=total_lines,
            valid_lines=valid_lines,
            rejected_lines=rejected_lines,
            total_distance_km=total_distance,
            flight_duration_minutes=flight_duration,
            overall_quality=overall_quality,
            quality_score=quality_score,
            lines=lines,
            sensor_health=sensor_health,
            compliance=compliance,
            recommendations=recommendations
        )
    
    def apply_corrections(self, gnss_data: List[GNSSPosition],
                         imu_data: List[IMUData],
                         sensor_name: str) -> List[Tuple[float, float, float]]:
        """
        Apply lever arm corrections to get sensor positions.
        
        Args:
            gnss_data: GNSS positions
            imu_data: IMU orientations
            sensor_name: Sensor to correct for
            
        Returns:
            List of corrected (lat, lon, alt) positions
        """
        if sensor_name not in self.lever_arms:
            return [(g.latitude, g.longitude, g.altitude_msl) for g in gnss_data]
            
        lever_arm = self.lever_arms[sensor_name]
        corrected = []
        
        for gnss, imu in zip(gnss_data, imu_data):
            lat, lon, alt = lever_arm.apply_correction(gnss, imu)
            corrected.append((lat, lon, alt))
            
        return corrected


# Factory functions
def create_telemetry_normalizer(target_altitude: float = 50.0,
                               target_speed: float = 10.0) -> DroneTelemetryNormalizer:
    """Create drone telemetry normalizer."""
    return DroneTelemetryNormalizer(target_altitude, target_speed)


def create_time_synchronizer() -> TimeSynchronizer:
    """Create time synchronizer."""
    return TimeSynchronizer()


def create_bad_line_detector() -> BadLineDetector:
    """Create bad line detector."""
    return BadLineDetector()
