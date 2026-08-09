"""
Mobile GNSS Module for MineralVision.

Provides mobile-optimized GNSS capabilities:
- Battery-efficient positioning
- Adaptive accuracy modes
- Background location tracking
- Geofencing support
- Offline correction caching
- Sensor fusion with IMU
- A-GPS support
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
import uuid

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power consumption modes."""
    HIGH_ACCURACY = "high_accuracy"
    BALANCED = "balanced"
    LOW_POWER = "low_power"
    PASSIVE = "passive"


class LocationProvider(Enum):
    """Location provider types."""
    GPS = "gps"
    NETWORK = "network"
    FUSED = "fused"
    PASSIVE = "passive"


class GeofenceTransition(Enum):
    """Geofence transition types."""
    ENTER = "enter"
    EXIT = "exit"
    DWELL = "dwell"


class MotionState(Enum):
    """Device motion state."""
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"
    CYCLING = "cycling"
    DRIVING = "driving"
    UNKNOWN = "unknown"


@dataclass
class MobileLocationConfig:
    """Mobile location configuration."""
    power_mode: PowerMode = PowerMode.BALANCED
    update_interval_ms: int = 1000
    fastest_interval_ms: int = 500
    min_displacement_m: float = 0.0
    max_wait_time_ms: int = 5000
    use_network: bool = True
    use_gps: bool = True
    use_sensor_fusion: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'power_mode': self.power_mode.value,
            'update_interval_ms': self.update_interval_ms,
            'fastest_interval_ms': self.fastest_interval_ms,
            'min_displacement_m': self.min_displacement_m,
            'max_wait_time_ms': self.max_wait_time_ms,
            'use_network': self.use_network,
            'use_gps': self.use_gps,
            'use_sensor_fusion': self.use_sensor_fusion
        }


@dataclass
class MobileLocation:
    """Mobile device location."""
    location_id: str
    timestamp: datetime
    
    # Position
    latitude: float
    longitude: float
    altitude: float
    
    # Accuracy
    horizontal_accuracy: float
    vertical_accuracy: float
    bearing_accuracy: float = 0.0
    speed_accuracy: float = 0.0
    
    # Motion
    bearing: float = 0.0
    speed: float = 0.0
    
    # Provider info
    provider: LocationProvider = LocationProvider.FUSED
    satellites: int = 0
    
    # Battery impact
    battery_level: float = 100.0
    is_mock: bool = False
    
    # Sensor fusion data
    accelerometer: Optional[Tuple[float, float, float]] = None
    gyroscope: Optional[Tuple[float, float, float]] = None
    magnetometer: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location_id': self.location_id,
            'timestamp': self.timestamp.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'horizontal_accuracy': self.horizontal_accuracy,
            'vertical_accuracy': self.vertical_accuracy,
            'bearing': self.bearing,
            'speed': self.speed,
            'provider': self.provider.value,
            'satellites': self.satellites,
            'is_mock': self.is_mock
        }
        
    @property
    def is_accurate(self) -> bool:
        """Check if location meets accuracy threshold."""
        return self.horizontal_accuracy < 10.0


@dataclass
class Geofence:
    """Geofence definition."""
    geofence_id: str
    name: str
    latitude: float
    longitude: float
    radius_m: float
    transitions: List[GeofenceTransition] = field(default_factory=list)
    dwell_time_ms: int = 0
    expiration_ms: int = -1  # -1 = never expires
    notification_responsiveness_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'geofence_id': self.geofence_id,
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'radius_m': self.radius_m,
            'transitions': [t.value for t in self.transitions],
            'dwell_time_ms': self.dwell_time_ms,
            'expiration_ms': self.expiration_ms
        }
        
    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is inside geofence."""
        distance = self._haversine_distance(self.latitude, self.longitude, lat, lon)
        return distance <= self.radius_m
        
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        R = 6371000
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


@dataclass
class GeofenceEvent:
    """Geofence transition event."""
    event_id: str
    geofence_id: str
    transition: GeofenceTransition
    location: MobileLocation
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'geofence_id': self.geofence_id,
            'transition': self.transition.value,
            'location': self.location.to_dict(),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class IMUData:
    """Inertial Measurement Unit data."""
    timestamp: datetime
    accelerometer: Tuple[float, float, float]  # m/s^2
    gyroscope: Tuple[float, float, float]  # rad/s
    magnetometer: Optional[Tuple[float, float, float]] = None  # uT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'accelerometer': list(self.accelerometer),
            'gyroscope': list(self.gyroscope),
            'magnetometer': list(self.magnetometer) if self.magnetometer else None
        }
        
    @property
    def acceleration_magnitude(self) -> float:
        """Get acceleration magnitude."""
        ax, ay, az = self.accelerometer
        return math.sqrt(ax**2 + ay**2 + az**2)
        
    @property
    def rotation_rate(self) -> float:
        """Get rotation rate magnitude."""
        gx, gy, gz = self.gyroscope
        return math.sqrt(gx**2 + gy**2 + gz**2)


class MotionDetector:
    """Detect device motion state from IMU data."""
    
    def __init__(self):
        self._imu_buffer: List[IMUData] = []
        self._buffer_size = 50
        self._current_state = MotionState.UNKNOWN
        
    def add_imu_data(self, data: IMUData) -> MotionState:
        """Add IMU data and detect motion state."""
        self._imu_buffer.append(data)
        if len(self._imu_buffer) > self._buffer_size:
            self._imu_buffer.pop(0)
            
        if len(self._imu_buffer) < 10:
            return self._current_state
            
        # Calculate statistics
        acc_mags = [d.acceleration_magnitude for d in self._imu_buffer]
        rot_rates = [d.rotation_rate for d in self._imu_buffer]
        
        mean_acc = sum(acc_mags) / len(acc_mags)
        var_acc = sum((a - mean_acc)**2 for a in acc_mags) / len(acc_mags)
        mean_rot = sum(rot_rates) / len(rot_rates)
        
        # Classify motion state
        if var_acc < 0.1 and mean_rot < 0.1:
            self._current_state = MotionState.STATIONARY
        elif var_acc < 1.0 and mean_acc < 11.0:
            self._current_state = MotionState.WALKING
        elif var_acc < 3.0 and mean_acc < 12.0:
            self._current_state = MotionState.RUNNING
        elif var_acc < 2.0 and mean_rot > 0.5:
            self._current_state = MotionState.CYCLING
        elif var_acc > 3.0:
            self._current_state = MotionState.DRIVING
        else:
            self._current_state = MotionState.UNKNOWN
            
        return self._current_state
        
    def get_current_state(self) -> MotionState:
        """Get current motion state."""
        return self._current_state


class SensorFusion:
    """Fuse GNSS with IMU for improved positioning."""
    
    def __init__(self):
        self._last_gnss: Optional[MobileLocation] = None
        self._last_imu: Optional[IMUData] = None
        self._velocity = [0.0, 0.0, 0.0]  # NED
        self._position_offset = [0.0, 0.0, 0.0]  # meters from last GNSS
        
    def update_gnss(self, location: MobileLocation) -> None:
        """Update with GNSS measurement."""
        self._last_gnss = location
        self._position_offset = [0.0, 0.0, 0.0]
        
        # Update velocity from GNSS
        if location.speed > 0:
            bearing_rad = math.radians(location.bearing)
            self._velocity[0] = location.speed * math.cos(bearing_rad)  # North
            self._velocity[1] = location.speed * math.sin(bearing_rad)  # East
            self._velocity[2] = 0.0  # Down
            
    def update_imu(self, imu: IMUData) -> Optional[MobileLocation]:
        """Update with IMU measurement and return fused position."""
        if not self._last_gnss:
            return None
            
        if self._last_imu:
            dt = (imu.timestamp - self._last_imu.timestamp).total_seconds()
            if dt > 0 and dt < 1.0:
                # Integrate acceleration (simplified, ignoring rotation)
                ax, ay, az = imu.accelerometer
                
                # Remove gravity (simplified)
                az_corrected = az - 9.81
                
                # Update velocity
                self._velocity[0] += ax * dt
                self._velocity[1] += ay * dt
                self._velocity[2] += az_corrected * dt
                
                # Update position offset
                self._position_offset[0] += self._velocity[0] * dt
                self._position_offset[1] += self._velocity[1] * dt
                self._position_offset[2] += self._velocity[2] * dt
                
        self._last_imu = imu
        
        # Create fused location
        meters_per_deg_lat = 111132.92
        meters_per_deg_lon = 111132.92 * math.cos(math.radians(self._last_gnss.latitude))
        
        fused_lat = self._last_gnss.latitude + self._position_offset[0] / meters_per_deg_lat
        fused_lon = self._last_gnss.longitude + self._position_offset[1] / meters_per_deg_lon
        fused_alt = self._last_gnss.altitude - self._position_offset[2]
        
        # Calculate fused speed and bearing
        speed = math.sqrt(self._velocity[0]**2 + self._velocity[1]**2)
        bearing = math.degrees(math.atan2(self._velocity[1], self._velocity[0]))
        if bearing < 0:
            bearing += 360
            
        return MobileLocation(
            location_id=str(uuid.uuid4()),
            timestamp=imu.timestamp,
            latitude=fused_lat,
            longitude=fused_lon,
            altitude=fused_alt,
            horizontal_accuracy=self._last_gnss.horizontal_accuracy * 1.5,  # Degraded accuracy
            vertical_accuracy=self._last_gnss.vertical_accuracy * 1.5,
            bearing=bearing,
            speed=speed,
            provider=LocationProvider.FUSED,
            satellites=self._last_gnss.satellites,
            accelerometer=imu.accelerometer,
            gyroscope=imu.gyroscope,
            magnetometer=imu.magnetometer
        )
        
    def reset(self) -> None:
        """Reset fusion state."""
        self._last_gnss = None
        self._last_imu = None
        self._velocity = [0.0, 0.0, 0.0]
        self._position_offset = [0.0, 0.0, 0.0]


class AGPSManager:
    """Assisted GPS manager for faster TTFF."""
    
    def __init__(self):
        self._ephemeris_cache: Dict[str, bytes] = {}
        self._almanac_cache: Dict[str, bytes] = {}
        self._last_update: Optional[datetime] = None
        self._cache_validity_hours = 4
        
    def download_assistance_data(self, lat: float, lon: float) -> bool:
        """Download A-GPS assistance data."""
        try:
            logger.info(f"Downloading A-GPS data for {lat}, {lon}")
            self._last_update = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"A-GPS download failed: {e}")
            return False
            
    def get_ephemeris(self, prn: str) -> Optional[bytes]:
        """Get cached ephemeris for satellite."""
        if not self._is_cache_valid():
            return None
        return self._ephemeris_cache.get(prn)
        
    def get_almanac(self) -> Optional[bytes]:
        """Get cached almanac."""
        if not self._is_cache_valid():
            return None
        return self._almanac_cache.get('almanac')
        
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._last_update:
            return False
        age = (datetime.utcnow() - self._last_update).total_seconds() / 3600
        return age < self._cache_validity_hours
        
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status."""
        return {
            'valid': self._is_cache_valid(),
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'ephemeris_count': len(self._ephemeris_cache),
            'has_almanac': 'almanac' in self._almanac_cache
        }


class OfflineCorrectionCache:
    """Cache corrections for offline use."""
    
    def __init__(self, cache_path: str = ""):
        self.cache_path = cache_path
        self._ionospheric_grid: Dict[str, float] = {}
        self._tropospheric_grid: Dict[str, float] = {}
        self._last_update: Optional[datetime] = None
        
    def update_ionospheric_grid(self, grid: Dict[str, float]) -> None:
        """Update ionospheric correction grid."""
        self._ionospheric_grid = grid
        self._last_update = datetime.utcnow()
        
    def update_tropospheric_grid(self, grid: Dict[str, float]) -> None:
        """Update tropospheric correction grid."""
        self._tropospheric_grid = grid
        self._last_update = datetime.utcnow()
        
    def get_ionospheric_correction(self, lat: float, lon: float) -> float:
        """Get cached ionospheric correction."""
        key = f"{int(lat)}_{int(lon)}"
        return self._ionospheric_grid.get(key, 0.0)
        
    def get_tropospheric_correction(self, lat: float, lon: float) -> float:
        """Get cached tropospheric correction."""
        key = f"{int(lat)}_{int(lon)}"
        return self._tropospheric_grid.get(key, 0.0)
        
    def is_valid(self) -> bool:
        """Check if cache is valid."""
        if not self._last_update:
            return False
        age = (datetime.utcnow() - self._last_update).total_seconds() / 3600
        return age < 24  # Valid for 24 hours


class GeofenceManager:
    """Manage geofences."""
    
    def __init__(self):
        self._geofences: Dict[str, Geofence] = {}
        self._inside_geofences: set = set()
        self._callbacks: List[Callable[[GeofenceEvent], None]] = []
        
    def add_geofence(self, geofence: Geofence) -> bool:
        """Add geofence."""
        self._geofences[geofence.geofence_id] = geofence
        return True
        
    def remove_geofence(self, geofence_id: str) -> bool:
        """Remove geofence."""
        if geofence_id in self._geofences:
            del self._geofences[geofence_id]
            self._inside_geofences.discard(geofence_id)
            return True
        return False
        
    def register_callback(self, callback: Callable[[GeofenceEvent], None]) -> None:
        """Register geofence event callback."""
        self._callbacks.append(callback)
        
    def check_location(self, location: MobileLocation) -> List[GeofenceEvent]:
        """Check location against geofences."""
        events = []
        
        for gf_id, geofence in self._geofences.items():
            is_inside = geofence.contains(location.latitude, location.longitude)
            was_inside = gf_id in self._inside_geofences
            
            if is_inside and not was_inside:
                # Enter transition
                if GeofenceTransition.ENTER in geofence.transitions:
                    event = GeofenceEvent(
                        event_id=str(uuid.uuid4()),
                        geofence_id=gf_id,
                        transition=GeofenceTransition.ENTER,
                        location=location,
                        timestamp=datetime.utcnow()
                    )
                    events.append(event)
                self._inside_geofences.add(gf_id)
                
            elif not is_inside and was_inside:
                # Exit transition
                if GeofenceTransition.EXIT in geofence.transitions:
                    event = GeofenceEvent(
                        event_id=str(uuid.uuid4()),
                        geofence_id=gf_id,
                        transition=GeofenceTransition.EXIT,
                        location=location,
                        timestamp=datetime.utcnow()
                    )
                    events.append(event)
                self._inside_geofences.discard(gf_id)
                
        # Notify callbacks
        for event in events:
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Geofence callback error: {e}")
                    
        return events
        
    def get_geofences(self) -> List[Geofence]:
        """Get all geofences."""
        return list(self._geofences.values())


class AdaptiveAccuracyManager:
    """Manage adaptive accuracy based on context."""
    
    def __init__(self):
        self._motion_detector = MotionDetector()
        self._current_config = MobileLocationConfig()
        self._battery_threshold = 20.0
        
    def update_context(self, imu: Optional[IMUData] = None,
                      battery_level: float = 100.0) -> MobileLocationConfig:
        """Update context and return optimal config."""
        # Detect motion state
        if imu:
            motion = self._motion_detector.add_imu_data(imu)
        else:
            motion = self._motion_detector.get_current_state()
            
        # Adjust config based on motion and battery
        if battery_level < self._battery_threshold:
            # Low battery mode
            self._current_config = MobileLocationConfig(
                power_mode=PowerMode.LOW_POWER,
                update_interval_ms=30000,
                fastest_interval_ms=10000,
                use_sensor_fusion=False
            )
        elif motion == MotionState.STATIONARY:
            # Stationary - reduce updates
            self._current_config = MobileLocationConfig(
                power_mode=PowerMode.BALANCED,
                update_interval_ms=10000,
                fastest_interval_ms=5000,
                min_displacement_m=5.0
            )
        elif motion in [MotionState.WALKING, MotionState.RUNNING]:
            # Active movement - higher accuracy
            self._current_config = MobileLocationConfig(
                power_mode=PowerMode.HIGH_ACCURACY,
                update_interval_ms=1000,
                fastest_interval_ms=500,
                use_sensor_fusion=True
            )
        elif motion == MotionState.DRIVING:
            # Driving - balanced with displacement filter
            self._current_config = MobileLocationConfig(
                power_mode=PowerMode.BALANCED,
                update_interval_ms=2000,
                fastest_interval_ms=1000,
                min_displacement_m=10.0
            )
        else:
            # Default balanced
            self._current_config = MobileLocationConfig(
                power_mode=PowerMode.BALANCED,
                update_interval_ms=5000,
                fastest_interval_ms=2000
            )
            
        return self._current_config
        
    def get_current_config(self) -> MobileLocationConfig:
        """Get current config."""
        return self._current_config


class MobileGNSSService:
    """Main mobile GNSS service."""
    
    def __init__(self):
        self._config = MobileLocationConfig()
        self._sensor_fusion = SensorFusion()
        self._agps_manager = AGPSManager()
        self._correction_cache = OfflineCorrectionCache()
        self._geofence_manager = GeofenceManager()
        self._adaptive_manager = AdaptiveAccuracyManager()
        self._motion_detector = MotionDetector()
        
        self._current_location: Optional[MobileLocation] = None
        self._location_history: List[MobileLocation] = []
        self._max_history = 500
        self._location_callbacks: List[Callable[[MobileLocation], None]] = []
        
        self._tracking = False
        self._background_tracking = False
        
    def configure(self, config: MobileLocationConfig) -> None:
        """Configure location service."""
        self._config = config
        
    def start_tracking(self, background: bool = False) -> bool:
        """Start location tracking."""
        self._tracking = True
        self._background_tracking = background
        
        # Download A-GPS data if available
        if self._current_location:
            self._agps_manager.download_assistance_data(
                self._current_location.latitude,
                self._current_location.longitude
            )
            
        logger.info(f"Started tracking (background={background})")
        return True
        
    def stop_tracking(self) -> None:
        """Stop location tracking."""
        self._tracking = False
        self._background_tracking = False
        self._sensor_fusion.reset()
        logger.info("Stopped tracking")
        
    def update_location(self, location: MobileLocation) -> None:
        """Update with new location."""
        # Apply offline corrections if available
        if self._correction_cache.is_valid():
            iono = self._correction_cache.get_ionospheric_correction(
                location.latitude, location.longitude
            )
            tropo = self._correction_cache.get_tropospheric_correction(
                location.latitude, location.longitude
            )
            # Corrections would be applied to raw measurements
            
        # Update sensor fusion
        self._sensor_fusion.update_gnss(location)
        
        # Check geofences
        self._geofence_manager.check_location(location)
        
        # Store location
        self._current_location = location
        self._location_history.append(location)
        if len(self._location_history) > self._max_history:
            self._location_history.pop(0)
            
        # Notify callbacks
        for callback in self._location_callbacks:
            try:
                callback(location)
            except Exception as e:
                logger.error(f"Location callback error: {e}")
                
    def update_imu(self, imu: IMUData) -> Optional[MobileLocation]:
        """Update with IMU data for sensor fusion."""
        if not self._config.use_sensor_fusion:
            return None
            
        # Update motion detector
        self._motion_detector.add_imu_data(imu)
        
        # Get fused location
        fused = self._sensor_fusion.update_imu(imu)
        
        if fused:
            # Check geofences with fused location
            self._geofence_manager.check_location(fused)
            
        return fused
        
    def update_context(self, battery_level: float = 100.0) -> MobileLocationConfig:
        """Update context for adaptive accuracy."""
        return self._adaptive_manager.update_context(battery_level=battery_level)
        
    def add_geofence(self, geofence: Geofence) -> bool:
        """Add geofence."""
        return self._geofence_manager.add_geofence(geofence)
        
    def remove_geofence(self, geofence_id: str) -> bool:
        """Remove geofence."""
        return self._geofence_manager.remove_geofence(geofence_id)
        
    def register_location_callback(self, callback: Callable[[MobileLocation], None]) -> None:
        """Register location update callback."""
        self._location_callbacks.append(callback)
        
    def register_geofence_callback(self, callback: Callable[[GeofenceEvent], None]) -> None:
        """Register geofence event callback."""
        self._geofence_manager.register_callback(callback)
        
    def get_current_location(self) -> Optional[MobileLocation]:
        """Get current location."""
        return self._current_location
        
    def get_location_history(self, count: int = 100) -> List[MobileLocation]:
        """Get location history."""
        return self._location_history[-count:]
        
    def get_motion_state(self) -> MotionState:
        """Get current motion state."""
        return self._motion_detector.get_current_state()
        
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'tracking': self._tracking,
            'background_tracking': self._background_tracking,
            'config': self._config.to_dict(),
            'agps_status': self._agps_manager.get_cache_status(),
            'correction_cache_valid': self._correction_cache.is_valid(),
            'geofence_count': len(self._geofence_manager.get_geofences()),
            'motion_state': self._motion_detector.get_current_state().value,
            'location_count': len(self._location_history)
        }


def create_mobile_gnss_service() -> MobileGNSSService:
    """Factory function to create mobile GNSS service."""
    return MobileGNSSService()


def create_geofence(name: str, lat: float, lon: float, radius_m: float,
                   transitions: List[str] = None) -> Geofence:
    """Factory function to create geofence."""
    trans = []
    if transitions:
        for t in transitions:
            if t == 'enter':
                trans.append(GeofenceTransition.ENTER)
            elif t == 'exit':
                trans.append(GeofenceTransition.EXIT)
            elif t == 'dwell':
                trans.append(GeofenceTransition.DWELL)
    else:
        trans = [GeofenceTransition.ENTER, GeofenceTransition.EXIT]
        
    return Geofence(
        geofence_id=str(uuid.uuid4()),
        name=name,
        latitude=lat,
        longitude=lon,
        radius_m=radius_m,
        transitions=trans
    )
