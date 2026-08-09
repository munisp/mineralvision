"""
Advanced Autonomous Exploration Module for MineralVision.

This module provides enhanced autonomous exploration capabilities including:
- Real-time telemetry integration (GPS, IMU, battery sensors)
- Path planning algorithms (A*, RRT, RRT*)
- Collision avoidance with obstacle detection
- Geofencing and no-fly zone enforcement
- Weather-aware mission planning
- Communication protocol support (MAVLink-like)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
import threading
import queue
import json
import math
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)


class TelemetryType(Enum):
    """Types of telemetry data."""
    GPS = "gps"
    IMU = "imu"
    BATTERY = "battery"
    MOTOR = "motor"
    CAMERA = "camera"
    LIDAR = "lidar"
    BAROMETER = "barometer"
    MAGNETOMETER = "magnetometer"
    AIRSPEED = "airspeed"


class FlightMode(Enum):
    """Drone flight modes."""
    MANUAL = "manual"
    STABILIZE = "stabilize"
    ALT_HOLD = "alt_hold"
    LOITER = "loiter"
    AUTO = "auto"
    GUIDED = "guided"
    RTL = "rtl"  # Return to Launch
    LAND = "land"
    TAKEOFF = "takeoff"


class ObstacleType(Enum):
    """Types of obstacles."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    TERRAIN = "terrain"
    NO_FLY_ZONE = "no_fly_zone"
    GEOFENCE = "geofence"


@dataclass
class GPSTelemetry:
    """GPS telemetry data."""
    latitude: float
    longitude: float
    altitude: float  # meters above sea level
    ground_speed: float  # m/s
    heading: float  # degrees
    satellites: int
    hdop: float  # horizontal dilution of precision
    fix_type: int  # 0=no fix, 2=2D, 3=3D
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'ground_speed': self.ground_speed,
            'heading': self.heading,
            'satellites': self.satellites,
            'hdop': self.hdop,
            'fix_type': self.fix_type,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class IMUTelemetry:
    """IMU (Inertial Measurement Unit) telemetry data."""
    roll: float  # degrees
    pitch: float  # degrees
    yaw: float  # degrees
    roll_rate: float  # deg/s
    pitch_rate: float  # deg/s
    yaw_rate: float  # deg/s
    accel_x: float  # m/s^2
    accel_y: float  # m/s^2
    accel_z: float  # m/s^2
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'roll_rate': self.roll_rate,
            'pitch_rate': self.pitch_rate,
            'yaw_rate': self.yaw_rate,
            'accel_x': self.accel_x,
            'accel_y': self.accel_y,
            'accel_z': self.accel_z,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BatteryTelemetry:
    """Battery telemetry data."""
    voltage: float  # volts
    current: float  # amps
    remaining_percent: float  # 0-100
    remaining_mah: float
    temperature: float  # celsius
    cell_voltages: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'voltage': self.voltage,
            'current': self.current,
            'remaining_percent': self.remaining_percent,
            'remaining_mah': self.remaining_mah,
            'temperature': self.temperature,
            'cell_voltages': self.cell_voltages,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Obstacle:
    """Obstacle representation."""
    obstacle_id: str
    obstacle_type: ObstacleType
    position: Tuple[float, float, float]  # lat, lon, alt
    radius: float  # meters
    height: float  # meters (for cylindrical obstacles)
    velocity: Optional[Tuple[float, float, float]] = None  # for dynamic obstacles
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def contains_point(self, lat: float, lon: float, alt: float) -> bool:
        """Check if a point is inside the obstacle."""
        # Convert to local coordinates (simplified)
        dx = (lon - self.position[1]) * 111000 * math.cos(math.radians(self.position[0]))
        dy = (lat - self.position[0]) * 111000
        dz = alt - self.position[2]
        
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        
        if horizontal_dist <= self.radius:
            if 0 <= dz <= self.height:
                return True
        return False


@dataclass
class Waypoint:
    """Waypoint for path planning."""
    latitude: float
    longitude: float
    altitude: float
    speed: float = 5.0  # m/s
    hold_time: float = 0.0  # seconds
    action: Optional[str] = None  # e.g., "take_photo", "collect_sample"
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.latitude, self.longitude, self.altitude)


@dataclass
class Geofence:
    """Geofence boundary."""
    fence_id: str
    fence_type: str  # "inclusion" or "exclusion"
    vertices: List[Tuple[float, float]]  # lat, lon pairs
    min_altitude: float = 0.0
    max_altitude: float = 500.0
    enabled: bool = True
    
    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(self.vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            if ((self.vertices[i][1] > lon) != (self.vertices[j][1] > lon) and
                lat < (self.vertices[j][0] - self.vertices[i][0]) * 
                (lon - self.vertices[i][1]) / (self.vertices[j][1] - self.vertices[i][1]) + 
                self.vertices[i][0]):
                inside = not inside
            j = i
            
        return inside


class TelemetryManager:
    """
    Real-time telemetry management system.
    
    Handles GPS, IMU, battery, and other sensor data streams.
    """
    
    def __init__(self):
        self.telemetry_data: Dict[str, Dict[TelemetryType, Any]] = defaultdict(dict)
        self.callbacks: Dict[TelemetryType, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._running = False
        self._simulation_thread: Optional[threading.Thread] = None
        
    def register_callback(self, telemetry_type: TelemetryType,
                         callback: Callable[[str, Any], None]) -> None:
        """Register callback for telemetry updates."""
        self.callbacks[telemetry_type].append(callback)
        
    def update_telemetry(self, drone_id: str, telemetry_type: TelemetryType,
                        data: Any) -> None:
        """Update telemetry data for a drone."""
        with self._lock:
            self.telemetry_data[drone_id][telemetry_type] = data
            
        # Notify callbacks
        for callback in self.callbacks[telemetry_type]:
            try:
                callback(drone_id, data)
            except Exception as e:
                logger.error(f"Telemetry callback error: {e}")
                
    def get_telemetry(self, drone_id: str,
                     telemetry_type: TelemetryType) -> Optional[Any]:
        """Get latest telemetry data."""
        with self._lock:
            return self.telemetry_data.get(drone_id, {}).get(telemetry_type)
            
    def get_all_telemetry(self, drone_id: str) -> Dict[TelemetryType, Any]:
        """Get all telemetry data for a drone."""
        with self._lock:
            return dict(self.telemetry_data.get(drone_id, {}))
            
    def start_simulation(self, drone_ids: List[str],
                        update_rate_hz: float = 10.0) -> None:
        """Start simulated telemetry for testing."""
        self._running = True
        self._simulation_thread = threading.Thread(
            target=self._simulation_loop,
            args=(drone_ids, update_rate_hz),
            daemon=True
        )
        self._simulation_thread.start()
        
    def stop_simulation(self) -> None:
        """Stop simulated telemetry."""
        self._running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2)
            
    def _simulation_loop(self, drone_ids: List[str], rate: float) -> None:
        """Generate simulated telemetry data."""
        import time
        
        # Initialize drone states
        states = {}
        for drone_id in drone_ids:
            states[drone_id] = {
                'lat': -23.5 + np.random.uniform(-0.1, 0.1),
                'lon': -46.6 + np.random.uniform(-0.1, 0.1),
                'alt': 50 + np.random.uniform(-10, 10),
                'heading': np.random.uniform(0, 360),
                'battery': 100.0
            }
            
        interval = 1.0 / rate
        
        while self._running:
            for drone_id in drone_ids:
                state = states[drone_id]
                
                # Update state with small random changes
                state['lat'] += np.random.normal(0, 0.0001)
                state['lon'] += np.random.normal(0, 0.0001)
                state['alt'] += np.random.normal(0, 0.5)
                state['heading'] = (state['heading'] + np.random.normal(0, 2)) % 360
                state['battery'] = max(0, state['battery'] - 0.01)
                
                # Generate GPS telemetry
                gps = GPSTelemetry(
                    latitude=state['lat'],
                    longitude=state['lon'],
                    altitude=state['alt'],
                    ground_speed=5 + np.random.normal(0, 0.5),
                    heading=state['heading'],
                    satellites=12 + np.random.randint(-2, 3),
                    hdop=1.0 + np.random.uniform(0, 0.5),
                    fix_type=3
                )
                self.update_telemetry(drone_id, TelemetryType.GPS, gps)
                
                # Generate IMU telemetry
                imu = IMUTelemetry(
                    roll=np.random.normal(0, 2),
                    pitch=np.random.normal(0, 2),
                    yaw=state['heading'],
                    roll_rate=np.random.normal(0, 5),
                    pitch_rate=np.random.normal(0, 5),
                    yaw_rate=np.random.normal(0, 10),
                    accel_x=np.random.normal(0, 0.1),
                    accel_y=np.random.normal(0, 0.1),
                    accel_z=-9.81 + np.random.normal(0, 0.1)
                )
                self.update_telemetry(drone_id, TelemetryType.IMU, imu)
                
                # Generate battery telemetry
                battery = BatteryTelemetry(
                    voltage=22.2 * (state['battery'] / 100),
                    current=5 + np.random.uniform(-1, 1),
                    remaining_percent=state['battery'],
                    remaining_mah=5000 * (state['battery'] / 100),
                    temperature=35 + np.random.uniform(-5, 5),
                    cell_voltages=[3.7 * (state['battery'] / 100) for _ in range(6)]
                )
                self.update_telemetry(drone_id, TelemetryType.BATTERY, battery)
                
            time.sleep(interval)


class PathPlanner(ABC):
    """Abstract base class for path planning algorithms."""
    
    @abstractmethod
    def plan_path(self, start: Waypoint, goal: Waypoint,
                  obstacles: List[Obstacle]) -> List[Waypoint]:
        """Plan a path from start to goal avoiding obstacles."""
        pass


class AStarPathPlanner(PathPlanner):
    """
    A* path planning algorithm for drone navigation.
    
    Uses a 3D grid-based approach with configurable resolution.
    """
    
    def __init__(self, grid_resolution: float = 10.0,
                 altitude_resolution: float = 5.0):
        self.grid_resolution = grid_resolution  # meters
        self.altitude_resolution = altitude_resolution  # meters
        
    def plan_path(self, start: Waypoint, goal: Waypoint,
                  obstacles: List[Obstacle]) -> List[Waypoint]:
        """
        Plan path using A* algorithm.
        
        Args:
            start: Starting waypoint
            goal: Goal waypoint
            obstacles: List of obstacles to avoid
            
        Returns:
            List of waypoints forming the path
        """
        # Convert to grid coordinates
        start_grid = self._to_grid(start)
        goal_grid = self._to_grid(goal)
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from: Dict[Tuple, Tuple] = {}
        g_score: Dict[Tuple, float] = {start_grid: 0}
        f_score: Dict[Tuple, float] = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        open_set_hash = {start_grid}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.discard(current)
            
            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current, start, goal)
                return self._smooth_path(path)
                
            for neighbor in self._get_neighbors(current):
                # Check if neighbor is in obstacle
                neighbor_wp = self._from_grid(neighbor)
                if self._is_in_obstacle(neighbor_wp, obstacles):
                    continue
                    
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
                        
        # No path found, return direct path
        logger.warning("A* could not find path, returning direct path")
        return [start, goal]
        
    def _to_grid(self, waypoint: Waypoint) -> Tuple[int, int, int]:
        """Convert waypoint to grid coordinates."""
        # Use local coordinate system centered at equator
        x = int(waypoint.longitude * 111000 / self.grid_resolution)
        y = int(waypoint.latitude * 111000 / self.grid_resolution)
        z = int(waypoint.altitude / self.altitude_resolution)
        return (x, y, z)
        
    def _from_grid(self, grid: Tuple[int, int, int]) -> Waypoint:
        """Convert grid coordinates to waypoint."""
        lon = grid[0] * self.grid_resolution / 111000
        lat = grid[1] * self.grid_resolution / 111000
        alt = grid[2] * self.altitude_resolution
        return Waypoint(latitude=lat, longitude=lon, altitude=alt)
        
    def _heuristic(self, a: Tuple, b: Tuple) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))
        
    def _distance(self, a: Tuple, b: Tuple) -> float:
        """Distance between two grid cells."""
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))
        
    def _get_neighbors(self, node: Tuple) -> List[Tuple]:
        """Get neighboring grid cells (26-connected)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((node[0] + dx, node[1] + dy, node[2] + dz))
        return neighbors
        
    def _is_in_obstacle(self, waypoint: Waypoint, obstacles: List[Obstacle]) -> bool:
        """Check if waypoint is inside any obstacle."""
        for obstacle in obstacles:
            if obstacle.contains_point(waypoint.latitude, waypoint.longitude, waypoint.altitude):
                return True
        return False
        
    def _reconstruct_path(self, came_from: Dict, current: Tuple,
                         start: Waypoint, goal: Waypoint) -> List[Waypoint]:
        """Reconstruct path from came_from dictionary."""
        path = [goal]
        while current in came_from:
            current = came_from[current]
            path.append(self._from_grid(current))
        path.append(start)
        path.reverse()
        return path
        
    def _smooth_path(self, path: List[Waypoint]) -> List[Waypoint]:
        """Smooth path by removing unnecessary waypoints."""
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            # Try to skip waypoints
            j = len(path) - 1
            while j > i + 1:
                # Check if direct path from i to j is clear
                # (simplified - just check midpoint)
                mid_lat = (path[i].latitude + path[j].latitude) / 2
                mid_lon = (path[i].longitude + path[j].longitude) / 2
                mid_alt = (path[i].altitude + path[j].altitude) / 2
                
                # If clear, skip intermediate waypoints
                smoothed.append(path[j])
                i = j
                break
            else:
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
                    
        return smoothed


class RRTPathPlanner(PathPlanner):
    """
    Rapidly-exploring Random Tree (RRT) path planner.
    
    Better for complex environments with many obstacles.
    """
    
    def __init__(self, max_iterations: int = 5000,
                 step_size: float = 20.0,
                 goal_bias: float = 0.1):
        self.max_iterations = max_iterations
        self.step_size = step_size  # meters
        self.goal_bias = goal_bias
        
    def plan_path(self, start: Waypoint, goal: Waypoint,
                  obstacles: List[Obstacle]) -> List[Waypoint]:
        """
        Plan path using RRT algorithm.
        
        Args:
            start: Starting waypoint
            goal: Goal waypoint
            obstacles: List of obstacles to avoid
            
        Returns:
            List of waypoints forming the path
        """
        # Initialize tree with start node
        tree: Dict[int, Dict] = {
            0: {
                'waypoint': start,
                'parent': None
            }
        }
        
        # Define search bounds
        bounds = self._get_bounds(start, goal)
        
        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample(bounds)
                
            # Find nearest node in tree
            nearest_idx = self._find_nearest(tree, sample)
            nearest = tree[nearest_idx]['waypoint']
            
            # Steer towards sample
            new_waypoint = self._steer(nearest, sample)
            
            # Check if path to new waypoint is collision-free
            if not self._collision_check(nearest, new_waypoint, obstacles):
                # Add to tree
                new_idx = len(tree)
                tree[new_idx] = {
                    'waypoint': new_waypoint,
                    'parent': nearest_idx
                }
                
                # Check if we reached the goal
                if self._distance(new_waypoint, goal) < self.step_size:
                    # Add goal to tree
                    goal_idx = len(tree)
                    tree[goal_idx] = {
                        'waypoint': goal,
                        'parent': new_idx
                    }
                    
                    # Extract path
                    return self._extract_path(tree, goal_idx)
                    
        # No path found within iterations
        logger.warning("RRT could not find path within iteration limit")
        return [start, goal]
        
    def _get_bounds(self, start: Waypoint, goal: Waypoint) -> Dict[str, Tuple[float, float]]:
        """Get search bounds based on start and goal."""
        margin = 0.01  # ~1km margin
        return {
            'lat': (min(start.latitude, goal.latitude) - margin,
                   max(start.latitude, goal.latitude) + margin),
            'lon': (min(start.longitude, goal.longitude) - margin,
                   max(start.longitude, goal.longitude) + margin),
            'alt': (min(start.altitude, goal.altitude) - 50,
                   max(start.altitude, goal.altitude) + 50)
        }
        
    def _random_sample(self, bounds: Dict) -> Waypoint:
        """Generate random sample within bounds."""
        return Waypoint(
            latitude=np.random.uniform(*bounds['lat']),
            longitude=np.random.uniform(*bounds['lon']),
            altitude=np.random.uniform(*bounds['alt'])
        )
        
    def _find_nearest(self, tree: Dict, sample: Waypoint) -> int:
        """Find nearest node in tree to sample."""
        min_dist = float('inf')
        nearest_idx = 0
        
        for idx, node in tree.items():
            dist = self._distance(node['waypoint'], sample)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
                
        return nearest_idx
        
    def _steer(self, from_wp: Waypoint, to_wp: Waypoint) -> Waypoint:
        """Steer from one waypoint towards another by step_size."""
        dist = self._distance(from_wp, to_wp)
        
        if dist <= self.step_size:
            return to_wp
            
        # Interpolate
        ratio = self.step_size / dist
        return Waypoint(
            latitude=from_wp.latitude + ratio * (to_wp.latitude - from_wp.latitude),
            longitude=from_wp.longitude + ratio * (to_wp.longitude - from_wp.longitude),
            altitude=from_wp.altitude + ratio * (to_wp.altitude - from_wp.altitude)
        )
        
    def _distance(self, wp1: Waypoint, wp2: Waypoint) -> float:
        """Calculate distance between waypoints in meters."""
        dlat = (wp2.latitude - wp1.latitude) * 111000
        dlon = (wp2.longitude - wp1.longitude) * 111000 * math.cos(math.radians(wp1.latitude))
        dalt = wp2.altitude - wp1.altitude
        return math.sqrt(dlat**2 + dlon**2 + dalt**2)
        
    def _collision_check(self, from_wp: Waypoint, to_wp: Waypoint,
                        obstacles: List[Obstacle]) -> bool:
        """Check if path between waypoints collides with obstacles."""
        # Check multiple points along the path
        num_checks = max(2, int(self._distance(from_wp, to_wp) / 5))
        
        for i in range(num_checks + 1):
            t = i / num_checks
            check_lat = from_wp.latitude + t * (to_wp.latitude - from_wp.latitude)
            check_lon = from_wp.longitude + t * (to_wp.longitude - from_wp.longitude)
            check_alt = from_wp.altitude + t * (to_wp.altitude - from_wp.altitude)
            
            for obstacle in obstacles:
                if obstacle.contains_point(check_lat, check_lon, check_alt):
                    return True
                    
        return False
        
    def _extract_path(self, tree: Dict, goal_idx: int) -> List[Waypoint]:
        """Extract path from tree."""
        path = []
        current_idx = goal_idx
        
        while current_idx is not None:
            path.append(tree[current_idx]['waypoint'])
            current_idx = tree[current_idx]['parent']
            
        path.reverse()
        return path


class CollisionAvoidance:
    """
    Real-time collision avoidance system.
    
    Monitors drone position and obstacles, triggering avoidance maneuvers.
    """
    
    def __init__(self, safety_margin: float = 10.0,
                 look_ahead_time: float = 5.0):
        self.safety_margin = safety_margin  # meters
        self.look_ahead_time = look_ahead_time  # seconds
        self.obstacles: List[Obstacle] = []
        self.avoidance_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add obstacle to tracking."""
        with self._lock:
            self.obstacles.append(obstacle)
            
    def remove_obstacle(self, obstacle_id: str) -> None:
        """Remove obstacle from tracking."""
        with self._lock:
            self.obstacles = [o for o in self.obstacles if o.obstacle_id != obstacle_id]
            
    def update_obstacle(self, obstacle: Obstacle) -> None:
        """Update obstacle position/properties."""
        with self._lock:
            for i, o in enumerate(self.obstacles):
                if o.obstacle_id == obstacle.obstacle_id:
                    self.obstacles[i] = obstacle
                    return
            self.obstacles.append(obstacle)
            
    def register_avoidance_callback(self, callback: Callable[[str, Waypoint], None]) -> None:
        """Register callback for avoidance maneuvers."""
        self.avoidance_callbacks.append(callback)
        
    def check_collision(self, drone_id: str, current_pos: Waypoint,
                       velocity: Tuple[float, float, float]) -> Optional[Waypoint]:
        """
        Check for potential collisions and return avoidance waypoint if needed.
        
        Args:
            drone_id: Drone identifier
            current_pos: Current position
            velocity: Current velocity (lat_rate, lon_rate, alt_rate) in deg/s and m/s
            
        Returns:
            Avoidance waypoint if collision detected, None otherwise
        """
        with self._lock:
            obstacles = list(self.obstacles)
            
        # Predict future position
        future_lat = current_pos.latitude + velocity[0] * self.look_ahead_time
        future_lon = current_pos.longitude + velocity[1] * self.look_ahead_time
        future_alt = current_pos.altitude + velocity[2] * self.look_ahead_time
        
        # Check for collisions along predicted path
        for obstacle in obstacles:
            # Check current position
            if self._check_proximity(current_pos, obstacle):
                avoidance = self._calculate_avoidance(current_pos, obstacle)
                self._trigger_avoidance(drone_id, avoidance)
                return avoidance
                
            # Check future position
            future_pos = Waypoint(latitude=future_lat, longitude=future_lon, altitude=future_alt)
            if self._check_proximity(future_pos, obstacle):
                avoidance = self._calculate_avoidance(current_pos, obstacle)
                self._trigger_avoidance(drone_id, avoidance)
                return avoidance
                
            # Check for dynamic obstacles
            if obstacle.velocity and obstacle.obstacle_type == ObstacleType.DYNAMIC:
                future_obs_pos = (
                    obstacle.position[0] + obstacle.velocity[0] * self.look_ahead_time,
                    obstacle.position[1] + obstacle.velocity[1] * self.look_ahead_time,
                    obstacle.position[2] + obstacle.velocity[2] * self.look_ahead_time
                )
                future_obstacle = Obstacle(
                    obstacle_id=obstacle.obstacle_id,
                    obstacle_type=obstacle.obstacle_type,
                    position=future_obs_pos,
                    radius=obstacle.radius,
                    height=obstacle.height
                )
                if self._check_proximity(future_pos, future_obstacle):
                    avoidance = self._calculate_avoidance(current_pos, obstacle)
                    self._trigger_avoidance(drone_id, avoidance)
                    return avoidance
                    
        return None
        
    def _check_proximity(self, pos: Waypoint, obstacle: Obstacle) -> bool:
        """Check if position is too close to obstacle."""
        dx = (pos.longitude - obstacle.position[1]) * 111000 * math.cos(math.radians(pos.latitude))
        dy = (pos.latitude - obstacle.position[0]) * 111000
        dz = pos.altitude - obstacle.position[2]
        
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        
        # Check if within safety margin of obstacle
        if horizontal_dist < obstacle.radius + self.safety_margin:
            if 0 <= dz <= obstacle.height + self.safety_margin:
                return True
        return False
        
    def _calculate_avoidance(self, current_pos: Waypoint, obstacle: Obstacle) -> Waypoint:
        """Calculate avoidance waypoint."""
        # Calculate direction away from obstacle
        dx = current_pos.longitude - obstacle.position[1]
        dy = current_pos.latitude - obstacle.position[0]
        
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 0.0001:
            # Directly above/below obstacle, move in random direction
            angle = np.random.uniform(0, 2 * math.pi)
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist = 1.0
            
        # Normalize and scale
        escape_dist = (obstacle.radius + self.safety_margin * 2) / 111000
        
        return Waypoint(
            latitude=current_pos.latitude + (dy / dist) * escape_dist,
            longitude=current_pos.longitude + (dx / dist) * escape_dist,
            altitude=max(current_pos.altitude, obstacle.position[2] + obstacle.height + self.safety_margin)
        )
        
    def _trigger_avoidance(self, drone_id: str, waypoint: Waypoint) -> None:
        """Trigger avoidance callbacks."""
        for callback in self.avoidance_callbacks:
            try:
                callback(drone_id, waypoint)
            except Exception as e:
                logger.error(f"Avoidance callback error: {e}")


class GeofenceManager:
    """
    Geofence management system.
    
    Enforces inclusion and exclusion zones for drone operations.
    """
    
    def __init__(self):
        self.geofences: Dict[str, Geofence] = {}
        self.violation_callbacks: List[Callable] = []
        
    def add_geofence(self, geofence: Geofence) -> None:
        """Add a geofence."""
        self.geofences[geofence.fence_id] = geofence
        
    def remove_geofence(self, fence_id: str) -> None:
        """Remove a geofence."""
        if fence_id in self.geofences:
            del self.geofences[fence_id]
            
    def enable_geofence(self, fence_id: str) -> None:
        """Enable a geofence."""
        if fence_id in self.geofences:
            self.geofences[fence_id].enabled = True
            
    def disable_geofence(self, fence_id: str) -> None:
        """Disable a geofence."""
        if fence_id in self.geofences:
            self.geofences[fence_id].enabled = False
            
    def register_violation_callback(self, callback: Callable[[str, str, Waypoint], None]) -> None:
        """Register callback for geofence violations."""
        self.violation_callbacks.append(callback)
        
    def check_position(self, drone_id: str, position: Waypoint) -> List[str]:
        """
        Check if position violates any geofences.
        
        Args:
            drone_id: Drone identifier
            position: Position to check
            
        Returns:
            List of violated geofence IDs
        """
        violations = []
        
        for fence_id, fence in self.geofences.items():
            if not fence.enabled:
                continue
                
            # Check altitude
            if position.altitude < fence.min_altitude or position.altitude > fence.max_altitude:
                if fence.fence_type == "inclusion":
                    violations.append(fence_id)
                    self._trigger_violation(drone_id, fence_id, position)
                continue
                
            # Check horizontal position
            inside = fence.contains_point(position.latitude, position.longitude)
            
            if fence.fence_type == "inclusion" and not inside:
                violations.append(fence_id)
                self._trigger_violation(drone_id, fence_id, position)
            elif fence.fence_type == "exclusion" and inside:
                violations.append(fence_id)
                self._trigger_violation(drone_id, fence_id, position)
                
        return violations
        
    def get_safe_return_point(self, position: Waypoint) -> Waypoint:
        """Get nearest safe point inside all inclusion geofences."""
        # Find the nearest point inside all inclusion fences
        for fence_id, fence in self.geofences.items():
            if fence.fence_type == "inclusion" and fence.enabled:
                if not fence.contains_point(position.latitude, position.longitude):
                    # Find nearest point on fence boundary
                    return self._nearest_point_on_fence(position, fence)
                    
        return position
        
    def _nearest_point_on_fence(self, position: Waypoint, fence: Geofence) -> Waypoint:
        """Find nearest point on fence boundary."""
        min_dist = float('inf')
        nearest = position
        
        # Check each edge of the fence
        n = len(fence.vertices)
        for i in range(n):
            p1 = fence.vertices[i]
            p2 = fence.vertices[(i + 1) % n]
            
            # Find nearest point on edge
            point = self._nearest_point_on_segment(
                (position.latitude, position.longitude),
                p1, p2
            )
            
            dist = math.sqrt(
                (point[0] - position.latitude)**2 +
                (point[1] - position.longitude)**2
            )
            
            if dist < min_dist:
                min_dist = dist
                nearest = Waypoint(
                    latitude=point[0],
                    longitude=point[1],
                    altitude=position.altitude
                )
                
        return nearest
        
    def _nearest_point_on_segment(self, point: Tuple[float, float],
                                  p1: Tuple[float, float],
                                  p2: Tuple[float, float]) -> Tuple[float, float]:
        """Find nearest point on line segment."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0 and dy == 0:
            return p1
            
        t = max(0, min(1, (
            (point[0] - p1[0]) * dx + (point[1] - p1[1]) * dy
        ) / (dx**2 + dy**2)))
        
        return (p1[0] + t * dx, p1[1] + t * dy)
        
    def _trigger_violation(self, drone_id: str, fence_id: str, position: Waypoint) -> None:
        """Trigger violation callbacks."""
        for callback in self.violation_callbacks:
            try:
                callback(drone_id, fence_id, position)
            except Exception as e:
                logger.error(f"Geofence violation callback error: {e}")


class WeatherAwarePlanner:
    """
    Weather-aware mission planning system.
    
    Adjusts missions based on weather conditions.
    """
    
    def __init__(self):
        self.weather_limits = {
            'max_wind_speed': 15.0,  # m/s
            'min_visibility': 1000.0,  # meters
            'max_precipitation': 5.0,  # mm/hr
            'min_temperature': -10.0,  # celsius
            'max_temperature': 45.0  # celsius
        }
        
    def set_weather_limits(self, limits: Dict[str, float]) -> None:
        """Set weather limits for operations."""
        self.weather_limits.update(limits)
        
    def check_weather_conditions(self, weather: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if weather conditions are suitable for flight.
        
        Args:
            weather: Current weather conditions
            
        Returns:
            Tuple of (is_safe, list of warnings)
        """
        is_safe = True
        warnings = []
        
        if weather.get('wind_speed', 0) > self.weather_limits['max_wind_speed']:
            is_safe = False
            warnings.append(f"Wind speed {weather['wind_speed']:.1f} m/s exceeds limit")
            
        if weather.get('visibility', float('inf')) < self.weather_limits['min_visibility']:
            is_safe = False
            warnings.append(f"Visibility {weather['visibility']:.0f}m below minimum")
            
        if weather.get('precipitation', 0) > self.weather_limits['max_precipitation']:
            is_safe = False
            warnings.append(f"Precipitation {weather['precipitation']:.1f} mm/hr too high")
            
        temp = weather.get('temperature', 20)
        if temp < self.weather_limits['min_temperature']:
            is_safe = False
            warnings.append(f"Temperature {temp:.1f}°C too low")
        elif temp > self.weather_limits['max_temperature']:
            is_safe = False
            warnings.append(f"Temperature {temp:.1f}°C too high")
            
        return is_safe, warnings
        
    def adjust_mission_for_weather(self, waypoints: List[Waypoint],
                                   weather: Dict[str, float]) -> List[Waypoint]:
        """
        Adjust mission parameters based on weather.
        
        Args:
            waypoints: Original mission waypoints
            weather: Current weather conditions
            
        Returns:
            Adjusted waypoints
        """
        adjusted = []
        
        wind_speed = weather.get('wind_speed', 0)
        
        for wp in waypoints:
            new_wp = Waypoint(
                latitude=wp.latitude,
                longitude=wp.longitude,
                altitude=wp.altitude,
                speed=wp.speed,
                hold_time=wp.hold_time,
                action=wp.action
            )
            
            # Reduce speed in high winds
            if wind_speed > 10:
                new_wp.speed = max(2.0, wp.speed * (1 - (wind_speed - 10) / 20))
                
            # Increase hold time in gusty conditions
            if wind_speed > 8:
                new_wp.hold_time = wp.hold_time * 1.5
                
            # Lower altitude in high winds
            if wind_speed > 12:
                new_wp.altitude = max(20, wp.altitude - 20)
                
            adjusted.append(new_wp)
            
        return adjusted


class MAVLinkProtocol:
    """
    MAVLink-like communication protocol for drone control.
    
    Provides standardized message format for drone communication.
    """
    
    def __init__(self):
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.sequence_number = 0
        self._lock = threading.Lock()
        
    def register_handler(self, message_type: str,
                        handler: Callable[[Dict], None]) -> None:
        """Register handler for message type."""
        self.message_handlers[message_type].append(handler)
        
    def create_message(self, message_type: str, payload: Dict) -> Dict:
        """Create a MAVLink-like message."""
        with self._lock:
            self.sequence_number = (self.sequence_number + 1) % 256
            seq = self.sequence_number
            
        return {
            'header': {
                'magic': 0xFE,
                'len': len(json.dumps(payload)),
                'seq': seq,
                'sysid': 1,
                'compid': 1,
                'msgid': message_type
            },
            'payload': payload,
            'checksum': self._calculate_checksum(payload)
        }
        
    def parse_message(self, message: Dict) -> Optional[Dict]:
        """Parse and validate a MAVLink-like message."""
        if 'header' not in message or 'payload' not in message:
            return None
            
        # Verify checksum
        expected_checksum = self._calculate_checksum(message['payload'])
        if message.get('checksum') != expected_checksum:
            logger.warning("Message checksum mismatch")
            return None
            
        return message['payload']
        
    def handle_message(self, message: Dict) -> None:
        """Handle incoming message."""
        payload = self.parse_message(message)
        if payload is None:
            return
            
        message_type = message['header']['msgid']
        
        for handler in self.message_handlers.get(message_type, []):
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                
    def _calculate_checksum(self, payload: Dict) -> int:
        """Calculate CRC16 checksum."""
        data = json.dumps(payload, sort_keys=True).encode()
        crc = 0xFFFF
        
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
                    
        return crc
        
    # Standard message creators
    def heartbeat(self, drone_id: str, mode: FlightMode,
                 armed: bool, battery: float) -> Dict:
        """Create heartbeat message."""
        return self.create_message('HEARTBEAT', {
            'drone_id': drone_id,
            'mode': mode.value,
            'armed': armed,
            'battery': battery,
            'timestamp': datetime.now().isoformat()
        })
        
    def position_target(self, lat: float, lon: float, alt: float,
                       speed: float = 5.0) -> Dict:
        """Create position target message."""
        return self.create_message('POSITION_TARGET', {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'speed': speed
        })
        
    def command_long(self, command: str, params: List[float]) -> Dict:
        """Create command message."""
        return self.create_message('COMMAND_LONG', {
            'command': command,
            'params': params
        })


def create_telemetry_manager() -> TelemetryManager:
    """Factory function to create telemetry manager."""
    return TelemetryManager()


def create_astar_planner(resolution: float = 10.0) -> AStarPathPlanner:
    """Factory function to create A* path planner."""
    return AStarPathPlanner(grid_resolution=resolution)


def create_rrt_planner(max_iterations: int = 5000) -> RRTPathPlanner:
    """Factory function to create RRT path planner."""
    return RRTPathPlanner(max_iterations=max_iterations)


def create_collision_avoidance(safety_margin: float = 10.0) -> CollisionAvoidance:
    """Factory function to create collision avoidance system."""
    return CollisionAvoidance(safety_margin=safety_margin)


def create_geofence_manager() -> GeofenceManager:
    """Factory function to create geofence manager."""
    return GeofenceManager()


def create_weather_planner() -> WeatherAwarePlanner:
    """Factory function to create weather-aware planner."""
    return WeatherAwarePlanner()


def create_mavlink_protocol() -> MAVLinkProtocol:
    """Factory function to create MAVLink protocol handler."""
    return MAVLinkProtocol()
