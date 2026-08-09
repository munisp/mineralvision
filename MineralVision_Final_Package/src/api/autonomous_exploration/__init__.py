"""
Autonomous Exploration Module for MineralVision.

This module provides comprehensive autonomous exploration capabilities including:
- Real-time telemetry integration (GPS, IMU, battery sensors)
- Path planning algorithms (A*, RRT)
- Collision avoidance with obstacle detection
- Geofencing and no-fly zone enforcement
- Weather-aware mission planning
- MAVLink-like communication protocol
"""

from .core import (
    DroneSpecification,
    DroneState,
    SamplingPoint,
    MissionPlan,
    ExplorationArea,
    AutonomousExplorationSystem
)
from .advanced_exploration import (
    TelemetryType,
    FlightMode,
    ObstacleType,
    GPSTelemetry,
    IMUTelemetry,
    BatteryTelemetry,
    Obstacle,
    Waypoint,
    Geofence,
    TelemetryManager,
    PathPlanner,
    AStarPathPlanner,
    RRTPathPlanner,
    CollisionAvoidance,
    GeofenceManager,
    WeatherAwarePlanner,
    MAVLinkProtocol,
    create_telemetry_manager,
    create_astar_planner,
    create_rrt_planner,
    create_collision_avoidance,
    create_geofence_manager,
    create_weather_planner,
    create_mavlink_protocol
)

__all__ = [
    'DroneSpecification',
    'DroneState',
    'SamplingPoint',
    'MissionPlan',
    'ExplorationArea',
    'AutonomousExplorationSystem',
    'TelemetryType',
    'FlightMode',
    'ObstacleType',
    'GPSTelemetry',
    'IMUTelemetry',
    'BatteryTelemetry',
    'Obstacle',
    'Waypoint',
    'Geofence',
    'TelemetryManager',
    'PathPlanner',
    'AStarPathPlanner',
    'RRTPathPlanner',
    'CollisionAvoidance',
    'GeofenceManager',
    'WeatherAwarePlanner',
    'MAVLinkProtocol',
    'create_telemetry_manager',
    'create_astar_planner',
    'create_rrt_planner',
    'create_collision_avoidance',
    'create_geofence_manager',
    'create_weather_planner',
    'create_mavlink_protocol'
]
