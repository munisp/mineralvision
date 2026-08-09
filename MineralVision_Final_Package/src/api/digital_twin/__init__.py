"""
MineralVision Digital Twin Module.

This module provides comprehensive digital twin capabilities including:
- Core entity modeling and simulation framework
- Advanced physics-based simulations
- 3D visualization engine
- Real-time data streaming
- Temporal state evolution modeling
"""

from .core import (
    DigitalTwinEntity,
    SpatialEntity,
    MineralDeposit,
    ExplorationArea,
    Equipment,
    DigitalTwinSimulation,
    ExtractionSimulation,
    EnvironmentalImpactSimulation,
    DigitalTwinManager,
)

from .advanced_simulation import (
    SimulationState,
    PhysicsModel,
    SimulationConfig,
    SimulationResult,
    AbstractSimulation,
    PhysicsEngine,
    RigidBody,
    TerrainModel,
    Constraint,
    DistanceConstraint,
    EquipmentSimulation,
    GeologicalSimulation,
    TemporalStateModel,
    create_equipment_simulation,
    create_geological_simulation,
)

from .plant_simulation import (
    TelemetryPoint,
    TelemetryStream,
    AnomalyEvent,
    Zone,
    Equipment as PlantEquipment,
    PlantTwin,
    create_plant_twin,
)

from .visualization_3d import (
    RenderMode,
    CameraMode,
    Color,
    Transform,
    Camera,
    Mesh,
    TerrainMesh,
    SceneObject,
    Scene,
    Visualization3DEngine,
    DigitalTwinVisualizer,
    create_visualization_engine,
    create_digital_twin_visualizer,
)

from .realtime_streaming import (
    StreamType,
    StreamPriority,
    StreamMessage,
    StreamMetrics,
    StreamBuffer,
    DataStream,
    StreamHub,
    DigitalTwinStreamManager,
    AsyncDigitalTwinStreamer,
    SimulationStreamBridge,
    create_stream_manager,
    create_async_streamer,
    create_simulation_bridge,
)

__all__ = [
    "DigitalTwinEntity",
    "SpatialEntity",
    "MineralDeposit",
    "ExplorationArea",
    "Equipment",
    "DigitalTwinSimulation",
    "ExtractionSimulation",
    "EnvironmentalImpactSimulation",
    "DigitalTwinManager",
    "SimulationState",
    "PhysicsModel",
    "SimulationConfig",
    "SimulationResult",
    "AbstractSimulation",
    "PhysicsEngine",
    "RigidBody",
    "TerrainModel",
    "Constraint",
    "DistanceConstraint",
    "EquipmentSimulation",
    "GeologicalSimulation",
    "TemporalStateModel",
    "create_equipment_simulation",
    "create_geological_simulation",
    "TelemetryPoint",
    "TelemetryStream",
    "AnomalyEvent",
    "Zone",
    "PlantEquipment",
    "PlantTwin",
    "create_plant_twin",
    "RenderMode",
    "CameraMode",
    "Color",
    "Transform",
    "Camera",
    "Mesh",
    "TerrainMesh",
    "SceneObject",
    "Scene",
    "Visualization3DEngine",
    "DigitalTwinVisualizer",
    "create_visualization_engine",
    "create_digital_twin_visualizer",
    "StreamType",
    "StreamPriority",
    "StreamMessage",
    "StreamMetrics",
    "StreamBuffer",
    "DataStream",
    "StreamHub",
    "DigitalTwinStreamManager",
    "AsyncDigitalTwinStreamer",
    "SimulationStreamBridge",
    "create_stream_manager",
    "create_async_streamer",
    "create_simulation_bridge",
]
