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
    DigitalTwinManager
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
    create_geological_simulation
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
    create_digital_twin_visualizer
)

from .realtime_streaming import (
    StreamType,
    StreamPriority,
    StreamMessage,
    StreamConfig as DTStreamConfig,
    StreamMetrics,
    StreamBuffer,
    DataStream,
    StreamHub,
    DigitalTwinStreamManager,
    AsyncDigitalTwinStreamer,
    SimulationStreamBridge,
    create_stream_manager,
    create_async_streamer,
    create_simulation_bridge
)

__all__ = [
    # Core
    'DigitalTwinEntity',
    'SpatialEntity',
    'MineralDeposit',
    'ExplorationArea',
    'Equipment',
    'DigitalTwinSimulation',
    'ExtractionSimulation',
    'EnvironmentalImpactSimulation',
    'DigitalTwinManager',
    
    # Advanced Simulation
    'SimulationState',
    'PhysicsModel',
    'SimulationConfig',
    'SimulationResult',
    'AbstractSimulation',
    'PhysicsEngine',
    'RigidBody',
    'TerrainModel',
    'Constraint',
    'DistanceConstraint',
    'EquipmentSimulation',
    'GeologicalSimulation',
    'TemporalStateModel',
    'create_equipment_simulation',
    'create_geological_simulation',
    
    # 3D Visualization
    'RenderMode',
    'CameraMode',
    'Color',
    'Transform',
    'Camera',
    'Mesh',
    'TerrainMesh',
    'SceneObject',
    'Scene',
    'Visualization3DEngine',
    'DigitalTwinVisualizer',
    'create_visualization_engine',
    'create_digital_twin_visualizer',
    
    # Real-time Streaming
    'StreamType',
    'StreamPriority',
    'StreamMessage',
    'DTStreamConfig',
    'StreamMetrics',
    'StreamBuffer',
    'DataStream',
    'StreamHub',
    'DigitalTwinStreamManager',
    'AsyncDigitalTwinStreamer',
    'SimulationStreamBridge',
    'create_stream_manager',
    'create_async_streamer',
    'create_simulation_bridge'
]
