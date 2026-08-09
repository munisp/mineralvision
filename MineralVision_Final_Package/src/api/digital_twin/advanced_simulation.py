"""
Advanced Digital Twin Simulation for MineralVision.

This module provides enhanced simulation capabilities including:
- Abstract base simulation with proper interfaces
- Physics-based terrain and equipment simulation
- Temporal state evolution modeling
- Real-time data streaming integration
"""

import numpy as np
import uuid
import datetime
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)


class SimulationState(Enum):
    """States of a simulation."""
    CREATED = "created"
    CONFIGURED = "configured"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class PhysicsModel(Enum):
    """Physics models for simulation."""
    RIGID_BODY = "rigid_body"
    PARTICLE = "particle"
    FLUID = "fluid"
    TERRAIN = "terrain"
    HYBRID = "hybrid"


@dataclass
class SimulationConfig:
    """Configuration for simulations."""
    time_step: float = 0.1  # seconds
    max_iterations: int = 10000
    convergence_threshold: float = 1e-6
    enable_physics: bool = True
    physics_model: PhysicsModel = PhysicsModel.RIGID_BODY
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    enable_collision: bool = True
    enable_visualization: bool = False
    random_seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Result container for simulations."""
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, float]
    timestamps: List[datetime.datetime]
    states: List[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'metrics': self.metrics,
            'timestamps': [t.isoformat() for t in self.timestamps],
            'states': self.states,
            'errors': self.errors
        }


class AbstractSimulation(ABC):
    """
    Abstract base class for all digital twin simulations.
    
    This replaces the placeholder DigitalTwinSimulation.run() method
    with a proper abstract interface.
    """
    
    def __init__(self, name: str, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation.
        
        Args:
            name: Name of the simulation
            config: Simulation configuration
        """
        self.simulation_id = str(uuid.uuid4())
        self.name = name
        self.config = config or SimulationConfig()
        self.state = SimulationState.CREATED
        self.created_at = datetime.datetime.now()
        self.started_at: Optional[datetime.datetime] = None
        self.completed_at: Optional[datetime.datetime] = None
        
        # Parameters and results
        self.parameters: Dict[str, Any] = {}
        self.result: Optional[SimulationResult] = None
        
        # State history for temporal modeling
        self.state_history: List[Dict[str, Any]] = []
        self.time_history: List[float] = []
        
        # Callbacks
        self._progress_callbacks: List[Callable[[float, str], None]] = []
        self._state_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Deterministic local RNG (never seeds/touches the global numpy RNG)
        self.rng = np.random.default_rng(
            0 if self.config.random_seed is None else self.config.random_seed
        )
            
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set simulation parameters."""
        self.parameters = parameters
        self.state = SimulationState.CONFIGURED
        
    def register_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Register a callback for progress updates."""
        self._progress_callbacks.append(callback)
        
    def register_state_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for state updates."""
        self._state_callbacks.append(callback)
        
    def _notify_progress(self, progress: float, message: str) -> None:
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
                
    def _notify_state(self, state: Dict[str, Any]) -> None:
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
                
    def _record_state(self, time: float, state: Dict[str, Any]) -> None:
        """Record state for temporal modeling."""
        self.time_history.append(time)
        self.state_history.append(state.copy())
        self._notify_state(state)
        
    @abstractmethod
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """
        Validate simulation parameters.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize simulation state before running."""
        pass
        
    @abstractmethod
    def step(self, dt: float) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Current state after step
        """
        pass
        
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        pass
        
    @abstractmethod
    def compute_results(self) -> SimulationResult:
        """Compute final simulation results."""
        pass
        
    def run(self) -> SimulationResult:
        """
        Run the simulation.
        
        This is the main entry point that orchestrates the simulation.
        """
        # Validate parameters
        is_valid, errors = self.validate_parameters()
        if not is_valid:
            self.state = SimulationState.FAILED
            return SimulationResult(
                success=False,
                data={},
                metrics={},
                timestamps=[],
                states=[],
                errors=errors
            )
            
        # Initialize
        self.state = SimulationState.RUNNING
        self.started_at = datetime.datetime.now()
        
        try:
            self.initialize()
            
            # Main simulation loop
            current_time = 0.0
            iteration = 0
            
            while not self.is_complete() and iteration < self.config.max_iterations:
                # Execute step
                state = self.step(self.config.time_step)
                current_time += self.config.time_step
                iteration += 1
                
                # Record state
                self._record_state(current_time, state)
                
                # Progress update
                progress = min(1.0, iteration / self.config.max_iterations)
                self._notify_progress(progress, f"Iteration {iteration}")
                
            # Compute results
            self.result = self.compute_results()
            self.state = SimulationState.COMPLETED
            self.completed_at = datetime.datetime.now()
            
            return self.result
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            self.state = SimulationState.FAILED
            return SimulationResult(
                success=False,
                data={},
                metrics={},
                timestamps=[],
                states=self.state_history,
                errors=[str(e)]
            )
            
    def pause(self) -> None:
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            
    def resume(self) -> None:
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            
    def get_state_at_time(self, time: float) -> Optional[Dict[str, Any]]:
        """Get interpolated state at a specific time."""
        if not self.time_history:
            return None
            
        # Find bracketing times
        for i, t in enumerate(self.time_history):
            if t >= time:
                if i == 0:
                    return self.state_history[0]
                    
                # Linear interpolation
                t0, t1 = self.time_history[i-1], self.time_history[i]
                s0, s1 = self.state_history[i-1], self.state_history[i]
                
                alpha = (time - t0) / (t1 - t0)
                
                # Interpolate numeric values
                interpolated = {}
                for key in s0:
                    if isinstance(s0[key], (int, float)) and isinstance(s1[key], (int, float)):
                        interpolated[key] = s0[key] + alpha * (s1[key] - s0[key])
                    else:
                        interpolated[key] = s1[key] if alpha > 0.5 else s0[key]
                        
                return interpolated
                
        return self.state_history[-1] if self.state_history else None


class PhysicsEngine:
    """
    Physics engine for digital twin simulations.
    
    Provides rigid body dynamics, collision detection, and terrain interaction.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gravity = np.array(config.gravity)
        
        # Objects in the simulation
        self.rigid_bodies: Dict[str, 'RigidBody'] = {}
        self.terrain: Optional['TerrainModel'] = None
        self.constraints: List['Constraint'] = []
        
    def add_rigid_body(self, body: 'RigidBody') -> None:
        """Add a rigid body to the simulation."""
        self.rigid_bodies[body.body_id] = body
        
    def remove_rigid_body(self, body_id: str) -> None:
        """Remove a rigid body from the simulation."""
        if body_id in self.rigid_bodies:
            del self.rigid_bodies[body_id]
            
    def set_terrain(self, terrain: 'TerrainModel') -> None:
        """Set the terrain model."""
        self.terrain = terrain
        
    def add_constraint(self, constraint: 'Constraint') -> None:
        """Add a constraint between bodies."""
        self.constraints.append(constraint)
        
    def step(self, dt: float) -> Dict[str, Any]:
        """
        Execute one physics step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Physics state after step
        """
        # Apply gravity to all bodies
        for body in self.rigid_bodies.values():
            if not body.is_static:
                body.apply_force(self.gravity * body.mass)
                
        # Integrate velocities and positions
        for body in self.rigid_bodies.values():
            body.integrate(dt)
            
        # Collision detection and response
        if self.config.enable_collision:
            self._handle_collisions()
            
        # Terrain interaction
        if self.terrain is not None:
            self._handle_terrain_interaction()
            
        # Apply constraints
        for constraint in self.constraints:
            constraint.apply()
            
        # Collect state
        state = {
            'bodies': {
                body_id: body.get_state()
                for body_id, body in self.rigid_bodies.items()
            },
            'total_energy': self._compute_total_energy()
        }
        
        return state
        
    def _handle_collisions(self) -> None:
        """Handle collisions between rigid bodies."""
        bodies = list(self.rigid_bodies.values())
        
        for i, body1 in enumerate(bodies):
            for body2 in bodies[i+1:]:
                if self._check_collision(body1, body2):
                    self._resolve_collision(body1, body2)
                    
    def _check_collision(self, body1: 'RigidBody', body2: 'RigidBody') -> bool:
        """Check if two bodies are colliding (sphere approximation)."""
        distance = np.linalg.norm(body1.position - body2.position)
        return distance < (body1.radius + body2.radius)
        
    def _resolve_collision(self, body1: 'RigidBody', body2: 'RigidBody') -> None:
        """Resolve collision between two bodies."""
        # Normal vector
        normal = body2.position - body1.position
        distance = np.linalg.norm(normal)
        if distance == 0:
            return
        normal = normal / distance
        
        # Relative velocity
        rel_vel = body1.velocity - body2.velocity
        vel_along_normal = np.dot(rel_vel, normal)
        
        # Don't resolve if velocities are separating
        if vel_along_normal > 0:
            return
            
        # Coefficient of restitution
        e = 0.5
        
        # Impulse magnitude
        j = -(1 + e) * vel_along_normal
        j /= (1/body1.mass + 1/body2.mass)
        
        # Apply impulse
        impulse = j * normal
        if not body1.is_static:
            body1.velocity += impulse / body1.mass
        if not body2.is_static:
            body2.velocity -= impulse / body2.mass
            
    def _handle_terrain_interaction(self) -> None:
        """Handle interaction between bodies and terrain."""
        for body in self.rigid_bodies.values():
            if body.is_static:
                continue
                
            # Get terrain height at body position
            terrain_height = self.terrain.get_height(body.position[0], body.position[1])
            
            # Check if body is below terrain
            if body.position[2] - body.radius < terrain_height:
                # Push body up
                body.position[2] = terrain_height + body.radius
                
                # Reflect velocity with damping
                if body.velocity[2] < 0:
                    body.velocity[2] = -body.velocity[2] * 0.3
                    
                # Apply friction
                body.velocity[0] *= 0.95
                body.velocity[1] *= 0.95
                
    def _compute_total_energy(self) -> float:
        """Compute total energy in the system."""
        total = 0.0
        
        for body in self.rigid_bodies.values():
            # Kinetic energy
            total += 0.5 * body.mass * np.dot(body.velocity, body.velocity)
            
            # Potential energy (relative to z=0)
            total += body.mass * abs(self.gravity[2]) * body.position[2]
            
        return total


class RigidBody:
    """Rigid body for physics simulation."""
    
    def __init__(self, body_id: str, mass: float = 1.0, radius: float = 1.0,
                 position: Optional[np.ndarray] = None,
                 velocity: Optional[np.ndarray] = None,
                 is_static: bool = False):
        self.body_id = body_id
        self.mass = mass
        self.radius = radius
        self.position = position if position is not None else np.zeros(3)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.acceleration = np.zeros(3)
        self.force_accumulator = np.zeros(3)
        self.is_static = is_static
        
        # Rotation (quaternion)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.zeros(3)
        
        # Inertia tensor (sphere approximation)
        self.inertia = (2/5) * mass * radius**2 * np.eye(3)
        
    def apply_force(self, force: np.ndarray, point: Optional[np.ndarray] = None) -> None:
        """Apply a force to the body."""
        self.force_accumulator += force
        
        # Apply torque if force is not at center of mass
        if point is not None:
            r = point - self.position
            torque = np.cross(r, force)
            # Would apply torque here for full rigid body dynamics
            
    def integrate(self, dt: float) -> None:
        """Integrate motion equations."""
        if self.is_static:
            return
            
        # Compute acceleration
        self.acceleration = self.force_accumulator / self.mass
        
        # Semi-implicit Euler integration
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Clear force accumulator
        self.force_accumulator = np.zeros(3)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist()
        }


class TerrainModel:
    """
    Terrain model for digital twin simulations.
    
    Supports heightmap-based terrain with various geological features.
    """
    
    def __init__(self, width: float, height: float, resolution: int = 100):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Heightmap
        self.heightmap = np.zeros((resolution, resolution))
        
        # Material properties
        self.material_map: Optional[np.ndarray] = None
        self.friction_map: Optional[np.ndarray] = None
        
    def generate_from_noise(self, octaves: int = 4, persistence: float = 0.5,
                           scale: float = 100.0, seed: Optional[int] = None) -> None:
        """
        Generate terrain using Perlin-like noise.

        Fully deterministic: a dedicated local RNG is used (default seed 0
        when none is given); the global numpy RNG is never touched.
        """
        rng = np.random.default_rng(0 if seed is None else seed)

        self.heightmap = np.zeros((self.resolution, self.resolution))

        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave

            # Generate noise at this octave
            noise = self._generate_noise(self.resolution // freq + 1, rng)

            # Upsample to full resolution
            noise = self._upsample(noise, self.resolution)

            self.heightmap += noise * amp * scale

    def _generate_noise(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Generate deterministic noise from a local RNG."""
        return rng.standard_normal((size, size))
        
    def _upsample(self, arr: np.ndarray, target_size: int) -> np.ndarray:
        """Upsample array to target size using bilinear interpolation."""
        from scipy import ndimage
        zoom_factor = target_size / arr.shape[0]
        return ndimage.zoom(arr, zoom_factor, order=1)
        
    def set_heightmap(self, heightmap: np.ndarray) -> None:
        """Set heightmap directly."""
        self.heightmap = heightmap
        self.resolution = heightmap.shape[0]
        
    def get_height(self, x: float, y: float) -> float:
        """Get terrain height at a point."""
        # Convert to grid coordinates
        gx = int((x / self.width + 0.5) * (self.resolution - 1))
        gy = int((y / self.height + 0.5) * (self.resolution - 1))
        
        # Clamp to valid range
        gx = max(0, min(self.resolution - 1, gx))
        gy = max(0, min(self.resolution - 1, gy))
        
        return self.heightmap[gy, gx]
        
    def get_normal(self, x: float, y: float) -> np.ndarray:
        """Get terrain normal at a point."""
        eps = self.width / self.resolution
        
        # Sample heights
        h_center = self.get_height(x, y)
        h_right = self.get_height(x + eps, y)
        h_up = self.get_height(x, y + eps)
        
        # Compute normal
        dx = np.array([eps, 0, h_right - h_center])
        dy = np.array([0, eps, h_up - h_center])
        
        normal = np.cross(dx, dy)
        return normal / np.linalg.norm(normal)
        
    def get_slope(self, x: float, y: float) -> float:
        """Get terrain slope angle at a point (in degrees)."""
        normal = self.get_normal(x, y)
        return np.degrees(np.arccos(normal[2]))
        
    def add_geological_feature(self, feature_type: str, center: Tuple[float, float],
                               radius: float, depth: float) -> None:
        """Add a geological feature to the terrain."""
        cx, cy = center
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = (i / (self.resolution - 1) - 0.5) * self.width
                y = (j / (self.resolution - 1) - 0.5) * self.height
                
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                if dist < radius:
                    if feature_type == 'crater':
                        # Crater profile
                        t = dist / radius
                        profile = depth * (t**2 - 1)
                        self.heightmap[j, i] += profile
                        
                    elif feature_type == 'hill':
                        # Gaussian hill
                        t = dist / radius
                        profile = depth * np.exp(-t**2 * 3)
                        self.heightmap[j, i] += profile
                        
                    elif feature_type == 'pit':
                        # Mining pit
                        t = dist / radius
                        profile = -depth * (1 - t**2)
                        self.heightmap[j, i] += profile


class Constraint:
    """Base class for physics constraints."""
    
    def __init__(self, body1: RigidBody, body2: Optional[RigidBody] = None):
        self.body1 = body1
        self.body2 = body2
        
    def apply(self) -> None:
        """Apply the constraint."""
        pass


class DistanceConstraint(Constraint):
    """Constraint to maintain distance between two bodies."""
    
    def __init__(self, body1: RigidBody, body2: RigidBody, distance: float,
                 stiffness: float = 0.5):
        super().__init__(body1, body2)
        self.distance = distance
        self.stiffness = stiffness
        
    def apply(self) -> None:
        """Apply distance constraint."""
        if self.body2 is None:
            return
            
        # Current distance
        delta = self.body2.position - self.body1.position
        current_dist = np.linalg.norm(delta)
        
        if current_dist == 0:
            return
            
        # Correction
        error = current_dist - self.distance
        correction = (delta / current_dist) * error * self.stiffness
        
        # Apply correction
        if not self.body1.is_static:
            self.body1.position += correction * 0.5
        if not self.body2.is_static:
            self.body2.position -= correction * 0.5


class EquipmentSimulation(AbstractSimulation):
    """
    Simulation of mining equipment operations.
    """
    
    def __init__(self, name: str, config: Optional[SimulationConfig] = None):
        super().__init__(name, config)
        
        # Equipment state
        self.equipment: Dict[str, Dict[str, Any]] = {}
        self.physics_engine: Optional[PhysicsEngine] = None
        
        # Operational parameters
        self.fuel_consumption_rate = 0.0
        self.production_rate = 0.0
        self.current_time = 0.0
        
    def add_equipment(self, equipment_id: str, equipment_type: str,
                     position: Tuple[float, float, float],
                     capacity: float = 100.0) -> None:
        """Add equipment to the simulation."""
        self.equipment[equipment_id] = {
            'type': equipment_type,
            'position': np.array(position),
            'capacity': capacity,
            'current_load': 0.0,
            'fuel_level': 100.0,
            'operational': True,
            'hours_operated': 0.0
        }
        
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate simulation parameters."""
        errors = []
        
        if not self.equipment:
            errors.append("No equipment defined")
            
        if 'operation_duration' not in self.parameters:
            errors.append("operation_duration parameter required")
            
        return len(errors) == 0, errors
        
    def initialize(self) -> None:
        """Initialize the simulation."""
        # Initialize physics engine if enabled
        if self.config.enable_physics:
            self.physics_engine = PhysicsEngine(self.config)
            
            # Add equipment as rigid bodies
            for eq_id, eq_data in self.equipment.items():
                body = RigidBody(
                    body_id=eq_id,
                    mass=10000.0,  # 10 tons
                    radius=5.0,
                    position=eq_data['position'].copy()
                )
                self.physics_engine.add_rigid_body(body)
                
        self.current_time = 0.0
        self.fuel_consumption_rate = self.parameters.get('fuel_consumption_rate', 10.0)
        self.production_rate = self.parameters.get('production_rate', 50.0)
        
    def step(self, dt: float) -> Dict[str, Any]:
        """Execute one simulation step."""
        self.current_time += dt
        
        # Update equipment state
        for eq_id, eq_data in self.equipment.items():
            if not eq_data['operational']:
                continue
                
            # Consume fuel
            eq_data['fuel_level'] -= self.fuel_consumption_rate * dt / 3600
            eq_data['hours_operated'] += dt / 3600
            
            # Check fuel
            if eq_data['fuel_level'] <= 0:
                eq_data['fuel_level'] = 0
                eq_data['operational'] = False
                
            # Production
            eq_data['current_load'] += self.production_rate * dt / 3600
            if eq_data['current_load'] > eq_data['capacity']:
                eq_data['current_load'] = eq_data['capacity']
                
        # Physics step
        physics_state = {}
        if self.physics_engine is not None:
            physics_state = self.physics_engine.step(dt)
            
        # Collect state
        state = {
            'time': self.current_time,
            'equipment': {
                eq_id: {
                    'fuel_level': eq_data['fuel_level'],
                    'current_load': eq_data['current_load'],
                    'operational': eq_data['operational'],
                    'hours_operated': eq_data['hours_operated']
                }
                for eq_id, eq_data in self.equipment.items()
            },
            'physics': physics_state
        }
        
        return state
        
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        operation_duration = self.parameters.get('operation_duration', 3600)
        return self.current_time >= operation_duration
        
    def compute_results(self) -> SimulationResult:
        """Compute final results."""
        total_production = sum(
            eq['current_load'] for eq in self.equipment.values()
        )
        total_fuel_consumed = sum(
            100 - eq['fuel_level'] for eq in self.equipment.values()
        )
        operational_equipment = sum(
            1 for eq in self.equipment.values() if eq['operational']
        )
        
        return SimulationResult(
            success=True,
            data={
                'final_equipment_state': {
                    eq_id: eq_data.copy()
                    for eq_id, eq_data in self.equipment.items()
                }
            },
            metrics={
                'total_production': total_production,
                'total_fuel_consumed': total_fuel_consumed,
                'operational_equipment': operational_equipment,
                'total_equipment': len(self.equipment),
                'simulation_duration': self.current_time
            },
            timestamps=[datetime.datetime.now()],
            states=self.state_history
        )


class GeologicalSimulation(AbstractSimulation):
    """
    Simulation of geological processes and mineral deposit evolution.
    """
    
    def __init__(self, name: str, config: Optional[SimulationConfig] = None):
        super().__init__(name, config)
        
        # Geological state
        self.terrain: Optional[TerrainModel] = None
        self.mineral_concentration: Optional[np.ndarray] = None
        self.current_time = 0.0
        
        # Process rates
        self.erosion_rate = 0.0
        self.deposition_rate = 0.0
        self.mineralization_rate = 0.0
        
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate parameters."""
        errors = []
        
        if 'terrain_size' not in self.parameters:
            errors.append("terrain_size parameter required")
            
        return len(errors) == 0, errors
        
    def initialize(self) -> None:
        """Initialize the simulation."""
        terrain_size = self.parameters.get('terrain_size', (1000, 1000))
        resolution = self.parameters.get('resolution', 100)
        
        # Create terrain
        self.terrain = TerrainModel(terrain_size[0], terrain_size[1], resolution)
        self.terrain.generate_from_noise(
            octaves=self.parameters.get('terrain_octaves', 4),
            scale=self.parameters.get('terrain_scale', 50.0),
            seed=self.config.random_seed
        )
        
        # Initialize mineral concentration (deterministic local RNG)
        rng = np.random.default_rng(
            0 if self.config.random_seed is None else self.config.random_seed
        )
        self.mineral_concentration = rng.random((resolution, resolution)) * 0.1
        
        # Add initial mineral deposits
        for deposit in self.parameters.get('initial_deposits', []):
            self._add_mineral_deposit(
                deposit['center'],
                deposit['radius'],
                deposit['concentration']
            )
            
        # Process rates
        self.erosion_rate = self.parameters.get('erosion_rate', 0.001)
        self.deposition_rate = self.parameters.get('deposition_rate', 0.0005)
        self.mineralization_rate = self.parameters.get('mineralization_rate', 0.0001)
        
        self.current_time = 0.0
        
    def _add_mineral_deposit(self, center: Tuple[float, float],
                            radius: float, concentration: float) -> None:
        """Add a mineral deposit."""
        res = self.terrain.resolution
        
        for i in range(res):
            for j in range(res):
                x = (i / (res - 1) - 0.5) * self.terrain.width
                y = (j / (res - 1) - 0.5) * self.terrain.height
                
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                
                if dist < radius:
                    t = dist / radius
                    self.mineral_concentration[j, i] += concentration * np.exp(-t**2 * 3)
                    
    def step(self, dt: float) -> Dict[str, Any]:
        """Execute one simulation step."""
        self.current_time += dt
        
        # Erosion
        self._apply_erosion(dt)
        
        # Deposition
        self._apply_deposition(dt)
        
        # Mineralization
        self._apply_mineralization(dt)
        
        # Collect state
        state = {
            'time': self.current_time,
            'mean_elevation': float(np.mean(self.terrain.heightmap)),
            'max_elevation': float(np.max(self.terrain.heightmap)),
            'min_elevation': float(np.min(self.terrain.heightmap)),
            'mean_concentration': float(np.mean(self.mineral_concentration)),
            'max_concentration': float(np.max(self.mineral_concentration)),
            'total_mineral_mass': float(np.sum(self.mineral_concentration))
        }
        
        return state
        
    def _apply_erosion(self, dt: float) -> None:
        """Apply erosion to terrain."""
        # Simple slope-based erosion
        from scipy import ndimage
        
        # Compute gradient magnitude
        gy, gx = np.gradient(self.terrain.heightmap)
        slope = np.sqrt(gx**2 + gy**2)
        
        # Erode based on slope
        erosion = slope * self.erosion_rate * dt
        self.terrain.heightmap -= erosion
        
        # Erode minerals too
        self.mineral_concentration -= erosion * self.mineral_concentration * 0.1
        self.mineral_concentration = np.maximum(0, self.mineral_concentration)
        
    def _apply_deposition(self, dt: float) -> None:
        """Apply deposition to terrain."""
        # Simple diffusion-based deposition
        from scipy import ndimage
        
        # Smooth terrain slightly
        smoothed = ndimage.gaussian_filter(self.terrain.heightmap, sigma=1)
        self.terrain.heightmap += (smoothed - self.terrain.heightmap) * self.deposition_rate * dt
        
    def _apply_mineralization(self, dt: float) -> None:
        """Apply mineralization process."""
        # Minerals concentrate in low areas
        elevation_factor = 1 - (self.terrain.heightmap - np.min(self.terrain.heightmap)) / \
                          (np.max(self.terrain.heightmap) - np.min(self.terrain.heightmap) + 1e-6)
        
        self.mineral_concentration += elevation_factor * self.mineralization_rate * dt
        
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        simulation_years = self.parameters.get('simulation_years', 1000)
        return self.current_time >= simulation_years
        
    def compute_results(self) -> SimulationResult:
        """Compute final results."""
        # Find high-concentration areas
        threshold = np.percentile(self.mineral_concentration, 90)
        high_conc_mask = self.mineral_concentration > threshold
        
        return SimulationResult(
            success=True,
            data={
                'final_heightmap': self.terrain.heightmap.tolist(),
                'final_concentration': self.mineral_concentration.tolist(),
                'high_concentration_areas': np.sum(high_conc_mask)
            },
            metrics={
                'mean_elevation': float(np.mean(self.terrain.heightmap)),
                'elevation_range': float(np.max(self.terrain.heightmap) - np.min(self.terrain.heightmap)),
                'total_mineral_mass': float(np.sum(self.mineral_concentration)),
                'mean_concentration': float(np.mean(self.mineral_concentration)),
                'max_concentration': float(np.max(self.mineral_concentration)),
                'simulation_duration': self.current_time
            },
            timestamps=[datetime.datetime.now()],
            states=self.state_history
        )


class TemporalStateModel:
    """
    Temporal state evolution model for digital twins.
    
    Tracks state changes over time and supports prediction.
    """
    
    def __init__(self, state_dim: int, history_length: int = 1000):
        self.state_dim = state_dim
        self.history_length = history_length
        
        # State history
        self.states: deque = deque(maxlen=history_length)
        self.timestamps: deque = deque(maxlen=history_length)
        
        # State transition model (simple linear)
        self.transition_matrix: Optional[np.ndarray] = None
        self.process_noise: Optional[np.ndarray] = None
        
        # Statistics
        self.mean_state: Optional[np.ndarray] = None
        self.state_covariance: Optional[np.ndarray] = None
        
    def record_state(self, state: np.ndarray, timestamp: datetime.datetime) -> None:
        """Record a state observation."""
        self.states.append(state.copy())
        self.timestamps.append(timestamp)
        
        # Update statistics
        self._update_statistics()
        
        # Update transition model if enough data
        if len(self.states) >= 10:
            self._update_transition_model()
            
    def _update_statistics(self) -> None:
        """Update running statistics."""
        states = np.array(self.states)
        self.mean_state = np.mean(states, axis=0)
        self.state_covariance = np.cov(states.T) if len(states) > 1 else np.eye(self.state_dim)
        
    def _update_transition_model(self) -> None:
        """Update state transition model using least squares."""
        states = np.array(self.states)
        
        if len(states) < 2:
            return
            
        # X[t+1] = A * X[t] + noise
        X = states[:-1]
        Y = states[1:]
        
        # Least squares: A = Y * X^T * (X * X^T)^-1
        try:
            self.transition_matrix = Y.T @ X @ np.linalg.pinv(X.T @ X)
            
            # Estimate process noise
            predictions = X @ self.transition_matrix.T
            residuals = Y - predictions
            self.process_noise = np.cov(residuals.T)
        except np.linalg.LinAlgError:
            pass
            
    def predict(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future states.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        if self.transition_matrix is None or len(self.states) == 0:
            return np.zeros((steps, self.state_dim)), np.ones((steps, self.state_dim))
            
        current_state = np.array(self.states[-1])
        predictions = []
        uncertainties = []
        
        uncertainty = np.diag(self.state_covariance) if self.state_covariance is not None else np.ones(self.state_dim)
        
        for _ in range(steps):
            # Predict next state
            next_state = self.transition_matrix @ current_state
            predictions.append(next_state)
            
            # Propagate uncertainty
            if self.process_noise is not None:
                uncertainty = np.diag(self.transition_matrix @ np.diag(uncertainty) @ self.transition_matrix.T + self.process_noise)
            uncertainties.append(uncertainty.copy())
            
            current_state = next_state
            
        return np.array(predictions), np.array(uncertainties)
        
    def get_trend(self) -> np.ndarray:
        """Get current trend (rate of change)."""
        if len(self.states) < 2:
            return np.zeros(self.state_dim)
            
        states = np.array(self.states)
        
        # Simple linear regression for trend
        t = np.arange(len(states))
        trends = []
        
        for i in range(self.state_dim):
            slope = np.polyfit(t, states[:, i], 1)[0]
            trends.append(slope)
            
        return np.array(trends)
        
    def detect_anomaly(self, state: np.ndarray, threshold: float = 3.0) -> Tuple[bool, float]:
        """
        Detect if a state is anomalous.
        
        Args:
            state: State to check
            threshold: Number of standard deviations for anomaly
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if self.mean_state is None or self.state_covariance is None:
            return False, 0.0
            
        # Mahalanobis distance
        diff = state - self.mean_state
        try:
            cov_inv = np.linalg.inv(self.state_covariance)
            distance = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            distance = np.linalg.norm(diff / (np.sqrt(np.diag(self.state_covariance)) + 1e-6))
            
        return distance > threshold, float(distance)


def create_equipment_simulation(
    name: str,
    equipment_list: List[Dict[str, Any]],
    operation_duration: float = 3600,
    **kwargs
) -> EquipmentSimulation:
    """
    Factory function to create an equipment simulation.
    
    Args:
        name: Simulation name
        equipment_list: List of equipment definitions
        operation_duration: Duration in seconds
        **kwargs: Additional parameters
        
    Returns:
        Configured EquipmentSimulation
    """
    config = SimulationConfig(
        time_step=kwargs.get('time_step', 1.0),
        enable_physics=kwargs.get('enable_physics', True)
    )
    
    sim = EquipmentSimulation(name, config)
    
    for eq in equipment_list:
        sim.add_equipment(
            equipment_id=eq['id'],
            equipment_type=eq['type'],
            position=eq['position'],
            capacity=eq.get('capacity', 100.0)
        )
        
    sim.set_parameters({
        'operation_duration': operation_duration,
        'fuel_consumption_rate': kwargs.get('fuel_consumption_rate', 10.0),
        'production_rate': kwargs.get('production_rate', 50.0)
    })
    
    return sim


def create_geological_simulation(
    name: str,
    terrain_size: Tuple[float, float],
    simulation_years: float = 1000,
    **kwargs
) -> GeologicalSimulation:
    """
    Factory function to create a geological simulation.
    
    Args:
        name: Simulation name
        terrain_size: Size of terrain (width, height)
        simulation_years: Duration in years
        **kwargs: Additional parameters
        
    Returns:
        Configured GeologicalSimulation
    """
    config = SimulationConfig(
        time_step=kwargs.get('time_step', 1.0),
        random_seed=kwargs.get('random_seed')
    )
    
    sim = GeologicalSimulation(name, config)
    
    sim.set_parameters({
        'terrain_size': terrain_size,
        'resolution': kwargs.get('resolution', 100),
        'terrain_octaves': kwargs.get('terrain_octaves', 4),
        'terrain_scale': kwargs.get('terrain_scale', 50.0),
        'simulation_years': simulation_years,
        'erosion_rate': kwargs.get('erosion_rate', 0.001),
        'deposition_rate': kwargs.get('deposition_rate', 0.0005),
        'mineralization_rate': kwargs.get('mineralization_rate', 0.0001),
        'initial_deposits': kwargs.get('initial_deposits', [])
    })
    
    return sim
