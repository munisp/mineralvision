"""
3D Visualization Engine for MineralVision Digital Twin.

This module provides 3D visualization capabilities for digital twin data,
including terrain rendering, equipment visualization, and real-time updates.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import queue
import logging
import base64
import io

logger = logging.getLogger(__name__)


class RenderMode(Enum):
    """Rendering modes for 3D visualization."""
    WIREFRAME = "wireframe"
    SOLID = "solid"
    TEXTURED = "textured"
    POINT_CLOUD = "point_cloud"
    VOLUMETRIC = "volumetric"


class CameraMode(Enum):
    """Camera modes for 3D visualization."""
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"
    FIRST_PERSON = "first_person"
    ORBIT = "orbit"


@dataclass
class Color:
    """RGBA color representation."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    
    def to_hex(self) -> str:
        """Convert to hex string."""
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}"
        
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple."""
        return (self.r, self.g, self.b, self.a)
        
    @classmethod
    def from_hex(cls, hex_str: str) -> 'Color':
        """Create from hex string."""
        hex_str = hex_str.lstrip('#')
        r = int(hex_str[0:2], 16) / 255
        g = int(hex_str[2:4], 16) / 255
        b = int(hex_str[4:6], 16) / 255
        return cls(r, g, b, 1.0)


@dataclass
class Transform:
    """3D transformation (position, rotation, scale)."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler angles
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        # Translation matrix
        T = np.eye(4)
        T[:3, 3] = self.position
        
        # Rotation matrices (Euler angles: roll, pitch, yaw)
        rx, ry, rz = self.rotation
        
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        # Scale matrix
        S = np.diag([self.scale[0], self.scale[1], self.scale[2], 1])
        
        return T @ R @ S
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'scale': self.scale.tolist()
        }


@dataclass
class Camera:
    """3D camera for visualization."""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, -10.0, 5.0]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fov: float = 60.0  # Field of view in degrees
    near: float = 0.1
    far: float = 10000.0
    mode: CameraMode = CameraMode.PERSPECTIVE
    
    def get_view_matrix(self) -> np.ndarray:
        """Compute view matrix."""
        # Forward vector
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        # Right vector
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        
        # View matrix
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([
            np.dot(right, self.position),
            np.dot(up, self.position),
            np.dot(-forward, self.position)
        ])
        
        return view
        
    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Compute projection matrix."""
        if self.mode == CameraMode.PERSPECTIVE:
            fov_rad = np.radians(self.fov)
            f = 1.0 / np.tan(fov_rad / 2)
            
            proj = np.zeros((4, 4))
            proj[0, 0] = f / aspect_ratio
            proj[1, 1] = f
            proj[2, 2] = (self.far + self.near) / (self.near - self.far)
            proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
            proj[3, 2] = -1
            
            return proj
        else:
            # Orthographic
            size = 100.0
            proj = np.zeros((4, 4))
            proj[0, 0] = 2 / (size * aspect_ratio)
            proj[1, 1] = 2 / size
            proj[2, 2] = -2 / (self.far - self.near)
            proj[2, 3] = -(self.far + self.near) / (self.far - self.near)
            proj[3, 3] = 1
            
            return proj
            
    def orbit(self, delta_azimuth: float, delta_elevation: float) -> None:
        """Orbit camera around target."""
        # Current position relative to target
        rel_pos = self.position - self.target
        
        # Convert to spherical coordinates
        r = np.linalg.norm(rel_pos)
        azimuth = np.arctan2(rel_pos[1], rel_pos[0])
        elevation = np.arcsin(rel_pos[2] / r)
        
        # Update angles
        azimuth += delta_azimuth
        elevation = np.clip(elevation + delta_elevation, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
        # Convert back to Cartesian
        self.position = self.target + r * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ])
        
    def zoom(self, factor: float) -> None:
        """Zoom camera by moving towards/away from target."""
        direction = self.position - self.target
        distance = np.linalg.norm(direction)
        new_distance = max(1.0, distance * factor)
        self.position = self.target + (direction / distance) * new_distance
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position.tolist(),
            'target': self.target.tolist(),
            'up': self.up.tolist(),
            'fov': self.fov,
            'near': self.near,
            'far': self.far,
            'mode': self.mode.value
        }


class Mesh:
    """3D mesh representation."""
    
    def __init__(self, name: str):
        self.name = name
        self.vertices: np.ndarray = np.array([])  # Nx3
        self.normals: np.ndarray = np.array([])   # Nx3
        self.uvs: np.ndarray = np.array([])       # Nx2
        self.indices: np.ndarray = np.array([])   # Mx3 (triangles)
        self.colors: np.ndarray = np.array([])    # Nx4
        
    def compute_normals(self) -> None:
        """Compute vertex normals from faces."""
        if len(self.vertices) == 0 or len(self.indices) == 0:
            return
            
        self.normals = np.zeros_like(self.vertices)
        
        for face in self.indices:
            v0, v1, v2 = self.vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            for idx in face:
                self.normals[idx] += normal
                
        # Normalize
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals = self.normals / (norms + 1e-10)
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'vertices': self.vertices.tolist(),
            'normals': self.normals.tolist(),
            'uvs': self.uvs.tolist() if len(self.uvs) > 0 else [],
            'indices': self.indices.tolist(),
            'colors': self.colors.tolist() if len(self.colors) > 0 else []
        }
        
    @classmethod
    def create_box(cls, name: str, size: Tuple[float, float, float] = (1, 1, 1)) -> 'Mesh':
        """Create a box mesh."""
        mesh = cls(name)
        
        sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
        
        mesh.vertices = np.array([
            # Front face
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
            # Back face
            [-sx, -sy, -sz], [-sx, sy, -sz], [sx, sy, -sz], [sx, -sy, -sz],
            # Top face
            [-sx, sy, -sz], [-sx, sy, sz], [sx, sy, sz], [sx, sy, -sz],
            # Bottom face
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, -sy, sz], [-sx, -sy, sz],
            # Right face
            [sx, -sy, -sz], [sx, sy, -sz], [sx, sy, sz], [sx, -sy, sz],
            # Left face
            [-sx, -sy, -sz], [-sx, -sy, sz], [-sx, sy, sz], [-sx, sy, -sz]
        ], dtype=np.float32)
        
        mesh.indices = np.array([
            [0, 1, 2], [0, 2, 3],     # Front
            [4, 5, 6], [4, 6, 7],     # Back
            [8, 9, 10], [8, 10, 11],  # Top
            [12, 13, 14], [12, 14, 15],  # Bottom
            [16, 17, 18], [16, 18, 19],  # Right
            [20, 21, 22], [20, 22, 23]   # Left
        ], dtype=np.int32)
        
        mesh.compute_normals()
        return mesh
        
    @classmethod
    def create_sphere(cls, name: str, radius: float = 1.0, 
                     segments: int = 16, rings: int = 16) -> 'Mesh':
        """Create a sphere mesh."""
        mesh = cls(name)
        
        vertices = []
        indices = []
        
        for ring in range(rings + 1):
            phi = np.pi * ring / rings
            for seg in range(segments + 1):
                theta = 2 * np.pi * seg / segments
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                vertices.append([x, y, z])
                
        mesh.vertices = np.array(vertices, dtype=np.float32)
        
        # Create triangles
        for ring in range(rings):
            for seg in range(segments):
                current = ring * (segments + 1) + seg
                next_ring = (ring + 1) * (segments + 1) + seg
                
                indices.append([current, next_ring, current + 1])
                indices.append([current + 1, next_ring, next_ring + 1])
                
        mesh.indices = np.array(indices, dtype=np.int32)
        mesh.compute_normals()
        
        return mesh
        
    @classmethod
    def create_cylinder(cls, name: str, radius: float = 1.0, 
                       height: float = 2.0, segments: int = 16) -> 'Mesh':
        """Create a cylinder mesh."""
        mesh = cls(name)
        
        vertices = []
        indices = []
        
        # Side vertices
        for i in range(segments + 1):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            vertices.append([x, y, height/2])
            vertices.append([x, y, -height/2])
            
        # Top and bottom centers
        top_center = len(vertices)
        vertices.append([0, 0, height/2])
        bottom_center = len(vertices)
        vertices.append([0, 0, -height/2])
        
        mesh.vertices = np.array(vertices, dtype=np.float32)
        
        # Side triangles
        for i in range(segments):
            v0 = i * 2
            v1 = i * 2 + 1
            v2 = (i + 1) * 2
            v3 = (i + 1) * 2 + 1
            
            indices.append([v0, v2, v1])
            indices.append([v1, v2, v3])
            
        # Top and bottom caps
        for i in range(segments):
            v0 = i * 2
            v1 = (i + 1) * 2
            indices.append([top_center, v0, v1])
            
            v0 = i * 2 + 1
            v1 = (i + 1) * 2 + 1
            indices.append([bottom_center, v1, v0])
            
        mesh.indices = np.array(indices, dtype=np.int32)
        mesh.compute_normals()
        
        return mesh


class TerrainMesh:
    """Terrain mesh generated from heightmap."""
    
    def __init__(self, name: str, heightmap: np.ndarray,
                 width: float, height: float,
                 vertical_scale: float = 1.0):
        self.name = name
        self.heightmap = heightmap
        self.width = width
        self.height = height
        self.vertical_scale = vertical_scale
        
        self.mesh = self._generate_mesh()
        
    def _generate_mesh(self) -> Mesh:
        """Generate mesh from heightmap."""
        mesh = Mesh(self.name)
        
        rows, cols = self.heightmap.shape
        
        # Generate vertices
        vertices = []
        uvs = []
        
        for i in range(rows):
            for j in range(cols):
                x = (j / (cols - 1) - 0.5) * self.width
                y = (i / (rows - 1) - 0.5) * self.height
                z = self.heightmap[i, j] * self.vertical_scale
                
                vertices.append([x, y, z])
                uvs.append([j / (cols - 1), i / (rows - 1)])
                
        mesh.vertices = np.array(vertices, dtype=np.float32)
        mesh.uvs = np.array(uvs, dtype=np.float32)
        
        # Generate triangles
        indices = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = i * cols + j + 1
                v2 = (i + 1) * cols + j
                v3 = (i + 1) * cols + j + 1
                
                indices.append([v0, v2, v1])
                indices.append([v1, v2, v3])
                
        mesh.indices = np.array(indices, dtype=np.int32)
        mesh.compute_normals()
        
        # Color by height
        min_h = np.min(self.heightmap)
        max_h = np.max(self.heightmap)
        range_h = max_h - min_h + 1e-6
        
        colors = []
        for v in mesh.vertices:
            t = (v[2] / self.vertical_scale - min_h) / range_h
            # Green to brown gradient
            r = 0.2 + 0.6 * t
            g = 0.6 - 0.3 * t
            b = 0.1 + 0.1 * t
            colors.append([r, g, b, 1.0])
            
        mesh.colors = np.array(colors, dtype=np.float32)
        
        return mesh
        
    def update_heightmap(self, heightmap: np.ndarray) -> None:
        """Update terrain with new heightmap."""
        self.heightmap = heightmap
        self.mesh = self._generate_mesh()


class SceneObject:
    """Object in the 3D scene."""
    
    def __init__(self, name: str, mesh: Optional[Mesh] = None):
        self.object_id = str(id(self))
        self.name = name
        self.mesh = mesh
        self.transform = Transform()
        self.color = Color()
        self.visible = True
        self.render_mode = RenderMode.SOLID
        self.metadata: Dict[str, Any] = {}
        self.children: List['SceneObject'] = []
        self.parent: Optional['SceneObject'] = None
        
    def add_child(self, child: 'SceneObject') -> None:
        """Add a child object."""
        child.parent = self
        self.children.append(child)
        
    def remove_child(self, child: 'SceneObject') -> None:
        """Remove a child object."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            
    def get_world_transform(self) -> np.ndarray:
        """Get world transformation matrix."""
        local = self.transform.to_matrix()
        if self.parent is not None:
            return self.parent.get_world_transform() @ local
        return local
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'object_id': self.object_id,
            'name': self.name,
            'transform': self.transform.to_dict(),
            'color': self.color.to_hex(),
            'visible': self.visible,
            'render_mode': self.render_mode.value,
            'metadata': self.metadata,
            'mesh': self.mesh.to_dict() if self.mesh else None,
            'children': [c.to_dict() for c in self.children]
        }


class Scene:
    """3D scene container."""
    
    def __init__(self, name: str = "MineralVision Scene"):
        self.name = name
        self.objects: Dict[str, SceneObject] = {}
        self.camera = Camera()
        self.ambient_light = Color(0.3, 0.3, 0.3, 1.0)
        self.directional_light_direction = np.array([0.5, 0.5, -1.0])
        self.directional_light_color = Color(1.0, 1.0, 0.9, 1.0)
        self.background_color = Color(0.1, 0.1, 0.15, 1.0)
        
        # Selection
        self.selected_object: Optional[str] = None
        
    def add_object(self, obj: SceneObject) -> None:
        """Add an object to the scene."""
        self.objects[obj.object_id] = obj
        
    def remove_object(self, object_id: str) -> None:
        """Remove an object from the scene."""
        if object_id in self.objects:
            del self.objects[object_id]
            
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get an object by ID."""
        return self.objects.get(object_id)
        
    def find_objects_by_name(self, name: str) -> List[SceneObject]:
        """Find objects by name."""
        return [obj for obj in self.objects.values() if obj.name == name]
        
    def clear(self) -> None:
        """Clear all objects from the scene."""
        self.objects.clear()
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scene bounding box."""
        if not self.objects:
            return np.zeros(3), np.zeros(3)
            
        all_min = np.array([float('inf')] * 3)
        all_max = np.array([float('-inf')] * 3)
        
        for obj in self.objects.values():
            if obj.mesh is not None:
                obj_min, obj_max = obj.mesh.get_bounds()
                transform = obj.get_world_transform()
                
                # Transform bounds (simplified - uses corners)
                for corner in [obj_min, obj_max]:
                    world_corner = (transform @ np.append(corner, 1))[:3]
                    all_min = np.minimum(all_min, world_corner)
                    all_max = np.maximum(all_max, world_corner)
                    
        return all_min, all_max
        
    def fit_camera_to_scene(self) -> None:
        """Adjust camera to fit entire scene."""
        scene_min, scene_max = self.get_bounds()
        center = (scene_min + scene_max) / 2
        size = np.linalg.norm(scene_max - scene_min)
        
        self.camera.target = center
        self.camera.position = center + np.array([0, -size, size * 0.5])
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene to dictionary."""
        return {
            'name': self.name,
            'objects': {oid: obj.to_dict() for oid, obj in self.objects.items()},
            'camera': self.camera.to_dict(),
            'ambient_light': self.ambient_light.to_hex(),
            'directional_light_direction': self.directional_light_direction.tolist(),
            'directional_light_color': self.directional_light_color.to_hex(),
            'background_color': self.background_color.to_hex()
        }
        
    def to_json(self) -> str:
        """Convert scene to JSON string."""
        return json.dumps(self.to_dict())


class Visualization3DEngine:
    """
    3D visualization engine for digital twin.
    
    Provides scene management, rendering, and export capabilities.
    """
    
    def __init__(self):
        self.scene = Scene()
        self.render_width = 1920
        self.render_height = 1080
        
        # Update callbacks
        self._update_callbacks: List[Callable[[Scene], None]] = []
        
        # Animation
        self._animation_running = False
        self._animation_thread: Optional[threading.Thread] = None
        self._animation_queue: queue.Queue = queue.Queue()
        
    def set_resolution(self, width: int, height: int) -> None:
        """Set render resolution."""
        self.render_width = width
        self.render_height = height
        
    def register_update_callback(self, callback: Callable[[Scene], None]) -> None:
        """Register callback for scene updates."""
        self._update_callbacks.append(callback)
        
    def _notify_update(self) -> None:
        """Notify all update callbacks."""
        for callback in self._update_callbacks:
            try:
                callback(self.scene)
            except Exception as e:
                logger.error(f"Update callback error: {e}")
                
    def add_terrain(self, heightmap: np.ndarray, width: float, height: float,
                   vertical_scale: float = 1.0, name: str = "Terrain") -> SceneObject:
        """Add terrain to the scene."""
        terrain_mesh = TerrainMesh(name, heightmap, width, height, vertical_scale)
        
        obj = SceneObject(name, terrain_mesh.mesh)
        obj.metadata['terrain'] = True
        obj.metadata['heightmap_shape'] = heightmap.shape
        
        self.scene.add_object(obj)
        self._notify_update()
        
        return obj
        
    def add_equipment(self, equipment_id: str, equipment_type: str,
                     position: Tuple[float, float, float],
                     scale: float = 1.0) -> SceneObject:
        """Add equipment to the scene."""
        # Create appropriate mesh based on type
        if equipment_type in ['excavator', 'loader']:
            mesh = Mesh.create_box(equipment_id, (3 * scale, 2 * scale, 2 * scale))
            color = Color(0.9, 0.7, 0.1)
        elif equipment_type in ['truck', 'hauler']:
            mesh = Mesh.create_box(equipment_id, (4 * scale, 2 * scale, 2 * scale))
            color = Color(0.8, 0.8, 0.2)
        elif equipment_type == 'drill':
            mesh = Mesh.create_cylinder(equipment_id, 0.5 * scale, 4 * scale)
            color = Color(0.6, 0.6, 0.6)
        else:
            mesh = Mesh.create_sphere(equipment_id, 1.0 * scale)
            color = Color(0.5, 0.5, 0.8)
            
        obj = SceneObject(equipment_id, mesh)
        obj.transform.position = np.array(position)
        obj.color = color
        obj.metadata['equipment_type'] = equipment_type
        obj.metadata['equipment_id'] = equipment_id
        
        self.scene.add_object(obj)
        self._notify_update()
        
        return obj
        
    def add_mineral_deposit(self, deposit_id: str, center: Tuple[float, float, float],
                           radius: float, concentration: float = 1.0) -> SceneObject:
        """Add mineral deposit visualization."""
        mesh = Mesh.create_sphere(deposit_id, radius, segments=24, rings=24)
        
        obj = SceneObject(deposit_id, mesh)
        obj.transform.position = np.array(center)
        
        # Color based on concentration
        obj.color = Color(
            0.8 * concentration,
            0.2 * (1 - concentration),
            0.1,
            0.7  # Semi-transparent
        )
        obj.metadata['deposit_id'] = deposit_id
        obj.metadata['concentration'] = concentration
        
        self.scene.add_object(obj)
        self._notify_update()
        
        return obj
        
    def add_point_cloud(self, name: str, points: np.ndarray,
                       colors: Optional[np.ndarray] = None) -> SceneObject:
        """Add point cloud to the scene."""
        mesh = Mesh(name)
        mesh.vertices = points.astype(np.float32)
        
        if colors is not None:
            mesh.colors = colors.astype(np.float32)
        else:
            # Default coloring by height
            z = points[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
            mesh.colors = np.column_stack([
                z_norm,
                1 - z_norm,
                np.zeros_like(z_norm),
                np.ones_like(z_norm)
            ]).astype(np.float32)
            
        obj = SceneObject(name, mesh)
        obj.render_mode = RenderMode.POINT_CLOUD
        
        self.scene.add_object(obj)
        self._notify_update()
        
        return obj
        
    def update_object_position(self, object_id: str, 
                              position: Tuple[float, float, float]) -> None:
        """Update object position."""
        obj = self.scene.get_object(object_id)
        if obj:
            obj.transform.position = np.array(position)
            self._notify_update()
            
    def update_object_transform(self, object_id: str, transform: Transform) -> None:
        """Update object transform."""
        obj = self.scene.get_object(object_id)
        if obj:
            obj.transform = transform
            self._notify_update()
            
    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        """Set object visibility."""
        obj = self.scene.get_object(object_id)
        if obj:
            obj.visible = visible
            self._notify_update()
            
    def remove_object(self, object_id: str) -> None:
        """Remove object from scene."""
        self.scene.remove_object(object_id)
        self._notify_update()
        
    def clear_scene(self) -> None:
        """Clear all objects from scene."""
        self.scene.clear()
        self._notify_update()
        
    def set_camera_position(self, position: Tuple[float, float, float],
                           target: Optional[Tuple[float, float, float]] = None) -> None:
        """Set camera position and target."""
        self.scene.camera.position = np.array(position)
        if target is not None:
            self.scene.camera.target = np.array(target)
        self._notify_update()
        
    def orbit_camera(self, delta_azimuth: float, delta_elevation: float) -> None:
        """Orbit camera around target."""
        self.scene.camera.orbit(delta_azimuth, delta_elevation)
        self._notify_update()
        
    def zoom_camera(self, factor: float) -> None:
        """Zoom camera."""
        self.scene.camera.zoom(factor)
        self._notify_update()
        
    def fit_camera(self) -> None:
        """Fit camera to scene."""
        self.scene.fit_camera_to_scene()
        self._notify_update()
        
    def export_scene_json(self) -> str:
        """Export scene as JSON."""
        return self.scene.to_json()
        
    def export_scene_gltf(self) -> Dict[str, Any]:
        """Export scene in glTF format."""
        gltf = {
            'asset': {'version': '2.0', 'generator': 'MineralVision'},
            'scene': 0,
            'scenes': [{'nodes': list(range(len(self.scene.objects)))}],
            'nodes': [],
            'meshes': [],
            'accessors': [],
            'bufferViews': [],
            'buffers': []
        }
        
        buffer_data = bytearray()
        
        for i, (obj_id, obj) in enumerate(self.scene.objects.items()):
            # Node
            node = {
                'name': obj.name,
                'translation': obj.transform.position.tolist(),
                'rotation': [0, 0, 0, 1],  # Quaternion
                'scale': obj.transform.scale.tolist()
            }
            
            if obj.mesh is not None:
                node['mesh'] = i
                
                # Mesh
                mesh_data = {
                    'name': obj.mesh.name,
                    'primitives': [{
                        'attributes': {},
                        'mode': 4  # TRIANGLES
                    }]
                }
                
                # Add vertex data to buffer
                if len(obj.mesh.vertices) > 0:
                    vertices_bytes = obj.mesh.vertices.astype(np.float32).tobytes()
                    buffer_view_idx = len(gltf['bufferViews'])
                    
                    gltf['bufferViews'].append({
                        'buffer': 0,
                        'byteOffset': len(buffer_data),
                        'byteLength': len(vertices_bytes)
                    })
                    buffer_data.extend(vertices_bytes)
                    
                    accessor_idx = len(gltf['accessors'])
                    gltf['accessors'].append({
                        'bufferView': buffer_view_idx,
                        'componentType': 5126,  # FLOAT
                        'count': len(obj.mesh.vertices),
                        'type': 'VEC3',
                        'min': obj.mesh.vertices.min(axis=0).tolist(),
                        'max': obj.mesh.vertices.max(axis=0).tolist()
                    })
                    
                    mesh_data['primitives'][0]['attributes']['POSITION'] = accessor_idx
                    
                gltf['meshes'].append(mesh_data)
                
            gltf['nodes'].append(node)
            
        # Buffer
        gltf['buffers'].append({
            'byteLength': len(buffer_data),
            'uri': f"data:application/octet-stream;base64,{base64.b64encode(buffer_data).decode()}"
        })
        
        return gltf
        
    def render_to_image(self) -> np.ndarray:
        """
        Render scene to image (software rasterizer).
        
        Returns:
            RGB image as numpy array
        """
        # Simple software rasterizer for basic visualization
        image = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        depth_buffer = np.full((self.render_height, self.render_width), float('inf'))
        
        # Background
        bg = self.scene.background_color
        image[:, :] = [int(bg.r * 255), int(bg.g * 255), int(bg.b * 255)]
        
        # Get camera matrices
        aspect = self.render_width / self.render_height
        view = self.scene.camera.get_view_matrix()
        proj = self.scene.camera.get_projection_matrix(aspect)
        
        # Render each object
        for obj in self.scene.objects.values():
            if not obj.visible or obj.mesh is None:
                continue
                
            model = obj.get_world_transform()
            mvp = proj @ view @ model
            
            # Transform vertices
            vertices = obj.mesh.vertices
            if len(vertices) == 0:
                continue
                
            # Homogeneous coordinates
            v_homo = np.column_stack([vertices, np.ones(len(vertices))])
            v_clip = (mvp @ v_homo.T).T
            
            # Perspective divide
            w = v_clip[:, 3:4]
            w[w == 0] = 1e-10
            v_ndc = v_clip[:, :3] / w
            
            # Screen coordinates
            v_screen = np.zeros((len(vertices), 2))
            v_screen[:, 0] = (v_ndc[:, 0] + 1) * 0.5 * self.render_width
            v_screen[:, 1] = (1 - v_ndc[:, 1]) * 0.5 * self.render_height
            
            # Render triangles (simplified - just draw edges)
            color = [int(obj.color.r * 255), int(obj.color.g * 255), int(obj.color.b * 255)]
            
            if obj.render_mode == RenderMode.POINT_CLOUD:
                # Draw points
                for i, (sx, sy) in enumerate(v_screen):
                    x, y = int(sx), int(sy)
                    if 0 <= x < self.render_width and 0 <= y < self.render_height:
                        if len(obj.mesh.colors) > i:
                            c = obj.mesh.colors[i]
                            color = [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)]
                        image[y, x] = color
            else:
                # Draw wireframe
                for face in obj.mesh.indices:
                    for j in range(3):
                        i0, i1 = face[j], face[(j + 1) % 3]
                        x0, y0 = int(v_screen[i0, 0]), int(v_screen[i0, 1])
                        x1, y1 = int(v_screen[i1, 0]), int(v_screen[i1, 1])
                        
                        # Bresenham line
                        self._draw_line(image, x0, y0, x1, y1, color)
                        
        return image
        
    def _draw_line(self, image: np.ndarray, x0: int, y0: int, 
                   x1: int, y1: int, color: List[int]) -> None:
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < self.render_width and 0 <= y0 < self.render_height:
                image[y0, x0] = color
                
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
    def start_animation(self, update_fn: Callable[[float], None], 
                       fps: float = 30.0) -> None:
        """Start animation loop."""
        if self._animation_running:
            return
            
        self._animation_running = True
        
        def animation_loop():
            dt = 1.0 / fps
            while self._animation_running:
                try:
                    update_fn(dt)
                    self._notify_update()
                except Exception as e:
                    logger.error(f"Animation error: {e}")
                    
                import time
                time.sleep(dt)
                
        self._animation_thread = threading.Thread(target=animation_loop, daemon=True)
        self._animation_thread.start()
        
    def stop_animation(self) -> None:
        """Stop animation loop."""
        self._animation_running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=1.0)
            self._animation_thread = None


class DigitalTwinVisualizer:
    """
    High-level visualizer for digital twin data.
    """
    
    def __init__(self):
        self.engine = Visualization3DEngine()
        
    def visualize_terrain(self, heightmap: np.ndarray, 
                         width: float, height: float,
                         mineral_concentration: Optional[np.ndarray] = None) -> None:
        """Visualize terrain with optional mineral concentration overlay."""
        self.engine.add_terrain(heightmap, width, height, vertical_scale=1.0)
        
        if mineral_concentration is not None:
            # Add mineral deposits as spheres at high-concentration areas
            threshold = np.percentile(mineral_concentration, 90)
            rows, cols = mineral_concentration.shape
            
            for i in range(0, rows, 5):
                for j in range(0, cols, 5):
                    if mineral_concentration[i, j] > threshold:
                        x = (j / (cols - 1) - 0.5) * width
                        y = (i / (rows - 1) - 0.5) * height
                        z = heightmap[i, j] - 5  # Below surface
                        
                        conc = mineral_concentration[i, j]
                        self.engine.add_mineral_deposit(
                            f"deposit_{i}_{j}",
                            (x, y, z),
                            radius=2.0 * conc,
                            concentration=conc
                        )
                        
    def visualize_equipment(self, equipment_data: Dict[str, Dict[str, Any]]) -> None:
        """Visualize equipment from simulation data."""
        for eq_id, eq_info in equipment_data.items():
            position = eq_info.get('position', [0, 0, 0])
            eq_type = eq_info.get('type', 'generic')
            
            self.engine.add_equipment(eq_id, eq_type, tuple(position))
            
    def visualize_simulation_state(self, state: Dict[str, Any]) -> None:
        """Update visualization from simulation state."""
        # Update equipment positions
        if 'equipment' in state:
            for eq_id, eq_state in state['equipment'].items():
                if 'position' in eq_state:
                    self.engine.update_object_position(eq_id, tuple(eq_state['position']))
                    
        # Update physics bodies
        if 'physics' in state and 'bodies' in state['physics']:
            for body_id, body_state in state['physics']['bodies'].items():
                if 'position' in body_state:
                    self.engine.update_object_position(body_id, tuple(body_state['position']))
                    
    def export_visualization(self, format: str = 'json') -> Union[str, Dict]:
        """Export visualization data."""
        if format == 'json':
            return self.engine.export_scene_json()
        elif format == 'gltf':
            return self.engine.export_scene_gltf()
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def render_frame(self) -> np.ndarray:
        """Render current frame."""
        return self.engine.render_to_image()


def create_visualization_engine() -> Visualization3DEngine:
    """Factory function to create a visualization engine."""
    return Visualization3DEngine()


def create_digital_twin_visualizer() -> DigitalTwinVisualizer:
    """Factory function to create a digital twin visualizer."""
    return DigitalTwinVisualizer()
