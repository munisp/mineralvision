"""
3D Visualization Module for MineralVision Platform.

Comprehensive 3D visualization including:
1. Drillhole visualization with traces and intervals
2. Block model rendering with grade coloring
3. Surface/wireframe display
4. Cross-section planes
5. Interactive camera controls
6. Export to images and videos
7. PyVista/VTK integration layer
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np


class RenderMode(Enum):
    """Rendering modes."""
    SURFACE = "surface"
    WIREFRAME = "wireframe"
    POINTS = "points"
    EDGES = "edges"
    SURFACE_EDGES = "surface_edges"


class ColorMap(Enum):
    """Color map presets."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    JET = "jet"
    RAINBOW = "rainbow"
    COOLWARM = "coolwarm"
    SPECTRAL = "spectral"
    GEOLOGY = "geology"
    GRADE = "grade"


class CameraView(Enum):
    """Preset camera views."""
    ISOMETRIC = "isometric"
    TOP = "top"
    BOTTOM = "bottom"
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    CUSTOM = "custom"


@dataclass
class Point3D:
    """3D point."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Color:
    """RGB color."""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)
    
    def to_rgba(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)
    
    def to_hex(self) -> str:
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}"
    
    @classmethod
    def from_hex(cls, hex_color: str) -> 'Color':
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return cls(r, g, b)


@dataclass
class BoundingBox:
    """3D bounding box."""
    min_point: Point3D
    max_point: Point3D
    
    @property
    def center(self) -> Point3D:
        return Point3D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )
    
    @property
    def size(self) -> Point3D:
        return Point3D(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )
    
    @property
    def diagonal(self) -> float:
        s = self.size
        return math.sqrt(s.x**2 + s.y**2 + s.z**2)


@dataclass
class CameraSettings:
    """Camera configuration."""
    position: Point3D
    focal_point: Point3D
    view_up: Point3D = field(default_factory=lambda: Point3D(0, 0, 1))
    view_angle: float = 30.0
    parallel_projection: bool = False
    parallel_scale: float = 1.0
    clipping_range: Tuple[float, float] = (0.1, 10000.0)


@dataclass
class LightSettings:
    """Light configuration."""
    position: Point3D
    focal_point: Point3D
    intensity: float = 1.0
    color: Color = field(default_factory=lambda: Color(1, 1, 1))
    light_type: str = "scene"


@dataclass
class ScalarBarSettings:
    """Scalar bar (legend) configuration."""
    title: str
    n_labels: int = 5
    position: Tuple[float, float] = (0.85, 0.1)
    width: float = 0.1
    height: float = 0.8
    vertical: bool = True
    label_format: str = "%.2f"
    font_size: int = 12


@dataclass
class VisualizationObject:
    """Base class for visualization objects."""
    name: str
    visible: bool = True
    opacity: float = 1.0
    color: Optional[Color] = None
    render_mode: RenderMode = RenderMode.SURFACE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrillholeVisualization(VisualizationObject):
    """Drillhole visualization data."""
    hole_id: str
    collar: Point3D
    trace_points: List[Point3D]
    intervals: List[Dict[str, Any]] = field(default_factory=list)
    tube_radius: float = 1.0
    show_collar: bool = True
    show_labels: bool = True
    color_by: Optional[str] = None
    scalar_values: List[float] = field(default_factory=list)


@dataclass
class BlockModelVisualization(VisualizationObject):
    """Block model visualization data."""
    blocks: List[Dict[str, Any]]
    block_size: Tuple[float, float, float]
    color_by: Optional[str] = None
    scalar_range: Tuple[float, float] = (0, 1)
    show_edges: bool = False
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None


@dataclass
class SurfaceVisualization(VisualizationObject):
    """Surface/wireframe visualization data."""
    vertices: List[Point3D]
    triangles: List[Tuple[int, int, int]]
    vertex_scalars: Optional[List[float]] = None
    scalar_name: str = "values"
    show_edges: bool = False
    edge_color: Color = field(default_factory=lambda: Color(0, 0, 0))


@dataclass
class PointCloudVisualization(VisualizationObject):
    """Point cloud visualization data."""
    points: List[Point3D]
    scalars: Optional[List[float]] = None
    scalar_name: str = "values"
    point_size: float = 5.0
    render_points_as_spheres: bool = False


@dataclass
class SectionPlaneVisualization(VisualizationObject):
    """Section plane visualization data."""
    origin: Point3D
    normal: Point3D
    width: float
    height: float
    show_outline: bool = True


class ColorMapper:
    """Map scalar values to colors."""
    
    COLORMAPS = {
        "viridis": [(0.267, 0.004, 0.329), (0.282, 0.140, 0.458), (0.254, 0.265, 0.530),
                   (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
                   (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
                   (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)],
        "jet": [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1), (0.5, 1, 0.5),
               (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)],
        "grade": [(0.2, 0.6, 0.2), (0.4, 0.8, 0.2), (0.8, 0.8, 0.2),
                 (1.0, 0.6, 0.2), (1.0, 0.2, 0.2)],
        "geology": [(0.8, 0.6, 0.4), (0.6, 0.4, 0.2), (0.4, 0.4, 0.4),
                   (0.6, 0.6, 0.8), (0.4, 0.6, 0.4)]
    }
    
    def __init__(self, colormap: str = "viridis", 
                 scalar_range: Tuple[float, float] = (0, 1)):
        self.colormap_name = colormap
        self.scalar_range = scalar_range
        self.colors = self.COLORMAPS.get(colormap, self.COLORMAPS["viridis"])
    
    def map_value(self, value: float) -> Color:
        """Map a scalar value to a color."""
        min_val, max_val = self.scalar_range
        
        if max_val == min_val:
            t = 0.5
        else:
            t = (value - min_val) / (max_val - min_val)
        
        t = max(0, min(1, t))
        
        n_colors = len(self.colors)
        idx = t * (n_colors - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n_colors - 1)
        frac = idx - idx_low
        
        c1 = self.colors[idx_low]
        c2 = self.colors[idx_high]
        
        r = c1[0] + frac * (c2[0] - c1[0])
        g = c1[1] + frac * (c2[1] - c1[1])
        b = c1[2] + frac * (c2[2] - c1[2])
        
        return Color(r, g, b)
    
    def map_values(self, values: List[float]) -> List[Color]:
        """Map multiple values to colors."""
        return [self.map_value(v) for v in values]


class MeshBuilder:
    """Build mesh geometry for visualization."""
    
    def __init__(self):
        self.vertices: List[Point3D] = []
        self.triangles: List[Tuple[int, int, int]] = []
        self.scalars: List[float] = []
    
    def clear(self):
        """Clear all geometry."""
        self.vertices = []
        self.triangles = []
        self.scalars = []
    
    def add_box(self, center: Point3D, size: Tuple[float, float, float],
               scalar: float = 0.0) -> int:
        """Add a box to the mesh."""
        sx, sy, sz = size
        hx, hy, hz = sx/2, sy/2, sz/2
        
        base_idx = len(self.vertices)
        
        corners = [
            Point3D(center.x - hx, center.y - hy, center.z - hz),
            Point3D(center.x + hx, center.y - hy, center.z - hz),
            Point3D(center.x + hx, center.y + hy, center.z - hz),
            Point3D(center.x - hx, center.y + hy, center.z - hz),
            Point3D(center.x - hx, center.y - hy, center.z + hz),
            Point3D(center.x + hx, center.y - hy, center.z + hz),
            Point3D(center.x + hx, center.y + hy, center.z + hz),
            Point3D(center.x - hx, center.y + hy, center.z + hz),
        ]
        
        self.vertices.extend(corners)
        self.scalars.extend([scalar] * 8)
        
        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (2, 6, 7), (2, 7, 3),
            (0, 3, 7), (0, 7, 4),
            (1, 5, 6), (1, 6, 2),
        ]
        
        for f in faces:
            self.triangles.append((base_idx + f[0], base_idx + f[1], base_idx + f[2]))
        
        return base_idx
    
    def add_tube(self, points: List[Point3D], radius: float,
                n_sides: int = 8, scalars: Optional[List[float]] = None) -> int:
        """Add a tube along a path."""
        if len(points) < 2:
            return len(self.vertices)
        
        base_idx = len(self.vertices)
        
        for i, point in enumerate(points):
            if i == 0:
                direction = Point3D(
                    points[1].x - points[0].x,
                    points[1].y - points[0].y,
                    points[1].z - points[0].z
                )
            elif i == len(points) - 1:
                direction = Point3D(
                    points[-1].x - points[-2].x,
                    points[-1].y - points[-2].y,
                    points[-1].z - points[-2].z
                )
            else:
                direction = Point3D(
                    points[i+1].x - points[i-1].x,
                    points[i+1].y - points[i-1].y,
                    points[i+1].z - points[i-1].z
                )
            
            length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
            if length > 0:
                direction = Point3D(direction.x/length, direction.y/length, direction.z/length)
            else:
                direction = Point3D(0, 0, 1)
            
            if abs(direction.z) < 0.9:
                perp1 = Point3D(-direction.y, direction.x, 0)
            else:
                perp1 = Point3D(1, 0, 0)
            
            length = math.sqrt(perp1.x**2 + perp1.y**2 + perp1.z**2)
            if length > 0:
                perp1 = Point3D(perp1.x/length, perp1.y/length, perp1.z/length)
            
            perp2 = Point3D(
                direction.y * perp1.z - direction.z * perp1.y,
                direction.z * perp1.x - direction.x * perp1.z,
                direction.x * perp1.y - direction.y * perp1.x
            )
            
            scalar = scalars[i] if scalars and i < len(scalars) else 0.0
            
            for j in range(n_sides):
                angle = 2 * math.pi * j / n_sides
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                offset_x = radius * (cos_a * perp1.x + sin_a * perp2.x)
                offset_y = radius * (cos_a * perp1.y + sin_a * perp2.y)
                offset_z = radius * (cos_a * perp1.z + sin_a * perp2.z)
                
                self.vertices.append(Point3D(
                    point.x + offset_x,
                    point.y + offset_y,
                    point.z + offset_z
                ))
                self.scalars.append(scalar)
        
        for i in range(len(points) - 1):
            for j in range(n_sides):
                j_next = (j + 1) % n_sides
                
                v0 = base_idx + i * n_sides + j
                v1 = base_idx + i * n_sides + j_next
                v2 = base_idx + (i + 1) * n_sides + j_next
                v3 = base_idx + (i + 1) * n_sides + j
                
                self.triangles.append((v0, v1, v2))
                self.triangles.append((v0, v2, v3))
        
        return base_idx
    
    def add_sphere(self, center: Point3D, radius: float,
                  n_lat: int = 8, n_lon: int = 16, scalar: float = 0.0) -> int:
        """Add a sphere to the mesh."""
        base_idx = len(self.vertices)
        
        self.vertices.append(Point3D(center.x, center.y, center.z + radius))
        self.scalars.append(scalar)
        
        for i in range(1, n_lat):
            phi = math.pi * i / n_lat
            z = center.z + radius * math.cos(phi)
            r_xy = radius * math.sin(phi)
            
            for j in range(n_lon):
                theta = 2 * math.pi * j / n_lon
                x = center.x + r_xy * math.cos(theta)
                y = center.y + r_xy * math.sin(theta)
                
                self.vertices.append(Point3D(x, y, z))
                self.scalars.append(scalar)
        
        self.vertices.append(Point3D(center.x, center.y, center.z - radius))
        self.scalars.append(scalar)
        
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            self.triangles.append((base_idx, base_idx + 1 + j, base_idx + 1 + j_next))
        
        for i in range(n_lat - 2):
            for j in range(n_lon):
                j_next = (j + 1) % n_lon
                
                v0 = base_idx + 1 + i * n_lon + j
                v1 = base_idx + 1 + i * n_lon + j_next
                v2 = base_idx + 1 + (i + 1) * n_lon + j_next
                v3 = base_idx + 1 + (i + 1) * n_lon + j
                
                self.triangles.append((v0, v1, v2))
                self.triangles.append((v0, v2, v3))
        
        bottom_idx = base_idx + 1 + (n_lat - 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            v0 = base_idx + 1 + (n_lat - 2) * n_lon + j
            v1 = base_idx + 1 + (n_lat - 2) * n_lon + j_next
            self.triangles.append((v0, v1, bottom_idx))
        
        return base_idx
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        vertices = np.array([[v.x, v.y, v.z] for v in self.vertices])
        triangles = np.array(self.triangles)
        scalars = np.array(self.scalars)
        
        return vertices, triangles, scalars


class Scene3D:
    """3D scene manager."""
    
    def __init__(self, name: str = "MineralVision Scene"):
        self.name = name
        self.objects: Dict[str, VisualizationObject] = {}
        self.camera = CameraSettings(
            position=Point3D(1000, 1000, 1000),
            focal_point=Point3D(0, 0, 0)
        )
        self.lights: List[LightSettings] = []
        self.background_color = Color(0.1, 0.1, 0.2)
        self.bounding_box: Optional[BoundingBox] = None
        self.colormap = ColorMapper()
    
    def add_object(self, obj: VisualizationObject):
        """Add an object to the scene."""
        self.objects[obj.name] = obj
        self._update_bounding_box()
    
    def remove_object(self, name: str):
        """Remove an object from the scene."""
        if name in self.objects:
            del self.objects[name]
            self._update_bounding_box()
    
    def get_object(self, name: str) -> Optional[VisualizationObject]:
        """Get an object by name."""
        return self.objects.get(name)
    
    def set_object_visibility(self, name: str, visible: bool):
        """Set object visibility."""
        if name in self.objects:
            self.objects[name].visible = visible
    
    def set_camera_view(self, view: CameraView):
        """Set camera to a preset view."""
        if self.bounding_box is None:
            return
        
        center = self.bounding_box.center
        dist = self.bounding_box.diagonal * 1.5
        
        if view == CameraView.ISOMETRIC:
            self.camera.position = Point3D(
                center.x + dist * 0.577,
                center.y + dist * 0.577,
                center.z + dist * 0.577
            )
            self.camera.view_up = Point3D(0, 0, 1)
        elif view == CameraView.TOP:
            self.camera.position = Point3D(center.x, center.y, center.z + dist)
            self.camera.view_up = Point3D(0, 1, 0)
        elif view == CameraView.BOTTOM:
            self.camera.position = Point3D(center.x, center.y, center.z - dist)
            self.camera.view_up = Point3D(0, 1, 0)
        elif view == CameraView.FRONT:
            self.camera.position = Point3D(center.x, center.y - dist, center.z)
            self.camera.view_up = Point3D(0, 0, 1)
        elif view == CameraView.BACK:
            self.camera.position = Point3D(center.x, center.y + dist, center.z)
            self.camera.view_up = Point3D(0, 0, 1)
        elif view == CameraView.LEFT:
            self.camera.position = Point3D(center.x - dist, center.y, center.z)
            self.camera.view_up = Point3D(0, 0, 1)
        elif view == CameraView.RIGHT:
            self.camera.position = Point3D(center.x + dist, center.y, center.z)
            self.camera.view_up = Point3D(0, 0, 1)
        
        self.camera.focal_point = center
    
    def fit_camera_to_scene(self):
        """Fit camera to show all objects."""
        if self.bounding_box:
            self.set_camera_view(CameraView.ISOMETRIC)
    
    def _update_bounding_box(self):
        """Update scene bounding box."""
        if not self.objects:
            self.bounding_box = None
            return
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in self.objects.values():
            if isinstance(obj, DrillholeVisualization):
                for p in obj.trace_points:
                    min_x = min(min_x, p.x)
                    min_y = min(min_y, p.y)
                    min_z = min(min_z, p.z)
                    max_x = max(max_x, p.x)
                    max_y = max(max_y, p.y)
                    max_z = max(max_z, p.z)
            elif isinstance(obj, SurfaceVisualization):
                for v in obj.vertices:
                    min_x = min(min_x, v.x)
                    min_y = min(min_y, v.y)
                    min_z = min(min_z, v.z)
                    max_x = max(max_x, v.x)
                    max_y = max(max_y, v.y)
                    max_z = max(max_z, v.z)
            elif isinstance(obj, PointCloudVisualization):
                for p in obj.points:
                    min_x = min(min_x, p.x)
                    min_y = min(min_y, p.y)
                    min_z = min(min_z, p.z)
                    max_x = max(max_x, p.x)
                    max_y = max(max_y, p.y)
                    max_z = max(max_z, p.z)
            elif isinstance(obj, BlockModelVisualization):
                for block in obj.blocks:
                    x = block.get('x', 0)
                    y = block.get('y', 0)
                    z = block.get('z', 0)
                    min_x = min(min_x, x - obj.block_size[0]/2)
                    min_y = min(min_y, y - obj.block_size[1]/2)
                    min_z = min(min_z, z - obj.block_size[2]/2)
                    max_x = max(max_x, x + obj.block_size[0]/2)
                    max_y = max(max_y, y + obj.block_size[1]/2)
                    max_z = max(max_z, z + obj.block_size[2]/2)
        
        if min_x != float('inf'):
            self.bounding_box = BoundingBox(
                Point3D(min_x, min_y, min_z),
                Point3D(max_x, max_y, max_z)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export scene to dictionary."""
        return {
            "name": self.name,
            "objects": list(self.objects.keys()),
            "camera": {
                "position": self.camera.position.to_tuple(),
                "focal_point": self.camera.focal_point.to_tuple(),
                "view_up": self.camera.view_up.to_tuple(),
                "view_angle": self.camera.view_angle
            },
            "background_color": self.background_color.to_hex(),
            "bounding_box": {
                "min": self.bounding_box.min_point.to_tuple(),
                "max": self.bounding_box.max_point.to_tuple()
            } if self.bounding_box else None
        }


class PyVistaRenderer:
    """
    PyVista-based renderer for 3D visualization.
    
    This class provides an abstraction layer for PyVista/VTK rendering.
    When PyVista is available, it uses native PyVista functions.
    Otherwise, it generates mesh data that can be exported.
    """
    
    def __init__(self, scene: Scene3D):
        self.scene = scene
        self.mesh_builder = MeshBuilder()
        self.pyvista_available = self._check_pyvista()
        self.plotter = None
    
    def _check_pyvista(self) -> bool:
        """Check if PyVista is available."""
        try:
            import pyvista
            return True
        except ImportError:
            return False
    
    def initialize_plotter(self, off_screen: bool = False, 
                          window_size: Tuple[int, int] = (1024, 768)):
        """Initialize PyVista plotter."""
        if not self.pyvista_available:
            return None
        
        import pyvista as pv
        
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        self.plotter.set_background(self.scene.background_color.to_tuple())
        
        return self.plotter
    
    def render_drillhole(self, dh: DrillholeVisualization):
        """Render a drillhole."""
        if not dh.trace_points:
            return None
        
        self.mesh_builder.clear()
        
        scalars = dh.scalar_values if dh.scalar_values else None
        self.mesh_builder.add_tube(dh.trace_points, dh.tube_radius, scalars=scalars)
        
        if dh.show_collar:
            self.mesh_builder.add_sphere(dh.collar, dh.tube_radius * 2)
        
        vertices, triangles, scalars = self.mesh_builder.to_arrays()
        
        if self.pyvista_available and self.plotter:
            import pyvista as pv
            
            faces = np.hstack([[3] + list(t) for t in triangles])
            mesh = pv.PolyData(vertices, faces)
            
            if len(scalars) > 0:
                mesh.point_data['values'] = scalars
            
            self.plotter.add_mesh(
                mesh,
                scalars='values' if dh.color_by else None,
                opacity=dh.opacity,
                show_edges=False,
                name=dh.name
            )
        
        return {
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "scalars": scalars.tolist()
        }
    
    def render_block_model(self, bm: BlockModelVisualization):
        """Render a block model."""
        if not bm.blocks:
            return None
        
        self.mesh_builder.clear()
        
        for block in bm.blocks:
            if bm.threshold_min is not None:
                value = block.get(bm.color_by, 0) if bm.color_by else 0
                if value < bm.threshold_min:
                    continue
            
            if bm.threshold_max is not None:
                value = block.get(bm.color_by, 0) if bm.color_by else 0
                if value > bm.threshold_max:
                    continue
            
            center = Point3D(
                block.get('x', 0),
                block.get('y', 0),
                block.get('z', 0)
            )
            
            scalar = block.get(bm.color_by, 0) if bm.color_by else 0
            self.mesh_builder.add_box(center, bm.block_size, scalar)
        
        vertices, triangles, scalars = self.mesh_builder.to_arrays()
        
        if self.pyvista_available and self.plotter:
            import pyvista as pv
            
            faces = np.hstack([[3] + list(t) for t in triangles])
            mesh = pv.PolyData(vertices, faces)
            
            if len(scalars) > 0:
                mesh.point_data['values'] = scalars
            
            self.plotter.add_mesh(
                mesh,
                scalars='values' if bm.color_by else None,
                clim=bm.scalar_range,
                opacity=bm.opacity,
                show_edges=bm.show_edges,
                name=bm.name
            )
        
        return {
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "scalars": scalars.tolist()
        }
    
    def render_surface(self, surf: SurfaceVisualization):
        """Render a surface."""
        if not surf.vertices or not surf.triangles:
            return None
        
        vertices = np.array([[v.x, v.y, v.z] for v in surf.vertices])
        triangles = np.array(surf.triangles)
        scalars = np.array(surf.vertex_scalars) if surf.vertex_scalars else None
        
        if self.pyvista_available and self.plotter:
            import pyvista as pv
            
            faces = np.hstack([[3] + list(t) for t in triangles])
            mesh = pv.PolyData(vertices, faces)
            
            if scalars is not None:
                mesh.point_data[surf.scalar_name] = scalars
            
            self.plotter.add_mesh(
                mesh,
                scalars=surf.scalar_name if scalars is not None else None,
                opacity=surf.opacity,
                show_edges=surf.show_edges,
                edge_color=surf.edge_color.to_tuple(),
                color=surf.color.to_tuple() if surf.color else None,
                name=surf.name
            )
        
        return {
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "scalars": scalars.tolist() if scalars is not None else None
        }
    
    def render_point_cloud(self, pc: PointCloudVisualization):
        """Render a point cloud."""
        if not pc.points:
            return None
        
        points = np.array([[p.x, p.y, p.z] for p in pc.points])
        scalars = np.array(pc.scalars) if pc.scalars else None
        
        if self.pyvista_available and self.plotter:
            import pyvista as pv
            
            cloud = pv.PolyData(points)
            
            if scalars is not None:
                cloud.point_data[pc.scalar_name] = scalars
            
            self.plotter.add_mesh(
                cloud,
                scalars=pc.scalar_name if scalars is not None else None,
                point_size=pc.point_size,
                render_points_as_spheres=pc.render_points_as_spheres,
                opacity=pc.opacity,
                name=pc.name
            )
        
        return {
            "points": points.tolist(),
            "scalars": scalars.tolist() if scalars is not None else None
        }
    
    def render_scene(self):
        """Render all objects in the scene."""
        results = {}
        
        for name, obj in self.scene.objects.items():
            if not obj.visible:
                continue
            
            if isinstance(obj, DrillholeVisualization):
                results[name] = self.render_drillhole(obj)
            elif isinstance(obj, BlockModelVisualization):
                results[name] = self.render_block_model(obj)
            elif isinstance(obj, SurfaceVisualization):
                results[name] = self.render_surface(obj)
            elif isinstance(obj, PointCloudVisualization):
                results[name] = self.render_point_cloud(obj)
        
        return results
    
    def set_camera(self):
        """Set camera from scene settings."""
        if self.pyvista_available and self.plotter:
            self.plotter.camera_position = [
                self.scene.camera.position.to_tuple(),
                self.scene.camera.focal_point.to_tuple(),
                self.scene.camera.view_up.to_tuple()
            ]
    
    def add_scalar_bar(self, settings: ScalarBarSettings):
        """Add a scalar bar (legend)."""
        if self.pyvista_available and self.plotter:
            self.plotter.add_scalar_bar(
                title=settings.title,
                n_labels=settings.n_labels,
                position_x=settings.position[0],
                position_y=settings.position[1],
                width=settings.width,
                height=settings.height,
                vertical=settings.vertical,
                fmt=settings.label_format,
                font_size=settings.font_size
            )
    
    def add_axes(self):
        """Add coordinate axes."""
        if self.pyvista_available and self.plotter:
            self.plotter.add_axes()
    
    def show(self):
        """Show the visualization."""
        if self.pyvista_available and self.plotter:
            self.plotter.show()
    
    def screenshot(self, filename: str):
        """Save screenshot."""
        if self.pyvista_available and self.plotter:
            self.plotter.screenshot(filename)
    
    def export_html(self, filename: str):
        """Export to interactive HTML."""
        if self.pyvista_available and self.plotter:
            self.plotter.export_html(filename)
    
    def close(self):
        """Close the plotter."""
        if self.pyvista_available and self.plotter:
            self.plotter.close()


class VisualizationWorkflow:
    """
    Complete 3D visualization workflow.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.scene = Scene3D(project_name)
        self.renderer: Optional[PyVistaRenderer] = None
    
    def add_drillholes(self, drillholes: List[Dict[str, Any]],
                      color_by: Optional[str] = None,
                      tube_radius: float = 1.0):
        """Add drillholes to the scene."""
        for dh in drillholes:
            hole_id = dh.get('hole_id', f"DH_{len(self.scene.objects)}")
            
            collar = Point3D(
                dh.get('collar_x', dh.get('easting', 0)),
                dh.get('collar_y', dh.get('northing', 0)),
                dh.get('collar_z', dh.get('elevation', 0))
            )
            
            trace = dh.get('trace', [])
            trace_points = [
                Point3D(p.get('x', 0), p.get('y', 0), p.get('z', 0))
                for p in trace
            ]
            
            if not trace_points:
                trace_points = [collar]
            
            scalars = []
            if color_by:
                for p in trace:
                    scalars.append(p.get(color_by, 0))
            
            vis = DrillholeVisualization(
                name=hole_id,
                hole_id=hole_id,
                collar=collar,
                trace_points=trace_points,
                tube_radius=tube_radius,
                color_by=color_by,
                scalar_values=scalars
            )
            
            self.scene.add_object(vis)
    
    def add_block_model(self, blocks: List[Dict[str, Any]],
                       block_size: Tuple[float, float, float],
                       name: str = "BlockModel",
                       color_by: Optional[str] = None,
                       scalar_range: Optional[Tuple[float, float]] = None,
                       threshold_min: Optional[float] = None):
        """Add a block model to the scene."""
        if scalar_range is None and color_by:
            values = [b.get(color_by, 0) for b in blocks]
            if values:
                scalar_range = (min(values), max(values))
            else:
                scalar_range = (0, 1)
        
        vis = BlockModelVisualization(
            name=name,
            blocks=blocks,
            block_size=block_size,
            color_by=color_by,
            scalar_range=scalar_range or (0, 1),
            threshold_min=threshold_min
        )
        
        self.scene.add_object(vis)
    
    def add_surface(self, vertices: List[Dict[str, float]],
                   triangles: List[Tuple[int, int, int]],
                   name: str = "Surface",
                   scalars: Optional[List[float]] = None,
                   color: Optional[str] = None):
        """Add a surface to the scene."""
        vertex_points = [
            Point3D(v.get('x', 0), v.get('y', 0), v.get('z', 0))
            for v in vertices
        ]
        
        vis = SurfaceVisualization(
            name=name,
            vertices=vertex_points,
            triangles=triangles,
            vertex_scalars=scalars,
            color=Color.from_hex(color) if color else None
        )
        
        self.scene.add_object(vis)
    
    def add_points(self, points: List[Dict[str, float]],
                  name: str = "Points",
                  scalars: Optional[List[float]] = None,
                  point_size: float = 5.0):
        """Add point cloud to the scene."""
        point_objects = [
            Point3D(p.get('x', 0), p.get('y', 0), p.get('z', 0))
            for p in points
        ]
        
        vis = PointCloudVisualization(
            name=name,
            points=point_objects,
            scalars=scalars,
            point_size=point_size
        )
        
        self.scene.add_object(vis)
    
    def set_camera(self, view: str = "isometric"):
        """Set camera view."""
        view_enum = CameraView(view) if view in [v.value for v in CameraView] else CameraView.ISOMETRIC
        self.scene.set_camera_view(view_enum)
    
    def render(self, off_screen: bool = False,
              window_size: Tuple[int, int] = (1024, 768)) -> Dict[str, Any]:
        """Render the scene."""
        self.renderer = PyVistaRenderer(self.scene)
        self.renderer.initialize_plotter(off_screen, window_size)
        
        results = self.renderer.render_scene()
        self.renderer.set_camera()
        self.renderer.add_axes()
        
        return results
    
    def show(self):
        """Show interactive visualization."""
        if self.renderer:
            self.renderer.show()
    
    def screenshot(self, filename: str):
        """Save screenshot."""
        if self.renderer:
            self.renderer.screenshot(filename)
    
    def export_html(self, filename: str):
        """Export to interactive HTML."""
        if self.renderer:
            self.renderer.export_html(filename)
    
    def close(self):
        """Close visualization."""
        if self.renderer:
            self.renderer.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        return {
            "project": self.project_name,
            "scene": self.scene.to_dict(),
            "n_objects": len(self.scene.objects),
            "pyvista_available": self.renderer.pyvista_available if self.renderer else False
        }


def create_visualization_workflow(project_name: str = "default") -> VisualizationWorkflow:
    """Factory function to create a visualization workflow."""
    return VisualizationWorkflow(project_name)
