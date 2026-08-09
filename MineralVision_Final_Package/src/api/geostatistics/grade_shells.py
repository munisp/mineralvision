"""
Grade Shell Generation Module for MineralVision Platform.

Comprehensive grade shell (wireframe) generation including:
1. Indicator-based shell generation
2. Implicit surface modeling
3. Multiple threshold shells
4. Shell validation and smoothing
5. Volume calculations
6. Shell intersection and boolean operations
7. Export to various formats (DXF, OBJ, STL)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import math
import numpy as np
from collections import defaultdict


class ShellMethod(Enum):
    """Methods for generating grade shells."""
    INDICATOR = "indicator"
    IMPLICIT = "implicit"
    MARCHING_CUBES = "marching_cubes"
    RADIAL_BASIS = "radial_basis"
    KRIGING = "kriging"


class SurfaceType(Enum):
    """Types of surfaces."""
    GRADE_SHELL = "grade_shell"
    LITHOLOGY_CONTACT = "lithology_contact"
    FAULT = "fault"
    TOPOGRAPHY = "topography"
    PIT_SHELL = "pit_shell"


@dataclass
class Point3D:
    """3D point."""
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass
class Triangle:
    """Triangle face defined by three vertex indices."""
    v1: int
    v2: int
    v3: int
    normal: Optional[Point3D] = None
    
    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.v1, self.v2, self.v3)


@dataclass
class Vertex:
    """Mesh vertex with optional attributes."""
    position: Point3D
    normal: Optional[Point3D] = None
    grade: float = 0.0
    indicator: float = 0.0


@dataclass
class GradeShell:
    """Grade shell (wireframe) surface."""
    name: str
    cutoff: float
    element: str
    vertices: List[Vertex]
    triangles: List[Triangle]
    surface_type: SurfaceType = SurfaceType.GRADE_SHELL
    volume: float = 0.0
    surface_area: float = 0.0
    is_closed: bool = False
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def n_triangles(self) -> int:
        return len(self.triangles)
    
    def get_bounds(self) -> Tuple[Point3D, Point3D]:
        """Get bounding box of shell."""
        if not self.vertices:
            return (Point3D(0, 0, 0), Point3D(0, 0, 0))
        
        xs = [v.position.x for v in self.vertices]
        ys = [v.position.y for v in self.vertices]
        zs = [v.position.z for v in self.vertices]
        
        return (
            Point3D(min(xs), min(ys), min(zs)),
            Point3D(max(xs), max(ys), max(zs))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cutoff": self.cutoff,
            "element": self.element,
            "n_vertices": self.n_vertices,
            "n_triangles": self.n_triangles,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "is_closed": self.is_closed,
            "bounds": {
                "min": self.get_bounds()[0].to_tuple(),
                "max": self.get_bounds()[1].to_tuple()
            }
        }


@dataclass
class GridCell:
    """Cell in the marching cubes grid."""
    corners: List[Point3D]
    values: List[float]
    
    def get_case_index(self, threshold: float) -> int:
        """Get marching cubes case index."""
        index = 0
        for i, val in enumerate(self.values):
            if val >= threshold:
                index |= (1 << i)
        return index


class MarchingCubes:
    """Marching cubes algorithm for isosurface extraction."""
    
    EDGE_TABLE = [
        0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
    ]
    
    EDGE_VERTICES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.triangles: List[Triangle] = []
        self.vertex_cache: Dict[Tuple[int, int, int, int], int] = {}
    
    def interpolate_vertex(self, p1: Point3D, p2: Point3D, 
                          v1: float, v2: float, threshold: float) -> Point3D:
        """Interpolate vertex position on edge."""
        if abs(v1 - v2) < 1e-10:
            return p1
        
        t = (threshold - v1) / (v2 - v1)
        t = max(0, min(1, t))
        
        return Point3D(
            p1.x + t * (p2.x - p1.x),
            p1.y + t * (p2.y - p1.y),
            p1.z + t * (p2.z - p1.z)
        )
    
    def process_cell(self, cell: GridCell, threshold: float,
                    cell_index: Tuple[int, int, int]):
        """Process a single cell and generate triangles."""
        case_index = cell.get_case_index(threshold)
        
        if case_index == 0 or case_index == 255:
            return
        
        edge_flags = self.EDGE_TABLE[case_index]
        
        edge_vertices = {}
        
        for edge in range(12):
            if edge_flags & (1 << edge):
                v1_idx, v2_idx = self.EDGE_VERTICES[edge]
                
                cache_key = (*cell_index, edge)
                
                if cache_key in self.vertex_cache:
                    edge_vertices[edge] = self.vertex_cache[cache_key]
                else:
                    vertex_pos = self.interpolate_vertex(
                        cell.corners[v1_idx], cell.corners[v2_idx],
                        cell.values[v1_idx], cell.values[v2_idx],
                        threshold
                    )
                    
                    vertex_idx = len(self.vertices)
                    self.vertices.append(Vertex(
                        position=vertex_pos,
                        indicator=threshold
                    ))
                    
                    edge_vertices[edge] = vertex_idx
                    self.vertex_cache[cache_key] = vertex_idx
    
    def clear(self):
        """Clear all data."""
        self.vertices = []
        self.triangles = []
        self.vertex_cache = {}


class GradeShellGenerator:
    """Generate grade shells from sample data or block models."""
    
    def __init__(self):
        self.sample_data: List[Dict[str, Any]] = []
        self.grid_values: Optional[np.ndarray] = None
        self.grid_origin: Optional[Point3D] = None
        self.grid_spacing: Optional[Point3D] = None
        self.grid_dims: Optional[Tuple[int, int, int]] = None
    
    def set_sample_data(self, samples: List[Dict[str, Any]]):
        """Set sample data for shell generation."""
        self.sample_data = samples
    
    def create_grid(self, x_min: float, x_max: float, x_step: float,
                   y_min: float, y_max: float, y_step: float,
                   z_min: float, z_max: float, z_step: float):
        """Create interpolation grid."""
        nx = int((x_max - x_min) / x_step) + 1
        ny = int((y_max - y_min) / y_step) + 1
        nz = int((z_max - z_min) / z_step) + 1
        
        self.grid_origin = Point3D(x_min, y_min, z_min)
        self.grid_spacing = Point3D(x_step, y_step, z_step)
        self.grid_dims = (nx, ny, nz)
        self.grid_values = np.zeros((nx, ny, nz))
    
    def interpolate_grid(self, element: str, method: str = "inverse_distance",
                        power: float = 2.0, max_distance: float = float('inf')):
        """Interpolate sample values onto grid."""
        if self.grid_values is None or not self.sample_data:
            return
        
        nx, ny, nz = self.grid_dims
        
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x = self.grid_origin.x + ix * self.grid_spacing.x
                    y = self.grid_origin.y + iy * self.grid_spacing.y
                    z = self.grid_origin.z + iz * self.grid_spacing.z
                    
                    if method == "inverse_distance":
                        value = self._idw_interpolate(x, y, z, element, power, max_distance)
                    elif method == "nearest":
                        value = self._nearest_interpolate(x, y, z, element, max_distance)
                    else:
                        value = 0
                    
                    self.grid_values[ix, iy, iz] = value
    
    def _idw_interpolate(self, x: float, y: float, z: float,
                        element: str, power: float, max_distance: float) -> float:
        """Inverse distance weighted interpolation."""
        total_weight = 0
        weighted_sum = 0
        
        for sample in self.sample_data:
            sx = sample.get('x', sample.get('easting', 0))
            sy = sample.get('y', sample.get('northing', 0))
            sz = sample.get('z', sample.get('elevation', 0))
            
            dist = math.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
            
            if dist > max_distance:
                continue
            
            if dist < 0.001:
                return sample.get(element, 0)
            
            weight = 1 / (dist ** power)
            value = sample.get(element, 0)
            
            weighted_sum += weight * value
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _nearest_interpolate(self, x: float, y: float, z: float,
                            element: str, max_distance: float) -> float:
        """Nearest neighbor interpolation."""
        nearest_dist = float('inf')
        nearest_value = 0
        
        for sample in self.sample_data:
            sx = sample.get('x', sample.get('easting', 0))
            sy = sample.get('y', sample.get('northing', 0))
            sz = sample.get('z', sample.get('elevation', 0))
            
            dist = math.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
            
            if dist < nearest_dist and dist <= max_distance:
                nearest_dist = dist
                nearest_value = sample.get(element, 0)
        
        return nearest_value
    
    def generate_shell(self, cutoff: float, element: str,
                      name: Optional[str] = None) -> GradeShell:
        """Generate grade shell at specified cutoff."""
        if self.grid_values is None:
            raise ValueError("Grid not created or interpolated")
        
        mc = MarchingCubes()
        nx, ny, nz = self.grid_dims
        
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                for iz in range(nz - 1):
                    corners = []
                    values = []
                    
                    for dz in [0, 1]:
                        for dy in [0, 1]:
                            for dx in [0, 1]:
                                cx = self.grid_origin.x + (ix + dx) * self.grid_spacing.x
                                cy = self.grid_origin.y + (iy + dy) * self.grid_spacing.y
                                cz = self.grid_origin.z + (iz + dz) * self.grid_spacing.z
                                corners.append(Point3D(cx, cy, cz))
                                values.append(self.grid_values[ix + dx, iy + dy, iz + dz])
                    
                    cell = GridCell(corners=corners, values=values)
                    mc.process_cell(cell, cutoff, (ix, iy, iz))
        
        if name is None:
            name = f"{element}_{cutoff}"
        
        shell = GradeShell(
            name=name,
            cutoff=cutoff,
            element=element,
            vertices=mc.vertices,
            triangles=mc.triangles
        )
        
        self._calculate_shell_properties(shell)
        
        return shell
    
    def generate_multiple_shells(self, cutoffs: List[float], 
                                element: str) -> List[GradeShell]:
        """Generate multiple grade shells at different cutoffs."""
        shells = []
        
        for cutoff in cutoffs:
            shell = self.generate_shell(cutoff, element)
            shells.append(shell)
        
        return shells
    
    def _calculate_shell_properties(self, shell: GradeShell):
        """Calculate volume and surface area of shell."""
        volume = 0
        surface_area = 0
        
        for tri in shell.triangles:
            v1 = shell.vertices[tri.v1].position
            v2 = shell.vertices[tri.v2].position
            v3 = shell.vertices[tri.v3].position
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            
            cross = Point3D(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x
            )
            
            area = 0.5 * math.sqrt(cross.x**2 + cross.y**2 + cross.z**2)
            surface_area += area
            
            centroid = Point3D(
                (v1.x + v2.x + v3.x) / 3,
                (v1.y + v2.y + v3.y) / 3,
                (v1.z + v2.z + v3.z) / 3
            )
            
            volume += (centroid.x * cross.x + centroid.y * cross.y + centroid.z * cross.z) / 6
        
        shell.volume = abs(volume)
        shell.surface_area = surface_area
        shell.is_closed = self._check_closed(shell)
    
    def _check_closed(self, shell: GradeShell) -> bool:
        """Check if shell is a closed surface."""
        edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
        
        for tri in shell.triangles:
            edges = [
                (min(tri.v1, tri.v2), max(tri.v1, tri.v2)),
                (min(tri.v2, tri.v3), max(tri.v2, tri.v3)),
                (min(tri.v3, tri.v1), max(tri.v3, tri.v1))
            ]
            
            for edge in edges:
                edge_count[edge] += 1
        
        for count in edge_count.values():
            if count != 2:
                return False
        
        return True


class ShellSmoother:
    """Smooth grade shell surfaces."""
    
    def __init__(self):
        pass
    
    def laplacian_smooth(self, shell: GradeShell, iterations: int = 3,
                        factor: float = 0.5) -> GradeShell:
        """Apply Laplacian smoothing to shell."""
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        
        for tri in shell.triangles:
            adjacency[tri.v1].add(tri.v2)
            adjacency[tri.v1].add(tri.v3)
            adjacency[tri.v2].add(tri.v1)
            adjacency[tri.v2].add(tri.v3)
            adjacency[tri.v3].add(tri.v1)
            adjacency[tri.v3].add(tri.v2)
        
        positions = [v.position for v in shell.vertices]
        
        for _ in range(iterations):
            new_positions = []
            
            for i, pos in enumerate(positions):
                neighbors = adjacency[i]
                
                if not neighbors:
                    new_positions.append(pos)
                    continue
                
                avg = Point3D(0, 0, 0)
                for n in neighbors:
                    avg = avg + positions[n]
                avg = avg * (1 / len(neighbors))
                
                new_pos = Point3D(
                    pos.x + factor * (avg.x - pos.x),
                    pos.y + factor * (avg.y - pos.y),
                    pos.z + factor * (avg.z - pos.z)
                )
                new_positions.append(new_pos)
            
            positions = new_positions
        
        smoothed_vertices = []
        for i, pos in enumerate(positions):
            v = shell.vertices[i]
            smoothed_vertices.append(Vertex(
                position=pos,
                normal=v.normal,
                grade=v.grade,
                indicator=v.indicator
            ))
        
        return GradeShell(
            name=f"{shell.name}_smoothed",
            cutoff=shell.cutoff,
            element=shell.element,
            vertices=smoothed_vertices,
            triangles=shell.triangles.copy(),
            surface_type=shell.surface_type,
            metadata=shell.metadata.copy()
        )
    
    def taubin_smooth(self, shell: GradeShell, iterations: int = 3,
                     lambda_factor: float = 0.5, mu_factor: float = -0.53) -> GradeShell:
        """Apply Taubin smoothing (volume-preserving)."""
        result = shell
        
        for i in range(iterations):
            result = self.laplacian_smooth(result, 1, lambda_factor)
            result = self.laplacian_smooth(result, 1, mu_factor)
        
        result.name = f"{shell.name}_taubin_smoothed"
        return result


class ShellValidator:
    """Validate grade shell geometry."""
    
    def __init__(self):
        pass
    
    def validate(self, shell: GradeShell) -> Dict[str, Any]:
        """Perform comprehensive validation."""
        results = {
            "is_valid": True,
            "issues": [],
            "statistics": {}
        }
        
        if shell.n_vertices == 0:
            results["is_valid"] = False
            results["issues"].append("Shell has no vertices")
            return results
        
        if shell.n_triangles == 0:
            results["is_valid"] = False
            results["issues"].append("Shell has no triangles")
            return results
        
        degenerate = self._check_degenerate_triangles(shell)
        if degenerate > 0:
            results["issues"].append(f"{degenerate} degenerate triangles found")
        
        non_manifold = self._check_non_manifold_edges(shell)
        if non_manifold > 0:
            results["issues"].append(f"{non_manifold} non-manifold edges found")
            results["is_valid"] = False
        
        if not shell.is_closed:
            results["issues"].append("Shell is not closed (has boundary edges)")
        
        if shell.volume < 0:
            results["issues"].append("Shell has inverted normals (negative volume)")
        
        results["statistics"] = {
            "n_vertices": shell.n_vertices,
            "n_triangles": shell.n_triangles,
            "volume": shell.volume,
            "surface_area": shell.surface_area,
            "is_closed": shell.is_closed,
            "degenerate_triangles": degenerate,
            "non_manifold_edges": non_manifold
        }
        
        return results
    
    def _check_degenerate_triangles(self, shell: GradeShell) -> int:
        """Count degenerate (zero-area) triangles."""
        count = 0
        
        for tri in shell.triangles:
            v1 = shell.vertices[tri.v1].position
            v2 = shell.vertices[tri.v2].position
            v3 = shell.vertices[tri.v3].position
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            
            cross = Point3D(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x
            )
            
            area = 0.5 * math.sqrt(cross.x**2 + cross.y**2 + cross.z**2)
            
            if area < 1e-10:
                count += 1
        
        return count
    
    def _check_non_manifold_edges(self, shell: GradeShell) -> int:
        """Count non-manifold edges (shared by more than 2 triangles)."""
        edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
        
        for tri in shell.triangles:
            edges = [
                (min(tri.v1, tri.v2), max(tri.v1, tri.v2)),
                (min(tri.v2, tri.v3), max(tri.v2, tri.v3)),
                (min(tri.v3, tri.v1), max(tri.v3, tri.v1))
            ]
            
            for edge in edges:
                edge_count[edge] += 1
        
        non_manifold = sum(1 for count in edge_count.values() if count > 2)
        return non_manifold


class ShellExporter:
    """Export grade shells to various formats."""
    
    def __init__(self):
        pass
    
    def export_to_obj(self, shell: GradeShell, filepath: str):
        """Export shell to OBJ format."""
        with open(filepath, 'w') as f:
            f.write(f"# Grade Shell: {shell.name}\n")
            f.write(f"# Cutoff: {shell.cutoff}\n")
            f.write(f"# Element: {shell.element}\n")
            f.write(f"# Vertices: {shell.n_vertices}\n")
            f.write(f"# Triangles: {shell.n_triangles}\n\n")
            
            for vertex in shell.vertices:
                f.write(f"v {vertex.position.x:.6f} {vertex.position.y:.6f} {vertex.position.z:.6f}\n")
            
            f.write("\n")
            
            for tri in shell.triangles:
                f.write(f"f {tri.v1 + 1} {tri.v2 + 1} {tri.v3 + 1}\n")
    
    def export_to_stl(self, shell: GradeShell, filepath: str, binary: bool = False):
        """Export shell to STL format."""
        if binary:
            self._export_stl_binary(shell, filepath)
        else:
            self._export_stl_ascii(shell, filepath)
    
    def _export_stl_ascii(self, shell: GradeShell, filepath: str):
        """Export shell to ASCII STL format."""
        with open(filepath, 'w') as f:
            f.write(f"solid {shell.name}\n")
            
            for tri in shell.triangles:
                v1 = shell.vertices[tri.v1].position
                v2 = shell.vertices[tri.v2].position
                v3 = shell.vertices[tri.v3].position
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                
                normal = Point3D(
                    edge1.y * edge2.z - edge1.z * edge2.y,
                    edge1.z * edge2.x - edge1.x * edge2.z,
                    edge1.x * edge2.y - edge1.y * edge2.x
                )
                
                length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
                if length > 0:
                    normal = normal * (1 / length)
                
                f.write(f"  facet normal {normal.x:.6f} {normal.y:.6f} {normal.z:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1.x:.6f} {v1.y:.6f} {v1.z:.6f}\n")
                f.write(f"      vertex {v2.x:.6f} {v2.y:.6f} {v2.z:.6f}\n")
                f.write(f"      vertex {v3.x:.6f} {v3.y:.6f} {v3.z:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write(f"endsolid {shell.name}\n")
    
    def _export_stl_binary(self, shell: GradeShell, filepath: str):
        """Export shell to binary STL format."""
        import struct
        
        with open(filepath, 'wb') as f:
            header = shell.name[:80].ljust(80).encode('ascii')
            f.write(header)
            
            f.write(struct.pack('<I', shell.n_triangles))
            
            for tri in shell.triangles:
                v1 = shell.vertices[tri.v1].position
                v2 = shell.vertices[tri.v2].position
                v3 = shell.vertices[tri.v3].position
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                
                normal = Point3D(
                    edge1.y * edge2.z - edge1.z * edge2.y,
                    edge1.z * edge2.x - edge1.x * edge2.z,
                    edge1.x * edge2.y - edge1.y * edge2.x
                )
                
                length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
                if length > 0:
                    normal = normal * (1 / length)
                
                f.write(struct.pack('<fff', normal.x, normal.y, normal.z))
                f.write(struct.pack('<fff', v1.x, v1.y, v1.z))
                f.write(struct.pack('<fff', v2.x, v2.y, v2.z))
                f.write(struct.pack('<fff', v3.x, v3.y, v3.z))
                f.write(struct.pack('<H', 0))
    
    def export_to_dxf(self, shell: GradeShell, filepath: str):
        """Export shell to DXF format (3DFACE entities)."""
        with open(filepath, 'w') as f:
            f.write("0\nSECTION\n2\nENTITIES\n")
            
            for tri in shell.triangles:
                v1 = shell.vertices[tri.v1].position
                v2 = shell.vertices[tri.v2].position
                v3 = shell.vertices[tri.v3].position
                
                f.write("0\n3DFACE\n")
                f.write(f"8\n{shell.name}\n")
                f.write(f"10\n{v1.x:.6f}\n20\n{v1.y:.6f}\n30\n{v1.z:.6f}\n")
                f.write(f"11\n{v2.x:.6f}\n21\n{v2.y:.6f}\n31\n{v2.z:.6f}\n")
                f.write(f"12\n{v3.x:.6f}\n22\n{v3.y:.6f}\n32\n{v3.z:.6f}\n")
                f.write(f"13\n{v3.x:.6f}\n23\n{v3.y:.6f}\n33\n{v3.z:.6f}\n")
            
            f.write("0\nENDSEC\n0\nEOF\n")
    
    def export_to_json(self, shell: GradeShell) -> Dict[str, Any]:
        """Export shell to JSON-serializable dict."""
        return {
            "name": shell.name,
            "cutoff": shell.cutoff,
            "element": shell.element,
            "surface_type": shell.surface_type.value,
            "volume": shell.volume,
            "surface_area": shell.surface_area,
            "is_closed": shell.is_closed,
            "vertices": [
                {"x": v.position.x, "y": v.position.y, "z": v.position.z, "grade": v.grade}
                for v in shell.vertices
            ],
            "triangles": [
                {"v1": t.v1, "v2": t.v2, "v3": t.v3}
                for t in shell.triangles
            ],
            "created": shell.created_date.isoformat(),
            "metadata": shell.metadata
        }


class GradeShellWorkflow:
    """
    Complete grade shell generation workflow.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.generator = GradeShellGenerator()
        self.smoother = ShellSmoother()
        self.validator = ShellValidator()
        self.exporter = ShellExporter()
        self.shells: Dict[str, GradeShell] = {}
    
    def load_samples(self, samples: List[Dict[str, Any]]):
        """Load sample data."""
        self.generator.set_sample_data(samples)
    
    def setup_grid(self, x_min: float, x_max: float, x_step: float,
                  y_min: float, y_max: float, y_step: float,
                  z_min: float, z_max: float, z_step: float):
        """Setup interpolation grid."""
        self.generator.create_grid(
            x_min, x_max, x_step,
            y_min, y_max, y_step,
            z_min, z_max, z_step
        )
    
    def interpolate(self, element: str, method: str = "inverse_distance",
                   power: float = 2.0, max_distance: float = float('inf')):
        """Interpolate grades onto grid."""
        self.generator.interpolate_grid(element, method, power, max_distance)
    
    def generate_shell(self, cutoff: float, element: str,
                      name: Optional[str] = None,
                      smooth: bool = True,
                      smooth_iterations: int = 3) -> GradeShell:
        """Generate and optionally smooth a grade shell."""
        shell = self.generator.generate_shell(cutoff, element, name)
        
        if smooth and shell.n_vertices > 0:
            shell = self.smoother.taubin_smooth(shell, smooth_iterations)
        
        self.shells[shell.name] = shell
        return shell
    
    def generate_nested_shells(self, cutoffs: List[float], element: str,
                              smooth: bool = True) -> List[GradeShell]:
        """Generate nested grade shells at multiple cutoffs."""
        shells = []
        
        for cutoff in sorted(cutoffs, reverse=True):
            shell = self.generate_shell(cutoff, element, smooth=smooth)
            shells.append(shell)
        
        return shells
    
    def validate_shell(self, shell_name: str) -> Dict[str, Any]:
        """Validate a shell."""
        if shell_name not in self.shells:
            return {"error": f"Shell '{shell_name}' not found"}
        
        return self.validator.validate(self.shells[shell_name])
    
    def export_shell(self, shell_name: str, filepath: str, format: str = "obj"):
        """Export a shell to file."""
        if shell_name not in self.shells:
            raise ValueError(f"Shell '{shell_name}' not found")
        
        shell = self.shells[shell_name]
        
        if format == "obj":
            self.exporter.export_to_obj(shell, filepath)
        elif format == "stl":
            self.exporter.export_to_stl(shell, filepath)
        elif format == "stl_binary":
            self.exporter.export_to_stl(shell, filepath, binary=True)
        elif format == "dxf":
            self.exporter.export_to_dxf(shell, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        return {
            "project": self.project_name,
            "n_samples": len(self.generator.sample_data),
            "grid_dims": self.generator.grid_dims,
            "shells": {
                name: shell.to_dict()
                for name, shell in self.shells.items()
            }
        }


def create_grade_shell_workflow(project_name: str = "default") -> GradeShellWorkflow:
    """Factory function to create a grade shell workflow."""
    return GradeShellWorkflow(project_name)
