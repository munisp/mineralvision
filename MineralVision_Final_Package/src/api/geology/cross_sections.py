"""
Cross-Section and Fence Diagram Module for MineralVision Platform.

Comprehensive cross-section generation including:
1. Vertical cross-sections along arbitrary lines
2. Fence diagrams (multiple connected sections)
3. Drillhole trace projection onto section planes
4. Grade/lithology interval display
5. Topography and surface projection
6. Annotation and labeling
7. Export to various formats
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np


class SectionType(Enum):
    """Types of geological sections."""
    VERTICAL = "vertical"
    INCLINED = "inclined"
    HORIZONTAL = "horizontal"
    FENCE = "fence"
    LONG_SECTION = "long_section"
    CROSS_SECTION = "cross_section"


class ProjectionMethod(Enum):
    """Methods for projecting data onto sections."""
    PERPENDICULAR = "perpendicular"
    PARALLEL = "parallel"
    NEAREST = "nearest"
    WEIGHTED = "weighted"


class DisplayMode(Enum):
    """Display modes for section data."""
    TRACE_ONLY = "trace_only"
    GRADE_INTERVALS = "grade_intervals"
    LITHOLOGY = "lithology"
    GRADE_COLORED = "grade_colored"
    COMPOSITE = "composite"


class ColorScale(Enum):
    """Color scale types for grade display."""
    LINEAR = "linear"
    LOG = "log"
    QUANTILE = "quantile"
    CUSTOM = "custom"


@dataclass
class Point2D:
    """2D point for section coordinates."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Point3D:
    """3D point in world coordinates."""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class SectionLine:
    """Definition of a section line in plan view."""
    name: str
    start_point: Point3D
    end_point: Point3D
    azimuth: float = 0.0
    width: float = 50.0
    look_direction: str = "right"
    
    def __post_init__(self):
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        self.azimuth = math.degrees(math.atan2(dx, dy)) % 360
        self.length = math.sqrt(dx**2 + dy**2)
    
    @property
    def midpoint(self) -> Point3D:
        return Point3D(
            (self.start_point.x + self.end_point.x) / 2,
            (self.start_point.y + self.end_point.y) / 2,
            (self.start_point.z + self.end_point.z) / 2
        )
    
    def point_at_distance(self, distance: float) -> Point3D:
        """Get point along section line at given distance from start."""
        if self.length == 0:
            return self.start_point
        
        t = distance / self.length
        return Point3D(
            self.start_point.x + t * (self.end_point.x - self.start_point.x),
            self.start_point.y + t * (self.end_point.y - self.start_point.y),
            self.start_point.z + t * (self.end_point.z - self.start_point.z)
        )
    
    def distance_to_point(self, point: Point3D) -> Tuple[float, float]:
        """
        Calculate perpendicular distance and along-section distance to a point.
        Returns (perpendicular_distance, along_section_distance)
        """
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        
        if self.length == 0:
            return (point.distance_to(self.start_point), 0)
        
        t = ((point.x - self.start_point.x) * dx + (point.y - self.start_point.y) * dy) / (self.length ** 2)
        
        closest_x = self.start_point.x + t * dx
        closest_y = self.start_point.y + t * dy
        
        perp_dist = math.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2)
        along_dist = t * self.length
        
        cross = dx * (point.y - self.start_point.y) - dy * (point.x - self.start_point.x)
        if (self.look_direction == "right" and cross < 0) or (self.look_direction == "left" and cross > 0):
            perp_dist = -perp_dist
        
        return (perp_dist, along_dist)


@dataclass
class SectionExtent:
    """Extent of a section view."""
    min_distance: float
    max_distance: float
    min_elevation: float
    max_elevation: float
    
    @property
    def width(self) -> float:
        return self.max_distance - self.min_distance
    
    @property
    def height(self) -> float:
        return self.max_elevation - self.min_elevation


@dataclass
class ProjectedPoint:
    """Point projected onto section plane."""
    world_point: Point3D
    section_x: float
    section_y: float
    perpendicular_distance: float
    hole_id: str = ""
    depth: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectedInterval:
    """Interval projected onto section plane."""
    hole_id: str
    from_point: ProjectedPoint
    to_point: ProjectedPoint
    from_depth: float
    to_depth: float
    values: Dict[str, Any] = field(default_factory=dict)
    lithology: str = ""
    color: str = "#808080"
    width: float = 2.0


@dataclass
class DrillholeTrace:
    """Drillhole trace on section."""
    hole_id: str
    collar_point: ProjectedPoint
    trace_points: List[ProjectedPoint]
    intervals: List[ProjectedInterval]
    perpendicular_distance: float
    within_width: bool
    collar_label_position: Point2D = None
    eoh_label_position: Point2D = None
    
    def __post_init__(self):
        if self.trace_points:
            self.collar_label_position = Point2D(
                self.trace_points[0].section_x,
                self.trace_points[0].section_y + 5
            )
            self.eoh_label_position = Point2D(
                self.trace_points[-1].section_x,
                self.trace_points[-1].section_y - 5
            )


@dataclass
class TopographyProfile:
    """Topography profile along section."""
    points: List[Point2D]
    source: str = "dtm"
    
    def get_elevation_at(self, distance: float) -> Optional[float]:
        """Interpolate elevation at given distance along section."""
        if not self.points:
            return None
        
        for i in range(len(self.points) - 1):
            if self.points[i].x <= distance <= self.points[i + 1].x:
                t = (distance - self.points[i].x) / (self.points[i + 1].x - self.points[i].x)
                return self.points[i].y + t * (self.points[i + 1].y - self.points[i].y)
        
        return None


@dataclass
class SectionAnnotation:
    """Annotation on section."""
    text: str
    position: Point2D
    font_size: float = 10.0
    rotation: float = 0.0
    anchor: str = "center"
    color: str = "#000000"
    annotation_type: str = "label"


@dataclass
class GradeColorMap:
    """Color mapping for grade values."""
    element: str
    min_value: float
    max_value: float
    colors: List[str]
    scale: ColorScale = ColorScale.LINEAR
    null_color: str = "#CCCCCC"
    
    def get_color(self, value: float) -> str:
        """Get color for a grade value."""
        if value is None or math.isnan(value):
            return self.null_color
        
        if value <= self.min_value:
            return self.colors[0]
        if value >= self.max_value:
            return self.colors[-1]
        
        if self.scale == ColorScale.LOG:
            if self.min_value <= 0:
                norm = (value - self.min_value) / (self.max_value - self.min_value)
            else:
                norm = (math.log10(value) - math.log10(self.min_value)) / \
                       (math.log10(self.max_value) - math.log10(self.min_value))
        else:
            norm = (value - self.min_value) / (self.max_value - self.min_value)
        
        idx = int(norm * (len(self.colors) - 1))
        idx = max(0, min(idx, len(self.colors) - 1))
        
        return self.colors[idx]


@dataclass
class Section:
    """Complete section with all projected data."""
    name: str
    section_line: SectionLine
    section_type: SectionType
    extent: SectionExtent
    drillhole_traces: List[DrillholeTrace]
    topography: Optional[TopographyProfile] = None
    annotations: List[SectionAnnotation] = field(default_factory=list)
    grid_spacing_x: float = 100.0
    grid_spacing_y: float = 50.0
    vertical_exaggeration: float = 1.0
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SectionGenerator:
    """Generate cross-sections from drillhole data."""
    
    def __init__(self):
        self.drillhole_data: Dict[str, Dict[str, Any]] = {}
        self.topography_points: List[Point3D] = []
        self.default_width = 50.0
        self.default_projection = ProjectionMethod.PERPENDICULAR
    
    def add_drillhole(self, hole_id: str, collar: Point3D, 
                     trace_points: List[Point3D],
                     assays: Optional[List[Dict[str, Any]]] = None,
                     lithology: Optional[List[Dict[str, Any]]] = None):
        """Add drillhole data for section generation."""
        self.drillhole_data[hole_id] = {
            "collar": collar,
            "trace": trace_points,
            "assays": assays or [],
            "lithology": lithology or []
        }
    
    def add_topography(self, points: List[Point3D]):
        """Add topography points."""
        self.topography_points.extend(points)
    
    def create_section_line(self, name: str, 
                           start_easting: float, start_northing: float,
                           end_easting: float, end_northing: float,
                           start_elevation: float = 0.0,
                           end_elevation: float = 0.0,
                           width: float = 50.0) -> SectionLine:
        """Create a section line definition."""
        return SectionLine(
            name=name,
            start_point=Point3D(start_easting, start_northing, start_elevation),
            end_point=Point3D(end_easting, end_northing, end_elevation),
            width=width
        )
    
    def project_point(self, point: Point3D, section_line: SectionLine) -> ProjectedPoint:
        """Project a 3D point onto the section plane."""
        perp_dist, along_dist = section_line.distance_to_point(point)
        
        return ProjectedPoint(
            world_point=point,
            section_x=along_dist,
            section_y=point.z,
            perpendicular_distance=perp_dist
        )
    
    def project_drillhole(self, hole_id: str, section_line: SectionLine,
                         display_mode: DisplayMode = DisplayMode.TRACE_ONLY,
                         element: Optional[str] = None,
                         color_map: Optional[GradeColorMap] = None) -> Optional[DrillholeTrace]:
        """Project a drillhole onto the section."""
        if hole_id not in self.drillhole_data:
            return None
        
        data = self.drillhole_data[hole_id]
        collar = data["collar"]
        trace = data["trace"]
        
        collar_proj = self.project_point(collar, section_line)
        
        if abs(collar_proj.perpendicular_distance) > section_line.width:
            return None
        
        trace_points = []
        for i, point in enumerate(trace):
            proj = self.project_point(point, section_line)
            proj.hole_id = hole_id
            proj.depth = i * (collar.z - trace[-1].z) / len(trace) if trace else 0
            trace_points.append(proj)
        
        intervals = []
        
        if display_mode == DisplayMode.GRADE_INTERVALS and element:
            for assay in data.get("assays", []):
                if element not in assay.get("values", {}):
                    continue
                
                from_depth = assay.get("from_depth", 0)
                to_depth = assay.get("to_depth", 0)
                value = assay["values"][element]
                
                from_point = self._get_point_at_depth(trace_points, from_depth, collar.z)
                to_point = self._get_point_at_depth(trace_points, to_depth, collar.z)
                
                if from_point and to_point:
                    color = color_map.get_color(value) if color_map else "#808080"
                    
                    intervals.append(ProjectedInterval(
                        hole_id=hole_id,
                        from_point=from_point,
                        to_point=to_point,
                        from_depth=from_depth,
                        to_depth=to_depth,
                        values={element: value},
                        color=color
                    ))
        
        elif display_mode == DisplayMode.LITHOLOGY:
            for lith in data.get("lithology", []):
                from_depth = lith.get("from_depth", 0)
                to_depth = lith.get("to_depth", 0)
                lith_code = lith.get("lithology_code", "UNK")
                
                from_point = self._get_point_at_depth(trace_points, from_depth, collar.z)
                to_point = self._get_point_at_depth(trace_points, to_depth, collar.z)
                
                if from_point and to_point:
                    color = self._get_lithology_color(lith_code)
                    
                    intervals.append(ProjectedInterval(
                        hole_id=hole_id,
                        from_point=from_point,
                        to_point=to_point,
                        from_depth=from_depth,
                        to_depth=to_depth,
                        lithology=lith_code,
                        color=color
                    ))
        
        return DrillholeTrace(
            hole_id=hole_id,
            collar_point=collar_proj,
            trace_points=trace_points,
            intervals=intervals,
            perpendicular_distance=collar_proj.perpendicular_distance,
            within_width=abs(collar_proj.perpendicular_distance) <= section_line.width
        )
    
    def _get_point_at_depth(self, trace_points: List[ProjectedPoint], 
                           depth: float, collar_z: float) -> Optional[ProjectedPoint]:
        """Get projected point at a specific depth."""
        if not trace_points:
            return None
        
        target_z = collar_z - depth
        
        for i in range(len(trace_points) - 1):
            z1 = trace_points[i].section_y
            z2 = trace_points[i + 1].section_y
            
            if (z1 >= target_z >= z2) or (z2 >= target_z >= z1):
                if z1 == z2:
                    t = 0.5
                else:
                    t = (z1 - target_z) / (z1 - z2)
                
                return ProjectedPoint(
                    world_point=Point3D(0, 0, target_z),
                    section_x=trace_points[i].section_x + t * (trace_points[i + 1].section_x - trace_points[i].section_x),
                    section_y=target_z,
                    perpendicular_distance=trace_points[i].perpendicular_distance,
                    hole_id=trace_points[i].hole_id,
                    depth=depth
                )
        
        return trace_points[-1] if trace_points else None
    
    def _get_lithology_color(self, lith_code: str) -> str:
        """Get color for lithology code."""
        colors = {
            "GR": "#FF6B6B",
            "GD": "#FF8E8E",
            "DI": "#4ECDC4",
            "GB": "#2C3E50",
            "BA": "#34495E",
            "AN": "#9B59B6",
            "RH": "#E74C3C",
            "DA": "#E91E63",
            "SS": "#F39C12",
            "SL": "#D35400",
            "MS": "#8B4513",
            "SH": "#696969",
            "LS": "#87CEEB",
            "DO": "#B0C4DE",
            "CG": "#CD853F",
            "BX": "#A0522D",
            "QZ": "#FFFFFF",
            "SC": "#708090",
            "GN": "#778899",
            "MB": "#F5F5F5",
            "PG": "#FFB6C1",
            "SK": "#20B2AA",
            "QV": "#FFFACD",
            "OX": "#B22222",
            "SU": "#FFD700",
            "GO": "#DAA520",
            "SAP": "#DEB887",
            "LAT": "#CD5C5C",
            "AL": "#F5DEB3",
            "OB": "#D2B48C",
            "UNK": "#808080"
        }
        return colors.get(lith_code.upper(), "#808080")
    
    def generate_topography_profile(self, section_line: SectionLine,
                                   sample_interval: float = 10.0) -> TopographyProfile:
        """Generate topography profile along section."""
        if not self.topography_points:
            return TopographyProfile(points=[])
        
        profile_points = []
        
        distance = 0.0
        while distance <= section_line.length:
            section_point = section_line.point_at_distance(distance)
            
            nearby = []
            for topo_point in self.topography_points:
                dist = math.sqrt((topo_point.x - section_point.x)**2 + 
                               (topo_point.y - section_point.y)**2)
                if dist < section_line.width * 2:
                    nearby.append((dist, topo_point.z))
            
            if nearby:
                total_weight = 0
                weighted_z = 0
                for dist, z in nearby:
                    weight = 1 / (dist + 0.001)**2
                    weighted_z += z * weight
                    total_weight += weight
                
                elevation = weighted_z / total_weight if total_weight > 0 else 0
                profile_points.append(Point2D(distance, elevation))
            
            distance += sample_interval
        
        return TopographyProfile(points=profile_points)
    
    def generate_section(self, section_line: SectionLine,
                        display_mode: DisplayMode = DisplayMode.TRACE_ONLY,
                        element: Optional[str] = None,
                        color_map: Optional[GradeColorMap] = None,
                        include_topography: bool = True,
                        vertical_exaggeration: float = 1.0) -> Section:
        """Generate a complete section."""
        
        traces = []
        for hole_id in self.drillhole_data:
            trace = self.project_drillhole(hole_id, section_line, display_mode, element, color_map)
            if trace and trace.within_width:
                traces.append(trace)
        
        min_x = 0
        max_x = section_line.length
        min_z = float('inf')
        max_z = float('-inf')
        
        for trace in traces:
            for point in trace.trace_points:
                min_z = min(min_z, point.section_y)
                max_z = max(max_z, point.section_y)
        
        if min_z == float('inf'):
            min_z = 0
            max_z = 100
        
        padding = (max_z - min_z) * 0.1
        min_z -= padding
        max_z += padding
        
        extent = SectionExtent(
            min_distance=min_x,
            max_distance=max_x,
            min_elevation=min_z,
            max_elevation=max_z
        )
        
        topography = None
        if include_topography and self.topography_points:
            topography = self.generate_topography_profile(section_line)
        
        annotations = self._generate_annotations(traces, section_line)
        
        return Section(
            name=section_line.name,
            section_line=section_line,
            section_type=SectionType.CROSS_SECTION,
            extent=extent,
            drillhole_traces=traces,
            topography=topography,
            annotations=annotations,
            vertical_exaggeration=vertical_exaggeration
        )
    
    def _generate_annotations(self, traces: List[DrillholeTrace],
                             section_line: SectionLine) -> List[SectionAnnotation]:
        """Generate standard annotations for section."""
        annotations = []
        
        for trace in traces:
            if trace.collar_label_position:
                annotations.append(SectionAnnotation(
                    text=trace.hole_id,
                    position=trace.collar_label_position,
                    font_size=8,
                    rotation=-90,
                    anchor="bottom",
                    annotation_type="hole_label"
                ))
        
        annotations.append(SectionAnnotation(
            text=f"Section: {section_line.name}",
            position=Point2D(section_line.length / 2, 0),
            font_size=12,
            annotation_type="title"
        ))
        
        annotations.append(SectionAnnotation(
            text=f"Azimuth: {section_line.azimuth:.1f}°",
            position=Point2D(section_line.length - 50, 0),
            font_size=8,
            annotation_type="info"
        ))
        
        return annotations


class FenceDiagramGenerator:
    """Generate fence diagrams from multiple sections."""
    
    def __init__(self, section_generator: SectionGenerator):
        self.section_generator = section_generator
        self.sections: List[Section] = []
    
    def add_section(self, section: Section):
        """Add a section to the fence diagram."""
        self.sections.append(section)
    
    def generate_fence(self, section_lines: List[SectionLine],
                      display_mode: DisplayMode = DisplayMode.TRACE_ONLY,
                      element: Optional[str] = None,
                      color_map: Optional[GradeColorMap] = None) -> List[Section]:
        """Generate multiple connected sections as a fence diagram."""
        self.sections = []
        
        for section_line in section_lines:
            section = self.section_generator.generate_section(
                section_line, display_mode, element, color_map
            )
            self.sections.append(section)
        
        return self.sections
    
    def get_3d_coordinates(self) -> List[Dict[str, Any]]:
        """Get 3D coordinates for fence diagram visualization."""
        fence_data = []
        
        for section in self.sections:
            section_data = {
                "name": section.name,
                "corners": [
                    section.section_line.start_point.to_tuple(),
                    section.section_line.end_point.to_tuple(),
                    (section.section_line.end_point.x, 
                     section.section_line.end_point.y,
                     section.extent.min_elevation),
                    (section.section_line.start_point.x,
                     section.section_line.start_point.y,
                     section.extent.min_elevation)
                ],
                "traces": []
            }
            
            for trace in section.drillhole_traces:
                trace_3d = []
                for point in trace.trace_points:
                    world = point.world_point
                    trace_3d.append(world.to_tuple())
                section_data["traces"].append({
                    "hole_id": trace.hole_id,
                    "points": trace_3d
                })
            
            fence_data.append(section_data)
        
        return fence_data


class SectionExporter:
    """Export sections to various formats."""
    
    def __init__(self):
        self.supported_formats = ["svg", "dxf", "json", "csv"]
    
    def export_to_svg(self, section: Section, 
                     width: int = 1200, height: int = 800,
                     margin: int = 50) -> str:
        """Export section to SVG format."""
        
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        scale_x = plot_width / section.extent.width if section.extent.width > 0 else 1
        scale_y = plot_height / section.extent.height if section.extent.height > 0 else 1
        scale_y *= section.vertical_exaggeration
        
        def to_svg_x(section_x: float) -> float:
            return margin + (section_x - section.extent.min_distance) * scale_x
        
        def to_svg_y(section_y: float) -> float:
            return height - margin - (section_y - section.extent.min_elevation) * scale_y
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<style>',
            '  .trace { stroke: #333; stroke-width: 1; fill: none; }',
            '  .interval { stroke-width: 4; }',
            '  .grid { stroke: #ddd; stroke-width: 0.5; }',
            '  .label { font-family: Arial; font-size: 8px; }',
            '  .title { font-family: Arial; font-size: 12px; font-weight: bold; }',
            '</style>',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        ]
        
        svg_parts.append('<g class="grid">')
        x = section.extent.min_distance
        while x <= section.extent.max_distance:
            svg_x = to_svg_x(x)
            svg_parts.append(f'<line x1="{svg_x}" y1="{margin}" x2="{svg_x}" y2="{height-margin}"/>')
            x += section.grid_spacing_x
        
        y = section.extent.min_elevation
        while y <= section.extent.max_elevation:
            svg_y = to_svg_y(y)
            svg_parts.append(f'<line x1="{margin}" y1="{svg_y}" x2="{width-margin}" y2="{svg_y}"/>')
            y += section.grid_spacing_y
        svg_parts.append('</g>')
        
        if section.topography:
            svg_parts.append('<g class="topography">')
            points = ' '.join([f'{to_svg_x(p.x)},{to_svg_y(p.y)}' for p in section.topography.points])
            svg_parts.append(f'<polyline points="{points}" fill="none" stroke="#8B4513" stroke-width="2"/>')
            svg_parts.append('</g>')
        
        svg_parts.append('<g class="drillholes">')
        for trace in section.drillhole_traces:
            points = ' '.join([f'{to_svg_x(p.section_x)},{to_svg_y(p.section_y)}' 
                             for p in trace.trace_points])
            svg_parts.append(f'<polyline points="{points}" class="trace"/>')
            
            for interval in trace.intervals:
                x1 = to_svg_x(interval.from_point.section_x)
                y1 = to_svg_y(interval.from_point.section_y)
                x2 = to_svg_x(interval.to_point.section_x)
                y2 = to_svg_y(interval.to_point.section_y)
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="{interval.color}" class="interval"/>'
                )
            
            if trace.trace_points:
                collar = trace.trace_points[0]
                svg_parts.append(
                    f'<text x="{to_svg_x(collar.section_x)}" y="{to_svg_y(collar.section_y) - 5}" '
                    f'class="label" text-anchor="middle">{trace.hole_id}</text>'
                )
        svg_parts.append('</g>')
        
        svg_parts.append(
            f'<text x="{width/2}" y="{margin/2}" class="title" text-anchor="middle">'
            f'{section.name}</text>'
        )
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def export_to_json(self, section: Section) -> Dict[str, Any]:
        """Export section to JSON format."""
        return {
            "name": section.name,
            "type": section.section_type.value,
            "section_line": {
                "start": section.section_line.start_point.to_tuple(),
                "end": section.section_line.end_point.to_tuple(),
                "azimuth": section.section_line.azimuth,
                "length": section.section_line.length,
                "width": section.section_line.width
            },
            "extent": {
                "min_distance": section.extent.min_distance,
                "max_distance": section.extent.max_distance,
                "min_elevation": section.extent.min_elevation,
                "max_elevation": section.extent.max_elevation
            },
            "drillholes": [{
                "hole_id": trace.hole_id,
                "perpendicular_distance": trace.perpendicular_distance,
                "trace": [{
                    "section_x": p.section_x,
                    "section_y": p.section_y,
                    "depth": p.depth
                } for p in trace.trace_points],
                "intervals": [{
                    "from_depth": i.from_depth,
                    "to_depth": i.to_depth,
                    "values": i.values,
                    "lithology": i.lithology,
                    "color": i.color
                } for i in trace.intervals]
            } for trace in section.drillhole_traces],
            "topography": [{
                "distance": p.x,
                "elevation": p.y
            } for p in (section.topography.points if section.topography else [])],
            "vertical_exaggeration": section.vertical_exaggeration,
            "created": section.created_date.isoformat()
        }
    
    def export_to_dxf(self, section: Section) -> str:
        """Export section to DXF format (simplified)."""
        dxf_parts = [
            "0", "SECTION",
            "2", "ENTITIES"
        ]
        
        for trace in section.drillhole_traces:
            if len(trace.trace_points) >= 2:
                dxf_parts.extend(["0", "POLYLINE", "8", "DRILLHOLES", "66", "1"])
                
                for point in trace.trace_points:
                    dxf_parts.extend([
                        "0", "VERTEX",
                        "8", "DRILLHOLES",
                        "10", str(point.section_x),
                        "20", str(point.section_y),
                        "30", "0"
                    ])
                
                dxf_parts.extend(["0", "SEQEND"])
        
        dxf_parts.extend(["0", "ENDSEC", "0", "EOF"])
        
        return '\n'.join(dxf_parts)


def create_section_generator() -> SectionGenerator:
    """Factory function to create a section generator."""
    return SectionGenerator()


def create_grade_color_map(element: str, min_value: float, max_value: float,
                          colors: Optional[List[str]] = None,
                          scale: str = "linear") -> GradeColorMap:
    """
    Factory function to create a grade color map.
    
    Args:
        element: Element name
        min_value: Minimum grade value
        max_value: Maximum grade value
        colors: List of colors (default: green-yellow-red)
        scale: 'linear' or 'log'
    
    Returns:
        Configured GradeColorMap
    """
    if colors is None:
        colors = ["#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
    
    scale_enum = ColorScale.LOG if scale == "log" else ColorScale.LINEAR
    
    return GradeColorMap(
        element=element,
        min_value=min_value,
        max_value=max_value,
        colors=colors,
        scale=scale_enum
    )
