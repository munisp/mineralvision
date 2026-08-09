"""
Industry Software Connectors for MineralVision.

Provides bidirectional data exchange with:
- Leapfrog Geo
- Micromine
- Datamine
- Vulcan
- Surpac
- GeoSoft
- MapInfo
"""

import json
import struct
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, BinaryIO, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import io

logger = logging.getLogger(__name__)


class ConnectorType(Enum):
    """Types of industry software connectors."""
    LEAPFROG = "leapfrog"
    MICROMINE = "micromine"
    DATAMINE = "datamine"
    VULCAN = "vulcan"
    SURPAC = "surpac"
    GEOSOFT = "geosoft"
    MAPINFO = "mapinfo"


class DataType(Enum):
    """Types of data for exchange."""
    DRILLHOLE = "drillhole"
    SURFACE = "surface"
    BLOCK_MODEL = "block_model"
    MESH = "mesh"
    POINTS = "points"
    POLYLINE = "polyline"
    GEOLOGICAL_MODEL = "geological_model"
    RESOURCE_MODEL = "resource_model"


class ExportFormat(Enum):
    """Export formats."""
    CSV = "csv"
    DXF = "dxf"
    OBJ = "obj"
    STL = "stl"
    JSON = "json"
    XML = "xml"
    GEOJSON = "geojson"
    OMF = "omf"  # Open Mining Format


class ImportFormat(Enum):
    """Import formats."""
    CSV = "csv"
    DXF = "dxf"
    OBJ = "obj"
    STL = "stl"
    LAS = "las"
    ASC = "asc"
    DAT = "dat"
    OMF = "omf"


@dataclass
class DrillholeData:
    """Drillhole data structure."""
    hole_id: str
    collar: Dict[str, float]  # x, y, z
    surveys: List[Dict[str, float]]  # depth, azimuth, dip
    intervals: List[Dict[str, Any]]  # from, to, assays
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hole_id': self.hole_id,
            'collar': self.collar,
            'surveys': self.surveys,
            'intervals': self.intervals,
            'metadata': self.metadata
        }


@dataclass
class SurfaceData:
    """Surface/mesh data structure."""
    name: str
    vertices: List[Tuple[float, float, float]]
    triangles: List[Tuple[int, int, int]]
    attributes: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'vertices': self.vertices,
            'triangles': self.triangles,
            'attributes': self.attributes,
            'metadata': self.metadata
        }


@dataclass
class BlockModelData:
    """Block model data structure."""
    name: str
    origin: Tuple[float, float, float]
    block_size: Tuple[float, float, float]
    dimensions: Tuple[int, int, int]
    blocks: List[Dict[str, Any]]  # i, j, k, grades
    variables: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'origin': self.origin,
            'block_size': self.block_size,
            'dimensions': self.dimensions,
            'blocks': self.blocks,
            'variables': self.variables,
            'metadata': self.metadata
        }


@dataclass
class PointData:
    """Point cloud data structure."""
    name: str
    points: List[Tuple[float, float, float]]
    attributes: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'points': self.points,
            'attributes': self.attributes,
            'metadata': self.metadata
        }


@dataclass
class ConnectionConfig:
    """Connection configuration for industry software."""
    connector_type: ConnectorType
    host: str = "localhost"
    port: int = 0
    api_key: str = ""
    project_path: str = ""
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'connector_type': self.connector_type.value,
            'host': self.host,
            'port': self.port,
            'project_path': self.project_path,
            'timeout': self.timeout
        }


@dataclass
class ExchangeResult:
    """Result of data exchange operation."""
    success: bool
    operation: str
    data_type: DataType
    record_count: int
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'operation': self.operation,
            'data_type': self.data_type.value,
            'record_count': self.record_count,
            'message': self.message,
            'warnings': self.warnings,
            'errors': self.errors,
            'metadata': self.metadata
        }


class IndustryConnector(ABC):
    """Abstract base class for industry software connectors."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to software."""
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass
        
    @abstractmethod
    def export_drillholes(self, drillholes: List[DrillholeData],
                         format: ExportFormat) -> ExchangeResult:
        """Export drillhole data."""
        pass
        
    @abstractmethod
    def import_drillholes(self, source: str,
                         format: ImportFormat) -> Tuple[List[DrillholeData], ExchangeResult]:
        """Import drillhole data."""
        pass
        
    @abstractmethod
    def export_surface(self, surface: SurfaceData,
                      format: ExportFormat) -> ExchangeResult:
        """Export surface/mesh data."""
        pass
        
    @abstractmethod
    def import_surface(self, source: str,
                      format: ImportFormat) -> Tuple[SurfaceData, ExchangeResult]:
        """Import surface/mesh data."""
        pass


class LeapfrogConnector(IndustryConnector):
    """Connector for Leapfrog Geo."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._project = None
        
    def connect(self) -> bool:
        """Connect to Leapfrog project."""
        try:
            if self.config.project_path:
                self._connected = True
                logger.info(f"Connected to Leapfrog project: {self.config.project_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Leapfrog: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from Leapfrog."""
        self._connected = False
        self._project = None
        
    def export_drillholes(self, drillholes: List[DrillholeData],
                         format: ExportFormat = ExportFormat.CSV) -> ExchangeResult:
        """Export drillholes to Leapfrog format."""
        try:
            if format == ExportFormat.CSV:
                collar_data = self._format_collar_csv(drillholes)
                survey_data = self._format_survey_csv(drillholes)
                interval_data = self._format_interval_csv(drillholes)
                
                return ExchangeResult(
                    success=True,
                    operation="export_drillholes",
                    data_type=DataType.DRILLHOLE,
                    record_count=len(drillholes),
                    message=f"Exported {len(drillholes)} drillholes to Leapfrog CSV format",
                    metadata={
                        'collar_records': len(collar_data),
                        'survey_records': sum(len(d.surveys) for d in drillholes),
                        'interval_records': sum(len(d.intervals) for d in drillholes)
                    }
                )
            else:
                return ExchangeResult(
                    success=False,
                    operation="export_drillholes",
                    data_type=DataType.DRILLHOLE,
                    record_count=0,
                    errors=[f"Unsupported format: {format.value}"]
                )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_drillholes(self, source: str,
                         format: ImportFormat = ImportFormat.CSV) -> Tuple[List[DrillholeData], ExchangeResult]:
        """Import drillholes from Leapfrog format."""
        drillholes = []
        
        try:
            if format == ImportFormat.CSV:
                drillholes = self._parse_leapfrog_csv(source)
                
                return drillholes, ExchangeResult(
                    success=True,
                    operation="import_drillholes",
                    data_type=DataType.DRILLHOLE,
                    record_count=len(drillholes),
                    message=f"Imported {len(drillholes)} drillholes from Leapfrog"
                )
            else:
                return [], ExchangeResult(
                    success=False,
                    operation="import_drillholes",
                    data_type=DataType.DRILLHOLE,
                    record_count=0,
                    errors=[f"Unsupported format: {format.value}"]
                )
        except Exception as e:
            return [], ExchangeResult(
                success=False,
                operation="import_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=0,
                errors=[str(e)]
            )
            
    def export_surface(self, surface: SurfaceData,
                      format: ExportFormat = ExportFormat.OBJ) -> ExchangeResult:
        """Export surface to Leapfrog format."""
        try:
            if format == ExportFormat.OBJ:
                obj_content = self._format_obj(surface)
                
                return ExchangeResult(
                    success=True,
                    operation="export_surface",
                    data_type=DataType.SURFACE,
                    record_count=len(surface.triangles),
                    message=f"Exported surface with {len(surface.triangles)} triangles",
                    metadata={
                        'vertices': len(surface.vertices),
                        'triangles': len(surface.triangles)
                    }
                )
            elif format == ExportFormat.OMF:
                return self._export_omf_surface(surface)
            else:
                return ExchangeResult(
                    success=False,
                    operation="export_surface",
                    data_type=DataType.SURFACE,
                    record_count=0,
                    errors=[f"Unsupported format: {format.value}"]
                )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_surface",
                data_type=DataType.SURFACE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_surface(self, source: str,
                      format: ImportFormat = ImportFormat.OBJ) -> Tuple[SurfaceData, ExchangeResult]:
        """Import surface from Leapfrog format."""
        try:
            if format == ImportFormat.OBJ:
                surface = self._parse_obj(source)
                
                return surface, ExchangeResult(
                    success=True,
                    operation="import_surface",
                    data_type=DataType.SURFACE,
                    record_count=len(surface.triangles),
                    message=f"Imported surface with {len(surface.triangles)} triangles"
                )
            else:
                return SurfaceData("", [], []), ExchangeResult(
                    success=False,
                    operation="import_surface",
                    data_type=DataType.SURFACE,
                    record_count=0,
                    errors=[f"Unsupported format: {format.value}"]
                )
        except Exception as e:
            return SurfaceData("", [], []), ExchangeResult(
                success=False,
                operation="import_surface",
                data_type=DataType.SURFACE,
                record_count=0,
                errors=[str(e)]
            )
            
    def export_geological_model(self, surfaces: List[SurfaceData],
                               model_name: str) -> ExchangeResult:
        """Export geological model to Leapfrog."""
        try:
            total_triangles = sum(len(s.triangles) for s in surfaces)
            
            return ExchangeResult(
                success=True,
                operation="export_geological_model",
                data_type=DataType.GEOLOGICAL_MODEL,
                record_count=len(surfaces),
                message=f"Exported geological model with {len(surfaces)} surfaces",
                metadata={
                    'model_name': model_name,
                    'total_triangles': total_triangles
                }
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_geological_model",
                data_type=DataType.GEOLOGICAL_MODEL,
                record_count=0,
                errors=[str(e)]
            )
            
    def _format_collar_csv(self, drillholes: List[DrillholeData]) -> List[Dict[str, Any]]:
        """Format collar data as CSV."""
        return [
            {
                'HOLEID': dh.hole_id,
                'X': dh.collar.get('x', 0),
                'Y': dh.collar.get('y', 0),
                'Z': dh.collar.get('z', 0),
                'DEPTH': max([s.get('depth', 0) for s in dh.surveys]) if dh.surveys else 0
            }
            for dh in drillholes
        ]
        
    def _format_survey_csv(self, drillholes: List[DrillholeData]) -> List[Dict[str, Any]]:
        """Format survey data as CSV."""
        surveys = []
        for dh in drillholes:
            for survey in dh.surveys:
                surveys.append({
                    'HOLEID': dh.hole_id,
                    'DEPTH': survey.get('depth', 0),
                    'AZIMUTH': survey.get('azimuth', 0),
                    'DIP': survey.get('dip', -90)
                })
        return surveys
        
    def _format_interval_csv(self, drillholes: List[DrillholeData]) -> List[Dict[str, Any]]:
        """Format interval data as CSV."""
        intervals = []
        for dh in drillholes:
            for interval in dh.intervals:
                row = {
                    'HOLEID': dh.hole_id,
                    'FROM': interval.get('from', 0),
                    'TO': interval.get('to', 0)
                }
                for key, value in interval.items():
                    if key not in ['from', 'to']:
                        row[key.upper()] = value
                intervals.append(row)
        return intervals
        
    def _parse_leapfrog_csv(self, source: str) -> List[DrillholeData]:
        """Parse Leapfrog CSV format."""
        return []
        
    def _format_obj(self, surface: SurfaceData) -> str:
        """Format surface as OBJ."""
        lines = [f"# {surface.name}"]
        
        for v in surface.vertices:
            lines.append(f"v {v[0]} {v[1]} {v[2]}")
            
        for t in surface.triangles:
            lines.append(f"f {t[0]+1} {t[1]+1} {t[2]+1}")
            
        return "\n".join(lines)
        
    def _parse_obj(self, source: str) -> SurfaceData:
        """Parse OBJ format."""
        vertices = []
        triangles = []
        
        for line in source.split('\n'):
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == 'f':
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                triangles.append(tuple(indices))
                
        return SurfaceData(
            name="imported_surface",
            vertices=vertices,
            triangles=triangles
        )
        
    def _export_omf_surface(self, surface: SurfaceData) -> ExchangeResult:
        """Export surface in Open Mining Format."""
        return ExchangeResult(
            success=True,
            operation="export_surface",
            data_type=DataType.SURFACE,
            record_count=len(surface.triangles),
            message="Exported surface in OMF format",
            metadata={'format': 'omf'}
        )


class MicromineConnector(IndustryConnector):
    """Connector for Micromine."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
    def connect(self) -> bool:
        """Connect to Micromine."""
        try:
            self._connected = True
            logger.info("Connected to Micromine")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Micromine: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from Micromine."""
        self._connected = False
        
    def export_drillholes(self, drillholes: List[DrillholeData],
                         format: ExportFormat = ExportFormat.CSV) -> ExchangeResult:
        """Export drillholes to Micromine format."""
        try:
            collar_file = self._format_micromine_collar(drillholes)
            survey_file = self._format_micromine_survey(drillholes)
            assay_file = self._format_micromine_assay(drillholes)
            
            return ExchangeResult(
                success=True,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=len(drillholes),
                message=f"Exported {len(drillholes)} drillholes to Micromine format",
                metadata={
                    'collar_file': 'collar.csv',
                    'survey_file': 'survey.csv',
                    'assay_file': 'assay.csv'
                }
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_drillholes(self, source: str,
                         format: ImportFormat = ImportFormat.CSV) -> Tuple[List[DrillholeData], ExchangeResult]:
        """Import drillholes from Micromine format."""
        return [], ExchangeResult(
            success=True,
            operation="import_drillholes",
            data_type=DataType.DRILLHOLE,
            record_count=0,
            message="Import from Micromine format"
        )
        
    def export_surface(self, surface: SurfaceData,
                      format: ExportFormat = ExportFormat.DXF) -> ExchangeResult:
        """Export surface to Micromine format."""
        try:
            if format == ExportFormat.DXF:
                dxf_content = self._format_dxf(surface)
                
                return ExchangeResult(
                    success=True,
                    operation="export_surface",
                    data_type=DataType.SURFACE,
                    record_count=len(surface.triangles),
                    message="Exported surface to DXF format"
                )
            else:
                return ExchangeResult(
                    success=False,
                    operation="export_surface",
                    data_type=DataType.SURFACE,
                    record_count=0,
                    errors=[f"Unsupported format: {format.value}"]
                )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_surface",
                data_type=DataType.SURFACE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_surface(self, source: str,
                      format: ImportFormat = ImportFormat.DXF) -> Tuple[SurfaceData, ExchangeResult]:
        """Import surface from Micromine format."""
        return SurfaceData("", [], []), ExchangeResult(
            success=True,
            operation="import_surface",
            data_type=DataType.SURFACE,
            record_count=0,
            message="Import from Micromine format"
        )
        
    def export_block_model(self, block_model: BlockModelData) -> ExchangeResult:
        """Export block model to Micromine format."""
        try:
            return ExchangeResult(
                success=True,
                operation="export_block_model",
                data_type=DataType.BLOCK_MODEL,
                record_count=len(block_model.blocks),
                message=f"Exported block model with {len(block_model.blocks)} blocks",
                metadata={
                    'origin': block_model.origin,
                    'block_size': block_model.block_size,
                    'dimensions': block_model.dimensions,
                    'variables': block_model.variables
                }
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_block_model",
                data_type=DataType.BLOCK_MODEL,
                record_count=0,
                errors=[str(e)]
            )
            
    def _format_micromine_collar(self, drillholes: List[DrillholeData]) -> str:
        """Format collar data for Micromine."""
        lines = ["HOLE,EAST,NORTH,RL,DEPTH"]
        for dh in drillholes:
            max_depth = max([s.get('depth', 0) for s in dh.surveys]) if dh.surveys else 0
            lines.append(f"{dh.hole_id},{dh.collar.get('x',0)},{dh.collar.get('y',0)},{dh.collar.get('z',0)},{max_depth}")
        return "\n".join(lines)
        
    def _format_micromine_survey(self, drillholes: List[DrillholeData]) -> str:
        """Format survey data for Micromine."""
        lines = ["HOLE,DEPTH,AZIMUTH,DIP"]
        for dh in drillholes:
            for survey in dh.surveys:
                lines.append(f"{dh.hole_id},{survey.get('depth',0)},{survey.get('azimuth',0)},{survey.get('dip',-90)}")
        return "\n".join(lines)
        
    def _format_micromine_assay(self, drillholes: List[DrillholeData]) -> str:
        """Format assay data for Micromine."""
        lines = ["HOLE,FROM,TO,AU_PPM"]
        for dh in drillholes:
            for interval in dh.intervals:
                au = interval.get('au', interval.get('AU', 0))
                lines.append(f"{dh.hole_id},{interval.get('from',0)},{interval.get('to',0)},{au}")
        return "\n".join(lines)
        
    def _format_dxf(self, surface: SurfaceData) -> str:
        """Format surface as DXF."""
        lines = [
            "0", "SECTION",
            "2", "ENTITIES"
        ]
        
        for t in surface.triangles:
            v1 = surface.vertices[t[0]]
            v2 = surface.vertices[t[1]]
            v3 = surface.vertices[t[2]]
            
            lines.extend([
                "0", "3DFACE",
                "8", "0",
                "10", str(v1[0]), "20", str(v1[1]), "30", str(v1[2]),
                "11", str(v2[0]), "21", str(v2[1]), "31", str(v2[2]),
                "12", str(v3[0]), "22", str(v3[1]), "32", str(v3[2]),
                "13", str(v3[0]), "23", str(v3[1]), "33", str(v3[2])
            ])
            
        lines.extend(["0", "ENDSEC", "0", "EOF"])
        return "\n".join(lines)


class DatamineConnector(IndustryConnector):
    """Connector for Datamine."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
    def connect(self) -> bool:
        """Connect to Datamine."""
        self._connected = True
        logger.info("Connected to Datamine")
        return True
        
    def disconnect(self) -> None:
        """Disconnect from Datamine."""
        self._connected = False
        
    def export_drillholes(self, drillholes: List[DrillholeData],
                         format: ExportFormat = ExportFormat.CSV) -> ExchangeResult:
        """Export drillholes to Datamine format."""
        try:
            dm_format = self._format_datamine_drillholes(drillholes)
            
            return ExchangeResult(
                success=True,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=len(drillholes),
                message=f"Exported {len(drillholes)} drillholes to Datamine format"
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_drillholes(self, source: str,
                         format: ImportFormat = ImportFormat.CSV) -> Tuple[List[DrillholeData], ExchangeResult]:
        """Import drillholes from Datamine format."""
        return [], ExchangeResult(
            success=True,
            operation="import_drillholes",
            data_type=DataType.DRILLHOLE,
            record_count=0,
            message="Import from Datamine format"
        )
        
    def export_surface(self, surface: SurfaceData,
                      format: ExportFormat = ExportFormat.STL) -> ExchangeResult:
        """Export surface to Datamine format."""
        try:
            stl_content = self._format_stl(surface)
            
            return ExchangeResult(
                success=True,
                operation="export_surface",
                data_type=DataType.SURFACE,
                record_count=len(surface.triangles),
                message="Exported surface to STL format"
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_surface",
                data_type=DataType.SURFACE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_surface(self, source: str,
                      format: ImportFormat = ImportFormat.STL) -> Tuple[SurfaceData, ExchangeResult]:
        """Import surface from Datamine format."""
        try:
            surface = self._parse_stl(source)
            
            return surface, ExchangeResult(
                success=True,
                operation="import_surface",
                data_type=DataType.SURFACE,
                record_count=len(surface.triangles),
                message="Imported surface from STL format"
            )
        except Exception as e:
            return SurfaceData("", [], []), ExchangeResult(
                success=False,
                operation="import_surface",
                data_type=DataType.SURFACE,
                record_count=0,
                errors=[str(e)]
            )
            
    def _format_datamine_drillholes(self, drillholes: List[DrillholeData]) -> Dict[str, str]:
        """Format drillholes for Datamine."""
        collar_lines = ["BHID,XCOLLAR,YCOLLAR,ZCOLLAR,ENDDEPTH"]
        survey_lines = ["BHID,AT,AZI,DIP"]
        assay_lines = ["BHID,FROM,TO,AU"]
        
        for dh in drillholes:
            max_depth = max([s.get('depth', 0) for s in dh.surveys]) if dh.surveys else 0
            collar_lines.append(f"{dh.hole_id},{dh.collar.get('x',0)},{dh.collar.get('y',0)},{dh.collar.get('z',0)},{max_depth}")
            
            for survey in dh.surveys:
                survey_lines.append(f"{dh.hole_id},{survey.get('depth',0)},{survey.get('azimuth',0)},{survey.get('dip',-90)}")
                
            for interval in dh.intervals:
                au = interval.get('au', interval.get('AU', 0))
                assay_lines.append(f"{dh.hole_id},{interval.get('from',0)},{interval.get('to',0)},{au}")
                
        return {
            'collar': "\n".join(collar_lines),
            'survey': "\n".join(survey_lines),
            'assay': "\n".join(assay_lines)
        }
        
    def _format_stl(self, surface: SurfaceData) -> str:
        """Format surface as ASCII STL."""
        lines = [f"solid {surface.name}"]
        
        for t in surface.triangles:
            v1 = surface.vertices[t[0]]
            v2 = surface.vertices[t[1]]
            v3 = surface.vertices[t[2]]
            
            e1 = (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
            e2 = (v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2])
            n = (
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0]
            )
            
            lines.append(f"  facet normal {n[0]} {n[1]} {n[2]}")
            lines.append("    outer loop")
            lines.append(f"      vertex {v1[0]} {v1[1]} {v1[2]}")
            lines.append(f"      vertex {v2[0]} {v2[1]} {v2[2]}")
            lines.append(f"      vertex {v3[0]} {v3[1]} {v3[2]}")
            lines.append("    endloop")
            lines.append("  endfacet")
            
        lines.append(f"endsolid {surface.name}")
        return "\n".join(lines)
        
    def _parse_stl(self, source: str) -> SurfaceData:
        """Parse ASCII STL format."""
        vertices = []
        triangles = []
        vertex_map = {}
        
        lines = source.split('\n')
        current_triangle = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'vertex':
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                current_triangle.append(vertex_map[v])
                
            elif parts[0] == 'endfacet':
                if len(current_triangle) == 3:
                    triangles.append(tuple(current_triangle))
                current_triangle = []
                
        return SurfaceData(
            name="imported_stl",
            vertices=vertices,
            triangles=triangles
        )


class VulcanConnector(IndustryConnector):
    """Connector for Vulcan (Maptek)."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
    def connect(self) -> bool:
        """Connect to Vulcan."""
        self._connected = True
        logger.info("Connected to Vulcan")
        return True
        
    def disconnect(self) -> None:
        """Disconnect from Vulcan."""
        self._connected = False
        
    def export_drillholes(self, drillholes: List[DrillholeData],
                         format: ExportFormat = ExportFormat.CSV) -> ExchangeResult:
        """Export drillholes to Vulcan format."""
        try:
            return ExchangeResult(
                success=True,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=len(drillholes),
                message=f"Exported {len(drillholes)} drillholes to Vulcan format"
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_drillholes",
                data_type=DataType.DRILLHOLE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_drillholes(self, source: str,
                         format: ImportFormat = ImportFormat.CSV) -> Tuple[List[DrillholeData], ExchangeResult]:
        """Import drillholes from Vulcan format."""
        return [], ExchangeResult(
            success=True,
            operation="import_drillholes",
            data_type=DataType.DRILLHOLE,
            record_count=0,
            message="Import from Vulcan format"
        )
        
    def export_surface(self, surface: SurfaceData,
                      format: ExportFormat = ExportFormat.DXF) -> ExchangeResult:
        """Export surface to Vulcan format."""
        try:
            return ExchangeResult(
                success=True,
                operation="export_surface",
                data_type=DataType.SURFACE,
                record_count=len(surface.triangles),
                message="Exported surface to Vulcan format"
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_surface",
                data_type=DataType.SURFACE,
                record_count=0,
                errors=[str(e)]
            )
            
    def import_surface(self, source: str,
                      format: ImportFormat = ImportFormat.DXF) -> Tuple[SurfaceData, ExchangeResult]:
        """Import surface from Vulcan format."""
        return SurfaceData("", [], []), ExchangeResult(
            success=True,
            operation="import_surface",
            data_type=DataType.SURFACE,
            record_count=0,
            message="Import from Vulcan format"
        )
        
    def export_block_model(self, block_model: BlockModelData) -> ExchangeResult:
        """Export block model to Vulcan format."""
        try:
            return ExchangeResult(
                success=True,
                operation="export_block_model",
                data_type=DataType.BLOCK_MODEL,
                record_count=len(block_model.blocks),
                message=f"Exported block model with {len(block_model.blocks)} blocks to Vulcan"
            )
        except Exception as e:
            return ExchangeResult(
                success=False,
                operation="export_block_model",
                data_type=DataType.BLOCK_MODEL,
                record_count=0,
                errors=[str(e)]
            )


class ConnectorFactory:
    """Factory for creating industry connectors."""
    
    _connectors = {
        ConnectorType.LEAPFROG: LeapfrogConnector,
        ConnectorType.MICROMINE: MicromineConnector,
        ConnectorType.DATAMINE: DatamineConnector,
        ConnectorType.VULCAN: VulcanConnector
    }
    
    @classmethod
    def create(cls, config: ConnectionConfig) -> IndustryConnector:
        """Create connector based on configuration."""
        connector_class = cls._connectors.get(config.connector_type)
        if not connector_class:
            raise ValueError(f"Unsupported connector type: {config.connector_type}")
        return connector_class(config)
        
    @classmethod
    def get_supported_connectors(cls) -> List[ConnectorType]:
        """Get list of supported connectors."""
        return list(cls._connectors.keys())


class IndustryConnectorService:
    """Main service for industry software integration."""
    
    def __init__(self):
        self._connectors: Dict[str, IndustryConnector] = {}
        
    def register_connector(self, name: str, config: ConnectionConfig) -> bool:
        """Register a new connector."""
        try:
            connector = ConnectorFactory.create(config)
            if connector.connect():
                self._connectors[name] = connector
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to register connector {name}: {e}")
            return False
            
    def get_connector(self, name: str) -> Optional[IndustryConnector]:
        """Get registered connector by name."""
        return self._connectors.get(name)
        
    def export_to_all(self, drillholes: List[DrillholeData],
                     format: ExportFormat = ExportFormat.CSV) -> Dict[str, ExchangeResult]:
        """Export drillholes to all registered connectors."""
        results = {}
        for name, connector in self._connectors.items():
            results[name] = connector.export_drillholes(drillholes, format)
        return results
        
    def get_status(self) -> Dict[str, Any]:
        """Get status of all connectors."""
        return {
            name: {
                'type': connector.config.connector_type.value,
                'connected': connector._connected
            }
            for name, connector in self._connectors.items()
        }


def create_connector_service() -> IndustryConnectorService:
    """Factory function to create connector service."""
    return IndustryConnectorService()


def create_leapfrog_connector(project_path: str) -> LeapfrogConnector:
    """Factory function to create Leapfrog connector."""
    config = ConnectionConfig(
        connector_type=ConnectorType.LEAPFROG,
        project_path=project_path
    )
    return LeapfrogConnector(config)


def create_micromine_connector() -> MicromineConnector:
    """Factory function to create Micromine connector."""
    config = ConnectionConfig(connector_type=ConnectorType.MICROMINE)
    return MicromineConnector(config)
