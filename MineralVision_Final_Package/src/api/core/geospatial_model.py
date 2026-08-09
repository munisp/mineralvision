"""
Unified Geospatial Data Model for MineralVision.

This module provides canonical CRS/units/vertical datum governance:
- Consistent coordinate reference system handling
- Unit conversion and validation
- Vertical datum management
- Grid conventions and alignment
- Schema validation for all geospatial data

Prevents integration bugs from EPSG mismatches, depth sign conventions,
altitude vs elevation confusion, and unit inconsistencies.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json
import re

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CoordinateSystem(Enum):
    """Coordinate system types."""
    GEOGRAPHIC = "geographic"  # lat/lon
    PROJECTED = "projected"    # easting/northing
    LOCAL = "local"           # local grid


class VerticalDatum(Enum):
    """Vertical datum types."""
    WGS84_ELLIPSOID = "WGS84"
    EGM96 = "EGM96"
    EGM2008 = "EGM2008"
    MSL = "mean_sea_level"
    AGL = "above_ground_level"
    LOCAL = "local_datum"


class DepthConvention(Enum):
    """Depth sign convention."""
    POSITIVE_DOWN = "positive_down"  # Depth increases downward (mining convention)
    POSITIVE_UP = "positive_up"      # Elevation increases upward (survey convention)


class LengthUnit(Enum):
    """Length units."""
    METERS = "m"
    KILOMETERS = "km"
    FEET = "ft"
    MILES = "mi"
    NAUTICAL_MILES = "nm"


class AreaUnit(Enum):
    """Area units."""
    SQUARE_METERS = "m2"
    SQUARE_KILOMETERS = "km2"
    HECTARES = "ha"
    ACRES = "ac"
    SQUARE_FEET = "ft2"


class AngleUnit(Enum):
    """Angle units."""
    DEGREES = "deg"
    RADIANS = "rad"
    GRADIANS = "grad"


class TimeUnit(Enum):
    """Time units."""
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"


@dataclass
class EPSGCode:
    """EPSG code with metadata."""
    code: int
    name: str
    coord_system: CoordinateSystem
    units: LengthUnit
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
    
    @classmethod
    def from_code(cls, code: int) -> 'EPSGCode':
        """Create from EPSG code."""
        # Common EPSG codes
        epsg_registry = {
            4326: cls(4326, "WGS 84", CoordinateSystem.GEOGRAPHIC, LengthUnit.METERS, (-180, -90, 180, 90)),
            4269: cls(4269, "NAD83", CoordinateSystem.GEOGRAPHIC, LengthUnit.METERS, (-180, -90, 180, 90)),
            32601: cls(32601, "WGS 84 / UTM zone 1N", CoordinateSystem.PROJECTED, LengthUnit.METERS),
            3857: cls(3857, "Web Mercator", CoordinateSystem.PROJECTED, LengthUnit.METERS),
            2154: cls(2154, "RGF93 / Lambert-93", CoordinateSystem.PROJECTED, LengthUnit.METERS),
        }
        
        if code in epsg_registry:
            return epsg_registry[code]
        
        # Default for unknown codes
        coord_sys = CoordinateSystem.PROJECTED if code > 10000 else CoordinateSystem.GEOGRAPHIC
        return cls(code, f"EPSG:{code}", coord_sys, LengthUnit.METERS)
    
    def __str__(self) -> str:
        return f"EPSG:{self.code}"


@dataclass
class CRSDefinition:
    """Complete CRS definition."""
    epsg: EPSGCode
    vertical_datum: VerticalDatum = VerticalDatum.WGS84_ELLIPSOID
    depth_convention: DepthConvention = DepthConvention.POSITIVE_DOWN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epsg': self.epsg.code,
            'epsg_name': self.epsg.name,
            'coord_system': self.epsg.coord_system.value,
            'horizontal_units': self.epsg.units.value,
            'vertical_datum': self.vertical_datum.value,
            'depth_convention': self.depth_convention.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRSDefinition':
        epsg = EPSGCode.from_code(data['epsg'])
        vertical_datum = VerticalDatum(data.get('vertical_datum', 'WGS84'))
        depth_convention = DepthConvention(data.get('depth_convention', 'positive_down'))
        return cls(epsg, vertical_datum, depth_convention)


class UnitConverter:
    """Unit conversion utilities."""
    
    # Length conversion factors to meters
    LENGTH_TO_METERS = {
        LengthUnit.METERS: 1.0,
        LengthUnit.KILOMETERS: 1000.0,
        LengthUnit.FEET: 0.3048,
        LengthUnit.MILES: 1609.344,
        LengthUnit.NAUTICAL_MILES: 1852.0
    }
    
    # Area conversion factors to square meters
    AREA_TO_SQ_METERS = {
        AreaUnit.SQUARE_METERS: 1.0,
        AreaUnit.SQUARE_KILOMETERS: 1_000_000.0,
        AreaUnit.HECTARES: 10_000.0,
        AreaUnit.ACRES: 4046.8564224,
        AreaUnit.SQUARE_FEET: 0.09290304
    }
    
    # Angle conversion factors to degrees
    ANGLE_TO_DEGREES = {
        AngleUnit.DEGREES: 1.0,
        AngleUnit.RADIANS: 180.0 / np.pi,
        AngleUnit.GRADIANS: 0.9
    }
    
    # Time conversion factors to seconds
    TIME_TO_SECONDS = {
        TimeUnit.SECONDS: 1.0,
        TimeUnit.MILLISECONDS: 0.001,
        TimeUnit.MICROSECONDS: 0.000001,
        TimeUnit.NANOSECONDS: 0.000000001
    }
    
    @classmethod
    def convert_length(cls, value: float, from_unit: LengthUnit, to_unit: LengthUnit) -> float:
        """Convert length between units."""
        meters = value * cls.LENGTH_TO_METERS[from_unit]
        return meters / cls.LENGTH_TO_METERS[to_unit]
    
    @classmethod
    def convert_area(cls, value: float, from_unit: AreaUnit, to_unit: AreaUnit) -> float:
        """Convert area between units."""
        sq_meters = value * cls.AREA_TO_SQ_METERS[from_unit]
        return sq_meters / cls.AREA_TO_SQ_METERS[to_unit]
    
    @classmethod
    def convert_angle(cls, value: float, from_unit: AngleUnit, to_unit: AngleUnit) -> float:
        """Convert angle between units."""
        degrees = value * cls.ANGLE_TO_DEGREES[from_unit]
        return degrees / cls.ANGLE_TO_DEGREES[to_unit]
    
    @classmethod
    def convert_time(cls, value: float, from_unit: TimeUnit, to_unit: TimeUnit) -> float:
        """Convert time between units."""
        seconds = value * cls.TIME_TO_SECONDS[from_unit]
        return seconds / cls.TIME_TO_SECONDS[to_unit]


@dataclass
class BoundingBox:
    """Geospatial bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: Optional[float] = None
    max_z: Optional[float] = None
    crs: Optional[CRSDefinition] = None
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def depth(self) -> Optional[float]:
        if self.min_z is not None and self.max_z is not None:
            return abs(self.max_z - self.min_z)
        return None
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is within bounds."""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if bounding boxes intersect."""
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                   self.max_y < other.min_y or self.min_y > other.max_y)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'min_x': self.min_x,
            'min_y': self.min_y,
            'max_x': self.max_x,
            'max_y': self.max_y
        }
        if self.min_z is not None:
            result['min_z'] = self.min_z
        if self.max_z is not None:
            result['max_z'] = self.max_z
        if self.crs is not None:
            result['crs'] = self.crs.to_dict()
        return result


@dataclass
class GridDefinition:
    """Grid definition for raster data."""
    origin: Tuple[float, float]  # (x, y) of grid origin
    cell_size: Tuple[float, float]  # (dx, dy) cell dimensions
    shape: Tuple[int, int]  # (rows, cols)
    crs: CRSDefinition
    nodata_value: float = -9999.0
    
    @property
    def bounds(self) -> BoundingBox:
        """Get grid bounding box."""
        min_x = self.origin[0]
        max_y = self.origin[1]
        max_x = min_x + self.shape[1] * self.cell_size[0]
        min_y = max_y - self.shape[0] * self.cell_size[1]
        return BoundingBox(min_x, min_y, max_x, max_y, crs=self.crs)
    
    def xy_to_rowcol(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to row/col indices."""
        col = int((x - self.origin[0]) / self.cell_size[0])
        row = int((self.origin[1] - y) / self.cell_size[1])
        return (row, col)
    
    def rowcol_to_xy(self, row: int, col: int) -> Tuple[float, float]:
        """Convert row/col indices to coordinates."""
        x = self.origin[0] + (col + 0.5) * self.cell_size[0]
        y = self.origin[1] - (row + 0.5) * self.cell_size[1]
        return (x, y)
    
    def is_aligned_with(self, other: 'GridDefinition', tolerance: float = 1e-6) -> bool:
        """Check if grids are aligned."""
        if self.crs.epsg.code != other.crs.epsg.code:
            return False
        
        # Check cell size alignment
        if abs(self.cell_size[0] - other.cell_size[0]) > tolerance:
            return False
        if abs(self.cell_size[1] - other.cell_size[1]) > tolerance:
            return False
        
        # Check origin alignment (modulo cell size)
        dx = (self.origin[0] - other.origin[0]) % self.cell_size[0]
        dy = (self.origin[1] - other.origin[1]) % self.cell_size[1]
        
        return dx < tolerance and dy < tolerance


@dataclass
class GeospatialMetadata:
    """Complete geospatial metadata for any dataset."""
    crs: CRSDefinition
    bounds: BoundingBox
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    accuracy: Optional[float] = None  # meters
    grid: Optional[GridDefinition] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'crs': self.crs.to_dict(),
            'bounds': self.bounds.to_dict()
        }
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        if self.source:
            result['source'] = self.source
        if self.accuracy:
            result['accuracy_m'] = self.accuracy
        if self.grid:
            result['grid'] = {
                'origin': self.grid.origin,
                'cell_size': self.grid.cell_size,
                'shape': self.grid.shape,
                'nodata': self.grid.nodata_value
            }
        return result


class CoordinateTransformer:
    """
    Transform coordinates between CRS.
    
    Handles both horizontal and vertical transformations.
    """
    
    def __init__(self, source_crs: CRSDefinition, target_crs: CRSDefinition):
        self.source_crs = source_crs
        self.target_crs = target_crs
        
    def transform_point(self, x: float, y: float, z: Optional[float] = None) -> Tuple[float, float, Optional[float]]:
        """
        Transform a single point.
        
        In production, this would use pyproj or similar.
        Here we implement basic transformations.
        """
        # Handle horizontal transformation
        if self.source_crs.epsg.code == self.target_crs.epsg.code:
            tx, ty = x, y
        else:
            # Simplified transformation (in production use pyproj)
            tx, ty = self._transform_horizontal(x, y)
        
        # Handle vertical transformation
        tz = None
        if z is not None:
            tz = self._transform_vertical(z)
            
        return (tx, ty, tz)
    
    def transform_points(self, x: np.ndarray, y: np.ndarray, 
                        z: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Transform arrays of points."""
        if self.source_crs.epsg.code == self.target_crs.epsg.code:
            tx, ty = x.copy(), y.copy()
        else:
            tx, ty = self._transform_horizontal_array(x, y)
            
        tz = None
        if z is not None:
            tz = self._transform_vertical_array(z)
            
        return (tx, ty, tz)
    
    def _transform_horizontal(self, x: float, y: float) -> Tuple[float, float]:
        """Transform horizontal coordinates."""
        # Simplified - in production use pyproj
        return (x, y)
    
    def _transform_horizontal_array(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform horizontal coordinate arrays."""
        return (x.copy(), y.copy())
    
    def _transform_vertical(self, z: float) -> float:
        """Transform vertical coordinate."""
        # Handle depth convention
        if self.source_crs.depth_convention != self.target_crs.depth_convention:
            z = -z
            
        # Handle vertical datum (simplified)
        # In production, use geoid models
        return z
    
    def _transform_vertical_array(self, z: np.ndarray) -> np.ndarray:
        """Transform vertical coordinate array."""
        result = z.copy()
        if self.source_crs.depth_convention != self.target_crs.depth_convention:
            result = -result
        return result


class GeospatialValidator:
    """
    Validate geospatial data against schema.
    
    Ensures consistency across all modules.
    """
    
    def __init__(self, canonical_crs: CRSDefinition = None):
        self.canonical_crs = canonical_crs or CRSDefinition(
            epsg=EPSGCode.from_code(4326),
            vertical_datum=VerticalDatum.WGS84_ELLIPSOID,
            depth_convention=DepthConvention.POSITIVE_DOWN
        )
        
    def validate_coordinates(self, x: np.ndarray, y: np.ndarray, 
                           crs: CRSDefinition) -> List[str]:
        """Validate coordinate arrays."""
        errors = []
        
        # Check for NaN/Inf
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            errors.append("Coordinates contain NaN values")
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            errors.append("Coordinates contain infinite values")
            
        # Check bounds for geographic coordinates
        if crs.epsg.coord_system == CoordinateSystem.GEOGRAPHIC:
            if np.any(x < -180) or np.any(x > 180):
                errors.append("Longitude values out of range [-180, 180]")
            if np.any(y < -90) or np.any(y > 90):
                errors.append("Latitude values out of range [-90, 90]")
                
        return errors
    
    def validate_grid(self, grid: GridDefinition) -> List[str]:
        """Validate grid definition."""
        errors = []
        
        # Check cell size
        if grid.cell_size[0] <= 0 or grid.cell_size[1] <= 0:
            errors.append("Cell size must be positive")
            
        # Check shape
        if grid.shape[0] <= 0 or grid.shape[1] <= 0:
            errors.append("Grid shape must be positive")
            
        # Check for reasonable dimensions
        if grid.shape[0] * grid.shape[1] > 1e9:
            errors.append("Grid exceeds 1 billion cells - consider tiling")
            
        return errors
    
    def validate_bounds(self, bounds: BoundingBox) -> List[str]:
        """Validate bounding box."""
        errors = []
        
        if bounds.min_x >= bounds.max_x:
            errors.append("min_x must be less than max_x")
        if bounds.min_y >= bounds.max_y:
            errors.append("min_y must be less than max_y")
            
        if bounds.min_z is not None and bounds.max_z is not None:
            if bounds.min_z >= bounds.max_z:
                errors.append("min_z must be less than max_z")
                
        return errors
    
    def check_crs_compatibility(self, crs1: CRSDefinition, crs2: CRSDefinition) -> Dict[str, Any]:
        """Check compatibility between two CRS definitions."""
        result = {
            'compatible': True,
            'warnings': [],
            'requires_transform': False
        }
        
        # Check horizontal CRS
        if crs1.epsg.code != crs2.epsg.code:
            result['requires_transform'] = True
            result['warnings'].append(f"Different EPSG codes: {crs1.epsg.code} vs {crs2.epsg.code}")
            
        # Check vertical datum
        if crs1.vertical_datum != crs2.vertical_datum:
            result['requires_transform'] = True
            result['warnings'].append(f"Different vertical datums: {crs1.vertical_datum.value} vs {crs2.vertical_datum.value}")
            
        # Check depth convention
        if crs1.depth_convention != crs2.depth_convention:
            result['requires_transform'] = True
            result['warnings'].append(f"Different depth conventions: {crs1.depth_convention.value} vs {crs2.depth_convention.value}")
            
        return result


class GeospatialDataModel:
    """
    Unified geospatial data model manager.
    
    Provides canonical representation and validation for all geospatial data.
    """
    
    def __init__(self, canonical_crs: CRSDefinition = None):
        self.canonical_crs = canonical_crs or CRSDefinition(
            epsg=EPSGCode.from_code(4326),
            vertical_datum=VerticalDatum.WGS84_ELLIPSOID,
            depth_convention=DepthConvention.POSITIVE_DOWN
        )
        self.validator = GeospatialValidator(self.canonical_crs)
        self._registered_datasets: Dict[str, GeospatialMetadata] = {}
        
    def register_dataset(self, dataset_id: str, metadata: GeospatialMetadata) -> Dict[str, Any]:
        """
        Register a dataset with the data model.
        
        Validates and optionally transforms to canonical CRS.
        """
        # Validate
        errors = []
        errors.extend(self.validator.validate_bounds(metadata.bounds))
        if metadata.grid:
            errors.extend(self.validator.validate_grid(metadata.grid))
            
        if errors:
            return {'success': False, 'errors': errors}
            
        # Check CRS compatibility
        compat = self.validator.check_crs_compatibility(metadata.crs, self.canonical_crs)
        
        # Register
        self._registered_datasets[dataset_id] = metadata
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'requires_transform': compat['requires_transform'],
            'warnings': compat['warnings']
        }
    
    def get_transformer(self, source_crs: CRSDefinition) -> CoordinateTransformer:
        """Get transformer to canonical CRS."""
        return CoordinateTransformer(source_crs, self.canonical_crs)
    
    def transform_to_canonical(self, x: np.ndarray, y: np.ndarray,
                              z: Optional[np.ndarray], 
                              source_crs: CRSDefinition) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Transform coordinates to canonical CRS."""
        transformer = self.get_transformer(source_crs)
        return transformer.transform_points(x, y, z)
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[GeospatialMetadata]:
        """Get metadata for registered dataset."""
        return self._registered_datasets.get(dataset_id)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._registered_datasets.keys())
    
    def check_alignment(self, dataset_ids: List[str]) -> Dict[str, Any]:
        """Check if multiple datasets are aligned."""
        if len(dataset_ids) < 2:
            return {'aligned': True, 'issues': []}
            
        issues = []
        reference = self._registered_datasets.get(dataset_ids[0])
        
        if reference is None:
            return {'aligned': False, 'issues': [f"Dataset {dataset_ids[0]} not found"]}
            
        for ds_id in dataset_ids[1:]:
            ds = self._registered_datasets.get(ds_id)
            if ds is None:
                issues.append(f"Dataset {ds_id} not found")
                continue
                
            # Check CRS
            compat = self.validator.check_crs_compatibility(reference.crs, ds.crs)
            if compat['requires_transform']:
                issues.extend([f"{ds_id}: {w}" for w in compat['warnings']])
                
            # Check grid alignment if both have grids
            if reference.grid and ds.grid:
                if not reference.grid.is_aligned_with(ds.grid):
                    issues.append(f"{ds_id}: Grid not aligned with reference")
                    
        return {
            'aligned': len(issues) == 0,
            'issues': issues
        }


# Factory functions
def create_geospatial_model(epsg: int = 4326,
                           vertical_datum: str = "WGS84",
                           depth_convention: str = "positive_down") -> GeospatialDataModel:
    """Create geospatial data model with specified canonical CRS."""
    crs = CRSDefinition(
        epsg=EPSGCode.from_code(epsg),
        vertical_datum=VerticalDatum(vertical_datum),
        depth_convention=DepthConvention(depth_convention)
    )
    return GeospatialDataModel(crs)


def create_grid_definition(origin: Tuple[float, float],
                          cell_size: float,
                          shape: Tuple[int, int],
                          epsg: int = 4326) -> GridDefinition:
    """Create grid definition."""
    crs = CRSDefinition(epsg=EPSGCode.from_code(epsg))
    return GridDefinition(
        origin=origin,
        cell_size=(cell_size, cell_size),
        shape=shape,
        crs=crs
    )
