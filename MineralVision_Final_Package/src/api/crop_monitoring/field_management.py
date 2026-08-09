"""
Field Management System for MineralVision Crop Monitoring.

Comprehensive field boundary management:
- Field boundary upload (SHP, KML, KMZ, GeoJSON)
- Interactive field drawing
- Field metadata and grouping
- Area calculation and validation
- Multi-farm hierarchy
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date
import numpy as np
import json
import uuid
import logging
import math

logger = logging.getLogger(__name__)


class FieldStatus(Enum):
    """Field status in crop cycle."""
    FALLOW = "fallow"
    PLANTED = "planted"
    GROWING = "growing"
    MATURE = "mature"
    HARVESTING = "harvesting"
    POST_HARVEST = "post_harvest"


class CropStage(Enum):
    """Crop growth stages."""
    # Oil Palm stages
    NURSERY = "nursery"
    IMMATURE = "immature"  # 0-3 years
    YOUNG_MATURE = "young_mature"  # 3-7 years
    PRIME = "prime"  # 7-18 years
    OLD = "old"  # 18+ years
    
    # Annual crop stages
    GERMINATION = "germination"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    RIPENING = "ripening"
    SENESCENCE = "senescence"


class BoundaryFormat(Enum):
    """Supported boundary file formats."""
    SHAPEFILE = "shp"
    KML = "kml"
    KMZ = "kmz"
    GEOJSON = "geojson"
    WKT = "wkt"
    GPX = "gpx"


@dataclass
class Coordinate:
    """Geographic coordinate."""
    longitude: float
    latitude: float
    elevation: Optional[float] = None
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.longitude, self.latitude)
    
    def to_dict(self) -> Dict[str, float]:
        result = {'longitude': self.longitude, 'latitude': self.latitude}
        if self.elevation is not None:
            result['elevation'] = self.elevation
        return result


@dataclass
class FieldBoundary:
    """Field boundary polygon."""
    boundary_id: str
    coordinates: List[Coordinate]  # Ring of coordinates (closed polygon)
    holes: List[List[Coordinate]] = field(default_factory=list)  # Interior holes
    
    # Calculated properties
    area_ha: float = 0.0
    perimeter_m: float = 0.0
    centroid: Optional[Coordinate] = None
    
    # Metadata
    source_format: BoundaryFormat = BoundaryFormat.GEOJSON
    source_file: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON polygon."""
        exterior = [[c.longitude, c.latitude] for c in self.coordinates]
        # Ensure closed ring
        if exterior[0] != exterior[-1]:
            exterior.append(exterior[0])
        
        rings = [exterior]
        for hole in self.holes:
            hole_coords = [[c.longitude, c.latitude] for c in hole]
            if hole_coords[0] != hole_coords[-1]:
                hole_coords.append(hole_coords[0])
            rings.append(hole_coords)
        
        return {
            'type': 'Polygon',
            'coordinates': rings
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'boundary_id': self.boundary_id,
            'coordinates': [c.to_dict() for c in self.coordinates],
            'area_ha': self.area_ha,
            'perimeter_m': self.perimeter_m,
            'centroid': self.centroid.to_dict() if self.centroid else None,
            'geojson': self.to_geojson()
        }


@dataclass
class CropInfo:
    """Crop information for a field."""
    crop_type: str  # oil_palm, cocoa, ginger, etc.
    variety: str = ""
    planting_date: Optional[date] = None
    expected_harvest_date: Optional[date] = None
    current_stage: CropStage = CropStage.VEGETATIVE
    plant_density: float = 0.0  # plants per hectare
    row_spacing_m: float = 0.0
    plant_spacing_m: float = 0.0
    age_years: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'crop_type': self.crop_type,
            'variety': self.variety,
            'planting_date': self.planting_date.isoformat() if self.planting_date else None,
            'expected_harvest_date': self.expected_harvest_date.isoformat() if self.expected_harvest_date else None,
            'current_stage': self.current_stage.value,
            'plant_density': self.plant_density,
            'row_spacing_m': self.row_spacing_m,
            'plant_spacing_m': self.plant_spacing_m,
            'age_years': self.age_years
        }


@dataclass
class SoilInfo:
    """Soil information for a field."""
    soil_type: str = ""
    texture_class: str = ""
    ph: float = 0.0
    organic_matter_percent: float = 0.0
    drainage_class: str = ""
    slope_percent: float = 0.0
    elevation_m: float = 0.0
    last_soil_test_date: Optional[date] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'soil_type': self.soil_type,
            'texture_class': self.texture_class,
            'ph': self.ph,
            'organic_matter_percent': self.organic_matter_percent,
            'drainage_class': self.drainage_class,
            'slope_percent': self.slope_percent,
            'elevation_m': self.elevation_m,
            'last_soil_test_date': self.last_soil_test_date.isoformat() if self.last_soil_test_date else None
        }


@dataclass
class Field:
    """Agricultural field with all metadata."""
    field_id: str
    name: str
    boundary: FieldBoundary
    
    # Organization
    farm_id: str = ""
    farm_name: str = ""
    block_id: str = ""
    block_name: str = ""
    
    # Status
    status: FieldStatus = FieldStatus.FALLOW
    is_active: bool = True
    
    # Crop information
    crop_info: Optional[CropInfo] = None
    crop_history: List[CropInfo] = field(default_factory=list)
    
    # Soil information
    soil_info: Optional[SoilInfo] = None
    
    # Irrigation
    irrigation_type: str = ""  # none, drip, sprinkler, flood
    water_source: str = ""
    
    # Management
    manager_name: str = ""
    manager_contact: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_id': self.field_id,
            'name': self.name,
            'boundary': self.boundary.to_dict(),
            'farm_id': self.farm_id,
            'farm_name': self.farm_name,
            'block_id': self.block_id,
            'block_name': self.block_name,
            'status': self.status.value,
            'is_active': self.is_active,
            'crop_info': self.crop_info.to_dict() if self.crop_info else None,
            'soil_info': self.soil_info.to_dict() if self.soil_info else None,
            'irrigation_type': self.irrigation_type,
            'area_ha': self.boundary.area_ha,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat()
        }
    
    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert to GeoJSON Feature."""
        return {
            'type': 'Feature',
            'id': self.field_id,
            'geometry': self.boundary.to_geojson(),
            'properties': {
                'field_id': self.field_id,
                'name': self.name,
                'farm_name': self.farm_name,
                'status': self.status.value,
                'crop_type': self.crop_info.crop_type if self.crop_info else None,
                'area_ha': self.boundary.area_ha,
                'tags': self.tags
            }
        }


@dataclass
class Farm:
    """Farm containing multiple fields."""
    farm_id: str
    name: str
    
    # Location
    country: str = ""
    region: str = ""
    district: str = ""
    address: str = ""
    
    # Fields
    fields: List[Field] = field(default_factory=list)
    
    # Statistics
    total_area_ha: float = 0.0
    active_fields_count: int = 0
    
    # Contact
    owner_name: str = ""
    owner_contact: str = ""
    manager_name: str = ""
    manager_contact: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'farm_id': self.farm_id,
            'name': self.name,
            'country': self.country,
            'region': self.region,
            'district': self.district,
            'total_area_ha': self.total_area_ha,
            'active_fields_count': self.active_fields_count,
            'fields_count': len(self.fields),
            'owner_name': self.owner_name,
            'created_at': self.created_at.isoformat()
        }


class GeometryCalculator:
    """Calculate geometric properties of field boundaries."""
    
    EARTH_RADIUS_M = 6371000  # Earth radius in meters
    
    @staticmethod
    def calculate_area_ha(coordinates: List[Coordinate]) -> float:
        """
        Calculate polygon area in hectares using Shoelace formula
        with geodetic correction.
        """
        if len(coordinates) < 3:
            return 0.0
        
        # Convert to radians
        coords_rad = [
            (math.radians(c.longitude), math.radians(c.latitude))
            for c in coordinates
        ]
        
        # Shoelace formula with spherical correction
        n = len(coords_rad)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            lon1, lat1 = coords_rad[i]
            lon2, lat2 = coords_rad[j]
            
            area += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
        
        area = abs(area) * GeometryCalculator.EARTH_RADIUS_M ** 2 / 2
        
        # Convert to hectares
        return area / 10000
    
    @staticmethod
    def calculate_perimeter_m(coordinates: List[Coordinate]) -> float:
        """Calculate polygon perimeter in meters using Haversine formula."""
        if len(coordinates) < 2:
            return 0.0
        
        perimeter = 0.0
        n = len(coordinates)
        
        for i in range(n):
            j = (i + 1) % n
            perimeter += GeometryCalculator.haversine_distance(
                coordinates[i], coordinates[j]
            )
        
        return perimeter
    
    @staticmethod
    def haversine_distance(c1: Coordinate, c2: Coordinate) -> float:
        """Calculate distance between two coordinates in meters."""
        lat1 = math.radians(c1.latitude)
        lat2 = math.radians(c2.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(c2.longitude - c1.longitude)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return GeometryCalculator.EARTH_RADIUS_M * c
    
    @staticmethod
    def calculate_centroid(coordinates: List[Coordinate]) -> Coordinate:
        """Calculate polygon centroid."""
        if not coordinates:
            return Coordinate(0, 0)
        
        sum_lon = sum(c.longitude for c in coordinates)
        sum_lat = sum(c.latitude for c in coordinates)
        n = len(coordinates)
        
        return Coordinate(
            longitude=sum_lon / n,
            latitude=sum_lat / n
        )
    
    @staticmethod
    def get_bounding_box(coordinates: List[Coordinate]) -> Tuple[float, float, float, float]:
        """Get bounding box (min_lon, min_lat, max_lon, max_lat)."""
        if not coordinates:
            return (0, 0, 0, 0)
        
        lons = [c.longitude for c in coordinates]
        lats = [c.latitude for c in coordinates]
        
        return (min(lons), min(lats), max(lons), max(lats))
    
    @staticmethod
    def point_in_polygon(point: Coordinate, polygon: List[Coordinate]) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            if ((polygon[i].latitude > point.latitude) != (polygon[j].latitude > point.latitude) and
                point.longitude < (polygon[j].longitude - polygon[i].longitude) * 
                (point.latitude - polygon[i].latitude) / 
                (polygon[j].latitude - polygon[i].latitude) + polygon[i].longitude):
                inside = not inside
            j = i
        
        return inside


class BoundaryParser:
    """Parse field boundaries from various file formats."""
    
    def parse(self, data: Union[str, bytes, Dict], 
              format_type: BoundaryFormat) -> List[FieldBoundary]:
        """Parse boundary data based on format."""
        if format_type == BoundaryFormat.GEOJSON:
            return self._parse_geojson(data)
        elif format_type == BoundaryFormat.KML:
            return self._parse_kml(data)
        elif format_type == BoundaryFormat.WKT:
            return self._parse_wkt(data)
        elif format_type == BoundaryFormat.SHAPEFILE:
            return self._parse_shapefile(data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _parse_geojson(self, data: Union[str, Dict]) -> List[FieldBoundary]:
        """Parse GeoJSON data."""
        if isinstance(data, str):
            geojson = json.loads(data)
        else:
            geojson = data
        
        boundaries = []
        
        # Handle different GeoJSON types
        if geojson.get('type') == 'FeatureCollection':
            features = geojson.get('features', [])
        elif geojson.get('type') == 'Feature':
            features = [geojson]
        elif geojson.get('type') in ['Polygon', 'MultiPolygon']:
            features = [{'type': 'Feature', 'geometry': geojson, 'properties': {}}]
        else:
            features = []
        
        for feature in features:
            geometry = feature.get('geometry', {})
            
            if geometry.get('type') == 'Polygon':
                boundary = self._polygon_to_boundary(geometry)
                if boundary:
                    boundaries.append(boundary)
            elif geometry.get('type') == 'MultiPolygon':
                for polygon_coords in geometry.get('coordinates', []):
                    poly_geom = {'type': 'Polygon', 'coordinates': polygon_coords}
                    boundary = self._polygon_to_boundary(poly_geom)
                    if boundary:
                        boundaries.append(boundary)
        
        return boundaries
    
    def _polygon_to_boundary(self, geometry: Dict) -> Optional[FieldBoundary]:
        """Convert GeoJSON polygon to FieldBoundary."""
        coords = geometry.get('coordinates', [])
        if not coords:
            return None
        
        # Exterior ring
        exterior = coords[0]
        coordinates = [
            Coordinate(longitude=c[0], latitude=c[1], 
                      elevation=c[2] if len(c) > 2 else None)
            for c in exterior
        ]
        
        # Interior holes
        holes = []
        for hole_coords in coords[1:]:
            hole = [
                Coordinate(longitude=c[0], latitude=c[1])
                for c in hole_coords
            ]
            holes.append(hole)
        
        boundary = FieldBoundary(
            boundary_id=str(uuid.uuid4()),
            coordinates=coordinates,
            holes=holes,
            source_format=BoundaryFormat.GEOJSON
        )
        
        # Calculate properties
        boundary.area_ha = GeometryCalculator.calculate_area_ha(coordinates)
        boundary.perimeter_m = GeometryCalculator.calculate_perimeter_m(coordinates)
        boundary.centroid = GeometryCalculator.calculate_centroid(coordinates)
        
        return boundary
    
    def _parse_kml(self, data: Union[str, bytes]) -> List[FieldBoundary]:
        """Parse KML data (simplified parser)."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        boundaries = []
        
        # Simple regex-based KML parsing
        import re
        
        # Find all coordinate strings
        coord_pattern = r'<coordinates>(.*?)</coordinates>'
        matches = re.findall(coord_pattern, data, re.DOTALL)
        
        for match in matches:
            coords_str = match.strip()
            coordinates = []
            
            for coord_str in coords_str.split():
                parts = coord_str.split(',')
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        elev = float(parts[2]) if len(parts) > 2 else None
                        coordinates.append(Coordinate(lon, lat, elev))
                    except ValueError:
                        continue
            
            if len(coordinates) >= 3:
                boundary = FieldBoundary(
                    boundary_id=str(uuid.uuid4()),
                    coordinates=coordinates,
                    source_format=BoundaryFormat.KML
                )
                boundary.area_ha = GeometryCalculator.calculate_area_ha(coordinates)
                boundary.perimeter_m = GeometryCalculator.calculate_perimeter_m(coordinates)
                boundary.centroid = GeometryCalculator.calculate_centroid(coordinates)
                boundaries.append(boundary)
        
        return boundaries
    
    def _parse_wkt(self, data: str) -> List[FieldBoundary]:
        """Parse WKT (Well-Known Text) data."""
        boundaries = []
        
        # Simple WKT polygon parser
        import re
        
        # Match POLYGON or MULTIPOLYGON
        polygon_pattern = r'POLYGON\s*\(\((.*?)\)\)'
        matches = re.findall(polygon_pattern, data, re.IGNORECASE)
        
        for match in matches:
            coordinates = []
            for coord_str in match.split(','):
                parts = coord_str.strip().split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coordinates.append(Coordinate(lon, lat))
                    except ValueError:
                        continue
            
            if len(coordinates) >= 3:
                boundary = FieldBoundary(
                    boundary_id=str(uuid.uuid4()),
                    coordinates=coordinates,
                    source_format=BoundaryFormat.WKT
                )
                boundary.area_ha = GeometryCalculator.calculate_area_ha(coordinates)
                boundary.perimeter_m = GeometryCalculator.calculate_perimeter_m(coordinates)
                boundary.centroid = GeometryCalculator.calculate_centroid(coordinates)
                boundaries.append(boundary)
        
        return boundaries
    
    def _parse_shapefile(self, data: bytes) -> List[FieldBoundary]:
        """Parse Shapefile data (placeholder - requires pyshp library)."""
        logger.warning("Shapefile parsing requires pyshp library")
        return []


class FieldManager:
    """Manage fields and farms."""
    
    def __init__(self):
        self._farms: Dict[str, Farm] = {}
        self._fields: Dict[str, Field] = {}
        self._parser = BoundaryParser()
    
    def create_farm(self, name: str, **kwargs) -> Farm:
        """Create a new farm."""
        farm_id = str(uuid.uuid4())
        farm = Farm(
            farm_id=farm_id,
            name=name,
            **kwargs
        )
        self._farms[farm_id] = farm
        return farm
    
    def get_farm(self, farm_id: str) -> Optional[Farm]:
        """Get farm by ID."""
        return self._farms.get(farm_id)
    
    def list_farms(self) -> List[Farm]:
        """List all farms."""
        return list(self._farms.values())
    
    def create_field(
        self,
        name: str,
        boundary: FieldBoundary,
        farm_id: str = "",
        crop_type: str = "",
        **kwargs
    ) -> Field:
        """Create a new field."""
        field_id = str(uuid.uuid4())
        
        # Create crop info if provided
        crop_info = None
        if crop_type:
            crop_info = CropInfo(crop_type=crop_type)
        
        field_obj = Field(
            field_id=field_id,
            name=name,
            boundary=boundary,
            farm_id=farm_id,
            crop_info=crop_info,
            **kwargs
        )
        
        self._fields[field_id] = field_obj
        
        # Add to farm if specified
        if farm_id and farm_id in self._farms:
            self._farms[farm_id].fields.append(field_obj)
            self._update_farm_stats(farm_id)
        
        return field_obj
    
    def get_field(self, field_id: str) -> Optional[Field]:
        """Get field by ID."""
        return self._fields.get(field_id)
    
    def list_fields(self, farm_id: str = "", 
                   crop_type: str = "",
                   status: Optional[FieldStatus] = None,
                   tags: List[str] = None) -> List[Field]:
        """List fields with optional filters."""
        fields = list(self._fields.values())
        
        if farm_id:
            fields = [f for f in fields if f.farm_id == farm_id]
        
        if crop_type:
            fields = [f for f in fields if f.crop_info and f.crop_info.crop_type == crop_type]
        
        if status:
            fields = [f for f in fields if f.status == status]
        
        if tags:
            fields = [f for f in fields if any(t in f.tags for t in tags)]
        
        return fields
    
    def update_field(self, field_id: str, **kwargs) -> Optional[Field]:
        """Update field properties."""
        field_obj = self._fields.get(field_id)
        if not field_obj:
            return None
        
        for key, value in kwargs.items():
            if hasattr(field_obj, key):
                setattr(field_obj, key, value)
        
        field_obj.modified_at = datetime.utcnow()
        return field_obj
    
    def delete_field(self, field_id: str) -> bool:
        """Delete a field."""
        field_obj = self._fields.get(field_id)
        if not field_obj:
            return False
        
        # Remove from farm
        if field_obj.farm_id and field_obj.farm_id in self._farms:
            farm = self._farms[field_obj.farm_id]
            farm.fields = [f for f in farm.fields if f.field_id != field_id]
            self._update_farm_stats(field_obj.farm_id)
        
        del self._fields[field_id]
        return True
    
    def import_boundaries(
        self,
        data: Union[str, bytes, Dict],
        format_type: BoundaryFormat,
        farm_id: str = "",
        crop_type: str = "",
        name_prefix: str = "Field"
    ) -> List[Field]:
        """Import field boundaries from file data."""
        boundaries = self._parser.parse(data, format_type)
        
        fields = []
        for i, boundary in enumerate(boundaries):
            name = f"{name_prefix}_{i + 1}"
            field_obj = self.create_field(
                name=name,
                boundary=boundary,
                farm_id=farm_id,
                crop_type=crop_type
            )
            fields.append(field_obj)
        
        return fields
    
    def export_fields_geojson(self, field_ids: List[str] = None) -> Dict[str, Any]:
        """Export fields as GeoJSON FeatureCollection."""
        if field_ids:
            fields = [self._fields[fid] for fid in field_ids if fid in self._fields]
        else:
            fields = list(self._fields.values())
        
        return {
            'type': 'FeatureCollection',
            'features': [f.to_geojson_feature() for f in fields]
        }
    
    def get_total_area(self, farm_id: str = "") -> float:
        """Get total area of all fields (or fields in a farm)."""
        fields = self.list_fields(farm_id=farm_id)
        return sum(f.boundary.area_ha for f in fields)
    
    def get_fields_by_location(
        self,
        point: Coordinate,
        radius_km: float = 10.0
    ) -> List[Field]:
        """Get fields within radius of a point."""
        fields = []
        
        for field_obj in self._fields.values():
            if field_obj.boundary.centroid:
                distance = GeometryCalculator.haversine_distance(
                    point, field_obj.boundary.centroid
                ) / 1000  # Convert to km
                
                if distance <= radius_km:
                    fields.append(field_obj)
        
        return fields
    
    def _update_farm_stats(self, farm_id: str) -> None:
        """Update farm statistics."""
        farm = self._farms.get(farm_id)
        if not farm:
            return
        
        farm.total_area_ha = sum(f.boundary.area_ha for f in farm.fields)
        farm.active_fields_count = sum(1 for f in farm.fields if f.is_active)
        farm.modified_at = datetime.utcnow()


class CropCalendar:
    """Manage crop growth stages and calendar events."""
    
    # Crop-specific growth stage durations (days)
    CROP_STAGES = {
        'oil_palm': {
            CropStage.NURSERY: (0, 365),  # 0-12 months
            CropStage.IMMATURE: (365, 1095),  # 1-3 years
            CropStage.YOUNG_MATURE: (1095, 2555),  # 3-7 years
            CropStage.PRIME: (2555, 6570),  # 7-18 years
            CropStage.OLD: (6570, 10950)  # 18-30 years
        },
        'cocoa': {
            CropStage.NURSERY: (0, 180),
            CropStage.IMMATURE: (180, 1095),
            CropStage.YOUNG_MATURE: (1095, 1825),
            CropStage.PRIME: (1825, 7300),
            CropStage.OLD: (7300, 14600)
        },
        'ginger': {
            CropStage.GERMINATION: (0, 21),
            CropStage.VEGETATIVE: (21, 120),
            CropStage.FLOWERING: (120, 180),
            CropStage.RIPENING: (180, 240),
            CropStage.SENESCENCE: (240, 270)
        }
    }
    
    def get_current_stage(self, crop_type: str, planting_date: date) -> CropStage:
        """Determine current growth stage based on planting date."""
        if crop_type not in self.CROP_STAGES:
            return CropStage.VEGETATIVE
        
        days_since_planting = (date.today() - planting_date).days
        
        for stage, (start_day, end_day) in self.CROP_STAGES[crop_type].items():
            if start_day <= days_since_planting < end_day:
                return stage
        
        # Return last stage if beyond all defined stages
        stages = list(self.CROP_STAGES[crop_type].keys())
        return stages[-1] if stages else CropStage.VEGETATIVE
    
    def get_expected_harvest_date(self, crop_type: str, planting_date: date) -> Optional[date]:
        """Calculate expected harvest date."""
        harvest_days = {
            'oil_palm': 1095,  # First harvest ~3 years
            'cocoa': 1095,  # First harvest ~3 years
            'ginger': 240,  # ~8 months
            'maize': 120,
            'rice': 120,
            'cassava': 365
        }
        
        days = harvest_days.get(crop_type)
        if days:
            return planting_date + timedelta(days=days)
        return None
    
    def get_recommended_activities(self, crop_type: str, stage: CropStage) -> List[Dict[str, str]]:
        """Get recommended activities for current growth stage."""
        activities = {
            'oil_palm': {
                CropStage.NURSERY: [
                    {'activity': 'Watering', 'frequency': 'Daily'},
                    {'activity': 'Fertilization', 'frequency': 'Monthly'},
                    {'activity': 'Pest monitoring', 'frequency': 'Weekly'}
                ],
                CropStage.IMMATURE: [
                    {'activity': 'Circle weeding', 'frequency': 'Monthly'},
                    {'activity': 'Fertilization', 'frequency': 'Quarterly'},
                    {'activity': 'Frond pruning', 'frequency': 'As needed'}
                ],
                CropStage.PRIME: [
                    {'activity': 'Harvesting', 'frequency': 'Every 10-14 days'},
                    {'activity': 'Fertilization', 'frequency': 'Quarterly'},
                    {'activity': 'Pest/disease monitoring', 'frequency': 'Weekly'}
                ]
            },
            'ginger': {
                CropStage.VEGETATIVE: [
                    {'activity': 'Weeding', 'frequency': 'Bi-weekly'},
                    {'activity': 'Earthing up', 'frequency': 'Monthly'},
                    {'activity': 'Fertilization', 'frequency': 'Monthly'}
                ],
                CropStage.RIPENING: [
                    {'activity': 'Reduce irrigation', 'frequency': 'Gradual'},
                    {'activity': 'Harvest preparation', 'frequency': 'Once'}
                ]
            }
        }
        
        crop_activities = activities.get(crop_type, {})
        return crop_activities.get(stage, [])


def create_field_manager() -> FieldManager:
    """Factory function to create field manager."""
    return FieldManager()


def create_sample_fields(manager: FieldManager, farm_name: str = "Demo Farm") -> Farm:
    """Create sample fields for demonstration."""
    # Create farm
    farm = manager.create_farm(
        name=farm_name,
        country="Nigeria",
        region="Cross River",
        district="Calabar"
    )
    
    # Sample field boundaries (simplified polygons)
    sample_boundaries = [
        {
            'name': 'Block A - Oil Palm',
            'coords': [
                (8.3500, 4.9500), (8.3550, 4.9500),
                (8.3550, 4.9550), (8.3500, 4.9550)
            ],
            'crop': 'oil_palm'
        },
        {
            'name': 'Block B - Cocoa',
            'coords': [
                (8.3560, 4.9500), (8.3610, 4.9500),
                (8.3610, 4.9550), (8.3560, 4.9550)
            ],
            'crop': 'cocoa'
        },
        {
            'name': 'Block C - Ginger',
            'coords': [
                (8.3500, 4.9560), (8.3550, 4.9560),
                (8.3550, 4.9610), (8.3500, 4.9610)
            ],
            'crop': 'ginger'
        }
    ]
    
    for sample in sample_boundaries:
        coordinates = [Coordinate(lon, lat) for lon, lat in sample['coords']]
        boundary = FieldBoundary(
            boundary_id=str(uuid.uuid4()),
            coordinates=coordinates
        )
        boundary.area_ha = GeometryCalculator.calculate_area_ha(coordinates)
        boundary.perimeter_m = GeometryCalculator.calculate_perimeter_m(coordinates)
        boundary.centroid = GeometryCalculator.calculate_centroid(coordinates)
        
        manager.create_field(
            name=sample['name'],
            boundary=boundary,
            farm_id=farm.farm_id,
            crop_type=sample['crop'],
            status=FieldStatus.GROWING
        )
    
    return farm
