"""
Variable Rate Application (VRA) Maps Module for MineralVision Crop Monitoring.

Comprehensive VRA map generation:
- Sowing/seeding maps
- Nitrogen fertilization maps
- P&K fertilization maps
- Custom multi-layer maps
- Zone-based productivity mapping
- Export to agricultural machinery formats (ISO-XML, Shapefile)
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


class VRAMapType(Enum):
    """Types of VRA maps."""
    SOWING = "sowing"
    NITROGEN = "nitrogen"
    PHOSPHORUS = "phosphorus"
    POTASSIUM = "potassium"
    PK_COMBINED = "pk_combined"
    LIME = "lime"
    IRRIGATION = "irrigation"
    CUSTOM = "custom"


class ZoneMethod(Enum):
    """Methods for creating management zones."""
    VEGETATION_INDEX = "vegetation_index"
    PRODUCTIVITY = "productivity"
    SOIL_SAMPLING = "soil_sampling"
    ELEVATION = "elevation"
    MULTI_LAYER = "multi_layer"
    MANUAL = "manual"


class ExportFormat(Enum):
    """Export formats for agricultural machinery."""
    ISO_XML = "iso_xml"
    SHAPEFILE = "shapefile"
    GEOJSON = "geojson"
    CSV = "csv"
    ISOBUS = "isobus"
    JOHN_DEERE = "john_deere"
    TRIMBLE = "trimble"
    AG_LEADER = "ag_leader"


class ApplicationUnit(Enum):
    """Units for application rates."""
    KG_HA = "kg/ha"
    L_HA = "l/ha"
    SEEDS_HA = "seeds/ha"
    SEEDS_M2 = "seeds/m2"
    T_HA = "t/ha"
    MM = "mm"


@dataclass
class ManagementZone:
    """Management zone within a field."""
    zone_id: str
    zone_number: int
    
    # Geometry
    boundary: List[Tuple[float, float]]  # List of (lon, lat) coordinates
    area_ha: float = 0.0
    centroid: Tuple[float, float] = (0, 0)
    
    # Zone characteristics
    productivity_class: str = "medium"  # low, medium, high
    vegetation_index_avg: float = 0.0
    soil_type: str = ""
    
    # Application rates
    application_rate: float = 0.0
    application_unit: ApplicationUnit = ApplicationUnit.KG_HA
    
    # Statistics
    ndvi_mean: float = 0.0
    ndvi_std: float = 0.0
    elevation_mean: float = 0.0
    
    # Color for visualization
    color: str = "#00FF00"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'zone_id': self.zone_id,
            'zone_number': self.zone_number,
            'area_ha': self.area_ha,
            'productivity_class': self.productivity_class,
            'vegetation_index_avg': self.vegetation_index_avg,
            'application_rate': self.application_rate,
            'application_unit': self.application_unit.value,
            'ndvi_mean': self.ndvi_mean,
            'color': self.color
        }
    
    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert to GeoJSON Feature."""
        # Close the polygon ring
        coords = list(self.boundary)
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        
        return {
            'type': 'Feature',
            'id': self.zone_id,
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            },
            'properties': {
                'zone_number': self.zone_number,
                'area_ha': self.area_ha,
                'productivity_class': self.productivity_class,
                'application_rate': self.application_rate,
                'application_unit': self.application_unit.value,
                'color': self.color
            }
        }


@dataclass
class VRAMap:
    """Variable Rate Application map."""
    map_id: str
    field_id: str
    map_type: VRAMapType
    
    # Metadata
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Zones
    zones: List[ManagementZone] = field(default_factory=list)
    num_zones: int = 3
    
    # Zone creation parameters
    zone_method: ZoneMethod = ZoneMethod.VEGETATION_INDEX
    source_layers: List[str] = field(default_factory=list)
    
    # Application parameters
    product_name: str = ""
    product_type: str = ""
    application_unit: ApplicationUnit = ApplicationUnit.KG_HA
    min_rate: float = 0.0
    max_rate: float = 0.0
    avg_rate: float = 0.0
    total_product: float = 0.0
    
    # Field info
    field_area_ha: float = 0.0
    
    # Savings calculation
    uniform_rate: float = 0.0
    vra_savings_percent: float = 0.0
    vra_savings_amount: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'map_id': self.map_id,
            'field_id': self.field_id,
            'map_type': self.map_type.value,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'num_zones': self.num_zones,
            'zone_method': self.zone_method.value,
            'product_name': self.product_name,
            'application_unit': self.application_unit.value,
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,
            'avg_rate': self.avg_rate,
            'total_product': self.total_product,
            'field_area_ha': self.field_area_ha,
            'vra_savings_percent': self.vra_savings_percent,
            'zones': [z.to_dict() for z in self.zones]
        }
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON FeatureCollection."""
        return {
            'type': 'FeatureCollection',
            'properties': {
                'map_id': self.map_id,
                'map_type': self.map_type.value,
                'product_name': self.product_name,
                'application_unit': self.application_unit.value
            },
            'features': [z.to_geojson_feature() for z in self.zones]
        }


@dataclass
class SowingMapParameters:
    """Parameters for sowing/seeding map."""
    crop_type: str
    variety: str = ""
    target_plant_density: float = 0.0  # plants/ha
    seed_weight_g: float = 0.0  # grams per seed
    germination_rate: float = 0.95
    
    # Zone adjustments
    low_productivity_adjustment: float = -0.15  # -15%
    high_productivity_adjustment: float = 0.10  # +10%
    
    def calculate_seeding_rate(self, productivity_class: str) -> float:
        """Calculate seeding rate based on productivity."""
        base_rate = self.target_plant_density / self.germination_rate
        
        if productivity_class == "low":
            return base_rate * (1 + self.low_productivity_adjustment)
        elif productivity_class == "high":
            return base_rate * (1 + self.high_productivity_adjustment)
        return base_rate


@dataclass
class NitrogenMapParameters:
    """Parameters for nitrogen fertilization map."""
    crop_type: str
    growth_stage: str = "vegetative"
    target_yield: float = 0.0  # t/ha
    
    # Nitrogen uptake coefficients
    n_uptake_per_tonne: float = 25.0  # kg N per tonne yield
    
    # Soil N supply
    soil_n_supply: float = 30.0  # kg/ha
    previous_crop_credit: float = 0.0  # kg/ha
    organic_matter_credit: float = 0.0  # kg/ha
    
    # Efficiency
    fertilizer_efficiency: float = 0.6
    
    # Zone adjustments based on NDVI
    ndvi_low_threshold: float = 0.4
    ndvi_high_threshold: float = 0.7
    low_ndvi_adjustment: float = 0.20  # +20% for stressed areas
    high_ndvi_adjustment: float = -0.15  # -15% for vigorous areas
    
    def calculate_n_rate(self, ndvi: float) -> float:
        """Calculate N rate based on NDVI."""
        # Base N requirement
        crop_demand = self.target_yield * self.n_uptake_per_tonne
        soil_supply = self.soil_n_supply + self.previous_crop_credit + self.organic_matter_credit
        base_rate = (crop_demand - soil_supply) / self.fertilizer_efficiency
        
        # Adjust based on NDVI
        if ndvi < self.ndvi_low_threshold:
            return base_rate * (1 + self.low_ndvi_adjustment)
        elif ndvi > self.ndvi_high_threshold:
            return base_rate * (1 + self.high_ndvi_adjustment)
        return base_rate


@dataclass
class PKMapParameters:
    """Parameters for P&K fertilization map."""
    crop_type: str
    
    # Soil test values
    soil_p_ppm: float = 15.0
    soil_k_ppm: float = 150.0
    
    # Target levels
    target_p_ppm: float = 25.0
    target_k_ppm: float = 200.0
    
    # Crop removal rates
    p_removal_per_tonne: float = 4.0  # kg P2O5/t
    k_removal_per_tonne: float = 8.0  # kg K2O/t
    
    # Build-up rates
    p_buildup_factor: float = 9.0  # kg P2O5 to raise 1 ppm
    k_buildup_factor: float = 4.0  # kg K2O to raise 1 ppm
    
    def calculate_p_rate(self, target_yield: float) -> float:
        """Calculate P fertilizer rate."""
        maintenance = target_yield * self.p_removal_per_tonne
        buildup = max(0, (self.target_p_ppm - self.soil_p_ppm) * self.p_buildup_factor)
        return maintenance + buildup
    
    def calculate_k_rate(self, target_yield: float) -> float:
        """Calculate K fertilizer rate."""
        maintenance = target_yield * self.k_removal_per_tonne
        buildup = max(0, (self.target_k_ppm - self.soil_k_ppm) * self.k_buildup_factor)
        return maintenance + buildup


class ZoneCreator:
    """Create management zones from various data sources."""
    
    # Zone colors by productivity
    ZONE_COLORS = {
        'very_low': '#FF0000',
        'low': '#FF8C00',
        'medium': '#FFFF00',
        'high': '#90EE90',
        'very_high': '#008000'
    }
    
    def create_zones_from_ndvi(
        self,
        ndvi_values: np.ndarray,
        field_boundary: List[Tuple[float, float]],
        num_zones: int = 3,
        bounds: Tuple[float, float, float, float] = (0, 0, 1, 1)
    ) -> List[ManagementZone]:
        """Create management zones from NDVI values."""
        # Calculate zone thresholds using quantiles
        valid_values = ndvi_values[~np.isnan(ndvi_values)]
        
        if len(valid_values) == 0:
            return []
        
        # Create zone boundaries using percentiles
        percentiles = np.linspace(0, 100, num_zones + 1)
        thresholds = np.percentile(valid_values, percentiles)
        
        zones = []
        min_lon, min_lat, max_lon, max_lat = bounds
        
        for i in range(num_zones):
            zone_mask = (ndvi_values >= thresholds[i]) & (ndvi_values < thresholds[i + 1])
            
            if i == num_zones - 1:  # Include upper bound for last zone
                zone_mask = (ndvi_values >= thresholds[i]) & (ndvi_values <= thresholds[i + 1])
            
            zone_values = ndvi_values[zone_mask]
            
            if len(zone_values) == 0:
                continue
            
            # Determine productivity class
            if num_zones == 3:
                productivity = ['low', 'medium', 'high'][i]
            elif num_zones == 5:
                productivity = ['very_low', 'low', 'medium', 'high', 'very_high'][i]
            else:
                productivity = 'medium'
            
            # Calculate zone area (simplified - based on pixel count)
            total_pixels = ndvi_values.size
            zone_pixels = np.sum(zone_mask)
            
            # Estimate area from field boundary
            field_area = self._calculate_polygon_area(field_boundary)
            zone_area = field_area * (zone_pixels / total_pixels)
            
            # Create simplified zone boundary (convex hull of zone pixels)
            zone_boundary = self._create_zone_boundary(
                zone_mask, bounds, field_boundary
            )
            
            zone = ManagementZone(
                zone_id=str(uuid.uuid4()),
                zone_number=i + 1,
                boundary=zone_boundary,
                area_ha=zone_area,
                productivity_class=productivity,
                vegetation_index_avg=float(np.mean(zone_values)),
                ndvi_mean=float(np.mean(zone_values)),
                ndvi_std=float(np.std(zone_values)),
                color=self.ZONE_COLORS.get(productivity, '#FFFF00')
            )
            
            zones.append(zone)
        
        return zones
    
    def create_zones_from_productivity(
        self,
        historical_yields: List[float],
        field_boundary: List[Tuple[float, float]],
        num_zones: int = 3
    ) -> List[ManagementZone]:
        """Create zones from historical productivity data."""
        if not historical_yields:
            return []
        
        # Calculate productivity classes
        mean_yield = np.mean(historical_yields)
        std_yield = np.std(historical_yields)
        
        zones = []
        field_area = self._calculate_polygon_area(field_boundary)
        
        for i in range(num_zones):
            if num_zones == 3:
                productivity = ['low', 'medium', 'high'][i]
                area_fraction = [0.25, 0.50, 0.25][i]
            else:
                productivity = 'medium'
                area_fraction = 1.0 / num_zones
            
            zone = ManagementZone(
                zone_id=str(uuid.uuid4()),
                zone_number=i + 1,
                boundary=field_boundary,  # Simplified - use field boundary
                area_ha=field_area * area_fraction,
                productivity_class=productivity,
                color=self.ZONE_COLORS.get(productivity, '#FFFF00')
            )
            zones.append(zone)
        
        return zones
    
    def _calculate_polygon_area(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate polygon area in hectares."""
        if len(coords) < 3:
            return 0.0
        
        # Shoelace formula with geodetic correction
        n = len(coords)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[j]
            
            area += math.radians(lon2 - lon1) * (
                2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2))
            )
        
        area = abs(area) * 6371000 ** 2 / 2  # Earth radius in meters
        return area / 10000  # Convert to hectares
    
    def _create_zone_boundary(
        self,
        zone_mask: np.ndarray,
        bounds: Tuple[float, float, float, float],
        field_boundary: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Create simplified zone boundary from mask."""
        # For simplicity, return field boundary subdivided
        # In production, this would use contour finding
        return field_boundary


class VRAMapGenerator:
    """Generate VRA maps for different applications."""
    
    def __init__(self):
        self.zone_creator = ZoneCreator()
    
    def generate_sowing_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        ndvi_values: np.ndarray,
        params: SowingMapParameters,
        num_zones: int = 3
    ) -> VRAMap:
        """Generate sowing/seeding rate map."""
        # Create zones from NDVI
        zones = self.zone_creator.create_zones_from_ndvi(
            ndvi_values, field_boundary, num_zones
        )
        
        # Calculate seeding rates for each zone
        rates = []
        for zone in zones:
            rate = params.calculate_seeding_rate(zone.productivity_class)
            zone.application_rate = rate
            zone.application_unit = ApplicationUnit.SEEDS_HA
            rates.append(rate)
        
        # Calculate field area
        field_area = self.zone_creator._calculate_polygon_area(field_boundary)
        
        # Calculate total seed requirement
        total_seeds = sum(z.application_rate * z.area_ha for z in zones)
        
        # Calculate savings vs uniform rate
        uniform_rate = params.target_plant_density / params.germination_rate
        uniform_total = uniform_rate * field_area
        savings = (uniform_total - total_seeds) / uniform_total * 100 if uniform_total > 0 else 0
        
        return VRAMap(
            map_id=str(uuid.uuid4()),
            field_id=field_id,
            map_type=VRAMapType.SOWING,
            name=f"Sowing Map - {params.crop_type}",
            description=f"Variable rate seeding map for {params.variety or params.crop_type}",
            zones=zones,
            num_zones=num_zones,
            zone_method=ZoneMethod.VEGETATION_INDEX,
            product_name=f"{params.crop_type} seed",
            product_type="seed",
            application_unit=ApplicationUnit.SEEDS_HA,
            min_rate=min(rates) if rates else 0,
            max_rate=max(rates) if rates else 0,
            avg_rate=sum(rates) / len(rates) if rates else 0,
            total_product=total_seeds,
            field_area_ha=field_area,
            uniform_rate=uniform_rate,
            vra_savings_percent=savings
        )
    
    def generate_nitrogen_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        ndvi_values: np.ndarray,
        params: NitrogenMapParameters,
        num_zones: int = 3
    ) -> VRAMap:
        """Generate nitrogen fertilization map."""
        # Create zones from NDVI
        zones = self.zone_creator.create_zones_from_ndvi(
            ndvi_values, field_boundary, num_zones
        )
        
        # Calculate N rates for each zone
        rates = []
        for zone in zones:
            rate = params.calculate_n_rate(zone.ndvi_mean)
            zone.application_rate = max(0, rate)
            zone.application_unit = ApplicationUnit.KG_HA
            rates.append(zone.application_rate)
        
        field_area = self.zone_creator._calculate_polygon_area(field_boundary)
        total_n = sum(z.application_rate * z.area_ha for z in zones)
        
        # Calculate uniform rate
        uniform_rate = params.calculate_n_rate(0.5)  # Use mid NDVI
        uniform_total = uniform_rate * field_area
        savings = (uniform_total - total_n) / uniform_total * 100 if uniform_total > 0 else 0
        
        return VRAMap(
            map_id=str(uuid.uuid4()),
            field_id=field_id,
            map_type=VRAMapType.NITROGEN,
            name=f"Nitrogen Map - {params.crop_type}",
            description=f"Variable rate N fertilization for {params.growth_stage} stage",
            zones=zones,
            num_zones=num_zones,
            zone_method=ZoneMethod.VEGETATION_INDEX,
            product_name="Nitrogen fertilizer",
            product_type="fertilizer",
            application_unit=ApplicationUnit.KG_HA,
            min_rate=min(rates) if rates else 0,
            max_rate=max(rates) if rates else 0,
            avg_rate=sum(rates) / len(rates) if rates else 0,
            total_product=total_n,
            field_area_ha=field_area,
            uniform_rate=uniform_rate,
            vra_savings_percent=savings
        )
    
    def generate_pk_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        soil_samples: List[Dict[str, Any]],
        params: PKMapParameters,
        target_yield: float,
        num_zones: int = 3
    ) -> VRAMap:
        """Generate P&K fertilization map."""
        # Create zones from soil sampling data
        zones = self.zone_creator.create_zones_from_productivity(
            [s.get('yield', 0) for s in soil_samples] if soil_samples else [1.0],
            field_boundary,
            num_zones
        )
        
        # Calculate P&K rates for each zone
        p_rate = params.calculate_p_rate(target_yield)
        k_rate = params.calculate_k_rate(target_yield)
        
        rates = []
        for zone in zones:
            # Adjust based on productivity class
            if zone.productivity_class == 'low':
                adjustment = 1.2  # More fertilizer for low productivity
            elif zone.productivity_class == 'high':
                adjustment = 0.9  # Less for high productivity
            else:
                adjustment = 1.0
            
            combined_rate = (p_rate + k_rate) * adjustment
            zone.application_rate = combined_rate
            zone.application_unit = ApplicationUnit.KG_HA
            rates.append(combined_rate)
        
        field_area = self.zone_creator._calculate_polygon_area(field_boundary)
        total_pk = sum(z.application_rate * z.area_ha for z in zones)
        
        uniform_rate = p_rate + k_rate
        uniform_total = uniform_rate * field_area
        savings = (uniform_total - total_pk) / uniform_total * 100 if uniform_total > 0 else 0
        
        return VRAMap(
            map_id=str(uuid.uuid4()),
            field_id=field_id,
            map_type=VRAMapType.PK_COMBINED,
            name=f"P&K Map - {params.crop_type}",
            description="Variable rate P&K fertilization based on soil tests",
            zones=zones,
            num_zones=num_zones,
            zone_method=ZoneMethod.SOIL_SAMPLING,
            product_name="P&K fertilizer blend",
            product_type="fertilizer",
            application_unit=ApplicationUnit.KG_HA,
            min_rate=min(rates) if rates else 0,
            max_rate=max(rates) if rates else 0,
            avg_rate=sum(rates) / len(rates) if rates else 0,
            total_product=total_pk,
            field_area_ha=field_area,
            uniform_rate=uniform_rate,
            vra_savings_percent=savings
        )
    
    def generate_custom_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        layers: Dict[str, np.ndarray],
        weights: Dict[str, float],
        rate_function: callable,
        product_name: str,
        num_zones: int = 3
    ) -> VRAMap:
        """Generate custom multi-layer VRA map."""
        # Combine layers with weights
        combined = None
        for layer_name, layer_data in layers.items():
            weight = weights.get(layer_name, 1.0)
            normalized = (layer_data - np.nanmin(layer_data)) / (np.nanmax(layer_data) - np.nanmin(layer_data) + 1e-10)
            
            if combined is None:
                combined = normalized * weight
            else:
                combined += normalized * weight
        
        if combined is None:
            return VRAMap(
                map_id=str(uuid.uuid4()),
                field_id=field_id,
                map_type=VRAMapType.CUSTOM,
                name="Custom VRA Map"
            )
        
        # Normalize combined layer
        combined = combined / sum(weights.values())
        
        # Create zones
        zones = self.zone_creator.create_zones_from_ndvi(
            combined, field_boundary, num_zones
        )
        
        # Apply rate function to each zone
        rates = []
        for zone in zones:
            rate = rate_function(zone.vegetation_index_avg)
            zone.application_rate = rate
            rates.append(rate)
        
        field_area = self.zone_creator._calculate_polygon_area(field_boundary)
        
        return VRAMap(
            map_id=str(uuid.uuid4()),
            field_id=field_id,
            map_type=VRAMapType.CUSTOM,
            name=f"Custom Map - {product_name}",
            zones=zones,
            num_zones=num_zones,
            zone_method=ZoneMethod.MULTI_LAYER,
            source_layers=list(layers.keys()),
            product_name=product_name,
            application_unit=ApplicationUnit.KG_HA,
            min_rate=min(rates) if rates else 0,
            max_rate=max(rates) if rates else 0,
            avg_rate=sum(rates) / len(rates) if rates else 0,
            field_area_ha=field_area
        )


class VRAExporter:
    """Export VRA maps to various formats."""
    
    def export(self, vra_map: VRAMap, format_type: ExportFormat) -> Union[str, bytes, Dict]:
        """Export VRA map to specified format."""
        if format_type == ExportFormat.GEOJSON:
            return self._export_geojson(vra_map)
        elif format_type == ExportFormat.ISO_XML:
            return self._export_iso_xml(vra_map)
        elif format_type == ExportFormat.CSV:
            return self._export_csv(vra_map)
        elif format_type == ExportFormat.SHAPEFILE:
            return self._export_shapefile_info(vra_map)
        else:
            return self._export_geojson(vra_map)
    
    def _export_geojson(self, vra_map: VRAMap) -> Dict[str, Any]:
        """Export as GeoJSON."""
        return vra_map.to_geojson()
    
    def _export_iso_xml(self, vra_map: VRAMap) -> str:
        """Export as ISO-XML for ISOBUS compatible equipment."""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<ISO11783_TaskData VersionMajor="4" VersionMinor="0">',
            f'  <TSK A="{vra_map.map_id}" B="{vra_map.name}" G="1">',
            f'    <TZN A="1" B="{vra_map.product_name}" C="1">',
        ]
        
        for zone in vra_map.zones:
            xml_parts.append(
                f'      <PDV A="{zone.zone_number}" B="{zone.application_rate}" '
                f'C="{vra_map.application_unit.value}"/>'
            )
        
        xml_parts.extend([
            '    </TZN>',
            '  </TSK>',
            '</ISO11783_TaskData>'
        ])
        
        return '\n'.join(xml_parts)
    
    def _export_csv(self, vra_map: VRAMap) -> str:
        """Export as CSV."""
        lines = [
            f"# VRA Map: {vra_map.name}",
            f"# Product: {vra_map.product_name}",
            f"# Unit: {vra_map.application_unit.value}",
            f"# Total Area: {vra_map.field_area_ha:.2f} ha",
            "",
            "zone_number,productivity_class,area_ha,application_rate,ndvi_mean"
        ]
        
        for zone in vra_map.zones:
            lines.append(
                f"{zone.zone_number},{zone.productivity_class},"
                f"{zone.area_ha:.2f},{zone.application_rate:.2f},{zone.ndvi_mean:.3f}"
            )
        
        return '\n'.join(lines)
    
    def _export_shapefile_info(self, vra_map: VRAMap) -> Dict[str, Any]:
        """Return shapefile export info (actual export requires pyshp)."""
        return {
            'format': 'shapefile',
            'message': 'Shapefile export requires pyshp library',
            'geojson_alternative': vra_map.to_geojson(),
            'fields': [
                {'name': 'zone_num', 'type': 'N', 'size': 10},
                {'name': 'prod_class', 'type': 'C', 'size': 20},
                {'name': 'area_ha', 'type': 'N', 'size': 10, 'decimal': 2},
                {'name': 'app_rate', 'type': 'N', 'size': 10, 'decimal': 2}
            ]
        }


class SavingsCalculator:
    """Calculate savings from VRA vs uniform application."""
    
    def __init__(self):
        # Product prices (USD per unit)
        self.prices = {
            'nitrogen': 1.2,  # per kg N
            'phosphorus': 1.5,  # per kg P2O5
            'potassium': 0.8,  # per kg K2O
            'seed_oil_palm': 5.0,  # per seed
            'seed_cocoa': 0.5,  # per seed
            'seed_ginger': 0.1  # per kg
        }
    
    def calculate_savings(
        self,
        vra_map: VRAMap,
        product_price: float = None
    ) -> Dict[str, Any]:
        """Calculate cost savings from VRA application."""
        if product_price is None:
            product_price = self.prices.get(
                vra_map.map_type.value, 1.0
            )
        
        # Calculate VRA total cost
        vra_total = sum(z.application_rate * z.area_ha for z in vra_map.zones)
        vra_cost = vra_total * product_price
        
        # Calculate uniform application cost
        uniform_total = vra_map.uniform_rate * vra_map.field_area_ha
        uniform_cost = uniform_total * product_price
        
        # Savings
        product_saved = uniform_total - vra_total
        cost_saved = uniform_cost - vra_cost
        percent_saved = (product_saved / uniform_total * 100) if uniform_total > 0 else 0
        
        return {
            'vra_total_product': vra_total,
            'uniform_total_product': uniform_total,
            'product_saved': product_saved,
            'vra_cost': vra_cost,
            'uniform_cost': uniform_cost,
            'cost_saved': cost_saved,
            'percent_saved': percent_saved,
            'product_unit': vra_map.application_unit.value,
            'currency': 'USD'
        }


class VRAService:
    """Main service for VRA map generation and management."""
    
    def __init__(self):
        self.generator = VRAMapGenerator()
        self.exporter = VRAExporter()
        self.savings_calc = SavingsCalculator()
        
        # Cache
        self._maps: Dict[str, VRAMap] = {}
    
    def create_sowing_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        ndvi_values: np.ndarray,
        crop_type: str,
        target_density: float,
        num_zones: int = 3,
        **kwargs
    ) -> VRAMap:
        """Create sowing rate map."""
        params = SowingMapParameters(
            crop_type=crop_type,
            target_plant_density=target_density,
            **kwargs
        )
        
        vra_map = self.generator.generate_sowing_map(
            field_id, field_boundary, ndvi_values, params, num_zones
        )
        
        self._maps[vra_map.map_id] = vra_map
        return vra_map
    
    def create_nitrogen_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        ndvi_values: np.ndarray,
        crop_type: str,
        target_yield: float,
        growth_stage: str = "vegetative",
        num_zones: int = 3,
        **kwargs
    ) -> VRAMap:
        """Create nitrogen fertilization map."""
        params = NitrogenMapParameters(
            crop_type=crop_type,
            target_yield=target_yield,
            growth_stage=growth_stage,
            **kwargs
        )
        
        vra_map = self.generator.generate_nitrogen_map(
            field_id, field_boundary, ndvi_values, params, num_zones
        )
        
        self._maps[vra_map.map_id] = vra_map
        return vra_map
    
    def create_pk_map(
        self,
        field_id: str,
        field_boundary: List[Tuple[float, float]],
        crop_type: str,
        target_yield: float,
        soil_p_ppm: float,
        soil_k_ppm: float,
        num_zones: int = 3
    ) -> VRAMap:
        """Create P&K fertilization map."""
        params = PKMapParameters(
            crop_type=crop_type,
            soil_p_ppm=soil_p_ppm,
            soil_k_ppm=soil_k_ppm
        )
        
        vra_map = self.generator.generate_pk_map(
            field_id, field_boundary, [], params, target_yield, num_zones
        )
        
        self._maps[vra_map.map_id] = vra_map
        return vra_map
    
    def get_map(self, map_id: str) -> Optional[VRAMap]:
        """Get VRA map by ID."""
        return self._maps.get(map_id)
    
    def list_maps(self, field_id: str = "") -> List[VRAMap]:
        """List VRA maps, optionally filtered by field."""
        maps = list(self._maps.values())
        if field_id:
            maps = [m for m in maps if m.field_id == field_id]
        return maps
    
    def export_map(self, map_id: str, format_type: ExportFormat) -> Union[str, bytes, Dict]:
        """Export VRA map to specified format."""
        vra_map = self._maps.get(map_id)
        if not vra_map:
            raise ValueError(f"Map not found: {map_id}")
        
        return self.exporter.export(vra_map, format_type)
    
    def calculate_savings(self, map_id: str, product_price: float = None) -> Dict[str, Any]:
        """Calculate savings for a VRA map."""
        vra_map = self._maps.get(map_id)
        if not vra_map:
            raise ValueError(f"Map not found: {map_id}")
        
        return self.savings_calc.calculate_savings(vra_map, product_price)


def create_vra_service() -> VRAService:
    """Factory function to create VRA service."""
    return VRAService()


def create_sample_vra_map(
    field_id: str = "sample_field",
    map_type: VRAMapType = VRAMapType.NITROGEN
) -> VRAMap:
    """Create sample VRA map for demonstration."""
    # Sample field boundary (small polygon)
    field_boundary = [
        (8.35, 4.95), (8.36, 4.95),
        (8.36, 4.96), (8.35, 4.96)
    ]
    
    # Generate synthetic NDVI
    np.random.seed(42)
    ndvi = 0.5 + 0.2 * np.random.randn(100, 100)
    ndvi = np.clip(ndvi, 0, 1)
    
    service = create_vra_service()
    
    if map_type == VRAMapType.SOWING:
        return service.create_sowing_map(
            field_id, field_boundary, ndvi,
            crop_type="oil_palm",
            target_density=143,  # palms/ha
            num_zones=3
        )
    elif map_type == VRAMapType.NITROGEN:
        return service.create_nitrogen_map(
            field_id, field_boundary, ndvi,
            crop_type="oil_palm",
            target_yield=20,  # t FFB/ha
            num_zones=3
        )
    else:
        return service.create_pk_map(
            field_id, field_boundary,
            crop_type="oil_palm",
            target_yield=20,
            soil_p_ppm=15,
            soil_k_ppm=150,
            num_zones=3
        )
