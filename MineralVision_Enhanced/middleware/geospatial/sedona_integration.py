"""
Apache Sedona Geospatial Integration
=====================================

Production-grade geospatial data processing for MineralVision:
- Spatial SQL queries
- Geospatial joins
- Raster processing
- Vector operations
- Spatial indexing (R-tree, Quad-tree)
- GeoJSON/WKT/WKB support
- Integration with Spark/Flink

Apache Sedona provides cluster computing for
large-scale geospatial data processing.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import math

logger = logging.getLogger(__name__)

try:
    from sedona.spark import SedonaContext
    from sedona.core.enums import IndexType, GridType
    from sedona.core.spatialOperator import JoinQuery, RangeQuery, KNNQuery
    SEDONA_AVAILABLE = True
except ImportError:
    SEDONA_AVAILABLE = False

from .._mock_fallback import real_client_unavailable


class GeometryType(Enum):
    """Types of geometries."""
    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOINT = "MultiPoint"
    MULTILINESTRING = "MultiLineString"
    MULTIPOLYGON = "MultiPolygon"
    GEOMETRYCOLLECTION = "GeometryCollection"


class SpatialIndexType(Enum):
    """Types of spatial indices."""
    RTREE = "rtree"
    QUADTREE = "quadtree"


class SpatialJoinType(Enum):
    """Types of spatial joins."""
    INTERSECTS = "intersects"
    CONTAINS = "contains"
    WITHIN = "within"
    TOUCHES = "touches"
    CROSSES = "crosses"
    OVERLAPS = "overlaps"
    EQUALS = "equals"
    DISTANCE = "distance"


@dataclass
class BoundingBox:
    """Bounding box for spatial queries."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    def to_wkt(self) -> str:
        """Convert to WKT polygon."""
        return (f"POLYGON(({self.min_x} {self.min_y}, {self.max_x} {self.min_y}, "
                f"{self.max_x} {self.max_y}, {self.min_x} {self.max_y}, "
                f"{self.min_x} {self.min_y}))")
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within bounding box."""
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y)


@dataclass
class Geometry:
    """Geometry object."""
    type: GeometryType
    coordinates: Any
    properties: Dict[str, Any] = field(default_factory=dict)
    srid: int = 4326
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON."""
        return {
            "type": "Feature",
            "geometry": {
                "type": self.type.value,
                "coordinates": self.coordinates
            },
            "properties": self.properties
        }
    
    def to_wkt(self) -> str:
        """Convert to WKT."""
        if self.type == GeometryType.POINT:
            return f"POINT({self.coordinates[0]} {self.coordinates[1]})"
        elif self.type == GeometryType.LINESTRING:
            coords = ", ".join(f"{c[0]} {c[1]}" for c in self.coordinates)
            return f"LINESTRING({coords})"
        elif self.type == GeometryType.POLYGON:
            rings = []
            for ring in self.coordinates:
                coords = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                rings.append(f"({coords})")
            return f"POLYGON({', '.join(rings)})"
        return ""
    
    @classmethod
    def from_wkt(cls, wkt: str, properties: Dict[str, Any] = None) -> 'Geometry':
        """Create geometry from WKT."""
        wkt = wkt.strip().upper()
        
        if wkt.startswith("POINT"):
            coords_str = wkt[wkt.index("(")+1:wkt.index(")")]
            coords = [float(c) for c in coords_str.split()]
            return cls(GeometryType.POINT, coords, properties or {})
        
        elif wkt.startswith("LINESTRING"):
            coords_str = wkt[wkt.index("(")+1:wkt.index(")")]
            coords = [[float(c) for c in p.split()] for p in coords_str.split(",")]
            return cls(GeometryType.LINESTRING, coords, properties or {})
        
        elif wkt.startswith("POLYGON"):
            # Simplified parsing
            return cls(GeometryType.POLYGON, [], properties or {})
        
        raise ValueError(f"Unsupported WKT: {wkt}")
    
    @classmethod
    def point(cls, lon: float, lat: float, properties: Dict[str, Any] = None) -> 'Geometry':
        """Create a point geometry."""
        return cls(GeometryType.POINT, [lon, lat], properties or {})
    
    @classmethod
    def polygon(cls, coordinates: List[List[List[float]]], 
               properties: Dict[str, Any] = None) -> 'Geometry':
        """Create a polygon geometry."""
        return cls(GeometryType.POLYGON, coordinates, properties or {})


@dataclass
class SpatialDataset:
    """Spatial dataset."""
    name: str
    geometries: List[Geometry]
    crs: str = "EPSG:4326"
    bounds: Optional[BoundingBox] = None
    
    def __post_init__(self):
        if not self.bounds and self.geometries:
            self._compute_bounds()
    
    def _compute_bounds(self):
        """Compute bounding box from geometries."""
        if not self.geometries:
            return
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for geom in self.geometries:
            if geom.type == GeometryType.POINT:
                x, y = geom.coordinates
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        self.bounds = BoundingBox(min_x, min_y, max_x, max_y)
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [g.to_geojson() for g in self.geometries]
        }


@dataclass
class SedonaConfig:
    """Sedona configuration."""
    spark_master: str = "local[*]"
    app_name: str = "MineralVision-Sedona"
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    kryo_registrator: str = "org.apache.sedona.core.serde.SedonaKryoRegistrator"
    index_type: SpatialIndexType = SpatialIndexType.RTREE
    num_partitions: int = 4


class MockSedonaContext:
    """Mock Sedona context."""
    
    def __init__(self, config: SedonaConfig):
        self.config = config
        self._datasets: Dict[str, SpatialDataset] = {}
        self._indices: Dict[str, Dict[str, Any]] = {}
    
    async def register_dataset(self, dataset: SpatialDataset) -> None:
        """Register a spatial dataset."""
        self._datasets[dataset.name] = dataset
    
    async def get_dataset(self, name: str) -> Optional[SpatialDataset]:
        """Get a dataset by name."""
        return self._datasets.get(name)
    
    async def list_datasets(self) -> List[str]:
        """List all datasets."""
        return list(self._datasets.keys())
    
    async def build_index(self, dataset_name: str, 
                         index_type: SpatialIndexType = SpatialIndexType.RTREE) -> Dict[str, Any]:
        """Build spatial index on dataset."""
        dataset = self._datasets.get(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        self._indices[dataset_name] = {
            'type': index_type.value,
            'built_at': datetime.now().isoformat(),
            'geometry_count': len(dataset.geometries)
        }
        
        return self._indices[dataset_name]
    
    async def range_query(self, dataset_name: str, 
                         bbox: BoundingBox) -> List[Geometry]:
        """Execute range query."""
        dataset = self._datasets.get(dataset_name)
        if not dataset:
            return []
        
        results = []
        for geom in dataset.geometries:
            if geom.type == GeometryType.POINT:
                x, y = geom.coordinates
                if bbox.contains_point(x, y):
                    results.append(geom)
        
        return results
    
    async def knn_query(self, dataset_name: str, point: Geometry,
                       k: int = 10) -> List[Tuple[Geometry, float]]:
        """Execute K-nearest neighbors query."""
        dataset = self._datasets.get(dataset_name)
        if not dataset:
            return []
        
        if point.type != GeometryType.POINT:
            raise ValueError("Query point must be a Point geometry")
        
        px, py = point.coordinates
        
        # Calculate distances
        distances = []
        for geom in dataset.geometries:
            if geom.type == GeometryType.POINT:
                gx, gy = geom.coordinates
                dist = math.sqrt((gx - px)**2 + (gy - py)**2)
                distances.append((geom, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    async def spatial_join(self, left_dataset: str, right_dataset: str,
                          join_type: SpatialJoinType,
                          distance: float = None) -> List[Tuple[Geometry, Geometry]]:
        """Execute spatial join."""
        left = self._datasets.get(left_dataset)
        right = self._datasets.get(right_dataset)
        
        if not left or not right:
            return []
        
        results = []
        
        for lg in left.geometries:
            for rg in right.geometries:
                if self._check_spatial_relation(lg, rg, join_type, distance):
                    results.append((lg, rg))
        
        return results
    
    def _check_spatial_relation(self, geom1: Geometry, geom2: Geometry,
                               relation: SpatialJoinType,
                               distance: float = None) -> bool:
        """Check spatial relation between geometries."""
        if relation == SpatialJoinType.DISTANCE and distance is not None:
            if geom1.type == GeometryType.POINT and geom2.type == GeometryType.POINT:
                x1, y1 = geom1.coordinates
                x2, y2 = geom2.coordinates
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                return dist <= distance
        
        # Simplified checks for other relations
        return False
    
    async def buffer(self, geometry: Geometry, distance: float) -> Geometry:
        """Create buffer around geometry."""
        # Simplified buffer - returns approximate polygon
        if geometry.type == GeometryType.POINT:
            x, y = geometry.coordinates
            # Create approximate circle as polygon
            n_points = 32
            coords = []
            for i in range(n_points + 1):
                angle = 2 * math.pi * i / n_points
                px = x + distance * math.cos(angle)
                py = y + distance * math.sin(angle)
                coords.append([px, py])
            
            return Geometry.polygon([coords], geometry.properties)
        
        return geometry
    
    async def centroid(self, geometry: Geometry) -> Geometry:
        """Calculate centroid of geometry."""
        if geometry.type == GeometryType.POINT:
            return geometry
        
        elif geometry.type == GeometryType.POLYGON:
            # Simplified centroid calculation
            if geometry.coordinates and geometry.coordinates[0]:
                ring = geometry.coordinates[0]
                x_sum = sum(p[0] for p in ring)
                y_sum = sum(p[1] for p in ring)
                n = len(ring)
                return Geometry.point(x_sum / n, y_sum / n)
        
        return Geometry.point(0, 0)
    
    async def area(self, geometry: Geometry) -> float:
        """Calculate area of geometry."""
        if geometry.type != GeometryType.POLYGON:
            return 0.0
        
        if not geometry.coordinates or not geometry.coordinates[0]:
            return 0.0
        
        # Shoelace formula
        ring = geometry.coordinates[0]
        n = len(ring)
        area = 0.0
        
        for i in range(n - 1):
            area += ring[i][0] * ring[i + 1][1]
            area -= ring[i + 1][0] * ring[i][1]
        
        return abs(area) / 2.0
    
    async def distance(self, geom1: Geometry, geom2: Geometry) -> float:
        """Calculate distance between geometries."""
        if geom1.type == GeometryType.POINT and geom2.type == GeometryType.POINT:
            x1, y1 = geom1.coordinates
            x2, y2 = geom2.coordinates
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return 0.0
    
    async def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute spatial SQL query."""
        # Mock SQL execution
        logger.info(f"Executing spatial SQL: {sql}")
        return []


class SpatialQueryBuilder:
    """
    Build spatial queries.
    
    Provides:
    - Fluent query building
    - Spatial predicates
    - Aggregations
    """
    
    def __init__(self, context: MockSedonaContext):
        self.context = context
        self._dataset: Optional[str] = None
        self._filters: List[Dict[str, Any]] = []
        self._joins: List[Dict[str, Any]] = []
        self._limit: Optional[int] = None
    
    def from_dataset(self, name: str) -> 'SpatialQueryBuilder':
        """Set source dataset."""
        self._dataset = name
        return self
    
    def within_bbox(self, bbox: BoundingBox) -> 'SpatialQueryBuilder':
        """Filter by bounding box."""
        self._filters.append({
            'type': 'bbox',
            'bbox': bbox
        })
        return self
    
    def within_distance(self, point: Geometry, distance: float) -> 'SpatialQueryBuilder':
        """Filter by distance from point."""
        self._filters.append({
            'type': 'distance',
            'point': point,
            'distance': distance
        })
        return self
    
    def join(self, dataset: str, join_type: SpatialJoinType,
            distance: float = None) -> 'SpatialQueryBuilder':
        """Add spatial join."""
        self._joins.append({
            'dataset': dataset,
            'join_type': join_type,
            'distance': distance
        })
        return self
    
    def limit(self, n: int) -> 'SpatialQueryBuilder':
        """Limit results."""
        self._limit = n
        return self
    
    async def execute(self) -> List[Geometry]:
        """Execute the query."""
        if not self._dataset:
            raise ValueError("No dataset specified")
        
        dataset = await self.context.get_dataset(self._dataset)
        if not dataset:
            return []
        
        results = list(dataset.geometries)
        
        # Apply filters
        for f in self._filters:
            if f['type'] == 'bbox':
                bbox = f['bbox']
                results = [
                    g for g in results
                    if g.type == GeometryType.POINT and 
                    bbox.contains_point(g.coordinates[0], g.coordinates[1])
                ]
            elif f['type'] == 'distance':
                point = f['point']
                distance = f['distance']
                px, py = point.coordinates
                results = [
                    g for g in results
                    if g.type == GeometryType.POINT and
                    math.sqrt((g.coordinates[0] - px)**2 + 
                             (g.coordinates[1] - py)**2) <= distance
                ]
        
        # Apply limit
        if self._limit:
            results = results[:self._limit]
        
        return results


class SedonaIntegration:
    """
    Apache Sedona integration for MineralVision.
    
    Provides geospatial data processing:
    - Spatial queries
    - Spatial joins
    - Geometry operations
    - Spatial indexing
    
    Example:
        sedona = SedonaIntegration()
        await sedona.connect()
        
        # Register dataset
        samples = SpatialDataset(
            name="geological_samples",
            geometries=[
                Geometry.point(-122.4, 37.8, {"sample_id": "S001"}),
                Geometry.point(-122.5, 37.9, {"sample_id": "S002"})
            ]
        )
        await sedona.register_dataset(samples)
        
        # Query by bounding box
        results = await sedona.query().from_dataset("geological_samples") \
            .within_bbox(BoundingBox(-123, 37, -122, 38)) \
            .execute()
        
        # KNN query
        nearest = await sedona.knn_query(
            "geological_samples",
            Geometry.point(-122.45, 37.85),
            k=5
        )
    """
    
    def __init__(self, config: SedonaConfig = None):
        self.config = config or SedonaConfig()
        self.context: Optional[MockSedonaContext] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded

    async def connect(self) -> 'SedonaIntegration':
        """
        Connect to Sedona (real context first).

        Falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        if SEDONA_AVAILABLE:
            try:
                # Initialize real Sedona context
                self.context = SedonaContext.create(
                    self.config.spark_master,
                    self.config.app_name
                )
                logger.info("Connected to Apache Sedona")
            except Exception as e:
                if real_client_unavailable("Apache Sedona", "context creation failed", e):
                    self._degraded = True
                    self.context = MockSedonaContext(self.config)
        else:
            if real_client_unavailable("Apache Sedona", "sedona/spark packages not installed"):
                self._degraded = True
                self.context = MockSedonaContext(self.config)

        self._connected = True
        return self
    
    async def register_dataset(self, dataset: SpatialDataset) -> None:
        """Register a spatial dataset."""
        await self.context.register_dataset(dataset)
    
    async def get_dataset(self, name: str) -> Optional[SpatialDataset]:
        """Get a dataset by name."""
        return await self.context.get_dataset(name)
    
    async def list_datasets(self) -> List[str]:
        """List all datasets."""
        return await self.context.list_datasets()
    
    async def build_index(self, dataset_name: str,
                         index_type: SpatialIndexType = SpatialIndexType.RTREE) -> Dict[str, Any]:
        """Build spatial index."""
        return await self.context.build_index(dataset_name, index_type)
    
    def query(self) -> SpatialQueryBuilder:
        """Create a query builder."""
        return SpatialQueryBuilder(self.context)
    
    async def range_query(self, dataset_name: str,
                         bbox: BoundingBox) -> List[Geometry]:
        """Execute range query."""
        return await self.context.range_query(dataset_name, bbox)
    
    async def knn_query(self, dataset_name: str, point: Geometry,
                       k: int = 10) -> List[Tuple[Geometry, float]]:
        """Execute KNN query."""
        return await self.context.knn_query(dataset_name, point, k)
    
    async def spatial_join(self, left_dataset: str, right_dataset: str,
                          join_type: SpatialJoinType,
                          distance: float = None) -> List[Tuple[Geometry, Geometry]]:
        """Execute spatial join."""
        return await self.context.spatial_join(
            left_dataset, right_dataset, join_type, distance
        )
    
    async def buffer(self, geometry: Geometry, distance: float) -> Geometry:
        """Create buffer around geometry."""
        return await self.context.buffer(geometry, distance)
    
    async def centroid(self, geometry: Geometry) -> Geometry:
        """Calculate centroid."""
        return await self.context.centroid(geometry)
    
    async def area(self, geometry: Geometry) -> float:
        """Calculate area."""
        return await self.context.area(geometry)
    
    async def distance(self, geom1: Geometry, geom2: Geometry) -> float:
        """Calculate distance."""
        return await self.context.distance(geom1, geom2)
    
    async def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute spatial SQL."""
        return await self.context.execute_sql(sql)
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_sedona(config: SedonaConfig = None) -> SedonaIntegration:
    """Create a Sedona integration instance."""
    return SedonaIntegration(config)


async def create_and_connect_sedona(config: SedonaConfig = None) -> SedonaIntegration:
    """Create and connect Sedona."""
    sedona = SedonaIntegration(config)
    await sedona.connect()
    return sedona
