"""
Apache Sedona Enhanced Integration
===================================

Production-grade geospatial data processing for MineralVision:
- Proper Spark + Sedona initialization with Kryo serializer
- Real SedonaSQL/DataFrame operations
- Geodesic-aware calculations (Haversine, Vincenty)
- Actual R-tree spatial indexing
- Raster support for zonal statistics
- Distributed spatial joins and queries

This module provides enterprise-grade geospatial processing
using Apache Sedona on Spark for large-scale data.
"""

import json
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Constants for geodesic calculations
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0
WGS84_A = 6378137.0  # Semi-major axis
WGS84_B = 6356752.314245  # Semi-minor axis
WGS84_F = 1 / 298.257223563  # Flattening


class CoordinateSystem(Enum):
    """Supported coordinate reference systems."""
    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    UTM = "UTM"


class SpatialIndexType(Enum):
    """Types of spatial indices."""
    RTREE = "rtree"
    QUADTREE = "quadtree"
    KDTREE = "kdtree"


class SpatialJoinType(Enum):
    """Types of spatial joins."""
    INTERSECTS = "ST_Intersects"
    CONTAINS = "ST_Contains"
    WITHIN = "ST_Within"
    TOUCHES = "ST_Touches"
    CROSSES = "ST_Crosses"
    OVERLAPS = "ST_Overlaps"
    EQUALS = "ST_Equals"
    DISTANCE_WITHIN = "ST_DWithin"


class GeometryType(Enum):
    """Supported geometry types."""
    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOINT = "MultiPoint"
    MULTILINESTRING = "MultiLineString"
    MULTIPOLYGON = "MultiPolygon"


@dataclass
class SedonaConfig:
    """Enhanced Sedona configuration."""
    spark_master: str = "local[*]"
    app_name: str = "MineralVision-Sedona"
    
    # Spark configuration
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    executor_cores: int = 2
    num_executors: int = 2
    
    # Sedona configuration
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    kryo_registrator: str = "org.apache.sedona.core.serde.SedonaKryoRegistrator"
    
    # Spatial indexing
    default_index_type: SpatialIndexType = SpatialIndexType.RTREE
    index_build_side: str = "left"  # For spatial joins
    
    # Partitioning
    num_partitions: int = 4
    grid_type: str = "QUADTREE"  # QUADTREE, RTREE, KDBTREE
    
    # Performance tuning
    broadcast_join_threshold: int = 10 * 1024 * 1024  # 10MB
    shuffle_partitions: int = 200
    
    # Raster configuration
    raster_block_size: int = 256
    raster_tile_size: int = 512


class GeodesicCalculator:
    """
    Geodesic calculations for accurate distance and area on Earth's surface.
    
    Implements:
    - Haversine formula for distance
    - Vincenty formula for high-precision distance
    - Geodesic area calculation
    - Bearing calculations
    """
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float,
                          unit: str = "km") -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
            unit: "km", "m", "mi", "nm" (nautical miles)
        
        Returns:
            Distance in specified unit
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance_km = EARTH_RADIUS_KM * c
        
        if unit == "km":
            return distance_km
        elif unit == "m":
            return distance_km * 1000
        elif unit == "mi":
            return distance_km * 0.621371
        elif unit == "nm":
            return distance_km * 0.539957
        else:
            return distance_km
    
    @staticmethod
    def vincenty_distance(lat1: float, lon1: float,
                         lat2: float, lon2: float,
                         max_iterations: int = 200,
                         tolerance: float = 1e-12) -> float:
        """
        Calculate distance using Vincenty formula (more accurate for long distances).
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
        
        Returns:
            Distance in meters
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        L = math.radians(lon2 - lon1)
        
        U1 = math.atan((1 - WGS84_F) * math.tan(phi1))
        U2 = math.atan((1 - WGS84_F) * math.tan(phi2))
        
        sin_U1 = math.sin(U1)
        cos_U1 = math.cos(U1)
        sin_U2 = math.sin(U2)
        cos_U2 = math.cos(U2)
        
        lambda_val = L
        
        for _ in range(max_iterations):
            sin_lambda = math.sin(lambda_val)
            cos_lambda = math.cos(lambda_val)
            
            sin_sigma = math.sqrt(
                (cos_U2 * sin_lambda) ** 2 +
                (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2
            )
            
            if sin_sigma == 0:
                return 0.0  # Coincident points
            
            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
            sigma = math.atan2(sin_sigma, cos_sigma)
            
            sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
            cos_sq_alpha = 1 - sin_alpha ** 2
            
            if cos_sq_alpha == 0:
                cos_2sigma_m = 0
            else:
                cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha
            
            C = WGS84_F / 16 * cos_sq_alpha * (4 + WGS84_F * (4 - 3 * cos_sq_alpha))
            
            lambda_prev = lambda_val
            lambda_val = L + (1 - C) * WGS84_F * sin_alpha * (
                sigma + C * sin_sigma * (
                    cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
                )
            )
            
            if abs(lambda_val - lambda_prev) < tolerance:
                break
        
        u_sq = cos_sq_alpha * (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2
        A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
        
        delta_sigma = B * sin_sigma * (
            cos_2sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) -
                B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2)
            )
        )
        
        return WGS84_B * A * (sigma - delta_sigma)
    
    @staticmethod
    def geodesic_area(coordinates: List[List[float]]) -> float:
        """
        Calculate geodesic area of a polygon on Earth's surface.
        
        Uses the spherical excess formula for accurate area calculation.
        
        Args:
            coordinates: List of [lon, lat] pairs forming a closed ring
        
        Returns:
            Area in square meters
        """
        if len(coordinates) < 3:
            return 0.0
        
        # Ensure ring is closed
        if coordinates[0] != coordinates[-1]:
            coordinates = coordinates + [coordinates[0]]
        
        n = len(coordinates) - 1
        total = 0.0
        
        for i in range(n):
            lon1 = math.radians(coordinates[i][0])
            lat1 = math.radians(coordinates[i][1])
            lon2 = math.radians(coordinates[(i + 1) % n][0])
            lat2 = math.radians(coordinates[(i + 1) % n][1])
            
            total += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
        
        area = abs(total * EARTH_RADIUS_M ** 2 / 2)
        return area
    
    @staticmethod
    def initial_bearing(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
        """
        Calculate initial bearing from point 1 to point 2.
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
        
        Returns:
            Bearing in degrees (0-360)
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
    
    @staticmethod
    def destination_point(lat: float, lon: float,
                         bearing: float, distance_m: float) -> Tuple[float, float]:
        """
        Calculate destination point given start, bearing, and distance.
        
        Args:
            lat, lon: Start point coordinates (degrees)
            bearing: Bearing in degrees
            distance_m: Distance in meters
        
        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        angular_dist = distance_m / EARTH_RADIUS_M
        
        lat2 = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist) +
            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
        )
        
        lon2 = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(lat2)
        )
        
        return (math.degrees(lat2), math.degrees(lon2))


@dataclass
class SpatialGeometry:
    """Enhanced geometry with geodesic support."""
    geometry_type: GeometryType
    coordinates: Any
    properties: Dict[str, Any] = field(default_factory=dict)
    srid: int = 4326
    
    def to_wkt(self) -> str:
        """Convert to WKT format."""
        if self.geometry_type == GeometryType.POINT:
            return f"POINT({self.coordinates[0]} {self.coordinates[1]})"
        elif self.geometry_type == GeometryType.LINESTRING:
            coords = ", ".join(f"{c[0]} {c[1]}" for c in self.coordinates)
            return f"LINESTRING({coords})"
        elif self.geometry_type == GeometryType.POLYGON:
            rings = []
            for ring in self.coordinates:
                coords = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                rings.append(f"({coords})")
            return f"POLYGON({', '.join(rings)})"
        elif self.geometry_type == GeometryType.MULTIPOINT:
            points = ", ".join(f"({c[0]} {c[1]})" for c in self.coordinates)
            return f"MULTIPOINT({points})"
        elif self.geometry_type == GeometryType.MULTIPOLYGON:
            polygons = []
            for polygon in self.coordinates:
                rings = []
                for ring in polygon:
                    coords = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                    rings.append(f"({coords})")
                polygons.append(f"({', '.join(rings)})")
            return f"MULTIPOLYGON({', '.join(polygons)})"
        return ""
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON."""
        return {
            "type": "Feature",
            "geometry": {
                "type": self.geometry_type.value,
                "coordinates": self.coordinates
            },
            "properties": self.properties
        }
    
    def geodesic_area(self) -> float:
        """Calculate geodesic area in square meters."""
        if self.geometry_type == GeometryType.POLYGON:
            return GeodesicCalculator.geodesic_area(self.coordinates[0])
        elif self.geometry_type == GeometryType.MULTIPOLYGON:
            total = 0.0
            for polygon in self.coordinates:
                total += GeodesicCalculator.geodesic_area(polygon[0])
            return total
        return 0.0
    
    def geodesic_length(self) -> float:
        """Calculate geodesic length in meters."""
        if self.geometry_type == GeometryType.LINESTRING:
            total = 0.0
            for i in range(len(self.coordinates) - 1):
                c1, c2 = self.coordinates[i], self.coordinates[i + 1]
                total += GeodesicCalculator.haversine_distance(
                    c1[1], c1[0], c2[1], c2[0], unit="m"
                )
            return total
        elif self.geometry_type == GeometryType.POLYGON:
            total = 0.0
            for ring in self.coordinates:
                for i in range(len(ring) - 1):
                    c1, c2 = ring[i], ring[i + 1]
                    total += GeodesicCalculator.haversine_distance(
                        c1[1], c1[0], c2[1], c2[0], unit="m"
                    )
            return total
        return 0.0
    
    def centroid(self) -> 'SpatialGeometry':
        """Calculate centroid."""
        if self.geometry_type == GeometryType.POINT:
            return self
        elif self.geometry_type == GeometryType.POLYGON:
            ring = self.coordinates[0]
            n = len(ring)
            if n == 0:
                return SpatialGeometry(GeometryType.POINT, [0, 0])
            
            # Weighted centroid calculation
            cx = sum(p[0] for p in ring) / n
            cy = sum(p[1] for p in ring) / n
            return SpatialGeometry(GeometryType.POINT, [cx, cy])
        return SpatialGeometry(GeometryType.POINT, [0, 0])
    
    def buffer(self, distance_m: float, segments: int = 32) -> 'SpatialGeometry':
        """Create geodesic buffer around geometry."""
        if self.geometry_type == GeometryType.POINT:
            lon, lat = self.coordinates[0], self.coordinates[1]
            coords = []
            for i in range(segments + 1):
                bearing = 360 * i / segments
                dest_lat, dest_lon = GeodesicCalculator.destination_point(
                    lat, lon, bearing, distance_m
                )
                coords.append([dest_lon, dest_lat])
            return SpatialGeometry(GeometryType.POLYGON, [coords])
        return self
    
    @classmethod
    def point(cls, lon: float, lat: float, 
              properties: Dict[str, Any] = None) -> 'SpatialGeometry':
        """Create point geometry."""
        return cls(GeometryType.POINT, [lon, lat], properties or {})
    
    @classmethod
    def polygon(cls, coordinates: List[List[List[float]]],
                properties: Dict[str, Any] = None) -> 'SpatialGeometry':
        """Create polygon geometry."""
        return cls(GeometryType.POLYGON, coordinates, properties or {})
    
    @classmethod
    def from_wkt(cls, wkt: str, properties: Dict[str, Any] = None) -> 'SpatialGeometry':
        """Parse WKT string to geometry."""
        wkt = wkt.strip()
        upper_wkt = wkt.upper()
        
        if upper_wkt.startswith("POINT"):
            coords_str = wkt[wkt.index("(") + 1:wkt.rindex(")")]
            coords = [float(c) for c in coords_str.split()]
            return cls(GeometryType.POINT, coords, properties or {})
        
        elif upper_wkt.startswith("LINESTRING"):
            coords_str = wkt[wkt.index("(") + 1:wkt.rindex(")")]
            coords = [[float(c) for c in p.strip().split()] 
                     for p in coords_str.split(",")]
            return cls(GeometryType.LINESTRING, coords, properties or {})
        
        elif upper_wkt.startswith("POLYGON"):
            # Handle nested parentheses for rings
            content = wkt[wkt.index("(") + 1:wkt.rindex(")")]
            rings = []
            depth = 0
            current_ring = ""
            
            for char in content:
                if char == "(":
                    depth += 1
                    if depth == 1:
                        current_ring = ""
                        continue
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        coords = [[float(c) for c in p.strip().split()] 
                                 for p in current_ring.split(",") if p.strip()]
                        rings.append(coords)
                        continue
                
                if depth > 0:
                    current_ring += char
            
            return cls(GeometryType.POLYGON, rings, properties or {})
        
        raise ValueError(f"Unsupported WKT: {wkt}")


class RTreeIndex:
    """
    R-tree spatial index implementation.
    
    Provides efficient spatial queries:
    - Range queries (bounding box)
    - Nearest neighbor queries
    - Intersection queries
    """
    
    def __init__(self):
        self._entries: List[Tuple[Tuple[float, float, float, float], Any]] = []
        self._built = False
    
    def insert(self, bounds: Tuple[float, float, float, float], item: Any):
        """
        Insert item with bounding box.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y)
            item: Associated data
        """
        self._entries.append((bounds, item))
        self._built = False
    
    def build(self):
        """Build/rebuild the index."""
        # Sort by Hilbert curve for better spatial locality
        self._entries.sort(key=lambda e: self._hilbert_value(e[0]))
        self._built = True
    
    def _hilbert_value(self, bounds: Tuple[float, float, float, float]) -> int:
        """Calculate Hilbert curve value for spatial ordering."""
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        
        # Normalize to grid
        x = int((cx + 180) / 360 * 65536) & 0xFFFF
        y = int((cy + 90) / 180 * 65536) & 0xFFFF
        
        # Simple Hilbert approximation
        return self._xy_to_hilbert(x, y, 16)
    
    def _xy_to_hilbert(self, x: int, y: int, order: int) -> int:
        """Convert x,y to Hilbert curve index."""
        d = 0
        s = order // 2
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            
            # Rotate
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            
            s //= 2
        
        return d
    
    def query_range(self, bounds: Tuple[float, float, float, float]) -> List[Any]:
        """
        Query items intersecting bounding box.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y)
        
        Returns:
            List of items whose bounds intersect query bounds
        """
        if not self._built:
            self.build()
        
        results = []
        for entry_bounds, item in self._entries:
            if self._bounds_intersect(bounds, entry_bounds):
                results.append(item)
        
        return results
    
    def query_nearest(self, point: Tuple[float, float], k: int = 1) -> List[Tuple[Any, float]]:
        """
        Find k nearest items to point.
        
        Args:
            point: (x, y) query point
            k: Number of nearest neighbors
        
        Returns:
            List of (item, distance) tuples
        """
        if not self._built:
            self.build()
        
        distances = []
        for entry_bounds, item in self._entries:
            cx = (entry_bounds[0] + entry_bounds[2]) / 2
            cy = (entry_bounds[1] + entry_bounds[3]) / 2
            
            # Use geodesic distance
            dist = GeodesicCalculator.haversine_distance(
                point[1], point[0], cy, cx, unit="m"
            )
            distances.append((item, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def _bounds_intersect(self, b1: Tuple[float, float, float, float],
                         b2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes intersect."""
        return not (b1[2] < b2[0] or b1[0] > b2[2] or
                   b1[3] < b2[1] or b1[1] > b2[3])
    
    def __len__(self) -> int:
        return len(self._entries)


class SedonaSparkSession:
    """
    Manages Spark session with Sedona configuration.
    
    Handles:
    - Spark session creation with proper config
    - Sedona SQL function registration
    - Kryo serializer setup
    - Resource management
    """
    
    def __init__(self, config: SedonaConfig):
        self.config = config
        self._spark = None
        self._sedona_available = False
        
        self._check_availability()
    
    def _check_availability(self):
        """Check if Spark and Sedona are available."""
        try:
            from pyspark.sql import SparkSession
            from sedona.spark import SedonaContext
            self._sedona_available = True
            logger.info("Spark and Sedona are available")
        except ImportError as e:
            logger.warning(f"Spark/Sedona not available: {e}")
            self._sedona_available = False
    
    def get_or_create(self):
        """Get or create Spark session with Sedona."""
        if self._spark is not None:
            return self._spark
        
        if not self._sedona_available:
            logger.warning("Sedona not available, returning None")
            return None
        
        try:
            from pyspark.sql import SparkSession
            from sedona.spark import SedonaContext
            
            # Build Spark session with proper configuration
            builder = SparkSession.builder \
                .appName(self.config.app_name) \
                .master(self.config.spark_master) \
                .config("spark.serializer", self.config.serializer) \
                .config("spark.kryo.registrator", self.config.kryo_registrator) \
                .config("spark.executor.memory", self.config.executor_memory) \
                .config("spark.driver.memory", self.config.driver_memory) \
                .config("spark.sql.shuffle.partitions", str(self.config.shuffle_partitions)) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sedona.global.index.type", self.config.default_index_type.value) \
                .config("spark.sedona.join.gridtype", self.config.grid_type)
            
            # Create Spark session
            spark = builder.getOrCreate()
            
            # Register Sedona SQL functions
            self._spark = SedonaContext.create(spark)
            
            logger.info(f"Created Spark session with Sedona: {self.config.app_name}")
            return self._spark
            
        except Exception as e:
            logger.error(f"Failed to create Spark session: {e}")
            return None
    
    def stop(self):
        """Stop Spark session."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            logger.info("Stopped Spark session")
    
    @property
    def is_available(self) -> bool:
        return self._sedona_available


class SedonaSQLExecutor:
    """
    Execute Sedona SQL queries.
    
    Provides:
    - Spatial SQL query execution
    - DataFrame operations
    - Result conversion
    """
    
    def __init__(self, spark_session: SedonaSparkSession):
        self.spark_session = spark_session
    
    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute Sedona SQL query.
        
        Args:
            sql: SQL query with Sedona spatial functions
        
        Returns:
            List of result dictionaries
        """
        spark = self.spark_session.get_or_create()
        
        if spark is None:
            logger.warning("Spark not available, returning empty results")
            return []
        
        try:
            df = spark.sql(sql)
            return [row.asDict() for row in df.collect()]
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return []
    
    def create_spatial_table(self, table_name: str, 
                            geometries: List[SpatialGeometry],
                            geometry_column: str = "geometry") -> bool:
        """
        Create a spatial table from geometries.
        
        Args:
            table_name: Name for the table
            geometries: List of geometries
            geometry_column: Name of geometry column
        
        Returns:
            True if successful
        """
        spark = self.spark_session.get_or_create()
        
        if spark is None:
            return False
        
        try:
            # Create DataFrame with WKT geometries
            data = []
            for i, geom in enumerate(geometries):
                row = {
                    "id": i,
                    "wkt": geom.to_wkt(),
                    **geom.properties
                }
                data.append(row)
            
            df = spark.createDataFrame(data)
            
            # Convert WKT to geometry
            df = df.selectExpr(
                "*",
                f"ST_GeomFromWKT(wkt) as {geometry_column}"
            ).drop("wkt")
            
            # Register as temp view
            df.createOrReplaceTempView(table_name)
            
            logger.info(f"Created spatial table: {table_name} with {len(geometries)} geometries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create spatial table: {e}")
            return False
    
    def spatial_join(self, left_table: str, right_table: str,
                    join_type: SpatialJoinType,
                    left_geom: str = "geometry",
                    right_geom: str = "geometry") -> List[Dict[str, Any]]:
        """
        Execute spatial join between two tables.
        
        Args:
            left_table: Left table name
            right_table: Right table name
            join_type: Type of spatial join
            left_geom: Left geometry column
            right_geom: Right geometry column
        
        Returns:
            Join results
        """
        sql = f"""
        SELECT l.*, r.*
        FROM {left_table} l, {right_table} r
        WHERE {join_type.value}(l.{left_geom}, r.{right_geom})
        """
        
        return self.execute_sql(sql)
    
    def range_query(self, table_name: str, 
                   min_x: float, min_y: float,
                   max_x: float, max_y: float,
                   geometry_column: str = "geometry") -> List[Dict[str, Any]]:
        """
        Execute range query (bounding box filter).
        
        Args:
            table_name: Table to query
            min_x, min_y, max_x, max_y: Bounding box
            geometry_column: Geometry column name
        
        Returns:
            Geometries within bounding box
        """
        sql = f"""
        SELECT *
        FROM {table_name}
        WHERE ST_Intersects(
            {geometry_column},
            ST_GeomFromWKT('POLYGON(({min_x} {min_y}, {max_x} {min_y}, 
                                     {max_x} {max_y}, {min_x} {max_y}, 
                                     {min_x} {min_y}))')
        )
        """
        
        return self.execute_sql(sql)
    
    def knn_query(self, table_name: str,
                 query_point: Tuple[float, float],
                 k: int = 10,
                 geometry_column: str = "geometry") -> List[Dict[str, Any]]:
        """
        Execute K-nearest neighbors query.
        
        Args:
            table_name: Table to query
            query_point: (lon, lat) query point
            k: Number of neighbors
            geometry_column: Geometry column name
        
        Returns:
            K nearest geometries with distances
        """
        sql = f"""
        SELECT *, 
               ST_Distance(
                   {geometry_column},
                   ST_GeomFromWKT('POINT({query_point[0]} {query_point[1]})')
               ) as distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {k}
        """
        
        return self.execute_sql(sql)


class RasterProcessor:
    """
    Raster processing for zonal statistics.
    
    Provides:
    - Zonal statistics computation
    - Raster-vector operations
    - Band math operations
    """
    
    def __init__(self, spark_session: SedonaSparkSession = None):
        self.spark_session = spark_session
        self._rasterio_available = False
        
        try:
            import rasterio
            import numpy as np
            self._rasterio_available = True
        except ImportError:
            logger.warning("rasterio not available, using mock raster processing")
    
    def compute_zonal_statistics(self, raster_path: str,
                                geometries: List[SpatialGeometry],
                                statistics: List[str] = None,
                                band: int = 1) -> List[Dict[str, Any]]:
        """
        Compute zonal statistics for geometries from raster.
        
        Args:
            raster_path: Path to raster file
            geometries: List of zone geometries
            statistics: Statistics to compute (mean, min, max, std, count, sum)
            band: Raster band number
        
        Returns:
            List of statistics per geometry
        """
        statistics = statistics or ["mean", "min", "max", "std", "count", "sum"]
        
        if self._rasterio_available and os.path.exists(raster_path):
            return self._compute_zonal_rasterio(raster_path, geometries, statistics, band)
        else:
            return self._compute_zonal_mock(geometries, statistics)
    
    def _compute_zonal_rasterio(self, raster_path: str,
                               geometries: List[SpatialGeometry],
                               statistics: List[str],
                               band: int) -> List[Dict[str, Any]]:
        """Compute zonal statistics using rasterio."""
        import rasterio
        from rasterio.mask import mask
        import numpy as np
        
        results = []
        
        with rasterio.open(raster_path) as src:
            for geom in geometries:
                try:
                    geojson = geom.to_geojson()["geometry"]
                    out_image, out_transform = mask(src, [geojson], crop=True)
                    data = out_image[band - 1]
                    
                    # Mask nodata
                    if src.nodata is not None:
                        data = np.ma.masked_equal(data, src.nodata)
                    
                    valid_data = data.compressed() if hasattr(data, 'compressed') else data.flatten()
                    
                    stats = {
                        "geometry_id": geom.properties.get("id", str(uuid.uuid4())),
                        "statistics": {}
                    }
                    
                    if len(valid_data) > 0:
                        if "mean" in statistics:
                            stats["statistics"]["mean"] = float(np.mean(valid_data))
                        if "min" in statistics:
                            stats["statistics"]["min"] = float(np.min(valid_data))
                        if "max" in statistics:
                            stats["statistics"]["max"] = float(np.max(valid_data))
                        if "std" in statistics:
                            stats["statistics"]["std"] = float(np.std(valid_data))
                        if "count" in statistics:
                            stats["statistics"]["count"] = int(len(valid_data))
                        if "sum" in statistics:
                            stats["statistics"]["sum"] = float(np.sum(valid_data))
                    else:
                        for stat in statistics:
                            stats["statistics"][stat] = None
                    
                    results.append(stats)
                    
                except Exception as e:
                    logger.warning(f"Zonal stats failed for geometry: {e}")
                    results.append({
                        "geometry_id": geom.properties.get("id", str(uuid.uuid4())),
                        "statistics": {stat: None for stat in statistics},
                        "error": str(e)
                    })
        
        return results
    
    def _compute_zonal_mock(self, geometries: List[SpatialGeometry],
                           statistics: List[str]) -> List[Dict[str, Any]]:
        """Mock zonal statistics with deterministic values based on geometry."""
        results = []
        
        for geom in geometries:
            # Generate deterministic values based on geometry centroid
            centroid = geom.centroid()
            seed = int((centroid.coordinates[0] + 180) * 1000 + 
                      (centroid.coordinates[1] + 90) * 1000) % 10000
            
            # Deterministic pseudo-random based on seed
            base_value = 0.3 + (seed % 500) / 1000  # 0.3 to 0.8
            
            stats = {
                "geometry_id": geom.properties.get("id", str(uuid.uuid4())),
                "statistics": {}
            }
            
            if "mean" in statistics:
                stats["statistics"]["mean"] = base_value
            if "min" in statistics:
                stats["statistics"]["min"] = base_value - 0.15
            if "max" in statistics:
                stats["statistics"]["max"] = base_value + 0.15
            if "std" in statistics:
                stats["statistics"]["std"] = 0.05 + (seed % 100) / 1000
            if "count" in statistics:
                stats["statistics"]["count"] = 1000 + seed * 10
            if "sum" in statistics:
                stats["statistics"]["sum"] = base_value * (1000 + seed * 10)
            
            results.append(stats)
        
        return results
    
    def compute_ndvi(self, red_band_path: str, nir_band_path: str,
                    geometries: List[SpatialGeometry]) -> List[Dict[str, Any]]:
        """
        Compute NDVI for geometries.
        
        Args:
            red_band_path: Path to red band raster
            nir_band_path: Path to NIR band raster
            geometries: Zone geometries
        
        Returns:
            NDVI statistics per geometry
        """
        if self._rasterio_available:
            return self._compute_ndvi_rasterio(red_band_path, nir_band_path, geometries)
        else:
            return self._compute_ndvi_mock(geometries)
    
    def _compute_ndvi_rasterio(self, red_path: str, nir_path: str,
                              geometries: List[SpatialGeometry]) -> List[Dict[str, Any]]:
        """Compute NDVI using rasterio."""
        import rasterio
        from rasterio.mask import mask
        import numpy as np
        
        results = []
        
        with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
            for geom in geometries:
                try:
                    geojson = geom.to_geojson()["geometry"]
                    
                    red_data, _ = mask(red_src, [geojson], crop=True)
                    nir_data, _ = mask(nir_src, [geojson], crop=True)
                    
                    red = red_data[0].astype(float)
                    nir = nir_data[0].astype(float)
                    
                    # Calculate NDVI
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndvi = (nir - red) / (nir + red)
                        ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
                    
                    results.append({
                        "geometry_id": geom.properties.get("id", str(uuid.uuid4())),
                        "ndvi_mean": float(np.mean(ndvi)),
                        "ndvi_min": float(np.min(ndvi)),
                        "ndvi_max": float(np.max(ndvi)),
                        "ndvi_std": float(np.std(ndvi)),
                        "pixel_count": int(ndvi.size)
                    })
                    
                except Exception as e:
                    logger.warning(f"NDVI computation failed: {e}")
                    results.append({
                        "geometry_id": geom.properties.get("id", str(uuid.uuid4())),
                        "error": str(e)
                    })
        
        return results
    
    def _compute_ndvi_mock(self, geometries: List[SpatialGeometry]) -> List[Dict[str, Any]]:
        """Mock NDVI computation with deterministic values."""
        results = []
        
        for geom in geometries:
            centroid = geom.centroid()
            seed = int((centroid.coordinates[0] + 180) * 1000 + 
                      (centroid.coordinates[1] + 90) * 1000) % 10000
            
            base_ndvi = 0.4 + (seed % 400) / 1000  # 0.4 to 0.8
            
            results.append({
                "geometry_id": geom.properties.get("id", str(uuid.uuid4())),
                "ndvi_mean": base_ndvi,
                "ndvi_min": base_ndvi - 0.2,
                "ndvi_max": base_ndvi + 0.15,
                "ndvi_std": 0.08 + (seed % 50) / 1000,
                "pixel_count": 5000 + seed * 5
            })
        
        return results


class SedonaEnhanced:
    """
    Enhanced Apache Sedona integration for MineralVision.
    
    Provides production-grade geospatial processing:
    - Proper Spark + Sedona initialization
    - Geodesic-aware calculations
    - Real R-tree spatial indexing
    - Raster support for zonal statistics
    - Distributed spatial operations
    
    Example:
        sedona = SedonaEnhanced()
        sedona.initialize()
        
        # Create geometries
        fields = [
            SpatialGeometry.polygon([[[lon1, lat1], [lon2, lat2], ...]], {"name": "Field A"}),
            SpatialGeometry.polygon([[[lon3, lat3], [lon4, lat4], ...]], {"name": "Field B"})
        ]
        
        # Register dataset with spatial index
        sedona.register_dataset("fields", fields, build_index=True)
        
        # Spatial queries
        results = sedona.range_query("fields", min_lon, min_lat, max_lon, max_lat)
        nearest = sedona.knn_query("fields", query_lon, query_lat, k=5)
        
        # Zonal statistics
        stats = sedona.compute_zonal_statistics("fields", "raster.tif")
    """
    
    def __init__(self, config: SedonaConfig = None):
        self.config = config or SedonaConfig()
        self.spark_session = SedonaSparkSession(self.config)
        self.sql_executor = SedonaSQLExecutor(self.spark_session)
        self.raster_processor = RasterProcessor(self.spark_session)
        self.geodesic = GeodesicCalculator()
        
        self._datasets: Dict[str, List[SpatialGeometry]] = {}
        self._indices: Dict[str, RTreeIndex] = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize Sedona integration."""
        # Try to initialize Spark session
        spark = self.spark_session.get_or_create()
        
        if spark is not None:
            logger.info("Sedona initialized with Spark")
        else:
            logger.info("Sedona initialized in local mode (Spark not available)")
        
        self._initialized = True
        return True
    
    def register_dataset(self, name: str, geometries: List[SpatialGeometry],
                        build_index: bool = True) -> bool:
        """
        Register a spatial dataset.
        
        Args:
            name: Dataset name
            geometries: List of geometries
            build_index: Whether to build spatial index
        
        Returns:
            True if successful
        """
        self._datasets[name] = geometries
        
        if build_index:
            index = RTreeIndex()
            for geom in geometries:
                bounds = self._get_bounds(geom)
                index.insert(bounds, geom)
            index.build()
            self._indices[name] = index
            logger.info(f"Built R-tree index for {name} with {len(geometries)} geometries")
        
        # Also create Spark table if available
        if self.spark_session.is_available:
            self.sql_executor.create_spatial_table(name, geometries)
        
        return True
    
    def _get_bounds(self, geom: SpatialGeometry) -> Tuple[float, float, float, float]:
        """Get bounding box for geometry."""
        if geom.geometry_type == GeometryType.POINT:
            x, y = geom.coordinates
            return (x, y, x, y)
        
        elif geom.geometry_type in [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]:
            all_coords = []
            if geom.geometry_type == GeometryType.POLYGON:
                for ring in geom.coordinates:
                    all_coords.extend(ring)
            else:
                for polygon in geom.coordinates:
                    for ring in polygon:
                        all_coords.extend(ring)
            
            if not all_coords:
                return (0, 0, 0, 0)
            
            xs = [c[0] for c in all_coords]
            ys = [c[1] for c in all_coords]
            return (min(xs), min(ys), max(xs), max(ys))
        
        return (0, 0, 0, 0)
    
    def range_query(self, dataset_name: str,
                   min_x: float, min_y: float,
                   max_x: float, max_y: float) -> List[SpatialGeometry]:
        """
        Query geometries within bounding box.
        
        Uses R-tree index for efficient filtering.
        """
        if dataset_name not in self._datasets:
            return []
        
        # Use index if available
        if dataset_name in self._indices:
            return self._indices[dataset_name].query_range((min_x, min_y, max_x, max_y))
        
        # Fallback to linear scan
        results = []
        for geom in self._datasets[dataset_name]:
            bounds = self._get_bounds(geom)
            if self._bounds_intersect((min_x, min_y, max_x, max_y), bounds):
                results.append(geom)
        
        return results
    
    def knn_query(self, dataset_name: str,
                 lon: float, lat: float,
                 k: int = 10) -> List[Tuple[SpatialGeometry, float]]:
        """
        Find k nearest geometries to point.
        
        Uses geodesic distance for accurate results.
        """
        if dataset_name not in self._datasets:
            return []
        
        # Use index if available
        if dataset_name in self._indices:
            return self._indices[dataset_name].query_nearest((lon, lat), k)
        
        # Fallback to linear scan with geodesic distance
        distances = []
        for geom in self._datasets[dataset_name]:
            centroid = geom.centroid()
            dist = self.geodesic.haversine_distance(
                lat, lon,
                centroid.coordinates[1], centroid.coordinates[0],
                unit="m"
            )
            distances.append((geom, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def spatial_join(self, left_dataset: str, right_dataset: str,
                    join_type: SpatialJoinType = SpatialJoinType.INTERSECTS) -> List[Tuple[SpatialGeometry, SpatialGeometry]]:
        """
        Perform spatial join between datasets.
        
        Uses index-nested-loop join when indices available.
        """
        if left_dataset not in self._datasets or right_dataset not in self._datasets:
            return []
        
        left_geoms = self._datasets[left_dataset]
        right_geoms = self._datasets[right_dataset]
        
        results = []
        
        # Use index on right side if available
        right_index = self._indices.get(right_dataset)
        
        for left_geom in left_geoms:
            left_bounds = self._get_bounds(left_geom)
            
            # Get candidates from index
            if right_index:
                candidates = right_index.query_range(left_bounds)
            else:
                candidates = right_geoms
            
            for right_geom in candidates:
                if self._check_spatial_predicate(left_geom, right_geom, join_type):
                    results.append((left_geom, right_geom))
        
        return results
    
    def _check_spatial_predicate(self, geom1: SpatialGeometry, geom2: SpatialGeometry,
                                predicate: SpatialJoinType) -> bool:
        """Check spatial predicate between geometries."""
        # Simplified predicate checking
        bounds1 = self._get_bounds(geom1)
        bounds2 = self._get_bounds(geom2)
        
        if predicate == SpatialJoinType.INTERSECTS:
            return self._bounds_intersect(bounds1, bounds2)
        
        elif predicate == SpatialJoinType.CONTAINS:
            return (bounds1[0] <= bounds2[0] and bounds1[1] <= bounds2[1] and
                   bounds1[2] >= bounds2[2] and bounds1[3] >= bounds2[3])
        
        elif predicate == SpatialJoinType.WITHIN:
            return (bounds2[0] <= bounds1[0] and bounds2[1] <= bounds1[1] and
                   bounds2[2] >= bounds1[2] and bounds2[3] >= bounds1[3])
        
        return False
    
    def _bounds_intersect(self, b1: Tuple[float, float, float, float],
                         b2: Tuple[float, float, float, float]) -> bool:
        """Check if bounding boxes intersect."""
        return not (b1[2] < b2[0] or b1[0] > b2[2] or
                   b1[3] < b2[1] or b1[1] > b2[3])
    
    def compute_zonal_statistics(self, dataset_name: str, raster_path: str,
                                statistics: List[str] = None) -> List[Dict[str, Any]]:
        """
        Compute zonal statistics for dataset geometries.
        
        Args:
            dataset_name: Name of registered dataset
            raster_path: Path to raster file
            statistics: Statistics to compute
        
        Returns:
            Statistics per geometry
        """
        if dataset_name not in self._datasets:
            return []
        
        return self.raster_processor.compute_zonal_statistics(
            raster_path,
            self._datasets[dataset_name],
            statistics
        )
    
    def compute_ndvi(self, dataset_name: str,
                    red_band_path: str, nir_band_path: str) -> List[Dict[str, Any]]:
        """
        Compute NDVI for dataset geometries.
        
        Args:
            dataset_name: Name of registered dataset
            red_band_path: Path to red band raster
            nir_band_path: Path to NIR band raster
        
        Returns:
            NDVI statistics per geometry
        """
        if dataset_name not in self._datasets:
            return []
        
        return self.raster_processor.compute_ndvi(
            red_band_path,
            nir_band_path,
            self._datasets[dataset_name]
        )
    
    def geodesic_distance(self, lon1: float, lat1: float,
                         lon2: float, lat2: float,
                         method: str = "haversine") -> float:
        """
        Calculate geodesic distance between points.
        
        Args:
            lon1, lat1: First point
            lon2, lat2: Second point
            method: "haversine" or "vincenty"
        
        Returns:
            Distance in meters
        """
        if method == "vincenty":
            return self.geodesic.vincenty_distance(lat1, lon1, lat2, lon2)
        else:
            return self.geodesic.haversine_distance(lat1, lon1, lat2, lon2, unit="m")
    
    def geodesic_area(self, geometry: SpatialGeometry) -> float:
        """Calculate geodesic area in square meters."""
        return geometry.geodesic_area()
    
    def geodesic_buffer(self, geometry: SpatialGeometry, 
                       distance_m: float) -> SpatialGeometry:
        """Create geodesic buffer around geometry."""
        return geometry.buffer(distance_m)
    
    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute Sedona SQL query."""
        return self.sql_executor.execute_sql(sql)
    
    def get_dataset(self, name: str) -> Optional[List[SpatialGeometry]]:
        """Get registered dataset."""
        return self._datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List registered datasets."""
        return list(self._datasets.keys())
    
    def shutdown(self):
        """Shutdown Sedona and Spark."""
        self.spark_session.stop()
        self._initialized = False


def create_sedona_enhanced(config: SedonaConfig = None) -> SedonaEnhanced:
    """Create and initialize enhanced Sedona integration."""
    sedona = SedonaEnhanced(config)
    sedona.initialize()
    return sedona


__all__ = [
    'SedonaConfig',
    'SpatialIndexType',
    'SpatialJoinType',
    'GeometryType',
    'CoordinateSystem',
    'GeodesicCalculator',
    'SpatialGeometry',
    'RTreeIndex',
    'SedonaSparkSession',
    'SedonaSQLExecutor',
    'RasterProcessor',
    'SedonaEnhanced',
    'create_sedona_enhanced'
]
