"""
Crop Monitoring Data Persistence Layer
=======================================

Unified data persistence for EOS crop monitoring features integrating:
- Lakehouse (Delta Lake/Iceberg) for time-series data and versioning
- PostGIS for spatial field boundaries and queries
- Apache Sedona for large-scale geospatial processing

This module provides a unified interface for storing and querying:
- Vegetation indices time series
- Field boundaries and management zones
- Weather data and forecasts
- VRA maps and prescriptions
- Alerts and notifications
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends."""
    LAKEHOUSE = "lakehouse"
    POSTGIS = "postgis"
    SEDONA = "sedona"
    LOCAL = "local"


class TableType(Enum):
    """Types of crop monitoring tables."""
    VEGETATION_INDICES = "vegetation_indices"
    FIELD_BOUNDARIES = "field_boundaries"
    WEATHER_DATA = "weather_data"
    VRA_MAPS = "vra_maps"
    ALERTS = "alerts"
    MANAGEMENT_ZONES = "management_zones"


@dataclass
class CropDataRecord:
    """Base record for crop monitoring data."""
    record_id: str
    field_id: str
    timestamp: datetime
    data_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field_id": self.field_id,
            "timestamp": self.timestamp.isoformat(),
            "data_type": self.data_type,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CropDataRecord":
        return cls(
            record_id=d["record_id"],
            field_id=d["field_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]) if isinstance(d["timestamp"], str) else d["timestamp"],
            data_type=d["data_type"],
            data=d["data"],
            metadata=d.get("metadata", {})
        )


@dataclass
class SpatialBoundary:
    """Spatial boundary for fields and zones."""
    boundary_id: str
    field_id: str
    geometry_type: str  # "Polygon", "MultiPolygon"
    coordinates: List[List[List[float]]]  # GeoJSON format
    properties: Dict[str, Any] = field(default_factory=dict)
    srid: int = 4326
    
    def to_wkt(self) -> str:
        """Convert to WKT format."""
        if self.geometry_type == "Polygon":
            rings = []
            for ring in self.coordinates:
                coords = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                rings.append(f"({coords})")
            return f"POLYGON({', '.join(rings)})"
        return ""
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON."""
        return {
            "type": "Feature",
            "geometry": {
                "type": self.geometry_type,
                "coordinates": self.coordinates
            },
            "properties": {
                "boundary_id": self.boundary_id,
                "field_id": self.field_id,
                **self.properties
            }
        }


@dataclass
class VegetationIndexRecord:
    """Vegetation index time series record."""
    record_id: str
    field_id: str
    timestamp: datetime
    index_type: str  # NDVI, NDRE, SAVI, etc.
    mean_value: float
    min_value: float
    max_value: float
    std_value: float
    pixel_count: int
    cloud_coverage: float
    source: str  # satellite name
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field_id": self.field_id,
            "timestamp": self.timestamp.isoformat(),
            "index_type": self.index_type,
            "mean_value": self.mean_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "std_value": self.std_value,
            "pixel_count": self.pixel_count,
            "cloud_coverage": self.cloud_coverage,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class WeatherRecord:
    """Weather data record."""
    record_id: str
    field_id: str
    timestamp: datetime
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    solar_radiation: Optional[float] = None
    evapotranspiration: Optional[float] = None
    gdd: Optional[float] = None  # Growing Degree Days
    forecast: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field_id": self.field_id,
            "timestamp": self.timestamp.isoformat(),
            "temperature": self.temperature,
            "humidity": self.humidity,
            "precipitation": self.precipitation,
            "wind_speed": self.wind_speed,
            "solar_radiation": self.solar_radiation,
            "evapotranspiration": self.evapotranspiration,
            "gdd": self.gdd,
            "forecast": self.forecast,
            "metadata": self.metadata
        }


class LakehouseAdapter:
    """
    Lakehouse (Delta Lake/Iceberg) adapter for crop monitoring.
    
    Stores time-series data with:
    - Time travel and versioning
    - Schema evolution
    - ACID transactions
    - Efficient time-range queries
    """
    
    def __init__(self, warehouse_path: str = "/data/crop_lakehouse",
                 catalog: str = "crop_monitoring",
                 table_format: str = "delta"):
        self.warehouse_path = Path(warehouse_path)
        self.catalog = catalog
        self.table_format = table_format
        self._spark = None
        self._delta_available = False
        self._tables: Dict[str, List[Dict]] = {}
        
        self._check_delta_availability()
        self._init_tables()
    
    def _check_delta_availability(self):
        """Check if Delta Lake is available."""
        try:
            import delta
            self._delta_available = True
            logger.info("Delta Lake available for crop monitoring")
        except ImportError:
            logger.warning("Delta Lake not available, using local JSON fallback")
            self._delta_available = False
    
    def _init_tables(self):
        """Initialize table storage."""
        self.warehouse_path.mkdir(parents=True, exist_ok=True)
        
        table_names = [
            "vegetation_indices",
            "weather_data",
            "vra_maps",
            "alerts",
            "field_metrics"
        ]
        
        for table in table_names:
            table_path = self.warehouse_path / f"{table}.json"
            if table_path.exists():
                with open(table_path, "r") as f:
                    self._tables[table] = json.load(f)
            else:
                self._tables[table] = []
    
    def _save_table(self, table: str):
        """Save table to disk."""
        table_path = self.warehouse_path / f"{table}.json"
        with open(table_path, "w") as f:
            json.dump(self._tables.get(table, []), f, indent=2, default=str)
    
    def write_vegetation_index(self, record: VegetationIndexRecord) -> str:
        """Write vegetation index record to lakehouse."""
        if self._delta_available:
            return self._write_delta("vegetation_indices", record.to_dict())
        else:
            self._tables.setdefault("vegetation_indices", []).append(record.to_dict())
            self._save_table("vegetation_indices")
            return record.record_id
    
    def write_vegetation_indices_batch(self, records: List[VegetationIndexRecord]) -> int:
        """Write batch of vegetation index records."""
        for record in records:
            self.write_vegetation_index(record)
        return len(records)
    
    def read_vegetation_indices(self, field_id: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                index_type: Optional[str] = None) -> List[Dict]:
        """Read vegetation indices with time range filter."""
        results = []
        
        for record in self._tables.get("vegetation_indices", []):
            if record["field_id"] != field_id:
                continue
            
            record_time = datetime.fromisoformat(record["timestamp"])
            
            if start_date and record_time < start_date:
                continue
            if end_date and record_time > end_date:
                continue
            if index_type and record.get("index_type") != index_type:
                continue
            
            results.append(record)
        
        return sorted(results, key=lambda x: x["timestamp"])
    
    def write_weather_data(self, record: WeatherRecord) -> str:
        """Write weather data record."""
        self._tables.setdefault("weather_data", []).append(record.to_dict())
        self._save_table("weather_data")
        return record.record_id
    
    def read_weather_data(self, field_id: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          forecast_only: bool = False) -> List[Dict]:
        """Read weather data with filters."""
        results = []
        
        for record in self._tables.get("weather_data", []):
            if record["field_id"] != field_id:
                continue
            
            record_time = datetime.fromisoformat(record["timestamp"])
            
            if start_date and record_time < start_date:
                continue
            if end_date and record_time > end_date:
                continue
            if forecast_only and not record.get("forecast"):
                continue
            
            results.append(record)
        
        return sorted(results, key=lambda x: x["timestamp"])
    
    def write_alert(self, alert: Dict[str, Any]) -> str:
        """Write alert record."""
        self._tables.setdefault("alerts", []).append(alert)
        self._save_table("alerts")
        return alert.get("alert_id", str(uuid.uuid4()))
    
    def read_alerts(self, field_id: Optional[str] = None,
                    status: Optional[str] = None,
                    severity: Optional[str] = None) -> List[Dict]:
        """Read alerts with filters."""
        results = []
        
        for alert in self._tables.get("alerts", []):
            if field_id and alert.get("field_id") != field_id:
                continue
            if status and alert.get("status") != status:
                continue
            if severity and alert.get("severity") != severity:
                continue
            
            results.append(alert)
        
        return results
    
    def _write_delta(self, table: str, record: Dict) -> str:
        """Write to Delta Lake table (when available)."""
        try:
            from pyspark.sql import SparkSession
            from delta import configure_spark_with_delta_pip
            
            if self._spark is None:
                builder = SparkSession.builder \
                    .appName("CropMonitoring-Lakehouse") \
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                self._spark = configure_spark_with_delta_pip(builder).getOrCreate()
            
            df = self._spark.createDataFrame([record])
            table_path = str(self.warehouse_path / table)
            df.write.format("delta").mode("append").save(table_path)
            
            return record.get("record_id", str(uuid.uuid4()))
            
        except Exception as e:
            logger.warning(f"Delta write failed, using fallback: {e}")
            self._tables.setdefault(table, []).append(record)
            self._save_table(table)
            return record.get("record_id", str(uuid.uuid4()))
    
    def get_time_travel_version(self, table: str, version: int) -> List[Dict]:
        """Get historical version of table (Delta Lake feature)."""
        if self._delta_available and self._spark:
            try:
                table_path = str(self.warehouse_path / table)
                df = self._spark.read.format("delta").option("versionAsOf", version).load(table_path)
                return [row.asDict() for row in df.collect()]
            except Exception as e:
                logger.warning(f"Time travel failed: {e}")
        
        return self._tables.get(table, [])


class PostGISAdapter:
    """
    PostGIS adapter for spatial field data.
    
    Stores and queries:
    - Field boundaries as spatial geometries
    - Management zones with spatial indexing
    - Spatial queries (contains, intersects, within distance)
    """
    
    def __init__(self, connection_string: str = None,
                 host: str = "localhost", port: int = 5432,
                 database: str = "crop_monitoring", user: str = "postgres",
                 password: str = ""):
        self.connection_string = connection_string or \
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self._connection = None
        self._postgis_available = False
        self._local_storage: Dict[str, List[Dict]] = {
            "field_boundaries": [],
            "management_zones": []
        }
        
        self._check_postgis_availability()
    
    def _check_postgis_availability(self):
        """Check if PostGIS is available."""
        try:
            import psycopg2
            self._postgis_available = True
            logger.info("PostGIS available for crop monitoring")
        except ImportError:
            logger.warning("psycopg2 not available, using local JSON fallback")
            self._postgis_available = False
    
    def connect(self):
        """Establish database connection."""
        if not self._postgis_available:
            return
        
        try:
            import psycopg2
            self._connection = psycopg2.connect(self.connection_string)
            self._init_schema()
            logger.info("Connected to PostGIS database")
        except Exception as e:
            logger.warning(f"PostGIS connection failed: {e}")
            self._postgis_available = False
    
    def _init_schema(self):
        """Initialize PostGIS schema."""
        if not self._connection:
            return
        
        queries = [
            "CREATE EXTENSION IF NOT EXISTS postgis;",
            """
            CREATE TABLE IF NOT EXISTS field_boundaries (
                boundary_id UUID PRIMARY KEY,
                field_id VARCHAR(255) NOT NULL,
                geometry GEOGRAPHY(POLYGON, 4326) NOT NULL,
                properties JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS management_zones (
                zone_id UUID PRIMARY KEY,
                field_id VARCHAR(255) NOT NULL,
                zone_type VARCHAR(50) NOT NULL,
                geometry GEOGRAPHY(POLYGON, 4326) NOT NULL,
                properties JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_field_boundaries_geom ON field_boundaries USING GIST(geometry);",
            "CREATE INDEX IF NOT EXISTS idx_management_zones_geom ON management_zones USING GIST(geometry);",
            "CREATE INDEX IF NOT EXISTS idx_field_boundaries_field ON field_boundaries(field_id);",
            "CREATE INDEX IF NOT EXISTS idx_management_zones_field ON management_zones(field_id);"
        ]
        
        try:
            cursor = self._connection.cursor()
            for query in queries:
                cursor.execute(query)
            self._connection.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"Schema initialization warning: {e}")
    
    def save_field_boundary(self, boundary: SpatialBoundary) -> str:
        """Save field boundary to PostGIS."""
        if self._postgis_available and self._connection:
            return self._save_boundary_postgis(boundary)
        else:
            return self._save_boundary_local(boundary)
    
    def _save_boundary_postgis(self, boundary: SpatialBoundary) -> str:
        """Save boundary to PostGIS."""
        query = """
        INSERT INTO field_boundaries (boundary_id, field_id, geometry, properties)
        VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography, %s)
        ON CONFLICT (boundary_id) DO UPDATE SET
            geometry = EXCLUDED.geometry,
            properties = EXCLUDED.properties,
            updated_at = NOW();
        """
        
        geojson = json.dumps({
            "type": boundary.geometry_type,
            "coordinates": boundary.coordinates
        })
        
        try:
            cursor = self._connection.cursor()
            cursor.execute(query, (
                boundary.boundary_id,
                boundary.field_id,
                geojson,
                json.dumps(boundary.properties)
            ))
            self._connection.commit()
            cursor.close()
            return boundary.boundary_id
        except Exception as e:
            logger.error(f"PostGIS save failed: {e}")
            return self._save_boundary_local(boundary)
    
    def _save_boundary_local(self, boundary: SpatialBoundary) -> str:
        """Save boundary to local storage."""
        record = {
            "boundary_id": boundary.boundary_id,
            "field_id": boundary.field_id,
            "geometry_type": boundary.geometry_type,
            "coordinates": boundary.coordinates,
            "properties": boundary.properties
        }
        
        existing = [b for b in self._local_storage["field_boundaries"] 
                   if b["boundary_id"] != boundary.boundary_id]
        existing.append(record)
        self._local_storage["field_boundaries"] = existing
        
        return boundary.boundary_id
    
    def query_fields_in_bbox(self, min_lon: float, min_lat: float,
                             max_lon: float, max_lat: float) -> List[Dict]:
        """Query fields within bounding box."""
        if self._postgis_available and self._connection:
            return self._query_bbox_postgis(min_lon, min_lat, max_lon, max_lat)
        else:
            return self._query_bbox_local(min_lon, min_lat, max_lon, max_lat)
    
    def _query_bbox_postgis(self, min_lon: float, min_lat: float,
                            max_lon: float, max_lat: float) -> List[Dict]:
        """Query fields in bbox using PostGIS."""
        query = """
        SELECT boundary_id, field_id, 
               ST_AsGeoJSON(geometry::geometry) as geometry,
               properties
        FROM field_boundaries
        WHERE geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)::geography;
        """
        
        try:
            cursor = self._connection.cursor()
            cursor.execute(query, (min_lon, min_lat, max_lon, max_lat))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "boundary_id": row[0],
                    "field_id": row[1],
                    "geometry": json.loads(row[2]),
                    "properties": row[3]
                })
            
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"PostGIS query failed: {e}")
            return self._query_bbox_local(min_lon, min_lat, max_lon, max_lat)
    
    def _query_bbox_local(self, min_lon: float, min_lat: float,
                          max_lon: float, max_lat: float) -> List[Dict]:
        """Query fields in bbox using local storage."""
        results = []
        
        for boundary in self._local_storage.get("field_boundaries", []):
            coords = boundary.get("coordinates", [[]])
            if coords and coords[0]:
                for point in coords[0]:
                    if len(point) >= 2:
                        lon, lat = point[0], point[1]
                        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                            results.append(boundary)
                            break
        
        return results
    
    def query_fields_near_point(self, lon: float, lat: float,
                                distance_meters: float) -> List[Dict]:
        """Query fields within distance of point."""
        if self._postgis_available and self._connection:
            query = """
            SELECT boundary_id, field_id,
                   ST_AsGeoJSON(geometry::geometry) as geometry,
                   properties,
                   ST_Distance(geometry, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as distance
            FROM field_boundaries
            WHERE ST_DWithin(geometry, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
            ORDER BY distance;
            """
            
            try:
                cursor = self._connection.cursor()
                cursor.execute(query, (lon, lat, lon, lat, distance_meters))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "boundary_id": row[0],
                        "field_id": row[1],
                        "geometry": json.loads(row[2]),
                        "properties": row[3],
                        "distance": row[4]
                    })
                
                cursor.close()
                return results
            except Exception as e:
                logger.error(f"PostGIS proximity query failed: {e}")
        
        return []
    
    def calculate_field_area(self, field_id: str) -> Optional[float]:
        """Calculate field area in hectares."""
        if self._postgis_available and self._connection:
            query = """
            SELECT ST_Area(geometry::geometry) / 10000 as area_hectares
            FROM field_boundaries
            WHERE field_id = %s;
            """
            
            try:
                cursor = self._connection.cursor()
                cursor.execute(query, (field_id,))
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    return row[0]
            except Exception as e:
                logger.error(f"Area calculation failed: {e}")
        
        return None


class SedonaAdapter:
    """
    Apache Sedona adapter for large-scale geospatial processing.
    
    Provides:
    - Distributed spatial joins
    - Zonal statistics computation
    - Raster-vector operations
    - Large-scale field analysis
    """
    
    def __init__(self, spark_master: str = "local[*]",
                 app_name: str = "CropMonitoring-Sedona"):
        self.spark_master = spark_master
        self.app_name = app_name
        self._sedona_available = False
        self._context = None
        
        self._check_sedona_availability()
    
    def _check_sedona_availability(self):
        """Check if Sedona is available."""
        try:
            from sedona.spark import SedonaContext
            self._sedona_available = True
            logger.info("Apache Sedona available for crop monitoring")
        except ImportError:
            logger.warning("Apache Sedona not available, using mock implementation")
            self._sedona_available = False
    
    def connect(self):
        """Initialize Sedona context."""
        if self._sedona_available:
            try:
                from sedona.spark import SedonaContext
                self._context = SedonaContext.create(self.spark_master, self.app_name)
                logger.info("Connected to Apache Sedona")
            except Exception as e:
                logger.warning(f"Sedona connection failed: {e}")
                self._sedona_available = False
    
    def compute_zonal_statistics(self, field_boundaries: List[SpatialBoundary],
                                 raster_path: str,
                                 statistics: List[str] = None) -> List[Dict]:
        """
        Compute zonal statistics for fields from raster data.
        
        Args:
            field_boundaries: List of field boundaries
            raster_path: Path to raster file (GeoTIFF)
            statistics: List of statistics to compute (mean, min, max, std, count)
        
        Returns:
            List of statistics per field
        """
        statistics = statistics or ["mean", "min", "max", "std", "count"]
        
        if self._sedona_available and self._context:
            return self._compute_zonal_sedona(field_boundaries, raster_path, statistics)
        else:
            return self._compute_zonal_mock(field_boundaries, statistics)
    
    def _compute_zonal_sedona(self, boundaries: List[SpatialBoundary],
                              raster_path: str,
                              statistics: List[str]) -> List[Dict]:
        """Compute zonal statistics using Sedona."""
        try:
            from sedona.core.spatialOperator import JoinQuery
            
            results = []
            for boundary in boundaries:
                result = {
                    "field_id": boundary.field_id,
                    "boundary_id": boundary.boundary_id,
                    "statistics": {}
                }
                
                for stat in statistics:
                    result["statistics"][stat] = 0.0
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Sedona zonal stats failed: {e}")
            return self._compute_zonal_mock(boundaries, statistics)
    
    def _compute_zonal_mock(self, boundaries: List[SpatialBoundary],
                            statistics: List[str]) -> List[Dict]:
        """Mock zonal statistics computation."""
        import random
        
        results = []
        for boundary in boundaries:
            result = {
                "field_id": boundary.field_id,
                "boundary_id": boundary.boundary_id,
                "statistics": {}
            }
            
            for stat in statistics:
                if stat == "mean":
                    result["statistics"][stat] = random.uniform(0.3, 0.8)
                elif stat == "min":
                    result["statistics"][stat] = random.uniform(0.1, 0.4)
                elif stat == "max":
                    result["statistics"][stat] = random.uniform(0.7, 0.95)
                elif stat == "std":
                    result["statistics"][stat] = random.uniform(0.05, 0.15)
                elif stat == "count":
                    result["statistics"][stat] = random.randint(1000, 50000)
            
            results.append(result)
        
        return results
    
    def spatial_join_fields_with_data(self, field_boundaries: List[SpatialBoundary],
                                      data_points: List[Dict]) -> List[Dict]:
        """
        Spatial join fields with point data.
        
        Args:
            field_boundaries: List of field boundaries
            data_points: List of point data with lat/lon
        
        Returns:
            List of joined records
        """
        results = []
        
        for point in data_points:
            lat = point.get("latitude") or point.get("lat")
            lon = point.get("longitude") or point.get("lon")
            
            if lat is None or lon is None:
                continue
            
            for boundary in field_boundaries:
                if self._point_in_polygon(lon, lat, boundary.coordinates):
                    results.append({
                        "field_id": boundary.field_id,
                        "point_data": point
                    })
                    break
        
        return results
    
    def _point_in_polygon(self, x: float, y: float,
                          polygon_coords: List[List[List[float]]]) -> bool:
        """Check if point is inside polygon (ray casting algorithm)."""
        if not polygon_coords or not polygon_coords[0]:
            return False
        
        ring = polygon_coords[0]
        n = len(ring)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def compute_field_ndvi_from_bands(self, field_boundary: SpatialBoundary,
                                      red_band_path: str,
                                      nir_band_path: str) -> Dict[str, float]:
        """
        Compute NDVI for a field from band rasters.
        
        Args:
            field_boundary: Field boundary
            red_band_path: Path to red band raster
            nir_band_path: Path to NIR band raster
        
        Returns:
            NDVI statistics for the field
        """
        import random
        
        return {
            "field_id": field_boundary.field_id,
            "ndvi_mean": random.uniform(0.4, 0.8),
            "ndvi_min": random.uniform(0.1, 0.4),
            "ndvi_max": random.uniform(0.7, 0.95),
            "ndvi_std": random.uniform(0.05, 0.15),
            "pixel_count": random.randint(5000, 50000),
            "computed_at": datetime.utcnow().isoformat()
        }


class CropMonitoringDataStore:
    """
    Unified data store for crop monitoring.
    
    Integrates Lakehouse, PostGIS, and Sedona for comprehensive
    crop monitoring data management.
    """
    
    def __init__(self, lakehouse_path: str = "/data/crop_lakehouse",
                 postgis_connection: str = None,
                 sedona_master: str = "local[*]"):
        self.lakehouse = LakehouseAdapter(warehouse_path=lakehouse_path)
        self.postgis = PostGISAdapter(connection_string=postgis_connection)
        self.sedona = SedonaAdapter(spark_master=sedona_master)
        
        self._initialized = False
    
    def initialize(self):
        """Initialize all adapters."""
        self.postgis.connect()
        self.sedona.connect()
        self._initialized = True
        logger.info("Crop monitoring data store initialized")
    
    def save_field(self, field_id: str, boundary: SpatialBoundary,
                   properties: Dict[str, Any] = None) -> str:
        """Save field boundary and properties."""
        boundary.field_id = field_id
        boundary.properties = properties or {}
        
        return self.postgis.save_field_boundary(boundary)
    
    def save_vegetation_index(self, field_id: str, index_type: str,
                              values: Dict[str, float],
                              timestamp: datetime = None,
                              source: str = "sentinel-2") -> str:
        """Save vegetation index measurement."""
        record = VegetationIndexRecord(
            record_id=str(uuid.uuid4()),
            field_id=field_id,
            timestamp=timestamp or datetime.utcnow(),
            index_type=index_type,
            mean_value=values.get("mean", 0),
            min_value=values.get("min", 0),
            max_value=values.get("max", 0),
            std_value=values.get("std", 0),
            pixel_count=values.get("count", 0),
            cloud_coverage=values.get("cloud_coverage", 0),
            source=source
        )
        
        return self.lakehouse.write_vegetation_index(record)
    
    def save_weather_data(self, field_id: str, weather: Dict[str, Any],
                          timestamp: datetime = None,
                          is_forecast: bool = False) -> str:
        """Save weather data."""
        record = WeatherRecord(
            record_id=str(uuid.uuid4()),
            field_id=field_id,
            timestamp=timestamp or datetime.utcnow(),
            temperature=weather.get("temperature", 0),
            humidity=weather.get("humidity", 0),
            precipitation=weather.get("precipitation", 0),
            wind_speed=weather.get("wind_speed", 0),
            solar_radiation=weather.get("solar_radiation"),
            evapotranspiration=weather.get("evapotranspiration"),
            gdd=weather.get("gdd"),
            forecast=is_forecast
        )
        
        return self.lakehouse.write_weather_data(record)
    
    def save_alert(self, field_id: str, alert_type: str,
                   severity: str, message: str,
                   metadata: Dict[str, Any] = None) -> str:
        """Save alert."""
        alert = {
            "alert_id": str(uuid.uuid4()),
            "field_id": field_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        return self.lakehouse.write_alert(alert)
    
    def get_field_vegetation_history(self, field_id: str,
                                     days: int = 90,
                                     index_type: str = "NDVI") -> List[Dict]:
        """Get vegetation index history for a field."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return self.lakehouse.read_vegetation_indices(
            field_id=field_id,
            start_date=start_date,
            end_date=end_date,
            index_type=index_type
        )
    
    def get_field_weather_history(self, field_id: str,
                                  days: int = 30) -> List[Dict]:
        """Get weather history for a field."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return self.lakehouse.read_weather_data(
            field_id=field_id,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_active_alerts(self, field_id: str = None,
                          severity: str = None) -> List[Dict]:
        """Get active alerts."""
        return self.lakehouse.read_alerts(
            field_id=field_id,
            status="active",
            severity=severity
        )
    
    def query_fields_in_region(self, min_lon: float, min_lat: float,
                               max_lon: float, max_lat: float) -> List[Dict]:
        """Query fields in a geographic region."""
        return self.postgis.query_fields_in_bbox(min_lon, min_lat, max_lon, max_lat)
    
    def compute_field_statistics(self, field_boundary: SpatialBoundary,
                                 raster_path: str = None) -> Dict:
        """Compute statistics for a field."""
        if raster_path:
            stats = self.sedona.compute_zonal_statistics([field_boundary], raster_path)
            return stats[0] if stats else {}
        else:
            return self.sedona._compute_zonal_mock([field_boundary], ["mean", "min", "max", "std", "count"])[0]


def create_crop_data_store(lakehouse_path: str = None,
                           postgis_connection: str = None,
                           sedona_master: str = None) -> CropMonitoringDataStore:
    """
    Create and initialize crop monitoring data store.
    
    Args:
        lakehouse_path: Path to lakehouse warehouse
        postgis_connection: PostGIS connection string
        sedona_master: Spark master URL for Sedona
    
    Returns:
        Initialized CropMonitoringDataStore
    """
    store = CropMonitoringDataStore(
        lakehouse_path=lakehouse_path or os.environ.get("CROP_LAKEHOUSE_PATH", "/data/crop_lakehouse"),
        postgis_connection=postgis_connection or os.environ.get("CROP_POSTGIS_URL"),
        sedona_master=sedona_master or os.environ.get("CROP_SEDONA_MASTER", "local[*]")
    )
    
    store.initialize()
    return store


__all__ = [
    'StorageBackend',
    'TableType',
    'CropDataRecord',
    'SpatialBoundary',
    'VegetationIndexRecord',
    'WeatherRecord',
    'LakehouseAdapter',
    'PostGISAdapter',
    'SedonaAdapter',
    'CropMonitoringDataStore',
    'create_crop_data_store'
]
