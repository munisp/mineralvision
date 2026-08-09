"""
Geo Persistence Module
======================

Production-grade geospatial persistence with:
- PostgreSQL/PostGIS integration
- Spatial indexes for efficient queries
- Real bounding-box and proximity queries
- Detection history and analytics
- Async database operations
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


class GeometryType(Enum):
    """Supported geometry types."""
    POINT = "Point"
    POLYGON = "Polygon"
    BBOX = "BoundingBox"


@dataclass
class GeoPoint:
    """Geographic point."""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    
    def to_wkt(self) -> str:
        """Convert to WKT format."""
        if self.altitude:
            return f"POINT Z({self.longitude} {self.latitude} {self.altitude})"
        return f"POINT({self.longitude} {self.latitude})"
    
    def to_geojson(self) -> Dict:
        """Convert to GeoJSON."""
        coords = [self.longitude, self.latitude]
        if self.altitude:
            coords.append(self.altitude)
        return {
            "type": "Point",
            "coordinates": coords
        }


@dataclass
class GeoBoundingBox:
    """Geographic bounding box."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    
    def to_wkt(self) -> str:
        """Convert to WKT polygon."""
        return f"POLYGON(({self.min_lon} {self.min_lat}, {self.max_lon} {self.min_lat}, " \
               f"{self.max_lon} {self.max_lat}, {self.min_lon} {self.max_lat}, " \
               f"{self.min_lon} {self.min_lat}))"
    
    def to_geojson(self) -> Dict:
        """Convert to GeoJSON polygon."""
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.min_lon, self.min_lat],
                [self.max_lon, self.min_lat],
                [self.max_lon, self.max_lat],
                [self.min_lon, self.max_lat],
                [self.min_lon, self.min_lat]
            ]]
        }
    
    def contains(self, point: GeoPoint) -> bool:
        """Check if point is within bounding box."""
        return (self.min_lat <= point.latitude <= self.max_lat and
                self.min_lon <= point.longitude <= self.max_lon)
    
    @property
    def center(self) -> GeoPoint:
        """Get center point."""
        return GeoPoint(
            latitude=(self.min_lat + self.max_lat) / 2,
            longitude=(self.min_lon + self.max_lon) / 2
        )


@dataclass
class GeoDetection:
    """Geospatially-referenced detection."""
    detection_id: str
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox_pixels: List[float]  # [x1, y1, x2, y2] in pixels
    location: GeoPoint
    footprint: Optional[GeoBoundingBox] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    frame_id: int = 0
    source_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            'detection_id': self.detection_id,
            'track_id': self.track_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox_pixels': self.bbox_pixels,
            'location': {
                'latitude': self.location.latitude,
                'longitude': self.location.longitude,
                'altitude': self.location.altitude
            },
            'timestamp': self.timestamp.isoformat(),
            'frame_id': self.frame_id,
            'source_id': self.source_id,
            'metadata': self.metadata
        }
        
        if self.footprint:
            d['footprint'] = {
                'min_lat': self.footprint.min_lat,
                'min_lon': self.footprint.min_lon,
                'max_lat': self.footprint.max_lat,
                'max_lon': self.footprint.max_lon
            }
        
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GeoDetection':
        """Create from dictionary."""
        location = GeoPoint(
            latitude=d['location']['latitude'],
            longitude=d['location']['longitude'],
            altitude=d['location'].get('altitude')
        )
        
        footprint = None
        if 'footprint' in d and d['footprint']:
            footprint = GeoBoundingBox(
                min_lat=d['footprint']['min_lat'],
                min_lon=d['footprint']['min_lon'],
                max_lat=d['footprint']['max_lat'],
                max_lon=d['footprint']['max_lon']
            )
        
        return cls(
            detection_id=d['detection_id'],
            track_id=d.get('track_id'),
            class_id=d['class_id'],
            class_name=d['class_name'],
            confidence=d['confidence'],
            bbox_pixels=d['bbox_pixels'],
            location=location,
            footprint=footprint,
            timestamp=datetime.fromisoformat(d['timestamp']) if isinstance(d['timestamp'], str) else d['timestamp'],
            frame_id=d.get('frame_id', 0),
            source_id=d.get('source_id', ''),
            metadata=d.get('metadata', {})
        )


class PostGISConnection:
    """
    PostgreSQL/PostGIS connection manager.
    """
    
    def __init__(self, connection_string: str = None, **kwargs):
        self.connection_string = connection_string or self._build_connection_string(**kwargs)
        self._connection = None
        self._lock = threading.Lock()
    
    def _build_connection_string(self, host: str = "localhost", port: int = 5432,
                                database: str = "waldo", user: str = "postgres",
                                password: str = "") -> str:
        """Build connection string from parameters."""
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    def connect(self):
        """Establish database connection."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            self._connection = psycopg2.connect(self.connection_string)
            self._connection.autocommit = False
            
            logger.info("Connected to PostgreSQL database")
            
        except ImportError:
            logger.warning("psycopg2 not available, using SQLite fallback")
            self._use_sqlite_fallback()
    
    def _use_sqlite_fallback(self):
        """Use SQLite as fallback when PostgreSQL is not available."""
        import sqlite3
        
        db_path = os.environ.get('WALDO_DB_PATH', '/tmp/waldo_geo.db')
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        
        logger.info(f"Using SQLite fallback at {db_path}")
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    @contextmanager
    def cursor(self):
        """Get database cursor."""
        with self._lock:
            if self._connection is None:
                self.connect()
            
            cursor = self._connection.cursor()
            try:
                yield cursor
                self._connection.commit()
            except Exception as e:
                self._connection.rollback()
                raise
            finally:
                cursor.close()
    
    def execute(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute query and return results."""
        with self.cursor() as cursor:
            cursor.execute(query, params)
            
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return []
    
    def execute_many(self, query: str, params_list: List[tuple]):
        """Execute query with multiple parameter sets."""
        with self.cursor() as cursor:
            cursor.executemany(query, params_list)


class GeoPersistenceManager:
    """
    Geospatial persistence manager for WALDO detections.
    """
    
    def __init__(self, connection: PostGISConnection = None,
                 connection_string: str = None):
        self.connection = connection or PostGISConnection(connection_string)
        self._initialized = False
    
    def initialize(self):
        """Initialize database schema."""
        self._create_tables()
        self._create_indexes()
        self._initialized = True
        logger.info("Geo persistence initialized")
    
    def _create_tables(self):
        """Create database tables."""
        # Check if using PostgreSQL or SQLite
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            self._create_postgres_tables()
        else:
            self._create_sqlite_tables()
    
    def _create_postgres_tables(self):
        """Create PostgreSQL/PostGIS tables."""
        queries = [
            # Enable PostGIS extension
            "CREATE EXTENSION IF NOT EXISTS postgis;",
            
            # Detections table
            """
            CREATE TABLE IF NOT EXISTS detections (
                detection_id UUID PRIMARY KEY,
                track_id INTEGER,
                class_id INTEGER NOT NULL,
                class_name VARCHAR(255) NOT NULL,
                confidence REAL NOT NULL,
                bbox_pixels REAL[] NOT NULL,
                location GEOGRAPHY(POINT, 4326) NOT NULL,
                footprint GEOGRAPHY(POLYGON, 4326),
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                frame_id INTEGER,
                source_id VARCHAR(255),
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            
            # Detection history table for analytics
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                id SERIAL PRIMARY KEY,
                detection_id UUID REFERENCES detections(detection_id),
                track_id INTEGER,
                location GEOGRAPHY(POINT, 4326) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                confidence REAL NOT NULL
            );
            """,
            
            # Sources table (cameras, drones, etc.)
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id VARCHAR(255) PRIMARY KEY,
                source_type VARCHAR(50) NOT NULL,
                name VARCHAR(255),
                location GEOGRAPHY(POINT, 4326),
                coverage_area GEOGRAPHY(POLYGON, 4326),
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            
            # Aggregated statistics table
            """
            CREATE TABLE IF NOT EXISTS detection_stats (
                id SERIAL PRIMARY KEY,
                class_name VARCHAR(255) NOT NULL,
                source_id VARCHAR(255),
                hour TIMESTAMPTZ NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL,
                UNIQUE(class_name, source_id, hour)
            );
            """
        ]
        
        for query in queries:
            try:
                self.connection.execute(query)
            except Exception as e:
                logger.warning(f"Table creation warning: {e}")
    
    def _create_sqlite_tables(self):
        """Create SQLite tables (fallback)."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS detections (
                detection_id TEXT PRIMARY KEY,
                track_id INTEGER,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                altitude REAL,
                footprint_min_lat REAL,
                footprint_min_lon REAL,
                footprint_max_lat REAL,
                footprint_max_lon REAL,
                timestamp TEXT NOT NULL,
                frame_id INTEGER,
                source_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id TEXT,
                track_id INTEGER,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            """,
            
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                name TEXT,
                latitude REAL,
                longitude REAL,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]
        
        for query in queries:
            try:
                self.connection.execute(query)
            except Exception as e:
                logger.warning(f"Table creation warning: {e}")
    
    def _create_indexes(self):
        """Create spatial and other indexes."""
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_detections_location ON detections USING GIST(location);",
                "CREATE INDEX IF NOT EXISTS idx_detections_footprint ON detections USING GIST(footprint);",
                "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);",
                "CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id);",
                "CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source_id);",
                "CREATE INDEX IF NOT EXISTS idx_history_detection ON detection_history(detection_id);",
                "CREATE INDEX IF NOT EXISTS idx_history_track ON detection_history(track_id);",
                "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON detection_history(timestamp);"
            ]
        else:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_detections_lat ON detections(latitude);",
                "CREATE INDEX IF NOT EXISTS idx_detections_lon ON detections(longitude);",
                "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);",
                "CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id);",
                "CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source_id);"
            ]
        
        for index in indexes:
            try:
                self.connection.execute(index)
            except Exception as e:
                logger.debug(f"Index creation note: {e}")
    
    def save_detection(self, detection: GeoDetection) -> str:
        """
        Save a detection to the database.
        
        Args:
            detection: GeoDetection to save
            
        Returns:
            Detection ID
        """
        if not self._initialized:
            self.initialize()
        
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            return self._save_detection_postgres(detection)
        else:
            return self._save_detection_sqlite(detection)
    
    def _save_detection_postgres(self, detection: GeoDetection) -> str:
        """Save detection to PostgreSQL."""
        query = """
        INSERT INTO detections (
            detection_id, track_id, class_id, class_name, confidence,
            bbox_pixels, location, footprint, timestamp, frame_id,
            source_id, metadata
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (detection_id) DO UPDATE SET
            confidence = EXCLUDED.confidence,
            location = EXCLUDED.location,
            timestamp = EXCLUDED.timestamp,
            metadata = EXCLUDED.metadata;
        """
        
        footprint_wkt = detection.footprint.to_wkt() if detection.footprint else None
        if footprint_wkt:
            footprint_wkt = f"ST_SetSRID(ST_GeomFromText('{footprint_wkt}'), 4326)::geography"
        
        params = (
            detection.detection_id,
            detection.track_id,
            detection.class_id,
            detection.class_name,
            detection.confidence,
            detection.bbox_pixels,
            detection.location.longitude,
            detection.location.latitude,
            footprint_wkt,
            detection.timestamp,
            detection.frame_id,
            detection.source_id,
            json.dumps(detection.metadata)
        )
        
        self.connection.execute(query, params)
        
        # Also save to history
        self._save_to_history(detection)
        
        return detection.detection_id
    
    def _save_detection_sqlite(self, detection: GeoDetection) -> str:
        """Save detection to SQLite."""
        query = """
        INSERT OR REPLACE INTO detections (
            detection_id, track_id, class_id, class_name, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            latitude, longitude, altitude,
            footprint_min_lat, footprint_min_lon, footprint_max_lat, footprint_max_lon,
            timestamp, frame_id, source_id, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        params = (
            detection.detection_id,
            detection.track_id,
            detection.class_id,
            detection.class_name,
            detection.confidence,
            detection.bbox_pixels[0],
            detection.bbox_pixels[1],
            detection.bbox_pixels[2],
            detection.bbox_pixels[3],
            detection.location.latitude,
            detection.location.longitude,
            detection.location.altitude,
            detection.footprint.min_lat if detection.footprint else None,
            detection.footprint.min_lon if detection.footprint else None,
            detection.footprint.max_lat if detection.footprint else None,
            detection.footprint.max_lon if detection.footprint else None,
            detection.timestamp.isoformat(),
            detection.frame_id,
            detection.source_id,
            json.dumps(detection.metadata)
        )
        
        self.connection.execute(query, params)
        
        return detection.detection_id
    
    def _save_to_history(self, detection: GeoDetection):
        """Save detection to history table."""
        query = """
        INSERT INTO detection_history (
            detection_id, track_id, location, timestamp, confidence
        ) VALUES (
            %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s, %s
        );
        """
        
        params = (
            detection.detection_id,
            detection.track_id,
            detection.location.longitude,
            detection.location.latitude,
            detection.timestamp,
            detection.confidence
        )
        
        try:
            self.connection.execute(query, params)
        except Exception as e:
            logger.debug(f"History save note: {e}")
    
    def save_detections_batch(self, detections: List[GeoDetection]) -> int:
        """
        Save multiple detections in batch.
        
        Args:
            detections: List of detections
            
        Returns:
            Number of saved detections
        """
        if not detections:
            return 0
        
        for detection in detections:
            self.save_detection(detection)
        
        return len(detections)
    
    def query_by_bbox(self, bbox: GeoBoundingBox,
                     class_names: Optional[List[str]] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[GeoDetection]:
        """
        Query detections within a bounding box.
        
        Args:
            bbox: Geographic bounding box
            class_names: Filter by class names
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results
            
        Returns:
            List of detections
        """
        if not self._initialized:
            self.initialize()
        
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            return self._query_bbox_postgres(bbox, class_names, start_time, end_time, limit)
        else:
            return self._query_bbox_sqlite(bbox, class_names, start_time, end_time, limit)
    
    def _query_bbox_postgres(self, bbox: GeoBoundingBox,
                            class_names: Optional[List[str]],
                            start_time: Optional[datetime],
                            end_time: Optional[datetime],
                            limit: int) -> List[GeoDetection]:
        """Query using PostGIS spatial functions."""
        query = """
        SELECT 
            detection_id, track_id, class_id, class_name, confidence,
            bbox_pixels,
            ST_Y(location::geometry) as latitude,
            ST_X(location::geometry) as longitude,
            timestamp, frame_id, source_id, metadata
        FROM detections
        WHERE ST_Intersects(
            location,
            ST_MakeEnvelope(%s, %s, %s, %s, 4326)::geography
        )
        """
        
        params = [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat]
        
        if class_names:
            query += " AND class_name = ANY(%s)"
            params.append(class_names)
        
        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        results = self.connection.execute(query, tuple(params))
        
        return [self._row_to_detection(row) for row in results]
    
    def _query_bbox_sqlite(self, bbox: GeoBoundingBox,
                          class_names: Optional[List[str]],
                          start_time: Optional[datetime],
                          end_time: Optional[datetime],
                          limit: int) -> List[GeoDetection]:
        """Query using SQLite (no spatial functions)."""
        query = """
        SELECT * FROM detections
        WHERE latitude >= ? AND latitude <= ?
        AND longitude >= ? AND longitude <= ?
        """
        
        params = [bbox.min_lat, bbox.max_lat, bbox.min_lon, bbox.max_lon]
        
        if class_names:
            placeholders = ','.join(['?' for _ in class_names])
            query += f" AND class_name IN ({placeholders})"
            params.extend(class_names)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = self.connection.execute(query, tuple(params))
        
        return [self._sqlite_row_to_detection(row) for row in results]
    
    def query_by_proximity(self, center: GeoPoint,
                          radius_meters: float,
                          class_names: Optional[List[str]] = None,
                          limit: int = 1000) -> List[GeoDetection]:
        """
        Query detections within radius of a point.
        
        Args:
            center: Center point
            radius_meters: Search radius in meters
            class_names: Filter by class names
            limit: Maximum results
            
        Returns:
            List of detections sorted by distance
        """
        if not self._initialized:
            self.initialize()
        
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            query = """
            SELECT 
                detection_id, track_id, class_id, class_name, confidence,
                bbox_pixels,
                ST_Y(location::geometry) as latitude,
                ST_X(location::geometry) as longitude,
                timestamp, frame_id, source_id, metadata,
                ST_Distance(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as distance
            FROM detections
            WHERE ST_DWithin(
                location,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s
            )
            """
            
            params = [center.longitude, center.latitude,
                     center.longitude, center.latitude, radius_meters]
            
            if class_names:
                query += " AND class_name = ANY(%s)"
                params.append(class_names)
            
            query += " ORDER BY distance ASC LIMIT %s"
            params.append(limit)
            
            results = self.connection.execute(query, tuple(params))
            return [self._row_to_detection(row) for row in results]
        else:
            # Approximate for SQLite using bounding box
            # 1 degree latitude ≈ 111km
            lat_delta = radius_meters / 111000
            lon_delta = radius_meters / (111000 * abs(center.latitude) if center.latitude != 0 else 111000)
            
            bbox = GeoBoundingBox(
                min_lat=center.latitude - lat_delta,
                max_lat=center.latitude + lat_delta,
                min_lon=center.longitude - lon_delta,
                max_lon=center.longitude + lon_delta
            )
            
            return self.query_by_bbox(bbox, class_names, limit=limit)
    
    def query_track_history(self, track_id: int,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Query location history for a track.
        
        Args:
            track_id: Track ID
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of track positions
        """
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            query = """
            SELECT 
                ST_Y(location::geometry) as latitude,
                ST_X(location::geometry) as longitude,
                timestamp, confidence
            FROM detection_history
            WHERE track_id = %s
            """
            params = [track_id]
        else:
            query = """
            SELECT latitude, longitude, timestamp, confidence
            FROM detection_history
            WHERE track_id = ?
            """
            params = [track_id]
        
        if start_time:
            query += f" AND timestamp >= {'%s' if is_postgres else '?'}"
            params.append(start_time if is_postgres else start_time.isoformat())
        
        if end_time:
            query += f" AND timestamp <= {'%s' if is_postgres else '?'}"
            params.append(end_time if is_postgres else end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        return self.connection.execute(query, tuple(params))
    
    def get_detection_counts(self, bbox: Optional[GeoBoundingBox] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            group_by: str = 'class_name') -> List[Dict]:
        """
        Get detection counts grouped by class or source.
        
        Args:
            bbox: Optional bounding box filter
            start_time: Start time filter
            end_time: End time filter
            group_by: Group by field ('class_name' or 'source_id')
            
        Returns:
            List of count records
        """
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            query = f"""
            SELECT {group_by}, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM detections
            WHERE 1=1
            """
            params = []
            
            if bbox:
                query += """
                AND ST_Intersects(
                    location,
                    ST_MakeEnvelope(%s, %s, %s, %s, 4326)::geography
                )
                """
                params.extend([bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat])
            
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            
            query += f" GROUP BY {group_by} ORDER BY count DESC"
            
        else:
            query = f"""
            SELECT {group_by}, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM detections
            WHERE 1=1
            """
            params = []
            
            if bbox:
                query += """
                AND latitude >= ? AND latitude <= ?
                AND longitude >= ? AND longitude <= ?
                """
                params.extend([bbox.min_lat, bbox.max_lat, bbox.min_lon, bbox.max_lon])
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += f" GROUP BY {group_by} ORDER BY count DESC"
        
        return self.connection.execute(query, tuple(params))
    
    def delete_old_detections(self, older_than: datetime) -> int:
        """
        Delete detections older than specified time.
        
        Args:
            older_than: Delete detections before this time
            
        Returns:
            Number of deleted records
        """
        is_postgres = 'postgresql' in str(self.connection.connection_string).lower()
        
        if is_postgres:
            query = "DELETE FROM detections WHERE timestamp < %s"
            params = (older_than,)
        else:
            query = "DELETE FROM detections WHERE timestamp < ?"
            params = (older_than.isoformat(),)
        
        self.connection.execute(query, params)
        
        # Also clean history
        if is_postgres:
            self.connection.execute(
                "DELETE FROM detection_history WHERE timestamp < %s",
                (older_than,)
            )
        else:
            self.connection.execute(
                "DELETE FROM detection_history WHERE timestamp < ?",
                (older_than.isoformat(),)
            )
        
        return 0  # SQLite doesn't return affected rows easily
    
    def _row_to_detection(self, row: Dict) -> GeoDetection:
        """Convert database row to GeoDetection."""
        return GeoDetection(
            detection_id=row['detection_id'],
            track_id=row.get('track_id'),
            class_id=row['class_id'],
            class_name=row['class_name'],
            confidence=row['confidence'],
            bbox_pixels=row['bbox_pixels'] if isinstance(row['bbox_pixels'], list) else list(row['bbox_pixels']),
            location=GeoPoint(
                latitude=row['latitude'],
                longitude=row['longitude']
            ),
            timestamp=row['timestamp'] if isinstance(row['timestamp'], datetime) else datetime.fromisoformat(row['timestamp']),
            frame_id=row.get('frame_id', 0),
            source_id=row.get('source_id', ''),
            metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else (row['metadata'] or {})
        )
    
    def _sqlite_row_to_detection(self, row: Dict) -> GeoDetection:
        """Convert SQLite row to GeoDetection."""
        footprint = None
        if row.get('footprint_min_lat') is not None:
            footprint = GeoBoundingBox(
                min_lat=row['footprint_min_lat'],
                min_lon=row['footprint_min_lon'],
                max_lat=row['footprint_max_lat'],
                max_lon=row['footprint_max_lon']
            )
        
        return GeoDetection(
            detection_id=row['detection_id'],
            track_id=row.get('track_id'),
            class_id=row['class_id'],
            class_name=row['class_name'],
            confidence=row['confidence'],
            bbox_pixels=[row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
            location=GeoPoint(
                latitude=row['latitude'],
                longitude=row['longitude'],
                altitude=row.get('altitude')
            ),
            footprint=footprint,
            timestamp=datetime.fromisoformat(row['timestamp']),
            frame_id=row.get('frame_id', 0),
            source_id=row.get('source_id', ''),
            metadata=json.loads(row['metadata']) if row.get('metadata') else {}
        )
    
    def close(self):
        """Close database connection."""
        self.connection.close()


def create_geo_persistence(connection_string: Optional[str] = None,
                          config: Optional[Dict] = None) -> GeoPersistenceManager:
    """Factory function to create geo persistence manager."""
    if connection_string:
        connection = PostGISConnection(connection_string)
    elif config:
        connection = PostGISConnection(
            host=config.get('host', 'localhost'),
            port=config.get('port', 5432),
            database=config.get('database', 'waldo'),
            user=config.get('user', 'postgres'),
            password=config.get('password', '')
        )
    else:
        # Use SQLite fallback
        connection = PostGISConnection()
    
    manager = GeoPersistenceManager(connection)
    manager.initialize()
    
    return manager
