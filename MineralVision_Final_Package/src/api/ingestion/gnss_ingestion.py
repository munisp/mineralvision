"""
GNSS/Survey Data Ingestion Module for MineralVision Platform.

Supports ingestion of GNSS data from various formats:
- RINEX (observation and navigation)
- NMEA sentences
- CSV/ASCII exports from survey software
- GPX tracks
- Trimble/Leica/Topcon proprietary formats (via conversion)

Provides trajectory QA, coordinate transformation, and sample geotagging.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import numpy as np
from datetime import datetime, timedelta
import re
import math


class GNSSFormat(Enum):
    """Supported GNSS data formats."""
    RINEX_OBS = "rinex_obs"
    RINEX_NAV = "rinex_nav"
    NMEA = "nmea"
    CSV = "csv"
    GPX = "gpx"
    KML = "kml"
    GEOJSON = "geojson"
    TRIMBLE_JXL = "trimble_jxl"
    LEICA_GSI = "leica_gsi"


class FixQuality(Enum):
    """GNSS fix quality indicators."""
    INVALID = 0
    GPS_FIX = 1
    DGPS = 2
    PPS = 3
    RTK_FIXED = 4
    RTK_FLOAT = 5
    ESTIMATED = 6
    MANUAL = 7
    SIMULATION = 8
    PPK = 9  # Post-processed kinematic


class CoordinateSystem(Enum):
    """Common coordinate systems."""
    WGS84 = "EPSG:4326"
    WGS84_UTM = "UTM"
    NAD83 = "EPSG:4269"
    ETRS89 = "EPSG:4258"
    LOCAL = "local"


@dataclass
class GNSSObservation:
    """Single GNSS observation/position."""
    observation_id: str
    timestamp: datetime
    
    # Position (WGS84)
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters (ellipsoidal)
    
    # Position accuracy
    horizontal_accuracy: float = 0.0  # meters
    vertical_accuracy: float = 0.0  # meters
    pdop: float = 0.0
    hdop: float = 0.0
    vdop: float = 0.0
    
    # Fix information
    fix_quality: FixQuality = FixQuality.GPS_FIX
    num_satellites: int = 0
    satellites_used: List[str] = field(default_factory=list)
    
    # Velocity (if available)
    speed: float = 0.0  # m/s
    heading: float = 0.0  # degrees from north
    
    # Geoid separation
    geoid_separation: float = 0.0  # meters
    
    # Age of differential correction
    dgps_age: float = 0.0  # seconds
    dgps_station_id: str = ""
    
    # Raw data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    # QC flags
    is_valid: bool = True
    qc_flags: List[str] = field(default_factory=list)
    
    @property
    def orthometric_height(self) -> float:
        """Calculate orthometric height (MSL)."""
        return self.altitude - self.geoid_separation
    
    def to_utm(self) -> Tuple[float, float, int, str]:
        """
        Convert to UTM coordinates.
        
        Returns:
            (easting, northing, zone_number, zone_letter)
        """
        return self._latlon_to_utm(self.latitude, self.longitude)
    
    @staticmethod
    def _latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
        """Convert lat/lon to UTM."""
        # UTM zone calculation
        zone_number = int((lon + 180) / 6) + 1
        
        # Special zones for Norway and Svalbard
        if 56 <= lat < 64 and 3 <= lon < 12:
            zone_number = 32
        elif 72 <= lat < 84:
            if 0 <= lon < 9:
                zone_number = 31
            elif 9 <= lon < 21:
                zone_number = 33
            elif 21 <= lon < 33:
                zone_number = 35
            elif 33 <= lon < 42:
                zone_number = 37
        
        # Zone letter
        if 84 >= lat >= 72:
            zone_letter = 'X'
        elif 72 > lat >= 64:
            zone_letter = 'W'
        elif 64 > lat >= 56:
            zone_letter = 'V'
        elif 56 > lat >= 48:
            zone_letter = 'U'
        elif 48 > lat >= 40:
            zone_letter = 'T'
        elif 40 > lat >= 32:
            zone_letter = 'S'
        elif 32 > lat >= 24:
            zone_letter = 'R'
        elif 24 > lat >= 16:
            zone_letter = 'Q'
        elif 16 > lat >= 8:
            zone_letter = 'P'
        elif 8 > lat >= 0:
            zone_letter = 'N'
        elif 0 > lat >= -8:
            zone_letter = 'M'
        elif -8 > lat >= -16:
            zone_letter = 'L'
        elif -16 > lat >= -24:
            zone_letter = 'K'
        elif -24 > lat >= -32:
            zone_letter = 'J'
        elif -32 > lat >= -40:
            zone_letter = 'H'
        elif -40 > lat >= -48:
            zone_letter = 'G'
        elif -48 > lat >= -56:
            zone_letter = 'F'
        elif -56 > lat >= -64:
            zone_letter = 'E'
        elif -64 > lat >= -72:
            zone_letter = 'D'
        elif -72 > lat >= -80:
            zone_letter = 'C'
        else:
            zone_letter = 'Z'  # Outside UTM limits
        
        # UTM projection parameters
        a = 6378137.0  # WGS84 semi-major axis
        f = 1 / 298.257223563  # WGS84 flattening
        k0 = 0.9996  # UTM scale factor
        
        e2 = 2 * f - f * f  # First eccentricity squared
        e_prime2 = e2 / (1 - e2)  # Second eccentricity squared
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # Central meridian
        lon0 = math.radians((zone_number - 1) * 6 - 180 + 3)
        
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
        T = math.tan(lat_rad) ** 2
        C = e_prime2 * math.cos(lat_rad) ** 2
        A = math.cos(lat_rad) * (lon_rad - lon0)
        
        M = a * (
            (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_rad
            - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_rad)
            + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_rad)
            - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_rad)
        )
        
        easting = k0 * N * (
            A + (1 - T + C) * A ** 3 / 6
            + (5 - 18 * T + T ** 2 + 72 * C - 58 * e_prime2) * A ** 5 / 120
        ) + 500000
        
        northing = k0 * (
            M + N * math.tan(lat_rad) * (
                A ** 2 / 2
                + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                + (61 - 58 * T + T ** 2 + 600 * C - 330 * e_prime2) * A ** 6 / 720
            )
        )
        
        if lat < 0:
            northing += 10000000  # Southern hemisphere offset
        
        return easting, northing, zone_number, zone_letter


@dataclass
class GNSSTrajectory:
    """Collection of GNSS observations forming a trajectory."""
    trajectory_id: str
    observations: List[GNSSObservation] = field(default_factory=list)
    
    # Metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Statistics
    total_distance: float = 0.0  # meters
    mean_speed: float = 0.0  # m/s
    max_speed: float = 0.0  # m/s
    
    # Bounding box
    min_lat: float = 90.0
    max_lat: float = -90.0
    min_lon: float = 180.0
    max_lon: float = -180.0
    min_alt: float = float('inf')
    max_alt: float = float('-inf')
    
    # Quality metrics
    mean_hdop: float = 0.0
    mean_vdop: float = 0.0
    rtk_fix_percentage: float = 0.0
    gap_count: int = 0  # Number of data gaps
    
    def add_observation(self, obs: GNSSObservation) -> None:
        """Add observation and update statistics."""
        self.observations.append(obs)
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update trajectory statistics."""
        if not self.observations:
            return
        
        # Sort by timestamp
        self.observations.sort(key=lambda x: x.timestamp)
        
        # Time bounds
        self.start_time = self.observations[0].timestamp
        self.end_time = self.observations[-1].timestamp
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Spatial bounds
        for obs in self.observations:
            self.min_lat = min(self.min_lat, obs.latitude)
            self.max_lat = max(self.max_lat, obs.latitude)
            self.min_lon = min(self.min_lon, obs.longitude)
            self.max_lon = max(self.max_lon, obs.longitude)
            self.min_alt = min(self.min_alt, obs.altitude)
            self.max_alt = max(self.max_alt, obs.altitude)
        
        # Calculate distances and speeds
        self.total_distance = 0.0
        speeds = []
        
        for i in range(1, len(self.observations)):
            prev = self.observations[i - 1]
            curr = self.observations[i]
            
            # Haversine distance
            dist = self._haversine_distance(
                prev.latitude, prev.longitude,
                curr.latitude, curr.longitude
            )
            self.total_distance += dist
            
            # Speed
            dt = (curr.timestamp - prev.timestamp).total_seconds()
            if dt > 0:
                speeds.append(dist / dt)
        
        if speeds:
            self.mean_speed = np.mean(speeds)
            self.max_speed = np.max(speeds)
        
        # Quality metrics
        hdops = [obs.hdop for obs in self.observations if obs.hdop > 0]
        vdops = [obs.vdop for obs in self.observations if obs.vdop > 0]
        
        if hdops:
            self.mean_hdop = np.mean(hdops)
        if vdops:
            self.mean_vdop = np.mean(vdops)
        
        # RTK fix percentage
        rtk_count = sum(
            1 for obs in self.observations
            if obs.fix_quality in [FixQuality.RTK_FIXED, FixQuality.PPK]
        )
        self.rtk_fix_percentage = (rtk_count / len(self.observations)) * 100
        
        # Count gaps (>5 seconds between observations)
        self.gap_count = 0
        for i in range(1, len(self.observations)):
            dt = (self.observations[i].timestamp - self.observations[i-1].timestamp).total_seconds()
            if dt > 5:
                self.gap_count += 1
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


class NMEAParser:
    """Parser for NMEA 0183 sentences."""
    
    def parse_file(self, file_path: str) -> List[GNSSObservation]:
        """Parse NMEA file and return observations."""
        observations = []
        current_date = datetime.now().date()
        
        with open(file_path, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('$'):
                    continue
                
                obs = self._parse_sentence(line, current_date)
                if obs:
                    observations.append(obs)
                    
                    # Update date from RMC sentence
                    if 'RMC' in line and obs.timestamp:
                        current_date = obs.timestamp.date()
        
        return observations
    
    def _parse_sentence(self, sentence: str, current_date) -> Optional[GNSSObservation]:
        """Parse single NMEA sentence."""
        # Remove checksum
        if '*' in sentence:
            sentence = sentence.split('*')[0]
        
        parts = sentence.split(',')
        if len(parts) < 2:
            return None
        
        sentence_type = parts[0][-3:]  # GGA, RMC, etc.
        
        if sentence_type == 'GGA':
            return self._parse_gga(parts, current_date)
        elif sentence_type == 'RMC':
            return self._parse_rmc(parts)
        elif sentence_type == 'GLL':
            return self._parse_gll(parts, current_date)
        
        return None
    
    def _parse_gga(self, parts: List[str], current_date) -> Optional[GNSSObservation]:
        """Parse GGA sentence (fix data)."""
        if len(parts) < 15:
            return None
        
        try:
            # Time
            time_str = parts[1]
            if len(time_str) >= 6:
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                second = float(time_str[4:])
                timestamp = datetime.combine(
                    current_date,
                    datetime.min.time().replace(
                        hour=hour, minute=minute,
                        second=int(second), microsecond=int((second % 1) * 1000000)
                    )
                )
            else:
                timestamp = datetime.now()
            
            # Latitude
            lat_str = parts[2]
            lat_dir = parts[3]
            lat = self._parse_coordinate(lat_str, lat_dir)
            
            # Longitude
            lon_str = parts[4]
            lon_dir = parts[5]
            lon = self._parse_coordinate(lon_str, lon_dir, is_longitude=True)
            
            # Fix quality
            fix_quality = int(parts[6]) if parts[6] else 0
            fix_enum = FixQuality(min(fix_quality, 8))
            
            # Satellites
            num_sats = int(parts[7]) if parts[7] else 0
            
            # HDOP
            hdop = float(parts[8]) if parts[8] else 0.0
            
            # Altitude
            altitude = float(parts[9]) if parts[9] else 0.0
            
            # Geoid separation
            geoid_sep = float(parts[11]) if len(parts) > 11 and parts[11] else 0.0
            
            # DGPS age
            dgps_age = float(parts[13]) if len(parts) > 13 and parts[13] else 0.0
            
            return GNSSObservation(
                observation_id=f"GGA_{timestamp.isoformat()}",
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                hdop=hdop,
                fix_quality=fix_enum,
                num_satellites=num_sats,
                geoid_separation=geoid_sep,
                dgps_age=dgps_age,
                raw_data={'sentence': ','.join(parts)}
            )
        
        except (ValueError, IndexError):
            return None
    
    def _parse_rmc(self, parts: List[str]) -> Optional[GNSSObservation]:
        """Parse RMC sentence (recommended minimum)."""
        if len(parts) < 12:
            return None
        
        try:
            # Time and date
            time_str = parts[1]
            date_str = parts[9]
            
            if len(time_str) >= 6 and len(date_str) >= 6:
                day = int(date_str[0:2])
                month = int(date_str[2:4])
                year = 2000 + int(date_str[4:6])
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                second = float(time_str[4:])
                
                timestamp = datetime(
                    year, month, day, hour, minute,
                    int(second), int((second % 1) * 1000000)
                )
            else:
                timestamp = datetime.now()
            
            # Status
            status = parts[2]
            is_valid = status == 'A'
            
            # Latitude
            lat = self._parse_coordinate(parts[3], parts[4])
            
            # Longitude
            lon = self._parse_coordinate(parts[5], parts[6], is_longitude=True)
            
            # Speed (knots to m/s)
            speed = float(parts[7]) * 0.514444 if parts[7] else 0.0
            
            # Heading
            heading = float(parts[8]) if parts[8] else 0.0
            
            return GNSSObservation(
                observation_id=f"RMC_{timestamp.isoformat()}",
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=0.0,  # RMC doesn't have altitude
                speed=speed,
                heading=heading,
                is_valid=is_valid,
                fix_quality=FixQuality.GPS_FIX if is_valid else FixQuality.INVALID,
                raw_data={'sentence': ','.join(parts)}
            )
        
        except (ValueError, IndexError):
            return None
    
    def _parse_gll(self, parts: List[str], current_date) -> Optional[GNSSObservation]:
        """Parse GLL sentence (geographic position)."""
        if len(parts) < 7:
            return None
        
        try:
            lat = self._parse_coordinate(parts[1], parts[2])
            lon = self._parse_coordinate(parts[3], parts[4], is_longitude=True)
            
            time_str = parts[5]
            if len(time_str) >= 6:
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                second = float(time_str[4:])
                timestamp = datetime.combine(
                    current_date,
                    datetime.min.time().replace(
                        hour=hour, minute=minute, second=int(second)
                    )
                )
            else:
                timestamp = datetime.now()
            
            status = parts[6] if len(parts) > 6 else 'A'
            is_valid = status == 'A'
            
            return GNSSObservation(
                observation_id=f"GLL_{timestamp.isoformat()}",
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=0.0,
                is_valid=is_valid,
                fix_quality=FixQuality.GPS_FIX if is_valid else FixQuality.INVALID
            )
        
        except (ValueError, IndexError):
            return None
    
    def _parse_coordinate(
        self, coord_str: str, direction: str, is_longitude: bool = False
    ) -> float:
        """Parse NMEA coordinate format (DDDMM.MMMM)."""
        if not coord_str:
            return 0.0
        
        if is_longitude:
            degrees = int(coord_str[:3])
            minutes = float(coord_str[3:])
        else:
            degrees = int(coord_str[:2])
            minutes = float(coord_str[2:])
        
        decimal = degrees + minutes / 60
        
        if direction in ['S', 'W']:
            decimal = -decimal
        
        return decimal


class CSVParser:
    """Parser for CSV/ASCII GNSS exports."""
    
    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize parser.
        
        Args:
            column_mapping: Mapping of CSV columns to standard names
        """
        self.column_mapping = column_mapping or {}
        
        # Default column name patterns
        self.patterns = {
            'latitude': ['lat', 'latitude', 'y', 'northing'],
            'longitude': ['lon', 'longitude', 'lng', 'x', 'easting'],
            'altitude': ['alt', 'altitude', 'elevation', 'height', 'z'],
            'timestamp': ['time', 'timestamp', 'datetime', 'date_time', 'gps_time'],
            'hdop': ['hdop', 'h_dop', 'horizontal_dop'],
            'vdop': ['vdop', 'v_dop', 'vertical_dop'],
            'pdop': ['pdop', 'p_dop', 'position_dop'],
            'fix_quality': ['fix', 'fix_quality', 'fix_type', 'quality'],
            'num_satellites': ['sats', 'satellites', 'num_sats', 'sv_count'],
            'h_accuracy': ['h_acc', 'horizontal_accuracy', 'h_rms', 'ce90'],
            'v_accuracy': ['v_acc', 'vertical_accuracy', 'v_rms', 'le90']
        }
    
    def parse_file(self, file_path: str) -> List[GNSSObservation]:
        """Parse CSV file and return observations."""
        import csv
        
        observations = []
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # Detect delimiter
            sample = f.read(4096)
            f.seek(0)
            
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Map columns
            column_map = self._map_columns(reader.fieldnames or [])
            
            for i, row in enumerate(reader):
                obs = self._parse_row(row, column_map, i)
                if obs:
                    observations.append(obs)
        
        return observations
    
    def _map_columns(self, fieldnames: List[str]) -> Dict[str, str]:
        """Map CSV columns to standard names."""
        column_map = {}
        
        for field in fieldnames:
            field_lower = field.lower().strip()
            
            # Check custom mapping first
            if field in self.column_mapping:
                column_map[self.column_mapping[field]] = field
                continue
            
            # Check patterns
            for standard_name, patterns in self.patterns.items():
                for pattern in patterns:
                    if pattern in field_lower:
                        column_map[standard_name] = field
                        break
        
        return column_map
    
    def _parse_row(
        self, row: Dict[str, str], column_map: Dict[str, str], index: int
    ) -> Optional[GNSSObservation]:
        """Parse single CSV row."""
        try:
            # Get latitude and longitude
            lat_col = column_map.get('latitude')
            lon_col = column_map.get('longitude')
            
            if not lat_col or not lon_col:
                return None
            
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            
            # Validate coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return None
            
            # Altitude
            alt_col = column_map.get('altitude')
            altitude = float(row[alt_col]) if alt_col and row.get(alt_col) else 0.0
            
            # Timestamp
            time_col = column_map.get('timestamp')
            if time_col and row.get(time_col):
                timestamp = self._parse_timestamp(row[time_col])
            else:
                timestamp = datetime.now()
            
            # DOP values
            hdop = float(row[column_map['hdop']]) if 'hdop' in column_map and row.get(column_map['hdop']) else 0.0
            vdop = float(row[column_map['vdop']]) if 'vdop' in column_map and row.get(column_map['vdop']) else 0.0
            pdop = float(row[column_map['pdop']]) if 'pdop' in column_map and row.get(column_map['pdop']) else 0.0
            
            # Accuracy
            h_acc = float(row[column_map['h_accuracy']]) if 'h_accuracy' in column_map and row.get(column_map['h_accuracy']) else 0.0
            v_acc = float(row[column_map['v_accuracy']]) if 'v_accuracy' in column_map and row.get(column_map['v_accuracy']) else 0.0
            
            # Fix quality
            fix_col = column_map.get('fix_quality')
            fix_quality = FixQuality.GPS_FIX
            if fix_col and row.get(fix_col):
                fix_val = row[fix_col].lower()
                if 'rtk' in fix_val and 'fixed' in fix_val:
                    fix_quality = FixQuality.RTK_FIXED
                elif 'rtk' in fix_val:
                    fix_quality = FixQuality.RTK_FLOAT
                elif 'dgps' in fix_val:
                    fix_quality = FixQuality.DGPS
                elif 'ppk' in fix_val:
                    fix_quality = FixQuality.PPK
            
            # Satellites
            sat_col = column_map.get('num_satellites')
            num_sats = int(float(row[sat_col])) if sat_col and row.get(sat_col) else 0
            
            return GNSSObservation(
                observation_id=f"CSV_{index}",
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                horizontal_accuracy=h_acc,
                vertical_accuracy=v_acc,
                hdop=hdop,
                vdop=vdop,
                pdop=pdop,
                fix_quality=fix_quality,
                num_satellites=num_sats,
                raw_data=dict(row)
            )
        
        except (ValueError, KeyError, TypeError):
            return None
    
    def _parse_timestamp(self, time_str: str) -> datetime:
        """Parse various timestamp formats."""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt)
            except ValueError:
                continue
        
        # Try parsing as Unix timestamp
        try:
            return datetime.fromtimestamp(float(time_str))
        except (ValueError, OSError):
            pass
        
        return datetime.now()


class GPXParser:
    """Parser for GPX (GPS Exchange Format) files."""
    
    def parse_file(self, file_path: str) -> List[GNSSObservation]:
        """Parse GPX file and return observations."""
        import xml.etree.ElementTree as ET
        
        observations = []
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle namespace
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        if root.tag.startswith('{'):
            ns['gpx'] = root.tag.split('}')[0][1:]
        
        # Parse track points
        for trkpt in root.findall('.//gpx:trkpt', ns) or root.findall('.//trkpt'):
            obs = self._parse_trackpoint(trkpt, ns)
            if obs:
                observations.append(obs)
        
        # Parse waypoints
        for wpt in root.findall('.//gpx:wpt', ns) or root.findall('.//wpt'):
            obs = self._parse_waypoint(wpt, ns)
            if obs:
                observations.append(obs)
        
        return observations
    
    def _parse_trackpoint(self, trkpt, ns: Dict) -> Optional[GNSSObservation]:
        """Parse GPX trackpoint element."""
        try:
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            
            # Elevation
            ele_elem = trkpt.find('gpx:ele', ns) or trkpt.find('ele')
            altitude = float(ele_elem.text) if ele_elem is not None else 0.0
            
            # Time
            time_elem = trkpt.find('gpx:time', ns) or trkpt.find('time')
            if time_elem is not None:
                timestamp = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # HDOP (if available in extensions)
            hdop = 0.0
            ext = trkpt.find('gpx:extensions', ns) or trkpt.find('extensions')
            if ext is not None:
                hdop_elem = ext.find('.//hdop')
                if hdop_elem is not None:
                    hdop = float(hdop_elem.text)
            
            return GNSSObservation(
                observation_id=f"GPX_{timestamp.isoformat()}",
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                hdop=hdop,
                fix_quality=FixQuality.GPS_FIX
            )
        
        except (ValueError, TypeError):
            return None
    
    def _parse_waypoint(self, wpt, ns: Dict) -> Optional[GNSSObservation]:
        """Parse GPX waypoint element."""
        try:
            lat = float(wpt.get('lat'))
            lon = float(wpt.get('lon'))
            
            ele_elem = wpt.find('gpx:ele', ns) or wpt.find('ele')
            altitude = float(ele_elem.text) if ele_elem is not None else 0.0
            
            name_elem = wpt.find('gpx:name', ns) or wpt.find('name')
            name = name_elem.text if name_elem is not None else ""
            
            time_elem = wpt.find('gpx:time', ns) or wpt.find('time')
            if time_elem is not None:
                timestamp = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            return GNSSObservation(
                observation_id=f"WPT_{name or timestamp.isoformat()}",
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                fix_quality=FixQuality.GPS_FIX,
                raw_data={'name': name}
            )
        
        except (ValueError, TypeError):
            return None


class TrajectoryQA:
    """Quality assurance for GNSS trajectories."""
    
    def __init__(
        self,
        max_speed: float = 100.0,  # m/s
        max_hdop: float = 5.0,
        max_gap_seconds: float = 10.0,
        min_satellites: int = 4
    ):
        self.max_speed = max_speed
        self.max_hdop = max_hdop
        self.max_gap_seconds = max_gap_seconds
        self.min_satellites = min_satellites
    
    def check_trajectory(self, trajectory: GNSSTrajectory) -> Dict[str, Any]:
        """
        Perform QA checks on trajectory.
        
        Returns:
            Dictionary with QA results and issues
        """
        issues = []
        warnings = []
        
        # Check for empty trajectory
        if not trajectory.observations:
            issues.append("Empty trajectory")
            return {"valid": False, "issues": issues, "warnings": warnings}
        
        # Check time span
        if trajectory.duration_seconds < 1:
            warnings.append("Very short trajectory duration")
        
        # Check for data gaps
        if trajectory.gap_count > 0:
            warnings.append(f"{trajectory.gap_count} data gaps detected")
        
        # Check individual observations
        invalid_count = 0
        high_hdop_count = 0
        low_sat_count = 0
        speed_violations = 0
        
        prev_obs = None
        for obs in trajectory.observations:
            # Check fix quality
            if obs.fix_quality == FixQuality.INVALID:
                invalid_count += 1
            
            # Check HDOP
            if obs.hdop > self.max_hdop:
                high_hdop_count += 1
            
            # Check satellites
            if 0 < obs.num_satellites < self.min_satellites:
                low_sat_count += 1
            
            # Check speed
            if prev_obs:
                dt = (obs.timestamp - prev_obs.timestamp).total_seconds()
                if dt > 0:
                    dist = trajectory._haversine_distance(
                        prev_obs.latitude, prev_obs.longitude,
                        obs.latitude, obs.longitude
                    )
                    speed = dist / dt
                    if speed > self.max_speed:
                        speed_violations += 1
            
            prev_obs = obs
        
        n_obs = len(trajectory.observations)
        
        if invalid_count > n_obs * 0.1:
            issues.append(f"{invalid_count} invalid fixes ({invalid_count/n_obs*100:.1f}%)")
        
        if high_hdop_count > n_obs * 0.2:
            warnings.append(f"{high_hdop_count} observations with high HDOP")
        
        if low_sat_count > n_obs * 0.2:
            warnings.append(f"{low_sat_count} observations with few satellites")
        
        if speed_violations > 0:
            issues.append(f"{speed_violations} unrealistic speed jumps")
        
        # RTK quality
        if trajectory.rtk_fix_percentage < 50 and trajectory.rtk_fix_percentage > 0:
            warnings.append(f"Low RTK fix rate: {trajectory.rtk_fix_percentage:.1f}%")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "statistics": {
                "observation_count": n_obs,
                "duration_seconds": trajectory.duration_seconds,
                "total_distance_m": trajectory.total_distance,
                "mean_hdop": trajectory.mean_hdop,
                "rtk_fix_percentage": trajectory.rtk_fix_percentage,
                "invalid_fixes": invalid_count,
                "high_hdop_count": high_hdop_count,
                "speed_violations": speed_violations
            }
        }
    
    def filter_trajectory(
        self, trajectory: GNSSTrajectory, remove_invalid: bool = True
    ) -> GNSSTrajectory:
        """
        Filter trajectory to remove low-quality observations.
        """
        filtered_obs = []
        
        for obs in trajectory.observations:
            # Skip invalid fixes
            if remove_invalid and obs.fix_quality == FixQuality.INVALID:
                continue
            
            # Skip high HDOP
            if obs.hdop > self.max_hdop:
                continue
            
            # Skip low satellite count
            if 0 < obs.num_satellites < self.min_satellites:
                continue
            
            filtered_obs.append(obs)
        
        new_trajectory = GNSSTrajectory(
            trajectory_id=f"{trajectory.trajectory_id}_filtered",
            observations=filtered_obs
        )
        new_trajectory._update_statistics()
        
        return new_trajectory


class GNSSIngestionPipeline:
    """
    Complete GNSS data ingestion pipeline.
    
    Handles parsing, QA, coordinate transformation, and sample geotagging.
    """
    
    def __init__(
        self,
        target_crs: str = "EPSG:4326",
        apply_qa: bool = True
    ):
        self.target_crs = target_crs
        self.apply_qa = apply_qa
        
        self.qa_checker = TrajectoryQA()
        self.trajectories: List[GNSSTrajectory] = []
        self.observations: List[GNSSObservation] = []
    
    def ingest(self, file_path: str, format: Optional[GNSSFormat] = None) -> Dict[str, Any]:
        """
        Ingest GNSS data file.
        
        Args:
            file_path: Path to GNSS data file
            format: File format (auto-detected if not specified)
        
        Returns:
            Summary of ingested data
        """
        # Auto-detect format
        if format is None:
            format = self._detect_format(file_path)
        
        # Select parser
        parser = self._get_parser(format)
        
        # Parse file
        observations = parser.parse_file(file_path)
        
        if not observations:
            return {
                "file_path": file_path,
                "format": format.value,
                "observation_count": 0,
                "error": "No observations parsed"
            }
        
        # Create trajectory
        trajectory = GNSSTrajectory(
            trajectory_id=file_path.split('/')[-1],
            observations=observations
        )
        trajectory._update_statistics()
        
        # Apply QA
        qa_result = None
        if self.apply_qa:
            qa_result = self.qa_checker.check_trajectory(trajectory)
            trajectory = self.qa_checker.filter_trajectory(trajectory)
        
        self.trajectories.append(trajectory)
        self.observations.extend(trajectory.observations)
        
        return {
            "file_path": file_path,
            "format": format.value,
            "observation_count": len(trajectory.observations),
            "duration_seconds": trajectory.duration_seconds,
            "total_distance_m": trajectory.total_distance,
            "bounds": {
                "min_lat": trajectory.min_lat,
                "max_lat": trajectory.max_lat,
                "min_lon": trajectory.min_lon,
                "max_lon": trajectory.max_lon
            },
            "quality": qa_result
        }
    
    def _detect_format(self, file_path: str) -> GNSSFormat:
        """Auto-detect file format."""
        ext = file_path.lower().split('.')[-1]
        
        if ext == 'gpx':
            return GNSSFormat.GPX
        elif ext == 'kml':
            return GNSSFormat.KML
        elif ext in ['csv', 'txt']:
            # Check content
            with open(file_path, 'r', errors='ignore') as f:
                first_line = f.readline()
                if first_line.startswith('$GP') or first_line.startswith('$GN'):
                    return GNSSFormat.NMEA
            return GNSSFormat.CSV
        elif ext in ['nmea', 'nme']:
            return GNSSFormat.NMEA
        elif ext in ['obs', 'o']:
            return GNSSFormat.RINEX_OBS
        else:
            return GNSSFormat.CSV
    
    def _get_parser(self, format: GNSSFormat):
        """Get appropriate parser for format."""
        if format == GNSSFormat.NMEA:
            return NMEAParser()
        elif format == GNSSFormat.GPX:
            return GPXParser()
        else:
            return CSVParser()
    
    def geotag_samples(
        self,
        samples: List[Dict[str, Any]],
        timestamp_field: str = "timestamp",
        max_time_diff: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Geotag samples using GNSS trajectory.
        
        Args:
            samples: List of sample dictionaries with timestamps
            timestamp_field: Field name containing sample timestamp
            max_time_diff: Maximum time difference for matching (seconds)
        
        Returns:
            Samples with added coordinates
        """
        if not self.observations:
            return samples
        
        # Sort observations by time
        sorted_obs = sorted(self.observations, key=lambda x: x.timestamp)
        
        for sample in samples:
            sample_time = sample.get(timestamp_field)
            if not sample_time:
                continue
            
            if isinstance(sample_time, str):
                try:
                    sample_time = datetime.fromisoformat(sample_time)
                except ValueError:
                    continue
            
            # Find closest observation
            best_obs = None
            best_diff = float('inf')
            
            for obs in sorted_obs:
                diff = abs((obs.timestamp - sample_time).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_obs = obs
            
            if best_obs and best_diff <= max_time_diff:
                sample['latitude'] = best_obs.latitude
                sample['longitude'] = best_obs.longitude
                sample['altitude'] = best_obs.altitude
                sample['gnss_accuracy'] = best_obs.horizontal_accuracy
                sample['gnss_time_diff'] = best_diff
                sample['gnss_fix_quality'] = best_obs.fix_quality.name
        
        return samples
    
    def export_csv(self, output_path: str) -> None:
        """Export observations to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'observation_id', 'timestamp', 'latitude', 'longitude', 'altitude',
                'h_accuracy', 'v_accuracy', 'hdop', 'vdop', 'pdop',
                'fix_quality', 'num_satellites', 'speed', 'heading'
            ])
            
            for obs in self.observations:
                writer.writerow([
                    obs.observation_id,
                    obs.timestamp.isoformat(),
                    obs.latitude,
                    obs.longitude,
                    obs.altitude,
                    obs.horizontal_accuracy,
                    obs.vertical_accuracy,
                    obs.hdop,
                    obs.vdop,
                    obs.pdop,
                    obs.fix_quality.name,
                    obs.num_satellites,
                    obs.speed,
                    obs.heading
                ])
    
    def get_interpolated_position(self, timestamp: datetime) -> Optional[GNSSObservation]:
        """
        Get interpolated position for a given timestamp.
        """
        if not self.observations:
            return None
        
        sorted_obs = sorted(self.observations, key=lambda x: x.timestamp)
        
        # Find bracketing observations
        before = None
        after = None
        
        for obs in sorted_obs:
            if obs.timestamp <= timestamp:
                before = obs
            elif obs.timestamp > timestamp and after is None:
                after = obs
                break
        
        if before is None and after is None:
            return None
        
        if before is None:
            return after
        
        if after is None:
            return before
        
        # Interpolate
        dt_total = (after.timestamp - before.timestamp).total_seconds()
        dt_target = (timestamp - before.timestamp).total_seconds()
        
        if dt_total == 0:
            return before
        
        ratio = dt_target / dt_total
        
        return GNSSObservation(
            observation_id=f"INTERP_{timestamp.isoformat()}",
            timestamp=timestamp,
            latitude=before.latitude + ratio * (after.latitude - before.latitude),
            longitude=before.longitude + ratio * (after.longitude - before.longitude),
            altitude=before.altitude + ratio * (after.altitude - before.altitude),
            horizontal_accuracy=max(before.horizontal_accuracy, after.horizontal_accuracy),
            vertical_accuracy=max(before.vertical_accuracy, after.vertical_accuracy),
            fix_quality=FixQuality.ESTIMATED
        )


def create_gnss_pipeline(apply_qa: bool = True) -> GNSSIngestionPipeline:
    """
    Factory function to create GNSS ingestion pipeline.
    
    Args:
        apply_qa: Whether to apply QA checks
    
    Returns:
        Configured GNSSIngestionPipeline
    """
    return GNSSIngestionPipeline(apply_qa=apply_qa)
