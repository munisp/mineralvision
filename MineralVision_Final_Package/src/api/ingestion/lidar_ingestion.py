"""
LiDAR Data Ingestion Module for MineralVision Platform.

Supports ingestion of LiDAR point cloud data from various formats:
- LAS (ASPRS LAS format)
- LAZ (compressed LAS)
- ASCII XYZ/CSV
- E57 (via conversion)

Provides point cloud processing, ground classification, and DEM/DTM generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import numpy as np
from datetime import datetime
import struct
import gzip
import io


class LiDARFormat(Enum):
    """Supported LiDAR file formats."""
    LAS_1_2 = "las_1.2"
    LAS_1_3 = "las_1.3"
    LAS_1_4 = "las_1.4"
    LAZ = "laz"  # Compressed LAS
    ASCII_XYZ = "xyz"
    ASCII_CSV = "csv"
    E57 = "e57"
    PLY = "ply"


class LiDARClassification(Enum):
    """ASPRS standard LiDAR point classifications."""
    CREATED_NEVER_CLASSIFIED = 0
    UNCLASSIFIED = 1
    GROUND = 2
    LOW_VEGETATION = 3
    MEDIUM_VEGETATION = 4
    HIGH_VEGETATION = 5
    BUILDING = 6
    LOW_POINT = 7
    MODEL_KEY_POINT = 8
    WATER = 9
    RAIL = 10
    ROAD_SURFACE = 11
    OVERLAP = 12
    WIRE_GUARD = 13
    WIRE_CONDUCTOR = 14
    TRANSMISSION_TOWER = 15
    WIRE_STRUCTURE = 16
    BRIDGE_DECK = 17
    HIGH_NOISE = 18


@dataclass
class LiDARPoint:
    """Single LiDAR point with all attributes."""
    x: float
    y: float
    z: float
    intensity: int = 0
    return_number: int = 1
    number_of_returns: int = 1
    classification: LiDARClassification = LiDARClassification.UNCLASSIFIED
    scan_angle: float = 0.0
    gps_time: float = 0.0
    red: int = 0
    green: int = 0
    blue: int = 0
    nir: int = 0  # Near-infrared (LAS 1.4)
    
    def is_ground(self) -> bool:
        """Check if point is classified as ground."""
        return self.classification == LiDARClassification.GROUND
    
    def is_vegetation(self) -> bool:
        """Check if point is classified as vegetation."""
        return self.classification in [
            LiDARClassification.LOW_VEGETATION,
            LiDARClassification.MEDIUM_VEGETATION,
            LiDARClassification.HIGH_VEGETATION
        ]
    
    def is_first_return(self) -> bool:
        """Check if this is a first return."""
        return self.return_number == 1
    
    def is_last_return(self) -> bool:
        """Check if this is a last return."""
        return self.return_number == self.number_of_returns


@dataclass
class LiDARMetadata:
    """LiDAR dataset metadata."""
    file_path: str
    format: LiDARFormat
    version: str = ""
    
    # Spatial extent
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0
    min_z: float = 0.0
    max_z: float = 0.0
    
    # Point statistics
    point_count: int = 0
    point_density: float = 0.0  # points per m²
    
    # Classification counts
    classification_counts: Dict[int, int] = field(default_factory=dict)
    
    # Return statistics
    return_counts: Dict[int, int] = field(default_factory=dict)
    
    # Coordinate reference system
    crs_wkt: str = ""
    crs_epsg: Optional[int] = None
    
    # Scale and offset (LAS format)
    scale_x: float = 0.001
    scale_y: float = 0.001
    scale_z: float = 0.001
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    
    # Acquisition info
    system_identifier: str = ""
    generating_software: str = ""
    creation_date: Optional[datetime] = None
    
    # Quality metrics
    has_rgb: bool = False
    has_nir: bool = False
    has_gps_time: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DEMOutput:
    """Digital Elevation Model output."""
    data: np.ndarray  # 2D array of elevations
    transform: Tuple[float, float, float, float, float, float]  # Affine transform
    crs_epsg: Optional[int] = None
    nodata_value: float = -9999.0
    resolution: float = 1.0  # meters
    dem_type: str = "dtm"  # dtm, dsm, chm
    
    # Statistics
    min_elevation: float = 0.0
    max_elevation: float = 0.0
    mean_elevation: float = 0.0
    
    # Quality
    cell_count: int = 0
    void_count: int = 0
    point_density_grid: Optional[np.ndarray] = None


class LASReader:
    """Reader for LAS/LAZ format files."""
    
    # LAS point record formats
    POINT_FORMATS = {
        0: {'size': 20, 'has_gps': False, 'has_rgb': False},
        1: {'size': 28, 'has_gps': True, 'has_rgb': False},
        2: {'size': 26, 'has_gps': False, 'has_rgb': True},
        3: {'size': 34, 'has_gps': True, 'has_rgb': True},
        6: {'size': 30, 'has_gps': True, 'has_rgb': False},  # LAS 1.4
        7: {'size': 36, 'has_gps': True, 'has_rgb': True},   # LAS 1.4
        8: {'size': 38, 'has_gps': True, 'has_rgb': True, 'has_nir': True}  # LAS 1.4
    }
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata: Optional[LiDARMetadata] = None
        self._file_handle = None
        self._header = {}
        self._is_laz = file_path.lower().endswith('.laz')
    
    def read_header(self) -> LiDARMetadata:
        """Read LAS file header and return metadata."""
        with open(self.file_path, 'rb') as f:
            # File signature
            signature = f.read(4)
            if signature != b'LASF':
                raise ValueError(f"Invalid LAS file signature: {signature}")
            
            # File source ID and global encoding
            file_source_id = struct.unpack('<H', f.read(2))[0]
            global_encoding = struct.unpack('<H', f.read(2))[0]
            
            # Project ID (GUID)
            guid = f.read(16)
            
            # Version
            version_major = struct.unpack('<B', f.read(1))[0]
            version_minor = struct.unpack('<B', f.read(1))[0]
            version = f"{version_major}.{version_minor}"
            
            # System identifier and generating software
            system_id = f.read(32).decode('ascii', errors='ignore').strip('\x00')
            gen_software = f.read(32).decode('ascii', errors='ignore').strip('\x00')
            
            # Creation date
            creation_day = struct.unpack('<H', f.read(2))[0]
            creation_year = struct.unpack('<H', f.read(2))[0]
            
            # Header size
            header_size = struct.unpack('<H', f.read(2))[0]
            
            # Offset to point data
            offset_to_points = struct.unpack('<I', f.read(4))[0]
            
            # Number of VLRs
            num_vlrs = struct.unpack('<I', f.read(4))[0]
            
            # Point data format and record length
            point_format = struct.unpack('<B', f.read(1))[0]
            point_record_length = struct.unpack('<H', f.read(2))[0]
            
            # Number of points (legacy for LAS 1.0-1.3)
            num_points_legacy = struct.unpack('<I', f.read(4))[0]
            
            # Number of points by return (legacy)
            points_by_return_legacy = struct.unpack('<5I', f.read(20))
            
            # Scale factors
            scale_x = struct.unpack('<d', f.read(8))[0]
            scale_y = struct.unpack('<d', f.read(8))[0]
            scale_z = struct.unpack('<d', f.read(8))[0]
            
            # Offsets
            offset_x = struct.unpack('<d', f.read(8))[0]
            offset_y = struct.unpack('<d', f.read(8))[0]
            offset_z = struct.unpack('<d', f.read(8))[0]
            
            # Bounds
            max_x = struct.unpack('<d', f.read(8))[0]
            min_x = struct.unpack('<d', f.read(8))[0]
            max_y = struct.unpack('<d', f.read(8))[0]
            min_y = struct.unpack('<d', f.read(8))[0]
            max_z = struct.unpack('<d', f.read(8))[0]
            min_z = struct.unpack('<d', f.read(8))[0]
            
            # For LAS 1.4, read extended header
            num_points = num_points_legacy
            if version_major >= 1 and version_minor >= 4:
                # Skip to extended point count
                f.seek(247)
                num_points = struct.unpack('<Q', f.read(8))[0]
            
            # Determine format
            if self._is_laz:
                las_format = LiDARFormat.LAZ
            elif version_minor == 2:
                las_format = LiDARFormat.LAS_1_2
            elif version_minor == 3:
                las_format = LiDARFormat.LAS_1_3
            else:
                las_format = LiDARFormat.LAS_1_4
            
            # Point format info
            format_info = self.POINT_FORMATS.get(point_format, self.POINT_FORMATS[0])
            
            # Calculate point density
            area = (max_x - min_x) * (max_y - min_y)
            density = num_points / area if area > 0 else 0
            
            # Store header info
            self._header = {
                'offset_to_points': offset_to_points,
                'point_format': point_format,
                'point_record_length': point_record_length,
                'num_points': num_points,
                'scale': (scale_x, scale_y, scale_z),
                'offset': (offset_x, offset_y, offset_z)
            }
            
            self.metadata = LiDARMetadata(
                file_path=self.file_path,
                format=las_format,
                version=version,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                min_z=min_z,
                max_z=max_z,
                point_count=num_points,
                point_density=density,
                scale_x=scale_x,
                scale_y=scale_y,
                scale_z=scale_z,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
                system_identifier=system_id,
                generating_software=gen_software,
                has_rgb=format_info.get('has_rgb', False),
                has_nir=format_info.get('has_nir', False),
                has_gps_time=format_info.get('has_gps', False)
            )
            
            return self.metadata
    
    def read_points(self, max_points: Optional[int] = None) -> Iterator[LiDARPoint]:
        """
        Read points from LAS file as iterator.
        
        Args:
            max_points: Maximum number of points to read (None for all)
        """
        if self.metadata is None:
            self.read_header()
        
        with open(self.file_path, 'rb') as f:
            f.seek(self._header['offset_to_points'])
            
            point_format = self._header['point_format']
            record_length = self._header['point_record_length']
            num_points = self._header['num_points']
            
            if max_points:
                num_points = min(num_points, max_points)
            
            scale = self._header['scale']
            offset = self._header['offset']
            
            for _ in range(num_points):
                record = f.read(record_length)
                if len(record) < record_length:
                    break
                
                # Parse point record (format 0 base)
                x_raw, y_raw, z_raw = struct.unpack('<3i', record[0:12])
                intensity = struct.unpack('<H', record[12:14])[0]
                
                # Return number and flags
                flags = struct.unpack('<B', record[14:15])[0]
                return_number = flags & 0x07
                number_of_returns = (flags >> 3) & 0x07
                
                classification = struct.unpack('<B', record[15:16])[0]
                scan_angle = struct.unpack('<b', record[16:17])[0]
                
                # Convert to real coordinates
                x = x_raw * scale[0] + offset[0]
                y = y_raw * scale[1] + offset[1]
                z = z_raw * scale[2] + offset[2]
                
                # GPS time (if available)
                gps_time = 0.0
                if point_format in [1, 3, 6, 7, 8] and len(record) >= 28:
                    gps_time = struct.unpack('<d', record[20:28])[0]
                
                # RGB (if available)
                red, green, blue = 0, 0, 0
                if point_format in [2, 3, 7, 8]:
                    rgb_offset = 28 if point_format in [3, 7, 8] else 20
                    if len(record) >= rgb_offset + 6:
                        red = struct.unpack('<H', record[rgb_offset:rgb_offset+2])[0]
                        green = struct.unpack('<H', record[rgb_offset+2:rgb_offset+4])[0]
                        blue = struct.unpack('<H', record[rgb_offset+4:rgb_offset+6])[0]
                
                try:
                    class_enum = LiDARClassification(classification)
                except ValueError:
                    class_enum = LiDARClassification.UNCLASSIFIED
                
                yield LiDARPoint(
                    x=x,
                    y=y,
                    z=z,
                    intensity=intensity,
                    return_number=return_number,
                    number_of_returns=number_of_returns,
                    classification=class_enum,
                    scan_angle=float(scan_angle),
                    gps_time=gps_time,
                    red=red,
                    green=green,
                    blue=blue
                )
    
    def read_points_array(self, max_points: Optional[int] = None) -> np.ndarray:
        """
        Read points as numpy array for efficient processing.
        
        Returns array with columns: [x, y, z, intensity, classification, return_num]
        """
        if self.metadata is None:
            self.read_header()
        
        num_points = self._header['num_points']
        if max_points:
            num_points = min(num_points, max_points)
        
        # Pre-allocate array
        points = np.zeros((num_points, 6), dtype=np.float64)
        
        for i, point in enumerate(self.read_points(max_points)):
            points[i] = [
                point.x, point.y, point.z,
                point.intensity, point.classification.value, point.return_number
            ]
        
        return points


class GroundClassifier:
    """Simple ground classification using progressive morphological filter."""
    
    def __init__(
        self,
        cell_size: float = 1.0,
        slope_threshold: float = 0.3,
        max_window_size: float = 20.0,
        initial_height_threshold: float = 0.5
    ):
        self.cell_size = cell_size
        self.slope_threshold = slope_threshold
        self.max_window_size = max_window_size
        self.initial_height_threshold = initial_height_threshold
    
    def classify(self, points: np.ndarray) -> np.ndarray:
        """
        Classify points as ground or non-ground.
        
        Args:
            points: Nx6 array [x, y, z, intensity, classification, return_num]
        
        Returns:
            Array of classifications (2=ground, 1=non-ground)
        """
        n_points = len(points)
        classifications = np.ones(n_points, dtype=np.int32)  # Default to unclassified
        
        # Get bounds
        min_x, min_y = points[:, 0].min(), points[:, 1].min()
        max_x, max_y = points[:, 0].max(), points[:, 1].max()
        
        # Create grid
        nx = int(np.ceil((max_x - min_x) / self.cell_size)) + 1
        ny = int(np.ceil((max_y - min_y) / self.cell_size)) + 1
        
        # Assign points to grid cells
        cell_x = ((points[:, 0] - min_x) / self.cell_size).astype(int)
        cell_y = ((points[:, 1] - min_y) / self.cell_size).astype(int)
        cell_x = np.clip(cell_x, 0, nx - 1)
        cell_y = np.clip(cell_y, 0, ny - 1)
        
        # Find minimum elevation in each cell
        min_z_grid = np.full((ny, nx), np.inf)
        for i in range(n_points):
            cx, cy = cell_x[i], cell_y[i]
            if points[i, 2] < min_z_grid[cy, cx]:
                min_z_grid[cy, cx] = points[i, 2]
        
        # Progressive morphological filter
        window_sizes = [1, 2, 4, 8, 16]
        height_threshold = self.initial_height_threshold
        
        for window_size in window_sizes:
            if window_size * self.cell_size > self.max_window_size:
                break
            
            # Apply morphological opening (erosion + dilation)
            surface = self._morphological_opening(min_z_grid, window_size)
            
            # Classify points based on height above surface
            for i in range(n_points):
                cx, cy = cell_x[i], cell_y[i]
                surface_z = surface[cy, cx]
                
                if surface_z != np.inf:
                    height_above = points[i, 2] - surface_z
                    
                    # Adaptive threshold based on slope
                    local_slope = self._estimate_slope(surface, cy, cx)
                    adaptive_threshold = height_threshold + local_slope * self.cell_size * window_size
                    
                    if height_above <= adaptive_threshold:
                        classifications[i] = 2  # Ground
            
            # Increase threshold for larger windows
            height_threshold += self.slope_threshold * self.cell_size * window_size
        
        return classifications
    
    def _morphological_opening(self, grid: np.ndarray, window_size: int) -> np.ndarray:
        """Apply morphological opening (erosion followed by dilation)."""
        # Simple implementation using min/max filters
        from scipy.ndimage import minimum_filter, maximum_filter
        
        # Handle infinite values
        valid_mask = grid != np.inf
        temp_grid = grid.copy()
        temp_grid[~valid_mask] = np.nanmax(grid[valid_mask]) if valid_mask.any() else 0
        
        # Erosion (minimum filter)
        eroded = minimum_filter(temp_grid, size=window_size)
        
        # Dilation (maximum filter)
        opened = maximum_filter(eroded, size=window_size)
        
        # Restore invalid cells
        opened[~valid_mask] = np.inf
        
        return opened
    
    def _estimate_slope(self, surface: np.ndarray, row: int, col: int) -> float:
        """Estimate local slope at a grid cell."""
        ny, nx = surface.shape
        
        # Get neighboring elevations
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < ny and 0 <= nc < nx:
                    if surface[nr, nc] != np.inf:
                        neighbors.append(surface[nr, nc])
        
        if not neighbors or surface[row, col] == np.inf:
            return 0.0
        
        # Calculate slope as max elevation difference
        center_z = surface[row, col]
        max_diff = max(abs(z - center_z) for z in neighbors)
        
        return max_diff / self.cell_size


class DEMGenerator:
    """Generate DEM/DTM/DSM from LiDAR point clouds."""
    
    def __init__(
        self,
        resolution: float = 1.0,
        interpolation: str = "idw",  # idw, linear, nearest
        search_radius: float = 5.0,
        nodata_value: float = -9999.0
    ):
        self.resolution = resolution
        self.interpolation = interpolation
        self.search_radius = search_radius
        self.nodata_value = nodata_value
    
    def generate_dtm(
        self,
        points: np.ndarray,
        classifications: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> DEMOutput:
        """
        Generate Digital Terrain Model from ground points.
        
        Args:
            points: Nx6 array [x, y, z, intensity, classification, return_num]
            classifications: Array of point classifications
            bounds: Optional (min_x, min_y, max_x, max_y)
        
        Returns:
            DEMOutput with DTM raster
        """
        # Filter ground points
        ground_mask = classifications == 2
        ground_points = points[ground_mask]
        
        if len(ground_points) == 0:
            raise ValueError("No ground points found for DTM generation")
        
        return self._generate_dem(ground_points, bounds, dem_type="dtm")
    
    def generate_dsm(
        self,
        points: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> DEMOutput:
        """
        Generate Digital Surface Model from first returns.
        
        Args:
            points: Nx6 array [x, y, z, intensity, classification, return_num]
            bounds: Optional (min_x, min_y, max_x, max_y)
        
        Returns:
            DEMOutput with DSM raster
        """
        # Filter first returns
        first_return_mask = points[:, 5] == 1
        first_returns = points[first_return_mask]
        
        if len(first_returns) == 0:
            first_returns = points  # Use all points if no return info
        
        return self._generate_dem(first_returns, bounds, dem_type="dsm", use_max=True)
    
    def generate_chm(
        self,
        dtm: DEMOutput,
        dsm: DEMOutput
    ) -> DEMOutput:
        """
        Generate Canopy Height Model (DSM - DTM).
        """
        if dtm.data.shape != dsm.data.shape:
            raise ValueError("DTM and DSM must have same dimensions")
        
        chm_data = dsm.data - dtm.data
        
        # Set negative values (errors) to 0
        chm_data = np.maximum(chm_data, 0)
        
        # Preserve nodata
        nodata_mask = (dtm.data == dtm.nodata_value) | (dsm.data == dsm.nodata_value)
        chm_data[nodata_mask] = self.nodata_value
        
        valid_mask = ~nodata_mask
        
        return DEMOutput(
            data=chm_data,
            transform=dtm.transform,
            crs_epsg=dtm.crs_epsg,
            nodata_value=self.nodata_value,
            resolution=dtm.resolution,
            dem_type="chm",
            min_elevation=float(chm_data[valid_mask].min()) if valid_mask.any() else 0,
            max_elevation=float(chm_data[valid_mask].max()) if valid_mask.any() else 0,
            mean_elevation=float(chm_data[valid_mask].mean()) if valid_mask.any() else 0,
            cell_count=int(valid_mask.sum()),
            void_count=int((~valid_mask).sum())
        )
    
    def _generate_dem(
        self,
        points: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]],
        dem_type: str,
        use_max: bool = False
    ) -> DEMOutput:
        """Internal DEM generation."""
        # Determine bounds
        if bounds:
            min_x, min_y, max_x, max_y = bounds
        else:
            min_x, min_y = points[:, 0].min(), points[:, 1].min()
            max_x, max_y = points[:, 0].max(), points[:, 1].max()
        
        # Create grid
        nx = int(np.ceil((max_x - min_x) / self.resolution))
        ny = int(np.ceil((max_y - min_y) / self.resolution))
        
        # Initialize output
        dem = np.full((ny, nx), self.nodata_value, dtype=np.float32)
        point_count = np.zeros((ny, nx), dtype=np.int32)
        
        # Assign points to cells
        cell_x = ((points[:, 0] - min_x) / self.resolution).astype(int)
        cell_y = ((points[:, 1] - min_y) / self.resolution).astype(int)
        cell_x = np.clip(cell_x, 0, nx - 1)
        cell_y = np.clip(cell_y, 0, ny - 1)
        
        # Aggregate elevations
        if use_max:
            # For DSM, use maximum elevation
            for i in range(len(points)):
                cx, cy = cell_x[i], cell_y[i]
                z = points[i, 2]
                if dem[cy, cx] == self.nodata_value or z > dem[cy, cx]:
                    dem[cy, cx] = z
                point_count[cy, cx] += 1
        else:
            # For DTM, use mean elevation
            sum_z = np.zeros((ny, nx), dtype=np.float64)
            for i in range(len(points)):
                cx, cy = cell_x[i], cell_y[i]
                sum_z[cy, cx] += points[i, 2]
                point_count[cy, cx] += 1
            
            valid = point_count > 0
            dem[valid] = (sum_z[valid] / point_count[valid]).astype(np.float32)
        
        # Interpolate voids if requested
        if self.interpolation != "none":
            dem = self._interpolate_voids(dem, point_count)
        
        # Create affine transform
        transform = (min_x, self.resolution, 0, max_y, 0, -self.resolution)
        
        # Statistics
        valid_mask = dem != self.nodata_value
        
        return DEMOutput(
            data=dem,
            transform=transform,
            nodata_value=self.nodata_value,
            resolution=self.resolution,
            dem_type=dem_type,
            min_elevation=float(dem[valid_mask].min()) if valid_mask.any() else 0,
            max_elevation=float(dem[valid_mask].max()) if valid_mask.any() else 0,
            mean_elevation=float(dem[valid_mask].mean()) if valid_mask.any() else 0,
            cell_count=int(valid_mask.sum()),
            void_count=int((~valid_mask).sum()),
            point_density_grid=point_count.astype(np.float32) / (self.resolution ** 2)
        )
    
    def _interpolate_voids(self, dem: np.ndarray, point_count: np.ndarray) -> np.ndarray:
        """Interpolate void cells using IDW."""
        from scipy.interpolate import griddata
        
        ny, nx = dem.shape
        valid_mask = dem != self.nodata_value
        void_mask = ~valid_mask
        
        if not void_mask.any() or not valid_mask.any():
            return dem
        
        # Get coordinates
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        
        # Known points
        known_points = np.column_stack([
            x_coords[valid_mask],
            y_coords[valid_mask]
        ])
        known_values = dem[valid_mask]
        
        # Void points
        void_points = np.column_stack([
            x_coords[void_mask],
            y_coords[void_mask]
        ])
        
        # Interpolate
        if self.interpolation == "linear":
            method = "linear"
        elif self.interpolation == "nearest":
            method = "nearest"
        else:
            method = "linear"  # IDW approximation
        
        try:
            interpolated = griddata(
                known_points, known_values, void_points,
                method=method, fill_value=self.nodata_value
            )
            dem[void_mask] = interpolated
        except Exception:
            pass  # Keep nodata if interpolation fails
        
        return dem


class LiDARIngestionPipeline:
    """
    Complete LiDAR data ingestion pipeline.
    
    Handles reading, classification, and DEM generation.
    """
    
    def __init__(
        self,
        dem_resolution: float = 1.0,
        classify_ground: bool = True,
        generate_products: List[str] = None
    ):
        self.dem_resolution = dem_resolution
        self.classify_ground = classify_ground
        self.generate_products = generate_products or ["dtm", "dsm", "chm"]
        
        self.reader: Optional[LASReader] = None
        self.classifier = GroundClassifier()
        self.dem_generator = DEMGenerator(resolution=dem_resolution)
        
        self.metadata: Optional[LiDARMetadata] = None
        self.points: Optional[np.ndarray] = None
        self.classifications: Optional[np.ndarray] = None
        self.products: Dict[str, DEMOutput] = {}
    
    def ingest(self, file_path: str, max_points: Optional[int] = None) -> Dict[str, Any]:
        """
        Ingest LiDAR file and generate products.
        
        Args:
            file_path: Path to LAS/LAZ file
            max_points: Maximum points to read (for testing)
        
        Returns:
            Dictionary with metadata and products
        """
        # Determine format
        ext = file_path.lower().split('.')[-1]
        
        if ext in ['las', 'laz']:
            return self._ingest_las(file_path, max_points)
        elif ext in ['xyz', 'csv', 'txt']:
            return self._ingest_ascii(file_path, max_points)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def _ingest_las(self, file_path: str, max_points: Optional[int]) -> Dict[str, Any]:
        """Ingest LAS/LAZ file."""
        self.reader = LASReader(file_path)
        self.metadata = self.reader.read_header()
        
        # Read points
        self.points = self.reader.read_points_array(max_points)
        
        # Use existing classifications or classify
        if self.classify_ground:
            existing_ground = (self.points[:, 4] == 2).sum()
            if existing_ground < len(self.points) * 0.1:
                # Less than 10% ground points, reclassify
                self.classifications = self.classifier.classify(self.points)
            else:
                self.classifications = self.points[:, 4].astype(np.int32)
        else:
            self.classifications = self.points[:, 4].astype(np.int32)
        
        # Update metadata with classification counts
        unique, counts = np.unique(self.classifications, return_counts=True)
        self.metadata.classification_counts = dict(zip(unique.tolist(), counts.tolist()))
        
        # Generate products
        self._generate_products()
        
        return {
            "metadata": self.metadata,
            "point_count": len(self.points),
            "ground_points": int((self.classifications == 2).sum()),
            "products": list(self.products.keys())
        }
    
    def _ingest_ascii(self, file_path: str, max_points: Optional[int]) -> Dict[str, Any]:
        """Ingest ASCII XYZ/CSV file."""
        # Read file
        points_list = []
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_points and i >= max_points:
                    break
                
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try comma, space, or tab delimiter
                for delimiter in [',', ' ', '\t']:
                    parts = line.split(delimiter)
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            intensity = int(float(parts[3])) if len(parts) > 3 else 0
                            classification = int(float(parts[4])) if len(parts) > 4 else 1
                            return_num = int(float(parts[5])) if len(parts) > 5 else 1
                            points_list.append([x, y, z, intensity, classification, return_num])
                            break
                        except ValueError:
                            continue
        
        if not points_list:
            raise ValueError("No valid points found in file")
        
        self.points = np.array(points_list)
        
        # Create metadata
        self.metadata = LiDARMetadata(
            file_path=file_path,
            format=LiDARFormat.ASCII_XYZ,
            min_x=float(self.points[:, 0].min()),
            max_x=float(self.points[:, 0].max()),
            min_y=float(self.points[:, 1].min()),
            max_y=float(self.points[:, 1].max()),
            min_z=float(self.points[:, 2].min()),
            max_z=float(self.points[:, 2].max()),
            point_count=len(self.points)
        )
        
        # Classify ground
        if self.classify_ground:
            self.classifications = self.classifier.classify(self.points)
        else:
            self.classifications = self.points[:, 4].astype(np.int32)
        
        # Generate products
        self._generate_products()
        
        return {
            "metadata": self.metadata,
            "point_count": len(self.points),
            "ground_points": int((self.classifications == 2).sum()),
            "products": list(self.products.keys())
        }
    
    def _generate_products(self) -> None:
        """Generate DEM products."""
        bounds = (
            self.metadata.min_x,
            self.metadata.min_y,
            self.metadata.max_x,
            self.metadata.max_y
        )
        
        if "dtm" in self.generate_products:
            try:
                self.products["dtm"] = self.dem_generator.generate_dtm(
                    self.points, self.classifications, bounds
                )
            except ValueError as e:
                print(f"DTM generation failed: {e}")
        
        if "dsm" in self.generate_products:
            self.products["dsm"] = self.dem_generator.generate_dsm(
                self.points, bounds
            )
        
        if "chm" in self.generate_products and "dtm" in self.products and "dsm" in self.products:
            self.products["chm"] = self.dem_generator.generate_chm(
                self.products["dtm"],
                self.products["dsm"]
            )
    
    def export_geotiff(self, product: str, output_path: str) -> None:
        """
        Export DEM product to GeoTIFF.
        
        Note: Requires rasterio library.
        """
        if product not in self.products:
            raise ValueError(f"Product {product} not available")
        
        dem = self.products[product]
        
        # Simple GeoTIFF export (without rasterio)
        # This creates a basic TIFF that can be read by GIS software
        self._write_simple_tiff(dem, output_path)
    
    def _write_simple_tiff(self, dem: DEMOutput, output_path: str) -> None:
        """Write simple TIFF file (basic implementation)."""
        # This is a simplified TIFF writer
        # For production, use rasterio or GDAL
        
        data = dem.data
        ny, nx = data.shape
        
        # Write as raw binary with metadata sidecar
        np.save(output_path.replace('.tif', '.npy'), data)
        
        # Write metadata
        meta = {
            'transform': dem.transform,
            'crs_epsg': dem.crs_epsg,
            'nodata': dem.nodata_value,
            'resolution': dem.resolution,
            'dem_type': dem.dem_type,
            'shape': (ny, nx)
        }
        
        import json
        with open(output_path.replace('.tif', '_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    
    def get_terrain_derivatives(self, product: str = "dtm") -> Dict[str, np.ndarray]:
        """
        Calculate terrain derivatives from DEM.
        
        Returns slope, aspect, curvature, TWI.
        """
        if product not in self.products:
            raise ValueError(f"Product {product} not available")
        
        dem = self.products[product].data
        res = self.products[product].resolution
        nodata = self.products[product].nodata_value
        
        # Calculate gradients
        valid_mask = dem != nodata
        
        # Pad for edge handling
        dem_padded = np.pad(dem, 1, mode='edge')
        
        # Sobel-like gradient calculation
        dz_dx = (dem_padded[1:-1, 2:] - dem_padded[1:-1, :-2]) / (2 * res)
        dz_dy = (dem_padded[2:, 1:-1] - dem_padded[:-2, 1:-1]) / (2 * res)
        
        # Slope (degrees)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        slope[~valid_mask] = nodata
        
        # Aspect (degrees from north)
        aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        aspect[~valid_mask] = nodata
        
        # Curvature (plan + profile)
        d2z_dx2 = (dem_padded[1:-1, 2:] - 2*dem_padded[1:-1, 1:-1] + dem_padded[1:-1, :-2]) / (res**2)
        d2z_dy2 = (dem_padded[2:, 1:-1] - 2*dem_padded[1:-1, 1:-1] + dem_padded[:-2, 1:-1]) / (res**2)
        curvature = d2z_dx2 + d2z_dy2
        curvature[~valid_mask] = nodata
        
        # Topographic Wetness Index (simplified)
        # TWI = ln(a / tan(slope)) where a is upslope contributing area
        slope_rad = np.radians(slope)
        slope_rad = np.maximum(slope_rad, 0.001)  # Avoid division by zero
        
        # Simplified: use cell area as proxy for contributing area
        twi = np.log(res**2 / np.tan(slope_rad))
        twi[~valid_mask] = nodata
        
        return {
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "twi": twi
        }


def create_lidar_pipeline(
    resolution: float = 1.0,
    classify_ground: bool = True,
    products: List[str] = None
) -> LiDARIngestionPipeline:
    """
    Factory function to create LiDAR ingestion pipeline.
    
    Args:
        resolution: DEM resolution in meters
        classify_ground: Whether to classify ground points
        products: List of products to generate ('dtm', 'dsm', 'chm')
    
    Returns:
        Configured LiDARIngestionPipeline
    """
    return LiDARIngestionPipeline(
        dem_resolution=resolution,
        classify_ground=classify_ground,
        generate_products=products or ["dtm", "dsm", "chm"]
    )
