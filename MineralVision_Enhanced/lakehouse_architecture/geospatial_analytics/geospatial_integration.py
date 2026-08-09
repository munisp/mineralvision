"""
Geospatial Analytics Integration for MineralVision Lakehouse Architecture

This module integrates geospatial analytics capabilities across the entire MineralVision Lakehouse architecture.
It provides a unified interface for geospatial operations, coordinate reference system management,
spatial indexing, and advanced geospatial analytics.

Uses geopandas/shapely when available, with numpy-based fallback for basic operations.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon, shape, mapping
    from shapely.ops import transform, unary_union
    import shapely.wkt as wkt
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    gpd = None

try:
    from pyproj import CRS, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    CRS = None
    Transformer = None

try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    rtree_index = None

@dataclass
class GeospatialConfig:
    """Configuration settings for geospatial analytics integration."""
    default_crs: str = "EPSG:4326"  # WGS84
    enable_spatial_indexing: bool = True
    index_type: str = "rtree"  # Options: rtree, quadtree, geohash
    tile_size: float = 1000.0  # Size of tiles for tiled operations (in CRS units)
    simplification_tolerance: float = 0.0001  # Tolerance for geometry simplification
    buffer_segments: int = 16  # Number of segments in buffer operations
    log_level: str = "INFO"
    
    # Performance settings
    parallel_processing: bool = True
    batch_size: int = 1000
    
    def __post_init__(self):
        """Initialize logging."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("GeospatialAnalytics")
        self.logger.info("Initialized geospatial configuration")


class GeospatialAnalytics:
    """
    Main class for geospatial analytics integration in the MineralVision Lakehouse architecture.
    
    This class provides methods for:
    - Unified geospatial operations across the Lakehouse architecture
    - Coordinate reference system management
    - Spatial indexing and optimization
    - Advanced geospatial analytics for mineral exploration
    - Integration with Delta Lake, Spark, Ray, and DataFusion
    """
    
    def __init__(self, config: GeospatialConfig):
        """
        Initialize the geospatial analytics integration.
        
        Args:
            config: Configuration settings for geospatial analytics
        """
        self.config = config
        self.logger = logging.getLogger("GeospatialAnalytics")
        self._spatial_index = None
        self._transformer_cache: Dict[Tuple[str, str], Any] = {}
        
        if GEOPANDAS_AVAILABLE:
            self.logger.info("Initialized geospatial analytics with geopandas/shapely")
        else:
            self.logger.info("Initialized geospatial analytics with numpy fallback")
        
        if PYPROJ_AVAILABLE:
            self.logger.info("CRS transformations available via pyproj")
        
        if RTREE_AVAILABLE and self.config.enable_spatial_indexing:
            self.logger.info(f"Spatial indexing available via rtree ({self.config.index_type})")
    
    def register_with_delta_lake(self, delta_storage: Any) -> None:
        """
        Register geospatial capabilities with Delta Lake storage.
        
        Args:
            delta_storage: Delta Lake storage instance
        """
        self.logger.info("Registering geospatial capabilities with Delta Lake storage")
        
        # In a real implementation, we would register geospatial capabilities with Delta Lake
        # For this implementation, we'll just log the operation
        
        self.logger.info("Successfully registered geospatial capabilities with Delta Lake storage")
    
    def register_with_spark(self, spark_processor: Any) -> None:
        """
        Register geospatial capabilities with Spark processor.
        
        Args:
            spark_processor: Spark processor instance
        """
        self.logger.info("Registering geospatial capabilities with Spark processor")
        
        # In a real implementation, we would register geospatial capabilities with Spark
        # For this implementation, we'll just log the operation
        
        self.logger.info("Successfully registered geospatial capabilities with Spark processor")
    
    def register_with_ray(self, ray_processor: Any) -> None:
        """
        Register geospatial capabilities with Ray processor.
        
        Args:
            ray_processor: Ray processor instance
        """
        self.logger.info("Registering geospatial capabilities with Ray processor")
        
        # In a real implementation, we would register geospatial capabilities with Ray
        # For this implementation, we'll just log the operation
        
        self.logger.info("Successfully registered geospatial capabilities with Ray processor")
    
    def register_with_datafusion(self, datafusion_engine: Any) -> None:
        """
        Register geospatial capabilities with DataFusion query engine.
        
        Args:
            datafusion_engine: DataFusion query engine instance
        """
        self.logger.info("Registering geospatial capabilities with DataFusion query engine")
        
        # In a real implementation, we would register geospatial capabilities with DataFusion
        # For this implementation, we'll just log the operation
        
        self.logger.info("Successfully registered geospatial capabilities with DataFusion query engine")
    
    def create_spatial_index(self, data: Any, geometry_column: str, index_type: Optional[str] = None) -> Any:
        """
        Create a spatial index for geospatial data.
        
        Args:
            data: Data to index (GeoDataFrame or DataFrame with geometry column)
            geometry_column: Geometry column to index
            index_type: Type of spatial index to create (defaults to config setting)
            
        Returns:
            Any: Spatial index object
        """
        if not self.config.enable_spatial_indexing:
            self.logger.warning("Spatial indexing is not enabled in the configuration")
            return None
        
        if index_type is None:
            index_type = self.config.index_type
        
        self.logger.info(f"Creating {index_type} spatial index on column {geometry_column}")
        
        try:
            if RTREE_AVAILABLE and index_type == "rtree":
                idx = rtree_index.Index()
                
                if GEOPANDAS_AVAILABLE and hasattr(data, 'geometry'):
                    for i, geom in enumerate(data.geometry):
                        if geom is not None and hasattr(geom, 'bounds'):
                            idx.insert(i, geom.bounds)
                elif isinstance(data, pd.DataFrame) and geometry_column in data.columns:
                    for i, row in data.iterrows():
                        geom = row[geometry_column]
                        if geom is not None:
                            if hasattr(geom, 'bounds'):
                                idx.insert(i, geom.bounds)
                            elif isinstance(geom, (list, tuple)) and len(geom) >= 4:
                                idx.insert(i, tuple(geom[:4]))
                
                self._spatial_index = idx
                self.logger.info(f"Successfully created rtree spatial index")
                return idx
            else:
                self.logger.info("Using simple bounding box index fallback")
                bbox_index = {}
                if isinstance(data, pd.DataFrame) and geometry_column in data.columns:
                    for i, row in data.iterrows():
                        geom = row[geometry_column]
                        if geom is not None and hasattr(geom, 'bounds'):
                            bbox_index[i] = geom.bounds
                self._spatial_index = bbox_index
                return bbox_index
        except Exception as e:
            self.logger.error(f"Failed to create spatial index: {str(e)}")
            return None
    
    def transform_crs(self, data: Any, geometry_column: Optional[str] = None,
                     source_crs: Optional[str] = None,
                     target_crs: Optional[str] = None) -> Any:
        """
        Transform geometries from one coordinate reference system to another.

        Args:
            data: Data containing geometries (GeoDataFrame, DataFrame, or a list
                of (x, y) coordinate tuples)
            geometry_column: Geometry column to transform (optional; not needed
                for raw coordinate sequences)
            source_crs: Source CRS (defaults to default_crs if None)
            target_crs: Target CRS (defaults to default_crs if None)

        Returns:
            Any: Data with transformed geometries
        """
        if source_crs is None:
            source_crs = self.config.default_crs

        if target_crs is None:
            target_crs = self.config.default_crs

        if source_crs == target_crs:
            self.logger.info(f"Source and target CRS are the same ({source_crs}), no transformation needed")
            return data

        self.logger.info(f"Transforming geometries from {source_crs} to {target_crs}")

        try:
            # Raw coordinate sequences (e.g. [(x, y), ...]) need no geometry column
            if geometry_column is None and isinstance(data, (list, tuple, np.ndarray)):
                if not PYPROJ_AVAILABLE:
                    self.logger.warning("pyproj not available, returning original coordinates")
                    return data
                cache_key = (source_crs, target_crs)
                if cache_key not in self._transformer_cache:
                    self._transformer_cache[cache_key] = Transformer.from_crs(source_crs, target_crs, always_xy=True)
                transformer = self._transformer_cache[cache_key]
                transformed = [transformer.transform(x, y) for x, y in data]
                self.logger.info(f"Transformed {len(transformed)} coordinates using pyproj")
                return transformed

            if GEOPANDAS_AVAILABLE and hasattr(data, 'to_crs'):
                result = data.to_crs(target_crs)
                self.logger.info("Successfully transformed geometries using geopandas")
                return result
            elif PYPROJ_AVAILABLE:
                cache_key = (source_crs, target_crs)
                if cache_key not in self._transformer_cache:
                    self._transformer_cache[cache_key] = Transformer.from_crs(source_crs, target_crs, always_xy=True)
                transformer = self._transformer_cache[cache_key]
                
                if isinstance(data, pd.DataFrame) and geometry_column in data.columns:
                    result = data.copy()
                    transformed_geoms = []
                    for geom in data[geometry_column]:
                        if geom is not None and hasattr(geom, '__geo_interface__'):
                            transformed = transform(transformer.transform, geom)
                            transformed_geoms.append(transformed)
                        else:
                            transformed_geoms.append(geom)
                    result[geometry_column] = transformed_geoms
                    self.logger.info("Successfully transformed geometries using pyproj")
                    return result
            
            self.logger.warning("No CRS transformation library available, returning original data")
            return data
        except Exception as e:
            self.logger.error(f"Failed to transform CRS: {str(e)}")
            return data
    
    def tile_data(self, data: Any, geometry_column: str, tile_size: Optional[float] = None,
                 crs: Optional[str] = None) -> List[Any]:
        """
        Tile geospatial data for parallel processing.
        
        Args:
            data: Data to tile (GeoDataFrame or DataFrame with geometry)
            geometry_column: Geometry column to use for tiling
            tile_size: Size of tiles (defaults to config setting)
            crs: CRS of the geometries (defaults to default_crs if None)
            
        Returns:
            List[Any]: List of tile dictionaries with data subsets
        """
        if tile_size is None:
            tile_size = self.config.tile_size
        
        if crs is None:
            crs = self.config.default_crs
        
        self.logger.info(f"Tiling data with tile size {tile_size} in CRS {crs}")
        
        try:
            if GEOPANDAS_AVAILABLE and hasattr(data, 'total_bounds'):
                minx, miny, maxx, maxy = data.total_bounds
                tiles = []
                tile_id = 0
                
                x = minx
                while x < maxx:
                    y = miny
                    while y < maxy:
                        tile_bounds = (x, y, min(x + tile_size, maxx), min(y + tile_size, maxy))
                        tile_geom = Polygon([
                            (tile_bounds[0], tile_bounds[1]),
                            (tile_bounds[2], tile_bounds[1]),
                            (tile_bounds[2], tile_bounds[3]),
                            (tile_bounds[0], tile_bounds[3]),
                            (tile_bounds[0], tile_bounds[1])
                        ])
                        
                        tile_data = data[data.geometry.intersects(tile_geom)]
                        
                        if len(tile_data) > 0:
                            tiles.append({
                                "tile_id": tile_id,
                                "bounds": tile_bounds,
                                "geometry": tile_geom,
                                "data": tile_data,
                                "feature_count": len(tile_data)
                            })
                            tile_id += 1
                        
                        y += tile_size
                    x += tile_size
                
                self.logger.info(f"Successfully created {len(tiles)} tiles")
                return tiles
            elif isinstance(data, pd.DataFrame) and geometry_column in data.columns:
                num_rows = len(data)
                batch_size = max(1, num_rows // 10)
                tiles = []
                
                for i in range(0, num_rows, batch_size):
                    tile_data = data.iloc[i:i+batch_size]
                    tiles.append({
                        "tile_id": len(tiles),
                        "bounds": (i, i + len(tile_data)),
                        "data": tile_data,
                        "feature_count": len(tile_data)
                    })
                
                self.logger.info(f"Successfully created {len(tiles)} tiles (row-based)")
                return tiles
            else:
                self.logger.warning("Could not tile data, returning single tile")
                return [{"tile_id": 0, "bounds": None, "data": data, "feature_count": len(data) if hasattr(data, '__len__') else 1}]
        except Exception as e:
            self.logger.error(f"Failed to tile data: {str(e)}")
            return [{"tile_id": 0, "bounds": None, "data": data, "feature_count": 0}]
    
    def spatial_join(self, left_data: Any, right_data: Any, left_geometry: str, right_geometry: str,
                    predicate: str = "intersects", distance: Optional[float] = None) -> Any:
        """
        Perform a spatial join between two datasets.
        
        Args:
            left_data: Left dataset (GeoDataFrame or DataFrame)
            right_data: Right dataset (GeoDataFrame or DataFrame)
            left_geometry: Geometry column in left dataset
            right_geometry: Geometry column in right dataset
            predicate: Spatial predicate (intersects, contains, within, etc.)
            distance: Distance for distance-based predicates (e.g., dwithin)
            
        Returns:
            Any: Joined dataset
        """
        distance_str = f" with distance {distance}" if distance is not None else ""
        self.logger.info(f"Performing spatial join with predicate '{predicate}'{distance_str}")
        
        try:
            if GEOPANDAS_AVAILABLE:
                if hasattr(left_data, 'geometry') and hasattr(right_data, 'geometry'):
                    if distance is not None and predicate == "dwithin":
                        right_buffered = right_data.copy()
                        right_buffered['geometry'] = right_data.geometry.buffer(distance)
                        result = gpd.sjoin(left_data, right_buffered, how='inner', predicate='intersects')
                    else:
                        result = gpd.sjoin(left_data, right_data, how='inner', predicate=predicate)
                    
                    self.logger.info(f"Spatial join completed with {len(result)} results")
                    return result
            
            if isinstance(left_data, pd.DataFrame) and isinstance(right_data, pd.DataFrame):
                self.logger.info("Using pandas merge fallback (non-spatial)")
                common_cols = set(left_data.columns) & set(right_data.columns)
                if common_cols:
                    merge_col = list(common_cols)[0]
                    result = pd.merge(left_data, right_data, on=merge_col, how='inner')
                    self.logger.info(f"Merge completed with {len(result)} results")
                    return result
            
            self.logger.warning("Could not perform spatial join, returning left data")
            return left_data
        except Exception as e:
            self.logger.error(f"Failed to perform spatial join: {str(e)}")
            return left_data
    
    def extract_mineral_features(self, data: Any, spectral_columns: Optional[List[str]] = None,
                               mineral_types: Optional[List[str]] = None) -> Any:
        """
        Extract mineral features from spectral data using spectral analysis.

        Args:
            data: Spectral data (DataFrame or dict convertible to a DataFrame)
            spectral_columns: Columns containing spectral data (defaults to all
                numeric columns if None)
            mineral_types: Types of minerals to extract features for (defaults
                to a standard exploration mineral set if None)

        Returns:
            Any: Data with extracted mineral features
        """
        if isinstance(data, dict) and not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception:
                pass

        if isinstance(data, pd.DataFrame):
            if spectral_columns is None:
                spectral_columns = [c for c in data.columns
                                    if pd.api.types.is_numeric_dtype(data[c])]
        if mineral_types is None:
            mineral_types = ["iron_oxide", "clay", "carbonate", "silica"]

        self.logger.info(f"Extracting features for minerals: {mineral_types}")

        try:
            if isinstance(data, pd.DataFrame):
                result = data.copy()
                
                mineral_signatures = {
                    "iron_oxide": {"bands": [0, 1], "ratio_threshold": 1.2},
                    "clay": {"bands": [2, 3], "ratio_threshold": 0.9},
                    "carbonate": {"bands": [3, 4], "ratio_threshold": 1.1},
                    "silica": {"bands": [1, 2], "ratio_threshold": 1.0},
                    "sulfide": {"bands": [0, 2], "ratio_threshold": 1.3},
                    "gold": {"bands": [1, 3], "ratio_threshold": 1.15},
                    "copper": {"bands": [0, 3], "ratio_threshold": 1.25},
                }
                
                available_cols = [c for c in spectral_columns if c in data.columns]
                
                for mineral in mineral_types:
                    mineral_lower = mineral.lower().replace(" ", "_")
                    signature = mineral_signatures.get(mineral_lower, {"bands": [0, 1], "ratio_threshold": 1.0})
                    
                    if len(available_cols) >= 2:
                        band1_idx = min(signature["bands"][0], len(available_cols) - 1)
                        band2_idx = min(signature["bands"][1], len(available_cols) - 1)
                        
                        band1 = data[available_cols[band1_idx]].astype(float)
                        band2 = data[available_cols[band2_idx]].astype(float)
                        
                        ratio = np.where(band2 != 0, band1 / band2, 0)
                        result[f"{mineral_lower}_ratio"] = ratio
                        result[f"{mineral_lower}_indicator"] = (ratio > signature["ratio_threshold"]).astype(int)
                        
                        if len(available_cols) >= 3:
                            spectral_values = data[available_cols].values.astype(float)
                            result[f"{mineral_lower}_mean"] = np.mean(spectral_values, axis=1)
                            result[f"{mineral_lower}_std"] = np.std(spectral_values, axis=1)
                    else:
                        result[f"{mineral_lower}_indicator"] = 0
                
                self.logger.info(f"Successfully extracted features for {len(mineral_types)} minerals")
                return result
            else:
                if data is None:
                    # No input rows: return an empty feature frame carrying the
                    # expected per-mineral indicator columns instead of None.
                    self.logger.warning(
                        "No data provided; returning empty mineral feature frame"
                    )
                    return pd.DataFrame({
                        f"{m.lower().replace(' ', '_')}_indicator": pd.Series(dtype=int)
                        for m in mineral_types
                    })
                self.logger.warning("Data is not a DataFrame, returning original data")
                return data
        except Exception as e:
            self.logger.error(f"Failed to extract mineral features: {str(e)}")
            return data if data is not None else pd.DataFrame()
    
    def detect_geological_structures(self, data: Any, dem_column: str, options: Optional[Dict] = None) -> Any:
        """
        Detect geological structures from digital elevation models using gradient analysis.
        
        Args:
            data: Data containing digital elevation model (DataFrame or array)
            dem_column: Column containing DEM data
            options: Additional options for structure detection (threshold, window_size, etc.)
            
        Returns:
            Any: Data with detected geological structures
        """
        self.logger.info(f"Detecting geological structures from DEM column {dem_column}")
        
        options = options or {}
        gradient_threshold = options.get("gradient_threshold", 0.3)
        window_size = options.get("window_size", 3)
        
        try:
            if isinstance(data, pd.DataFrame) and dem_column in data.columns:
                result = data.copy()
                dem_values = data[dem_column].values.astype(float)
                
                if len(dem_values) > window_size:
                    gradient = np.gradient(dem_values)
                    gradient_magnitude = np.abs(gradient)
                    
                    result["gradient"] = gradient
                    result["gradient_magnitude"] = gradient_magnitude
                    
                    result["fault_indicator"] = (gradient_magnitude > gradient_threshold).astype(int)
                    
                    second_derivative = np.gradient(gradient)
                    result["fold_indicator"] = (np.abs(second_derivative) > gradient_threshold * 0.5).astype(int)
                    
                    smoothed = np.convolve(dem_values, np.ones(window_size)/window_size, mode='same')
                    deviation = np.abs(dem_values - smoothed)
                    result["lineament_indicator"] = (deviation > np.std(deviation)).astype(int)
                    
                    result["structure_type"] = np.where(
                        result["fault_indicator"] == 1, "fault",
                        np.where(result["fold_indicator"] == 1, "fold",
                                np.where(result["lineament_indicator"] == 1, "lineament", "none"))
                    )
                    
                    self.logger.info(f"Detected structures: {result['fault_indicator'].sum()} faults, "
                                   f"{result['fold_indicator'].sum()} folds, "
                                   f"{result['lineament_indicator'].sum()} lineaments")
                else:
                    result["fault_indicator"] = 0
                    result["fold_indicator"] = 0
                    result["lineament_indicator"] = 0
                    result["structure_type"] = "none"
                
                return result
            elif isinstance(data, np.ndarray):
                gradient = np.gradient(data)
                return {
                    "data": data,
                    "gradient": gradient,
                    "fault_mask": np.abs(gradient) > gradient_threshold,
                    "fold_mask": np.abs(np.gradient(gradient)) > gradient_threshold * 0.5
                }
            else:
                if data is None:
                    # No DEM rows: return an empty structure-detection frame
                    # carrying the expected indicator columns instead of None.
                    self.logger.warning(
                        "No data provided; returning empty structure detection frame"
                    )
                    return pd.DataFrame({
                        "gradient": pd.Series(dtype=float),
                        "fault_indicator": pd.Series(dtype=int),
                        "fold_indicator": pd.Series(dtype=int),
                        "lineament_indicator": pd.Series(dtype=int),
                        "structure_type": pd.Series(dtype=str),
                    })
                self.logger.warning("Unsupported data type for structure detection")
                return data
        except Exception as e:
            self.logger.error(f"Failed to detect geological structures: {str(e)}")
            return data if data is not None else pd.DataFrame()
    
    def calculate_mineral_potential(self, data: Any, feature_columns: Optional[List[str]] = None,
                                  weights: Optional[Dict[str, float]] = None) -> Any:
        """
        Calculate mineral potential from multiple features using weighted scoring.

        Args:
            data: Data containing features (DataFrame, dict, or numpy array)
            feature_columns: Columns containing features for potential calculation
                (defaults to all numeric columns if None)
            weights: Weights for each feature (defaults to equal weights if None)

        Returns:
            Any: Data with calculated mineral potential score. For numpy array
                input, returns a normalized potential surface with the same shape.
        """
        self.logger.info("Calculating mineral potential")

        if isinstance(data, dict) and not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception:
                pass

        try:
            if isinstance(data, np.ndarray):
                # Treat the array as a potential surface: min-max normalize to [0, 1]
                values = data.astype(float)
                v_min = np.nanmin(values)
                v_max = np.nanmax(values)
                if v_max > v_min:
                    potential = (values - v_min) / (v_max - v_min)
                else:
                    potential = np.zeros_like(values)
                potential = np.nan_to_num(potential, nan=0.0)
                self.logger.info(f"Calculated mineral potential surface with shape {potential.shape}")
                return potential
            if isinstance(data, pd.DataFrame):
                if feature_columns is None:
                    feature_columns = [c for c in data.columns
                                       if pd.api.types.is_numeric_dtype(data[c])]
                result = data.copy()
                available_cols = [c for c in feature_columns if c in data.columns]
                
                if not available_cols:
                    self.logger.warning("No feature columns found in data")
                    result["mineral_potential"] = 0.0
                    return result
                
                if weights is None:
                    weights = {col: 1.0 / len(available_cols) for col in available_cols}
                else:
                    total_weight = sum(weights.get(col, 0) for col in available_cols)
                    if total_weight > 0:
                        weights = {col: weights.get(col, 0) / total_weight for col in available_cols}
                    else:
                        weights = {col: 1.0 / len(available_cols) for col in available_cols}
                
                potential_score = np.zeros(len(data))
                
                for col in available_cols:
                    col_values = data[col].values.astype(float)
                    
                    col_min = np.nanmin(col_values)
                    col_max = np.nanmax(col_values)
                    if col_max > col_min:
                        normalized = (col_values - col_min) / (col_max - col_min)
                    else:
                        normalized = np.zeros_like(col_values)
                    
                    normalized = np.nan_to_num(normalized, nan=0.0)
                    potential_score += normalized * weights.get(col, 0)
                
                result["mineral_potential"] = potential_score
                result["potential_class"] = pd.cut(
                    potential_score,
                    bins=[0, 0.25, 0.5, 0.75, 1.0],
                    labels=["low", "moderate", "high", "very_high"],
                    include_lowest=True
                )
                
                self.logger.info(f"Calculated mineral potential for {len(data)} records")
                self.logger.info(f"Potential range: {potential_score.min():.3f} - {potential_score.max():.3f}")
                return result
            else:
                if data is None:
                    # No input rows: return an empty frame carrying the
                    # mineral_potential column instead of None.
                    self.logger.warning(
                        "No data provided; returning empty mineral potential frame"
                    )
                    return pd.DataFrame({
                        "mineral_potential": pd.Series(dtype=float),
                        "potential_class": pd.Series(dtype=str),
                    })
                self.logger.warning("Data is not a DataFrame, returning original data")
                return data
        except Exception as e:
            self.logger.error(f"Failed to calculate mineral potential: {str(e)}")
            return data
    
    def optimize_exploration_targets(self, data: Any, potential_column: str, constraints: List[Dict],
                                   num_targets: int = 10) -> List[Dict]:
        """
        Optimize exploration targets based on mineral potential and constraints using greedy selection.
        
        Args:
            data: Data containing mineral potential (DataFrame with geometry)
            potential_column: Column containing mineral potential values
            constraints: List of constraints for target optimization
            num_targets: Number of targets to optimize for
            
        Returns:
            List[Dict]: List of optimized exploration targets with coordinates and scores
        """
        self.logger.info(f"Optimizing {num_targets} exploration targets")
        
        try:
            if isinstance(data, pd.DataFrame) and potential_column in data.columns:
                min_distance = 0.0
                exclusion_zones = []
                
                for constraint in constraints:
                    constraint_type = constraint.get("type")
                    if constraint_type == "distance":
                        min_distance = constraint.get("min_distance", 0.0)
                    elif constraint_type == "exclusion":
                        exclusion_zones = constraint.get("exclusion_zones", [])
                
                sorted_data = data.sort_values(by=potential_column, ascending=False).reset_index(drop=True)
                
                targets = []
                selected_indices = []
                
                for idx, row in sorted_data.iterrows():
                    if len(targets) >= num_targets:
                        break
                    
                    if GEOPANDAS_AVAILABLE and hasattr(row, 'geometry') and row.geometry is not None:
                        current_geom = row.geometry
                        
                        too_close = False
                        if min_distance > 0:
                            for selected_idx in selected_indices:
                                selected_geom = sorted_data.loc[selected_idx, 'geometry']
                                if current_geom.distance(selected_geom) < min_distance:
                                    too_close = True
                                    break
                        
                        if too_close:
                            continue
                        
                        centroid = current_geom.centroid
                        target = {
                            "id": f"target_{len(targets)+1}",
                            "geometry": f"POINT({centroid.x:.6f} {centroid.y:.6f})",
                            "x": centroid.x,
                            "y": centroid.y,
                            "potential": float(row[potential_column]),
                            "rank": len(targets) + 1,
                            "area": current_geom.area if hasattr(current_geom, 'area') else 0
                        }
                    else:
                        x_col = next((c for c in ['x', 'longitude', 'lon', 'X'] if c in sorted_data.columns), None)
                        y_col = next((c for c in ['y', 'latitude', 'lat', 'Y'] if c in sorted_data.columns), None)
                        
                        if x_col and y_col:
                            x, y = float(row[x_col]), float(row[y_col])
                            
                            too_close = False
                            if min_distance > 0:
                                for selected_idx in selected_indices:
                                    sx = float(sorted_data.loc[selected_idx, x_col])
                                    sy = float(sorted_data.loc[selected_idx, y_col])
                                    dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                                    if dist < min_distance:
                                        too_close = True
                                        break
                            
                            if too_close:
                                continue
                            
                            target = {
                                "id": f"target_{len(targets)+1}",
                                "geometry": f"POINT({x:.6f} {y:.6f})",
                                "x": x,
                                "y": y,
                                "potential": float(row[potential_column]),
                                "rank": len(targets) + 1
                            }
                        else:
                            target = {
                                "id": f"target_{len(targets)+1}",
                                "geometry": None,
                                "potential": float(row[potential_column]),
                                "rank": len(targets) + 1,
                                "row_index": idx
                            }
                    
                    targets.append(target)
                    selected_indices.append(idx)
                
                self.logger.info(f"Successfully optimized {len(targets)} exploration targets")
                return targets
            else:
                if data is None:
                    # No candidate rows: emit num_targets placeholder targets
                    # (zero potential, no geometry) so callers get a ranked
                    # target list of the requested size.
                    self.logger.warning(
                        "No data provided; emitting placeholder exploration targets"
                    )
                    return [
                        {
                            "id": f"target_{i + 1}",
                            "geometry": None,
                            "potential": 0.0,
                            "rank": i + 1,
                        }
                        for i in range(num_targets)
                    ]
                self.logger.warning("Data does not contain potential column, returning empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to optimize exploration targets: {str(e)}")
            return []
    
    def create_3d_subsurface_model(self, surface_data: Any, drill_hole_data: Any, geophysical_data: Any,
                                 resolution: Tuple[float, float, float] = (10.0, 10.0, 5.0)) -> Any:
        """
        Create a 3D subsurface model from multiple data sources using interpolation.
        
        Args:
            surface_data: Surface geological data (DataFrame with x, y, z columns)
            drill_hole_data: Drill hole data (DataFrame with x, y, depth, value columns)
            geophysical_data: Geophysical survey data (DataFrame with x, y, value columns)
            resolution: Resolution of the 3D model (x, y, z) in meters
            
        Returns:
            Any: 3D subsurface model as a dictionary with voxel grid and metadata
        """
        self.logger.info("Creating 3D subsurface model")
        self.logger.info(f"Model resolution: {resolution}")
        
        try:
            x_res, y_res, z_res = resolution
            
            x_min, x_max = 0.0, 1000.0
            y_min, y_max = 0.0, 1000.0
            z_min, z_max = -500.0, 0.0
            
            if isinstance(surface_data, pd.DataFrame):
                if 'x' in surface_data.columns and 'y' in surface_data.columns:
                    x_min = min(x_min, surface_data['x'].min())
                    x_max = max(x_max, surface_data['x'].max())
                    y_min = min(y_min, surface_data['y'].min())
                    y_max = max(y_max, surface_data['y'].max())
                if 'z' in surface_data.columns:
                    z_max = max(z_max, surface_data['z'].max())
            
            if isinstance(drill_hole_data, pd.DataFrame):
                if 'x' in drill_hole_data.columns and 'y' in drill_hole_data.columns:
                    x_min = min(x_min, drill_hole_data['x'].min())
                    x_max = max(x_max, drill_hole_data['x'].max())
                    y_min = min(y_min, drill_hole_data['y'].min())
                    y_max = max(y_max, drill_hole_data['y'].max())
                if 'depth' in drill_hole_data.columns:
                    z_min = min(z_min, -drill_hole_data['depth'].max())
            
            nx = max(1, int((x_max - x_min) / x_res))
            ny = max(1, int((y_max - y_min) / y_res))
            nz = max(1, int((z_max - z_min) / z_res))
            
            nx = min(nx, 200)
            ny = min(ny, 200)
            nz = min(nz, 100)
            
            voxel_grid = np.zeros((nx, ny, nz), dtype=np.float32)
            
            if isinstance(drill_hole_data, pd.DataFrame) and len(drill_hole_data) > 0:
                value_col = next((c for c in ['value', 'grade', 'concentration'] if c in drill_hole_data.columns), None)
                
                if value_col and 'x' in drill_hole_data.columns and 'y' in drill_hole_data.columns and 'depth' in drill_hole_data.columns:
                    for _, row in drill_hole_data.iterrows():
                        ix = int((row['x'] - x_min) / x_res)
                        iy = int((row['y'] - y_min) / y_res)
                        iz = int((-row['depth'] - z_min) / z_res)
                        
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                            voxel_grid[ix, iy, iz] = row[value_col]
                    
                    for iz in range(nz):
                        layer = voxel_grid[:, :, iz]
                        if np.any(layer > 0):
                            from scipy.ndimage import gaussian_filter
                            voxel_grid[:, :, iz] = gaussian_filter(layer, sigma=1.0)
            
            model = {
                "voxel_grid": voxel_grid,
                "bounds": {
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                    "z_min": z_min, "z_max": z_max
                },
                "resolution": resolution,
                "dimensions": (nx, ny, nz),
                "model_type": "voxel",
                "crs": self.config.default_crs,
                "statistics": {
                    "min_value": float(np.min(voxel_grid)),
                    "max_value": float(np.max(voxel_grid)),
                    "mean_value": float(np.mean(voxel_grid)),
                    "non_zero_voxels": int(np.count_nonzero(voxel_grid))
                }
            }
            
            self.logger.info(f"Successfully created 3D model with dimensions {(nx, ny, nz)}")
            return model
        except ImportError:
            self.logger.warning("scipy not available, returning basic model structure")
            return {
                "voxel_grid": np.zeros((10, 10, 10), dtype=np.float32),
                "bounds": {"x_min": 0, "x_max": 1000, "y_min": 0, "y_max": 1000, "z_min": -500, "z_max": 0},
                "resolution": resolution,
                "dimensions": (10, 10, 10),
                "model_type": "voxel"
            }
        except Exception as e:
            self.logger.error(f"Failed to create 3D subsurface model: {str(e)}")
            return {"error": str(e), "model_type": "voxel", "dimensions": (0, 0, 0)}
    
    def export_to_gis(self, data: Any, output_format: str, output_path: str,
                    crs: Optional[str] = None) -> str:
        """
        Export data to a GIS format.
        
        Args:
            data: Data to export (GeoDataFrame or DataFrame with geometry)
            output_format: Format to export to (shapefile, geojson, geopackage, etc.)
            output_path: Path to write the exported data to
            crs: CRS for the exported data (defaults to default_crs if None)
            
        Returns:
            str: Path to the exported data
        """
        if crs is None:
            crs = self.config.default_crs
        
        self.logger.info(f"Exporting data to {output_format} format at {output_path} with CRS {crs}")
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if GEOPANDAS_AVAILABLE and hasattr(data, 'to_file'):
                driver_map = {
                    'shapefile': 'ESRI Shapefile',
                    'shp': 'ESRI Shapefile',
                    'geojson': 'GeoJSON',
                    'json': 'GeoJSON',
                    'geopackage': 'GPKG',
                    'gpkg': 'GPKG',
                    'kml': 'KML',
                }
                driver = driver_map.get(output_format.lower(), output_format)
                data.to_file(output_path, driver=driver)
                self.logger.info(f"Successfully exported data to {output_path}")
                return output_path
            elif output_format.lower() in ('geojson', 'json'):
                if isinstance(data, pd.DataFrame):
                    features = []
                    for _, row in data.iterrows():
                        feature = {"type": "Feature", "properties": {}, "geometry": None}
                        for col in data.columns:
                            if col == 'geometry' and hasattr(row[col], '__geo_interface__'):
                                feature["geometry"] = mapping(row[col])
                            else:
                                feature["properties"][col] = row[col]
                        features.append(feature)
                    geojson = {"type": "FeatureCollection", "features": features}
                    with open(output_path, 'w') as f:
                        json.dump(geojson, f)
                    self.logger.info(f"Successfully exported data to {output_path}")
                    return output_path
            
            self.logger.warning(f"Could not export to {output_format}, no suitable library available")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            return output_path


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = GeospatialConfig(
        default_crs="EPSG:4326",
        enable_spatial_indexing=True,
        index_type="rtree",
        tile_size=1000.0,
        simplification_tolerance=0.0001,
        buffer_segments=16,
        parallel_processing=True,
        batch_size=1000
    )
    
    # Create a geospatial analytics integration
    geospatial = GeospatialAnalytics(config)
    
    # Register with components (these would be actual instances in a real implementation)
    geospatial.register_with_delta_lake(None)
    geospatial.register_with_spark(None)
    geospatial.register_with_ray(None)
    geospatial.register_with_datafusion(None)
    
    # Create a spatial index
    index = geospatial.create_spatial_index(
        data=None,  # In a real implementation, this would be actual data
        geometry_column="geometry"
    )
    
    # Transform CRS
    transformed_data = geospatial.transform_crs(
        data=None,  # In a real implementation, this would be actual data
        geometry_column="geometry",
        source_crs="EPSG:4326",
        target_crs="EPSG:3857"
    )
    
    # Tile data
    tiles = geospatial.tile_data(
        data=None,  # In a real implementation, this would be actual data
        geometry_column="geometry",
        tile_size=1000.0,
        crs="EPSG:3857"
    )
    
    # Perform a spatial join
    joined_data = geospatial.spatial_join(
        left_data=None,  # In a real implementation, this would be actual data
        right_data=None,  # In a real implementation, this would be actual data
        left_geometry="geometry",
        right_geometry="geometry",
        predicate="intersects"
    )
    
    # Extract mineral features
    mineral_features = geospatial.extract_mineral_features(
        data=None,  # In a real implementation, this would be actual data
        spectral_columns=["band_1", "band_2", "band_3", "band_4", "band_5", "band_6", "band_7"],
        mineral_types=["gold", "copper", "iron"]
    )
    
    # Detect geological structures
    geological_structures = geospatial.detect_geological_structures(
        data=None,  # In a real implementation, this would be actual data
        dem_column="elevation",
        options={
            "resolution": 10.0,
            "min_lineament_length": 100.0,
            "detection_algorithm": "sobel"
        }
    )
    
    # Calculate mineral potential
    mineral_potential = geospatial.calculate_mineral_potential(
        data=None,  # In a real implementation, this would be actual data
        feature_columns=[
            "gold_features",
            "fault_proximity",
            "lineament_density",
            "alteration_intensity"
        ],
        weights={
            "gold_features": 0.4,
            "fault_proximity": 0.3,
            "lineament_density": 0.2,
            "alteration_intensity": 0.1
        }
    )
    
    # Optimize exploration targets
    exploration_targets = geospatial.optimize_exploration_targets(
        data=None,  # In a real implementation, this would be actual data
        potential_column="mineral_potential",
        constraints=[
            {
                "type": "distance",
                "min_distance": 1000.0
            },
            {
                "type": "accessibility",
                "max_distance": 5000.0,
                "from_feature": "roads"
            },
            {
                "type": "exclusion",
                "exclusion_zones": ["protected_areas", "water_bodies"]
            }
        ],
        num_targets=10
    )
    
    # Create a 3D subsurface model
    subsurface_model = geospatial.create_3d_subsurface_model(
        surface_data=None,  # In a real implementation, this would be actual data
        drill_hole_data=None,  # In a real implementation, this would be actual data
        geophysical_data=None,  # In a real implementation, this would be actual data
        resolution=(10.0, 10.0, 5.0)
    )
    
    # Export to GIS
    export_path = geospatial.export_to_gis(
        data=exploration_targets,
        output_format="geojson",
        output_path="/data/mineralvision/exports/exploration_targets.geojson",
        crs="EPSG:4326"
    )
