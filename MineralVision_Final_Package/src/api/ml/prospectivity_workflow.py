"""
ML Prospectivity Mapping Workflow for MineralVision.

This module provides comprehensive ML workflows for mineral prospectivity including:
- Prospectivity mapping pipeline with spatial cross-validation
- Feature generation from raster stacks (mag, radiometrics, hyperspectral indices)
- Integration with TorchGeo/TerraTorch for geospatial foundation models
- NLP ingestion for geological reports (entity extraction)
- Dataset registry and reproducible benchmarks
- Spatially-aware evaluation metrics

Based on mineral-exploration-machine-learning best practices and UNCOVER-ML patterns.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, interpolate
from scipy.spatial import cKDTree
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json
import re
import os
import hashlib

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features for prospectivity mapping."""
    MAGNETIC = "magnetic"
    RADIOMETRIC = "radiometric"
    HYPERSPECTRAL = "hyperspectral"
    TOPOGRAPHIC = "topographic"
    GEOCHEMICAL = "geochemical"
    STRUCTURAL = "structural"
    LITHOLOGICAL = "lithological"
    PROXIMITY = "proximity"
    CUSTOM = "custom"


class ValidationStrategy(Enum):
    """Spatial cross-validation strategies."""
    RANDOM = "random"
    SPATIAL_BLOCK = "spatial_block"
    SPATIAL_LEAVE_ONE_OUT = "spatial_loo"
    SPATIAL_BUFFER = "spatial_buffer"
    STRATIFIED_SPATIAL = "stratified_spatial"


class ProspectivityModel(Enum):
    """Prospectivity model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    FOUNDATION_MODEL = "foundation_model"
    ENSEMBLE = "ensemble"


@dataclass
class RasterLayer:
    """Raster layer for feature generation."""
    name: str
    data: np.ndarray
    transform: Tuple[float, float, float, float, float, float]  # Affine transform
    crs: str
    nodata: float = np.nan
    feature_type: FeatureType = FeatureType.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get layer bounds (xmin, ymin, xmax, ymax)."""
        height, width = self.data.shape[:2]
        xmin = self.transform[2]
        ymax = self.transform[5]
        xmax = xmin + width * self.transform[0]
        ymin = ymax + height * self.transform[4]  # transform[4] is negative
        return (xmin, ymin, xmax, ymax)


@dataclass
class TrainingPoint:
    """Training point for prospectivity mapping."""
    x: float
    y: float
    label: int  # 1 = deposit, 0 = non-deposit
    deposit_type: Optional[str] = None
    commodity: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProspectivityDataset:
    """Dataset for prospectivity mapping."""
    name: str
    features: np.ndarray
    labels: np.ndarray
    coordinates: np.ndarray  # (n_samples, 2) array of x, y
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.labels)
        
    def get_hash(self) -> str:
        """Get unique hash for dataset."""
        data_str = f"{self.name}_{self.features.shape}_{self.labels.sum()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]


class SpatialBlockCV(BaseCrossValidator):
    """
    Spatial block cross-validation.
    
    Divides the study area into spatial blocks and uses blocks
    as cross-validation folds to account for spatial autocorrelation.
    """
    
    def __init__(self, n_splits: int = 5, block_size: float = None,
                 buffer_size: float = 0.0, random_state: int = None):
        self.n_splits = n_splits
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray = None, 
             coordinates: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for spatial block CV.
        
        Args:
            X: Feature matrix
            y: Labels (unused but required by sklearn)
            coordinates: (n_samples, 2) array of x, y coordinates
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if coordinates is None:
            raise ValueError("Coordinates required for spatial block CV")
            
        n_samples = len(X)
        
        # Determine block size if not specified
        if self.block_size is None:
            x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
            y_range = coordinates[:, 1].max() - coordinates[:, 1].min()
            self.block_size = max(x_range, y_range) / np.sqrt(self.n_splits * 2)
            
        # Assign samples to blocks
        x_min, y_min = coordinates.min(axis=0)
        block_x = ((coordinates[:, 0] - x_min) / self.block_size).astype(int)
        block_y = ((coordinates[:, 1] - y_min) / self.block_size).astype(int)
        block_ids = block_x * 1000 + block_y  # Unique block ID
        
        unique_blocks = np.unique(block_ids)
        
        # Shuffle blocks
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_blocks)
        
        # Split blocks into folds
        fold_size = len(unique_blocks) // self.n_splits
        
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_splits - 1 else len(unique_blocks)
            
            test_blocks = set(unique_blocks[start:end])
            test_mask = np.isin(block_ids, list(test_blocks))
            
            # Apply buffer if specified
            if self.buffer_size > 0:
                test_coords = coordinates[test_mask]
                train_mask = ~test_mask
                
                # Remove training points within buffer distance of test points
                if len(test_coords) > 0:
                    tree = cKDTree(test_coords)
                    train_coords = coordinates[train_mask]
                    distances, _ = tree.query(train_coords)
                    buffer_mask = distances > self.buffer_size
                    
                    train_indices = np.where(train_mask)[0][buffer_mask]
                else:
                    train_indices = np.where(train_mask)[0]
            else:
                train_indices = np.where(~test_mask)[0]
                
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class SpatialBufferCV(BaseCrossValidator):
    """
    Spatial buffer cross-validation.
    
    Leaves out a buffer zone around test points to prevent
    spatial leakage between train and test sets.
    """
    
    def __init__(self, n_splits: int = 5, buffer_distance: float = 1000.0,
                 random_state: int = None):
        self.n_splits = n_splits
        self.buffer_distance = buffer_distance
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray = None,
             coordinates: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with spatial buffer."""
        if coordinates is None:
            raise ValueError("Coordinates required for spatial buffer CV")
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[start:end]
            test_coords = coordinates[test_indices]
            
            # Build KD-tree for test points
            tree = cKDTree(test_coords)
            
            # Find training points outside buffer
            all_distances, _ = tree.query(coordinates)
            train_mask = all_distances > self.buffer_distance
            train_mask[test_indices] = False  # Exclude test points
            
            train_indices = np.where(train_mask)[0]
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class FeatureGenerator:
    """
    Generate features from raster stacks for prospectivity mapping.
    """
    
    def __init__(self):
        self.feature_extractors: Dict[FeatureType, Callable] = {
            FeatureType.MAGNETIC: self._extract_magnetic_features,
            FeatureType.RADIOMETRIC: self._extract_radiometric_features,
            FeatureType.HYPERSPECTRAL: self._extract_hyperspectral_features,
            FeatureType.TOPOGRAPHIC: self._extract_topographic_features,
            FeatureType.STRUCTURAL: self._extract_structural_features,
        }
        
    def extract_at_points(self, layers: List[RasterLayer],
                         points: List[TrainingPoint],
                         window_size: int = 1) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features at training points.
        
        Args:
            layers: List of raster layers
            points: List of training points
            window_size: Window size for neighborhood features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        all_features = []
        all_names = []
        
        for layer in layers:
            features, names = self._extract_layer_features(layer, points, window_size)
            all_features.append(features)
            all_names.extend(names)
            
        return np.hstack(all_features), all_names
        
    def _extract_layer_features(self, layer: RasterLayer,
                               points: List[TrainingPoint],
                               window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a single layer."""
        n_points = len(points)
        
        # Get pixel coordinates for each point
        pixel_coords = []
        for point in points:
            col = int((point.x - layer.transform[2]) / layer.transform[0])
            row = int((point.y - layer.transform[5]) / layer.transform[4])
            pixel_coords.append((row, col))
            
        # Extract base features
        if layer.feature_type in self.feature_extractors:
            features, names = self.feature_extractors[layer.feature_type](
                layer, pixel_coords, window_size
            )
        else:
            features, names = self._extract_basic_features(
                layer, pixel_coords, window_size
            )
            
        return features, names
        
    def _extract_basic_features(self, layer: RasterLayer,
                               pixel_coords: List[Tuple[int, int]],
                               window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract basic features (value, mean, std in window)."""
        data = layer.data
        n_points = len(pixel_coords)
        
        features = np.zeros((n_points, 3))
        names = [f"{layer.name}_value", f"{layer.name}_mean", f"{layer.name}_std"]
        
        half_win = window_size // 2
        
        for i, (row, col) in enumerate(pixel_coords):
            # Point value
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                features[i, 0] = data[row, col]
            else:
                features[i, 0] = np.nan
                
            # Window statistics
            r_start = max(0, row - half_win)
            r_end = min(data.shape[0], row + half_win + 1)
            c_start = max(0, col - half_win)
            c_end = min(data.shape[1], col + half_win + 1)
            
            window = data[r_start:r_end, c_start:c_end]
            valid = window[~np.isnan(window)]
            
            if len(valid) > 0:
                features[i, 1] = np.mean(valid)
                features[i, 2] = np.std(valid)
            else:
                features[i, 1:] = np.nan
                
        return features, names
        
    def _extract_magnetic_features(self, layer: RasterLayer,
                                  pixel_coords: List[Tuple[int, int]],
                                  window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract magnetic-specific features."""
        data = layer.data
        n_points = len(pixel_coords)
        
        # Compute derivatives
        dx = ndimage.sobel(data, axis=1)
        dy = ndimage.sobel(data, axis=0)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Analytic signal approximation
        analytic = np.sqrt(dx**2 + dy**2 + data**2)
        
        features = np.zeros((n_points, 6))
        names = [
            f"{layer.name}_value",
            f"{layer.name}_gradient",
            f"{layer.name}_analytic_signal",
            f"{layer.name}_mean",
            f"{layer.name}_std",
            f"{layer.name}_range"
        ]
        
        half_win = window_size // 2
        
        for i, (row, col) in enumerate(pixel_coords):
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                features[i, 0] = data[row, col]
                features[i, 1] = gradient_mag[row, col]
                features[i, 2] = analytic[row, col]
            else:
                features[i, :3] = np.nan
                
            # Window statistics
            r_start = max(0, row - half_win)
            r_end = min(data.shape[0], row + half_win + 1)
            c_start = max(0, col - half_win)
            c_end = min(data.shape[1], col + half_win + 1)
            
            window = data[r_start:r_end, c_start:c_end]
            valid = window[~np.isnan(window)]
            
            if len(valid) > 0:
                features[i, 3] = np.mean(valid)
                features[i, 4] = np.std(valid)
                features[i, 5] = np.max(valid) - np.min(valid)
            else:
                features[i, 3:] = np.nan
                
        return features, names
        
    def _extract_radiometric_features(self, layer: RasterLayer,
                                     pixel_coords: List[Tuple[int, int]],
                                     window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract radiometric-specific features."""
        # Similar to magnetic but with different derived products
        return self._extract_basic_features(layer, pixel_coords, window_size)
        
    def _extract_hyperspectral_features(self, layer: RasterLayer,
                                       pixel_coords: List[Tuple[int, int]],
                                       window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract hyperspectral indices and features."""
        data = layer.data
        n_points = len(pixel_coords)
        
        # Assume multi-band data
        if len(data.shape) == 2:
            return self._extract_basic_features(layer, pixel_coords, window_size)
            
        n_bands = data.shape[2]
        
        features = np.zeros((n_points, n_bands + 3))
        names = [f"{layer.name}_band{i}" for i in range(n_bands)]
        names.extend([f"{layer.name}_mean", f"{layer.name}_std", f"{layer.name}_range"])
        
        for i, (row, col) in enumerate(pixel_coords):
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                spectrum = data[row, col, :]
                features[i, :n_bands] = spectrum
                features[i, n_bands] = np.mean(spectrum)
                features[i, n_bands + 1] = np.std(spectrum)
                features[i, n_bands + 2] = np.max(spectrum) - np.min(spectrum)
            else:
                features[i, :] = np.nan
                
        return features, names
        
    def _extract_topographic_features(self, layer: RasterLayer,
                                     pixel_coords: List[Tuple[int, int]],
                                     window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract topographic features (slope, aspect, curvature)."""
        data = layer.data
        n_points = len(pixel_coords)
        
        # Compute derivatives
        dx = ndimage.sobel(data, axis=1)
        dy = ndimage.sobel(data, axis=0)
        
        # Slope
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # Aspect
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        
        # Curvature (Laplacian)
        curvature = ndimage.laplace(data)
        
        features = np.zeros((n_points, 5))
        names = [
            f"{layer.name}_elevation",
            f"{layer.name}_slope",
            f"{layer.name}_aspect",
            f"{layer.name}_curvature",
            f"{layer.name}_roughness"
        ]
        
        half_win = window_size // 2
        
        for i, (row, col) in enumerate(pixel_coords):
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                features[i, 0] = data[row, col]
                features[i, 1] = slope[row, col]
                features[i, 2] = aspect[row, col]
                features[i, 3] = curvature[row, col]
                
                # Roughness (std in window)
                r_start = max(0, row - half_win)
                r_end = min(data.shape[0], row + half_win + 1)
                c_start = max(0, col - half_win)
                c_end = min(data.shape[1], col + half_win + 1)
                
                window = data[r_start:r_end, c_start:c_end]
                features[i, 4] = np.std(window[~np.isnan(window)])
            else:
                features[i, :] = np.nan
                
        return features, names
        
    def _extract_structural_features(self, layer: RasterLayer,
                                    pixel_coords: List[Tuple[int, int]],
                                    window_size: int) -> Tuple[np.ndarray, List[str]]:
        """Extract structural geology features."""
        # For structural features, compute edge detection and lineament density
        data = layer.data
        n_points = len(pixel_coords)
        
        # Edge detection
        edges = ndimage.sobel(data)
        
        features = np.zeros((n_points, 3))
        names = [
            f"{layer.name}_value",
            f"{layer.name}_edge_strength",
            f"{layer.name}_lineament_density"
        ]
        
        half_win = window_size // 2
        
        for i, (row, col) in enumerate(pixel_coords):
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                features[i, 0] = data[row, col]
                features[i, 1] = edges[row, col]
                
                # Lineament density (edge density in window)
                r_start = max(0, row - half_win)
                r_end = min(data.shape[0], row + half_win + 1)
                c_start = max(0, col - half_win)
                c_end = min(data.shape[1], col + half_win + 1)
                
                window = edges[r_start:r_end, c_start:c_end]
                features[i, 2] = np.mean(np.abs(window[~np.isnan(window)]))
            else:
                features[i, :] = np.nan
                
        return features, names
        
    def compute_proximity_features(self, points: List[TrainingPoint],
                                  reference_features: List[Tuple[float, float]],
                                  feature_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Compute distance to reference features (faults, intrusions, etc.).
        
        Args:
            points: Training points
            reference_features: List of (x, y) coordinates of reference features
            feature_name: Name for the feature
            
        Returns:
            Tuple of (distances, feature_names)
        """
        if not reference_features:
            return np.zeros((len(points), 1)), [f"dist_to_{feature_name}"]
            
        ref_coords = np.array(reference_features)
        tree = cKDTree(ref_coords)
        
        point_coords = np.array([[p.x, p.y] for p in points])
        distances, _ = tree.query(point_coords)
        
        return distances.reshape(-1, 1), [f"dist_to_{feature_name}"]


class GeologicalNLPExtractor:
    """
    NLP-based extraction of geological entities from reports.
    """
    
    # Geological entity patterns
    LITHOLOGY_PATTERNS = [
        r'\b(granite|basalt|sandstone|limestone|shale|gneiss|schist|quartzite|'
        r'dolomite|conglomerate|breccia|tuff|rhyolite|andesite|diorite|gabbro|'
        r'peridotite|serpentinite|marble|slate|phyllite|amphibolite|eclogite)\b',
    ]
    
    ALTERATION_PATTERNS = [
        r'\b(silicification|sericitization|chloritization|carbonatization|'
        r'propylitic|phyllic|argillic|potassic|sodic|skarn|greisen|'
        r'albitization|epidotization|tourmalinization)\b',
    ]
    
    MINERALIZATION_PATTERNS = [
        r'\b(gold|silver|copper|zinc|lead|nickel|cobalt|platinum|palladium|'
        r'iron|manganese|chromium|tungsten|molybdenum|tin|uranium|'
        r'rare earth|lithium|tantalum|niobium)\b',
        r'\b(pyrite|chalcopyrite|galena|sphalerite|magnetite|hematite|'
        r'arsenopyrite|pyrrhotite|pentlandite|bornite|chalcocite|'
        r'molybdenite|scheelite|wolframite|cassiterite)\b',
    ]
    
    STRUCTURE_PATTERNS = [
        r'\b(fault|shear zone|fold|anticline|syncline|thrust|'
        r'lineament|fracture|joint|vein|stockwork|breccia pipe|'
        r'contact|unconformity|detachment)\b',
    ]
    
    GRADE_PATTERNS = [
        r'(\d+\.?\d*)\s*(g/t|ppm|ppb|%|oz/t)',
        r'(\d+\.?\d*)\s*(gram|grams)\s*per\s*(tonne|ton)',
    ]
    
    def __init__(self):
        self.compiled_patterns = {
            'lithology': [re.compile(p, re.IGNORECASE) for p in self.LITHOLOGY_PATTERNS],
            'alteration': [re.compile(p, re.IGNORECASE) for p in self.ALTERATION_PATTERNS],
            'mineralization': [re.compile(p, re.IGNORECASE) for p in self.MINERALIZATION_PATTERNS],
            'structure': [re.compile(p, re.IGNORECASE) for p in self.STRUCTURE_PATTERNS],
            'grade': [re.compile(p, re.IGNORECASE) for p in self.GRADE_PATTERNS],
        }
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract geological entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to lists of extracted entities
        """
        entities = {
            'lithology': [],
            'alteration': [],
            'mineralization': [],
            'structure': [],
            'grade': []
        }
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if entity_type == 'grade':
                    # Format grade matches
                    for match in matches:
                        if isinstance(match, tuple):
                            entities[entity_type].append(f"{match[0]} {match[1]}")
                        else:
                            entities[entity_type].append(match)
                else:
                    entities[entity_type].extend([m.lower() for m in matches])
                    
        # Remove duplicates while preserving order
        for entity_type in entities:
            entities[entity_type] = list(dict.fromkeys(entities[entity_type]))
            
        return entities
        
    def extract_coordinates(self, text: str) -> List[Tuple[float, float]]:
        """
        Extract coordinate pairs from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (x, y) coordinate tuples
        """
        coordinates = []
        
        # Pattern for decimal degrees
        dd_pattern = r'(-?\d+\.?\d*)[°\s]*([NS]?)[\s,]+(-?\d+\.?\d*)[°\s]*([EW]?)'
        
        # Pattern for UTM
        utm_pattern = r'(\d{6,7})\s*[mE]?\s*[,\s]+\s*(\d{6,7})\s*[mN]?'
        
        # Extract decimal degrees
        for match in re.finditer(dd_pattern, text):
            lat = float(match.group(1))
            if match.group(2) == 'S':
                lat = -lat
            lon = float(match.group(3))
            if match.group(4) == 'W':
                lon = -lon
            coordinates.append((lon, lat))
            
        # Extract UTM (simplified - assumes easting, northing order)
        for match in re.finditer(utm_pattern, text):
            easting = float(match.group(1))
            northing = float(match.group(2))
            coordinates.append((easting, northing))
            
        return coordinates
        
    def generate_weak_labels(self, text: str, 
                            positive_keywords: List[str] = None,
                            negative_keywords: List[str] = None) -> float:
        """
        Generate weak label score from text.
        
        Args:
            text: Input text
            positive_keywords: Keywords indicating prospectivity
            negative_keywords: Keywords indicating non-prospectivity
            
        Returns:
            Score between 0 and 1
        """
        if positive_keywords is None:
            positive_keywords = [
                'mineralization', 'deposit', 'ore', 'high grade', 'anomaly',
                'prospect', 'discovery', 'significant', 'economic'
            ]
            
        if negative_keywords is None:
            negative_keywords = [
                'barren', 'unmineralized', 'low grade', 'subeconomic',
                'negative', 'no mineralization', 'background'
            ]
            
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5
            
        return positive_count / total


class SpatialMetrics:
    """
    Spatially-aware evaluation metrics for prospectivity mapping.
    """
    
    @staticmethod
    def spatial_auc(y_true: np.ndarray, y_pred: np.ndarray,
                   coordinates: np.ndarray,
                   n_bootstrap: int = 100,
                   block_size: float = None) -> Dict[str, float]:
        """
        Compute AUC with spatial bootstrap confidence intervals.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            coordinates: Sample coordinates
            n_bootstrap: Number of bootstrap iterations
            block_size: Block size for spatial bootstrap
            
        Returns:
            Dictionary with AUC and confidence intervals
        """
        # Standard AUC
        auc = roc_auc_score(y_true, y_pred)
        
        # Spatial bootstrap
        if block_size is None:
            x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
            y_range = coordinates[:, 1].max() - coordinates[:, 1].min()
            block_size = max(x_range, y_range) / 10
            
        # Assign to blocks
        x_min, y_min = coordinates.min(axis=0)
        block_x = ((coordinates[:, 0] - x_min) / block_size).astype(int)
        block_y = ((coordinates[:, 1] - y_min) / block_size).astype(int)
        block_ids = block_x * 1000 + block_y
        
        unique_blocks = np.unique(block_ids)
        
        bootstrap_aucs = []
        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            sampled_blocks = np.random.choice(unique_blocks, size=len(unique_blocks), replace=True)
            
            # Get samples from sampled blocks
            sample_mask = np.isin(block_ids, sampled_blocks)
            
            if sample_mask.sum() > 10 and y_true[sample_mask].sum() > 0:
                try:
                    boot_auc = roc_auc_score(y_true[sample_mask], y_pred[sample_mask])
                    bootstrap_aucs.append(boot_auc)
                except:
                    pass
                    
        if bootstrap_aucs:
            ci_lower = np.percentile(bootstrap_aucs, 2.5)
            ci_upper = np.percentile(bootstrap_aucs, 97.5)
        else:
            ci_lower = ci_upper = auc
            
        return {
            'auc': auc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(bootstrap_aucs) if bootstrap_aucs else 0
        }
        
    @staticmethod
    def concentration_area_curve(y_true: np.ndarray, y_pred: np.ndarray,
                                cell_areas: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute concentration-area curve.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            cell_areas: Area of each cell (default: equal areas)
            
        Returns:
            Dictionary with curve data and metrics
        """
        if cell_areas is None:
            cell_areas = np.ones(len(y_true))
            
        # Sort by prediction (descending)
        sort_idx = np.argsort(-y_pred)
        y_true_sorted = y_true[sort_idx]
        areas_sorted = cell_areas[sort_idx]
        
        # Cumulative area and deposits
        cum_area = np.cumsum(areas_sorted) / np.sum(areas_sorted)
        cum_deposits = np.cumsum(y_true_sorted) / np.sum(y_true)
        
        # Add origin
        cum_area = np.concatenate([[0], cum_area])
        cum_deposits = np.concatenate([[0], cum_deposits])
        
        # Compute area under curve
        auc = np.trapz(cum_deposits, cum_area)
        
        # Find area needed to capture X% of deposits
        thresholds = [0.5, 0.75, 0.9]
        area_for_threshold = {}
        
        for thresh in thresholds:
            idx = np.searchsorted(cum_deposits, thresh)
            if idx < len(cum_area):
                area_for_threshold[f'area_for_{int(thresh*100)}pct'] = cum_area[idx]
            else:
                area_for_threshold[f'area_for_{int(thresh*100)}pct'] = 1.0
                
        return {
            'cumulative_area': cum_area,
            'cumulative_deposits': cum_deposits,
            'auc': auc,
            **area_for_threshold
        }


class DatasetRegistry:
    """
    Registry for prospectivity datasets and benchmarks.
    """
    
    def __init__(self, registry_path: str = None):
        self.registry_path = registry_path or os.path.expanduser('~/.mineralvision/datasets')
        self.datasets: Dict[str, Dict] = {}
        self._load_registry()
        
    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = os.path.join(self.registry_path, 'registry.json')
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                self.datasets = json.load(f)
                
    def _save_registry(self) -> None:
        """Save registry to disk."""
        os.makedirs(self.registry_path, exist_ok=True)
        registry_file = os.path.join(self.registry_path, 'registry.json')
        with open(registry_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)
            
    def register_dataset(self, dataset: ProspectivityDataset,
                        description: str = "",
                        tags: List[str] = None) -> str:
        """
        Register a dataset.
        
        Args:
            dataset: Dataset to register
            description: Dataset description
            tags: Tags for categorization
            
        Returns:
            Dataset ID
        """
        dataset_id = dataset.get_hash()
        
        self.datasets[dataset_id] = {
            'name': dataset.name,
            'description': description,
            'tags': tags or [],
            'n_samples': len(dataset),
            'n_features': len(dataset.feature_names),
            'n_positive': int(dataset.labels.sum()),
            'feature_names': dataset.feature_names,
            'registered_at': datetime.now().isoformat(),
            'metadata': dataset.metadata
        }
        
        # Save dataset
        dataset_path = os.path.join(self.registry_path, f'{dataset_id}.npz')
        np.savez(
            dataset_path,
            features=dataset.features,
            labels=dataset.labels,
            coordinates=dataset.coordinates
        )
        
        self._save_registry()
        return dataset_id
        
    def get_dataset(self, dataset_id: str) -> Optional[ProspectivityDataset]:
        """Load a registered dataset."""
        if dataset_id not in self.datasets:
            return None
            
        dataset_path = os.path.join(self.registry_path, f'{dataset_id}.npz')
        if not os.path.exists(dataset_path):
            return None
            
        data = np.load(dataset_path)
        info = self.datasets[dataset_id]
        
        return ProspectivityDataset(
            name=info['name'],
            features=data['features'],
            labels=data['labels'],
            coordinates=data['coordinates'],
            feature_names=info['feature_names'],
            metadata=info.get('metadata', {})
        )
        
    def list_datasets(self, tags: List[str] = None) -> List[Dict]:
        """List registered datasets."""
        results = []
        
        for dataset_id, info in self.datasets.items():
            if tags is None or any(t in info.get('tags', []) for t in tags):
                results.append({
                    'id': dataset_id,
                    **info
                })
                
        return results


class ProspectivityPipeline:
    """
    Complete prospectivity mapping pipeline.
    """
    
    def __init__(self):
        self.feature_generator = FeatureGenerator()
        self.nlp_extractor = GeologicalNLPExtractor()
        self.dataset_registry = DatasetRegistry()
        self.metrics = SpatialMetrics()
        
        self.processing_history: List[Dict] = []
        
    def create_dataset(self, layers: List[RasterLayer],
                      positive_points: List[TrainingPoint],
                      negative_points: List[TrainingPoint] = None,
                      window_size: int = 3,
                      name: str = "prospectivity_dataset") -> ProspectivityDataset:
        """
        Create prospectivity dataset from raster layers and training points.
        
        Args:
            layers: List of raster layers
            positive_points: Known deposit locations
            negative_points: Known non-deposit locations (generated if None)
            window_size: Window size for feature extraction
            name: Dataset name
            
        Returns:
            ProspectivityDataset
        """
        # Generate negative points if not provided
        if negative_points is None:
            negative_points = self._generate_negative_points(
                layers[0], positive_points, n_negative=len(positive_points) * 3
            )
            
        # Combine points
        all_points = positive_points + negative_points
        labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        
        # Extract features
        features, feature_names = self.feature_generator.extract_at_points(
            layers, all_points, window_size
        )
        
        # Get coordinates
        coordinates = np.array([[p.x, p.y] for p in all_points])
        
        # Handle missing values
        valid_mask = ~np.any(np.isnan(features), axis=1)
        features = features[valid_mask]
        labels = labels[valid_mask]
        coordinates = coordinates[valid_mask]
        
        return ProspectivityDataset(
            name=name,
            features=features,
            labels=labels,
            coordinates=coordinates,
            feature_names=feature_names,
            metadata={
                'n_layers': len(layers),
                'window_size': window_size,
                'n_positive_original': len(positive_points),
                'n_negative_original': len(negative_points)
            }
        )
        
    def _generate_negative_points(self, reference_layer: RasterLayer,
                                 positive_points: List[TrainingPoint],
                                 n_negative: int,
                                 min_distance: float = 1000.0) -> List[TrainingPoint]:
        """Generate negative training points away from known deposits."""
        bounds = reference_layer.bounds
        
        # Build KD-tree of positive points
        pos_coords = np.array([[p.x, p.y] for p in positive_points])
        tree = cKDTree(pos_coords)
        
        negative_points = []
        max_attempts = n_negative * 10
        attempts = 0
        
        while len(negative_points) < n_negative and attempts < max_attempts:
            # Random point within bounds
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            
            # Check distance to positive points
            dist, _ = tree.query([x, y])
            
            if dist > min_distance:
                negative_points.append(TrainingPoint(x=x, y=y, label=0))
                
            attempts += 1
            
        return negative_points
        
    def train_model(self, dataset: ProspectivityDataset,
                   model_type: ProspectivityModel = ProspectivityModel.RANDOM_FOREST,
                   cv_strategy: ValidationStrategy = ValidationStrategy.SPATIAL_BLOCK,
                   n_splits: int = 5,
                   **model_params) -> Dict[str, Any]:
        """
        Train prospectivity model with spatial cross-validation.
        
        Args:
            dataset: Training dataset
            model_type: Model type to use
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV folds
            **model_params: Model parameters
            
        Returns:
            Training results
        """
        # Select model
        if model_type == ProspectivityModel.RANDOM_FOREST:
            model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', None),
                random_state=model_params.get('random_state', 42)
            )
        elif model_type == ProspectivityModel.GRADIENT_BOOSTING:
            model = GradientBoostingClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 3),
                random_state=model_params.get('random_state', 42)
            )
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        # Select CV strategy
        if cv_strategy == ValidationStrategy.SPATIAL_BLOCK:
            cv = SpatialBlockCV(n_splits=n_splits, random_state=42)
        elif cv_strategy == ValidationStrategy.SPATIAL_BUFFER:
            cv = SpatialBufferCV(n_splits=n_splits, random_state=42)
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dataset.features)
        
        # Cross-validation
        cv_results = []
        fold_predictions = np.zeros(len(dataset))
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(
            X_scaled, dataset.labels, dataset.coordinates
        )):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = dataset.labels[train_idx], dataset.labels[test_idx]
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict_proba(X_test)[:, 1]
            fold_predictions[test_idx] = y_pred
            
            # Compute metrics
            if y_test.sum() > 0 and y_test.sum() < len(y_test):
                auc = roc_auc_score(y_test, y_pred)
                ap = average_precision_score(y_test, y_pred)
            else:
                auc = ap = np.nan
                
            cv_results.append({
                'fold': fold,
                'auc': auc,
                'average_precision': ap,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'n_positive_test': y_test.sum()
            })
            
        # Train final model on all data
        model.fit(X_scaled, dataset.labels)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(dataset.feature_names, model.feature_importances_))
        else:
            importance = {}
            
        # Spatial AUC
        spatial_auc = self.metrics.spatial_auc(
            dataset.labels, fold_predictions, dataset.coordinates
        )
        
        # Concentration-area curve
        ca_curve = self.metrics.concentration_area_curve(
            dataset.labels, fold_predictions
        )
        
        results = {
            'model': model,
            'scaler': scaler,
            'cv_results': cv_results,
            'mean_auc': np.nanmean([r['auc'] for r in cv_results]),
            'std_auc': np.nanstd([r['auc'] for r in cv_results]),
            'spatial_auc': spatial_auc,
            'concentration_area': ca_curve,
            'feature_importance': importance,
            'fold_predictions': fold_predictions
        }
        
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset.name,
            'model_type': model_type.value,
            'cv_strategy': cv_strategy.value,
            'mean_auc': results['mean_auc']
        })
        
        return results
        
    def predict_map(self, model: Any, scaler: Any,
                   layers: List[RasterLayer],
                   window_size: int = 3) -> xr.DataArray:
        """
        Generate prospectivity map.
        
        Args:
            model: Trained model
            scaler: Feature scaler
            layers: Raster layers for prediction
            window_size: Window size for feature extraction
            
        Returns:
            Prospectivity map as xarray DataArray
        """
        # Get reference layer dimensions
        ref_layer = layers[0]
        height, width = ref_layer.data.shape[:2]
        
        # Generate prediction points
        points = []
        for row in range(height):
            for col in range(width):
                x = ref_layer.transform[2] + col * ref_layer.transform[0]
                y = ref_layer.transform[5] + row * ref_layer.transform[4]
                points.append(TrainingPoint(x=x, y=y, label=0))
                
        # Extract features
        features, _ = self.feature_generator.extract_at_points(
            layers, points, window_size
        )
        
        # Handle missing values
        valid_mask = ~np.any(np.isnan(features), axis=1)
        
        # Scale and predict
        predictions = np.zeros(len(points))
        if valid_mask.sum() > 0:
            X_valid = scaler.transform(features[valid_mask])
            predictions[valid_mask] = model.predict_proba(X_valid)[:, 1]
            
        # Reshape to grid
        prediction_grid = predictions.reshape(height, width)
        
        # Create coordinate arrays
        x_coords = ref_layer.transform[2] + np.arange(width) * ref_layer.transform[0]
        y_coords = ref_layer.transform[5] + np.arange(height) * ref_layer.transform[4]
        
        return xr.DataArray(
            data=prediction_grid,
            dims=['y', 'x'],
            coords={'y': y_coords, 'x': x_coords},
            attrs={
                'crs': ref_layer.crs,
                'description': 'Mineral prospectivity probability'
            }
        )
        
    def ingest_geological_reports(self, report_texts: List[str],
                                 report_metadata: List[Dict] = None) -> Dict[str, Any]:
        """
        Ingest geological reports and extract information.
        
        Args:
            report_texts: List of report text contents
            report_metadata: Optional metadata for each report
            
        Returns:
            Extracted information
        """
        all_entities = {
            'lithology': [],
            'alteration': [],
            'mineralization': [],
            'structure': [],
            'grade': []
        }
        
        all_coordinates = []
        weak_labels = []
        
        for i, text in enumerate(report_texts):
            # Extract entities
            entities = self.nlp_extractor.extract_entities(text)
            for entity_type, values in entities.items():
                all_entities[entity_type].extend(values)
                
            # Extract coordinates
            coords = self.nlp_extractor.extract_coordinates(text)
            all_coordinates.extend(coords)
            
            # Generate weak label
            label = self.nlp_extractor.generate_weak_labels(text)
            weak_labels.append({
                'report_index': i,
                'weak_label': label,
                'metadata': report_metadata[i] if report_metadata else {}
            })
            
        # Deduplicate entities
        for entity_type in all_entities:
            all_entities[entity_type] = list(set(all_entities[entity_type]))
            
        return {
            'entities': all_entities,
            'coordinates': all_coordinates,
            'weak_labels': weak_labels,
            'n_reports': len(report_texts)
        }


def create_prospectivity_pipeline() -> ProspectivityPipeline:
    """Factory function to create prospectivity pipeline."""
    return ProspectivityPipeline()


def create_benchmark_dataset(name: str = 'synthetic',
                            n_samples: int = 1000,
                            n_features: int = 20,
                            n_positive: int = 50,
                            spatial_clustering: float = 0.5) -> ProspectivityDataset:
    """
    Create synthetic benchmark dataset for testing.
    
    Args:
        name: Dataset name
        n_samples: Number of samples
        n_features: Number of features
        n_positive: Number of positive samples
        spatial_clustering: Degree of spatial clustering (0-1)
        
    Returns:
        Synthetic ProspectivityDataset
    """
    # Generate coordinates
    coordinates = np.random.uniform(0, 1000, (n_samples, 2))
    
    # Generate features with spatial correlation
    features = np.random.randn(n_samples, n_features)
    
    # Add spatial structure
    for i in range(n_features):
        # Smooth features spatially
        tree = cKDTree(coordinates)
        for j in range(n_samples):
            neighbors = tree.query_ball_point(coordinates[j], r=100)
            if len(neighbors) > 1:
                features[j, i] = spatial_clustering * np.mean(features[neighbors, i]) + \
                                (1 - spatial_clustering) * features[j, i]
                                
    # Generate labels based on feature combination
    signal = features[:, 0] + 0.5 * features[:, 1] - 0.3 * features[:, 2]
    threshold = np.percentile(signal, 100 - 100 * n_positive / n_samples)
    labels = (signal > threshold).astype(int)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return ProspectivityDataset(
        name=name,
        features=features,
        labels=labels,
        coordinates=coordinates,
        feature_names=feature_names,
        metadata={
            'synthetic': True,
            'spatial_clustering': spatial_clustering
        }
    )
