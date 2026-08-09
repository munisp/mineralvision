"""
Spatial Cross-Validation for MineralVision.

Implements proper spatial cross-validation strategies to prevent
data leakage from spatial autocorrelation in geospatial ML models.

Key strategies:
- Block CV: Spatial blocking to ensure train/test independence
- Buffered Leave-One-Out: Buffer zones around test points
- Spatial K-Fold: Spatially stratified k-fold cross-validation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable
import json

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseCrossValidator = object


class CVStrategy(str, Enum):
    """Cross-validation strategies."""
    BLOCK = "block"
    BUFFERED_LOO = "buffered_loo"
    SPATIAL_KFOLD = "spatial_kfold"
    CLUSTER = "cluster"
    RANDOM = "random"  # For comparison


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_coords: Optional[np.ndarray] = None
    test_coords: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_train(self) -> int:
        return len(self.train_indices)
    
    @property
    def n_test(self) -> int:
        return len(self.test_indices)


@dataclass
class CVResult:
    """Results from cross-validation."""
    strategy: CVStrategy
    n_folds: int
    scores: Dict[str, List[float]]
    fold_details: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mean_scores(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.scores.items()}
    
    @property
    def std_scores(self) -> Dict[str, float]:
        return {k: np.std(v) for k, v in self.scores.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'n_folds': self.n_folds,
            'mean_scores': self.mean_scores,
            'std_scores': self.std_scores,
            'fold_details': self.fold_details,
            'metadata': self.metadata
        }


class BlockCV:
    """
    Block Cross-Validation for spatial data.
    
    Divides the study area into spatial blocks and assigns
    entire blocks to train or test sets to prevent spatial leakage.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        block_size: Optional[float] = None,
        n_blocks_x: Optional[int] = None,
        n_blocks_y: Optional[int] = None,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.block_size = block_size
        self.n_blocks_x = n_blocks_x
        self.n_blocks_y = n_blocks_y
        self.random_state = random_state
    
    def split(
        self,
        coordinates: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[CVFold]:
        """Generate train/test splits."""
        np.random.seed(self.random_state)
        
        # Determine block grid
        x_min, y_min = coordinates.min(axis=0)
        x_max, y_max = coordinates.max(axis=0)
        
        if self.block_size is not None:
            n_blocks_x = int(np.ceil((x_max - x_min) / self.block_size))
            n_blocks_y = int(np.ceil((y_max - y_min) / self.block_size))
        else:
            n_blocks_x = self.n_blocks_x or 5
            n_blocks_y = self.n_blocks_y or 5
        
        block_width = (x_max - x_min) / n_blocks_x
        block_height = (y_max - y_min) / n_blocks_y
        
        # Assign points to blocks
        block_x = ((coordinates[:, 0] - x_min) / block_width).astype(int)
        block_y = ((coordinates[:, 1] - y_min) / block_height).astype(int)
        block_x = np.clip(block_x, 0, n_blocks_x - 1)
        block_y = np.clip(block_y, 0, n_blocks_y - 1)
        
        block_ids = block_x * n_blocks_y + block_y
        unique_blocks = np.unique(block_ids)
        
        # Shuffle blocks
        np.random.shuffle(unique_blocks)
        
        # Assign blocks to folds
        fold_size = len(unique_blocks) // self.n_folds
        
        for fold_id in range(self.n_folds):
            start_idx = fold_id * fold_size
            if fold_id == self.n_folds - 1:
                test_blocks = unique_blocks[start_idx:]
            else:
                test_blocks = unique_blocks[start_idx:start_idx + fold_size]
            
            test_mask = np.isin(block_ids, test_blocks)
            train_indices = np.where(~test_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(test_indices) == 0:
                continue
            
            yield CVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                train_coords=coordinates[train_indices],
                test_coords=coordinates[test_indices],
                metadata={
                    'n_blocks': len(unique_blocks),
                    'test_blocks': test_blocks.tolist(),
                    'block_size': (block_width, block_height)
                }
            )
    
    def get_n_splits(self, coordinates: np.ndarray = None) -> int:
        return self.n_folds


class BufferedLeaveOneOut:
    """
    Buffered Leave-One-Out Cross-Validation.
    
    Excludes points within a buffer distance of the test point
    from the training set to prevent spatial autocorrelation leakage.
    """
    
    def __init__(
        self,
        buffer_distance: float,
        max_samples: Optional[int] = None,
        random_state: int = 42
    ):
        self.buffer_distance = buffer_distance
        self.max_samples = max_samples
        self.random_state = random_state
    
    def split(
        self,
        coordinates: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[CVFold]:
        """Generate train/test splits."""
        np.random.seed(self.random_state)
        
        n_samples = len(coordinates)
        
        # Optionally subsample for large datasets
        if self.max_samples is not None and n_samples > self.max_samples:
            sample_indices = np.random.choice(
                n_samples, self.max_samples, replace=False
            )
        else:
            sample_indices = np.arange(n_samples)
        
        for fold_id, test_idx in enumerate(sample_indices):
            test_coord = coordinates[test_idx]
            
            # Calculate distances to all other points
            distances = np.sqrt(
                np.sum((coordinates - test_coord) ** 2, axis=1)
            )
            
            # Exclude points within buffer
            train_mask = distances > self.buffer_distance
            train_mask[test_idx] = False  # Exclude test point
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.array([test_idx])
            
            if len(train_indices) == 0:
                continue
            
            yield CVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                train_coords=coordinates[train_indices],
                test_coords=coordinates[test_indices],
                metadata={
                    'buffer_distance': self.buffer_distance,
                    'n_excluded': n_samples - len(train_indices) - 1
                }
            )
    
    def get_n_splits(self, coordinates: np.ndarray = None) -> int:
        if coordinates is None:
            return 0
        n = len(coordinates)
        if self.max_samples is not None:
            return min(n, self.max_samples)
        return n


class SpatialKFold:
    """
    Spatial K-Fold Cross-Validation.
    
    Uses clustering to create spatially coherent folds
    that respect spatial structure in the data.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        method: str = "kmeans",
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.method = method
        self.random_state = random_state
    
    def split(
        self,
        coordinates: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[CVFold]:
        """Generate train/test splits using spatial clustering."""
        np.random.seed(self.random_state)
        
        # Cluster coordinates
        if self.method == "kmeans":
            cluster_labels = self._kmeans_cluster(coordinates)
        else:
            cluster_labels = self._grid_cluster(coordinates)
        
        unique_clusters = np.unique(cluster_labels)
        np.random.shuffle(unique_clusters)
        
        # Assign clusters to folds
        fold_assignments = np.zeros(len(unique_clusters), dtype=int)
        for i, cluster in enumerate(unique_clusters):
            fold_assignments[i] = i % self.n_folds
        
        cluster_to_fold = dict(zip(unique_clusters, fold_assignments))
        point_folds = np.array([cluster_to_fold[c] for c in cluster_labels])
        
        for fold_id in range(self.n_folds):
            test_mask = point_folds == fold_id
            train_indices = np.where(~test_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(test_indices) == 0:
                continue
            
            yield CVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                train_coords=coordinates[train_indices],
                test_coords=coordinates[test_indices],
                metadata={
                    'method': self.method,
                    'n_clusters': len(unique_clusters)
                }
            )
    
    def _kmeans_cluster(self, coordinates: np.ndarray) -> np.ndarray:
        """Cluster using k-means."""
        try:
            from sklearn.cluster import KMeans
            
            # Use more clusters than folds for better spatial coverage
            n_clusters = min(self.n_folds * 3, len(coordinates) // 5)
            n_clusters = max(n_clusters, self.n_folds)
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            return kmeans.fit_predict(coordinates)
        except ImportError:
            return self._grid_cluster(coordinates)
    
    def _grid_cluster(self, coordinates: np.ndarray) -> np.ndarray:
        """Fallback grid-based clustering."""
        x_min, y_min = coordinates.min(axis=0)
        x_max, y_max = coordinates.max(axis=0)
        
        n_cells = int(np.ceil(np.sqrt(self.n_folds * 3)))
        
        cell_width = (x_max - x_min) / n_cells
        cell_height = (y_max - y_min) / n_cells
        
        cell_x = ((coordinates[:, 0] - x_min) / cell_width).astype(int)
        cell_y = ((coordinates[:, 1] - y_min) / cell_height).astype(int)
        
        cell_x = np.clip(cell_x, 0, n_cells - 1)
        cell_y = np.clip(cell_y, 0, n_cells - 1)
        
        return cell_x * n_cells + cell_y
    
    def get_n_splits(self, coordinates: np.ndarray = None) -> int:
        return self.n_folds


class SpatialCrossValidator:
    """
    Main spatial cross-validation class.
    
    Provides a unified interface for different spatial CV strategies
    and includes evaluation metrics.
    """
    
    def __init__(
        self,
        strategy: CVStrategy = CVStrategy.BLOCK,
        n_folds: int = 5,
        block_size: Optional[float] = None,
        buffer_distance: Optional[float] = None,
        random_state: int = 42
    ):
        self.strategy = strategy
        self.n_folds = n_folds
        self.block_size = block_size
        self.buffer_distance = buffer_distance
        self.random_state = random_state
        
        self._cv = self._create_cv()
    
    def _create_cv(self):
        """Create the appropriate CV object."""
        if self.strategy == CVStrategy.BLOCK:
            return BlockCV(
                n_folds=self.n_folds,
                block_size=self.block_size,
                random_state=self.random_state
            )
        elif self.strategy == CVStrategy.BUFFERED_LOO:
            return BufferedLeaveOneOut(
                buffer_distance=self.buffer_distance or 1000.0,
                max_samples=min(self.n_folds * 20, 200),
                random_state=self.random_state
            )
        elif self.strategy == CVStrategy.SPATIAL_KFOLD:
            return SpatialKFold(
                n_folds=self.n_folds,
                random_state=self.random_state
            )
        else:
            return BlockCV(
                n_folds=self.n_folds,
                random_state=self.random_state
            )
    
    def split(
        self,
        coordinates: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[CVFold]:
        """Generate cross-validation splits."""
        return self._cv.split(coordinates, y)
    
    def validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        coordinates: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> CVResult:
        """
        Perform cross-validation with a model.
        
        Args:
            model: Sklearn-compatible model with fit/predict methods
            X: Feature matrix
            y: Target labels
            coordinates: Spatial coordinates
            metrics: List of metrics to compute
        
        Returns:
            CVResult with scores for each fold
        """
        metrics = metrics or ['auc', 'accuracy', 'f1']
        
        scores = {m: [] for m in metrics}
        fold_details = []
        
        for fold in self.split(coordinates, y):
            X_train = X[fold.train_indices]
            X_test = X[fold.test_indices]
            y_train = y[fold.train_indices]
            y_test = y[fold.test_indices]
            
            # Handle missing values
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)
            
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            fold_scores = {}
            
            if 'auc' in metrics:
                try:
                    auc = roc_auc_score(y_test, y_prob)
                    scores['auc'].append(auc)
                    fold_scores['auc'] = auc
                except ValueError:
                    pass
            
            if 'accuracy' in metrics:
                acc = accuracy_score(y_test, y_pred)
                scores['accuracy'].append(acc)
                fold_scores['accuracy'] = acc
            
            if 'f1' in metrics:
                f1 = f1_score(y_test, y_pred, zero_division=0)
                scores['f1'].append(f1)
                fold_scores['f1'] = f1
            
            fold_details.append({
                'fold_id': fold.fold_id,
                'n_train': fold.n_train,
                'n_test': fold.n_test,
                'scores': fold_scores,
                'metadata': fold.metadata
            })
        
        return CVResult(
            strategy=self.strategy,
            n_folds=len(fold_details),
            scores=scores,
            fold_details=fold_details,
            metadata={
                'block_size': self.block_size,
                'buffer_distance': self.buffer_distance
            }
        )
    
    def compare_with_random(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        coordinates: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare spatial CV with random CV to detect spatial leakage.
        
        A large difference between spatial and random CV scores
        indicates spatial autocorrelation is inflating random CV scores.
        """
        # Spatial CV
        spatial_result = self.validate(model, X, y, coordinates)
        
        # Random CV (ignoring spatial structure)
        from sklearn.model_selection import cross_val_score
        
        random_scores = cross_val_score(
            model, X, y, cv=self.n_folds, scoring='roc_auc'
        )
        
        spatial_auc = spatial_result.mean_scores.get('auc', 0)
        random_auc = np.mean(random_scores)
        
        leakage_indicator = random_auc - spatial_auc
        
        return {
            'spatial_cv': spatial_result.to_dict(),
            'random_cv': {
                'mean_auc': float(random_auc),
                'std_auc': float(np.std(random_scores)),
                'fold_scores': random_scores.tolist()
            },
            'leakage_indicator': float(leakage_indicator),
            'leakage_detected': leakage_indicator > 0.05,
            'recommendation': (
                "Significant spatial leakage detected. Use spatial CV for reliable estimates."
                if leakage_indicator > 0.05
                else "Minimal spatial leakage. Random CV may be acceptable."
            )
        }


def validate_spatial_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    coordinates: np.ndarray,
    strategy: str = "block",
    n_folds: int = 5,
    block_size: Optional[float] = None,
    buffer_distance: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function for spatial model validation.
    
    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target labels
        coordinates: Spatial coordinates (N, 2)
        strategy: CV strategy ('block', 'buffered_loo', 'spatial_kfold')
        n_folds: Number of folds
        block_size: Block size for block CV
        buffer_distance: Buffer distance for buffered LOO
    
    Returns:
        Dictionary with validation results
    """
    cv = SpatialCrossValidator(
        strategy=CVStrategy(strategy),
        n_folds=n_folds,
        block_size=block_size,
        buffer_distance=buffer_distance
    )
    
    result = cv.validate(model, X, y, coordinates)
    comparison = cv.compare_with_random(model, X, y, coordinates)
    
    return {
        'spatial_cv_result': result.to_dict(),
        'leakage_comparison': comparison,
        'summary': {
            'spatial_auc': result.mean_scores.get('auc', 0),
            'spatial_auc_std': result.std_scores.get('auc', 0),
            'random_auc': comparison['random_cv']['mean_auc'],
            'leakage_detected': comparison['leakage_detected']
        }
    }
