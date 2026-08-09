"""
UNCOVER-ML Style Prospectivity Workflow for MineralVision.

Implements validated prospectivity mapping workflows inspired by
Geoscience Australia's UNCOVER-ML framework.

Key features:
- Multi-scale feature engineering
- Ensemble modeling with uncertainty quantification
- Spatial cross-validation
- Reproducible workflows
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class FeatureScale(str, Enum):
    """Feature extraction scales."""
    POINT = "point"
    LOCAL = "local"  # 3x3 window
    NEIGHBORHOOD = "neighborhood"  # 5x5 window
    REGIONAL = "regional"  # 11x11 window
    MULTI_SCALE = "multi_scale"


class AggregationType(str, Enum):
    """Aggregation types for window features."""
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    GRADIENT = "gradient"
    TEXTURE = "texture"


class ModelType(str, Enum):
    """Available model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"


@dataclass
class RasterStack:
    """A stack of raster layers for feature extraction."""
    layers: Dict[str, np.ndarray]
    transform: Optional[Any] = None
    crs: Optional[str] = None
    nodata: float = -9999.0
    
    @property
    def shape(self) -> Tuple[int, int]:
        if self.layers:
            first_layer = next(iter(self.layers.values()))
            return first_layer.shape
        return (0, 0)
    
    @property
    def n_layers(self) -> int:
        return len(self.layers)


@dataclass
class TrainingData:
    """Training data for prospectivity modeling."""
    X: np.ndarray
    y: np.ndarray
    coordinates: np.ndarray
    feature_names: List[str]
    weights: Optional[np.ndarray] = None
    
    @property
    def n_samples(self) -> int:
        return len(self.y)
    
    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1
    
    @property
    def positive_ratio(self) -> float:
        return np.mean(self.y)


@dataclass
class PredictionResult:
    """Result of prospectivity prediction."""
    probability: np.ndarray
    uncertainty: np.ndarray
    feature_importance: Dict[str, float]
    model_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureEngineering:
    """
    Multi-scale feature engineering for prospectivity mapping.
    
    Extracts features at multiple spatial scales using
    various aggregation methods.
    """
    
    def __init__(
        self,
        scales: List[FeatureScale] = None,
        aggregations: List[AggregationType] = None
    ):
        self.scales = scales or [
            FeatureScale.POINT,
            FeatureScale.LOCAL,
            FeatureScale.NEIGHBORHOOD
        ]
        self.aggregations = aggregations or [
            AggregationType.MEAN,
            AggregationType.STD,
            AggregationType.GRADIENT
        ]
        
        self._scale_windows = {
            FeatureScale.POINT: 1,
            FeatureScale.LOCAL: 3,
            FeatureScale.NEIGHBORHOOD: 5,
            FeatureScale.REGIONAL: 11,
        }
    
    def extract_features(
        self,
        raster_stack: RasterStack,
        coordinates: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features at specified coordinates."""
        features = []
        feature_names = []
        
        for layer_name, layer_data in raster_stack.layers.items():
            for scale in self.scales:
                if scale == FeatureScale.MULTI_SCALE:
                    # Extract at all scales
                    for s in [FeatureScale.POINT, FeatureScale.LOCAL, 
                              FeatureScale.NEIGHBORHOOD, FeatureScale.REGIONAL]:
                        f, n = self._extract_scale_features(
                            layer_data, coordinates, layer_name, s
                        )
                        features.append(f)
                        feature_names.extend(n)
                else:
                    f, n = self._extract_scale_features(
                        layer_data, coordinates, layer_name, scale
                    )
                    features.append(f)
                    feature_names.extend(n)
        
        X = np.hstack(features) if features else np.array([])
        return X, feature_names
    
    def _extract_scale_features(
        self,
        layer: np.ndarray,
        coordinates: np.ndarray,
        layer_name: str,
        scale: FeatureScale
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features at a specific scale."""
        window_size = self._scale_windows.get(scale, 1)
        half_window = window_size // 2
        
        n_samples = len(coordinates)
        features = []
        names = []
        
        if scale == FeatureScale.POINT:
            # Point value only
            values = np.zeros(n_samples)
            for i, (row, col) in enumerate(coordinates.astype(int)):
                if 0 <= row < layer.shape[0] and 0 <= col < layer.shape[1]:
                    values[i] = layer[row, col]
                else:
                    values[i] = np.nan
            features.append(values.reshape(-1, 1))
            names.append(f"{layer_name}_point")
        else:
            # Window aggregations
            for agg in self.aggregations:
                values = np.zeros(n_samples)
                for i, (row, col) in enumerate(coordinates.astype(int)):
                    row, col = int(row), int(col)
                    r_start = max(0, row - half_window)
                    r_end = min(layer.shape[0], row + half_window + 1)
                    c_start = max(0, col - half_window)
                    c_end = min(layer.shape[1], col + half_window + 1)
                    
                    window = layer[r_start:r_end, c_start:c_end]
                    values[i] = self._aggregate(window, agg)
                
                features.append(values.reshape(-1, 1))
                names.append(f"{layer_name}_{scale.value}_{agg.value}")
        
        return np.hstack(features), names
    
    def _aggregate(self, window: np.ndarray, agg_type: AggregationType) -> float:
        """Apply aggregation to window."""
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            return np.nan
        
        if agg_type == AggregationType.MEAN:
            return np.mean(valid)
        elif agg_type == AggregationType.MEDIAN:
            return np.median(valid)
        elif agg_type == AggregationType.STD:
            return np.std(valid)
        elif agg_type == AggregationType.MIN:
            return np.min(valid)
        elif agg_type == AggregationType.MAX:
            return np.max(valid)
        elif agg_type == AggregationType.RANGE:
            return np.max(valid) - np.min(valid)
        elif agg_type == AggregationType.GRADIENT:
            if window.shape[0] >= 3 and window.shape[1] >= 3:
                gy, gx = np.gradient(window)
                return np.sqrt(np.nanmean(gx**2 + gy**2))
            return 0.0
        elif agg_type == AggregationType.TEXTURE:
            # Simple texture measure (local variance)
            return np.var(valid)
        else:
            return np.mean(valid)
    
    def generate_derived_features(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate derived features (ratios, products, etc.)."""
        derived = []
        derived_names = []
        
        # Add log transforms for positive features
        for i, name in enumerate(feature_names):
            if np.all(X[:, i] > 0):
                log_vals = np.log1p(X[:, i])
                derived.append(log_vals.reshape(-1, 1))
                derived_names.append(f"{name}_log")
        
        # Add feature interactions (limited to avoid explosion)
        n_features = min(X.shape[1], 10)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = X[:, i] / (X[:, j] + 1e-10)
                    ratio = np.clip(ratio, -100, 100)
                derived.append(ratio.reshape(-1, 1))
                derived_names.append(f"{feature_names[i]}_div_{feature_names[j]}")
        
        if derived:
            X_derived = np.hstack([X] + derived)
            all_names = feature_names + derived_names
        else:
            X_derived = X
            all_names = feature_names
        
        return X_derived, all_names


class UncertaintyQuantification:
    """
    Uncertainty quantification for prospectivity predictions.
    
    Provides multiple uncertainty measures including:
    - Model uncertainty (epistemic)
    - Data uncertainty (aleatoric)
    - Ensemble disagreement
    """
    
    def __init__(self, n_bootstrap: int = 100):
        self.n_bootstrap = n_bootstrap
    
    def bootstrap_uncertainty(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        X_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate uncertainty via bootstrap."""
        n_samples = len(y)
        predictions = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit and predict
            model.fit(X_boot, y_boot)
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_pred)[:, 1]
            else:
                pred = model.predict(X_pred)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def ensemble_uncertainty(
        self,
        predictions: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate uncertainty from ensemble predictions."""
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def prediction_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate prediction entropy."""
        p = np.clip(probabilities, 1e-10, 1 - 1e-10)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return entropy


class ModelEnsemble:
    """
    Ensemble of models for prospectivity prediction.
    
    Combines multiple model types with uncertainty quantification.
    """
    
    def __init__(
        self,
        model_types: List[ModelType] = None,
        n_estimators: int = 100
    ):
        self.model_types = model_types or [
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING
        ]
        self.n_estimators = n_estimators
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_names: List[str] = []
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create a model instance."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for modeling")
        
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fit(
        self,
        training_data: TrainingData,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """Fit ensemble models."""
        X = training_data.X
        y = training_data.y
        self.feature_names = training_data.feature_names
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        scores = {}
        for model_type in self.model_types:
            model = self._create_model(model_type)
            model.fit(X, y)
            self.models[model_type.value] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            scores[model_type.value] = float(np.mean(cv_scores))
        
        return scores
    
    def predict(
        self,
        X: np.ndarray,
        return_individual: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble."""
        X = np.nan_to_num(X, nan=0.0)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        predictions = []
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X).astype(float)
            predictions.append(pred)
        
        # Ensemble mean and uncertainty
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        if return_individual:
            return mean_pred, std_pred, predictions
        return mean_pred, std_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance."""
        importance = np.zeros(len(self.feature_names))
        n_models = 0
        
        for model in self.models.values():
            if hasattr(model, 'feature_importances_'):
                importance += model.feature_importances_
                n_models += 1
        
        if n_models > 0:
            importance /= n_models
        
        return dict(zip(self.feature_names, importance))


class ProspectivityWorkflow:
    """
    Complete prospectivity mapping workflow.
    
    Implements the full UNCOVER-ML style workflow:
    1. Feature engineering
    2. Spatial cross-validation
    3. Ensemble modeling
    4. Uncertainty quantification
    5. Map generation
    """
    
    def __init__(
        self,
        feature_scales: List[FeatureScale] = None,
        model_types: List[ModelType] = None,
        cv_block_size: float = 10000.0  # meters
    ):
        self.feature_engineering = FeatureEngineering(scales=feature_scales)
        self.ensemble = ModelEnsemble(model_types=model_types)
        self.uncertainty = UncertaintyQuantification()
        self.cv_block_size = cv_block_size
        
        self.training_data: Optional[TrainingData] = None
        self.model_scores: Dict[str, float] = {}
    
    def prepare_training_data(
        self,
        raster_stack: RasterStack,
        positive_coords: np.ndarray,
        negative_coords: np.ndarray
    ) -> TrainingData:
        """Prepare training data from raster stack and coordinates."""
        # Combine coordinates
        all_coords = np.vstack([positive_coords, negative_coords])
        y = np.hstack([
            np.ones(len(positive_coords)),
            np.zeros(len(negative_coords))
        ])
        
        # Extract features
        X, feature_names = self.feature_engineering.extract_features(
            raster_stack, all_coords
        )
        
        # Generate derived features
        X, feature_names = self.feature_engineering.generate_derived_features(
            X, feature_names
        )
        
        self.training_data = TrainingData(
            X=X,
            y=y,
            coordinates=all_coords,
            feature_names=feature_names
        )
        
        return self.training_data
    
    def train(self) -> Dict[str, float]:
        """Train the ensemble model."""
        if self.training_data is None:
            raise ValueError("No training data. Call prepare_training_data first.")
        
        self.model_scores = self.ensemble.fit(self.training_data)
        return self.model_scores
    
    def predict_map(
        self,
        raster_stack: RasterStack,
        chunk_size: int = 1000
    ) -> PredictionResult:
        """Generate prospectivity map."""
        height, width = raster_stack.shape
        
        # Generate all coordinates
        rows, cols = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'
        )
        all_coords = np.column_stack([rows.ravel(), cols.ravel()])
        
        # Predict in chunks
        n_pixels = len(all_coords)
        probabilities = np.zeros(n_pixels)
        uncertainties = np.zeros(n_pixels)
        
        for i in range(0, n_pixels, chunk_size):
            chunk_coords = all_coords[i:i + chunk_size]
            X_chunk, _ = self.feature_engineering.extract_features(
                raster_stack, chunk_coords
            )
            X_chunk, _ = self.feature_engineering.generate_derived_features(
                X_chunk, self.training_data.feature_names[:X_chunk.shape[1]]
            )
            
            # Pad if needed
            if X_chunk.shape[1] < self.training_data.n_features:
                padding = np.zeros((X_chunk.shape[0], 
                                   self.training_data.n_features - X_chunk.shape[1]))
                X_chunk = np.hstack([X_chunk, padding])
            elif X_chunk.shape[1] > self.training_data.n_features:
                X_chunk = X_chunk[:, :self.training_data.n_features]
            
            prob, unc = self.ensemble.predict(X_chunk)
            probabilities[i:i + chunk_size] = prob
            uncertainties[i:i + chunk_size] = unc
        
        # Reshape to map
        prob_map = probabilities.reshape(height, width)
        unc_map = uncertainties.reshape(height, width)
        
        return PredictionResult(
            probability=prob_map,
            uncertainty=unc_map,
            feature_importance=self.ensemble.get_feature_importance(),
            model_scores=self.model_scores,
            metadata={
                'n_training_samples': self.training_data.n_samples,
                'n_features': self.training_data.n_features,
                'positive_ratio': self.training_data.positive_ratio
            }
        )
    
    def cross_validate(self, n_folds: int = 5) -> Dict[str, Any]:
        """Perform spatial cross-validation."""
        if self.training_data is None:
            raise ValueError("No training data")
        
        coords = self.training_data.coordinates
        X = self.training_data.X
        y = self.training_data.y
        
        # Create spatial blocks
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        block_assignments = (
            (coords[:, 0] - x_min) // self.cv_block_size * 1000 +
            (coords[:, 1] - y_min) // self.cv_block_size
        ).astype(int)
        
        unique_blocks = np.unique(block_assignments)
        np.random.shuffle(unique_blocks)
        
        fold_size = len(unique_blocks) // n_folds
        fold_scores = []
        
        for fold in range(n_folds):
            test_blocks = unique_blocks[fold * fold_size:(fold + 1) * fold_size]
            test_mask = np.isin(block_assignments, test_blocks)
            
            X_train, X_test = X[~test_mask], X[test_mask]
            y_train, y_test = y[~test_mask], y[test_mask]
            
            if len(y_test) == 0 or len(np.unique(y_test)) < 2:
                continue
            
            # Train and evaluate
            temp_ensemble = ModelEnsemble(model_types=self.ensemble.model_types)
            temp_data = TrainingData(
                X=X_train, y=y_train,
                coordinates=coords[~test_mask],
                feature_names=self.training_data.feature_names
            )
            temp_ensemble.fit(temp_data)
            
            prob, _ = temp_ensemble.predict(X_test)
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_test, prob)
                fold_scores.append(auc)
            except ValueError:
                pass
        
        return {
            'mean_auc': np.mean(fold_scores) if fold_scores else 0.0,
            'std_auc': np.std(fold_scores) if fold_scores else 0.0,
            'fold_scores': fold_scores,
            'n_folds': len(fold_scores)
        }


class UncoverMLPipeline:
    """
    High-level UNCOVER-ML style pipeline.
    
    Provides a simple interface for prospectivity mapping.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.workflow = ProspectivityWorkflow(
            feature_scales=config.get('feature_scales'),
            model_types=config.get('model_types'),
            cv_block_size=config.get('cv_block_size', 10000.0)
        )
        
        self.results: Optional[PredictionResult] = None
        self.cv_results: Optional[Dict[str, Any]] = None
    
    def run(
        self,
        raster_stack: RasterStack,
        positive_coords: np.ndarray,
        negative_coords: np.ndarray,
        validate: bool = True
    ) -> PredictionResult:
        """Run the complete pipeline."""
        logger.info("Preparing training data...")
        self.workflow.prepare_training_data(
            raster_stack, positive_coords, negative_coords
        )
        
        if validate:
            logger.info("Running spatial cross-validation...")
            self.cv_results = self.workflow.cross_validate()
            logger.info(f"CV AUC: {self.cv_results['mean_auc']:.3f} +/- {self.cv_results['std_auc']:.3f}")
        
        logger.info("Training ensemble...")
        self.workflow.train()
        
        logger.info("Generating prospectivity map...")
        self.results = self.workflow.predict_map(raster_stack)
        
        return self.results
    
    def save_results(self, output_dir: str) -> None:
        """Save results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.results is not None:
            # Save probability map
            np.save(output_path / "probability.npy", self.results.probability)
            np.save(output_path / "uncertainty.npy", self.results.uncertainty)
            
            # Save metadata
            metadata = {
                'feature_importance': self.results.feature_importance,
                'model_scores': self.results.model_scores,
                'metadata': self.results.metadata,
                'cv_results': self.cv_results
            }
            with open(output_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)


def create_prospectivity_pipeline(
    config: Optional[Dict[str, Any]] = None
) -> UncoverMLPipeline:
    """Factory function to create UNCOVER-ML pipeline."""
    return UncoverMLPipeline(config)
