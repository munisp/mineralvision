"""
Explainable AI (XAI) Module for MineralVision.

Provides model interpretability for regulatory compliance:
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Partial dependence plots
- Individual prediction explanations
- Global model explanations
- Counterfactual explanations
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import hashlib

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations."""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    COUNTERFACTUAL = "counterfactual"
    ATTENTION = "attention"
    GRADIENT = "gradient"


class AggregationType(Enum):
    """Aggregation types for global explanations."""
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    SUM = "sum"


class PlotType(Enum):
    """Plot types for visualization."""
    BAR = "bar"
    WATERFALL = "waterfall"
    BEESWARM = "beeswarm"
    HEATMAP = "heatmap"
    FORCE = "force"
    DECISION = "decision"
    DEPENDENCE = "dependence"


@dataclass
class FeatureContribution:
    """Single feature contribution to prediction."""
    feature_name: str
    feature_value: Any
    contribution: float
    baseline_value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'feature_value': self.feature_value,
            'contribution': self.contribution,
            'baseline_value': self.baseline_value
        }


@dataclass
class LocalExplanation:
    """Explanation for a single prediction."""
    explanation_id: str
    prediction_id: str
    model_name: str
    model_version: str
    explanation_type: ExplanationType
    predicted_value: float
    baseline_value: float
    contributions: List[FeatureContribution]
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'explanation_id': self.explanation_id,
            'prediction_id': self.prediction_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'explanation_type': self.explanation_type.value,
            'predicted_value': self.predicted_value,
            'baseline_value': self.baseline_value,
            'contributions': [c.to_dict() for c in self.contributions],
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
        
    def get_top_features(self, n: int = 5, 
                        by_absolute: bool = True) -> List[FeatureContribution]:
        """Get top N contributing features."""
        if by_absolute:
            sorted_contribs = sorted(
                self.contributions, 
                key=lambda x: abs(x.contribution), 
                reverse=True
            )
        else:
            sorted_contribs = sorted(
                self.contributions,
                key=lambda x: x.contribution,
                reverse=True
            )
        return sorted_contribs[:n]
        
    def get_positive_contributions(self) -> List[FeatureContribution]:
        """Get features with positive contributions."""
        return [c for c in self.contributions if c.contribution > 0]
        
    def get_negative_contributions(self) -> List[FeatureContribution]:
        """Get features with negative contributions."""
        return [c for c in self.contributions if c.contribution < 0]


@dataclass
class GlobalExplanation:
    """Global model explanation."""
    explanation_id: str
    model_name: str
    model_version: str
    explanation_type: ExplanationType
    feature_importances: Dict[str, float]
    aggregation: AggregationType
    n_samples: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'explanation_id': self.explanation_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'explanation_type': self.explanation_type.value,
            'feature_importances': self.feature_importances,
            'aggregation': self.aggregation.value,
            'n_samples': self.n_samples,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
        
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N important features."""
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]


@dataclass
class PartialDependence:
    """Partial dependence for a feature."""
    feature_name: str
    feature_values: List[float]
    pdp_values: List[float]
    ice_values: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'feature_values': self.feature_values,
            'pdp_values': self.pdp_values,
            'ice_values': self.ice_values
        }


@dataclass
class Counterfactual:
    """Counterfactual explanation."""
    original_prediction: float
    target_prediction: float
    original_features: Dict[str, Any]
    counterfactual_features: Dict[str, Any]
    changes: Dict[str, Tuple[Any, Any]]
    distance: float
    validity: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_prediction': self.original_prediction,
            'target_prediction': self.target_prediction,
            'original_features': self.original_features,
            'counterfactual_features': self.counterfactual_features,
            'changes': {k: list(v) for k, v in self.changes.items()},
            'distance': self.distance,
            'validity': self.validity
        }


class Explainer(ABC):
    """Abstract base class for explainers."""
    
    @abstractmethod
    def explain_local(self, model: Any, X: np.ndarray,
                     feature_names: List[str]) -> LocalExplanation:
        """Generate local explanation for a single instance."""
        pass
        
    @abstractmethod
    def explain_global(self, model: Any, X: np.ndarray,
                      feature_names: List[str]) -> GlobalExplanation:
        """Generate global explanation for the model."""
        pass


class SHAPExplainer(Explainer):
    """SHAP-based explainer."""
    
    def __init__(self, model_type: str = "tree"):
        self.model_type = model_type
        self._background_data = None
        
    def set_background_data(self, X: np.ndarray) -> None:
        """Set background data for SHAP calculations."""
        if len(X) > 100:
            indices = np.random.choice(len(X), 100, replace=False)
            self._background_data = X[indices]
        else:
            self._background_data = X
            
    def explain_local(self, model: Any, X: np.ndarray,
                     feature_names: List[str],
                     prediction_id: str = None) -> LocalExplanation:
        """
        Generate SHAP explanation for a single instance.
        
        Uses KernelSHAP approximation when shap library not available.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        prediction = self._get_prediction(model, X)
        
        if self._background_data is None:
            baseline = 0.0
        else:
            baseline = np.mean(self._get_prediction(model, self._background_data))
            
        shap_values = self._compute_shap_values(model, X, feature_names)
        
        contributions = []
        for i, (name, value) in enumerate(zip(feature_names, X[0])):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(value) if isinstance(value, (int, float, np.number)) else value,
                contribution=float(shap_values[i]),
                baseline_value=baseline
            ))
            
        explanation_id = hashlib.md5(
            f"shap:{prediction_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return LocalExplanation(
            explanation_id=explanation_id,
            prediction_id=prediction_id or "unknown",
            model_name=getattr(model, '__class__.__name__', 'unknown'),
            model_version="1.0",
            explanation_type=ExplanationType.SHAP,
            predicted_value=float(prediction),
            baseline_value=float(baseline),
            contributions=contributions
        )
        
    def explain_global(self, model: Any, X: np.ndarray,
                      feature_names: List[str],
                      n_samples: int = 100) -> GlobalExplanation:
        """Generate global SHAP explanation."""
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            n_samples = len(X)
            
        all_shap_values = []
        for i in range(len(X_sample)):
            shap_values = self._compute_shap_values(model, X_sample[i:i+1], feature_names)
            all_shap_values.append(shap_values)
            
        all_shap_values = np.array(all_shap_values)
        
        mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
        
        feature_importances = {
            name: float(importance)
            for name, importance in zip(feature_names, mean_abs_shap)
        }
        
        explanation_id = hashlib.md5(
            f"shap_global:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return GlobalExplanation(
            explanation_id=explanation_id,
            model_name=getattr(model, '__class__.__name__', 'unknown'),
            model_version="1.0",
            explanation_type=ExplanationType.SHAP,
            feature_importances=feature_importances,
            aggregation=AggregationType.MEAN,
            n_samples=n_samples
        )
        
    def _get_prediction(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model prediction."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            return model.predict(X)
        elif callable(model):
            return model(X)
        else:
            raise ValueError("Model must have predict method or be callable")
            
    def _compute_shap_values(self, model: Any, X: np.ndarray,
                            feature_names: List[str]) -> np.ndarray:
        """
        Compute SHAP values using KernelSHAP approximation.
        
        This is a simplified implementation that approximates SHAP values
        using permutation-based feature importance.
        """
        n_features = X.shape[1]
        shap_values = np.zeros(n_features)
        
        if self._background_data is None:
            background = np.zeros_like(X)
        else:
            background = np.mean(self._background_data, axis=0, keepdims=True)
            
        base_pred = self._get_prediction(model, background)
        full_pred = self._get_prediction(model, X)
        
        n_permutations = min(100, 2 ** n_features)
        
        for _ in range(n_permutations):
            perm = np.random.permutation(n_features)
            
            for i, feature_idx in enumerate(perm):
                X_before = background.copy()
                X_before[0, perm[:i]] = X[0, perm[:i]]
                
                X_after = background.copy()
                X_after[0, perm[:i+1]] = X[0, perm[:i+1]]
                
                pred_before = self._get_prediction(model, X_before)
                pred_after = self._get_prediction(model, X_after)
                
                shap_values[feature_idx] += (pred_after - pred_before)[0]
                
        shap_values /= n_permutations
        
        return shap_values


class LIMEExplainer(Explainer):
    """LIME-based explainer."""
    
    def __init__(self, kernel_width: float = 0.75,
                 n_samples: int = 1000):
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        
    def explain_local(self, model: Any, X: np.ndarray,
                     feature_names: List[str],
                     prediction_id: str = None) -> LocalExplanation:
        """Generate LIME explanation for a single instance."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        prediction = self._get_prediction(model, X)
        
        perturbed_data, weights = self._generate_perturbations(X)
        
        perturbed_predictions = self._get_prediction(model, perturbed_data)
        
        coefficients = self._fit_linear_model(
            perturbed_data, perturbed_predictions, weights, X
        )
        
        contributions = []
        for i, (name, value) in enumerate(zip(feature_names, X[0])):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(value) if isinstance(value, (int, float, np.number)) else value,
                contribution=float(coefficients[i]),
                baseline_value=0.0
            ))
            
        explanation_id = hashlib.md5(
            f"lime:{prediction_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return LocalExplanation(
            explanation_id=explanation_id,
            prediction_id=prediction_id or "unknown",
            model_name=getattr(model, '__class__.__name__', 'unknown'),
            model_version="1.0",
            explanation_type=ExplanationType.LIME,
            predicted_value=float(prediction[0]),
            baseline_value=0.0,
            contributions=contributions
        )
        
    def explain_global(self, model: Any, X: np.ndarray,
                      feature_names: List[str],
                      n_samples: int = 100) -> GlobalExplanation:
        """Generate global explanation by aggregating local explanations."""
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            n_samples = len(X)
            
        all_importances = {name: [] for name in feature_names}
        
        for i in range(len(X_sample)):
            local_exp = self.explain_local(model, X_sample[i], feature_names)
            for contrib in local_exp.contributions:
                all_importances[contrib.feature_name].append(abs(contrib.contribution))
                
        feature_importances = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in all_importances.items()
        }
        
        explanation_id = hashlib.md5(
            f"lime_global:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return GlobalExplanation(
            explanation_id=explanation_id,
            model_name=getattr(model, '__class__.__name__', 'unknown'),
            model_version="1.0",
            explanation_type=ExplanationType.LIME,
            feature_importances=feature_importances,
            aggregation=AggregationType.MEAN,
            n_samples=n_samples
        )
        
    def _get_prediction(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model prediction."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            pred = model.predict(X)
            return pred.flatten() if hasattr(pred, 'flatten') else np.array([pred])
        elif callable(model):
            return model(X)
        else:
            raise ValueError("Model must have predict method or be callable")
            
    def _generate_perturbations(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate perturbed samples around the instance."""
        n_features = X.shape[1]
        
        perturbed = np.random.normal(0, 1, (self.n_samples, n_features))
        perturbed = X + perturbed * np.std(X) * 0.1
        
        distances = np.sqrt(np.sum((perturbed - X) ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        return perturbed, weights
        
    def _fit_linear_model(self, X_perturbed: np.ndarray,
                         y_perturbed: np.ndarray,
                         weights: np.ndarray,
                         X_original: np.ndarray) -> np.ndarray:
        """Fit weighted linear model."""
        X_centered = X_perturbed - X_original
        
        W = np.diag(weights)
        
        try:
            XtWX = X_centered.T @ W @ X_centered
            XtWy = X_centered.T @ W @ y_perturbed
            
            reg = 0.001 * np.eye(X_centered.shape[1])
            coefficients = np.linalg.solve(XtWX + reg, XtWy)
        except np.linalg.LinAlgError:
            coefficients = np.zeros(X_centered.shape[1])
            
        return coefficients


class FeatureImportanceExplainer(Explainer):
    """Feature importance based explainer."""
    
    def __init__(self, method: str = "permutation"):
        self.method = method
        
    def explain_local(self, model: Any, X: np.ndarray,
                     feature_names: List[str],
                     prediction_id: str = None) -> LocalExplanation:
        """Generate local explanation using feature importance."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        prediction = self._get_prediction(model, X)
        
        importances = self._compute_importance(model, X, feature_names)
        
        contributions = []
        for i, (name, value) in enumerate(zip(feature_names, X[0])):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(value) if isinstance(value, (int, float, np.number)) else value,
                contribution=float(importances[i] * value),
                baseline_value=0.0
            ))
            
        explanation_id = hashlib.md5(
            f"fi:{prediction_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return LocalExplanation(
            explanation_id=explanation_id,
            prediction_id=prediction_id or "unknown",
            model_name=getattr(model, '__class__.__name__', 'unknown'),
            model_version="1.0",
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            predicted_value=float(prediction[0]),
            baseline_value=0.0,
            contributions=contributions
        )
        
    def explain_global(self, model: Any, X: np.ndarray,
                      feature_names: List[str],
                      n_samples: int = 100) -> GlobalExplanation:
        """Generate global feature importance explanation."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = self._permutation_importance(model, X, feature_names)
            
        feature_importances = {
            name: float(imp)
            for name, imp in zip(feature_names, importances)
        }
        
        explanation_id = hashlib.md5(
            f"fi_global:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return GlobalExplanation(
            explanation_id=explanation_id,
            model_name=getattr(model, '__class__.__name__', 'unknown'),
            model_version="1.0",
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            feature_importances=feature_importances,
            aggregation=AggregationType.MEAN,
            n_samples=len(X)
        )
        
    def _get_prediction(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model prediction."""
        if hasattr(model, 'predict'):
            pred = model.predict(X)
            return pred.flatten() if hasattr(pred, 'flatten') else np.array([pred])
        elif callable(model):
            return model(X)
        else:
            raise ValueError("Model must have predict method or be callable")
            
    def _compute_importance(self, model: Any, X: np.ndarray,
                           feature_names: List[str]) -> np.ndarray:
        """Compute feature importance for single instance."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        else:
            return self._permutation_importance(model, X, feature_names)
            
    def _permutation_importance(self, model: Any, X: np.ndarray,
                               feature_names: List[str],
                               n_repeats: int = 10) -> np.ndarray:
        """Compute permutation importance."""
        baseline_pred = self._get_prediction(model, X)
        baseline_score = np.mean(baseline_pred)
        
        importances = np.zeros(len(feature_names))
        
        for i in range(len(feature_names)):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_pred = self._get_prediction(model, X_permuted)
                scores.append(np.mean(permuted_pred))
                
            importances[i] = abs(baseline_score - np.mean(scores))
            
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
            
        return importances


class PartialDependenceCalculator:
    """Calculate partial dependence plots."""
    
    def __init__(self, grid_resolution: int = 50):
        self.grid_resolution = grid_resolution
        
    def calculate(self, model: Any, X: np.ndarray,
                 feature_idx: int, feature_name: str) -> PartialDependence:
        """Calculate partial dependence for a feature."""
        feature_values = X[:, feature_idx]
        grid = np.linspace(
            np.min(feature_values),
            np.max(feature_values),
            self.grid_resolution
        )
        
        pdp_values = []
        ice_values = []
        
        for grid_value in grid:
            X_modified = X.copy()
            X_modified[:, feature_idx] = grid_value
            
            predictions = self._get_prediction(model, X_modified)
            pdp_values.append(np.mean(predictions))
            ice_values.append(predictions.tolist())
            
        return PartialDependence(
            feature_name=feature_name,
            feature_values=grid.tolist(),
            pdp_values=pdp_values,
            ice_values=ice_values
        )
        
    def calculate_2d(self, model: Any, X: np.ndarray,
                    feature_idx1: int, feature_idx2: int,
                    feature_name1: str, feature_name2: str) -> Dict[str, Any]:
        """Calculate 2D partial dependence."""
        grid1 = np.linspace(
            np.min(X[:, feature_idx1]),
            np.max(X[:, feature_idx1]),
            self.grid_resolution
        )
        grid2 = np.linspace(
            np.min(X[:, feature_idx2]),
            np.max(X[:, feature_idx2]),
            self.grid_resolution
        )
        
        pdp_values = np.zeros((len(grid1), len(grid2)))
        
        for i, val1 in enumerate(grid1):
            for j, val2 in enumerate(grid2):
                X_modified = X.copy()
                X_modified[:, feature_idx1] = val1
                X_modified[:, feature_idx2] = val2
                
                predictions = self._get_prediction(model, X_modified)
                pdp_values[i, j] = np.mean(predictions)
                
        return {
            'feature_name1': feature_name1,
            'feature_name2': feature_name2,
            'grid1': grid1.tolist(),
            'grid2': grid2.tolist(),
            'pdp_values': pdp_values.tolist()
        }
        
    def _get_prediction(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model prediction."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            pred = model.predict(X)
            return pred.flatten() if hasattr(pred, 'flatten') else np.array([pred])
        elif callable(model):
            return model(X)
        else:
            raise ValueError("Model must have predict method or be callable")


class CounterfactualGenerator:
    """Generate counterfactual explanations."""
    
    def __init__(self, n_iterations: int = 1000,
                 step_size: float = 0.1):
        self.n_iterations = n_iterations
        self.step_size = step_size
        
    def generate(self, model: Any, X: np.ndarray,
                target_prediction: float,
                feature_names: List[str],
                feature_ranges: Dict[str, Tuple[float, float]] = None,
                immutable_features: List[str] = None) -> Counterfactual:
        """
        Generate counterfactual explanation.
        
        Args:
            model: Prediction model
            X: Original instance
            target_prediction: Desired prediction value
            feature_names: Feature names
            feature_ranges: Valid ranges for features
            immutable_features: Features that cannot be changed
            
        Returns:
            Counterfactual explanation
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        original_pred = self._get_prediction(model, X)[0]
        
        immutable_indices = set()
        if immutable_features:
            for feat in immutable_features:
                if feat in feature_names:
                    immutable_indices.add(feature_names.index(feat))
                    
        X_cf = X.copy()
        
        for _ in range(self.n_iterations):
            current_pred = self._get_prediction(model, X_cf)[0]
            
            if abs(current_pred - target_prediction) < 0.01:
                break
                
            gradient = self._estimate_gradient(model, X_cf, target_prediction)
            
            for i in range(len(feature_names)):
                if i in immutable_indices:
                    continue
                    
                X_cf[0, i] += self.step_size * gradient[i]
                
                if feature_ranges and feature_names[i] in feature_ranges:
                    min_val, max_val = feature_ranges[feature_names[i]]
                    X_cf[0, i] = np.clip(X_cf[0, i], min_val, max_val)
                    
        final_pred = self._get_prediction(model, X_cf)[0]
        
        original_features = {
            name: float(X[0, i]) if isinstance(X[0, i], (int, float, np.number)) else X[0, i]
            for i, name in enumerate(feature_names)
        }
        counterfactual_features = {
            name: float(X_cf[0, i]) if isinstance(X_cf[0, i], (int, float, np.number)) else X_cf[0, i]
            for i, name in enumerate(feature_names)
        }
        
        changes = {}
        for name in feature_names:
            if abs(original_features[name] - counterfactual_features[name]) > 1e-6:
                changes[name] = (original_features[name], counterfactual_features[name])
                
        distance = np.sqrt(np.sum((X - X_cf) ** 2))
        validity = abs(final_pred - target_prediction) < 0.1
        
        return Counterfactual(
            original_prediction=float(original_pred),
            target_prediction=float(target_prediction),
            original_features=original_features,
            counterfactual_features=counterfactual_features,
            changes=changes,
            distance=float(distance),
            validity=validity
        )
        
    def _get_prediction(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model prediction."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            pred = model.predict(X)
            return pred.flatten() if hasattr(pred, 'flatten') else np.array([pred])
        elif callable(model):
            return model(X)
        else:
            raise ValueError("Model must have predict method or be callable")
            
    def _estimate_gradient(self, model: Any, X: np.ndarray,
                          target: float, epsilon: float = 1e-4) -> np.ndarray:
        """Estimate gradient using finite differences."""
        n_features = X.shape[1]
        gradient = np.zeros(n_features)
        
        current_pred = self._get_prediction(model, X)[0]
        loss = (current_pred - target) ** 2
        
        for i in range(n_features):
            X_plus = X.copy()
            X_plus[0, i] += epsilon
            
            pred_plus = self._get_prediction(model, X_plus)[0]
            loss_plus = (pred_plus - target) ** 2
            
            gradient[i] = -(loss_plus - loss) / epsilon
            
        return gradient


class ExplanationReport:
    """Generate explanation reports for regulatory compliance."""
    
    def __init__(self):
        self.sections = []
        
    def add_local_explanation(self, explanation: LocalExplanation,
                            title: str = "Prediction Explanation") -> None:
        """Add local explanation to report."""
        top_positive = explanation.get_positive_contributions()[:5]
        top_negative = explanation.get_negative_contributions()[:5]
        
        section = {
            'title': title,
            'type': 'local',
            'prediction': explanation.predicted_value,
            'baseline': explanation.baseline_value,
            'method': explanation.explanation_type.value,
            'top_positive_factors': [
                {'feature': c.feature_name, 'value': c.feature_value, 'contribution': c.contribution}
                for c in top_positive
            ],
            'top_negative_factors': [
                {'feature': c.feature_name, 'value': c.feature_value, 'contribution': c.contribution}
                for c in top_negative
            ]
        }
        self.sections.append(section)
        
    def add_global_explanation(self, explanation: GlobalExplanation,
                              title: str = "Model Feature Importance") -> None:
        """Add global explanation to report."""
        top_features = explanation.get_top_features(10)
        
        section = {
            'title': title,
            'type': 'global',
            'method': explanation.explanation_type.value,
            'n_samples': explanation.n_samples,
            'top_features': [
                {'feature': name, 'importance': importance}
                for name, importance in top_features
            ]
        }
        self.sections.append(section)
        
    def generate_ni43101_section(self, model_name: str,
                                global_exp: GlobalExplanation,
                                sample_explanations: List[LocalExplanation]) -> Dict[str, Any]:
        """Generate NI 43-101 compliant explanation section."""
        return {
            'section': 'Model Interpretability',
            'model_name': model_name,
            'methodology': f"Model explanations generated using {global_exp.explanation_type.value} methodology",
            'global_feature_importance': {
                'description': 'Features ranked by their average contribution to model predictions',
                'top_features': global_exp.get_top_features(10)
            },
            'sample_predictions': [
                {
                    'prediction': exp.predicted_value,
                    'key_factors': [
                        {'feature': c.feature_name, 'contribution': c.contribution}
                        for c in exp.get_top_features(5)
                    ]
                }
                for exp in sample_explanations[:5]
            ],
            'limitations': [
                'Explanations are approximations of model behavior',
                'Feature contributions may vary for different input ranges',
                'Interactions between features may not be fully captured'
            ]
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'generated_at': datetime.utcnow().isoformat(),
            'sections': self.sections
        }


class XAIService:
    """Main XAI service for MineralVision."""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.fi_explainer = FeatureImportanceExplainer()
        self.pdp_calculator = PartialDependenceCalculator()
        self.cf_generator = CounterfactualGenerator()
        
        self._explanation_cache: Dict[str, LocalExplanation] = {}
        
    def explain_prediction(self, model: Any, X: np.ndarray,
                          feature_names: List[str],
                          method: ExplanationType = ExplanationType.SHAP,
                          prediction_id: str = None) -> LocalExplanation:
        """Generate explanation for a prediction."""
        if method == ExplanationType.SHAP:
            return self.shap_explainer.explain_local(model, X, feature_names, prediction_id)
        elif method == ExplanationType.LIME:
            return self.lime_explainer.explain_local(model, X, feature_names, prediction_id)
        elif method == ExplanationType.FEATURE_IMPORTANCE:
            return self.fi_explainer.explain_local(model, X, feature_names, prediction_id)
        else:
            raise ValueError(f"Unsupported explanation type: {method}")
            
    def explain_model(self, model: Any, X: np.ndarray,
                     feature_names: List[str],
                     method: ExplanationType = ExplanationType.SHAP,
                     n_samples: int = 100) -> GlobalExplanation:
        """Generate global model explanation."""
        if method == ExplanationType.SHAP:
            return self.shap_explainer.explain_global(model, X, feature_names, n_samples)
        elif method == ExplanationType.LIME:
            return self.lime_explainer.explain_global(model, X, feature_names, n_samples)
        elif method == ExplanationType.FEATURE_IMPORTANCE:
            return self.fi_explainer.explain_global(model, X, feature_names, n_samples)
        else:
            raise ValueError(f"Unsupported explanation type: {method}")
            
    def calculate_pdp(self, model: Any, X: np.ndarray,
                     feature_name: str, feature_names: List[str]) -> PartialDependence:
        """Calculate partial dependence for a feature."""
        feature_idx = feature_names.index(feature_name)
        return self.pdp_calculator.calculate(model, X, feature_idx, feature_name)
        
    def generate_counterfactual(self, model: Any, X: np.ndarray,
                               target_prediction: float,
                               feature_names: List[str],
                               immutable_features: List[str] = None) -> Counterfactual:
        """Generate counterfactual explanation."""
        return self.cf_generator.generate(
            model, X, target_prediction, feature_names,
            immutable_features=immutable_features
        )
        
    def generate_compliance_report(self, model: Any, X: np.ndarray,
                                  feature_names: List[str],
                                  model_name: str) -> Dict[str, Any]:
        """Generate NI 43-101/JORC compliant explanation report."""
        global_exp = self.explain_model(model, X, feature_names)
        
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        sample_explanations = [
            self.explain_prediction(model, X[i], feature_names)
            for i in sample_indices
        ]
        
        report = ExplanationReport()
        report.add_global_explanation(global_exp)
        for exp in sample_explanations:
            report.add_local_explanation(exp)
            
        ni43101_section = report.generate_ni43101_section(
            model_name, global_exp, sample_explanations
        )
        
        return {
            'report': report.to_dict(),
            'ni43101_section': ni43101_section
        }


def create_xai_service() -> XAIService:
    """Factory function to create XAI service."""
    return XAIService()


def create_shap_explainer(model_type: str = "tree") -> SHAPExplainer:
    """Factory function to create SHAP explainer."""
    return SHAPExplainer(model_type)


def create_lime_explainer(kernel_width: float = 0.75) -> LIMEExplainer:
    """Factory function to create LIME explainer."""
    return LIMEExplainer(kernel_width)
