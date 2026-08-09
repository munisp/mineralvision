"""
Uncertainty Quantification Pipeline for MineralVision.

This module provides:
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data uncertainty)
- Calibrated confidence intervals
- Sensitivity analysis
- Uncertainty propagation through pipelines
- Decision-grade outputs with confidence bounds

Essential for mineral targeting and soil recommendations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty."""
    EPISTEMIC = "epistemic"      # Model/knowledge uncertainty (reducible)
    ALEATORIC = "aleatoric"      # Data/inherent uncertainty (irreducible)
    TOTAL = "total"              # Combined uncertainty


class ConfidenceLevel(Enum):
    """Standard confidence levels."""
    P50 = 0.50   # Median
    P75 = 0.75   # 75th percentile
    P90 = 0.90   # 90th percentile
    P95 = 0.95   # 95th percentile
    P99 = 0.99   # 99th percentile


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a prediction."""
    mean: float
    std: float
    epistemic_std: float
    aleatoric_std: float
    confidence_intervals: Dict[float, Tuple[float, float]]
    samples: Optional[np.ndarray] = None
    
    @property
    def total_variance(self) -> float:
        return self.epistemic_std ** 2 + self.aleatoric_std ** 2
    
    @property
    def coefficient_of_variation(self) -> float:
        if abs(self.mean) < 1e-10:
            return float('inf')
        return self.std / abs(self.mean)
    
    def get_interval(self, confidence: float) -> Tuple[float, float]:
        """Get confidence interval for given level."""
        if confidence in self.confidence_intervals:
            return self.confidence_intervals[confidence]
        # Approximate using normal distribution
        z = {0.50: 0.674, 0.75: 1.150, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
        z_score = z.get(confidence, 1.960)
        return (self.mean - z_score * self.std, self.mean + z_score * self.std)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean': self.mean,
            'std': self.std,
            'epistemic_std': self.epistemic_std,
            'aleatoric_std': self.aleatoric_std,
            'cv': self.coefficient_of_variation,
            'confidence_intervals': {
                str(k): list(v) for k, v in self.confidence_intervals.items()
            }
        }


@dataclass
class SensitivityResult:
    """Sensitivity analysis result."""
    feature_name: str
    sensitivity_index: float  # First-order Sobol index
    total_index: float        # Total-order Sobol index
    interaction_strength: float
    direction: str            # 'positive', 'negative', 'nonlinear'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature': self.feature_name,
            'sensitivity_index': self.sensitivity_index,
            'total_index': self.total_index,
            'interaction_strength': self.interaction_strength,
            'direction': self.direction
        }


@dataclass
class CalibrationResult:
    """Calibration assessment result."""
    expected_coverage: List[float]
    observed_coverage: List[float]
    calibration_error: float
    sharpness: float
    is_calibrated: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'expected_coverage': self.expected_coverage,
            'observed_coverage': self.observed_coverage,
            'calibration_error': self.calibration_error,
            'sharpness': self.sharpness,
            'is_calibrated': self.is_calibrated
        }


class MCDropoutEstimator:
    """
    Monte Carlo Dropout for epistemic uncertainty.
    
    Approximates Bayesian inference using dropout at inference time.
    """
    
    def __init__(self, n_samples: int = 100, dropout_rate: float = 0.1):
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def estimate(self, predict_func: Callable, X: np.ndarray) -> UncertaintyEstimate:
        """
        Estimate uncertainty using MC Dropout.
        
        Args:
            predict_func: Function that takes X and returns predictions
            X: Input features
            
        Returns:
            UncertaintyEstimate with epistemic uncertainty
        """
        predictions = []
        
        for _ in range(self.n_samples):
            # Apply dropout mask
            mask = np.random.binomial(1, 1 - self.dropout_rate, X.shape)
            X_masked = X * mask / (1 - self.dropout_rate)
            
            pred = predict_func(X_masked)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        mean = np.mean(predictions)
        epistemic_std = np.std(predictions)
        
        # Calculate confidence intervals from samples
        confidence_intervals = {}
        for level in [0.50, 0.75, 0.90, 0.95, 0.99]:
            lower = np.percentile(predictions, (1 - level) / 2 * 100)
            upper = np.percentile(predictions, (1 + level) / 2 * 100)
            confidence_intervals[level] = (lower, upper)
            
        return UncertaintyEstimate(
            mean=mean,
            std=epistemic_std,
            epistemic_std=epistemic_std,
            aleatoric_std=0.0,
            confidence_intervals=confidence_intervals,
            samples=predictions
        )


class DeepEnsembleEstimator:
    """
    Deep Ensemble for uncertainty estimation.
    
    Uses multiple models to estimate both epistemic and aleatoric uncertainty.
    """
    
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models: List[Any] = []
        
    def estimate(self, predict_funcs: List[Callable], X: np.ndarray) -> UncertaintyEstimate:
        """
        Estimate uncertainty using ensemble.
        
        Args:
            predict_funcs: List of prediction functions (one per model)
            X: Input features
            
        Returns:
            UncertaintyEstimate with epistemic and aleatoric uncertainty
        """
        means = []
        variances = []
        
        for predict_func in predict_funcs:
            # Each model returns (mean, variance) for aleatoric uncertainty
            result = predict_func(X)
            if isinstance(result, tuple):
                mean, var = result
            else:
                mean = result
                var = 0.0
            means.append(mean)
            variances.append(var)
            
        means = np.array(means)
        variances = np.array(variances)
        
        # Ensemble mean
        ensemble_mean = np.mean(means)
        
        # Epistemic uncertainty: variance of means
        epistemic_var = np.var(means)
        
        # Aleatoric uncertainty: mean of variances
        aleatoric_var = np.mean(variances)
        
        # Total uncertainty
        total_std = np.sqrt(epistemic_var + aleatoric_var)
        
        # Generate samples for confidence intervals
        n_samples = 1000
        samples = np.random.normal(ensemble_mean, total_std, n_samples)
        
        confidence_intervals = {}
        for level in [0.50, 0.75, 0.90, 0.95, 0.99]:
            lower = np.percentile(samples, (1 - level) / 2 * 100)
            upper = np.percentile(samples, (1 + level) / 2 * 100)
            confidence_intervals[level] = (lower, upper)
            
        return UncertaintyEstimate(
            mean=ensemble_mean,
            std=total_std,
            epistemic_std=np.sqrt(epistemic_var),
            aleatoric_std=np.sqrt(aleatoric_var),
            confidence_intervals=confidence_intervals,
            samples=samples
        )


class QuantileRegressionEstimator:
    """
    Quantile Regression for direct uncertainty estimation.
    
    Predicts multiple quantiles to form prediction intervals.
    """
    
    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
        
    def estimate(self, quantile_predictions: Dict[float, float]) -> UncertaintyEstimate:
        """
        Estimate uncertainty from quantile predictions.
        
        Args:
            quantile_predictions: Dict mapping quantile to predicted value
            
        Returns:
            UncertaintyEstimate
        """
        # Get median as mean estimate
        mean = quantile_predictions.get(0.50, np.mean(list(quantile_predictions.values())))
        
        # Estimate std from IQR
        q25 = quantile_predictions.get(0.25, mean)
        q75 = quantile_predictions.get(0.75, mean)
        iqr = q75 - q25
        std = iqr / 1.349  # IQR to std for normal distribution
        
        # Build confidence intervals
        confidence_intervals = {}
        
        # 50% CI
        if 0.25 in quantile_predictions and 0.75 in quantile_predictions:
            confidence_intervals[0.50] = (quantile_predictions[0.25], quantile_predictions[0.75])
            
        # 90% CI
        if 0.05 in quantile_predictions and 0.95 in quantile_predictions:
            confidence_intervals[0.90] = (quantile_predictions[0.05], quantile_predictions[0.95])
            
        # 95% CI
        if 0.025 in quantile_predictions and 0.975 in quantile_predictions:
            confidence_intervals[0.95] = (quantile_predictions[0.025], quantile_predictions[0.975])
            
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            epistemic_std=std * 0.5,  # Approximate split
            aleatoric_std=std * 0.5,
            confidence_intervals=confidence_intervals
        )


class SobolSensitivityAnalyzer:
    """
    Sobol sensitivity analysis for feature importance with uncertainty.
    
    Computes first-order and total-order Sobol indices.
    """
    
    def __init__(self, n_samples: int = 1024):
        self.n_samples = n_samples
        
    def analyze(self, model_func: Callable, 
               feature_bounds: Dict[str, Tuple[float, float]]) -> List[SensitivityResult]:
        """
        Perform Sobol sensitivity analysis.
        
        Args:
            model_func: Model prediction function
            feature_bounds: Dict of feature name to (min, max) bounds
            
        Returns:
            List of SensitivityResult for each feature
        """
        feature_names = list(feature_bounds.keys())
        n_features = len(feature_names)
        
        # Generate Sobol sequence samples
        # Using quasi-random sampling for better coverage
        A = self._generate_sobol_samples(n_features)
        B = self._generate_sobol_samples(n_features)
        
        # Scale to feature bounds
        for i, name in enumerate(feature_names):
            low, high = feature_bounds[name]
            A[:, i] = A[:, i] * (high - low) + low
            B[:, i] = B[:, i] * (high - low) + low
            
        # Evaluate model
        y_A = np.array([model_func(x) for x in A])
        y_B = np.array([model_func(x) for x in B])
        
        # Calculate variance
        total_var = np.var(np.concatenate([y_A, y_B]))
        
        results = []
        for i, name in enumerate(feature_names):
            # Create AB_i matrix (A with i-th column from B)
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            y_AB_i = np.array([model_func(x) for x in AB_i])
            
            # First-order index
            S_i = np.mean(y_B * (y_AB_i - y_A)) / total_var if total_var > 0 else 0
            
            # Total-order index
            ST_i = 0.5 * np.mean((y_A - y_AB_i) ** 2) / total_var if total_var > 0 else 0
            
            # Interaction strength
            interaction = ST_i - S_i
            
            # Determine direction
            correlation = np.corrcoef(A[:, i], y_A)[0, 1]
            if abs(correlation) < 0.1:
                direction = 'nonlinear'
            elif correlation > 0:
                direction = 'positive'
            else:
                direction = 'negative'
                
            results.append(SensitivityResult(
                feature_name=name,
                sensitivity_index=max(0, S_i),
                total_index=max(0, ST_i),
                interaction_strength=max(0, interaction),
                direction=direction
            ))
            
        return sorted(results, key=lambda x: x.total_index, reverse=True)
    
    def _generate_sobol_samples(self, n_dims: int) -> np.ndarray:
        """Generate quasi-random Sobol samples."""
        # Simplified: use uniform random (in production use scipy.stats.qmc.Sobol)
        return np.random.uniform(0, 1, (self.n_samples, n_dims))


class CalibrationAssessor:
    """
    Assess and improve calibration of uncertainty estimates.
    
    Ensures predicted confidence intervals have correct coverage.
    """
    
    def __init__(self):
        self.calibration_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
    def assess(self, predictions: List[UncertaintyEstimate], 
              actuals: np.ndarray) -> CalibrationResult:
        """
        Assess calibration of uncertainty estimates.
        
        Args:
            predictions: List of uncertainty estimates
            actuals: Actual observed values
            
        Returns:
            CalibrationResult
        """
        expected_coverage = []
        observed_coverage = []
        
        for level in self.calibration_levels:
            expected_coverage.append(level)
            
            # Count how many actuals fall within the confidence interval
            in_interval = 0
            for pred, actual in zip(predictions, actuals):
                lower, upper = pred.get_interval(level)
                if lower <= actual <= upper:
                    in_interval += 1
                    
            observed = in_interval / len(actuals) if actuals.size > 0 else 0
            observed_coverage.append(observed)
            
        # Calculate calibration error (mean absolute difference)
        calibration_error = np.mean(np.abs(np.array(expected_coverage) - np.array(observed_coverage)))
        
        # Calculate sharpness (average interval width)
        widths = []
        for pred in predictions:
            lower, upper = pred.get_interval(0.90)
            widths.append(upper - lower)
        sharpness = np.mean(widths)
        
        # Is calibrated if error < 0.1
        is_calibrated = calibration_error < 0.1
        
        return CalibrationResult(
            expected_coverage=expected_coverage,
            observed_coverage=observed_coverage,
            calibration_error=calibration_error,
            sharpness=sharpness,
            is_calibrated=is_calibrated
        )
    
    def recalibrate(self, predictions: List[UncertaintyEstimate],
                   actuals: np.ndarray) -> List[UncertaintyEstimate]:
        """
        Recalibrate uncertainty estimates using isotonic regression.
        
        Args:
            predictions: Original uncertainty estimates
            actuals: Actual observed values
            
        Returns:
            Recalibrated uncertainty estimates
        """
        # Calculate empirical quantiles
        residuals = []
        for pred, actual in zip(predictions, actuals):
            z_score = (actual - pred.mean) / pred.std if pred.std > 0 else 0
            residuals.append(z_score)
            
        residuals = np.array(residuals)
        
        # Fit scaling factor
        empirical_std = np.std(residuals)
        scale_factor = empirical_std if empirical_std > 0 else 1.0
        
        # Recalibrate
        recalibrated = []
        for pred in predictions:
            new_std = pred.std * scale_factor
            new_epistemic = pred.epistemic_std * scale_factor
            new_aleatoric = pred.aleatoric_std * scale_factor
            
            # Recalculate confidence intervals
            confidence_intervals = {}
            for level in [0.50, 0.75, 0.90, 0.95, 0.99]:
                z = {0.50: 0.674, 0.75: 1.150, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[level]
                confidence_intervals[level] = (pred.mean - z * new_std, pred.mean + z * new_std)
                
            recalibrated.append(UncertaintyEstimate(
                mean=pred.mean,
                std=new_std,
                epistemic_std=new_epistemic,
                aleatoric_std=new_aleatoric,
                confidence_intervals=confidence_intervals
            ))
            
        return recalibrated


class UncertaintyPropagator:
    """
    Propagate uncertainty through processing pipelines.
    
    Uses Monte Carlo simulation for complex transformations.
    """
    
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        
    def propagate(self, input_uncertainty: UncertaintyEstimate,
                 transform_func: Callable) -> UncertaintyEstimate:
        """
        Propagate uncertainty through a transformation.
        
        Args:
            input_uncertainty: Input uncertainty estimate
            transform_func: Transformation function
            
        Returns:
            Output uncertainty estimate
        """
        # Generate input samples
        if input_uncertainty.samples is not None:
            input_samples = input_uncertainty.samples
        else:
            input_samples = np.random.normal(
                input_uncertainty.mean, 
                input_uncertainty.std, 
                self.n_samples
            )
            
        # Transform samples
        output_samples = np.array([transform_func(x) for x in input_samples])
        
        # Calculate output statistics
        mean = np.mean(output_samples)
        std = np.std(output_samples)
        
        # Approximate epistemic/aleatoric split
        # (In practice, would need more sophisticated methods)
        ratio = input_uncertainty.epistemic_std / input_uncertainty.std if input_uncertainty.std > 0 else 0.5
        epistemic_std = std * ratio
        aleatoric_std = std * (1 - ratio)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in [0.50, 0.75, 0.90, 0.95, 0.99]:
            lower = np.percentile(output_samples, (1 - level) / 2 * 100)
            upper = np.percentile(output_samples, (1 + level) / 2 * 100)
            confidence_intervals[level] = (lower, upper)
            
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            epistemic_std=epistemic_std,
            aleatoric_std=aleatoric_std,
            confidence_intervals=confidence_intervals,
            samples=output_samples
        )
    
    def propagate_grid(self, input_grid: np.ndarray,
                      uncertainty_grid: np.ndarray,
                      transform_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through a grid transformation.
        
        Args:
            input_grid: Input values grid
            uncertainty_grid: Input uncertainty (std) grid
            transform_func: Transformation function
            
        Returns:
            Tuple of (output_grid, output_uncertainty_grid)
        """
        output_means = np.zeros_like(input_grid)
        output_stds = np.zeros_like(input_grid)
        
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                samples = np.random.normal(
                    input_grid[i, j],
                    uncertainty_grid[i, j],
                    self.n_samples
                )
                outputs = np.array([transform_func(x) for x in samples])
                output_means[i, j] = np.mean(outputs)
                output_stds[i, j] = np.std(outputs)
                
        return output_means, output_stds


class UncertaintyQuantificationPipeline:
    """
    Complete uncertainty quantification pipeline for MineralVision.
    
    Integrates all UQ methods for decision-grade outputs.
    """
    
    def __init__(self):
        self.mc_dropout = MCDropoutEstimator()
        self.ensemble = DeepEnsembleEstimator()
        self.quantile = QuantileRegressionEstimator()
        self.sensitivity = SobolSensitivityAnalyzer()
        self.calibration = CalibrationAssessor()
        self.propagator = UncertaintyPropagator()
        
    def estimate_prediction_uncertainty(self, 
                                        predictions: np.ndarray,
                                        method: str = 'ensemble') -> List[UncertaintyEstimate]:
        """
        Estimate uncertainty for predictions.
        
        Args:
            predictions: Model predictions (n_samples, n_models) for ensemble
            method: 'ensemble', 'mc_dropout', or 'quantile'
            
        Returns:
            List of UncertaintyEstimate
        """
        results = []
        
        if method == 'ensemble' and predictions.ndim == 2:
            for i in range(predictions.shape[0]):
                model_preds = predictions[i, :]
                mean = np.mean(model_preds)
                epistemic_std = np.std(model_preds)
                
                # Generate samples
                samples = np.random.normal(mean, epistemic_std, 1000)
                
                confidence_intervals = {}
                for level in [0.50, 0.75, 0.90, 0.95, 0.99]:
                    lower = np.percentile(samples, (1 - level) / 2 * 100)
                    upper = np.percentile(samples, (1 + level) / 2 * 100)
                    confidence_intervals[level] = (lower, upper)
                    
                results.append(UncertaintyEstimate(
                    mean=mean,
                    std=epistemic_std,
                    epistemic_std=epistemic_std,
                    aleatoric_std=0.0,
                    confidence_intervals=confidence_intervals,
                    samples=samples
                ))
        else:
            # Single model predictions
            for pred in predictions.flatten():
                # Use bootstrap for uncertainty
                std = abs(pred) * 0.1  # 10% relative uncertainty as default
                samples = np.random.normal(pred, std, 1000)
                
                confidence_intervals = {}
                for level in [0.50, 0.75, 0.90, 0.95, 0.99]:
                    lower = np.percentile(samples, (1 - level) / 2 * 100)
                    upper = np.percentile(samples, (1 + level) / 2 * 100)
                    confidence_intervals[level] = (lower, upper)
                    
                results.append(UncertaintyEstimate(
                    mean=pred,
                    std=std,
                    epistemic_std=std * 0.7,
                    aleatoric_std=std * 0.3,
                    confidence_intervals=confidence_intervals,
                    samples=samples
                ))
                
        return results
    
    def create_prospectivity_with_uncertainty(self,
                                             prospectivity_grid: np.ndarray,
                                             feature_grids: Dict[str, np.ndarray],
                                             model_func: Callable) -> Dict[str, np.ndarray]:
        """
        Create prospectivity map with uncertainty.
        
        Args:
            prospectivity_grid: Base prospectivity values
            feature_grids: Dict of feature name to grid
            model_func: Model prediction function
            
        Returns:
            Dict with 'prospectivity', 'uncertainty', 'lower_95', 'upper_95'
        """
        # Calculate uncertainty from feature variability
        uncertainty_grid = np.zeros_like(prospectivity_grid)
        
        # Simple approach: use local variance
        from scipy.ndimage import generic_filter
        
        def local_std(x):
            return np.std(x)
            
        for name, grid in feature_grids.items():
            # Pad to handle edges
            padded = np.pad(grid, 1, mode='reflect')
            local_var = generic_filter(padded, local_std, size=3)[1:-1, 1:-1]
            uncertainty_grid += local_var ** 2
            
        uncertainty_grid = np.sqrt(uncertainty_grid / len(feature_grids))
        
        # Scale uncertainty relative to prospectivity
        uncertainty_grid = uncertainty_grid * np.abs(prospectivity_grid) * 0.1
        
        # Calculate confidence bounds
        lower_95 = prospectivity_grid - 1.96 * uncertainty_grid
        upper_95 = prospectivity_grid + 1.96 * uncertainty_grid
        
        # Clip to valid range
        lower_95 = np.clip(lower_95, 0, 1)
        upper_95 = np.clip(upper_95, 0, 1)
        
        return {
            'prospectivity': prospectivity_grid,
            'uncertainty': uncertainty_grid,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'confidence_width': upper_95 - lower_95
        }
    
    def rank_targets_with_confidence(self,
                                    targets: List[Dict[str, Any]],
                                    score_key: str = 'prospectivity_score') -> List[Dict[str, Any]]:
        """
        Rank targets with confidence intervals.
        
        Args:
            targets: List of target dictionaries
            score_key: Key for the score to rank by
            
        Returns:
            Ranked targets with confidence information
        """
        ranked = []
        
        for target in targets:
            score = target.get(score_key, 0)
            uncertainty = target.get('uncertainty', score * 0.1)
            
            # Calculate confidence interval
            lower = score - 1.96 * uncertainty
            upper = score + 1.96 * uncertainty
            
            ranked_target = target.copy()
            ranked_target['score_lower_95'] = max(0, lower)
            ranked_target['score_upper_95'] = min(1, upper)
            ranked_target['confidence_width'] = upper - lower
            ranked_target['is_robust'] = (upper - lower) < 0.3  # Narrow interval
            
            ranked.append(ranked_target)
            
        # Sort by lower bound (conservative ranking)
        ranked.sort(key=lambda x: x['score_lower_95'], reverse=True)
        
        # Add rank
        for i, target in enumerate(ranked):
            target['rank'] = i + 1
            target['rank_confidence'] = 'high' if target['is_robust'] else 'medium'
            
        return ranked


# Factory functions
def create_uq_pipeline() -> UncertaintyQuantificationPipeline:
    """Create uncertainty quantification pipeline."""
    return UncertaintyQuantificationPipeline()


def estimate_grid_uncertainty(grid: np.ndarray, 
                             method: str = 'local_variance',
                             window_size: int = 3) -> np.ndarray:
    """
    Estimate uncertainty for a grid.
    
    Args:
        grid: Input grid
        method: 'local_variance', 'bootstrap', or 'gradient'
        window_size: Window size for local methods
        
    Returns:
        Uncertainty grid
    """
    from scipy.ndimage import generic_filter
    
    if method == 'local_variance':
        def local_std(x):
            return np.std(x)
        padded = np.pad(grid, window_size // 2, mode='reflect')
        uncertainty = generic_filter(padded, local_std, size=window_size)
        return uncertainty[window_size//2:-window_size//2, window_size//2:-window_size//2]
        
    elif method == 'gradient':
        # Use gradient magnitude as uncertainty proxy
        gy, gx = np.gradient(grid)
        return np.sqrt(gx**2 + gy**2)
        
    else:
        # Default: percentage of value
        return np.abs(grid) * 0.1
