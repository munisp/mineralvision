"""
MLOps Hardening for MineralVision.

This module provides unified model lifecycle management:
- Consistent training/inference packaging
- Dataset versioning
- Experiment tracking
- Model cards and documentation
- Automated evaluation suites
- Drift monitoring and re-training triggers

Ensures maintainability and reproducibility across all ML components.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
import hashlib
import uuid

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DatasetType(Enum):
    """Dataset types."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"


class DriftType(Enum):
    """Types of drift."""
    DATA_DRIFT = "data_drift"           # Input distribution change
    CONCEPT_DRIFT = "concept_drift"     # Relationship change
    PREDICTION_DRIFT = "prediction_drift"  # Output distribution change


@dataclass
class DatasetVersion:
    """Versioned dataset metadata."""
    dataset_id: str
    version: str
    dataset_type: DatasetType
    n_samples: int
    n_features: int
    feature_names: List[str]
    target_name: Optional[str]
    created_at: datetime
    checksum: str
    storage_path: str
    statistics: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'version': self.version,
            'type': self.dataset_type.value,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'created_at': self.created_at.isoformat(),
            'checksum': self.checksum,
            'storage_path': self.storage_path,
            'statistics': self.statistics,
            'lineage': self.lineage
        }


@dataclass
class ExperimentRun:
    """Experiment run tracking."""
    run_id: str
    experiment_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    dataset_versions: Dict[str, str]
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str = "running"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'dataset_versions': self.dataset_versions,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'tags': self.tags,
            'notes': self.notes
        }


@dataclass
class ModelCard:
    """Model documentation card."""
    model_id: str
    model_name: str
    version: str
    description: str
    intended_use: str
    limitations: List[str]
    training_data: str
    evaluation_data: str
    metrics: Dict[str, float]
    ethical_considerations: List[str]
    caveats: List[str]
    created_at: datetime
    authors: List[str]
    license: str = "Proprietary"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'description': self.description,
            'intended_use': self.intended_use,
            'limitations': self.limitations,
            'training_data': self.training_data,
            'evaluation_data': self.evaluation_data,
            'metrics': self.metrics,
            'ethical_considerations': self.ethical_considerations,
            'caveats': self.caveats,
            'created_at': self.created_at.isoformat(),
            'authors': self.authors,
            'license': self.license
        }
    
    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        md = f"# Model Card: {self.model_name}\n\n"
        md += f"**Version:** {self.version}\n"
        md += f"**Created:** {self.created_at.strftime('%Y-%m-%d')}\n"
        md += f"**Authors:** {', '.join(self.authors)}\n\n"
        
        md += f"## Description\n{self.description}\n\n"
        md += f"## Intended Use\n{self.intended_use}\n\n"
        
        md += "## Limitations\n"
        for lim in self.limitations:
            md += f"- {lim}\n"
        md += "\n"
        
        md += f"## Training Data\n{self.training_data}\n\n"
        md += f"## Evaluation Data\n{self.evaluation_data}\n\n"
        
        md += "## Metrics\n"
        for name, value in self.metrics.items():
            md += f"- **{name}:** {value:.4f}\n"
        md += "\n"
        
        md += "## Ethical Considerations\n"
        for eth in self.ethical_considerations:
            md += f"- {eth}\n"
        md += "\n"
        
        md += "## Caveats\n"
        for cav in self.caveats:
            md += f"- {cav}\n"
            
        return md


@dataclass
class DriftAlert:
    """Drift detection alert."""
    alert_id: str
    drift_type: DriftType
    model_id: str
    feature_name: Optional[str]
    drift_score: float
    threshold: float
    detected_at: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'drift_type': self.drift_type.value,
            'model_id': self.model_id,
            'feature_name': self.feature_name,
            'drift_score': self.drift_score,
            'threshold': self.threshold,
            'detected_at': self.detected_at.isoformat(),
            'severity': self.severity,
            'recommendation': self.recommendation
        }


class DatasetVersionManager:
    """
    Manage dataset versions.
    
    Provides versioning, lineage tracking, and statistics.
    """
    
    def __init__(self, storage_root: str = "/data/datasets"):
        self.storage_root = Path(storage_root)
        self._versions: Dict[str, List[DatasetVersion]] = {}
        
    def create_version(self, dataset_id: str, 
                      data: np.ndarray,
                      feature_names: List[str],
                      dataset_type: DatasetType,
                      target_name: Optional[str] = None,
                      parent_version: Optional[str] = None) -> DatasetVersion:
        """
        Create a new dataset version.
        
        Args:
            dataset_id: Dataset identifier
            data: Dataset array
            feature_names: Feature names
            dataset_type: Type of dataset
            target_name: Target column name
            parent_version: Parent version for lineage
            
        Returns:
            DatasetVersion
        """
        # Generate version
        version = f"v{len(self._versions.get(dataset_id, [])) + 1}"
        
        # Calculate checksum
        checksum = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        
        # Calculate statistics
        statistics = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'n_nulls': int(np.sum(np.isnan(data)))
        }
        
        # Build lineage
        lineage = []
        if parent_version:
            lineage.append(parent_version)
            
        # Storage path
        storage_path = str(self.storage_root / dataset_id / version)
        
        version_obj = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            dataset_type=dataset_type,
            n_samples=data.shape[0],
            n_features=data.shape[1] if data.ndim > 1 else 1,
            feature_names=feature_names,
            target_name=target_name,
            created_at=datetime.now(),
            checksum=checksum,
            storage_path=storage_path,
            statistics=statistics,
            lineage=lineage
        )
        
        if dataset_id not in self._versions:
            self._versions[dataset_id] = []
        self._versions[dataset_id].append(version_obj)
        
        return version_obj
    
    def get_version(self, dataset_id: str, version: str) -> Optional[DatasetVersion]:
        """Get specific dataset version."""
        versions = self._versions.get(dataset_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def get_latest(self, dataset_id: str) -> Optional[DatasetVersion]:
        """Get latest dataset version."""
        versions = self._versions.get(dataset_id, [])
        if versions:
            return versions[-1]
        return None
    
    def list_versions(self, dataset_id: str) -> List[DatasetVersion]:
        """List all versions of a dataset."""
        return self._versions.get(dataset_id, [])
    
    def compare_versions(self, dataset_id: str, 
                        version1: str, version2: str) -> Dict[str, Any]:
        """Compare two dataset versions."""
        v1 = self.get_version(dataset_id, version1)
        v2 = self.get_version(dataset_id, version2)
        
        if not v1 or not v2:
            return {'error': 'Version not found'}
            
        return {
            'sample_diff': v2.n_samples - v1.n_samples,
            'feature_diff': v2.n_features - v1.n_features,
            'mean_diff': v2.statistics['mean'] - v1.statistics['mean'],
            'std_diff': v2.statistics['std'] - v1.statistics['std'],
            'new_features': [f for f in v2.feature_names if f not in v1.feature_names],
            'removed_features': [f for f in v1.feature_names if f not in v2.feature_names]
        }


class ExperimentTracker:
    """
    Track ML experiments.
    
    Provides logging, comparison, and artifact management.
    """
    
    def __init__(self):
        self._experiments: Dict[str, List[ExperimentRun]] = {}
        self._active_runs: Dict[str, ExperimentRun] = {}
        
    def start_run(self, experiment_name: str,
                 model_type: str,
                 hyperparameters: Dict[str, Any],
                 dataset_versions: Dict[str, str],
                 tags: List[str] = None) -> ExperimentRun:
        """
        Start a new experiment run.
        
        Args:
            experiment_name: Name of experiment
            model_type: Type of model
            hyperparameters: Model hyperparameters
            dataset_versions: Dict of dataset type to version
            tags: Optional tags
            
        Returns:
            ExperimentRun
        """
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            model_type=model_type,
            hyperparameters=hyperparameters,
            metrics={},
            artifacts={},
            dataset_versions=dataset_versions,
            started_at=datetime.now(),
            tags=tags or []
        )
        
        if experiment_name not in self._experiments:
            self._experiments[experiment_name] = []
        self._experiments[experiment_name].append(run)
        self._active_runs[run_id] = run
        
        logger.info(f"Started experiment run: {run_id}")
        return run
    
    def log_metric(self, run_id: str, name: str, value: float):
        """Log a metric for a run."""
        if run_id in self._active_runs:
            self._active_runs[run_id].metrics[name] = value
            
    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        """Log multiple metrics."""
        if run_id in self._active_runs:
            self._active_runs[run_id].metrics.update(metrics)
            
    def log_artifact(self, run_id: str, name: str, path: str):
        """Log an artifact path."""
        if run_id in self._active_runs:
            self._active_runs[run_id].artifacts[name] = path
            
    def end_run(self, run_id: str, status: str = "completed"):
        """End an experiment run."""
        if run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.ended_at = datetime.now()
            run.status = status
            del self._active_runs[run_id]
            logger.info(f"Ended experiment run: {run_id} ({status})")
            
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run."""
        for runs in self._experiments.values():
            for run in runs:
                if run.run_id == run_id:
                    return run
        return None
    
    def list_runs(self, experiment_name: str) -> List[ExperimentRun]:
        """List all runs for an experiment."""
        return self._experiments.get(experiment_name, [])
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]
        
        if not runs:
            return {'error': 'No runs found'}
            
        # Collect all metrics
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())
            
        comparison = {
            'runs': [r.run_id for r in runs],
            'metrics': {}
        }
        
        for metric in all_metrics:
            values = [r.metrics.get(metric) for r in runs]
            comparison['metrics'][metric] = {
                'values': values,
                'best': max(v for v in values if v is not None) if any(v is not None for v in values) else None,
                'best_run': runs[values.index(max(v for v in values if v is not None))].run_id if any(v is not None for v in values) else None
            }
            
        return comparison
    
    def get_best_run(self, experiment_name: str, 
                    metric: str, higher_is_better: bool = True) -> Optional[ExperimentRun]:
        """Get best run by metric."""
        runs = self.list_runs(experiment_name)
        if not runs:
            return None
            
        valid_runs = [(r, r.metrics.get(metric)) for r in runs if metric in r.metrics]
        if not valid_runs:
            return None
            
        if higher_is_better:
            return max(valid_runs, key=lambda x: x[1])[0]
        else:
            return min(valid_runs, key=lambda x: x[1])[0]


class DriftMonitor:
    """
    Monitor for data and model drift.
    
    Detects distribution shifts and triggers re-training.
    """
    
    def __init__(self, reference_data: Optional[np.ndarray] = None):
        self.reference_data = reference_data
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self._alerts: List[DriftAlert] = []
        self.thresholds = {
            'data_drift': 0.1,      # KL divergence threshold
            'concept_drift': 0.15,  # Performance degradation threshold
            'prediction_drift': 0.1  # Output distribution shift threshold
        }
        
    def set_reference(self, data: np.ndarray, feature_names: List[str]):
        """Set reference data for drift detection."""
        self.reference_data = data
        
        for i, name in enumerate(feature_names):
            col = data[:, i] if data.ndim > 1 else data
            self.reference_stats[name] = {
                'mean': float(np.mean(col)),
                'std': float(np.std(col)),
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'median': float(np.median(col))
            }
            
    def detect_data_drift(self, current_data: np.ndarray,
                         feature_names: List[str]) -> List[DriftAlert]:
        """
        Detect data drift using statistical tests.
        
        Args:
            current_data: Current data batch
            feature_names: Feature names
            
        Returns:
            List of drift alerts
        """
        alerts = []
        
        for i, name in enumerate(feature_names):
            if name not in self.reference_stats:
                continue
                
            ref_stats = self.reference_stats[name]
            col = current_data[:, i] if current_data.ndim > 1 else current_data
            
            # Calculate drift score using normalized mean difference
            current_mean = np.mean(col)
            current_std = np.std(col)
            
            mean_drift = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-10)
            std_drift = abs(current_std - ref_stats['std']) / (ref_stats['std'] + 1e-10)
            
            drift_score = (mean_drift + std_drift) / 2
            
            if drift_score > self.thresholds['data_drift']:
                severity = 'low' if drift_score < 0.2 else ('medium' if drift_score < 0.5 else 'high')
                
                alert = DriftAlert(
                    alert_id=f"drift_{uuid.uuid4().hex[:8]}",
                    drift_type=DriftType.DATA_DRIFT,
                    model_id="",
                    feature_name=name,
                    drift_score=drift_score,
                    threshold=self.thresholds['data_drift'],
                    detected_at=datetime.now(),
                    severity=severity,
                    recommendation=f"Feature '{name}' shows significant drift. Consider retraining."
                )
                alerts.append(alert)
                self._alerts.append(alert)
                
        return alerts
    
    def detect_prediction_drift(self, reference_predictions: np.ndarray,
                               current_predictions: np.ndarray,
                               model_id: str) -> Optional[DriftAlert]:
        """
        Detect prediction distribution drift.
        
        Args:
            reference_predictions: Reference prediction distribution
            current_predictions: Current predictions
            model_id: Model identifier
            
        Returns:
            DriftAlert if drift detected
        """
        # Calculate distribution shift
        ref_mean = np.mean(reference_predictions)
        ref_std = np.std(reference_predictions)
        
        curr_mean = np.mean(current_predictions)
        curr_std = np.std(current_predictions)
        
        drift_score = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
        
        if drift_score > self.thresholds['prediction_drift']:
            severity = 'medium' if drift_score < 0.3 else 'high'
            
            alert = DriftAlert(
                alert_id=f"drift_{uuid.uuid4().hex[:8]}",
                drift_type=DriftType.PREDICTION_DRIFT,
                model_id=model_id,
                feature_name=None,
                drift_score=drift_score,
                threshold=self.thresholds['prediction_drift'],
                detected_at=datetime.now(),
                severity=severity,
                recommendation="Prediction distribution has shifted. Investigate model performance."
            )
            self._alerts.append(alert)
            return alert
            
        return None
    
    def get_alerts(self, since: Optional[datetime] = None) -> List[DriftAlert]:
        """Get drift alerts."""
        if since:
            return [a for a in self._alerts if a.detected_at >= since]
        return self._alerts
    
    def should_retrain(self, model_id: str) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        recent_alerts = [a for a in self._alerts 
                        if a.model_id == model_id 
                        and a.detected_at > datetime.now() - timedelta(days=7)]
        
        high_severity = [a for a in recent_alerts if a.severity in ['high', 'critical']]
        
        if high_severity:
            return True, f"High severity drift detected: {high_severity[0].recommendation}"
            
        if len(recent_alerts) >= 3:
            return True, f"Multiple drift alerts ({len(recent_alerts)}) in past week"
            
        return False, "No significant drift detected"


class EvaluationSuite:
    """
    Automated evaluation suite for models.
    
    Provides domain-specific evaluation metrics.
    """
    
    def __init__(self, domain: str = "mineral_exploration"):
        self.domain = domain
        self.metrics_registry: Dict[str, Callable] = {}
        self._register_default_metrics()
        
    def _register_default_metrics(self):
        """Register default evaluation metrics."""
        # Classification metrics
        self.metrics_registry['accuracy'] = lambda y, p: np.mean(y == p)
        self.metrics_registry['precision'] = self._precision
        self.metrics_registry['recall'] = self._recall
        self.metrics_registry['f1'] = self._f1_score
        
        # Regression metrics
        self.metrics_registry['rmse'] = lambda y, p: np.sqrt(np.mean((y - p) ** 2))
        self.metrics_registry['mae'] = lambda y, p: np.mean(np.abs(y - p))
        self.metrics_registry['r2'] = self._r2_score
        
        # Ranking metrics
        self.metrics_registry['auc'] = self._auc_score
        
    def _precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = self._precision(y_true, y_pred)
        r = self._recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    def _auc_score(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        # Simple AUC calculation
        sorted_indices = np.argsort(y_scores)[::-1]
        y_sorted = y_true[sorted_indices]
        
        tpr = np.cumsum(y_sorted) / np.sum(y_sorted)
        fpr = np.cumsum(1 - y_sorted) / np.sum(1 - y_sorted)
        
        return np.trapz(tpr, fpr)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to compute
            
        Returns:
            Dict of metric name to value
        """
        if metrics is None:
            metrics = list(self.metrics_registry.keys())
            
        results = {}
        for metric in metrics:
            if metric in self.metrics_registry:
                try:
                    results[metric] = float(self.metrics_registry[metric](y_true, y_pred))
                except Exception as e:
                    logger.warning(f"Failed to compute {metric}: {e}")
                    
        return results
    
    def evaluate_prospectivity(self, y_true: np.ndarray, 
                              y_scores: np.ndarray,
                              top_k: List[int] = None) -> Dict[str, float]:
        """
        Evaluate prospectivity model.
        
        Args:
            y_true: True deposit locations (binary)
            y_scores: Prospectivity scores
            top_k: List of k values for precision@k
            
        Returns:
            Evaluation metrics
        """
        top_k = top_k or [10, 50, 100, 500]
        
        results = {
            'auc': self._auc_score(y_true, y_scores)
        }
        
        # Precision at k
        sorted_indices = np.argsort(y_scores)[::-1]
        y_sorted = y_true[sorted_indices]
        
        for k in top_k:
            if k <= len(y_sorted):
                results[f'precision@{k}'] = np.mean(y_sorted[:k])
                
        # Capture rate
        total_positives = np.sum(y_true)
        for k in top_k:
            if k <= len(y_sorted) and total_positives > 0:
                results[f'capture@{k}'] = np.sum(y_sorted[:k]) / total_positives
                
        return results


class MLOpsPipeline:
    """
    Unified MLOps pipeline for MineralVision.
    
    Integrates all MLOps components.
    """
    
    def __init__(self):
        self.dataset_manager = DatasetVersionManager()
        self.experiment_tracker = ExperimentTracker()
        self.drift_monitor = DriftMonitor()
        self.evaluation_suite = EvaluationSuite()
        self._model_cards: Dict[str, ModelCard] = {}
        
    def create_model_card(self, model_id: str,
                         model_name: str,
                         version: str,
                         description: str,
                         intended_use: str,
                         metrics: Dict[str, float],
                         authors: List[str]) -> ModelCard:
        """
        Create model documentation card.
        
        Args:
            model_id: Model identifier
            model_name: Human-readable name
            version: Model version
            description: Model description
            intended_use: Intended use cases
            metrics: Evaluation metrics
            authors: Model authors
            
        Returns:
            ModelCard
        """
        card = ModelCard(
            model_id=model_id,
            model_name=model_name,
            version=version,
            description=description,
            intended_use=intended_use,
            limitations=[
                "Performance may vary with different geological settings",
                "Requires representative training data",
                "Should be validated with domain experts"
            ],
            training_data="See dataset version in experiment tracking",
            evaluation_data="See dataset version in experiment tracking",
            metrics=metrics,
            ethical_considerations=[
                "Model predictions should inform, not replace, expert judgment",
                "Consider environmental and social impacts of exploration",
                "Ensure fair access to model outputs"
            ],
            caveats=[
                "Model uncertainty should be considered in decision making",
                "Regular monitoring for drift is recommended"
            ],
            created_at=datetime.now(),
            authors=authors
        )
        
        self._model_cards[model_id] = card
        return card
    
    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        """Get model card."""
        return self._model_cards.get(model_id)
    
    def run_training_pipeline(self, 
                             experiment_name: str,
                             model_type: str,
                             train_data: np.ndarray,
                             val_data: np.ndarray,
                             feature_names: List[str],
                             hyperparameters: Dict[str, Any],
                             train_func: Callable) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            experiment_name: Experiment name
            model_type: Model type
            train_data: Training data
            val_data: Validation data
            feature_names: Feature names
            hyperparameters: Model hyperparameters
            train_func: Training function
            
        Returns:
            Pipeline results
        """
        # Version datasets
        train_version = self.dataset_manager.create_version(
            f"{experiment_name}_train",
            train_data,
            feature_names,
            DatasetType.TRAINING
        )
        
        val_version = self.dataset_manager.create_version(
            f"{experiment_name}_val",
            val_data,
            feature_names,
            DatasetType.VALIDATION
        )
        
        # Start experiment
        run = self.experiment_tracker.start_run(
            experiment_name=experiment_name,
            model_type=model_type,
            hyperparameters=hyperparameters,
            dataset_versions={
                'train': train_version.version,
                'validation': val_version.version
            }
        )
        
        try:
            # Train model
            model, metrics = train_func(train_data, val_data, hyperparameters)
            
            # Log metrics
            self.experiment_tracker.log_metrics(run.run_id, metrics)
            
            # Set drift reference
            self.drift_monitor.set_reference(train_data, feature_names)
            
            # End run
            self.experiment_tracker.end_run(run.run_id, "completed")
            
            return {
                'run_id': run.run_id,
                'metrics': metrics,
                'train_version': train_version.version,
                'val_version': val_version.version,
                'status': 'success'
            }
            
        except Exception as e:
            self.experiment_tracker.end_run(run.run_id, "failed")
            return {
                'run_id': run.run_id,
                'status': 'failed',
                'error': str(e)
            }


# Factory functions
def create_mlops_pipeline() -> MLOpsPipeline:
    """Create MLOps pipeline."""
    return MLOpsPipeline()


def create_experiment_tracker() -> ExperimentTracker:
    """Create experiment tracker."""
    return ExperimentTracker()


def create_drift_monitor(reference_data: np.ndarray = None,
                        feature_names: List[str] = None) -> DriftMonitor:
    """Create drift monitor."""
    monitor = DriftMonitor()
    if reference_data is not None and feature_names is not None:
        monitor.set_reference(reference_data, feature_names)
    return monitor
