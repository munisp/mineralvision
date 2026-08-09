"""
Automated Model Retraining Pipeline for MineralVision.

Provides:
- Scheduled retraining triggers
- Drift-based retraining triggers
- A/B testing framework
- Champion/challenger model comparison
- Automatic rollback on performance degradation
- Training job orchestration
- Model promotion workflows
"""

import time
import threading
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Types of retraining triggers."""
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"
    FEEDBACK_LOOP = "feedback_loop"


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    ARCHIVED = "archived"
    ROLLBACK = "rollback"


class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ABTestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CHAMPION_WINS = "champion_wins"
    CHALLENGER_WINS = "challenger_wins"
    INCONCLUSIVE = "inconclusive"


class RollbackReason(Enum):
    """Reasons for model rollback."""
    PERFORMANCE_DROP = "performance_drop"
    ERROR_RATE_SPIKE = "error_rate_spike"
    LATENCY_INCREASE = "latency_increase"
    MANUAL = "manual"
    AB_TEST_FAILURE = "ab_test_failure"


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    training_data_path: str
    validation_data_path: str
    test_data_path: str
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "r2"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'training_data_path': self.training_data_path,
            'validation_data_path': self.validation_data_path,
            'test_data_path': self.test_data_path,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'early_stopping_patience': self.early_stopping_patience,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'metrics': self.metrics
        }


@dataclass
class TrainingMetrics:
    """Training metrics."""
    loss: float
    val_loss: float
    metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    epoch: int
    training_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'loss': self.loss,
            'val_loss': self.val_loss,
            'metrics': self.metrics,
            'val_metrics': self.val_metrics,
            'epoch': self.epoch,
            'training_time_seconds': self.training_time_seconds
        }


@dataclass
class TrainingJob:
    """Training job."""
    job_id: str
    config: TrainingConfig
    trigger: RetrainingTrigger
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[TrainingMetrics] = None
    model_artifact_path: Optional[str] = None
    error_message: Optional[str] = None
    parent_model_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'config': self.config.to_dict(),
            'trigger': self.trigger.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'model_artifact_path': self.model_artifact_path,
            'error_message': self.error_message,
            'parent_model_version': self.parent_model_version
        }


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_name: str
    stage: ModelStage
    training_job_id: str
    metrics: Dict[str, float]
    created_at: datetime
    promoted_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    artifact_path: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'stage': self.stage.value,
            'training_job_id': self.training_job_id,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'promoted_at': self.promoted_at.isoformat() if self.promoted_at else None,
            'archived_at': self.archived_at.isoformat() if self.archived_at else None,
            'artifact_path': self.artifact_path,
            'tags': self.tags
        }


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    champion_version: str
    challenger_version: str
    traffic_split: float  # Fraction to challenger (0-1)
    min_samples: int
    max_duration_hours: int
    primary_metric: str
    significance_level: float = 0.05
    min_improvement: float = 0.01  # Minimum improvement to declare winner
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_id': self.test_id,
            'champion_version': self.champion_version,
            'challenger_version': self.challenger_version,
            'traffic_split': self.traffic_split,
            'min_samples': self.min_samples,
            'max_duration_hours': self.max_duration_hours,
            'primary_metric': self.primary_metric,
            'significance_level': self.significance_level,
            'min_improvement': self.min_improvement
        }


@dataclass
class ABTestResult:
    """A/B test result."""
    test_id: str
    status: ABTestStatus
    champion_samples: int
    challenger_samples: int
    champion_metric: float
    challenger_metric: float
    p_value: float
    improvement: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    winner: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_id': self.test_id,
            'status': self.status.value,
            'champion_samples': self.champion_samples,
            'challenger_samples': self.challenger_samples,
            'champion_metric': self.champion_metric,
            'challenger_metric': self.challenger_metric,
            'p_value': self.p_value,
            'improvement': self.improvement,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'winner': self.winner
        }


@dataclass
class RollbackEvent:
    """Model rollback event."""
    rollback_id: str
    model_name: str
    from_version: str
    to_version: str
    reason: RollbackReason
    triggered_at: datetime
    completed_at: Optional[datetime] = None
    triggered_by: str = "system"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rollback_id': self.rollback_id,
            'model_name': self.model_name,
            'from_version': self.from_version,
            'to_version': self.to_version,
            'reason': self.reason.value,
            'triggered_at': self.triggered_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'triggered_by': self.triggered_by,
            'details': self.details
        }


@dataclass
class RetrainingSchedule:
    """Retraining schedule configuration."""
    schedule_id: str
    model_name: str
    cron_expression: str  # e.g., "0 0 * * 0" for weekly
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    config_template: Optional[TrainingConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'schedule_id': self.schedule_id,
            'model_name': self.model_name,
            'cron_expression': self.cron_expression,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None
        }


class DriftTriggerEvaluator:
    """Evaluate drift-based retraining triggers."""
    
    def __init__(self, drift_threshold: float = 0.7,
                 consecutive_alerts: int = 3):
        self.drift_threshold = drift_threshold
        self.consecutive_alerts = consecutive_alerts
        self._drift_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
        
    def record_drift(self, model_name: str, drift_score: float,
                    drift_type: str) -> bool:
        """
        Record drift score and check if retraining should be triggered.
        
        Returns:
            True if retraining should be triggered
        """
        key = f"{model_name}:{drift_type}"
        
        with self._lock:
            if key not in self._drift_history:
                self._drift_history[key] = deque(maxlen=self.consecutive_alerts)
                
            self._drift_history[key].append(drift_score)
            
            if len(self._drift_history[key]) >= self.consecutive_alerts:
                if all(s >= self.drift_threshold for s in self._drift_history[key]):
                    self._drift_history[key].clear()
                    return True
                    
        return False
        
    def get_drift_status(self, model_name: str) -> Dict[str, Any]:
        """Get current drift status for a model."""
        with self._lock:
            status = {}
            for key, history in self._drift_history.items():
                if key.startswith(f"{model_name}:"):
                    drift_type = key.split(":")[1]
                    status[drift_type] = {
                        'recent_scores': list(history),
                        'above_threshold': sum(1 for s in history if s >= self.drift_threshold),
                        'trigger_ready': len(history) >= self.consecutive_alerts and 
                                        all(s >= self.drift_threshold for s in history)
                    }
            return status


class PerformanceMonitor:
    """Monitor model performance for degradation."""
    
    def __init__(self, window_size: int = 1000,
                 degradation_threshold: float = 0.1):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self._metrics: Dict[str, deque] = {}
        self._baselines: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def set_baseline(self, model_name: str, metric_name: str,
                    baseline_value: float) -> None:
        """Set baseline metric value."""
        key = f"{model_name}:{metric_name}"
        with self._lock:
            self._baselines[key] = baseline_value
            
    def record_metric(self, model_name: str, metric_name: str,
                     value: float) -> Optional[Dict[str, Any]]:
        """
        Record metric and check for degradation.
        
        Returns:
            Degradation info if detected, None otherwise
        """
        key = f"{model_name}:{metric_name}"
        
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=self.window_size)
                
            self._metrics[key].append(value)
            
            if key in self._baselines and len(self._metrics[key]) >= 100:
                baseline = self._baselines[key]
                current_mean = np.mean(list(self._metrics[key]))
                
                if metric_name in ['accuracy', 'r2', 'auc']:
                    degradation = (baseline - current_mean) / baseline
                else:
                    degradation = (current_mean - baseline) / baseline
                    
                if degradation > self.degradation_threshold:
                    return {
                        'model_name': model_name,
                        'metric_name': metric_name,
                        'baseline': baseline,
                        'current': current_mean,
                        'degradation_percent': degradation * 100
                    }
                    
        return None
        
    def get_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance summary for a model."""
        with self._lock:
            summary = {}
            for key, values in self._metrics.items():
                if key.startswith(f"{model_name}:"):
                    metric_name = key.split(":")[1]
                    values_list = list(values)
                    summary[metric_name] = {
                        'current_mean': np.mean(values_list) if values_list else None,
                        'current_std': np.std(values_list) if values_list else None,
                        'baseline': self._baselines.get(key),
                        'sample_count': len(values_list)
                    }
            return summary


class ABTestManager:
    """Manage A/B tests between model versions."""
    
    def __init__(self):
        self._tests: Dict[str, ABTestConfig] = {}
        self._results: Dict[str, ABTestResult] = {}
        self._samples: Dict[str, Dict[str, List[float]]] = {}
        self._lock = threading.Lock()
        
    def create_test(self, config: ABTestConfig) -> None:
        """Create a new A/B test."""
        with self._lock:
            self._tests[config.test_id] = config
            self._results[config.test_id] = ABTestResult(
                test_id=config.test_id,
                status=ABTestStatus.PENDING,
                champion_samples=0,
                challenger_samples=0,
                champion_metric=0.0,
                challenger_metric=0.0,
                p_value=1.0,
                improvement=0.0,
                started_at=datetime.utcnow()
            )
            self._samples[config.test_id] = {
                'champion': [],
                'challenger': []
            }
            
    def start_test(self, test_id: str) -> None:
        """Start an A/B test."""
        with self._lock:
            if test_id in self._results:
                self._results[test_id].status = ABTestStatus.RUNNING
                self._results[test_id].started_at = datetime.utcnow()
                
    def record_sample(self, test_id: str, variant: str,
                     metric_value: float) -> Optional[ABTestResult]:
        """
        Record a sample for A/B test.
        
        Returns:
            Test result if test is complete, None otherwise
        """
        with self._lock:
            if test_id not in self._tests:
                return None
                
            config = self._tests[test_id]
            result = self._results[test_id]
            
            if result.status != ABTestStatus.RUNNING:
                return None
                
            self._samples[test_id][variant].append(metric_value)
            
            champion_samples = self._samples[test_id]['champion']
            challenger_samples = self._samples[test_id]['challenger']
            
            result.champion_samples = len(champion_samples)
            result.challenger_samples = len(challenger_samples)
            
            if champion_samples:
                result.champion_metric = np.mean(champion_samples)
            if challenger_samples:
                result.challenger_metric = np.mean(challenger_samples)
                
            total_samples = result.champion_samples + result.challenger_samples
            
            if total_samples >= config.min_samples:
                result = self._evaluate_test(test_id)
                
            hours_elapsed = (datetime.utcnow() - result.started_at).total_seconds() / 3600
            if hours_elapsed >= config.max_duration_hours:
                result = self._evaluate_test(test_id, force=True)
                
            return result if result.status != ABTestStatus.RUNNING else None
            
    def _evaluate_test(self, test_id: str, force: bool = False) -> ABTestResult:
        """Evaluate A/B test results."""
        config = self._tests[test_id]
        result = self._results[test_id]
        
        champion_samples = self._samples[test_id]['champion']
        challenger_samples = self._samples[test_id]['challenger']
        
        if len(champion_samples) < 30 or len(challenger_samples) < 30:
            if force:
                result.status = ABTestStatus.INCONCLUSIVE
                result.completed_at = datetime.utcnow()
            return result
            
        champion_mean = np.mean(champion_samples)
        challenger_mean = np.mean(challenger_samples)
        champion_std = np.std(champion_samples)
        challenger_std = np.std(challenger_samples)
        
        pooled_std = np.sqrt(
            (champion_std**2 / len(champion_samples)) +
            (challenger_std**2 / len(challenger_samples))
        )
        
        if pooled_std > 0:
            t_stat = (challenger_mean - champion_mean) / pooled_std
            from scipy import stats
            try:
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            except ImportError:
                p_value = 0.5 if abs(t_stat) < 2 else 0.01
        else:
            p_value = 1.0
            
        result.p_value = p_value
        
        if champion_mean != 0:
            result.improvement = (challenger_mean - champion_mean) / abs(champion_mean)
        else:
            result.improvement = 0.0
            
        result.completed_at = datetime.utcnow()
        
        if p_value < config.significance_level:
            if result.improvement > config.min_improvement:
                result.status = ABTestStatus.CHALLENGER_WINS
                result.winner = config.challenger_version
            elif result.improvement < -config.min_improvement:
                result.status = ABTestStatus.CHAMPION_WINS
                result.winner = config.champion_version
            else:
                result.status = ABTestStatus.INCONCLUSIVE
        else:
            if force:
                result.status = ABTestStatus.INCONCLUSIVE
                
        return result
        
    def get_test_result(self, test_id: str) -> Optional[ABTestResult]:
        """Get A/B test result."""
        with self._lock:
            return self._results.get(test_id)
            
    def get_active_tests(self) -> List[ABTestResult]:
        """Get all active A/B tests."""
        with self._lock:
            return [r for r in self._results.values() 
                   if r.status == ABTestStatus.RUNNING]
                   
    def route_request(self, test_id: str) -> str:
        """
        Route a request to champion or challenger.
        
        Returns:
            'champion' or 'challenger'
        """
        with self._lock:
            if test_id not in self._tests:
                return 'champion'
                
            config = self._tests[test_id]
            result = self._results[test_id]
            
            if result.status != ABTestStatus.RUNNING:
                return 'champion'
                
            if np.random.random() < config.traffic_split:
                return 'challenger'
            return 'champion'


class RollbackManager:
    """Manage model rollbacks."""
    
    def __init__(self):
        self._rollback_history: List[RollbackEvent] = []
        self._current_versions: Dict[str, str] = {}
        self._version_history: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[RollbackEvent], None]] = []
        
    def register_callback(self, callback: Callable[[RollbackEvent], None]) -> None:
        """Register rollback callback."""
        self._callbacks.append(callback)
        
    def set_current_version(self, model_name: str, version_id: str) -> None:
        """Set current model version."""
        with self._lock:
            if model_name not in self._version_history:
                self._version_history[model_name] = []
            self._version_history[model_name].append(version_id)
            self._current_versions[model_name] = version_id
            
    def get_current_version(self, model_name: str) -> Optional[str]:
        """Get current model version."""
        with self._lock:
            return self._current_versions.get(model_name)
            
    def get_previous_version(self, model_name: str) -> Optional[str]:
        """Get previous model version."""
        with self._lock:
            history = self._version_history.get(model_name, [])
            if len(history) >= 2:
                return history[-2]
            return None
            
    def rollback(self, model_name: str, reason: RollbackReason,
                to_version: str = None, triggered_by: str = "system",
                details: Dict[str, Any] = None) -> RollbackEvent:
        """
        Perform model rollback.
        
        Args:
            model_name: Model to rollback
            reason: Reason for rollback
            to_version: Version to rollback to (default: previous)
            triggered_by: Who triggered the rollback
            details: Additional details
            
        Returns:
            Rollback event
        """
        with self._lock:
            current_version = self._current_versions.get(model_name)
            if not current_version:
                raise ValueError(f"No current version for model: {model_name}")
                
            if to_version is None:
                to_version = self.get_previous_version(model_name)
                if not to_version:
                    raise ValueError(f"No previous version to rollback to for: {model_name}")
                    
            rollback_id = hashlib.md5(
                f"{model_name}:{current_version}:{to_version}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16]
            
            event = RollbackEvent(
                rollback_id=rollback_id,
                model_name=model_name,
                from_version=current_version,
                to_version=to_version,
                reason=reason,
                triggered_at=datetime.utcnow(),
                triggered_by=triggered_by,
                details=details or {}
            )
            
            self._current_versions[model_name] = to_version
            self._rollback_history.append(event)
            event.completed_at = datetime.utcnow()
            
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")
                
        return event
        
    def get_rollback_history(self, model_name: str = None,
                            limit: int = 100) -> List[RollbackEvent]:
        """Get rollback history."""
        with self._lock:
            history = self._rollback_history
            if model_name:
                history = [e for e in history if e.model_name == model_name]
            return history[-limit:]


class TrainingJobQueue:
    """Queue and manage training jobs."""
    
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._pending: deque = deque()
        self._running: Dict[str, TrainingJob] = {}
        self._completed: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._job_callbacks: List[Callable[[TrainingJob], None]] = []
        
    def register_callback(self, callback: Callable[[TrainingJob], None]) -> None:
        """Register job completion callback."""
        self._job_callbacks.append(callback)
        
    def submit(self, job: TrainingJob) -> str:
        """Submit a training job."""
        with self._lock:
            self._pending.append(job)
            logger.info(f"Training job submitted: {job.job_id}")
            return job.job_id
            
    def start_next(self) -> Optional[TrainingJob]:
        """Start next pending job if capacity available."""
        with self._lock:
            if len(self._running) >= self.max_concurrent:
                return None
                
            if not self._pending:
                return None
                
            job = self._pending.popleft()
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            self._running[job.job_id] = job
            
            logger.info(f"Training job started: {job.job_id}")
            return job
            
    def complete_job(self, job_id: str, metrics: TrainingMetrics = None,
                    artifact_path: str = None, error: str = None) -> Optional[TrainingJob]:
        """Mark job as completed."""
        with self._lock:
            job = self._running.pop(job_id, None)
            if not job:
                return None
                
            job.completed_at = datetime.utcnow()
            
            if error:
                job.status = TrainingStatus.FAILED
                job.error_message = error
            else:
                job.status = TrainingStatus.COMPLETED
                job.metrics = metrics
                job.model_artifact_path = artifact_path
                
            self._completed[job_id] = job
            
        for callback in self._job_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Job callback error: {e}")
                
        logger.info(f"Training job completed: {job_id} - {job.status.value}")
        return job
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            for i, job in enumerate(self._pending):
                if job.job_id == job_id:
                    job.status = TrainingStatus.CANCELLED
                    self._pending.remove(job)
                    self._completed[job_id] = job
                    return True
                    
            if job_id in self._running:
                job = self._running.pop(job_id)
                job.status = TrainingStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                self._completed[job_id] = job
                return True
                
        return False
        
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        with self._lock:
            if job_id in self._running:
                return self._running[job_id]
            if job_id in self._completed:
                return self._completed[job_id]
            for job in self._pending:
                if job.job_id == job_id:
                    return job
        return None
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        with self._lock:
            return {
                'pending': len(self._pending),
                'running': len(self._running),
                'completed': len(self._completed),
                'max_concurrent': self.max_concurrent
            }


class AutoRetrainingPipeline:
    """Main auto-retraining pipeline orchestrator."""
    
    def __init__(self):
        self.drift_evaluator = DriftTriggerEvaluator()
        self.performance_monitor = PerformanceMonitor()
        self.ab_test_manager = ABTestManager()
        self.rollback_manager = RollbackManager()
        self.job_queue = TrainingJobQueue()
        
        self._model_configs: Dict[str, TrainingConfig] = {}
        self._schedules: Dict[str, RetrainingSchedule] = {}
        self._model_versions: Dict[str, List[ModelVersion]] = {}
        self._lock = threading.Lock()
        
        self.job_queue.register_callback(self._on_job_complete)
        
    def register_model(self, config: TrainingConfig,
                      schedule: RetrainingSchedule = None) -> None:
        """Register a model for auto-retraining."""
        with self._lock:
            self._model_configs[config.model_name] = config
            if schedule:
                self._schedules[schedule.schedule_id] = schedule
                
    def trigger_retraining(self, model_name: str,
                          trigger: RetrainingTrigger,
                          config_overrides: Dict[str, Any] = None) -> str:
        """
        Trigger model retraining.
        
        Returns:
            Job ID
        """
        with self._lock:
            base_config = self._model_configs.get(model_name)
            if not base_config:
                raise ValueError(f"Unknown model: {model_name}")
                
            config_dict = base_config.to_dict()
            if config_overrides:
                config_dict.update(config_overrides)
                
            config = TrainingConfig(**config_dict)
            
            job_id = hashlib.md5(
                f"{model_name}:{trigger.value}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16]
            
            current_version = self.rollback_manager.get_current_version(model_name)
            
            job = TrainingJob(
                job_id=job_id,
                config=config,
                trigger=trigger,
                status=TrainingStatus.PENDING,
                created_at=datetime.utcnow(),
                parent_model_version=current_version
            )
            
        self.job_queue.submit(job)
        return job_id
        
    def check_drift_trigger(self, model_name: str, drift_score: float,
                           drift_type: str) -> Optional[str]:
        """
        Check if drift should trigger retraining.
        
        Returns:
            Job ID if retraining triggered, None otherwise
        """
        should_retrain = self.drift_evaluator.record_drift(
            model_name, drift_score, drift_type
        )
        
        if should_retrain:
            logger.info(f"Drift-triggered retraining for {model_name}")
            return self.trigger_retraining(
                model_name, RetrainingTrigger.DRIFT_DETECTED
            )
        return None
        
    def check_performance_trigger(self, model_name: str, metric_name: str,
                                  value: float) -> Optional[str]:
        """
        Check if performance degradation should trigger retraining.
        
        Returns:
            Job ID if retraining triggered, None otherwise
        """
        degradation = self.performance_monitor.record_metric(
            model_name, metric_name, value
        )
        
        if degradation:
            logger.info(f"Performance-triggered retraining for {model_name}: {degradation}")
            return self.trigger_retraining(
                model_name, RetrainingTrigger.PERFORMANCE_DEGRADATION
            )
        return None
        
    def start_ab_test(self, model_name: str, challenger_version: str,
                     traffic_split: float = 0.1,
                     min_samples: int = 1000,
                     max_duration_hours: int = 24) -> str:
        """
        Start A/B test between champion and challenger.
        
        Returns:
            Test ID
        """
        champion_version = self.rollback_manager.get_current_version(model_name)
        if not champion_version:
            raise ValueError(f"No champion version for model: {model_name}")
            
        test_id = hashlib.md5(
            f"{model_name}:{champion_version}:{challenger_version}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        config = ABTestConfig(
            test_id=test_id,
            champion_version=champion_version,
            challenger_version=challenger_version,
            traffic_split=traffic_split,
            min_samples=min_samples,
            max_duration_hours=max_duration_hours,
            primary_metric='accuracy'
        )
        
        self.ab_test_manager.create_test(config)
        self.ab_test_manager.start_test(test_id)
        
        logger.info(f"A/B test started: {test_id}")
        return test_id
        
    def promote_model(self, model_name: str, version_id: str) -> None:
        """Promote a model version to champion."""
        with self._lock:
            if model_name not in self._model_versions:
                self._model_versions[model_name] = []
                
            for version in self._model_versions[model_name]:
                if version.version_id == version_id:
                    version.stage = ModelStage.CHAMPION
                    version.promoted_at = datetime.utcnow()
                    break
                    
        self.rollback_manager.set_current_version(model_name, version_id)
        logger.info(f"Model promoted: {model_name} -> {version_id}")
        
    def _on_job_complete(self, job: TrainingJob) -> None:
        """Handle job completion."""
        if job.status != TrainingStatus.COMPLETED:
            return
            
        version_id = hashlib.md5(
            f"{job.config.model_name}:{job.job_id}".encode()
        ).hexdigest()[:16]
        
        version = ModelVersion(
            version_id=version_id,
            model_name=job.config.model_name,
            stage=ModelStage.STAGING,
            training_job_id=job.job_id,
            metrics=job.metrics.val_metrics if job.metrics else {},
            created_at=datetime.utcnow(),
            artifact_path=job.model_artifact_path or ""
        )
        
        with self._lock:
            if job.config.model_name not in self._model_versions:
                self._model_versions[job.config.model_name] = []
            self._model_versions[job.config.model_name].append(version)
            
        logger.info(f"New model version created: {version_id}")
        
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        with self._lock:
            return self._model_versions.get(model_name, [])
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status."""
        return {
            'job_queue': self.job_queue.get_queue_status(),
            'active_ab_tests': len(self.ab_test_manager.get_active_tests()),
            'registered_models': list(self._model_configs.keys()),
            'schedules': len(self._schedules)
        }


def create_auto_retraining_pipeline() -> AutoRetrainingPipeline:
    """Factory function to create auto-retraining pipeline."""
    return AutoRetrainingPipeline()


def create_ab_test_manager() -> ABTestManager:
    """Factory function to create A/B test manager."""
    return ABTestManager()


def create_rollback_manager() -> RollbackManager:
    """Factory function to create rollback manager."""
    return RollbackManager()
