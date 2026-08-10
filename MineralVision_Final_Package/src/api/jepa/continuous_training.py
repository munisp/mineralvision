"""
Continuous Training Orchestrator for V-JEPA in MineralVision.

Enables automated continuous training with:
- Data drift detection triggers
- Scheduled retraining
- Checkpoint management and rollback
- Quality gates before model promotion
- Replay buffer to prevent catastrophic forgetting

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  ContinuousTrainingOrchestrator                                 │
│  - Monitors for new data / drift                                │
│  - Triggers training jobs                                       │
│  - Evaluates and promotes models                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Quality Gates                                                  │
│  - Embedding distribution checks                                │
│  - Validation set performance                                   │
│  - Anomaly rate stability                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Model Registry                                                 │
│  - Version tracking                                             │
│  - Promotion/rollback                                           │
│  - Lineage                                                      │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from api.jepa.continuous_training import (
        ContinuousTrainingOrchestrator,
        create_continuous_training_orchestrator,
    )
    
    # Create orchestrator
    orchestrator = create_continuous_training_orchestrator(
        lakehouse_store=store,
        model_registry_path="./models",
    )
    
    # Check if retraining is needed
    if orchestrator.should_retrain():
        result = orchestrator.run_training_cycle()
        if result.promoted:
            print(f"New model promoted: {result.model_version}")
    
    # Or run automated loop
    orchestrator.start_automated_loop(
        check_interval_hours=24,
        max_iterations=None,  # Run indefinitely
    )
"""

import json
import logging
import hashlib
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import math

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Reasons for triggering retraining."""
    SCHEDULED = "scheduled"
    DATA_VOLUME = "data_volume"
    EMBEDDING_DRIFT = "embedding_drift"
    ANOMALY_RATE_DRIFT = "anomaly_rate_drift"
    MANUAL = "manual"
    NEW_SENSOR = "new_sensor"
    NEW_SITE = "new_site"


class ModelStatus(Enum):
    """Status of a model version."""
    TRAINING = "training"
    EVALUATING = "evaluating"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class GateResult(Enum):
    """Result of a quality gate check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class DriftMetrics:
    """Metrics for detecting data/model drift."""
    embedding_mean_shift: float = 0.0
    embedding_std_shift: float = 0.0
    cosine_similarity_to_baseline: float = 1.0
    anomaly_rate_current: float = 0.0
    anomaly_rate_baseline: float = 0.0
    anomaly_rate_change: float = 0.0
    new_data_count: int = 0
    new_sites: List[str] = field(default_factory=list)
    new_sensors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_mean_shift": self.embedding_mean_shift,
            "embedding_std_shift": self.embedding_std_shift,
            "cosine_similarity_to_baseline": self.cosine_similarity_to_baseline,
            "anomaly_rate_current": self.anomaly_rate_current,
            "anomaly_rate_baseline": self.anomaly_rate_baseline,
            "anomaly_rate_change": self.anomaly_rate_change,
            "new_data_count": self.new_data_count,
            "new_sites": self.new_sites,
            "new_sensors": self.new_sensors,
        }


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    # Embedding distribution thresholds
    max_embedding_mean_shift: float = 0.5
    max_embedding_std_shift: float = 0.3
    min_cosine_similarity: float = 0.8
    
    # Anomaly rate thresholds
    max_anomaly_rate_change: float = 0.2  # 20% change
    
    # Training stability
    max_loss_increase: float = 0.1
    min_epochs_completed: int = 5
    
    # Validation performance
    min_retrieval_precision: float = 0.7
    max_validation_loss_increase: float = 0.15


@dataclass
class ModelVersion:
    """Represents a versioned model checkpoint."""
    version_id: str
    checkpoint_path: str
    checkpoint_hash: str
    
    # Training info
    training_run_id: str
    parent_version_id: Optional[str] = None
    dataset_manifest_path: Optional[str] = None
    
    # Metrics
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    embedding_stats: Optional[Dict[str, float]] = None
    
    # Status
    status: ModelStatus = ModelStatus.TRAINING
    promoted_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    git_sha: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "training_run_id": self.training_run_id,
            "parent_version_id": self.parent_version_id,
            "dataset_manifest_path": self.dataset_manifest_path,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "embedding_stats": self.embedding_stats,
            "status": self.status.value,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "created_at": self.created_at.isoformat(),
            "git_sha": self.git_sha,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(
            version_id=data["version_id"],
            checkpoint_path=data["checkpoint_path"],
            checkpoint_hash=data["checkpoint_hash"],
            training_run_id=data["training_run_id"],
            parent_version_id=data.get("parent_version_id"),
            dataset_manifest_path=data.get("dataset_manifest_path"),
            training_loss=data.get("training_loss"),
            validation_loss=data.get("validation_loss"),
            embedding_stats=data.get("embedding_stats"),
            status=ModelStatus(data.get("status", "training")),
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
            rolled_back_at=datetime.fromisoformat(data["rolled_back_at"]) if data.get("rolled_back_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            git_sha=data.get("git_sha"),
            config=data.get("config"),
        )


@dataclass
class TrainingCycleResult:
    """Result of a continuous training cycle."""
    success: bool
    model_version: Optional[ModelVersion] = None
    promoted: bool = False
    trigger: Optional[RetrainingTrigger] = None
    drift_metrics: Optional[DriftMetrics] = None
    gate_results: Dict[str, GateResult] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_version": self.model_version.to_dict() if self.model_version else None,
            "promoted": self.promoted,
            "trigger": self.trigger.value if self.trigger else None,
            "drift_metrics": self.drift_metrics.to_dict() if self.drift_metrics else None,
            "gate_results": {k: v.value for k, v in self.gate_results.items()},
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ContinuousTrainingConfig:
    """Configuration for continuous training."""
    # Retraining triggers
    min_new_samples_for_retrain: int = 1000
    max_days_between_retrains: int = 30
    embedding_drift_threshold: float = 0.3
    anomaly_rate_drift_threshold: float = 0.15
    
    # Training settings
    continuation_learning_rate: float = 1e-5  # Lower LR for continuation
    continuation_epochs: int = 10  # Fewer epochs for continuation
    replay_buffer_ratio: float = 0.3  # 30% old data mixed with new
    
    # Quality gates
    quality_gates: QualityGateConfig = field(default_factory=QualityGateConfig)
    
    # Model registry
    max_versions_to_keep: int = 10
    auto_promote: bool = False  # Require manual approval by default
    
    # Paths
    model_registry_path: str = "./model_registry"
    validation_set_path: Optional[str] = None


class ModelRegistry:
    """Registry for managing model versions."""
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.registry_path / "versions.json"
        self.current_file = self.registry_path / "current.json"
        self._versions: Dict[str, ModelVersion] = {}
        self._current_version_id: Optional[str] = None
        self._load()
    
    def _load(self) -> None:
        """Load registry from disk."""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                data = json.load(f)
                self._versions = {
                    k: ModelVersion.from_dict(v) for k, v in data.items()
                }
        
        if self.current_file.exists():
            with open(self.current_file, "r") as f:
                data = json.load(f)
                self._current_version_id = data.get("current_version_id")
    
    def _save(self) -> None:
        """Save registry to disk."""
        with open(self.versions_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._versions.items()}, f, indent=2)
        
        with open(self.current_file, "w") as f:
            json.dump({"current_version_id": self._current_version_id}, f, indent=2)
    
    def register_version(self, version: ModelVersion) -> None:
        """Register a new model version."""
        self._versions[version.version_id] = version
        self._save()
        logger.info(f"Registered model version: {version.version_id}")
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(version_id)
    
    def get_current_version(self) -> Optional[ModelVersion]:
        """Get the currently promoted model version."""
        if self._current_version_id:
            return self._versions.get(self._current_version_id)
        return None
    
    def promote_version(self, version_id: str) -> bool:
        """Promote a version to current production."""
        version = self._versions.get(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Update old current version status
        if self._current_version_id and self._current_version_id in self._versions:
            old_version = self._versions[self._current_version_id]
            if old_version.status == ModelStatus.PROMOTED:
                old_version.status = ModelStatus.CANDIDATE
        
        # Promote new version
        version.status = ModelStatus.PROMOTED
        version.promoted_at = datetime.utcnow()
        self._current_version_id = version_id
        self._save()
        logger.info(f"Promoted model version: {version_id}")
        return True
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a previous version."""
        version = self._versions.get(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Mark current as rolled back
        if self._current_version_id and self._current_version_id in self._versions:
            current = self._versions[self._current_version_id]
            current.status = ModelStatus.ROLLED_BACK
            current.rolled_back_at = datetime.utcnow()
        
        # Set new current
        version.status = ModelStatus.PROMOTED
        version.promoted_at = datetime.utcnow()
        self._current_version_id = version_id
        self._save()
        logger.info(f"Rolled back to model version: {version_id}")
        return True
    
    def list_versions(
        self,
        status: Optional[ModelStatus] = None,
        limit: int = 10,
    ) -> List[ModelVersion]:
        """List model versions."""
        versions = list(self._versions.values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by created_at descending
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return versions[:limit]
    
    def cleanup_old_versions(self, keep_count: int = 10) -> int:
        """Remove old versions, keeping the most recent ones."""
        versions = self.list_versions(limit=1000)
        
        # Always keep promoted and current versions
        to_keep = {self._current_version_id} if self._current_version_id else set()
        for v in versions:
            if v.status == ModelStatus.PROMOTED:
                to_keep.add(v.version_id)
        
        # Keep the most recent versions
        for v in versions[:keep_count]:
            to_keep.add(v.version_id)
        
        # Remove old versions
        removed = 0
        for version_id in list(self._versions.keys()):
            if version_id not in to_keep:
                version = self._versions[version_id]
                # Delete checkpoint file if it exists
                checkpoint_path = Path(version.checkpoint_path)
                if checkpoint_path.exists():
                    try:
                        if checkpoint_path.is_dir():
                            shutil.rmtree(checkpoint_path)
                        else:
                            checkpoint_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint: {e}")
                
                del self._versions[version_id]
                removed += 1
        
        if removed > 0:
            self._save()
            logger.info(f"Cleaned up {removed} old model versions")
        
        return removed


class DriftDetector:
    """Detects data and model drift."""
    
    def __init__(self, lakehouse_store: Any, baseline_stats: Optional[Dict[str, Any]] = None):
        self.lakehouse_store = lakehouse_store
        self.baseline_stats = baseline_stats or {}
    
    def compute_embedding_stats(
        self,
        embeddings: List[Any],
    ) -> Dict[str, float]:
        """Compute statistics over embeddings."""
        if not embeddings:
            return {"mean": 0.0, "std": 0.0, "norm_mean": 0.0}
        
        # Compute mean and std of embedding vectors
        all_vectors = [e.embedding_vector for e in embeddings if e.embedding_vector]
        
        if not all_vectors:
            return {"mean": 0.0, "std": 0.0, "norm_mean": 0.0}
        
        # Flatten and compute stats
        flat = []
        norms = []
        for vec in all_vectors:
            flat.extend(vec)
            norms.append(math.sqrt(sum(x * x for x in vec)))
        
        mean = sum(flat) / len(flat) if flat else 0.0
        variance = sum((x - mean) ** 2 for x in flat) / len(flat) if flat else 0.0
        std = math.sqrt(variance)
        norm_mean = sum(norms) / len(norms) if norms else 0.0
        
        return {"mean": mean, "std": std, "norm_mean": norm_mean}
    
    def compute_drift_metrics(
        self,
        since_timestamp: Optional[datetime] = None,
    ) -> DriftMetrics:
        """Compute drift metrics comparing recent data to baseline."""
        metrics = DriftMetrics()
        
        # Get recent embeddings
        recent_embeddings = self.lakehouse_store.read_embeddings(limit=1000)
        
        if since_timestamp:
            recent_embeddings = [
                e for e in recent_embeddings
                if e.created_at and e.created_at > since_timestamp
            ]
        
        metrics.new_data_count = len(recent_embeddings)
        
        # Compute current stats
        current_stats = self.compute_embedding_stats(recent_embeddings)
        
        # Compare to baseline
        if self.baseline_stats:
            baseline_mean = self.baseline_stats.get("mean", 0.0)
            baseline_std = self.baseline_stats.get("std", 1.0)
            
            if baseline_std > 0:
                metrics.embedding_mean_shift = abs(current_stats["mean"] - baseline_mean) / baseline_std
                metrics.embedding_std_shift = abs(current_stats["std"] - baseline_std) / baseline_std
        
        # Check for new sites/sensors
        sites = set()
        sensors = set()
        for e in recent_embeddings:
            if e.site:
                sites.add(e.site)
            if e.sensor:
                sensors.add(e.sensor)
        
        baseline_sites = set(self.baseline_stats.get("sites", []))
        baseline_sensors = set(self.baseline_stats.get("sensors", []))
        
        metrics.new_sites = list(sites - baseline_sites)
        metrics.new_sensors = list(sensors - baseline_sensors)
        
        # Compute anomaly rate
        findings = self.lakehouse_store.read_findings(limit=1000)
        if findings:
            anomaly_findings = [f for f in findings if f.finding_type == "anomaly"]
            metrics.anomaly_rate_current = len(anomaly_findings) / len(findings) if findings else 0.0
            metrics.anomaly_rate_baseline = self.baseline_stats.get("anomaly_rate", 0.0)
            metrics.anomaly_rate_change = abs(
                metrics.anomaly_rate_current - metrics.anomaly_rate_baseline
            )
        
        return metrics
    
    def update_baseline(self) -> Dict[str, Any]:
        """Update baseline statistics from current data."""
        embeddings = self.lakehouse_store.read_embeddings(limit=5000)
        stats = self.compute_embedding_stats(embeddings)
        
        # Collect sites and sensors
        sites = set()
        sensors = set()
        for e in embeddings:
            if e.site:
                sites.add(e.site)
            if e.sensor:
                sensors.add(e.sensor)
        
        stats["sites"] = list(sites)
        stats["sensors"] = list(sensors)
        
        # Compute anomaly rate
        findings = self.lakehouse_store.read_findings(limit=1000)
        if findings:
            anomaly_findings = [f for f in findings if f.finding_type == "anomaly"]
            stats["anomaly_rate"] = len(anomaly_findings) / len(findings)
        else:
            stats["anomaly_rate"] = 0.0
        
        self.baseline_stats = stats
        return stats


class QualityGateEvaluator:
    """Evaluates quality gates for model promotion."""
    
    def __init__(self, config: QualityGateConfig):
        self.config = config
    
    def evaluate_embedding_distribution(
        self,
        current_stats: Dict[str, float],
        baseline_stats: Dict[str, float],
    ) -> Tuple[GateResult, str]:
        """Check embedding distribution hasn't shifted too much."""
        if not baseline_stats:
            return GateResult.WARNING, "No baseline stats available"
        
        baseline_mean = baseline_stats.get("mean", 0.0)
        baseline_std = baseline_stats.get("std", 1.0)
        
        if baseline_std == 0:
            return GateResult.WARNING, "Baseline std is zero"
        
        mean_shift = abs(current_stats.get("mean", 0.0) - baseline_mean) / baseline_std
        std_shift = abs(current_stats.get("std", 0.0) - baseline_std) / baseline_std
        
        if mean_shift > self.config.max_embedding_mean_shift:
            return GateResult.FAILED, f"Mean shift {mean_shift:.3f} exceeds threshold {self.config.max_embedding_mean_shift}"
        
        if std_shift > self.config.max_embedding_std_shift:
            return GateResult.FAILED, f"Std shift {std_shift:.3f} exceeds threshold {self.config.max_embedding_std_shift}"
        
        return GateResult.PASSED, f"Mean shift: {mean_shift:.3f}, Std shift: {std_shift:.3f}"
    
    def evaluate_training_stability(
        self,
        training_loss: float,
        baseline_loss: Optional[float],
        epochs_completed: int,
    ) -> Tuple[GateResult, str]:
        """Check training completed stably."""
        if epochs_completed < self.config.min_epochs_completed:
            return GateResult.FAILED, f"Only {epochs_completed} epochs completed, need {self.config.min_epochs_completed}"
        
        if baseline_loss and training_loss > baseline_loss * (1 + self.config.max_loss_increase):
            return GateResult.FAILED, f"Training loss {training_loss:.4f} increased too much from baseline {baseline_loss:.4f}"
        
        return GateResult.PASSED, f"Training loss: {training_loss:.4f}, Epochs: {epochs_completed}"
    
    def evaluate_validation_performance(
        self,
        validation_loss: Optional[float],
        baseline_validation_loss: Optional[float],
    ) -> Tuple[GateResult, str]:
        """Check validation performance hasn't degraded."""
        if validation_loss is None:
            return GateResult.WARNING, "No validation loss available"
        
        if baseline_validation_loss is None:
            return GateResult.PASSED, f"Validation loss: {validation_loss:.4f} (no baseline)"
        
        if validation_loss > baseline_validation_loss * (1 + self.config.max_validation_loss_increase):
            return GateResult.FAILED, f"Validation loss {validation_loss:.4f} degraded from baseline {baseline_validation_loss:.4f}"
        
        return GateResult.PASSED, f"Validation loss: {validation_loss:.4f} (baseline: {baseline_validation_loss:.4f})"
    
    def evaluate_all_gates(
        self,
        model_version: ModelVersion,
        baseline_stats: Dict[str, Any],
        baseline_version: Optional[ModelVersion],
    ) -> Dict[str, Tuple[GateResult, str]]:
        """Evaluate all quality gates."""
        results = {}
        
        # Embedding distribution gate
        current_stats = model_version.embedding_stats or {}
        results["embedding_distribution"] = self.evaluate_embedding_distribution(
            current_stats, baseline_stats
        )
        
        # Training stability gate
        baseline_loss = baseline_version.training_loss if baseline_version else None
        results["training_stability"] = self.evaluate_training_stability(
            model_version.training_loss or 0.0,
            baseline_loss,
            model_version.config.get("epochs_completed", 0) if model_version.config else 0,
        )
        
        # Validation performance gate
        baseline_val_loss = baseline_version.validation_loss if baseline_version else None
        results["validation_performance"] = self.evaluate_validation_performance(
            model_version.validation_loss,
            baseline_val_loss,
        )
        
        return results


class ReplayBuffer:
    """Manages replay buffer for preventing catastrophic forgetting."""
    
    def __init__(self, lakehouse_store: Any, buffer_ratio: float = 0.3):
        self.lakehouse_store = lakehouse_store
        self.buffer_ratio = buffer_ratio
    
    def create_mixed_manifest(
        self,
        new_data_manifest_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Create a training manifest mixing new data with replay buffer."""
        # Load new data manifest
        with open(new_data_manifest_path, "r") as f:
            new_manifest = json.load(f)
        
        new_samples = new_manifest.get("samples", [])
        num_new = len(new_samples)
        
        # Calculate replay buffer size
        num_replay = int(num_new * self.buffer_ratio / (1 - self.buffer_ratio))
        
        # Get historical embeddings for replay
        historical = self.lakehouse_store.read_embeddings(limit=num_replay * 2)
        
        # Sample from historical (exclude samples already in new manifest)
        new_paths = {s.get("source_path") for s in new_samples}
        replay_candidates = [
            e for e in historical
            if e.source_path not in new_paths
        ]
        
        # Random sample (simplified - in production use stratified sampling)
        import random
        replay_samples = random.sample(
            replay_candidates,
            min(num_replay, len(replay_candidates))
        )
        
        # Create mixed manifest
        mixed_manifest = {
            "manifest_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "type": "mixed_replay",
            "new_samples_count": num_new,
            "replay_samples_count": len(replay_samples),
            "replay_ratio": self.buffer_ratio,
            "samples": new_samples + [
                {
                    "embedding_id": e.embedding_id,
                    "source_path": e.source_path,
                    "source_type": e.source_type,
                    "site": e.site,
                    "sensor": e.sensor,
                    "is_replay": True,
                }
                for e in replay_samples
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(mixed_manifest, f, indent=2)
        
        return mixed_manifest


class ContinuousTrainingOrchestrator:
    """Orchestrates continuous training for V-JEPA."""
    
    def __init__(
        self,
        lakehouse_store: Any,
        config: ContinuousTrainingConfig,
        model_registry: Optional[ModelRegistry] = None,
        pretraining_runner: Optional[Any] = None,
    ):
        self.lakehouse_store = lakehouse_store
        self.config = config
        
        self.model_registry = model_registry or ModelRegistry(config.model_registry_path)
        self.drift_detector = DriftDetector(lakehouse_store)
        self.gate_evaluator = QualityGateEvaluator(config.quality_gates)
        self.replay_buffer = ReplayBuffer(lakehouse_store, config.replay_buffer_ratio)
        self.pretraining_runner = pretraining_runner
        
        self._last_training_time: Optional[datetime] = None
        self._running = False
    
    def should_retrain(self) -> Tuple[bool, Optional[RetrainingTrigger], Optional[DriftMetrics]]:
        """Check if retraining should be triggered."""
        # Check scheduled trigger
        if self._last_training_time:
            days_since = (datetime.utcnow() - self._last_training_time).days
            if days_since >= self.config.max_days_between_retrains:
                return True, RetrainingTrigger.SCHEDULED, None
        
        # Compute drift metrics
        drift_metrics = self.drift_detector.compute_drift_metrics(
            since_timestamp=self._last_training_time
        )
        
        # Check data volume trigger
        if drift_metrics.new_data_count >= self.config.min_new_samples_for_retrain:
            return True, RetrainingTrigger.DATA_VOLUME, drift_metrics
        
        # Check embedding drift trigger
        if drift_metrics.embedding_mean_shift >= self.config.embedding_drift_threshold:
            return True, RetrainingTrigger.EMBEDDING_DRIFT, drift_metrics
        
        # Check anomaly rate drift trigger
        if drift_metrics.anomaly_rate_change >= self.config.anomaly_rate_drift_threshold:
            return True, RetrainingTrigger.ANOMALY_RATE_DRIFT, drift_metrics
        
        # Check new site/sensor trigger
        if drift_metrics.new_sites:
            return True, RetrainingTrigger.NEW_SITE, drift_metrics
        
        if drift_metrics.new_sensors:
            return True, RetrainingTrigger.NEW_SENSOR, drift_metrics
        
        return False, None, drift_metrics
    
    def run_training_cycle(
        self,
        trigger: Optional[RetrainingTrigger] = None,
        force: bool = False,
    ) -> TrainingCycleResult:
        """Run a complete training cycle."""
        start_time = time.time()
        
        # Check if we should retrain
        if not force:
            should_train, detected_trigger, drift_metrics = self.should_retrain()
            if not should_train:
                return TrainingCycleResult(
                    success=True,
                    promoted=False,
                    trigger=None,
                    drift_metrics=drift_metrics,
                    duration_seconds=time.time() - start_time,
                )
            trigger = trigger or detected_trigger
        else:
            trigger = trigger or RetrainingTrigger.MANUAL
            drift_metrics = self.drift_detector.compute_drift_metrics()
        
        logger.info(f"Starting training cycle, trigger: {trigger}")
        
        try:
            # Get current model version for continuation
            current_version = self.model_registry.get_current_version()
            resume_checkpoint = current_version.checkpoint_path if current_version else None
            
            # Create new version ID
            version_id = f"v-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            checkpoint_dir = Path(self.config.model_registry_path) / "checkpoints" / version_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create training manifest with replay buffer
            from .lakehouse_integration import TrainingDatasetManager
            dataset_manager = TrainingDatasetManager(self.lakehouse_store)
            
            new_manifest_path = str(checkpoint_dir / "new_data_manifest.json")
            dataset_manager.create_training_manifest(output_path=new_manifest_path)
            
            mixed_manifest_path = str(checkpoint_dir / "mixed_manifest.json")
            self.replay_buffer.create_mixed_manifest(
                new_manifest_path,
                mixed_manifest_path,
            )
            
            # Create model version record
            model_version = ModelVersion(
                version_id=version_id,
                checkpoint_path=str(checkpoint_dir / "checkpoint.pt"),
                checkpoint_hash="",  # Will be computed after training
                training_run_id=str(uuid.uuid4()),
                parent_version_id=current_version.version_id if current_version else None,
                dataset_manifest_path=mixed_manifest_path,
                status=ModelStatus.TRAINING,
                config={
                    "learning_rate": self.config.continuation_learning_rate,
                    "epochs": self.config.continuation_epochs,
                    "replay_ratio": self.config.replay_buffer_ratio,
                    "resume_from": resume_checkpoint,
                },
            )
            
            self.model_registry.register_version(model_version)
            
            # Run training — always through the real pretraining runner
            # (which is torch_core-backed and raises JEPAUnavailableError
            # when the backend is missing). The previous "simulated
            # training" branch (fabricated losses + dummy checkpoint) has
            # been removed: retraining must be real or fail LOUDLY.
            if self.pretraining_runner is None:
                from .vjepa_integration import JEPAUnavailableError

                model_version.status = ModelStatus.FAILED
                self.model_registry.register_version(model_version)
                raise JEPAUnavailableError(
                    "Continuous training requires a real pretraining runner "
                    "(PretrainingRunner -> VJEPAPretrainer -> torch_core), "
                    "but none was provided. Simulated training with "
                    "fabricated losses has been removed — no fake model "
                    "versions will be produced."
                )

            from .pretraining_pipeline import PretrainingJob
            
            from .vjepa_integration import VJEPAConfig

            job_config = VJEPAConfig(
                total_epochs=self.config.continuation_epochs,
                learning_rate=self.config.continuation_learning_rate,
                batch_size=32,
            )

            job = self.pretraining_runner.create_job(
                job_name=f"continuous_training_{version_id}",
                config=job_config,
                data_sources={},
                checkpoint_dir=str(checkpoint_dir),
                resume_from=resume_checkpoint,
            )
            
            result = self.pretraining_runner.run_job(job.job_id)
            model_version.training_loss = result.get("final_loss", 0.0)
            model_version.config["epochs_completed"] = result.get("epochs_completed", 0)
            
            # Compute checkpoint hash
            if Path(model_version.checkpoint_path).exists():
                with open(model_version.checkpoint_path, "rb") as f:
                    model_version.checkpoint_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Compute embedding stats on new model
            model_version.embedding_stats = self.drift_detector.compute_embedding_stats(
                self.lakehouse_store.read_embeddings(limit=1000)
            )
            
            model_version.status = ModelStatus.EVALUATING
            self.model_registry.register_version(model_version)
            
            # Evaluate quality gates
            baseline_stats = self.drift_detector.baseline_stats
            gate_results = self.gate_evaluator.evaluate_all_gates(
                model_version,
                baseline_stats,
                current_version,
            )
            
            # Check if all gates passed
            all_passed = all(
                result[0] in (GateResult.PASSED, GateResult.WARNING)
                for result in gate_results.values()
            )
            
            gate_result_dict = {k: v[0] for k, v in gate_results.items()}
            
            if all_passed:
                model_version.status = ModelStatus.CANDIDATE
                
                # Auto-promote if configured
                if self.config.auto_promote:
                    self.model_registry.promote_version(version_id)
                    model_version.status = ModelStatus.PROMOTED
                    logger.info(f"Auto-promoted model version: {version_id}")
                    
                    # Update baseline after promotion
                    self.drift_detector.update_baseline()
            else:
                model_version.status = ModelStatus.FAILED
                failed_gates = [k for k, v in gate_results.items() if v[0] == GateResult.FAILED]
                logger.warning(f"Model failed quality gates: {failed_gates}")
            
            self.model_registry.register_version(model_version)
            self._last_training_time = datetime.utcnow()
            
            # Cleanup old versions
            self.model_registry.cleanup_old_versions(self.config.max_versions_to_keep)
            
            return TrainingCycleResult(
                success=True,
                model_version=model_version,
                promoted=model_version.status == ModelStatus.PROMOTED,
                trigger=trigger,
                drift_metrics=drift_metrics,
                gate_results=gate_result_dict,
                duration_seconds=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"Training cycle failed: {e}")
            return TrainingCycleResult(
                success=False,
                error_message=str(e),
                trigger=trigger,
                duration_seconds=time.time() - start_time,
            )
    
    def promote_candidate(self, version_id: str) -> bool:
        """Manually promote a candidate model."""
        version = self.model_registry.get_version(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        if version.status != ModelStatus.CANDIDATE:
            logger.error(f"Version {version_id} is not a candidate (status: {version.status})")
            return False
        
        success = self.model_registry.promote_version(version_id)
        if success:
            self.drift_detector.update_baseline()
        
        return success
    
    def rollback(self, to_version_id: Optional[str] = None) -> bool:
        """Rollback to a previous model version."""
        if to_version_id:
            return self.model_registry.rollback_to_version(to_version_id)
        
        # Find the previous promoted version
        versions = self.model_registry.list_versions(limit=10)
        current = self.model_registry.get_current_version()
        
        for v in versions:
            if v.version_id != current.version_id and v.status in (
                ModelStatus.CANDIDATE, ModelStatus.PROMOTED
            ):
                return self.model_registry.rollback_to_version(v.version_id)
        
        logger.error("No previous version available for rollback")
        return False
    
    def start_automated_loop(
        self,
        check_interval_hours: float = 24,
        max_iterations: Optional[int] = None,
        callback: Optional[Callable[[TrainingCycleResult], None]] = None,
    ) -> None:
        """Start automated continuous training loop."""
        self._running = True
        iteration = 0
        
        logger.info(f"Starting automated training loop (interval: {check_interval_hours}h)")
        
        while self._running:
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations})")
                break
            
            # Check and run training if needed
            result = self.run_training_cycle()
            
            if callback:
                callback(result)
            
            if result.promoted:
                logger.info(f"New model promoted: {result.model_version.version_id}")
            
            iteration += 1
            
            # Wait for next check
            if self._running:
                time.sleep(check_interval_hours * 3600)
        
        logger.info("Automated training loop stopped")
    
    def stop_automated_loop(self) -> None:
        """Stop the automated training loop."""
        self._running = False


# Factory functions

def create_continuous_training_orchestrator(
    lakehouse_store: Any,
    model_registry_path: str = "./model_registry",
    auto_promote: bool = False,
    replay_buffer_ratio: float = 0.3,
    continuation_epochs: int = 10,
    continuation_learning_rate: float = 1e-5,
) -> ContinuousTrainingOrchestrator:
    """Create a continuous training orchestrator."""
    config = ContinuousTrainingConfig(
        model_registry_path=model_registry_path,
        auto_promote=auto_promote,
        replay_buffer_ratio=replay_buffer_ratio,
        continuation_epochs=continuation_epochs,
        continuation_learning_rate=continuation_learning_rate,
    )
    return ContinuousTrainingOrchestrator(lakehouse_store, config)


def create_model_registry(
    registry_path: str = "./model_registry",
) -> ModelRegistry:
    """Create a model registry."""
    return ModelRegistry(registry_path)
