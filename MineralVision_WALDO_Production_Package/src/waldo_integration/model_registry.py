"""
Model Registry and Artifact Management
======================================

Production-grade model management with:
- Model versioning and lifecycle management
- Artifact storage with checksums
- Model metadata and performance tracking
- A/B testing support
- Deployment management
"""

import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Model status."""
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time_ms: float = 0.0
    throughput_fps: float = 0.0
    model_size_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArtifact:
    """Model artifact information."""
    artifact_id: str
    artifact_type: str  # 'weights', 'config', 'onnx', 'tensorrt', 'metadata'
    file_path: str
    file_size: int
    checksum: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_name: str
    version: str
    description: str = ""
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: ModelStatus = ModelStatus.TRAINING
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    artifacts: List[ModelArtifact] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    input_size: Tuple[int, int] = (640, 640)
    framework: str = "pytorch"
    architecture: str = "yolov8"
    training_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'version': self.version,
            'description': self.description,
            'stage': self.stage.value,
            'status': self.status.value,
            'metrics': self.metrics.to_dict(),
            'artifacts': [a.to_dict() for a in self.artifacts],
            'class_names': self.class_names,
            'input_size': self.input_size,
            'framework': self.framework,
            'architecture': self.architecture,
            'training_config': self.training_config,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by,
            'parent_version': self.parent_version
        }
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelVersion':
        metrics = ModelMetrics(**d.get('metrics', {}))
        artifacts = [ModelArtifact(**a) for a in d.get('artifacts', [])]
        
        return cls(
            version_id=d['version_id'],
            model_name=d['model_name'],
            version=d['version'],
            description=d.get('description', ''),
            stage=ModelStage(d.get('stage', 'development')),
            status=ModelStatus(d.get('status', 'training')),
            metrics=metrics,
            artifacts=artifacts,
            class_names=d.get('class_names', []),
            input_size=tuple(d.get('input_size', (640, 640))),
            framework=d.get('framework', 'pytorch'),
            architecture=d.get('architecture', 'yolov8'),
            training_config=d.get('training_config', {}),
            tags=d.get('tags', []),
            created_at=datetime.fromisoformat(d['created_at']) if 'created_at' in d else datetime.now(),
            updated_at=datetime.fromisoformat(d['updated_at']) if 'updated_at' in d else datetime.now(),
            created_by=d.get('created_by', ''),
            parent_version=d.get('parent_version')
        )


class ArtifactStorage(ABC):
    """Abstract base class for artifact storage."""
    
    @abstractmethod
    def save(self, source_path: str, artifact_id: str) -> str:
        """Save artifact and return storage path."""
        pass
    
    @abstractmethod
    def load(self, artifact_id: str, dest_path: str) -> str:
        """Load artifact to destination path."""
        pass
    
    @abstractmethod
    def delete(self, artifact_id: str) -> bool:
        """Delete artifact."""
        pass
    
    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        pass


class LocalArtifactStorage(ArtifactStorage):
    """Local filesystem artifact storage."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, source_path: str, artifact_id: str) -> str:
        """Save artifact to local storage."""
        dest_path = self.base_path / artifact_id
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if os.path.isfile(source_path):
            shutil.copy2(source_path, dest_path)
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, dest_path)
        else:
            raise ValueError(f"Source path does not exist: {source_path}")
        
        return str(dest_path)
    
    def load(self, artifact_id: str, dest_path: str) -> str:
        """Load artifact from local storage."""
        source_path = self.base_path / artifact_id
        
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        
        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
        else:
            shutil.copytree(source_path, dest_path)
        
        return dest_path
    
    def delete(self, artifact_id: str) -> bool:
        """Delete artifact from local storage."""
        path = self.base_path / artifact_id
        
        if not path.exists():
            return False
        
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        
        return True
    
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        return (self.base_path / artifact_id).exists()


class S3ArtifactStorage(ArtifactStorage):
    """S3-compatible artifact storage."""
    
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint_url = endpoint_url
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('s3', endpoint_url=self.endpoint_url)
            except ImportError:
                raise ImportError("boto3 required for S3 storage")
        return self._client
    
    def _get_key(self, artifact_id: str) -> str:
        return f"{self.prefix}/{artifact_id}" if self.prefix else artifact_id
    
    def save(self, source_path: str, artifact_id: str) -> str:
        """Upload artifact to S3."""
        client = self._get_client()
        key = self._get_key(artifact_id)
        
        client.upload_file(source_path, self.bucket, key)
        
        return f"s3://{self.bucket}/{key}"
    
    def load(self, artifact_id: str, dest_path: str) -> str:
        """Download artifact from S3."""
        client = self._get_client()
        key = self._get_key(artifact_id)
        
        client.download_file(self.bucket, key, dest_path)
        
        return dest_path
    
    def delete(self, artifact_id: str) -> bool:
        """Delete artifact from S3."""
        client = self._get_client()
        key = self._get_key(artifact_id)
        
        try:
            client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
    
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists in S3."""
        client = self._get_client()
        key = self._get_key(artifact_id)
        
        try:
            client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


class ModelRegistry:
    """
    Model registry for versioning and managing detection models.
    """
    
    def __init__(self, storage_path: str, artifact_storage: Optional[ArtifactStorage] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.artifact_storage = artifact_storage or LocalArtifactStorage(
            str(self.storage_path / "artifacts")
        )
        
        self.registry_file = self.storage_path / "registry.json"
        self._lock = threading.Lock()
        
        self._models: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            for model_name, versions in data.items():
                self._models[model_name] = {}
                for version_id, version_data in versions.items():
                    self._models[model_name][version_id] = ModelVersion.from_dict(version_data)
    
    def _save_registry(self):
        """Save registry to disk."""
        data = {}
        for model_name, versions in self._models.items():
            data[model_name] = {}
            for version_id, version in versions.items():
                data[model_name][version_id] = version.to_dict()
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _generate_version_id(self, model_name: str, version: str) -> str:
        """Generate unique version ID."""
        return f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def register_model(self, model_name: str, version: str,
                      weights_path: str,
                      class_names: List[str],
                      description: str = "",
                      metrics: Optional[ModelMetrics] = None,
                      training_config: Optional[Dict] = None,
                      tags: Optional[List[str]] = None,
                      created_by: str = "",
                      parent_version: Optional[str] = None) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            version: Version string (e.g., "1.0.0")
            weights_path: Path to model weights file
            class_names: List of class names
            description: Model description
            metrics: Performance metrics
            training_config: Training configuration
            tags: Model tags
            created_by: Creator identifier
            parent_version: Parent version ID for lineage
            
        Returns:
            Registered ModelVersion
        """
        with self._lock:
            version_id = self._generate_version_id(model_name, version)
            
            # Compute checksum and get file size
            checksum = self._compute_checksum(weights_path)
            file_size = os.path.getsize(weights_path)
            
            # Store artifact
            artifact_id = f"{version_id}/weights.pt"
            stored_path = self.artifact_storage.save(weights_path, artifact_id)
            
            # Create artifact record
            artifact = ModelArtifact(
                artifact_id=artifact_id,
                artifact_type='weights',
                file_path=stored_path,
                file_size=file_size,
                checksum=checksum
            )
            
            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                version=version,
                description=description,
                stage=ModelStage.DEVELOPMENT,
                status=ModelStatus.READY,
                metrics=metrics or ModelMetrics(model_size_mb=file_size / (1024 * 1024)),
                artifacts=[artifact],
                class_names=class_names,
                training_config=training_config or {},
                tags=tags or [],
                created_by=created_by,
                parent_version=parent_version
            )
            
            # Add to registry
            if model_name not in self._models:
                self._models[model_name] = {}
            
            self._models[model_name][version_id] = model_version
            self._save_registry()
            
            logger.info(f"Registered model {model_name} version {version} (ID: {version_id})")
            
            return model_version
    
    def add_artifact(self, version_id: str, artifact_path: str,
                    artifact_type: str) -> ModelArtifact:
        """
        Add additional artifact to a model version.
        
        Args:
            version_id: Model version ID
            artifact_path: Path to artifact file
            artifact_type: Type of artifact ('onnx', 'tensorrt', 'config', etc.)
            
        Returns:
            Created ModelArtifact
        """
        with self._lock:
            # Find model version
            model_version = self._find_version(version_id)
            if model_version is None:
                raise ValueError(f"Model version not found: {version_id}")
            
            # Compute checksum and get file size
            checksum = self._compute_checksum(artifact_path)
            file_size = os.path.getsize(artifact_path)
            
            # Store artifact
            ext = os.path.splitext(artifact_path)[1]
            artifact_id = f"{version_id}/{artifact_type}{ext}"
            stored_path = self.artifact_storage.save(artifact_path, artifact_id)
            
            # Create artifact record
            artifact = ModelArtifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                file_path=stored_path,
                file_size=file_size,
                checksum=checksum
            )
            
            model_version.artifacts.append(artifact)
            model_version.updated_at = datetime.now()
            self._save_registry()
            
            return artifact
    
    def update_metrics(self, version_id: str, metrics: ModelMetrics):
        """Update model metrics."""
        with self._lock:
            model_version = self._find_version(version_id)
            if model_version is None:
                raise ValueError(f"Model version not found: {version_id}")
            
            model_version.metrics = metrics
            model_version.updated_at = datetime.now()
            self._save_registry()
    
    def transition_stage(self, version_id: str, stage: ModelStage):
        """Transition model to a new stage."""
        with self._lock:
            model_version = self._find_version(version_id)
            if model_version is None:
                raise ValueError(f"Model version not found: {version_id}")
            
            old_stage = model_version.stage
            model_version.stage = stage
            model_version.updated_at = datetime.now()
            self._save_registry()
            
            logger.info(f"Transitioned {version_id} from {old_stage.value} to {stage.value}")
    
    def get_model(self, model_name: str, version: Optional[str] = None,
                 stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """
        Get a model version.
        
        Args:
            model_name: Model name
            version: Specific version (optional)
            stage: Get latest model in stage (optional)
            
        Returns:
            ModelVersion or None
        """
        if model_name not in self._models:
            return None
        
        versions = self._models[model_name]
        
        if version:
            # Find specific version
            for v in versions.values():
                if v.version == version:
                    return v
            return None
        
        if stage:
            # Find latest in stage
            stage_versions = [v for v in versions.values() if v.stage == stage]
            if not stage_versions:
                return None
            return max(stage_versions, key=lambda x: x.created_at)
        
        # Return latest
        if not versions:
            return None
        return max(versions.values(), key=lambda x: x.created_at)
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the production model for a given name."""
        return self.get_model(model_name, stage=ModelStage.PRODUCTION)
    
    def list_models(self) -> List[str]:
        """List all model names."""
        return list(self._models.keys())
    
    def list_versions(self, model_name: str,
                     stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List all versions of a model."""
        if model_name not in self._models:
            return []
        
        versions = list(self._models[model_name].values())
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def load_model_weights(self, version_id: str, dest_path: str) -> str:
        """
        Load model weights to destination path.
        
        Args:
            version_id: Model version ID
            dest_path: Destination path for weights
            
        Returns:
            Path to loaded weights
        """
        model_version = self._find_version(version_id)
        if model_version is None:
            raise ValueError(f"Model version not found: {version_id}")
        
        # Find weights artifact
        weights_artifact = None
        for artifact in model_version.artifacts:
            if artifact.artifact_type == 'weights':
                weights_artifact = artifact
                break
        
        if weights_artifact is None:
            raise ValueError(f"No weights artifact found for {version_id}")
        
        # Verify checksum
        loaded_path = self.artifact_storage.load(weights_artifact.artifact_id, dest_path)
        
        actual_checksum = self._compute_checksum(loaded_path)
        if actual_checksum != weights_artifact.checksum:
            raise ValueError(f"Checksum mismatch for {version_id}")
        
        return loaded_path
    
    def delete_version(self, version_id: str) -> bool:
        """Delete a model version."""
        with self._lock:
            model_version = self._find_version(version_id)
            if model_version is None:
                return False
            
            # Delete artifacts
            for artifact in model_version.artifacts:
                self.artifact_storage.delete(artifact.artifact_id)
            
            # Remove from registry
            del self._models[model_version.model_name][version_id]
            
            if not self._models[model_version.model_name]:
                del self._models[model_version.model_name]
            
            self._save_registry()
            
            return True
    
    def _find_version(self, version_id: str) -> Optional[ModelVersion]:
        """Find model version by ID."""
        for versions in self._models.values():
            if version_id in versions:
                return versions[version_id]
        return None
    
    def compare_versions(self, version_id_a: str,
                        version_id_b: str) -> Dict[str, Any]:
        """Compare two model versions."""
        version_a = self._find_version(version_id_a)
        version_b = self._find_version(version_id_b)
        
        if version_a is None or version_b is None:
            raise ValueError("One or both versions not found")
        
        metrics_a = version_a.metrics
        metrics_b = version_b.metrics
        
        return {
            'version_a': version_id_a,
            'version_b': version_id_b,
            'metrics_comparison': {
                'mAP50': {
                    'a': metrics_a.mAP50,
                    'b': metrics_b.mAP50,
                    'diff': metrics_b.mAP50 - metrics_a.mAP50
                },
                'mAP50_95': {
                    'a': metrics_a.mAP50_95,
                    'b': metrics_b.mAP50_95,
                    'diff': metrics_b.mAP50_95 - metrics_a.mAP50_95
                },
                'precision': {
                    'a': metrics_a.precision,
                    'b': metrics_b.precision,
                    'diff': metrics_b.precision - metrics_a.precision
                },
                'recall': {
                    'a': metrics_a.recall,
                    'b': metrics_b.recall,
                    'diff': metrics_b.recall - metrics_a.recall
                },
                'inference_time_ms': {
                    'a': metrics_a.inference_time_ms,
                    'b': metrics_b.inference_time_ms,
                    'diff': metrics_b.inference_time_ms - metrics_a.inference_time_ms
                }
            },
            'class_changes': {
                'added': list(set(version_b.class_names) - set(version_a.class_names)),
                'removed': list(set(version_a.class_names) - set(version_b.class_names))
            }
        }


class ABTestManager:
    """
    A/B testing manager for model comparison.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_tests: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def create_test(self, test_name: str, model_a_id: str, model_b_id: str,
                   traffic_split: float = 0.5) -> Dict[str, Any]:
        """
        Create an A/B test.
        
        Args:
            test_name: Name of the test
            model_a_id: Control model version ID
            model_b_id: Treatment model version ID
            traffic_split: Fraction of traffic to model B (0.0-1.0)
            
        Returns:
            Test configuration
        """
        with self._lock:
            test = {
                'test_name': test_name,
                'model_a_id': model_a_id,
                'model_b_id': model_b_id,
                'traffic_split': traffic_split,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'results': {
                    'model_a': {'requests': 0, 'latency_sum': 0, 'errors': 0},
                    'model_b': {'requests': 0, 'latency_sum': 0, 'errors': 0}
                }
            }
            
            self.active_tests[test_name] = test
            
            return test
    
    def get_model_for_request(self, test_name: str) -> Tuple[str, str]:
        """
        Get model ID for a request based on traffic split.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Tuple of (model_id, variant) where variant is 'a' or 'b'
        """
        import random
        
        test = self.active_tests.get(test_name)
        if test is None or test['status'] != 'active':
            raise ValueError(f"Test not found or not active: {test_name}")
        
        if random.random() < test['traffic_split']:
            return test['model_b_id'], 'b'
        else:
            return test['model_a_id'], 'a'
    
    def record_result(self, test_name: str, variant: str,
                     latency_ms: float, error: bool = False):
        """Record result for a request."""
        with self._lock:
            test = self.active_tests.get(test_name)
            if test is None:
                return
            
            key = f'model_{variant}'
            test['results'][key]['requests'] += 1
            test['results'][key]['latency_sum'] += latency_ms
            if error:
                test['results'][key]['errors'] += 1
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get current test results."""
        test = self.active_tests.get(test_name)
        if test is None:
            raise ValueError(f"Test not found: {test_name}")
        
        results = test['results']
        
        def calc_stats(data):
            requests = data['requests']
            if requests == 0:
                return {'requests': 0, 'avg_latency_ms': 0, 'error_rate': 0}
            return {
                'requests': requests,
                'avg_latency_ms': data['latency_sum'] / requests,
                'error_rate': data['errors'] / requests
            }
        
        return {
            'test_name': test_name,
            'status': test['status'],
            'model_a': calc_stats(results['model_a']),
            'model_b': calc_stats(results['model_b'])
        }
    
    def conclude_test(self, test_name: str, winner: str) -> Dict[str, Any]:
        """
        Conclude an A/B test.
        
        Args:
            test_name: Name of the test
            winner: Winner variant ('a' or 'b')
            
        Returns:
            Final test results
        """
        with self._lock:
            test = self.active_tests.get(test_name)
            if test is None:
                raise ValueError(f"Test not found: {test_name}")
            
            test['status'] = 'concluded'
            test['winner'] = winner
            test['concluded_at'] = datetime.now().isoformat()
            
            # Promote winner to production
            winner_id = test[f'model_{winner}_id']
            self.registry.transition_stage(winner_id, ModelStage.PRODUCTION)
            
            return self.get_test_results(test_name)


def create_model_registry(storage_path: str,
                         use_s3: bool = False,
                         s3_bucket: Optional[str] = None,
                         s3_prefix: str = "") -> ModelRegistry:
    """Factory function to create model registry."""
    if use_s3 and s3_bucket:
        artifact_storage = S3ArtifactStorage(s3_bucket, s3_prefix)
    else:
        artifact_storage = None
    
    return ModelRegistry(storage_path, artifact_storage)
