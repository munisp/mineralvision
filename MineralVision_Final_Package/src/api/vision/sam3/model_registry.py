"""
SAM3 Model Registry and Adapter Management

Provides:
- Versioned model and adapter storage
- Model lifecycle management
- A/B testing support for adapters
- MLflow integration for experiment tracking
"""

import logging
import json
import hashlib
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class AdapterType(str, Enum):
    """Type of fine-tuned adapter."""
    LORA = "lora"
    FULL = "full"
    PROMPT = "prompt"


@dataclass
class AdapterMetadata:
    """Metadata for a fine-tuned adapter."""
    adapter_id: str
    name: str
    version: str
    adapter_type: AdapterType
    modality: str
    concepts: List[str]
    base_model: str
    status: ModelStatus = ModelStatus.DRAFT
    
    # Training info
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    training_epochs: int = 0
    
    # Evaluation metrics
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # File paths
    weights_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Additional metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["adapter_type"] = self.adapter_type.value
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdapterMetadata":
        d = d.copy()
        d["adapter_type"] = AdapterType(d.get("adapter_type", "lora"))
        d["status"] = ModelStatus(d.get("status", "draft"))
        return cls(**d)


@dataclass
class ModelDeployment:
    """Deployment configuration for a model/adapter."""
    deployment_id: str
    adapter_id: str
    environment: str
    traffic_percentage: float = 100.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SAM3ModelRegistry:
    """
    Registry for SAM3 models and fine-tuned adapters.
    
    Features:
    - Versioned adapter storage
    - Model lifecycle management (draft -> staging -> production)
    - A/B testing support
    - MLflow integration
    """
    
    def __init__(
        self,
        registry_dir: Union[str, Path],
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory for storing adapters
            mlflow_tracking_uri: Optional MLflow tracking URI
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_dir / "adapters").mkdir(exist_ok=True)
        (self.registry_dir / "deployments").mkdir(exist_ok=True)
        
        # Initialize MLflow if available
        self.mlflow_client = None
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
        
        # Load registry index
        self._index_path = self.registry_dir / "index.json"
        self._adapters: Dict[str, AdapterMetadata] = {}
        self._deployments: Dict[str, ModelDeployment] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load registry index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    data = json.load(f)
                    self._adapters = {
                        k: AdapterMetadata.from_dict(v)
                        for k, v in data.get("adapters", {}).items()
                    }
                    self._deployments = {
                        k: ModelDeployment(**v)
                        for k, v in data.get("deployments", {}).items()
                    }
            except Exception as e:
                logger.error(f"Failed to load registry index: {e}")
    
    def _save_index(self) -> None:
        """Save registry index to disk."""
        data = {
            "adapters": {k: v.to_dict() for k, v in self._adapters.items()},
            "deployments": {k: v.to_dict() for k, v in self._deployments.items()}
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def register_adapter(
        self,
        name: str,
        version: str,
        adapter_type: AdapterType,
        modality: str,
        concepts: List[str],
        base_model: str,
        weights_path: str,
        config_path: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> AdapterMetadata:
        """
        Register a new adapter in the registry.
        
        Args:
            name: Adapter name
            version: Semantic version
            adapter_type: Type of adapter (lora, full, prompt)
            modality: Target modality
            concepts: List of concepts this adapter handles
            base_model: Base SAM3 model version
            weights_path: Path to adapter weights
            config_path: Path to adapter config
            training_config: Training configuration used
            training_metrics: Final training metrics
            description: Human-readable description
            tags: Tags for categorization
            
        Returns:
            Registered AdapterMetadata
        """
        # Generate unique ID
        adapter_id = f"{name}_{version}_{hashlib.md5(f'{name}{version}'.encode()).hexdigest()[:8]}"
        
        # Copy weights to registry
        adapter_dir = self.registry_dir / "adapters" / adapter_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        weights_dest = adapter_dir / "weights.pt"
        if Path(weights_path).exists():
            shutil.copy(weights_path, weights_dest)
        
        config_dest = None
        if config_path and Path(config_path).exists():
            config_dest = adapter_dir / "config.json"
            shutil.copy(config_path, config_dest)
        
        # Create metadata
        metadata = AdapterMetadata(
            adapter_id=adapter_id,
            name=name,
            version=version,
            adapter_type=adapter_type,
            modality=modality,
            concepts=concepts,
            base_model=base_model,
            weights_path=str(weights_dest),
            config_path=str(config_dest) if config_dest else None,
            training_config=training_config or {},
            training_metrics=training_metrics or {},
            description=description,
            tags=tags or []
        )
        
        # Save metadata
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update index
        self._adapters[adapter_id] = metadata
        self._save_index()
        
        # Log to MLflow if available
        if self.mlflow_client:
            self._log_to_mlflow(metadata)
        
        logger.info(f"Registered adapter {adapter_id}")
        return metadata
    
    def _log_to_mlflow(self, metadata: AdapterMetadata) -> None:
        """Log adapter to MLflow."""
        if not MLFLOW_AVAILABLE or not self.mlflow_client:
            return
        
        try:
            with mlflow.start_run(run_name=f"{metadata.name}_{metadata.version}"):
                mlflow.log_params({
                    "adapter_type": metadata.adapter_type.value,
                    "modality": metadata.modality,
                    "base_model": metadata.base_model,
                    **{f"config_{k}": v for k, v in metadata.training_config.items()}
                })
                mlflow.log_metrics(metadata.training_metrics)
                mlflow.log_metrics(metadata.eval_metrics)
                
                if metadata.weights_path:
                    mlflow.log_artifact(metadata.weights_path)
                if metadata.config_path:
                    mlflow.log_artifact(metadata.config_path)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get adapter by ID."""
        return self._adapters.get(adapter_id)
    
    def list_adapters(
        self,
        modality: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        concept: Optional[str] = None
    ) -> List[AdapterMetadata]:
        """
        List adapters with optional filtering.
        
        Args:
            modality: Filter by modality
            status: Filter by status
            concept: Filter by concept
            
        Returns:
            List of matching adapters
        """
        adapters = list(self._adapters.values())
        
        if modality:
            adapters = [a for a in adapters if a.modality == modality]
        if status:
            adapters = [a for a in adapters if a.status == status]
        if concept:
            adapters = [a for a in adapters if concept in a.concepts]
        
        return sorted(adapters, key=lambda a: (a.name, a.version))
    
    def update_status(
        self,
        adapter_id: str,
        status: ModelStatus
    ) -> Optional[AdapterMetadata]:
        """
        Update adapter status.
        
        Args:
            adapter_id: Adapter ID
            status: New status
            
        Returns:
            Updated metadata or None if not found
        """
        if adapter_id not in self._adapters:
            logger.error(f"Adapter {adapter_id} not found")
            return None
        
        metadata = self._adapters[adapter_id]
        metadata.status = status
        metadata.updated_at = datetime.now().isoformat()
        
        # Update on disk
        adapter_dir = self.registry_dir / "adapters" / adapter_id
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        self._save_index()
        logger.info(f"Updated adapter {adapter_id} status to {status.value}")
        return metadata
    
    def promote_to_production(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Promote adapter to production status."""
        return self.update_status(adapter_id, ModelStatus.PRODUCTION)
    
    def archive_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Archive an adapter."""
        return self.update_status(adapter_id, ModelStatus.ARCHIVED)
    
    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter from the registry.
        
        Args:
            adapter_id: Adapter ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if adapter_id not in self._adapters:
            return False
        
        # Remove from disk
        adapter_dir = self.registry_dir / "adapters" / adapter_id
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        
        # Remove from index
        del self._adapters[adapter_id]
        self._save_index()
        
        logger.info(f"Deleted adapter {adapter_id}")
        return True
    
    def create_deployment(
        self,
        adapter_id: str,
        environment: str,
        traffic_percentage: float = 100.0,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelDeployment]:
        """
        Create a deployment for an adapter.
        
        Args:
            adapter_id: Adapter to deploy
            environment: Target environment (dev, staging, prod)
            traffic_percentage: Percentage of traffic to route
            config: Deployment configuration
            
        Returns:
            Created deployment or None if adapter not found
        """
        if adapter_id not in self._adapters:
            logger.error(f"Adapter {adapter_id} not found")
            return None
        
        deployment_id = f"{adapter_id}_{environment}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            adapter_id=adapter_id,
            environment=environment,
            traffic_percentage=traffic_percentage,
            config=config or {}
        )
        
        self._deployments[deployment_id] = deployment
        self._save_index()
        
        logger.info(f"Created deployment {deployment_id}")
        return deployment
    
    def get_production_adapter(self, modality: str) -> Optional[AdapterMetadata]:
        """Get the production adapter for a modality."""
        production_adapters = self.list_adapters(
            modality=modality,
            status=ModelStatus.PRODUCTION
        )
        if production_adapters:
            return production_adapters[-1]
        return None
    
    def get_adapter_for_concept(
        self,
        concept: str,
        modality: Optional[str] = None
    ) -> Optional[AdapterMetadata]:
        """Get the best adapter for a specific concept."""
        adapters = self.list_adapters(
            modality=modality,
            status=ModelStatus.PRODUCTION,
            concept=concept
        )
        if adapters:
            return adapters[-1]
        
        # Fall back to staging
        adapters = self.list_adapters(
            modality=modality,
            status=ModelStatus.STAGING,
            concept=concept
        )
        if adapters:
            return adapters[-1]
        
        return None


class ABTestManager:
    """
    A/B testing manager for adapter comparison.
    
    Supports:
    - Traffic splitting between adapters
    - Metric collection and comparison
    - Statistical significance testing
    """
    
    def __init__(self, registry: SAM3ModelRegistry):
        """
        Initialize A/B test manager.
        
        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self._active_tests: Dict[str, Dict[str, Any]] = {}
    
    def create_test(
        self,
        test_name: str,
        control_adapter_id: str,
        treatment_adapter_id: str,
        traffic_split: float = 0.5,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create an A/B test between two adapters.
        
        Args:
            test_name: Name for the test
            control_adapter_id: Control adapter (baseline)
            treatment_adapter_id: Treatment adapter (challenger)
            traffic_split: Fraction of traffic to treatment (0-1)
            metrics: Metrics to track
            
        Returns:
            Test configuration
        """
        control = self.registry.get_adapter(control_adapter_id)
        treatment = self.registry.get_adapter(treatment_adapter_id)
        
        if not control or not treatment:
            raise ValueError("Both adapters must exist in registry")
        
        test_config = {
            "test_name": test_name,
            "control_adapter_id": control_adapter_id,
            "treatment_adapter_id": treatment_adapter_id,
            "traffic_split": traffic_split,
            "metrics": metrics or ["iou", "dice", "latency"],
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "results": {
                "control": {"samples": 0, "metrics": {}},
                "treatment": {"samples": 0, "metrics": {}}
            }
        }
        
        self._active_tests[test_name] = test_config
        logger.info(f"Created A/B test: {test_name}")
        return test_config
    
    def route_request(self, test_name: str) -> str:
        """
        Route a request to control or treatment.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Adapter ID to use
        """
        if test_name not in self._active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        test = self._active_tests[test_name]
        import random
        
        if random.random() < test["traffic_split"]:
            return test["treatment_adapter_id"]
        return test["control_adapter_id"]
    
    def record_result(
        self,
        test_name: str,
        adapter_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Record a result for an A/B test.
        
        Args:
            test_name: Name of the test
            adapter_id: Adapter that was used
            metrics: Metrics from the request
        """
        if test_name not in self._active_tests:
            return
        
        test = self._active_tests[test_name]
        
        if adapter_id == test["control_adapter_id"]:
            group = "control"
        elif adapter_id == test["treatment_adapter_id"]:
            group = "treatment"
        else:
            return
        
        results = test["results"][group]
        results["samples"] += 1
        
        for metric, value in metrics.items():
            if metric not in results["metrics"]:
                results["metrics"][metric] = []
            results["metrics"][metric].append(value)
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """
        Get current results for an A/B test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test results with statistics
        """
        if test_name not in self._active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        test = self._active_tests[test_name]
        results = {"test_name": test_name, "status": test["status"]}
        
        for group in ["control", "treatment"]:
            group_results = test["results"][group]
            results[group] = {
                "samples": group_results["samples"],
                "metrics": {}
            }
            
            for metric, values in group_results["metrics"].items():
                if values:
                    import numpy as np
                    results[group]["metrics"][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values))
                    }
        
        return results
    
    def conclude_test(
        self,
        test_name: str,
        winner: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conclude an A/B test and optionally promote winner.
        
        Args:
            test_name: Name of the test
            winner: "control" or "treatment" to promote
            
        Returns:
            Final test results
        """
        if test_name not in self._active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        test = self._active_tests[test_name]
        test["status"] = "concluded"
        test["concluded_at"] = datetime.now().isoformat()
        
        results = self.get_test_results(test_name)
        
        if winner == "treatment":
            self.registry.promote_to_production(test["treatment_adapter_id"])
            self.registry.archive_adapter(test["control_adapter_id"])
            results["promoted"] = test["treatment_adapter_id"]
        elif winner == "control":
            self.registry.archive_adapter(test["treatment_adapter_id"])
            results["promoted"] = test["control_adapter_id"]
        
        logger.info(f"Concluded A/B test: {test_name}, winner: {winner}")
        return results


def create_model_registry(
    registry_dir: str = "./model_registry",
    mlflow_uri: Optional[str] = None
) -> SAM3ModelRegistry:
    """
    Factory function to create model registry.
    
    Args:
        registry_dir: Directory for registry storage
        mlflow_uri: Optional MLflow tracking URI
        
    Returns:
        Configured SAM3ModelRegistry
    """
    return SAM3ModelRegistry(
        registry_dir=registry_dir,
        mlflow_tracking_uri=mlflow_uri
    )
