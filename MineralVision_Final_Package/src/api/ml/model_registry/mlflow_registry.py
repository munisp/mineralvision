"""
Model Versioning and Registry with MLflow

This module provides comprehensive model versioning, registry, and A/B testing
capabilities for MineralVision ML models using MLflow.

Features:
- Model versioning with semantic versioning
- Model registry with stage management (Staging, Production, Archived)
- A/B testing framework for model comparison
- Model lineage tracking
- Automated model promotion based on metrics
"""

import os
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext
import mlflow.pytorch

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class MetricComparison(Enum):
    """Metric comparison strategies."""
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass
class ModelVersion:
    """Information about a model version."""
    name: str
    version: int
    stage: ModelStage
    run_id: str
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    description: str
    tags: Dict[str, str]
    metrics: Dict[str, float]
    source: str


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    name: str
    model_a_name: str
    model_a_version: int
    model_b_name: str
    model_b_version: int
    traffic_split: float = 0.5  # Fraction of traffic to model A
    primary_metric: str = "accuracy"
    metric_comparison: MetricComparison = MetricComparison.HIGHER_IS_BETTER
    min_samples: int = 1000
    confidence_level: float = 0.95
    max_duration_hours: int = 168  # 1 week


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    config: ABTestConfig
    model_a_samples: int
    model_b_samples: int
    model_a_metric: float
    model_b_metric: float
    p_value: float
    confidence_interval: Tuple[float, float]
    winner: Optional[str]
    is_significant: bool
    duration_hours: float


class MineralVisionModelWrapper(PythonModel):
    """
    MLflow PythonModel wrapper for MineralVision models.
    
    Provides a consistent interface for model serving and inference.
    """
    
    def __init__(self, model: nn.Module = None, preprocessing_fn: Callable = None,
                 postprocessing_fn: Callable = None):
        self.model = model
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
    
    def load_context(self, context: PythonModelContext):
        """Load model artifacts."""
        import torch
        
        model_path = context.artifacts.get("model_path")
        if model_path:
            self.model = torch.load(model_path)
            self.model.eval()
    
    def predict(self, context: PythonModelContext, model_input):
        """Make predictions."""
        import torch
        import numpy as np
        
        # Preprocess input
        if self.preprocessing_fn:
            model_input = self.preprocessing_fn(model_input)
        
        # Convert to tensor
        if isinstance(model_input, np.ndarray):
            model_input = torch.from_numpy(model_input).float()
        
        # Run inference
        with torch.no_grad():
            output = self.model(model_input)
        
        # Convert to numpy
        if isinstance(output, torch.Tensor):
            output = output.numpy()
        
        # Postprocess output
        if self.postprocessing_fn:
            output = self.postprocessing_fn(output)
        
        return output


class ModelRegistry:
    """
    Model Registry for managing ML model versions.
    
    Provides:
    - Model registration and versioning
    - Stage management (Staging, Production, Archived)
    - Model comparison and promotion
    - Lineage tracking
    """
    
    def __init__(self, tracking_uri: str = None, registry_uri: str = None):
        """
        Initialize the Model Registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
        """
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
        )
        self.registry_uri = registry_uri or self.tracking_uri
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient()
    
    def register_model(self, 
                      model: nn.Module,
                      model_name: str,
                      experiment_name: str,
                      metrics: Dict[str, float],
                      params: Dict[str, Any] = None,
                      tags: Dict[str, str] = None,
                      description: str = None,
                      input_example: np.ndarray = None,
                      preprocessing_fn: Callable = None,
                      postprocessing_fn: Callable = None) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model: PyTorch model to register
            model_name: Name for the registered model
            experiment_name: MLflow experiment name
            metrics: Model metrics (accuracy, loss, etc.)
            params: Model parameters/hyperparameters
            tags: Additional tags
            description: Model description
            input_example: Example input for signature inference
            preprocessing_fn: Preprocessing function
            postprocessing_fn: Postprocessing function
            
        Returns:
            ModelVersion object
        """
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Infer signature
            signature = None
            if input_example is not None:
                model.eval()
                with torch.no_grad():
                    if isinstance(input_example, np.ndarray):
                        input_tensor = torch.from_numpy(input_example).float()
                    else:
                        input_tensor = input_example
                    output = model(input_tensor)
                    if isinstance(output, torch.Tensor):
                        output = output.numpy()
                signature = infer_signature(input_example, output)
            
            # Create wrapper
            wrapper = MineralVisionModelWrapper(
                model=model,
                preprocessing_fn=preprocessing_fn,
                postprocessing_fn=postprocessing_fn
            )
            
            # Log model
            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature
            )
            
            # Get version info
            model_version = self.client.get_latest_versions(model_name)[0]
            
            # Update description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            
            return ModelVersion(
                name=model_name,
                version=int(model_version.version),
                stage=ModelStage(model_version.current_stage),
                run_id=run.info.run_id,
                creation_timestamp=datetime.fromtimestamp(model_version.creation_timestamp / 1000),
                last_updated_timestamp=datetime.fromtimestamp(model_version.last_updated_timestamp / 1000),
                description=description or "",
                tags=tags or {},
                metrics=metrics,
                source=model_version.source
            )
    
    def get_model(self, model_name: str, version: int = None, 
                 stage: ModelStage = None) -> nn.Module:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load (optional)
            stage: Stage to load from (optional)
            
        Returns:
            Loaded PyTorch model
        """
        if version is not None:
            model_uri = f"models:/{model_name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{model_name}/{stage.value}"
        else:
            # Get latest version
            versions = self.client.get_latest_versions(model_name)
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            model_uri = f"models:/{model_name}/{versions[0].version}"
        
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        
        return model
    
    def get_production_model(self, model_name: str) -> nn.Module:
        """Get the production version of a model."""
        return self.get_model(model_name, stage=ModelStage.PRODUCTION)
    
    def get_staging_model(self, model_name: str) -> nn.Module:
        """Get the staging version of a model."""
        return self.get_model(model_name, stage=ModelStage.STAGING)
    
    def transition_model_stage(self, model_name: str, version: int,
                              stage: ModelStage, archive_existing: bool = True):
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to transition
            stage: Target stage
            archive_existing: Whether to archive existing models in target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage.value,
            archive_existing_versions=archive_existing
        )
        
        logger.info(f"Transitioned {model_name} v{version} to {stage.value}")
    
    def promote_to_staging(self, model_name: str, version: int):
        """Promote a model version to staging."""
        self.transition_model_stage(model_name, version, ModelStage.STAGING)
    
    def promote_to_production(self, model_name: str, version: int):
        """Promote a model version to production."""
        self.transition_model_stage(model_name, version, ModelStage.PRODUCTION)
    
    def archive_model(self, model_name: str, version: int):
        """Archive a model version."""
        self.transition_model_stage(model_name, version, ModelStage.ARCHIVED, 
                                   archive_existing=False)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return [m.name for m in self.client.search_registered_models()]
    
    def list_versions(self, model_name: str, 
                     stage: ModelStage = None) -> List[ModelVersion]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the registered model
            stage: Optional filter by stage
            
        Returns:
            List of ModelVersion objects
        """
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage.value])
        else:
            filter_string = f"name='{model_name}'"
            versions = self.client.search_model_versions(filter_string)
        
        result = []
        for v in versions:
            # Get run metrics
            run = self.client.get_run(v.run_id)
            metrics = run.data.metrics
            
            result.append(ModelVersion(
                name=v.name,
                version=int(v.version),
                stage=ModelStage(v.current_stage),
                run_id=v.run_id,
                creation_timestamp=datetime.fromtimestamp(v.creation_timestamp / 1000),
                last_updated_timestamp=datetime.fromtimestamp(v.last_updated_timestamp / 1000),
                description=v.description or "",
                tags=dict(v.tags) if v.tags else {},
                metrics=metrics,
                source=v.source
            ))
        
        return result
    
    def compare_versions(self, model_name: str, version_a: int, 
                        version_b: int) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the registered model
            version_a: First version
            version_b: Second version
            
        Returns:
            Comparison results
        """
        versions = self.list_versions(model_name)
        
        v_a = next((v for v in versions if v.version == version_a), None)
        v_b = next((v for v in versions if v.version == version_b), None)
        
        if v_a is None or v_b is None:
            raise ValueError("One or both versions not found")
        
        # Compare metrics
        metric_comparison = {}
        all_metrics = set(v_a.metrics.keys()) | set(v_b.metrics.keys())
        
        for metric in all_metrics:
            val_a = v_a.metrics.get(metric)
            val_b = v_b.metrics.get(metric)
            
            if val_a is not None and val_b is not None:
                diff = val_b - val_a
                pct_change = (diff / val_a * 100) if val_a != 0 else float('inf')
                metric_comparison[metric] = {
                    'version_a': val_a,
                    'version_b': val_b,
                    'difference': diff,
                    'percent_change': pct_change
                }
        
        return {
            'model_name': model_name,
            'version_a': version_a,
            'version_b': version_b,
            'metrics': metric_comparison,
            'version_a_stage': v_a.stage.value,
            'version_b_stage': v_b.stage.value
        }
    
    def auto_promote(self, model_name: str, 
                    metric_name: str = "accuracy",
                    comparison: MetricComparison = MetricComparison.HIGHER_IS_BETTER,
                    threshold: float = 0.01) -> bool:
        """
        Automatically promote the best model to production.
        
        Args:
            model_name: Name of the registered model
            metric_name: Metric to use for comparison
            comparison: Whether higher or lower is better
            threshold: Minimum improvement required for promotion
            
        Returns:
            True if a model was promoted
        """
        versions = self.list_versions(model_name)
        
        if not versions:
            return False
        
        # Get current production model
        prod_versions = [v for v in versions if v.stage == ModelStage.PRODUCTION]
        prod_metric = prod_versions[0].metrics.get(metric_name, 0) if prod_versions else 0
        
        # Find best non-production model
        candidates = [v for v in versions if v.stage != ModelStage.PRODUCTION]
        
        if not candidates:
            return False
        
        if comparison == MetricComparison.HIGHER_IS_BETTER:
            best = max(candidates, key=lambda v: v.metrics.get(metric_name, 0))
            best_metric = best.metrics.get(metric_name, 0)
            should_promote = best_metric > prod_metric * (1 + threshold)
        else:
            best = min(candidates, key=lambda v: v.metrics.get(metric_name, float('inf')))
            best_metric = best.metrics.get(metric_name, float('inf'))
            should_promote = best_metric < prod_metric * (1 - threshold)
        
        if should_promote:
            self.promote_to_production(model_name, best.version)
            logger.info(f"Auto-promoted {model_name} v{best.version} to production "
                       f"({metric_name}: {best_metric:.4f} vs {prod_metric:.4f})")
            return True
        
        return False
    
    def delete_model(self, model_name: str, version: int = None):
        """
        Delete a model or specific version.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to delete (None deletes all)
        """
        if version is not None:
            self.client.delete_model_version(model_name, str(version))
            logger.info(f"Deleted {model_name} v{version}")
        else:
            self.client.delete_registered_model(model_name)
            logger.info(f"Deleted model {model_name}")


class ABTestingFramework:
    """
    A/B Testing Framework for comparing model versions.
    
    Provides:
    - Traffic splitting between model versions
    - Statistical significance testing
    - Automated winner selection
    - Experiment tracking
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize the A/B Testing Framework.
        
        Args:
            registry: ModelRegistry instance
        """
        self.registry = registry
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_test(self, config: ABTestConfig) -> str:
        """
        Create a new A/B test.
        
        Args:
            config: A/B test configuration
            
        Returns:
            Test ID
        """
        test_id = hashlib.md5(
            f"{config.name}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        self.active_tests[test_id] = config
        self.test_results[test_id] = []
        
        logger.info(f"Created A/B test {test_id}: {config.name}")
        
        return test_id
    
    def get_model_for_request(self, test_id: str, 
                             request_id: str = None) -> Tuple[nn.Module, str]:
        """
        Get the model to use for a request based on traffic split.
        
        Args:
            test_id: A/B test ID
            request_id: Optional request ID for consistent routing
            
        Returns:
            Tuple of (model, variant name)
        """
        config = self.active_tests.get(test_id)
        if config is None:
            raise ValueError(f"Unknown test ID: {test_id}")
        
        # Determine variant based on traffic split
        if request_id:
            # Use request ID for consistent routing
            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            use_model_a = (hash_value % 100) < (config.traffic_split * 100)
        else:
            # Random assignment
            use_model_a = np.random.random() < config.traffic_split
        
        if use_model_a:
            model = self.registry.get_model(config.model_a_name, config.model_a_version)
            variant = "A"
        else:
            model = self.registry.get_model(config.model_b_name, config.model_b_version)
            variant = "B"
        
        return model, variant
    
    def record_result(self, test_id: str, variant: str, 
                     metric_value: float, metadata: Dict[str, Any] = None):
        """
        Record a result for an A/B test.
        
        Args:
            test_id: A/B test ID
            variant: Model variant ("A" or "B")
            metric_value: Value of the primary metric
            metadata: Additional metadata
        """
        if test_id not in self.test_results:
            raise ValueError(f"Unknown test ID: {test_id}")
        
        self.test_results[test_id].append({
            'variant': variant,
            'metric_value': metric_value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """
        Get current status of an A/B test.
        
        Args:
            test_id: A/B test ID
            
        Returns:
            Test status dictionary
        """
        config = self.active_tests.get(test_id)
        results = self.test_results.get(test_id, [])
        
        if config is None:
            raise ValueError(f"Unknown test ID: {test_id}")
        
        # Separate results by variant
        a_results = [r['metric_value'] for r in results if r['variant'] == 'A']
        b_results = [r['metric_value'] for r in results if r['variant'] == 'B']
        
        return {
            'test_id': test_id,
            'config': config,
            'model_a_samples': len(a_results),
            'model_b_samples': len(b_results),
            'model_a_mean': np.mean(a_results) if a_results else None,
            'model_b_mean': np.mean(b_results) if b_results else None,
            'total_samples': len(results),
            'is_complete': len(results) >= config.min_samples
        }
    
    def analyze_test(self, test_id: str) -> ABTestResult:
        """
        Analyze an A/B test and determine the winner.
        
        Args:
            test_id: A/B test ID
            
        Returns:
            ABTestResult with statistical analysis
        """
        from scipy import stats
        
        config = self.active_tests.get(test_id)
        results = self.test_results.get(test_id, [])
        
        if config is None:
            raise ValueError(f"Unknown test ID: {test_id}")
        
        # Separate results by variant
        a_results = np.array([r['metric_value'] for r in results if r['variant'] == 'A'])
        b_results = np.array([r['metric_value'] for r in results if r['variant'] == 'B'])
        
        if len(a_results) < 2 or len(b_results) < 2:
            raise ValueError("Not enough samples for statistical analysis")
        
        # Calculate means
        a_mean = np.mean(a_results)
        b_mean = np.mean(b_results)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(a_results, b_results)
        
        # Calculate confidence interval for the difference
        diff = b_mean - a_mean
        se = np.sqrt(np.var(a_results) / len(a_results) + np.var(b_results) / len(b_results))
        z = stats.norm.ppf((1 + config.confidence_level) / 2)
        ci = (diff - z * se, diff + z * se)
        
        # Determine winner
        is_significant = p_value < (1 - config.confidence_level)
        
        if is_significant:
            if config.metric_comparison == MetricComparison.HIGHER_IS_BETTER:
                winner = "A" if a_mean > b_mean else "B"
            else:
                winner = "A" if a_mean < b_mean else "B"
        else:
            winner = None
        
        # Calculate duration
        if results:
            start_time = datetime.fromisoformat(results[0]['timestamp'])
            end_time = datetime.fromisoformat(results[-1]['timestamp'])
            duration_hours = (end_time - start_time).total_seconds() / 3600
        else:
            duration_hours = 0
        
        return ABTestResult(
            config=config,
            model_a_samples=len(a_results),
            model_b_samples=len(b_results),
            model_a_metric=a_mean,
            model_b_metric=b_mean,
            p_value=p_value,
            confidence_interval=ci,
            winner=winner,
            is_significant=is_significant,
            duration_hours=duration_hours
        )
    
    def conclude_test(self, test_id: str, 
                     promote_winner: bool = True) -> ABTestResult:
        """
        Conclude an A/B test and optionally promote the winner.
        
        Args:
            test_id: A/B test ID
            promote_winner: Whether to promote the winning model to production
            
        Returns:
            Final ABTestResult
        """
        result = self.analyze_test(test_id)
        config = self.active_tests[test_id]
        
        if promote_winner and result.winner:
            if result.winner == "A":
                self.registry.promote_to_production(
                    config.model_a_name, config.model_a_version
                )
                logger.info(f"Promoted Model A ({config.model_a_name} v{config.model_a_version}) to production")
            else:
                self.registry.promote_to_production(
                    config.model_b_name, config.model_b_version
                )
                logger.info(f"Promoted Model B ({config.model_b_name} v{config.model_b_version}) to production")
        
        # Archive test
        del self.active_tests[test_id]
        
        logger.info(f"Concluded A/B test {test_id}. Winner: {result.winner or 'No significant winner'}")
        
        return result
    
    def list_active_tests(self) -> List[str]:
        """List all active A/B tests."""
        return list(self.active_tests.keys())


class ModelLineageTracker:
    """
    Track model lineage and dependencies.
    
    Records:
    - Training data versions
    - Parent models (for transfer learning)
    - Feature engineering pipelines
    - Hyperparameter tuning history
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize the lineage tracker.
        
        Args:
            registry: ModelRegistry instance
        """
        self.registry = registry
        self.lineage_store: Dict[str, Dict[str, Any]] = {}
    
    def record_lineage(self, model_name: str, version: int,
                      parent_model: Tuple[str, int] = None,
                      training_data_version: str = None,
                      feature_pipeline_version: str = None,
                      hyperparameter_tuning_run: str = None,
                      custom_metadata: Dict[str, Any] = None):
        """
        Record lineage information for a model version.
        
        Args:
            model_name: Name of the model
            version: Model version
            parent_model: Parent model (name, version) for transfer learning
            training_data_version: Version of training data used
            feature_pipeline_version: Version of feature engineering pipeline
            hyperparameter_tuning_run: ID of hyperparameter tuning run
            custom_metadata: Additional custom metadata
        """
        key = f"{model_name}:{version}"
        
        self.lineage_store[key] = {
            'model_name': model_name,
            'version': version,
            'parent_model': parent_model,
            'training_data_version': training_data_version,
            'feature_pipeline_version': feature_pipeline_version,
            'hyperparameter_tuning_run': hyperparameter_tuning_run,
            'custom_metadata': custom_metadata or {},
            'recorded_at': datetime.now().isoformat()
        }
        
        logger.info(f"Recorded lineage for {key}")
    
    def get_lineage(self, model_name: str, version: int) -> Dict[str, Any]:
        """Get lineage information for a model version."""
        key = f"{model_name}:{version}"
        return self.lineage_store.get(key, {})
    
    def get_full_lineage_tree(self, model_name: str, 
                             version: int) -> List[Dict[str, Any]]:
        """
        Get the full lineage tree for a model.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            List of lineage records from root to current model
        """
        tree = []
        current = (model_name, version)
        
        while current:
            key = f"{current[0]}:{current[1]}"
            lineage = self.lineage_store.get(key)
            
            if lineage:
                tree.insert(0, lineage)
                current = lineage.get('parent_model')
            else:
                break
        
        return tree


# Convenience functions
def create_registry(tracking_uri: str = None) -> ModelRegistry:
    """Create a new ModelRegistry instance."""
    return ModelRegistry(tracking_uri=tracking_uri)


def register_model(model: nn.Module, model_name: str, 
                  metrics: Dict[str, float], **kwargs) -> ModelVersion:
    """Register a model with default settings."""
    registry = ModelRegistry()
    return registry.register_model(
        model=model,
        model_name=model_name,
        experiment_name=kwargs.get('experiment_name', 'default'),
        metrics=metrics,
        **kwargs
    )


def load_production_model(model_name: str) -> nn.Module:
    """Load the production version of a model."""
    registry = ModelRegistry()
    return registry.get_production_model(model_name)
