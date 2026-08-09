"""
Ensemble Methods for MineralVision

This module provides comprehensive ensemble learning methods for improved
prediction accuracy and robustness in mineral exploration tasks.

Supports:
- Model averaging (simple, weighted, Bayesian)
- Model stacking (meta-learning)
- Boosting (gradient boosting, AdaBoost)
- Bagging
- Snapshot ensembles
- Deep ensemble with uncertainty
- Multi-model fusion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold
import copy

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Supported ensemble methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_AVERAGE = "bayesian_average"
    STACKING = "stacking"
    BOOSTING = "boosting"
    BAGGING = "bagging"
    SNAPSHOT = "snapshot"
    DEEP_ENSEMBLE = "deep_ensemble"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    # General settings
    num_models: int = 5
    aggregation_method: str = "mean"  # mean, median, vote
    
    # Weighted average settings
    weight_by_performance: bool = True
    temperature: float = 1.0  # For softmax weighting
    
    # Stacking settings
    meta_learner: str = "logistic"  # logistic, mlp, xgboost
    use_probabilities: bool = True
    cv_folds: int = 5
    
    # Boosting settings
    learning_rate: float = 0.1
    max_depth: int = 3
    n_estimators: int = 100
    
    # Bagging settings
    sample_ratio: float = 0.8
    feature_ratio: float = 1.0
    bootstrap: bool = True
    
    # Snapshot ensemble settings
    snapshot_epochs: List[int] = field(default_factory=lambda: [20, 40, 60, 80, 100])
    cyclic_lr: bool = True
    
    # Deep ensemble settings
    diversity_weight: float = 0.1
    uncertainty_estimation: bool = True
    
    # Training settings
    epochs: int = 100
    batch_size: int = 32
    device: str = "cuda"


class ModelAveraging:
    """
    Model averaging ensemble methods.
    
    Combines predictions from multiple models using various averaging strategies.
    """
    
    def __init__(self, models: List[nn.Module], config: EnsembleConfig = None):
        """
        Initialize model averaging ensemble.
        
        Args:
            models: List of PyTorch models
            config: Ensemble configuration
        """
        self.models = models
        self.config = config or EnsembleConfig()
        self.weights = np.ones(len(models)) / len(models)
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def simple_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple average of model predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        
        if self.config.aggregation_method == "mean":
            return stacked.mean(dim=0)
        elif self.config.aggregation_method == "median":
            return stacked.median(dim=0)[0]
        else:
            return stacked.mean(dim=0)
    
    def weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weighted average of model predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted averaged predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        
        # Apply weights
        weights = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
        if len(stacked.shape) == 2:
            weights = weights.squeeze(-1)
        
        weighted = stacked * weights
        return weighted.sum(dim=0)
    
    def bayesian_average(self, x: torch.Tensor, 
                        return_uncertainty: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Bayesian model averaging with uncertainty estimation.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions and optionally uncertainty
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        
        # Mean prediction
        mean_pred = stacked.mean(dim=0)
        
        if return_uncertainty:
            # Epistemic uncertainty (model disagreement)
            variance = stacked.var(dim=0)
            return mean_pred, variance
        
        return mean_pred
    
    def set_weights(self, weights: np.ndarray):
        """Set model weights for weighted averaging."""
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        self.weights = weights / weights.sum()  # Normalize
    
    def compute_weights_from_validation(self, val_loader: DataLoader,
                                       criterion: nn.Module = None):
        """
        Compute weights based on validation performance.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function (default: CrossEntropyLoss)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        for model in self.models:
            model.eval()
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
        
        # Convert losses to weights (lower loss = higher weight)
        losses = np.array(losses)
        
        # Use softmax with temperature
        weights = np.exp(-losses / self.config.temperature)
        self.weights = weights / weights.sum()
        
        logger.info(f"Computed weights: {self.weights}")
    
    def predict(self, x: torch.Tensor, 
               method: str = "weighted") -> torch.Tensor:
        """
        Make predictions using specified averaging method.
        
        Args:
            x: Input tensor
            method: Averaging method (simple, weighted, bayesian)
            
        Returns:
            Predictions
        """
        x = x.to(self.device)
        
        if method == "simple":
            return self.simple_average(x)
        elif method == "weighted":
            return self.weighted_average(x)
        elif method == "bayesian":
            return self.bayesian_average(x, return_uncertainty=False)
        else:
            return self.simple_average(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, uncertainty)
        """
        x = x.to(self.device)
        return self.bayesian_average(x, return_uncertainty=True)


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    
    Uses predictions from base models as features for a meta-learner.
    """
    
    def __init__(self, base_models: List[nn.Module], config: EnsembleConfig = None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base PyTorch models
            config: Ensemble configuration
        """
        self.base_models = base_models
        self.config = config or EnsembleConfig()
        self.meta_learner = None
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Move base models to device
        for model in self.base_models:
            model.to(self.device)
    
    def _get_base_predictions(self, x: torch.Tensor, 
                             use_probabilities: bool = True) -> torch.Tensor:
        """Get predictions from all base models."""
        predictions = []
        
        with torch.no_grad():
            for model in self.base_models:
                model.eval()
                pred = model(x)
                
                if use_probabilities and pred.shape[-1] > 1:
                    pred = F.softmax(pred, dim=-1)
                
                predictions.append(pred)
        
        # Concatenate predictions
        return torch.cat(predictions, dim=-1)
    
    def _create_meta_learner(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create meta-learner model."""
        if self.config.meta_learner == "logistic":
            return nn.Linear(input_dim, output_dim)
        
        elif self.config.meta_learner == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(input_dim * 2, input_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(input_dim, output_dim)
            )
        
        else:
            return nn.Linear(input_dim, output_dim)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None,
           num_classes: int = 2):
        """
        Fit the stacking ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of output classes
        """
        # Collect base model predictions using cross-validation
        all_meta_features = []
        all_labels = []
        
        # Get meta-features from training data
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            
            meta_features = self._get_base_predictions(
                batch_x, 
                use_probabilities=self.config.use_probabilities
            )
            
            all_meta_features.append(meta_features.cpu())
            all_labels.append(batch_y)
        
        meta_features = torch.cat(all_meta_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Create and train meta-learner
        input_dim = meta_features.shape[-1]
        self.meta_learner = self._create_meta_learner(input_dim, num_classes)
        self.meta_learner.to(self.device)
        
        # Train meta-learner
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        meta_dataset = torch.utils.data.TensorDataset(meta_features, labels)
        meta_loader = DataLoader(meta_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        for epoch in range(self.config.epochs):
            self.meta_learner.train()
            total_loss = 0.0
            
            for batch_meta, batch_y in meta_loader:
                batch_meta = batch_meta.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.meta_learner(batch_meta)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Meta-learner epoch {epoch + 1}, loss: {total_loss / len(meta_loader):.4f}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using stacking ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions
        """
        x = x.to(self.device)
        
        # Get base model predictions
        meta_features = self._get_base_predictions(
            x, 
            use_probabilities=self.config.use_probabilities
        )
        
        # Get meta-learner predictions
        self.meta_learner.eval()
        with torch.no_grad():
            return self.meta_learner(meta_features)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.predict(x)
        return F.softmax(logits, dim=-1)


class BaggingEnsemble:
    """
    Bagging ensemble for neural networks.
    
    Trains multiple models on bootstrap samples of the data.
    """
    
    def __init__(self, model_factory: Callable[[], nn.Module], 
                config: EnsembleConfig = None):
        """
        Initialize bagging ensemble.
        
        Args:
            model_factory: Function that creates a new model instance
            config: Ensemble configuration
        """
        self.model_factory = model_factory
        self.config = config or EnsembleConfig()
        self.models: List[nn.Module] = []
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
    
    def _create_bootstrap_sample(self, dataset: Dataset) -> Dataset:
        """Create a bootstrap sample of the dataset."""
        n = len(dataset)
        sample_size = int(n * self.config.sample_ratio)
        
        if self.config.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n, size=sample_size, replace=True)
        else:
            # Sample without replacement
            indices = np.random.choice(n, size=sample_size, replace=False)
        
        return torch.utils.data.Subset(dataset, indices)
    
    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None,
           criterion: nn.Module = None, optimizer_factory: Callable = None):
        """
        Fit the bagging ensemble.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            criterion: Loss function
            optimizer_factory: Function to create optimizer
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.models = []
        
        for i in range(self.config.num_models):
            logger.info(f"Training model {i + 1}/{self.config.num_models}")
            
            # Create bootstrap sample
            bootstrap_dataset = self._create_bootstrap_sample(train_dataset)
            bootstrap_loader = DataLoader(
                bootstrap_dataset, 
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Create and train model
            model = self.model_factory()
            model.to(self.device)
            
            if optimizer_factory:
                optimizer = optimizer_factory(model.parameters())
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training loop
            for epoch in range(self.config.epochs):
                model.train()
                total_loss = 0.0
                
                for batch_x, batch_y in bootstrap_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            self.models.append(model)
            logger.info(f"Model {i + 1} trained, final loss: {total_loss / len(bootstrap_loader):.4f}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using bagging ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged predictions
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty from model disagreement."""
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0)
        
        return mean_pred, variance


class SnapshotEnsemble:
    """
    Snapshot ensemble using cyclic learning rates.
    
    Saves model snapshots at different points during training to create
    an ensemble without additional training cost.
    """
    
    def __init__(self, model: nn.Module, config: EnsembleConfig = None):
        """
        Initialize snapshot ensemble.
        
        Args:
            model: Base PyTorch model
            config: Ensemble configuration
        """
        self.base_model = model
        self.config = config or EnsembleConfig()
        self.snapshots: List[nn.Module] = []
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        self.base_model.to(self.device)
    
    def _cosine_annealing_lr(self, epoch: int, cycle_length: int, 
                            lr_max: float, lr_min: float = 0) -> float:
        """Calculate learning rate using cosine annealing."""
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + np.cos(np.pi * (epoch % cycle_length) / cycle_length)
        )
    
    def fit(self, train_loader: DataLoader, criterion: nn.Module = None,
           lr_max: float = 0.1, lr_min: float = 1e-6):
        """
        Train model and collect snapshots.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            lr_max: Maximum learning rate
            lr_min: Minimum learning rate
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.snapshots = []
        
        # Calculate cycle length
        num_snapshots = len(self.config.snapshot_epochs)
        
        optimizer = torch.optim.SGD(
            self.base_model.parameters(), 
            lr=lr_max, 
            momentum=0.9,
            weight_decay=1e-4
        )
        
        for epoch in range(self.config.epochs):
            # Update learning rate
            if self.config.cyclic_lr:
                cycle_length = self.config.epochs // num_snapshots
                lr = self._cosine_annealing_lr(epoch, cycle_length, lr_max, lr_min)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Training
            self.base_model.train()
            total_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.base_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Save snapshot at specified epochs
            if (epoch + 1) in self.config.snapshot_epochs:
                snapshot = copy.deepcopy(self.base_model)
                snapshot.eval()
                self.snapshots.append(snapshot)
                logger.info(f"Saved snapshot at epoch {epoch + 1}")
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, loss: {total_loss / len(train_loader):.4f}")
        
        logger.info(f"Training complete. Collected {len(self.snapshots)} snapshots.")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using snapshot ensemble."""
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for snapshot in self.snapshots:
                pred = snapshot(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)


class DeepEnsemble:
    """
    Deep ensemble with diversity regularization and uncertainty estimation.
    
    Trains multiple models with different initializations and optional
    diversity-promoting regularization.
    """
    
    def __init__(self, model_factory: Callable[[], nn.Module],
                config: EnsembleConfig = None):
        """
        Initialize deep ensemble.
        
        Args:
            model_factory: Function that creates a new model instance
            config: Ensemble configuration
        """
        self.model_factory = model_factory
        self.config = config or EnsembleConfig()
        self.models: List[nn.Module] = []
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
    
    def _diversity_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate diversity loss to encourage model disagreement.
        
        Args:
            predictions: List of predictions from each model
            
        Returns:
            Diversity loss (negative, to maximize diversity)
        """
        if len(predictions) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate pairwise disagreement
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0, keepdim=True)
        
        # Variance across models (higher = more diverse)
        variance = ((stacked - mean_pred) ** 2).mean()
        
        # Return negative variance to minimize (maximize diversity)
        return -variance
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None,
           criterion: nn.Module = None):
        """
        Train deep ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.models = []
        
        for i in range(self.config.num_models):
            logger.info(f"Training model {i + 1}/{self.config.num_models}")
            
            # Create model with random initialization
            model = self.model_factory()
            model.to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(self.config.epochs):
                model.train()
                total_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    
                    # Task loss
                    task_loss = criterion(outputs, batch_y)
                    
                    # Diversity loss (if we have other models)
                    if self.config.diversity_weight > 0 and len(self.models) > 0:
                        other_preds = []
                        with torch.no_grad():
                            for other_model in self.models:
                                other_model.eval()
                                other_preds.append(other_model(batch_x))
                        
                        other_preds.append(outputs)
                        div_loss = self._diversity_loss(other_preds)
                        loss = task_loss + self.config.diversity_weight * div_loss
                    else:
                        loss = task_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += task_loss.item()
                
                # Validation
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)
                            outputs = model(batch_x)
                            val_loss += criterion(outputs, batch_y).item()
                    
                    val_loss /= len(val_loader)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(model.state_dict())
            
            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model.eval()
            self.models.append(model)
            logger.info(f"Model {i + 1} trained")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using deep ensemble."""
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Returns:
            Tuple of (mean prediction, epistemic uncertainty, total uncertainty)
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                
                # Apply softmax for probabilities
                if pred.shape[-1] > 1:
                    pred = F.softmax(pred, dim=-1)
                
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        
        # Mean prediction
        mean_pred = stacked.mean(dim=0)
        
        # Epistemic uncertainty (model disagreement)
        epistemic = stacked.var(dim=0)
        
        # Aleatoric uncertainty (average entropy)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
        
        # Total uncertainty
        total = epistemic.mean(dim=-1) + entropy
        
        return mean_pred, epistemic, total


class GradientBoostingEnsemble:
    """
    Gradient boosting for neural networks.
    
    Trains models sequentially, with each model fitting the residuals
    of the previous models.
    """
    
    def __init__(self, model_factory: Callable[[], nn.Module],
                config: EnsembleConfig = None):
        """
        Initialize gradient boosting ensemble.
        
        Args:
            model_factory: Function that creates a new model instance
            config: Ensemble configuration
        """
        self.model_factory = model_factory
        self.config = config or EnsembleConfig()
        self.models: List[nn.Module] = []
        self.model_weights: List[float] = []
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
    
    def fit(self, train_loader: DataLoader, criterion: nn.Module = None):
        """
        Train gradient boosting ensemble.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.models = []
        self.model_weights = []
        
        # Initialize residuals
        residuals = None
        
        for i in range(self.config.n_estimators):
            logger.info(f"Training estimator {i + 1}/{self.config.n_estimators}")
            
            # Create model
            model = self.model_factory()
            model.to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Train on residuals
            for epoch in range(self.config.epochs // self.config.n_estimators):
                model.train()
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device).float()
                    
                    # Calculate current prediction
                    if residuals is None:
                        target = batch_y
                    else:
                        with torch.no_grad():
                            current_pred = self._predict_partial(batch_x)
                        target = batch_y - current_pred
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    
                    # Ensure output shape matches target
                    if outputs.shape != target.shape:
                        if len(target.shape) == 1:
                            target = target.unsqueeze(-1)
                        outputs = outputs.view_as(target)
                    
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            self.models.append(model)
            self.model_weights.append(self.config.learning_rate)
            
            # Update residuals flag
            residuals = True
    
    def _predict_partial(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from current ensemble."""
        prediction = torch.zeros(x.shape[0], 1, device=self.device)
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.model_weights):
                pred = model(x)
                if len(pred.shape) == 1:
                    pred = pred.unsqueeze(-1)
                prediction += weight * pred
        
        return prediction
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using gradient boosting ensemble."""
        x = x.to(self.device)
        return self._predict_partial(x)


class EnsembleFactory:
    """
    Factory for creating ensemble models.
    """
    
    @staticmethod
    def create_ensemble(method: EnsembleMethod, 
                       models: List[nn.Module] = None,
                       model_factory: Callable[[], nn.Module] = None,
                       config: EnsembleConfig = None) -> Any:
        """
        Create an ensemble model.
        
        Args:
            method: Ensemble method to use
            models: Pre-trained models (for averaging methods)
            model_factory: Factory function for creating models
            config: Ensemble configuration
            
        Returns:
            Ensemble model
        """
        config = config or EnsembleConfig()
        
        if method in [EnsembleMethod.SIMPLE_AVERAGE, EnsembleMethod.WEIGHTED_AVERAGE, 
                     EnsembleMethod.BAYESIAN_AVERAGE]:
            if models is None:
                raise ValueError("Models required for averaging methods")
            return ModelAveraging(models, config)
        
        elif method == EnsembleMethod.STACKING:
            if models is None:
                raise ValueError("Base models required for stacking")
            return StackingEnsemble(models, config)
        
        elif method == EnsembleMethod.BAGGING:
            if model_factory is None:
                raise ValueError("Model factory required for bagging")
            return BaggingEnsemble(model_factory, config)
        
        elif method == EnsembleMethod.SNAPSHOT:
            if models is None or len(models) == 0:
                raise ValueError("Base model required for snapshot ensemble")
            return SnapshotEnsemble(models[0], config)
        
        elif method == EnsembleMethod.DEEP_ENSEMBLE:
            if model_factory is None:
                raise ValueError("Model factory required for deep ensemble")
            return DeepEnsemble(model_factory, config)
        
        elif method == EnsembleMethod.BOOSTING:
            if model_factory is None:
                raise ValueError("Model factory required for boosting")
            return GradientBoostingEnsemble(model_factory, config)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


# Convenience functions
def create_model_averaging(models: List[nn.Module], 
                          method: str = "weighted") -> ModelAveraging:
    """Create a model averaging ensemble."""
    config = EnsembleConfig()
    return ModelAveraging(models, config)


def create_stacking_ensemble(base_models: List[nn.Module]) -> StackingEnsemble:
    """Create a stacking ensemble."""
    config = EnsembleConfig()
    return StackingEnsemble(base_models, config)


def create_deep_ensemble(model_factory: Callable[[], nn.Module],
                        num_models: int = 5) -> DeepEnsemble:
    """Create a deep ensemble."""
    config = EnsembleConfig(num_models=num_models)
    return DeepEnsemble(model_factory, config)
