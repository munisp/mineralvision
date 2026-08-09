"""
Automated Hyperparameter Tuning with Optuna

This module provides automated hyperparameter optimization for MineralVision ML models
using Optuna with support for:
- Bayesian optimization (TPE sampler)
- Pruning of unpromising trials
- Distributed optimization
- Multi-objective optimization
- Hyperparameter importance analysis
"""

import os
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import mlflow

logger = logging.getLogger(__name__)


class SamplerType(Enum):
    """Supported Optuna samplers."""
    TPE = "tpe"
    CMAES = "cmaes"
    RANDOM = "random"


class PrunerType(Enum):
    """Supported Optuna pruners."""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    
    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True
    
    # Batch size
    batch_size_choices: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    
    # Model architecture
    hidden_dims_min: int = 32
    hidden_dims_max: int = 512
    num_layers_min: int = 1
    num_layers_max: int = 6
    
    # Regularization
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-2
    
    # Optimizer
    optimizer_choices: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd", "rmsprop"])
    
    # Scheduler
    scheduler_choices: List[str] = field(default_factory=lambda: ["none", "cosine", "step", "exponential", "plateau"])
    
    # Activation
    activation_choices: List[str] = field(default_factory=lambda: ["relu", "leaky_relu", "gelu", "silu", "tanh"])
    
    # Normalization
    normalization_choices: List[str] = field(default_factory=lambda: ["batch", "layer", "instance", "none"])
    
    # Custom parameters
    custom_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Study settings
    study_name: str = "mineralvision_tuning"
    direction: str = "maximize"  # "maximize" or "minimize"
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    
    # Sampler settings
    sampler_type: SamplerType = SamplerType.TPE
    seed: int = 42
    
    # Pruner settings
    pruner_type: PrunerType = PrunerType.HYPERBAND
    pruner_n_startup_trials: int = 5
    pruner_n_warmup_steps: int = 10
    
    # Training settings
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Storage
    storage: Optional[str] = None  # SQLite or PostgreSQL URL
    load_if_exists: bool = True
    
    # Distributed settings
    n_jobs: int = 1  # Number of parallel jobs
    
    # Logging
    log_dir: str = "./optuna_logs"
    mlflow_tracking: bool = True
    mlflow_experiment_name: str = "hyperparameter_tuning"


class OptunaCallback(pl.Callback):
    """PyTorch Lightning callback for Optuna pruning."""
    
    def __init__(self, trial: Trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        current_value = trainer.callback_metrics.get(self.monitor)
        
        if current_value is None:
            return
        
        self.trial.report(current_value.item(), epoch)
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class HyperparameterTuner:
    """
    Automated hyperparameter tuning using Optuna.
    
    Supports:
    - Multiple sampling strategies (TPE, CMA-ES, Random)
    - Pruning of unpromising trials
    - Distributed optimization
    - MLflow integration for experiment tracking
    - Hyperparameter importance analysis
    """
    
    def __init__(self, config: TuningConfig, search_space: HyperparameterSpace = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            config: Tuning configuration
            search_space: Hyperparameter search space definition
        """
        self.config = config
        self.search_space = search_space or HyperparameterSpace()
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        
        # Create log directory
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize MLflow if enabled
        if config.mlflow_tracking:
            mlflow.set_experiment(config.mlflow_experiment_name)
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        if self.config.sampler_type == SamplerType.TPE:
            return TPESampler(seed=self.config.seed)
        elif self.config.sampler_type == SamplerType.CMAES:
            return CmaEsSampler(seed=self.config.seed)
        elif self.config.sampler_type == SamplerType.RANDOM:
            return RandomSampler(seed=self.config.seed)
        else:
            raise ValueError(f"Unknown sampler type: {self.config.sampler_type}")
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner based on configuration."""
        if self.config.pruner_type == PrunerType.MEDIAN:
            return MedianPruner(
                n_startup_trials=self.config.pruner_n_startup_trials,
                n_warmup_steps=self.config.pruner_n_warmup_steps
            )
        elif self.config.pruner_type == PrunerType.HYPERBAND:
            return HyperbandPruner(
                min_resource=1,
                max_resource=self.config.max_epochs,
                reduction_factor=3
            )
        elif self.config.pruner_type == PrunerType.SUCCESSIVE_HALVING:
            return SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        elif self.config.pruner_type == PrunerType.NONE:
            return None
        else:
            raise ValueError(f"Unknown pruner type: {self.config.pruner_type}")
    
    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {}
        
        # Learning rate
        params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            self.search_space.lr_min,
            self.search_space.lr_max,
            log=self.search_space.lr_log
        )
        
        # Batch size
        params['batch_size'] = trial.suggest_categorical(
            'batch_size',
            self.search_space.batch_size_choices
        )
        
        # Number of layers
        params['num_layers'] = trial.suggest_int(
            'num_layers',
            self.search_space.num_layers_min,
            self.search_space.num_layers_max
        )
        
        # Hidden dimensions for each layer
        params['hidden_dims'] = []
        for i in range(params['num_layers']):
            dim = trial.suggest_int(
                f'hidden_dim_{i}',
                self.search_space.hidden_dims_min,
                self.search_space.hidden_dims_max,
                step=32
            )
            params['hidden_dims'].append(dim)
        
        # Dropout
        params['dropout'] = trial.suggest_float(
            'dropout',
            self.search_space.dropout_min,
            self.search_space.dropout_max
        )
        
        # Weight decay
        params['weight_decay'] = trial.suggest_float(
            'weight_decay',
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            log=True
        )
        
        # Optimizer
        params['optimizer'] = trial.suggest_categorical(
            'optimizer',
            self.search_space.optimizer_choices
        )
        
        # Scheduler
        params['scheduler'] = trial.suggest_categorical(
            'scheduler',
            self.search_space.scheduler_choices
        )
        
        # Activation
        params['activation'] = trial.suggest_categorical(
            'activation',
            self.search_space.activation_choices
        )
        
        # Normalization
        params['normalization'] = trial.suggest_categorical(
            'normalization',
            self.search_space.normalization_choices
        )
        
        # Custom parameters
        for param_name, param_config in self.search_space.custom_params.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name, nn.ReLU())
    
    def _get_normalization(self, name: str, num_features: int) -> Optional[nn.Module]:
        """Get normalization layer by name."""
        if name == 'batch':
            return nn.BatchNorm1d(num_features)
        elif name == 'layer':
            return nn.LayerNorm(num_features)
        elif name == 'instance':
            return nn.InstanceNorm1d(num_features)
        else:
            return None
    
    def _build_model(self, params: Dict[str, Any], input_dim: int, 
                    output_dim: int) -> nn.Module:
        """Build a model with the given hyperparameters."""
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(params['hidden_dims']):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add normalization
            norm = self._get_normalization(params['normalization'], hidden_dim)
            if norm is not None:
                layers.append(norm)
            
            # Add activation
            layers.append(self._get_activation(params['activation']))
            
            # Add dropout
            if params['dropout'] > 0:
                layers.append(nn.Dropout(params['dropout']))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_optimizer(self, params: Dict[str, Any], 
                      model_params) -> torch.optim.Optimizer:
        """Get optimizer based on hyperparameters."""
        optimizer_name = params['optimizer']
        lr = params['learning_rate']
        weight_decay = params['weight_decay']
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    
    def _get_scheduler(self, params: Dict[str, Any], optimizer: torch.optim.Optimizer,
                      num_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on hyperparameters."""
        scheduler_name = params['scheduler']
        
        if scheduler_name == 'none':
            return None
        elif scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
        elif scheduler_name == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        else:
            return None
    
    def tune(self, 
            train_dataset: Dataset,
            val_dataset: Dataset,
            input_dim: int,
            output_dim: int,
            objective_fn: Optional[Callable[[Trial, nn.Module, DataLoader, DataLoader], float]] = None,
            model_builder: Optional[Callable[[Dict[str, Any], int, int], nn.Module]] = None
            ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            input_dim: Input dimension
            output_dim: Output dimension
            objective_fn: Custom objective function (optional)
            model_builder: Custom model builder function (optional)
            
        Returns:
            Dictionary with best hyperparameters and results
        """
        # Create study
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists
        )
        
        # Define objective function
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = self._suggest_hyperparameters(trial)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Use custom objective function if provided
            if objective_fn is not None:
                model = model_builder(params, input_dim, output_dim) if model_builder else self._build_model(params, input_dim, output_dim)
                return objective_fn(trial, model, train_loader, val_loader)
            
            # Default training loop
            model = model_builder(params, input_dim, output_dim) if model_builder else self._build_model(params, input_dim, output_dim)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            optimizer = self._get_optimizer(params, model.parameters())
            scheduler = self._get_scheduler(params, optimizer, self.config.max_epochs)
            criterion = nn.CrossEntropyLoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Training
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_accuracy = correct / total
                
                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Report to Optuna for pruning
                trial.report(val_accuracy if self.config.direction == "maximize" else val_loss, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break
            
            # Log to MLflow
            if self.config.mlflow_tracking:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("val_loss", best_val_loss)
                    mlflow.log_metric("val_accuracy", val_accuracy)
            
            return val_accuracy if self.config.direction == "maximize" else best_val_loss
        
        # Run optimization
        logger.info(f"Starting hyperparameter tuning with {self.config.n_trials} trials")
        
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Best trial value: {self.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'study_name': self.config.study_name
        }
    
    def get_best_model(self, input_dim: int, output_dim: int,
                      model_builder: Optional[Callable] = None) -> nn.Module:
        """
        Get a model with the best hyperparameters.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            model_builder: Custom model builder function
            
        Returns:
            Model with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No tuning has been performed yet")
        
        if model_builder is not None:
            return model_builder(self.best_params, input_dim, output_dim)
        
        return self._build_model(self.best_params, input_dim, output_dim)
    
    def get_param_importances(self) -> Dict[str, float]:
        """
        Get hyperparameter importances.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None:
            raise ValueError("No tuning has been performed yet")
        
        return optuna.importance.get_param_importances(self.study)
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet")
        
        fig = plot_optimization_history(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet")
        
        fig = plot_param_importances(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """Plot parallel coordinate visualization."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet")
        
        fig = plot_parallel_coordinate(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def save_results(self, path: str):
        """Save tuning results to JSON file."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet")
        
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'study_name': self.config.study_name,
            'param_importances': self.get_param_importances(),
            'trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in self.study.trials
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {path}")
    
    def load_results(self, path: str):
        """Load tuning results from JSON file."""
        with open(path, 'r') as f:
            results = json.load(f)
        
        self.best_params = results['best_params']
        self.best_value = results['best_value']
        
        logger.info(f"Results loaded from {path}")
        return results


class MultiObjectiveTuner(HyperparameterTuner):
    """
    Multi-objective hyperparameter tuning.
    
    Optimizes multiple objectives simultaneously (e.g., accuracy and model size).
    """
    
    def __init__(self, config: TuningConfig, search_space: HyperparameterSpace = None,
                 objectives: List[str] = None, directions: List[str] = None):
        """
        Initialize multi-objective tuner.
        
        Args:
            config: Tuning configuration
            search_space: Hyperparameter search space
            objectives: List of objective names
            directions: List of optimization directions ("maximize" or "minimize")
        """
        super().__init__(config, search_space)
        
        self.objectives = objectives or ["accuracy", "model_size"]
        self.directions = directions or ["maximize", "minimize"]
    
    def tune_multi_objective(self,
                            train_dataset: Dataset,
                            val_dataset: Dataset,
                            input_dim: int,
                            output_dim: int,
                            objective_fn: Callable[[Trial, nn.Module, DataLoader, DataLoader], Tuple[float, ...]]
                            ) -> Dict[str, Any]:
        """
        Run multi-objective hyperparameter tuning.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            input_dim: Input dimension
            output_dim: Output dimension
            objective_fn: Objective function returning tuple of values
            
        Returns:
            Dictionary with Pareto-optimal solutions
        """
        sampler = self._create_sampler()
        
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            directions=self.directions,
            sampler=sampler,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists
        )
        
        def objective(trial: Trial) -> Tuple[float, ...]:
            params = self._suggest_hyperparameters(trial)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            model = self._build_model(params, input_dim, output_dim)
            return objective_fn(trial, model, train_loader, val_loader)
        
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        # Get Pareto-optimal trials
        pareto_trials = self.study.best_trials
        
        return {
            'pareto_trials': [
                {
                    'values': t.values,
                    'params': t.params
                }
                for t in pareto_trials
            ],
            'n_trials': len(self.study.trials),
            'objectives': self.objectives,
            'directions': self.directions
        }


# Convenience functions
def quick_tune(train_dataset: Dataset, val_dataset: Dataset,
              input_dim: int, output_dim: int,
              n_trials: int = 50) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning with default settings.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        input_dim: Input dimension
        output_dim: Output dimension
        n_trials: Number of trials
        
    Returns:
        Best hyperparameters
    """
    config = TuningConfig(n_trials=n_trials)
    tuner = HyperparameterTuner(config)
    return tuner.tune(train_dataset, val_dataset, input_dim, output_dim)
