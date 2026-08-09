"""
AI-Powered Predictive Modeling for Mineral Deposit Prediction

This module implements deep learning models for mineral deposit prediction based on
multiple data sources including geological, geophysical, geochemical, and remote sensing data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import tensorflow_probability as tfp
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union

class MineralDepositDataset(Dataset):
    """
    Dataset class for mineral deposit prediction.
    
    Handles loading and preprocessing of multi-source geospatial data including:
    - Geological features (rock types, structures, etc.)
    - Geophysical measurements (magnetic, gravity, etc.)
    - Geochemical data (element concentrations)
    - Remote sensing data (hyperspectral, LiDAR, etc.)
    - Historical mining data
    """
    
    def __init__(self, 
                 data_dir: str,
                 geological_data: Optional[str] = None,
                 geophysical_data: Optional[str] = None,
                 geochemical_data: Optional[str] = None,
                 remote_sensing_data: Optional[str] = None,
                 historical_data: Optional[str] = None,
                 transform=None,
                 target_transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the data files
            geological_data: Path to geological data file
            geophysical_data: Path to geophysical data file
            geochemical_data: Path to geochemical data file
            remote_sensing_data: Path to remote sensing data file
            historical_data: Path to historical mining data file
            transform: Transforms to apply to features
            target_transform: Transforms to apply to targets
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data from various sources
        self.data = self._load_data(
            geological_data, 
            geophysical_data,
            geochemical_data,
            remote_sensing_data,
            historical_data
        )
        
        # Extract features and targets
        self.features, self.targets = self._prepare_data()
        
    def _load_data(self, 
                  geological_data: Optional[str],
                  geophysical_data: Optional[str],
                  geochemical_data: Optional[str],
                  remote_sensing_data: Optional[str],
                  historical_data: Optional[str]) -> Dict:
        """
        Load data from various sources and merge based on spatial location.
        
        Returns:
            Dictionary containing merged data
        """
        data = {}
        
        # Load geological data if provided
        if geological_data:
            geo_path = os.path.join(self.data_dir, geological_data)
            if geo_path.endswith('.shp'):
                data['geological'] = gpd.read_file(geo_path)
            elif geo_path.endswith('.csv'):
                data['geological'] = pd.read_csv(geo_path)
        
        # Load geophysical data if provided
        if geophysical_data:
            geophys_path = os.path.join(self.data_dir, geophysical_data)
            if geophys_path.endswith('.shp'):
                data['geophysical'] = gpd.read_file(geophys_path)
            elif geophys_path.endswith('.csv'):
                data['geophysical'] = pd.read_csv(geophys_path)
        
        # Load geochemical data if provided
        if geochemical_data:
            geochem_path = os.path.join(self.data_dir, geochemical_data)
            if geochem_path.endswith('.shp'):
                data['geochemical'] = gpd.read_file(geochem_path)
            elif geochem_path.endswith('.csv'):
                data['geochemical'] = pd.read_csv(geochem_path)
        
        # Load remote sensing data if provided
        if remote_sensing_data:
            rs_path = os.path.join(self.data_dir, remote_sensing_data)
            if rs_path.endswith('.shp'):
                data['remote_sensing'] = gpd.read_file(rs_path)
            elif rs_path.endswith('.csv'):
                data['remote_sensing'] = pd.read_csv(rs_path)
        
        # Load historical mining data if provided
        if historical_data:
            hist_path = os.path.join(self.data_dir, historical_data)
            if hist_path.endswith('.shp'):
                data['historical'] = gpd.read_file(hist_path)
            elif hist_path.endswith('.csv'):
                data['historical'] = pd.read_csv(hist_path)
        
        return data
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets from loaded data.
        
        Returns:
            Tuple of (features, targets)
        """
        # This is a simplified implementation
        # In a real-world scenario, this would involve complex spatial joins,
        # feature engineering, and handling of missing data
        
        # For demonstration purposes, we'll create a simple merged dataset
        merged_data = None
        target_column = 'deposit_present'
        
        # Merge data based on spatial location
        # This is a placeholder for actual spatial joining logic
        if 'geological' in self.data and 'historical' in self.data:
            # In reality, this would be a spatial join
            merged_data = pd.merge(
                self.data['geological'],
                self.data['historical'],
                on='location_id',
                how='left'
            )
        
        if merged_data is None:
            # Create dummy data for demonstration
            n_samples = 1000
            n_features = 50
            features = np.random.randn(n_samples, n_features)
            targets = np.random.randint(0, 2, size=(n_samples, 1))
            return features, targets
        
        # Extract features and targets
        features = merged_data.drop(columns=[target_column]).values
        targets = merged_data[target_column].values
        
        return features, targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        if self.target_transform:
            target = self.target_transform(target)
            
        return feature, target


class MineralDepositPredictor(pl.LightningModule):
    """
    PyTorch Lightning module for mineral deposit prediction.
    
    Implements a deep learning model with uncertainty quantification.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 uncertainty_estimation: bool = True):
        """
        Initialize the model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            uncertainty_estimation: Whether to estimate prediction uncertainty
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Build the network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer depends on uncertainty estimation
        if uncertainty_estimation:
            # For uncertainty estimation, output mean and log variance
            self.output_layer = nn.Linear(prev_dim, 2)
        else:
            # For point prediction, output just the prediction
            self.output_layer = nn.Linear(prev_dim, 1)
        
        self.uncertainty_estimation = uncertainty_estimation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output (prediction and uncertainty if enabled)
        """
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        
        if self.uncertainty_estimation:
            # Split output into mean and log variance
            mean, log_var = torch.split(output, 1, dim=1)
            # Ensure variance is positive
            var = torch.exp(log_var)
            return mean, var
        else:
            return torch.sigmoid(output)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        
        if self.uncertainty_estimation:
            mean, var = self(x)
            # Negative log likelihood loss with uncertainty
            loss = self._gaussian_nll_loss(y, mean, var)
        else:
            y_hat = self(x)
            # Binary cross entropy loss
            loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        x, y = batch
        
        if self.uncertainty_estimation:
            mean, var = self(x)
            # Negative log likelihood loss with uncertainty
            loss = self._gaussian_nll_loss(y, mean, var)
            # For metrics, use the mean prediction
            y_hat = torch.sigmoid(mean)
        else:
            y_hat = self(x)
            # Binary cross entropy loss
            loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        
        # Calculate metrics
        preds = (y_hat > 0.5).float()
        acc = (preds == y.float().view(-1, 1)).float().mean()
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        x, y = batch
        
        if self.uncertainty_estimation:
            mean, var = self(x)
            # Negative log likelihood loss with uncertainty
            loss = self._gaussian_nll_loss(y, mean, var)
            # For metrics, use the mean prediction
            y_hat = torch.sigmoid(mean)
        else:
            y_hat = self(x)
            # Binary cross entropy loss
            loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        
        # Calculate metrics
        preds = (y_hat > 0.5).float()
        acc = (preds == y.float().view(-1, 1)).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
    
    def _gaussian_nll_loss(self, y_true, mean, var):
        """
        Gaussian negative log likelihood loss.
        
        Args:
            y_true: True values
            mean: Predicted means
            var: Predicted variances
            
        Returns:
            Loss value
        """
        y_true = y_true.float().view(-1, 1)
        # Gaussian NLL loss
        return 0.5 * torch.mean(
            torch.log(var) + (y_true - mean)**2 / var
        )
    
    def configure_optimizers(self):
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty estimation using Monte Carlo sampling.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean prediction, prediction variance)
        """
        if not self.uncertainty_estimation:
            raise ValueError("Model was not trained with uncertainty estimation")
        
        self.eval()
        preds = []
        
        # Enable dropout at test time for MC Dropout
        def enable_dropout(model):
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        
        enable_dropout(self)
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, var = self(x)
                # Sample from the predicted distribution
                sample = mean + torch.sqrt(var) * torch.randn_like(mean)
                preds.append(torch.sigmoid(sample))
        
        # Stack predictions
        preds = torch.stack(preds)
        
        # Calculate mean and variance of predictions
        mean_pred = preds.mean(dim=0)
        var_pred = preds.var(dim=0)
        
        return mean_pred, var_pred


class FeatureExtractor:
    """
    Automated feature extraction from multiple data sources.
    
    Uses computer vision and signal processing techniques to extract
    relevant features from geospatial data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def extract_features_from_images(self, image_data):
        """
        Extract features from satellite or drone imagery.
        
        Args:
            image_data: Image data array (numpy array with shape HxWxC or HxW)
            
        Returns:
            Extracted features as numpy array
        """
        if image_data is None or not isinstance(image_data, np.ndarray):
            return np.zeros(100)
        
        features = []
        
        if len(image_data.shape) == 3:
            for c in range(image_data.shape[2]):
                channel = image_data[:, :, c]
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.min(channel),
                    np.max(channel),
                    np.median(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75),
                ])
        else:
            channel = image_data
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
            ])
        
        if len(image_data.shape) >= 2:
            gy, gx = np.gradient(image_data.astype(float) if len(image_data.shape) == 2 else image_data[:, :, 0].astype(float))
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude),
            ])
        
        hist, _ = np.histogram(image_data.flatten(), bins=20, density=True)
        features.extend(hist.tolist())
        
        while len(features) < 100:
            features.append(0.0)
        
        return np.array(features[:100])
    
    def extract_features_from_geophysical(self, geophysical_data):
        """
        Extract features from geophysical data (magnetic, gravity, seismic, etc.).
        
        Args:
            geophysical_data: Geophysical data array (1D signal or 2D grid)
            
        Returns:
            Extracted features as numpy array
        """
        if geophysical_data is None or not isinstance(geophysical_data, np.ndarray):
            return np.zeros(100)
        
        features = []
        data = geophysical_data.flatten() if len(geophysical_data.shape) > 1 else geophysical_data
        
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.median(data),
            np.percentile(data, 10),
            np.percentile(data, 90),
            np.var(data),
        ])
        
        try:
            from scipy import fft
            fft_result = np.abs(fft.fft(data))
            n = len(fft_result) // 2
            if n > 0:
                features.extend([
                    np.mean(fft_result[:n]),
                    np.std(fft_result[:n]),
                    np.max(fft_result[:n]),
                    np.argmax(fft_result[:n]) / n if n > 0 else 0,
                ])
                power_spectrum = fft_result[:n] ** 2
                features.extend([
                    np.sum(power_spectrum[:n//4]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0,
                    np.sum(power_spectrum[n//4:n//2]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0,
                    np.sum(power_spectrum[n//2:]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0,
                ])
        except ImportError:
            features.extend([0.0] * 7)
        
        if len(geophysical_data.shape) == 2:
            gy, gx = np.gradient(geophysical_data.astype(float))
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude),
            ])
            
            laplacian = np.gradient(np.gradient(geophysical_data.astype(float), axis=0), axis=0) + \
                        np.gradient(np.gradient(geophysical_data.astype(float), axis=1), axis=1)
            features.extend([
                np.mean(laplacian),
                np.std(laplacian),
            ])
        else:
            features.extend([0.0] * 5)
        
        hist, _ = np.histogram(data, bins=20, density=True)
        features.extend(hist.tolist())
        
        while len(features) < 100:
            features.append(0.0)
        
        return np.array(features[:100])
    
    def extract_features_from_geological(self, geological_data):
        """
        Extract features from geological data (rock types, structures, formations).
        
        Args:
            geological_data: Geological data (dict, DataFrame, or numpy array)
            
        Returns:
            Extracted features as numpy array
        """
        if geological_data is None:
            return np.zeros(100)
        
        features = []
        
        if isinstance(geological_data, dict):
            rock_types = geological_data.get('rock_types', [])
            if rock_types:
                rock_type_counts = {}
                for rt in rock_types:
                    rock_type_counts[rt] = rock_type_counts.get(rt, 0) + 1
                total = sum(rock_type_counts.values())
                for rt in sorted(rock_type_counts.keys())[:10]:
                    features.append(rock_type_counts[rt] / total)
            
            fault_density = geological_data.get('fault_density', 0)
            features.append(fault_density)
            
            fold_count = geological_data.get('fold_count', 0)
            features.append(fold_count)
            
            mineralization_index = geological_data.get('mineralization_index', 0)
            features.append(mineralization_index)
            
            alteration_intensity = geological_data.get('alteration_intensity', 0)
            features.append(alteration_intensity)
            
            porosity = geological_data.get('porosity', 0)
            features.append(porosity)
            
            permeability = geological_data.get('permeability', 0)
            features.append(permeability)
            
        elif isinstance(geological_data, np.ndarray):
            if len(geological_data.shape) == 2:
                unique_values, counts = np.unique(geological_data, return_counts=True)
                total = np.sum(counts)
                for i, (val, count) in enumerate(zip(unique_values[:10], counts[:10])):
                    features.append(count / total)
                
                gy, gx = np.gradient(geological_data.astype(float))
                gradient_magnitude = np.sqrt(gx**2 + gy**2)
                features.extend([
                    np.mean(gradient_magnitude),
                    np.std(gradient_magnitude),
                    np.max(gradient_magnitude),
                ])
                
                boundaries = gradient_magnitude > np.percentile(gradient_magnitude, 90)
                features.append(np.sum(boundaries) / boundaries.size)
            else:
                features.extend([
                    np.mean(geological_data),
                    np.std(geological_data),
                    np.min(geological_data),
                    np.max(geological_data),
                ])
        
        elif hasattr(geological_data, 'values'):
            data_array = geological_data.values if hasattr(geological_data, 'values') else np.array(geological_data)
            numeric_cols = []
            for col in range(data_array.shape[1] if len(data_array.shape) > 1 else 1):
                col_data = data_array[:, col] if len(data_array.shape) > 1 else data_array
                try:
                    numeric_data = col_data.astype(float)
                    features.extend([
                        np.mean(numeric_data),
                        np.std(numeric_data),
                        np.min(numeric_data),
                        np.max(numeric_data),
                    ])
                except (ValueError, TypeError):
                    pass
        
        while len(features) < 100:
            features.append(0.0)
        
        return np.array(features[:100])


class ContinuousLearningManager:
    """
    Manager for continuous learning from field validation results.
    
    Implements a feedback loop to improve model performance based on
    new data and validation results.
    """
    
    def __init__(self, model_path, mlflow_tracking_uri=None):
        """
        Initialize the continuous learning manager.
        
        Args:
            model_path: Path to the model
            mlflow_tracking_uri: URI for MLflow tracking server
        """
        self.model_path = model_path
        
        # Set up MLflow tracking
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Load the model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load the model from disk.
        
        Returns:
            Loaded model
        """
        try:
            return mlflow.pytorch.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def update_model(self, new_data, validation_results):
        """
        Update the model with new data and validation results.
        
        Args:
            new_data: New training data
            validation_results: Results from field validation
            
        Returns:
            Updated model
        """
        # Start an MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("update_data_size", len(new_data))
            
            # Prepare the data
            X_train, X_val, y_train, y_val = train_test_split(
                new_data['features'],
                new_data['targets'],
                test_size=0.2,
                random_state=42
            )
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False
            )
            
            # Fine-tune the model
            trainer = pl.Trainer(
                max_epochs=10,
                logger=pl.loggers.MLFlowLogger(
                    experiment_name="mineral_deposit_prediction",
                    tracking_uri=mlflow.get_tracking_uri()
                )
            )
            
            trainer.fit(
                self.model,
                train_loader,
                val_loader
            )
            
            # Log metrics
            mlflow.log_metrics({
                "final_val_loss": trainer.callback_metrics["val_loss"].item(),
                "final_val_acc": trainer.callback_metrics["val_acc"].item()
            })
            
            # Save the updated model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Update the local model
            self.model = trainer.model
            
            return self.model
    
    def evaluate_model(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        # Prepare the data
        X_test = test_data['features']
        y_test = test_data['targets']
        
        # Create data loader
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )
        
        # Evaluate the model
        trainer = pl.Trainer()
        results = trainer.test(self.model, test_loader)
        
        return results[0]


class MineralDepositPredictionService:
    """
    Service for mineral deposit prediction.
    
    Provides a high-level interface for making predictions with uncertainty
    estimation and continuous learning.
    """
    
    def __init__(self, 
                 model_path=None,
                 data_dir=None,
                 mlflow_tracking_uri=None,
                 uncertainty_estimation=True):
        """
        Initialize the prediction service.
        
        Args:
            model_path: Path to the model
            data_dir: Directory containing data files
            mlflow_tracking_uri: URI for MLflow tracking server
            uncertainty_estimation: Whether to estimate prediction uncertainty
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.uncertainty_estimation = uncertainty_estimation
        
        # Set up MLflow tracking
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        
        # Load or train the model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model()
        else:
            self.model = None
        
        # Initialize continuous learning manager if model exists
        if self.model:
            self.learning_manager = ContinuousLearningManager(
                model_path,
                mlflow_tracking_uri
            )
        else:
            self.learning_manager = None
    
    def _load_model(self):
        """
        Load the model from disk.
        
        Returns:
            Loaded model
        """
        try:
            return mlflow.pytorch.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def train_model(self, 
                   geological_data=None,
                   geophysical_data=None,
                   geochemical_data=None,
                   remote_sensing_data=None,
                   historical_data=None,
                   input_dim=None,
                   hidden_dims=None):
        """
        Train a new model.
        
        Args:
            geological_data: Path to geological data file
            geophysical_data: Path to geophysical data file
            geochemical_data: Path to geochemical data file
            remote_sensing_data: Path to remote sensing data file
            historical_data: Path to historical mining data file
            input_dim: Input dimension (required if not inferrable from data)
            hidden_dims: Hidden layer dimensions
            
        Returns:
            Trained model
        """
        # Start an MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("uncertainty_estimation", self.uncertainty_estimation)
            if hidden_dims:
                mlflow.log_param("hidden_dims", hidden_dims)
            
            # Create dataset
            dataset = MineralDepositDataset(
                self.data_dir,
                geological_data,
                geophysical_data,
                geochemical_data,
                remote_sensing_data,
                historical_data
            )
            
            # Split data
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False
            )
            
            # Determine input dimension if not provided
            if input_dim is None:
                sample_features, _ = dataset[0]
                input_dim = sample_features.shape[0]
            
            # Create model
            model = MineralDepositPredictor(
                input_dim=input_dim,
                hidden_dims=hidden_dims or [256, 128, 64],
                uncertainty_estimation=self.uncertainty_estimation
            )
            
            # Train model
            trainer = pl.Trainer(
                max_epochs=50,
                logger=pl.loggers.MLFlowLogger(
                    experiment_name="mineral_deposit_prediction",
                    tracking_uri=mlflow.get_tracking_uri()
                )
            )
            
            trainer.fit(
                model,
                train_loader,
                val_loader
            )
            
            # Test model
            results = trainer.test(model, test_loader)
            
            # Log metrics
            mlflow.log_metrics({
                "test_loss": results[0]["test_loss"],
                "test_acc": results[0]["test_acc"]
            })
            
            # Save model
            mlflow.pytorch.log_model(model, "model")
            
            # Update instance variables
            self.model = model
            self.learning_manager = ContinuousLearningManager(
                mlflow.get_artifact_uri("model"),
                self.mlflow_tracking_uri
            )
            
            return model
    
    def predict(self, features, with_uncertainty=None):
        """
        Make predictions for the given features.
        
        Args:
            features: Input features
            with_uncertainty: Whether to include uncertainty estimation
                (defaults to self.uncertainty_estimation)
                
        Returns:
            Predictions (and uncertainties if requested)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Default to instance setting if not specified
        if with_uncertainty is None:
            with_uncertainty = self.uncertainty_estimation
        
        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            if with_uncertainty and self.uncertainty_estimation:
                # Get predictions with uncertainty
                mean_pred, var_pred = self.model.predict_with_uncertainty(features)
                return mean_pred.numpy(), var_pred.numpy()
            elif self.uncertainty_estimation:
                # Get mean prediction from uncertainty model
                mean, _ = self.model(features)
                return torch.sigmoid(mean).numpy()
            else:
                # Get prediction from point prediction model
                return self.model(features).numpy()
    
    def update_from_validation(self, new_data, validation_results):
        """
        Update the model based on field validation results.
        
        Args:
            new_data: New data from field
            validation_results: Validation results
            
        Returns:
            Updated model
        """
        if self.learning_manager is None:
            raise ValueError("Continuous learning manager not initialized")
        
        return self.learning_manager.update_model(new_data, validation_results)
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        if self.learning_manager is None:
            raise ValueError("Continuous learning manager not initialized")
        
        return self.learning_manager.evaluate_model(test_data)
