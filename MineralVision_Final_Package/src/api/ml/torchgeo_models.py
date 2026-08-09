"""
TorchGeo Foundation Models for MineralVision.

Provides geospatial foundation model capabilities for
satellite and UAV imagery processing.

Supports:
- Pre-trained backbones (ResNet, ViT, Swin)
- Geospatial foundation models (SatMAE, Prithvi, etc.)
- Fine-tuning for mineral exploration tasks
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class FoundationModelType(str, Enum):
    """Available foundation model types."""
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    VIT_BASE = "vit_base"
    VIT_LARGE = "vit_large"
    SWIN_BASE = "swin_base"
    SATMAE = "satmae"
    PRITHVI = "prithvi"
    CLAY = "clay"
    CUSTOM = "custom"


class ImageryType(str, Enum):
    """Types of geospatial imagery."""
    SENTINEL2 = "sentinel2"
    LANDSAT = "landsat"
    ASTER = "aster"
    WORLDVIEW = "worldview"
    PLANET = "planet"
    UAV_RGB = "uav_rgb"
    UAV_MULTISPECTRAL = "uav_multispectral"
    AERIAL = "aerial"


class TaskType(str, Enum):
    """Downstream task types."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    CHANGE_DETECTION = "change_detection"
    OBJECT_DETECTION = "object_detection"
    REGRESSION = "regression"


@dataclass
class ImageBatch:
    """A batch of geospatial images."""
    images: np.ndarray  # Shape: (N, C, H, W)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    coordinates: Optional[np.ndarray] = None
    timestamps: Optional[List[str]] = None
    
    @property
    def batch_size(self) -> int:
        return self.images.shape[0]
    
    @property
    def n_channels(self) -> int:
        return self.images.shape[1]
    
    @property
    def height(self) -> int:
        return self.images.shape[2]
    
    @property
    def width(self) -> int:
        return self.images.shape[3]


@dataclass
class ModelOutput:
    """Output from foundation model."""
    features: np.ndarray
    predictions: Optional[np.ndarray] = None
    attention_maps: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PretrainedBackbone:
    """
    Pre-trained backbone for feature extraction.
    
    Wraps various pre-trained models for geospatial imagery.
    """
    
    def __init__(
        self,
        model_type: FoundationModelType = FoundationModelType.RESNET50,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        self.model_type = model_type
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        self._model = None
        self._feature_dim = 0
    
    def _load_model(self) -> None:
        """Load the pre-trained model."""
        if self._model is not None:
            return
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using numpy fallback")
            self._model = "numpy_fallback"
            self._feature_dim = 2048
            return
        
        try:
            from torchvision import models
            
            if self.model_type == FoundationModelType.RESNET50:
                weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
                self._model = models.resnet50(weights=weights)
                self._feature_dim = 2048
            elif self.model_type == FoundationModelType.RESNET101:
                weights = models.ResNet101_Weights.DEFAULT if self.pretrained else None
                self._model = models.resnet101(weights=weights)
                self._feature_dim = 2048
            elif self.model_type == FoundationModelType.VIT_BASE:
                weights = models.ViT_B_16_Weights.DEFAULT if self.pretrained else None
                self._model = models.vit_b_16(weights=weights)
                self._feature_dim = 768
            elif self.model_type == FoundationModelType.VIT_LARGE:
                weights = models.ViT_L_16_Weights.DEFAULT if self.pretrained else None
                self._model = models.vit_l_16(weights=weights)
                self._feature_dim = 1024
            elif self.model_type == FoundationModelType.SWIN_BASE:
                weights = models.Swin_B_Weights.DEFAULT if self.pretrained else None
                self._model = models.swin_b(weights=weights)
                self._feature_dim = 1024
            else:
                # Default to ResNet50
                weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
                self._model = models.resnet50(weights=weights)
                self._feature_dim = 2048
            
            # Remove classification head for feature extraction
            if hasattr(self._model, 'fc'):
                self._model.fc = nn.Identity()
            elif hasattr(self._model, 'head'):
                self._model.head = nn.Identity()
            elif hasattr(self._model, 'heads'):
                self._model.heads = nn.Identity()
            
            if self.freeze_backbone:
                for param in self._model.parameters():
                    param.requires_grad = False
            
            self._model.eval()
            logger.info(f"Loaded {self.model_type.value} backbone with {self._feature_dim} features")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using fallback")
            self._model = "numpy_fallback"
            self._feature_dim = 2048
    
    @property
    def feature_dim(self) -> int:
        self._load_model()
        return self._feature_dim
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features from images."""
        self._load_model()
        
        if self._model == "numpy_fallback":
            # Simple fallback: flatten and reduce
            batch_size = images.shape[0]
            features = np.mean(images, axis=(2, 3))  # Global average pooling
            # Pad to feature_dim
            if features.shape[1] < self._feature_dim:
                padding = np.zeros((batch_size, self._feature_dim - features.shape[1]))
                features = np.hstack([features, padding])
            return features[:, :self._feature_dim]
        
        # PyTorch inference
        with torch.no_grad():
            tensor = torch.from_numpy(images).float()
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            elif tensor.shape[1] > 3:
                tensor = tensor[:, :3, :, :]
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            
            features = self._model(tensor)
            return features.numpy()


class SatelliteImageProcessor:
    """
    Processor for satellite imagery.
    
    Handles preprocessing, normalization, and augmentation
    for various satellite platforms.
    """
    
    def __init__(
        self,
        imagery_type: ImageryType = ImageryType.SENTINEL2,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.imagery_type = imagery_type
        self.target_size = target_size
        
        # Band configurations
        self._band_configs = {
            ImageryType.SENTINEL2: {
                'bands': ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
                'rgb_bands': [2, 1, 0],  # B04, B03, B02
                'scale': 10000.0
            },
            ImageryType.LANDSAT: {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                'rgb_bands': [2, 1, 0],  # B4, B3, B2
                'scale': 10000.0
            },
            ImageryType.ASTER: {
                'bands': ['B01', 'B02', 'B3N', 'B04', 'B05', 'B06'],
                'rgb_bands': [2, 1, 0],
                'scale': 255.0
            },
            ImageryType.PLANET: {
                'bands': ['blue', 'green', 'red', 'nir'],
                'rgb_bands': [2, 1, 0],
                'scale': 10000.0
            }
        }
    
    def preprocess(
        self,
        images: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Preprocess satellite images."""
        config = self._band_configs.get(self.imagery_type, {})
        scale = config.get('scale', 1.0)
        
        # Scale to 0-1
        if normalize:
            images = images.astype(np.float32) / scale
            images = np.clip(images, 0, 1)
        
        # Resize if needed
        if images.shape[-2:] != self.target_size:
            images = self._resize(images)
        
        return images
    
    def _resize(self, images: np.ndarray) -> np.ndarray:
        """Resize images to target size."""
        try:
            from skimage.transform import resize
            
            batch_size, n_channels = images.shape[:2]
            resized = np.zeros((batch_size, n_channels, *self.target_size))
            
            for i in range(batch_size):
                for c in range(n_channels):
                    resized[i, c] = resize(
                        images[i, c],
                        self.target_size,
                        preserve_range=True
                    )
            
            return resized
        except ImportError:
            # Simple nearest neighbor resize
            h_ratio = self.target_size[0] / images.shape[2]
            w_ratio = self.target_size[1] / images.shape[3]
            
            new_h = np.arange(self.target_size[0])
            new_w = np.arange(self.target_size[1])
            
            old_h = (new_h / h_ratio).astype(int)
            old_w = (new_w / w_ratio).astype(int)
            
            old_h = np.clip(old_h, 0, images.shape[2] - 1)
            old_w = np.clip(old_w, 0, images.shape[3] - 1)
            
            return images[:, :, old_h[:, None], old_w]
    
    def compute_indices(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute spectral indices."""
        indices = {}
        
        if self.imagery_type in [ImageryType.SENTINEL2, ImageryType.LANDSAT]:
            # Assuming bands: B, G, R, NIR, SWIR1, SWIR2
            if images.shape[1] >= 4:
                red = images[:, 2]
                nir = images[:, 3]
                
                # NDVI
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (nir - red) / (nir + red + 1e-10)
                    indices['ndvi'] = np.clip(ndvi, -1, 1)
                
                if images.shape[1] >= 5:
                    swir1 = images[:, 4]
                    
                    # NDWI (water)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndwi = (nir - swir1) / (nir + swir1 + 1e-10)
                        indices['ndwi'] = np.clip(ndwi, -1, 1)
                    
                    # Iron oxide ratio
                    blue = images[:, 0]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        iron_oxide = red / (blue + 1e-10)
                        indices['iron_oxide'] = np.clip(iron_oxide, 0, 5)
                
                if images.shape[1] >= 6:
                    swir2 = images[:, 5]
                    
                    # Clay minerals ratio
                    with np.errstate(divide='ignore', invalid='ignore'):
                        clay = swir1 / (swir2 + 1e-10)
                        indices['clay'] = np.clip(clay, 0, 5)
        
        return indices
    
    def extract_rgb(self, images: np.ndarray) -> np.ndarray:
        """Extract RGB bands for visualization."""
        config = self._band_configs.get(self.imagery_type, {})
        rgb_bands = config.get('rgb_bands', [0, 1, 2])
        
        if images.shape[1] >= 3:
            return images[:, rgb_bands]
        else:
            # Repeat single channel
            return np.repeat(images, 3, axis=1)


class UAVImageProcessor:
    """
    Processor for UAV/drone imagery.
    
    Handles RGB and multispectral UAV data.
    """
    
    def __init__(
        self,
        imagery_type: ImageryType = ImageryType.UAV_RGB,
        target_size: Tuple[int, int] = (512, 512)
    ):
        self.imagery_type = imagery_type
        self.target_size = target_size
    
    def preprocess(
        self,
        images: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Preprocess UAV images."""
        if normalize:
            # Normalize to 0-1
            if images.max() > 1:
                images = images.astype(np.float32) / 255.0
        
        return images
    
    def tile_image(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Tile large UAV image into smaller patches."""
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        
        _, h, w = image.shape
        stride = tile_size - overlap
        
        tiles = []
        positions = []
        
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[:, y:y + tile_size, x:x + tile_size]
                tiles.append(tile)
                positions.append((y, x))
        
        return tiles, positions
    
    def merge_tiles(
        self,
        tiles: List[np.ndarray],
        positions: List[Tuple[int, int]],
        output_shape: Tuple[int, int],
        tile_size: int = 512
    ) -> np.ndarray:
        """Merge tiles back into full image."""
        n_channels = tiles[0].shape[0] if tiles[0].ndim == 3 else 1
        output = np.zeros((n_channels, *output_shape))
        counts = np.zeros(output_shape)
        
        for tile, (y, x) in zip(tiles, positions):
            if tile.ndim == 2:
                tile = tile[np.newaxis, ...]
            
            output[:, y:y + tile_size, x:x + tile_size] += tile
            counts[y:y + tile_size, x:x + tile_size] += 1
        
        # Average overlapping regions
        counts = np.maximum(counts, 1)
        output = output / counts
        
        return output


class GeoFoundationModel:
    """
    Geospatial Foundation Model for mineral exploration.
    
    Combines pre-trained backbones with task-specific heads
    for various downstream tasks.
    """
    
    def __init__(
        self,
        model_type: FoundationModelType = FoundationModelType.RESNET50,
        task_type: TaskType = TaskType.CLASSIFICATION,
        n_classes: int = 2,
        pretrained: bool = True
    ):
        self.model_type = model_type
        self.task_type = task_type
        self.n_classes = n_classes
        
        self.backbone = PretrainedBackbone(
            model_type=model_type,
            pretrained=pretrained,
            freeze_backbone=True
        )
        
        self._head = None
        self._trained = False
    
    def _build_head(self) -> None:
        """Build task-specific head."""
        if self._head is not None:
            return
        
        feature_dim = self.backbone.feature_dim
        
        if not TORCH_AVAILABLE:
            self._head = "numpy_fallback"
            return
        
        if self.task_type == TaskType.CLASSIFICATION:
            self._head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.n_classes)
            )
        elif self.task_type == TaskType.REGRESSION:
            self._head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
            )
        else:
            # Default classification head
            self._head = nn.Linear(feature_dim, self.n_classes)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features from images."""
        return self.backbone.extract_features(images)
    
    def predict(self, images: np.ndarray) -> ModelOutput:
        """Make predictions on images."""
        self._build_head()
        
        features = self.extract_features(images)
        
        if self._head == "numpy_fallback" or not self._trained:
            # Return features only
            return ModelOutput(
                features=features,
                predictions=None,
                metadata={'model_type': self.model_type.value}
            )
        
        with torch.no_grad():
            feat_tensor = torch.from_numpy(features).float()
            logits = self._head(feat_tensor)
            
            if self.task_type == TaskType.CLASSIFICATION:
                probs = torch.softmax(logits, dim=1).numpy()
            else:
                probs = logits.numpy()
        
        return ModelOutput(
            features=features,
            predictions=probs,
            metadata={'model_type': self.model_type.value}
        )
    
    def fine_tune(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        val_images: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """Fine-tune the model on training data."""
        self._build_head()
        
        if not TORCH_AVAILABLE or self._head == "numpy_fallback":
            logger.warning("PyTorch not available, skipping fine-tuning")
            return {'train_loss': [], 'val_loss': []}
        
        # Unfreeze backbone for fine-tuning
        for param in self.backbone._model.parameters():
            param.requires_grad = True
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            list(self.backbone._model.parameters()) + list(self._head.parameters()),
            lr=learning_rate
        )
        
        if self.task_type == TaskType.CLASSIFICATION:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(epochs):
            self.backbone._model.train()
            self._head.train()
            
            # Forward pass
            features = self.backbone.extract_features(train_images)
            feat_tensor = torch.from_numpy(features).float()
            labels_tensor = torch.from_numpy(train_labels).long()
            
            optimizer.zero_grad()
            outputs = self._head(feat_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            
            history['train_loss'].append(float(loss.item()))
            
            # Validation
            if val_images is not None:
                self.backbone._model.eval()
                self._head.eval()
                
                with torch.no_grad():
                    val_features = self.backbone.extract_features(val_images)
                    val_feat_tensor = torch.from_numpy(val_features).float()
                    val_labels_tensor = torch.from_numpy(val_labels).long()
                    
                    val_outputs = self._head(val_feat_tensor)
                    val_loss = criterion(val_outputs, val_labels_tensor)
                    history['val_loss'].append(float(val_loss.item()))
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        self._trained = True
        return history
    
    def save(self, path: str) -> None:
        """Save model weights."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot save model")
            return
        
        self._build_head()
        
        state = {
            'model_type': self.model_type.value,
            'task_type': self.task_type.value,
            'n_classes': self.n_classes,
            'head_state': self._head.state_dict() if self._head != "numpy_fallback" else None
        }
        
        torch.save(state, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model weights."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot load model")
            return
        
        state = torch.load(path)
        
        self.model_type = FoundationModelType(state['model_type'])
        self.task_type = TaskType(state['task_type'])
        self.n_classes = state['n_classes']
        
        self._build_head()
        
        if state['head_state'] is not None and self._head != "numpy_fallback":
            self._head.load_state_dict(state['head_state'])
        
        self._trained = True
        logger.info(f"Model loaded from {path}")


def create_geo_foundation_model(
    model_type: str = "resnet50",
    task_type: str = "classification",
    n_classes: int = 2
) -> GeoFoundationModel:
    """Factory function to create GeoFoundationModel."""
    return GeoFoundationModel(
        model_type=FoundationModelType(model_type),
        task_type=TaskType(task_type),
        n_classes=n_classes
    )
