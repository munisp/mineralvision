"""
Domain-Specific Foundation Models for MineralVision.

This module provides:
- Geoscience pretraining pipeline
- Self-supervised learning on unlabeled geoscience data
- Modality adapters for different data types
- Fine-tuning for specific commodities/regions
- Transfer learning utilities

Builds strong geological priors rather than relying solely on external pretrained models.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class DataModality(Enum):
    """Geoscience data modalities."""
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    DEM = "dem"
    MAGNETICS = "magnetics"
    RADIOMETRICS = "radiometrics"
    GRAVITY = "gravity"
    SEISMIC = "seismic"
    GPR = "gpr"
    GEOCHEMISTRY = "geochemistry"
    DRILL_LOGS = "drill_logs"
    GEOLOGICAL_TEXT = "geological_text"


class PretrainingTask(Enum):
    """Self-supervised pretraining tasks."""
    MASKED_PATCH = "masked_patch"           # Mask and predict patches
    CONTRASTIVE = "contrastive"             # Contrastive learning
    ROTATION = "rotation"                   # Predict rotation
    JIGSAW = "jigsaw"                       # Solve jigsaw puzzle
    COLORIZATION = "colorization"           # Predict colors
    INPAINTING = "inpainting"               # Fill missing regions
    TEMPORAL = "temporal"                   # Temporal prediction
    CROSS_MODAL = "cross_modal"             # Cross-modality prediction


class FineTuningStrategy(Enum):
    """Fine-tuning strategies."""
    FULL = "full"                           # Fine-tune all layers
    LINEAR_PROBE = "linear_probe"           # Only train classifier
    ADAPTER = "adapter"                     # Train adapter layers only
    LORA = "lora"                           # Low-rank adaptation
    PREFIX_TUNING = "prefix_tuning"         # Prefix tuning


@dataclass
class ModalityConfig:
    """Configuration for a data modality."""
    modality: DataModality
    input_channels: int
    patch_size: int
    embedding_dim: int
    normalization: str  # 'standard', 'minmax', 'robust'
    augmentations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'modality': self.modality.value,
            'input_channels': self.input_channels,
            'patch_size': self.patch_size,
            'embedding_dim': self.embedding_dim,
            'normalization': self.normalization,
            'augmentations': self.augmentations
        }


@dataclass
class PretrainingConfig:
    """Configuration for pretraining."""
    task: PretrainingTask
    modalities: List[DataModality]
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_epochs: int
    mask_ratio: float = 0.75
    temperature: float = 0.07
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task.value,
            'modalities': [m.value for m in self.modalities],
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'warmup_epochs': self.warmup_epochs,
            'mask_ratio': self.mask_ratio,
            'temperature': self.temperature
        }


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    strategy: FineTuningStrategy
    target_task: str
    learning_rate: float
    epochs: int
    freeze_backbone: bool
    adapter_dim: int = 64
    lora_rank: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'target_task': self.target_task,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'freeze_backbone': self.freeze_backbone,
            'adapter_dim': self.adapter_dim,
            'lora_rank': self.lora_rank
        }


@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata."""
    checkpoint_id: str
    model_name: str
    version: str
    modalities: List[DataModality]
    pretraining_task: PretrainingTask
    pretraining_epochs: int
    pretraining_loss: float
    created_at: datetime
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checkpoint_id': self.checkpoint_id,
            'model_name': self.model_name,
            'version': self.version,
            'modalities': [m.value for m in self.modalities],
            'pretraining_task': self.pretraining_task.value,
            'pretraining_epochs': self.pretraining_epochs,
            'pretraining_loss': self.pretraining_loss,
            'created_at': self.created_at.isoformat(),
            'file_path': self.file_path,
            'metadata': self.metadata
        }


class ModalityAdapter(ABC):
    """
    Abstract base class for modality adapters.
    
    Adapters transform raw data into a common embedding space.
    """
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data to embeddings."""
        pass
    
    @abstractmethod
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode embeddings back to data space."""
        pass
    
    @abstractmethod
    def get_config(self) -> ModalityConfig:
        """Get adapter configuration."""
        pass


class MultispectralAdapter(ModalityAdapter):
    """Adapter for multispectral imagery."""
    
    def __init__(self, n_bands: int = 4, patch_size: int = 16, embedding_dim: int = 768):
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode multispectral patches to embeddings.
        
        Args:
            data: (batch, bands, height, width)
            
        Returns:
            Embeddings (batch, n_patches, embedding_dim)
        """
        batch_size = data.shape[0]
        h, w = data.shape[2], data.shape[3]
        
        # Calculate number of patches
        n_patches_h = h // self.patch_size
        n_patches_w = w // self.patch_size
        n_patches = n_patches_h * n_patches_w
        
        # Flatten patches (simplified - in production use conv projection)
        patch_dim = self.n_bands * self.patch_size * self.patch_size
        
        # Random projection to embedding dim (placeholder for learned projection)
        np.random.seed(42)
        projection = np.random.randn(patch_dim, self.embedding_dim) / np.sqrt(patch_dim)
        
        # Reshape and project
        patches = data.reshape(batch_size, -1, patch_dim)
        embeddings = patches @ projection
        
        return embeddings
    
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode embeddings back to patches."""
        # Simplified inverse projection
        batch_size, n_patches, _ = embeddings.shape
        patch_dim = self.n_bands * self.patch_size * self.patch_size
        
        np.random.seed(42)
        projection = np.random.randn(patch_dim, self.embedding_dim) / np.sqrt(patch_dim)
        
        # Pseudo-inverse
        projection_inv = np.linalg.pinv(projection)
        patches = embeddings @ projection_inv.T
        
        return patches
    
    def get_config(self) -> ModalityConfig:
        return ModalityConfig(
            modality=DataModality.MULTISPECTRAL,
            input_channels=self.n_bands,
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            normalization='standard',
            augmentations=['flip', 'rotate', 'color_jitter']
        )


class GeophysicsAdapter(ModalityAdapter):
    """Adapter for geophysical grids (magnetics, radiometrics, gravity)."""
    
    def __init__(self, modality: DataModality, patch_size: int = 32, embedding_dim: int = 768):
        self.modality = modality
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode geophysical grid patches to embeddings.
        
        Args:
            data: (batch, channels, height, width)
            
        Returns:
            Embeddings (batch, n_patches, embedding_dim)
        """
        batch_size = data.shape[0]
        n_channels = data.shape[1]
        
        # Flatten patches
        patch_dim = n_channels * self.patch_size * self.patch_size
        
        np.random.seed(43)
        projection = np.random.randn(patch_dim, self.embedding_dim) / np.sqrt(patch_dim)
        
        patches = data.reshape(batch_size, -1, patch_dim)
        embeddings = patches @ projection
        
        return embeddings
    
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode embeddings back to patches."""
        batch_size, n_patches, _ = embeddings.shape
        n_channels = 1  # Single channel for geophysics
        patch_dim = n_channels * self.patch_size * self.patch_size
        
        np.random.seed(43)
        projection = np.random.randn(patch_dim, self.embedding_dim) / np.sqrt(patch_dim)
        projection_inv = np.linalg.pinv(projection)
        
        patches = embeddings @ projection_inv.T
        return patches
    
    def get_config(self) -> ModalityConfig:
        return ModalityConfig(
            modality=self.modality,
            input_channels=1,
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            normalization='robust',
            augmentations=['flip', 'rotate', 'noise']
        )


class TextAdapter(ModalityAdapter):
    """Adapter for geological text (reports, logs)."""
    
    def __init__(self, vocab_size: int = 30000, max_length: int = 512, embedding_dim: int = 768):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode tokenized text to embeddings.
        
        Args:
            data: (batch, sequence_length) token IDs
            
        Returns:
            Embeddings (batch, sequence_length, embedding_dim)
        """
        batch_size, seq_len = data.shape
        
        # Token embeddings (simplified)
        np.random.seed(44)
        token_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) / np.sqrt(self.embedding_dim)
        
        # Look up embeddings
        embeddings = token_embeddings[data.astype(int)]
        
        # Add positional embeddings
        positions = np.arange(seq_len)
        pos_embeddings = np.sin(positions[:, None] / 10000 ** (np.arange(self.embedding_dim) / self.embedding_dim))
        
        embeddings = embeddings + pos_embeddings
        
        return embeddings
    
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode embeddings to token logits."""
        np.random.seed(44)
        token_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) / np.sqrt(self.embedding_dim)
        
        # Compute similarity to all tokens
        logits = embeddings @ token_embeddings.T
        
        return logits
    
    def get_config(self) -> ModalityConfig:
        return ModalityConfig(
            modality=DataModality.GEOLOGICAL_TEXT,
            input_channels=1,
            patch_size=1,
            embedding_dim=self.embedding_dim,
            normalization='none',
            augmentations=['mask', 'shuffle']
        )


class PretrainingPipeline:
    """
    Self-supervised pretraining pipeline.
    
    Supports multiple pretraining tasks for geoscience data.
    """
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.adapters: Dict[DataModality, ModalityAdapter] = {}
        self._setup_adapters()
        
    def _setup_adapters(self):
        """Setup modality adapters."""
        for modality in self.config.modalities:
            if modality == DataModality.MULTISPECTRAL:
                self.adapters[modality] = MultispectralAdapter()
            elif modality in [DataModality.MAGNETICS, DataModality.RADIOMETRICS, DataModality.GRAVITY]:
                self.adapters[modality] = GeophysicsAdapter(modality)
            elif modality == DataModality.GEOLOGICAL_TEXT:
                self.adapters[modality] = TextAdapter()
                
    def create_masked_patches(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create masked patch prediction task.
        
        Args:
            embeddings: (batch, n_patches, embedding_dim)
            
        Returns:
            Tuple of (masked_embeddings, mask, targets)
        """
        batch_size, n_patches, embedding_dim = embeddings.shape
        
        # Create random mask
        n_masked = int(n_patches * self.config.mask_ratio)
        mask = np.zeros((batch_size, n_patches), dtype=bool)
        
        for i in range(batch_size):
            masked_indices = np.random.choice(n_patches, n_masked, replace=False)
            mask[i, masked_indices] = True
            
        # Create masked embeddings
        masked_embeddings = embeddings.copy()
        mask_token = np.zeros(embedding_dim)  # Learnable mask token
        masked_embeddings[mask] = mask_token
        
        # Targets are original embeddings at masked positions
        targets = embeddings[mask].reshape(batch_size, n_masked, embedding_dim)
        
        return masked_embeddings, mask, targets
    
    def create_contrastive_pairs(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create contrastive learning pairs.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (view1, view2) augmented views
        """
        # Apply different augmentations to create two views
        view1 = self._augment(data, strength=0.5)
        view2 = self._augment(data, strength=0.5)
        
        return view1, view2
    
    def _augment(self, data: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply random augmentations."""
        augmented = data.copy()
        
        # Random flip
        if np.random.random() < strength:
            augmented = np.flip(augmented, axis=-1)
            
        # Random rotation (90 degrees)
        if np.random.random() < strength:
            k = np.random.randint(1, 4)
            augmented = np.rot90(augmented, k, axes=(-2, -1))
            
        # Random noise
        if np.random.random() < strength:
            noise = np.random.randn(*augmented.shape) * 0.1
            augmented = augmented + noise
            
        return augmented
    
    def compute_contrastive_loss(self, embeddings1: np.ndarray, 
                                embeddings2: np.ndarray) -> float:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            embeddings1: First view embeddings (batch, dim)
            embeddings2: Second view embeddings (batch, dim)
            
        Returns:
            Loss value
        """
        # Normalize embeddings
        embeddings1 = embeddings1 / (np.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-8)
        embeddings2 = embeddings2 / (np.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity = embeddings1 @ embeddings2.T / self.config.temperature
        
        # Labels are diagonal (positive pairs)
        batch_size = embeddings1.shape[0]
        labels = np.arange(batch_size)
        
        # Cross-entropy loss
        exp_sim = np.exp(similarity - np.max(similarity, axis=1, keepdims=True))
        softmax = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        
        loss = -np.mean(np.log(softmax[np.arange(batch_size), labels] + 1e-8))
        
        return loss
    
    def pretrain_step(self, batch: Dict[DataModality, np.ndarray]) -> Dict[str, float]:
        """
        Execute one pretraining step.
        
        Args:
            batch: Dict of modality to data batch
            
        Returns:
            Dict of loss values
        """
        losses = {}
        
        for modality, data in batch.items():
            if modality not in self.adapters:
                continue
                
            adapter = self.adapters[modality]
            embeddings = adapter.encode(data)
            
            if self.config.task == PretrainingTask.MASKED_PATCH:
                masked, mask, targets = self.create_masked_patches(embeddings)
                # Simplified MSE loss
                predictions = masked[mask].reshape(targets.shape)
                loss = np.mean((predictions - targets) ** 2)
                losses[f'{modality.value}_masked'] = loss
                
            elif self.config.task == PretrainingTask.CONTRASTIVE:
                view1, view2 = self.create_contrastive_pairs(data)
                emb1 = adapter.encode(view1).mean(axis=1)  # Pool to single vector
                emb2 = adapter.encode(view2).mean(axis=1)
                loss = self.compute_contrastive_loss(emb1, emb2)
                losses[f'{modality.value}_contrastive'] = loss
                
        losses['total'] = sum(losses.values())
        return losses


class FineTuningPipeline:
    """
    Fine-tuning pipeline for downstream tasks.
    
    Supports multiple fine-tuning strategies.
    """
    
    def __init__(self, config: FineTuningConfig, base_model: Any = None):
        self.config = config
        self.base_model = base_model
        self.adapter_weights: Optional[np.ndarray] = None
        self.classifier_weights: Optional[np.ndarray] = None
        
    def setup_adapter(self, input_dim: int, output_dim: int):
        """Setup adapter layers for adapter tuning."""
        # Down projection
        self.adapter_down = np.random.randn(input_dim, self.config.adapter_dim) / np.sqrt(input_dim)
        # Up projection
        self.adapter_up = np.random.randn(self.config.adapter_dim, input_dim) / np.sqrt(self.config.adapter_dim)
        # Classifier
        self.classifier_weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        
    def setup_lora(self, input_dim: int):
        """Setup LoRA layers."""
        # Low-rank matrices
        self.lora_A = np.random.randn(input_dim, self.config.lora_rank) / np.sqrt(input_dim)
        self.lora_B = np.zeros((self.config.lora_rank, input_dim))
        
    def apply_adapter(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply adapter transformation."""
        if self.adapter_down is None:
            return embeddings
            
        # Down -> ReLU -> Up
        down = embeddings @ self.adapter_down
        down = np.maximum(down, 0)  # ReLU
        up = down @ self.adapter_up
        
        # Residual connection
        return embeddings + up
    
    def apply_lora(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply LoRA transformation."""
        if self.lora_A is None:
            return embeddings
            
        # Low-rank update
        lora_output = embeddings @ self.lora_A @ self.lora_B
        
        return embeddings + lora_output
    
    def finetune_step(self, embeddings: np.ndarray, 
                     labels: np.ndarray) -> Dict[str, float]:
        """
        Execute one fine-tuning step.
        
        Args:
            embeddings: Input embeddings
            labels: Target labels
            
        Returns:
            Dict of loss values
        """
        # Apply fine-tuning strategy
        if self.config.strategy == FineTuningStrategy.ADAPTER:
            embeddings = self.apply_adapter(embeddings)
        elif self.config.strategy == FineTuningStrategy.LORA:
            embeddings = self.apply_lora(embeddings)
            
        # Pool embeddings
        pooled = embeddings.mean(axis=1)
        
        # Classify
        if self.classifier_weights is not None:
            logits = pooled @ self.classifier_weights
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy loss
            n_classes = logits.shape[1]
            one_hot = np.eye(n_classes)[labels.astype(int)]
            loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-8), axis=1))
            
            # Accuracy
            predictions = np.argmax(logits, axis=1)
            accuracy = np.mean(predictions == labels)
            
            return {'loss': loss, 'accuracy': accuracy}
            
        return {'loss': 0.0}


class GeoscienceFoundationModel:
    """
    Complete geoscience foundation model.
    
    Integrates pretraining and fine-tuning pipelines.
    """
    
    def __init__(self, model_name: str = "geofm"):
        self.model_name = model_name
        self.version = "1.0.0"
        self.modality_adapters: Dict[DataModality, ModalityAdapter] = {}
        self.checkpoints: List[ModelCheckpoint] = []
        self._is_pretrained = False
        
    def add_modality(self, modality: DataModality, adapter: ModalityAdapter):
        """Add a modality adapter."""
        self.modality_adapters[modality] = adapter
        
    def pretrain(self, data_loader: Callable,
                config: PretrainingConfig,
                n_steps: int = 1000) -> Dict[str, Any]:
        """
        Pretrain the foundation model.
        
        Args:
            data_loader: Function that yields batches
            config: Pretraining configuration
            n_steps: Number of training steps
            
        Returns:
            Training history
        """
        pipeline = PretrainingPipeline(config)
        
        # Copy adapters
        for modality, adapter in self.modality_adapters.items():
            pipeline.adapters[modality] = adapter
            
        history = {'losses': [], 'steps': []}
        
        for step in range(n_steps):
            # Get batch (simulated)
            batch = {}
            for modality in config.modalities:
                if modality in self.modality_adapters:
                    # Simulated batch
                    batch[modality] = np.random.randn(config.batch_size, 4, 64, 64)
                    
            losses = pipeline.pretrain_step(batch)
            
            history['losses'].append(losses['total'])
            history['steps'].append(step)
            
            if step % 100 == 0:
                logger.info(f"Pretrain step {step}: loss = {losses['total']:.4f}")
                
        self._is_pretrained = True
        
        # Save checkpoint
        checkpoint = ModelCheckpoint(
            checkpoint_id=f"ckpt_{uuid.uuid4().hex[:8]}",
            model_name=self.model_name,
            version=self.version,
            modalities=config.modalities,
            pretraining_task=config.task,
            pretraining_epochs=config.epochs,
            pretraining_loss=history['losses'][-1] if history['losses'] else 0,
            created_at=datetime.now(),
            file_path=f"/models/{self.model_name}_{self.version}.pt"
        )
        self.checkpoints.append(checkpoint)
        
        return history
    
    def finetune(self, train_data: np.ndarray,
                train_labels: np.ndarray,
                config: FineTuningConfig,
                n_steps: int = 100) -> Dict[str, Any]:
        """
        Fine-tune for a downstream task.
        
        Args:
            train_data: Training data
            train_labels: Training labels
            config: Fine-tuning configuration
            n_steps: Number of training steps
            
        Returns:
            Training history
        """
        if not self._is_pretrained:
            logger.warning("Model not pretrained, fine-tuning from scratch")
            
        pipeline = FineTuningPipeline(config)
        
        # Setup classifier
        n_classes = len(np.unique(train_labels))
        embedding_dim = 768  # Default
        pipeline.setup_adapter(embedding_dim, n_classes)
        
        if config.strategy == FineTuningStrategy.LORA:
            pipeline.setup_lora(embedding_dim)
            
        history = {'losses': [], 'accuracies': [], 'steps': []}
        
        for step in range(n_steps):
            # Get batch
            batch_idx = np.random.choice(len(train_data), config.batch_size if hasattr(config, 'batch_size') else 32)
            batch_data = train_data[batch_idx]
            batch_labels = train_labels[batch_idx]
            
            # Encode to embeddings (simplified)
            embeddings = np.random.randn(len(batch_idx), 16, embedding_dim)
            
            metrics = pipeline.finetune_step(embeddings, batch_labels)
            
            history['losses'].append(metrics['loss'])
            history['accuracies'].append(metrics.get('accuracy', 0))
            history['steps'].append(step)
            
            if step % 10 == 0:
                logger.info(f"Finetune step {step}: loss = {metrics['loss']:.4f}, acc = {metrics.get('accuracy', 0):.4f}")
                
        return history
    
    def encode(self, data: np.ndarray, modality: DataModality) -> np.ndarray:
        """
        Encode data to embeddings.
        
        Args:
            data: Input data
            modality: Data modality
            
        Returns:
            Embeddings
        """
        if modality not in self.modality_adapters:
            raise ValueError(f"No adapter for modality: {modality}")
            
        adapter = self.modality_adapters[modality]
        return adapter.encode(data)
    
    def get_checkpoint(self, checkpoint_id: str = None) -> Optional[ModelCheckpoint]:
        """Get model checkpoint."""
        if checkpoint_id:
            for ckpt in self.checkpoints:
                if ckpt.checkpoint_id == checkpoint_id:
                    return ckpt
            return None
        elif self.checkpoints:
            return self.checkpoints[-1]
        return None


class FoundationModelRegistry:
    """
    Registry for foundation models.
    
    Manages model versions and checkpoints.
    """
    
    def __init__(self):
        self._models: Dict[str, GeoscienceFoundationModel] = {}
        self._default_configs: Dict[str, PretrainingConfig] = {}
        self._setup_defaults()
        
    def _setup_defaults(self):
        """Setup default configurations."""
        # Multispectral + DEM pretraining
        self._default_configs['geofm_imagery'] = PretrainingConfig(
            task=PretrainingTask.MASKED_PATCH,
            modalities=[DataModality.MULTISPECTRAL, DataModality.DEM],
            batch_size=32,
            learning_rate=1e-4,
            epochs=100,
            warmup_epochs=10,
            mask_ratio=0.75
        )
        
        # Geophysics pretraining
        self._default_configs['geofm_geophysics'] = PretrainingConfig(
            task=PretrainingTask.CONTRASTIVE,
            modalities=[DataModality.MAGNETICS, DataModality.RADIOMETRICS, DataModality.GRAVITY],
            batch_size=64,
            learning_rate=1e-4,
            epochs=50,
            warmup_epochs=5,
            temperature=0.1
        )
        
        # Text pretraining
        self._default_configs['geofm_text'] = PretrainingConfig(
            task=PretrainingTask.MASKED_PATCH,
            modalities=[DataModality.GEOLOGICAL_TEXT],
            batch_size=16,
            learning_rate=5e-5,
            epochs=20,
            warmup_epochs=2,
            mask_ratio=0.15
        )
        
    def register_model(self, model: GeoscienceFoundationModel):
        """Register a model."""
        self._models[model.model_name] = model
        
    def get_model(self, model_name: str) -> Optional[GeoscienceFoundationModel]:
        """Get a registered model."""
        return self._models.get(model_name)
    
    def create_model(self, model_name: str,
                    modalities: List[DataModality]) -> GeoscienceFoundationModel:
        """
        Create a new foundation model.
        
        Args:
            model_name: Model name
            modalities: Supported modalities
            
        Returns:
            GeoscienceFoundationModel
        """
        model = GeoscienceFoundationModel(model_name)
        
        for modality in modalities:
            if modality == DataModality.MULTISPECTRAL:
                model.add_modality(modality, MultispectralAdapter())
            elif modality in [DataModality.MAGNETICS, DataModality.RADIOMETRICS, DataModality.GRAVITY]:
                model.add_modality(modality, GeophysicsAdapter(modality))
            elif modality == DataModality.GEOLOGICAL_TEXT:
                model.add_modality(modality, TextAdapter())
                
        self.register_model(model)
        return model
    
    def get_default_config(self, config_name: str) -> Optional[PretrainingConfig]:
        """Get default pretraining config."""
        return self._default_configs.get(config_name)
    
    def list_models(self) -> List[str]:
        """List registered models."""
        return list(self._models.keys())
    
    def list_configs(self) -> List[str]:
        """List available default configs."""
        return list(self._default_configs.keys())


# Factory functions
def create_foundation_model(model_name: str = "geofm",
                           modalities: List[str] = None) -> GeoscienceFoundationModel:
    """
    Create a geoscience foundation model.
    
    Args:
        model_name: Model name
        modalities: List of modality names
        
    Returns:
        GeoscienceFoundationModel
    """
    modalities = modalities or ['multispectral', 'magnetics', 'radiometrics']
    modality_enums = [DataModality(m) for m in modalities]
    
    registry = FoundationModelRegistry()
    return registry.create_model(model_name, modality_enums)


def create_pretraining_config(task: str = "masked_patch",
                             modalities: List[str] = None,
                             epochs: int = 100) -> PretrainingConfig:
    """Create pretraining configuration."""
    modalities = modalities or ['multispectral']
    
    return PretrainingConfig(
        task=PretrainingTask(task),
        modalities=[DataModality(m) for m in modalities],
        batch_size=32,
        learning_rate=1e-4,
        epochs=epochs,
        warmup_epochs=epochs // 10
    )


def create_finetuning_config(strategy: str = "adapter",
                            target_task: str = "classification",
                            epochs: int = 50) -> FineTuningConfig:
    """Create fine-tuning configuration."""
    return FineTuningConfig(
        strategy=FineTuningStrategy(strategy),
        target_task=target_task,
        learning_rate=1e-4,
        epochs=epochs,
        freeze_backbone=strategy != "full"
    )
