"""
Deep Learning-based Sensor Fusion for MineralVision.

This module provides neural network-based fusion methods for combining
multi-modal sensor data using attention mechanisms, transformers, and
feature-level fusion networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from .core import SensorData, SensorFusionAlgorithm, SensorType, DataDimension

logger = logging.getLogger(__name__)


class FusionArchitecture(Enum):
    """Types of deep learning fusion architectures."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    ATTENTION_FUSION = "attention_fusion"
    TRANSFORMER_FUSION = "transformer_fusion"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    MULTIMODAL_AUTOENCODER = "multimodal_autoencoder"


@dataclass
class DeepFusionConfig:
    """Configuration for deep learning fusion models."""
    input_dims: List[int] = field(default_factory=lambda: [256, 256])
    hidden_dim: int = 512
    output_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    architecture: FusionArchitecture = FusionArchitecture.ATTENTION_FUSION
    use_residual: bool = True
    use_layer_norm: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based fusion."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SensorEncoder(nn.Module):
    """Encoder network for individual sensor modalities."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(next_dim))
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                
            current_dim = next_dim
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sensor data."""
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing different sensor modalities."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query tensor from one modality
            key: Key tensor from another modality
            value: Value tensor from another modality
            mask: Optional attention mask
            
        Returns:
            Attended features
        """
        batch_size = query.size(0)
        
        # Project to queries, keys, values
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        
        # Residual connection and layer norm
        out = self.layer_norm(query + out)
        
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for feature fusion."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head self-attention."""
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerFusionBlock(nn.Module):
    """Transformer block for multi-modal fusion."""
    
    def __init__(self, dim: int, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply transformer block."""
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class EarlyFusionNetwork(nn.Module):
    """
    Early fusion network that concatenates features before processing.
    
    Simple but effective for sensors with similar characteristics.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        total_input_dim = sum(config.input_dims)
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(total_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse inputs through early concatenation.
        
        Args:
            inputs: List of sensor feature tensors
            
        Returns:
            Fused features
        """
        # Concatenate all inputs
        concatenated = torch.cat(inputs, dim=-1)
        
        # Process through fusion network
        return self.fusion_net(concatenated)


class LateFusionNetwork(nn.Module):
    """
    Late fusion network that processes each modality separately
    then combines the outputs.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        
        # Individual encoders for each modality
        self.encoders = nn.ModuleList([
            SensorEncoder(
                input_dim=dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                dropout=config.dropout
            )
            for dim in config.input_dims
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(config.input_dims), config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse inputs through late fusion.
        
        Args:
            inputs: List of sensor feature tensors
            
        Returns:
            Fused features
        """
        # Encode each modality
        encoded = [encoder(inp) for encoder, inp in zip(self.encoders, inputs)]
        
        # Concatenate and fuse
        concatenated = torch.cat(encoded, dim=-1)
        return self.fusion(concatenated)


class AttentionFusionNetwork(nn.Module):
    """
    Attention-based fusion network that learns to weight different modalities.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        
        # Individual encoders
        self.encoders = nn.ModuleList([
            SensorEncoder(
                input_dim=dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                dropout=config.dropout
            )
            for dim in config.input_dims
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse inputs using attention weighting.
        
        Args:
            inputs: List of sensor feature tensors
            
        Returns:
            Tuple of (fused features, attention weights)
        """
        # Encode each modality
        encoded = torch.stack([encoder(inp) for encoder, inp in zip(self.encoders, inputs)], dim=1)
        
        # Compute attention weights
        attn_scores = self.attention(encoded).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum
        fused = (encoded * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # Output projection
        output = self.output_proj(fused)
        
        return output, attn_weights


class TransformerFusionNetwork(nn.Module):
    """
    Transformer-based fusion network for complex multi-modal interactions.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        
        # Individual encoders to project to common dimension
        self.encoders = nn.ModuleList([
            SensorEncoder(
                input_dim=dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                dropout=config.dropout
            )
            for dim in config.input_dims
        ])
        
        # Modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(len(config.input_dims), config.hidden_dim) * 0.02
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, dropout=config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerFusionBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse inputs using transformer architecture.
        
        Args:
            inputs: List of sensor feature tensors
            
        Returns:
            Fused features
        """
        batch_size = inputs[0].size(0)
        
        # Encode each modality and add modality embeddings
        encoded = []
        for i, (encoder, inp) in enumerate(zip(self.encoders, inputs)):
            enc = encoder(inp)
            if enc.dim() == 2:
                enc = enc.unsqueeze(1)
            enc = enc + self.modality_embeddings[i]
            encoded.append(enc)
            
        # Concatenate all modalities
        tokens = torch.cat(encoded, dim=1)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Apply positional encoding
        tokens = self.pos_encoding(tokens.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
            
        # Use CLS token output
        cls_output = tokens[:, 0]
        
        # Output projection
        output = self.output_proj(self.output_norm(cls_output))
        
        return output


class CrossModalAttentionNetwork(nn.Module):
    """
    Cross-modal attention network for bidirectional information flow
    between modalities.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        num_modalities = len(config.input_dims)
        
        # Individual encoders
        self.encoders = nn.ModuleList([
            SensorEncoder(
                input_dim=dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                dropout=config.dropout
            )
            for dim in config.input_dims
        ])
        
        # Cross-modal attention layers (all pairs)
        self.cross_attentions = nn.ModuleDict()
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i != j:
                    self.cross_attentions[f"{i}_{j}"] = CrossModalAttention(
                        dim=config.hidden_dim,
                        num_heads=config.num_heads,
                        dropout=config.dropout
                    )
                    
        # Self-attention for each modality
        self.self_attentions = nn.ModuleList([
            MultiHeadSelfAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(num_modalities)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * num_modalities, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse inputs using cross-modal attention.
        
        Args:
            inputs: List of sensor feature tensors
            
        Returns:
            Fused features
        """
        num_modalities = len(inputs)
        
        # Encode each modality
        encoded = [encoder(inp) for encoder, inp in zip(self.encoders, inputs)]
        
        # Ensure 3D tensors for attention
        for i in range(num_modalities):
            if encoded[i].dim() == 2:
                encoded[i] = encoded[i].unsqueeze(1)
                
        # Apply cross-modal attention
        attended = []
        for i in range(num_modalities):
            # Start with self-attention
            feat = self.self_attentions[i](encoded[i])
            
            # Add cross-modal attention from other modalities
            for j in range(num_modalities):
                if i != j:
                    cross_feat = self.cross_attentions[f"{i}_{j}"](
                        feat, encoded[j], encoded[j]
                    )
                    feat = feat + cross_feat
                    
            attended.append(feat.mean(dim=1))  # Pool over sequence dimension
            
        # Concatenate and fuse
        concatenated = torch.cat(attended, dim=-1)
        return self.fusion(concatenated)


class MultimodalAutoencoder(nn.Module):
    """
    Multimodal autoencoder for learning shared representations.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        
        # Individual encoders
        self.encoders = nn.ModuleList([
            SensorEncoder(
                input_dim=dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim // 2,
                dropout=config.dropout
            )
            for dim in config.input_dims
        ])
        
        # Shared latent space
        total_encoded_dim = (config.hidden_dim // 2) * len(config.input_dims)
        self.shared_encoder = nn.Sequential(
            nn.Linear(total_encoded_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Individual decoders for reconstruction
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.output_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, dim)
            )
            for dim in config.input_dims
        ])
        
    def encode(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Encode inputs to shared latent space."""
        encoded = [encoder(inp) for encoder, inp in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded, dim=-1)
        return self.shared_encoder(concatenated)
        
    def decode(self, latent: torch.Tensor) -> List[torch.Tensor]:
        """Decode from latent space to reconstructions."""
        return [decoder(latent) for decoder in self.decoders]
        
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with encoding and reconstruction.
        
        Args:
            inputs: List of sensor feature tensors
            
        Returns:
            Tuple of (latent representation, reconstructions)
        """
        latent = self.encode(inputs)
        reconstructions = self.decode(latent)
        return latent, reconstructions


class DeepFusionModel(nn.Module):
    """
    Unified deep fusion model supporting multiple architectures.
    """
    
    def __init__(self, config: DeepFusionConfig):
        super().__init__()
        
        self.config = config
        
        # Create appropriate fusion network
        if config.architecture == FusionArchitecture.EARLY_FUSION:
            self.fusion_net = EarlyFusionNetwork(config)
        elif config.architecture == FusionArchitecture.LATE_FUSION:
            self.fusion_net = LateFusionNetwork(config)
        elif config.architecture == FusionArchitecture.ATTENTION_FUSION:
            self.fusion_net = AttentionFusionNetwork(config)
        elif config.architecture == FusionArchitecture.TRANSFORMER_FUSION:
            self.fusion_net = TransformerFusionNetwork(config)
        elif config.architecture == FusionArchitecture.CROSS_MODAL_ATTENTION:
            self.fusion_net = CrossModalAttentionNetwork(config)
        elif config.architecture == FusionArchitecture.MULTIMODAL_AUTOENCODER:
            self.fusion_net = MultimodalAutoencoder(config)
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
            
    def forward(self, inputs: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        """Forward pass through fusion network."""
        return self.fusion_net(inputs)


class SensorFusionDataset(Dataset):
    """Dataset for training deep fusion models."""
    
    def __init__(self, sensor_data_list: List[List[np.ndarray]], 
                 labels: Optional[np.ndarray] = None):
        """
        Initialize dataset.
        
        Args:
            sensor_data_list: List of lists, each inner list contains
                              data from one sensor across all samples
            labels: Optional labels for supervised training
        """
        self.sensor_data = sensor_data_list
        self.labels = labels
        self.num_samples = len(sensor_data_list[0])
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple:
        inputs = [torch.FloatTensor(sensor[idx]) for sensor in self.sensor_data]
        
        if self.labels is not None:
            label = torch.FloatTensor(self.labels[idx])
            return inputs, label
        return inputs


class DeepLearningFusionAlgorithm(SensorFusionAlgorithm):
    """
    Deep learning-based sensor fusion algorithm.
    
    Uses neural networks to learn optimal fusion strategies from data.
    """
    
    def __init__(self, architecture: FusionArchitecture = FusionArchitecture.ATTENTION_FUSION,
                 device: Optional[str] = None):
        """
        Initialize the deep learning fusion algorithm.
        
        Args:
            architecture: Type of fusion architecture to use
            device: Device to run on ('cuda' or 'cpu')
        """
        self.architecture = architecture
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[DeepFusionModel] = None
        self.config: Optional[DeepFusionConfig] = None
        
        # Compatibility matrix
        self._compatibility_matrix = {
            (SensorType.HYPERSPECTRAL, SensorType.LIDAR): 0.95,
            (SensorType.HYPERSPECTRAL, SensorType.MAGNETOMETRY): 0.85,
            (SensorType.LIDAR, SensorType.MAGNETOMETRY): 0.80,
        }
        
        for sensor_type in SensorType:
            self._compatibility_matrix[(sensor_type, sensor_type)] = 1.0
            
        for (type1, type2), score in list(self._compatibility_matrix.items()):
            self._compatibility_matrix[(type2, type1)] = score
            
    def build_model(self, input_dims: List[int], output_dim: int,
                    **kwargs) -> DeepFusionModel:
        """
        Build the fusion model.
        
        Args:
            input_dims: Input dimensions for each sensor
            output_dim: Output dimension
            **kwargs: Additional configuration parameters
            
        Returns:
            Built model
        """
        self.config = DeepFusionConfig(
            input_dims=input_dims,
            hidden_dim=kwargs.get('hidden_dim', 512),
            output_dim=output_dim,
            num_heads=kwargs.get('num_heads', 8),
            num_layers=kwargs.get('num_layers', 4),
            dropout=kwargs.get('dropout', 0.1),
            architecture=self.architecture
        )
        
        self.model = DeepFusionModel(self.config).to(self.device)
        return self.model
        
    def train(self, sensor_data_list: List[List[np.ndarray]],
              labels: Optional[np.ndarray] = None,
              num_epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the fusion model.
        
        Args:
            sensor_data_list: Training data for each sensor
            labels: Optional labels for supervised training
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if self.model is None:
            # Infer dimensions and build model
            input_dims = [data[0].shape[-1] if len(data[0].shape) > 0 else 1 
                         for data in sensor_data_list]
            output_dim = labels[0].shape[-1] if labels is not None else input_dims[0]
            self.build_model(input_dims, output_dim)
            
        # Create dataset and dataloader
        dataset = SensorFusionDataset(sensor_data_list, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        # Loss function
        if self.architecture == FusionArchitecture.MULTIMODAL_AUTOENCODER:
            criterion = nn.MSELoss()
        else:
            criterion = nn.MSELoss() if labels is not None else None
            
        # Training loop
        history = {'loss': [], 'lr': []}
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                if labels is not None:
                    inputs, targets = batch
                    inputs = [inp.to(self.device) for inp in inputs]
                    targets = targets.to(self.device)
                else:
                    inputs = batch
                    inputs = [inp.to(self.device) for inp in inputs]
                    targets = None
                    
                optimizer.zero_grad()
                
                # Forward pass
                if self.architecture == FusionArchitecture.MULTIMODAL_AUTOENCODER:
                    latent, reconstructions = self.model(inputs)
                    loss = sum(criterion(recon, inp) for recon, inp in zip(reconstructions, inputs))
                elif self.architecture == FusionArchitecture.ATTENTION_FUSION:
                    output, _ = self.model(inputs)
                    if targets is not None:
                        loss = criterion(output, targets)
                    else:
                        # Self-supervised: predict concatenated inputs
                        target = torch.cat(inputs, dim=-1)[:, :output.size(-1)]
                        loss = F.mse_loss(output, target)
                else:
                    output = self.model(inputs)
                    if targets is not None:
                        loss = criterion(output, targets)
                    else:
                        target = torch.cat(inputs, dim=-1)[:, :output.size(-1)]
                        loss = F.mse_loss(output, target)
                        
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            history['lr'].append(scheduler.get_last_lr()[0])
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
                
        return history
        
    def fuse(self, sensor_data_list: List[SensorData], **kwargs) -> SensorData:
        """
        Fuse multiple sensor data using deep learning.
        
        Args:
            sensor_data_list: List of SensorData objects to fuse
            **kwargs: Additional parameters
            
        Returns:
            Fused SensorData object
        """
        if len(sensor_data_list) < 2:
            raise ValueError("At least two sensor data objects required for fusion")
            
        # Extract data arrays
        data_arrays = []
        for sensor_data in sensor_data_list:
            data = sensor_data.data
            if hasattr(data, 'values'):
                data = data.values
            data_arrays.append(data.flatten())
            
        # Build model if not already built
        if self.model is None:
            input_dims = [len(arr) for arr in data_arrays]
            output_dim = max(input_dims)
            self.build_model(input_dims, output_dim)
            
        # Convert to tensors
        self.model.eval()
        with torch.no_grad():
            inputs = [torch.FloatTensor(arr).unsqueeze(0).to(self.device) 
                     for arr in data_arrays]
            
            if self.architecture == FusionArchitecture.ATTENTION_FUSION:
                output, attention_weights = self.model(inputs)
                attention_weights = attention_weights.cpu().numpy()
            elif self.architecture == FusionArchitecture.MULTIMODAL_AUTOENCODER:
                output, _ = self.model(inputs)
                attention_weights = None
            else:
                output = self.model(inputs)
                attention_weights = None
                
            fused_data = output.squeeze(0).cpu().numpy()
            
        # Reshape to original shape if possible
        original_shape = sensor_data_list[0].data.shape if hasattr(sensor_data_list[0].data, 'shape') else fused_data.shape
        try:
            fused_data = fused_data[:np.prod(original_shape)].reshape(original_shape)
        except ValueError:
            pass
            
        # Create metadata
        metadata = {
            'fusion_method': 'deep_learning',
            'architecture': self.architecture.value,
            'source_sensors': [d.sensor_type.value for d in sensor_data_list]
        }
        
        if attention_weights is not None:
            metadata['attention_weights'] = attention_weights.tolist()
            
        return SensorData(
            data=fused_data,
            sensor_type=SensorType.CUSTOM,
            dimensions=sensor_data_list[0].dimensions,
            metadata=metadata,
            crs=sensor_data_list[0].crs
        )
        
    def get_compatibility_matrix(self) -> Dict[Tuple[SensorType, SensorType], float]:
        """Get the sensor compatibility matrix."""
        return self._compatibility_matrix.copy()
        
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'architecture': self.architecture
        }, path)
        
    def load_model(self, path: str) -> None:
        """Load a model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.architecture = checkpoint['architecture']
        
        self.model = DeepFusionModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def create_fusion_model(
    architecture: FusionArchitecture,
    input_dims: List[int],
    output_dim: int,
    **kwargs
) -> DeepFusionModel:
    """
    Factory function to create a fusion model.
    
    Args:
        architecture: Type of fusion architecture
        input_dims: Input dimensions for each sensor
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured fusion model
    """
    config = DeepFusionConfig(
        input_dims=input_dims,
        hidden_dim=kwargs.get('hidden_dim', 512),
        output_dim=output_dim,
        num_heads=kwargs.get('num_heads', 8),
        num_layers=kwargs.get('num_layers', 4),
        dropout=kwargs.get('dropout', 0.1),
        architecture=architecture
    )
    
    return DeepFusionModel(config)


def fuse_with_attention(
    sensor_data: List[np.ndarray],
    hidden_dim: int = 256,
    num_heads: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick fusion using attention mechanism.
    
    Args:
        sensor_data: List of sensor data arrays
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Tuple of (fused data, attention weights)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_dims = [arr.shape[-1] if len(arr.shape) > 0 else 1 for arr in sensor_data]
    output_dim = max(input_dims)
    
    config = DeepFusionConfig(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        architecture=FusionArchitecture.ATTENTION_FUSION
    )
    
    model = DeepFusionModel(config).to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = [torch.FloatTensor(arr).unsqueeze(0).to(device) for arr in sensor_data]
        output, attention_weights = model(inputs)
        
    return output.squeeze(0).cpu().numpy(), attention_weights.squeeze(0).cpu().numpy()
