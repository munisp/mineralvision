"""
Data Augmentation Pipeline for Geospatial Data

This module provides comprehensive data augmentation techniques specifically
designed for geospatial and remote sensing data used in mineral exploration.

Supports:
- Spectral augmentation (for hyperspectral data)
- Spatial augmentation (rotations, flips, crops)
- Noise injection (Gaussian, salt-and-pepper, speckle)
- Atmospheric correction simulation
- Sensor-specific augmentations
- Mixup and CutMix for geospatial data
- Elastic deformations
- Multi-scale augmentation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import ndimage
from scipy.interpolate import griddata
import random

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of augmentation."""
    SPECTRAL = "spectral"
    SPATIAL = "spatial"
    NOISE = "noise"
    ATMOSPHERIC = "atmospheric"
    GEOMETRIC = "geometric"
    MIXUP = "mixup"
    CUTMIX = "cutmix"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Probability of applying each augmentation
    apply_probability: float = 0.5
    
    # Spectral augmentations
    spectral_shift_range: Tuple[float, float] = (-0.1, 0.1)
    spectral_scale_range: Tuple[float, float] = (0.9, 1.1)
    spectral_noise_std: float = 0.02
    band_dropout_prob: float = 0.1
    
    # Spatial augmentations
    rotation_range: Tuple[float, float] = (-180, 180)
    flip_horizontal: bool = True
    flip_vertical: bool = True
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)
    crop_ratio_range: Tuple[float, float] = (0.9, 1.1)
    
    # Noise augmentations
    gaussian_noise_std: float = 0.05
    salt_pepper_prob: float = 0.01
    speckle_noise_std: float = 0.1
    
    # Atmospheric augmentations
    haze_intensity_range: Tuple[float, float] = (0.0, 0.3)
    cloud_coverage_range: Tuple[float, float] = (0.0, 0.2)
    shadow_intensity_range: Tuple[float, float] = (0.7, 1.0)
    
    # Geometric augmentations
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    perspective_scale: float = 0.1
    
    # Mixup/CutMix
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Multi-scale
    scale_factors: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])


class SpectralAugmentation:
    """Augmentation techniques for spectral/hyperspectral data."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def spectral_shift(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random spectral shift to simulate sensor calibration differences.
        
        Args:
            data: Input spectral data (samples x bands) or (H x W x bands)
            
        Returns:
            Augmented data
        """
        shift = np.random.uniform(*self.config.spectral_shift_range)
        return data + shift
    
    def spectral_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random spectral scaling to simulate illumination variations.
        
        Args:
            data: Input spectral data
            
        Returns:
            Augmented data
        """
        scale = np.random.uniform(*self.config.spectral_scale_range)
        return data * scale
    
    def spectral_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to spectral bands.
        
        Args:
            data: Input spectral data
            
        Returns:
            Augmented data
        """
        noise = np.random.normal(0, self.config.spectral_noise_std, data.shape)
        return data + noise
    
    def band_dropout(self, data: np.ndarray) -> np.ndarray:
        """
        Randomly drop spectral bands to simulate missing data.
        
        Args:
            data: Input spectral data (samples x bands) or (H x W x bands)
            
        Returns:
            Augmented data with some bands zeroed
        """
        result = data.copy()
        
        if len(data.shape) == 2:
            # (samples x bands)
            num_bands = data.shape[1]
            mask = np.random.random(num_bands) > self.config.band_dropout_prob
            result[:, ~mask] = 0
        elif len(data.shape) == 3:
            # (H x W x bands)
            num_bands = data.shape[2]
            mask = np.random.random(num_bands) > self.config.band_dropout_prob
            result[:, :, ~mask] = 0
        
        return result
    
    def spectral_interpolation(self, data: np.ndarray, 
                               target_bands: int) -> np.ndarray:
        """
        Interpolate spectral data to different number of bands.
        
        Args:
            data: Input spectral data
            target_bands: Target number of bands
            
        Returns:
            Interpolated data
        """
        if len(data.shape) == 2:
            # (samples x bands)
            current_bands = data.shape[1]
            x_old = np.linspace(0, 1, current_bands)
            x_new = np.linspace(0, 1, target_bands)
            
            result = np.zeros((data.shape[0], target_bands))
            for i in range(data.shape[0]):
                result[i] = np.interp(x_new, x_old, data[i])
            return result
        
        elif len(data.shape) == 3:
            # (H x W x bands)
            current_bands = data.shape[2]
            x_old = np.linspace(0, 1, current_bands)
            x_new = np.linspace(0, 1, target_bands)
            
            result = np.zeros((data.shape[0], data.shape[1], target_bands))
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    result[i, j] = np.interp(x_new, x_old, data[i, j])
            return result
        
        return data
    
    def continuum_removal(self, data: np.ndarray) -> np.ndarray:
        """
        Apply continuum removal to enhance absorption features.
        
        Args:
            data: Input spectral data
            
        Returns:
            Continuum-removed data
        """
        if len(data.shape) == 1:
            # Single spectrum
            hull = self._compute_convex_hull(data)
            return data / (hull + 1e-10)
        
        elif len(data.shape) == 2:
            # (samples x bands)
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                hull = self._compute_convex_hull(data[i])
                result[i] = data[i] / (hull + 1e-10)
            return result
        
        return data
    
    def _compute_convex_hull(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute convex hull for continuum removal."""
        n = len(spectrum)
        x = np.arange(n)
        
        # Simple linear interpolation between endpoints
        hull = np.linspace(spectrum[0], spectrum[-1], n)
        
        # Ensure hull is above spectrum
        for i in range(n):
            if hull[i] < spectrum[i]:
                hull[i] = spectrum[i]
        
        return hull
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random spectral augmentations."""
        if np.random.random() < self.config.apply_probability:
            aug_type = np.random.choice(['shift', 'scale', 'noise', 'dropout'])
            
            if aug_type == 'shift':
                data = self.spectral_shift(data)
            elif aug_type == 'scale':
                data = self.spectral_scale(data)
            elif aug_type == 'noise':
                data = self.spectral_noise(data)
            elif aug_type == 'dropout':
                data = self.band_dropout(data)
        
        return data


class SpatialAugmentation:
    """Augmentation techniques for spatial/image data."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def random_rotation(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random rotation.
        
        Args:
            data: Input image data (H x W) or (H x W x C)
            
        Returns:
            Rotated data
        """
        angle = np.random.uniform(*self.config.rotation_range)
        
        if len(data.shape) == 2:
            return ndimage.rotate(data, angle, reshape=False, mode='reflect')
        elif len(data.shape) == 3:
            result = np.zeros_like(data)
            for c in range(data.shape[2]):
                result[:, :, c] = ndimage.rotate(
                    data[:, :, c], angle, reshape=False, mode='reflect'
                )
            return result
        
        return data
    
    def random_flip(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random horizontal and/or vertical flip.
        
        Args:
            data: Input image data
            
        Returns:
            Flipped data
        """
        result = data.copy()
        
        if self.config.flip_horizontal and np.random.random() < 0.5:
            result = np.flip(result, axis=1)
        
        if self.config.flip_vertical and np.random.random() < 0.5:
            result = np.flip(result, axis=0)
        
        return np.ascontiguousarray(result)
    
    def random_crop(self, data: np.ndarray, 
                   target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Apply random crop with scale and ratio variations.
        
        Args:
            data: Input image data (H x W) or (H x W x C)
            target_size: Target crop size (default: same as input)
            
        Returns:
            Cropped data
        """
        h, w = data.shape[:2]
        
        if target_size is None:
            target_h, target_w = h, w
        else:
            target_h, target_w = target_size
        
        # Random scale
        scale = np.random.uniform(*self.config.crop_scale_range)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        # Random position
        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        
        # Crop
        if len(data.shape) == 2:
            cropped = data[top:top+crop_h, left:left+crop_w]
        else:
            cropped = data[top:top+crop_h, left:left+crop_w, :]
        
        # Resize to target size
        from scipy.ndimage import zoom
        
        if len(data.shape) == 2:
            zoom_factors = (target_h / crop_h, target_w / crop_w)
            return zoom(cropped, zoom_factors, order=1)
        else:
            zoom_factors = (target_h / crop_h, target_w / crop_w, 1)
            return zoom(cropped, zoom_factors, order=1)
    
    def center_crop(self, data: np.ndarray, 
                   crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply center crop.
        
        Args:
            data: Input image data
            crop_size: Size of crop (height, width)
            
        Returns:
            Cropped data
        """
        h, w = data.shape[:2]
        crop_h, crop_w = crop_size
        
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        
        if len(data.shape) == 2:
            return data[top:top+crop_h, left:left+crop_w]
        else:
            return data[top:top+crop_h, left:left+crop_w, :]
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random spatial augmentations."""
        if np.random.random() < self.config.apply_probability:
            data = self.random_rotation(data)
        
        data = self.random_flip(data)
        
        return data


class NoiseAugmentation:
    """Noise injection augmentations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.config.gaussian_noise_std, data.shape)
        return data + noise
    
    def salt_pepper_noise(self, data: np.ndarray) -> np.ndarray:
        """Add salt and pepper noise."""
        result = data.copy()
        
        # Salt
        salt_mask = np.random.random(data.shape) < self.config.salt_pepper_prob / 2
        result[salt_mask] = np.max(data)
        
        # Pepper
        pepper_mask = np.random.random(data.shape) < self.config.salt_pepper_prob / 2
        result[pepper_mask] = np.min(data)
        
        return result
    
    def speckle_noise(self, data: np.ndarray) -> np.ndarray:
        """Add multiplicative speckle noise (common in radar data)."""
        noise = np.random.normal(1, self.config.speckle_noise_std, data.shape)
        return data * noise
    
    def poisson_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Poisson noise (shot noise)."""
        # Scale data to reasonable range for Poisson
        scale = 1000
        scaled = np.clip(data * scale, 0, None)
        noisy = np.random.poisson(scaled) / scale
        return noisy
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random noise augmentation."""
        if np.random.random() < self.config.apply_probability:
            noise_type = np.random.choice(['gaussian', 'salt_pepper', 'speckle'])
            
            if noise_type == 'gaussian':
                data = self.gaussian_noise(data)
            elif noise_type == 'salt_pepper':
                data = self.salt_pepper_noise(data)
            elif noise_type == 'speckle':
                data = self.speckle_noise(data)
        
        return data


class AtmosphericAugmentation:
    """Simulate atmospheric effects for remote sensing data."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def add_haze(self, data: np.ndarray) -> np.ndarray:
        """
        Simulate atmospheric haze.
        
        Args:
            data: Input image data (H x W x C)
            
        Returns:
            Hazy image
        """
        intensity = np.random.uniform(*self.config.haze_intensity_range)
        
        # Create haze layer
        h, w = data.shape[:2]
        haze = np.ones((h, w)) * intensity
        
        # Add some variation
        haze += np.random.normal(0, intensity * 0.1, (h, w))
        haze = np.clip(haze, 0, 1)
        
        if len(data.shape) == 3:
            haze = haze[:, :, np.newaxis]
        
        # Blend with original
        return data * (1 - haze) + haze
    
    def add_clouds(self, data: np.ndarray) -> np.ndarray:
        """
        Simulate cloud coverage.
        
        Args:
            data: Input image data
            
        Returns:
            Image with simulated clouds
        """
        coverage = np.random.uniform(*self.config.cloud_coverage_range)
        
        h, w = data.shape[:2]
        
        # Generate cloud mask using Perlin-like noise
        cloud_mask = self._generate_cloud_mask(h, w, coverage)
        
        if len(data.shape) == 3:
            cloud_mask = cloud_mask[:, :, np.newaxis]
        
        # Clouds are bright
        cloud_value = np.max(data) * 0.9
        
        return data * (1 - cloud_mask) + cloud_value * cloud_mask
    
    def _generate_cloud_mask(self, h: int, w: int, coverage: float) -> np.ndarray:
        """Generate a cloud mask using multi-scale noise."""
        mask = np.zeros((h, w))
        
        # Multi-scale noise
        for scale in [4, 8, 16, 32]:
            noise = np.random.random((h // scale + 1, w // scale + 1))
            noise = ndimage.zoom(noise, scale, order=1)[:h, :w]
            mask += noise
        
        mask /= 4
        
        # Threshold to get coverage
        threshold = np.percentile(mask, (1 - coverage) * 100)
        mask = (mask > threshold).astype(float)
        
        # Smooth edges
        mask = ndimage.gaussian_filter(mask, sigma=3)
        
        return mask
    
    def add_shadows(self, data: np.ndarray) -> np.ndarray:
        """
        Simulate terrain shadows.
        
        Args:
            data: Input image data
            
        Returns:
            Image with simulated shadows
        """
        h, w = data.shape[:2]
        
        # Generate shadow mask
        shadow_mask = np.random.random((h, w))
        shadow_mask = ndimage.gaussian_filter(shadow_mask, sigma=10)
        
        # Threshold
        threshold = np.percentile(shadow_mask, 80)
        shadow_mask = (shadow_mask > threshold).astype(float)
        
        # Smooth
        shadow_mask = ndimage.gaussian_filter(shadow_mask, sigma=5)
        
        # Apply shadow intensity
        intensity = np.random.uniform(*self.config.shadow_intensity_range)
        
        if len(data.shape) == 3:
            shadow_mask = shadow_mask[:, :, np.newaxis]
        
        return data * (1 - shadow_mask * (1 - intensity))
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random atmospheric augmentation."""
        if np.random.random() < self.config.apply_probability:
            aug_type = np.random.choice(['haze', 'clouds', 'shadows'])
            
            if aug_type == 'haze':
                data = self.add_haze(data)
            elif aug_type == 'clouds':
                data = self.add_clouds(data)
            elif aug_type == 'shadows':
                data = self.add_shadows(data)
        
        return data


class GeometricAugmentation:
    """Geometric deformation augmentations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def elastic_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply elastic deformation.
        
        Args:
            data: Input image data (H x W) or (H x W x C)
            
        Returns:
            Elastically deformed data
        """
        h, w = data.shape[:2]
        
        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.random((h, w)) * 2 - 1),
            self.config.elastic_sigma
        ) * self.config.elastic_alpha
        
        dy = ndimage.gaussian_filter(
            (np.random.random((h, w)) * 2 - 1),
            self.config.elastic_sigma
        ) * self.config.elastic_alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        indices_x = np.clip(x + dx, 0, w - 1).astype(int)
        indices_y = np.clip(y + dy, 0, h - 1).astype(int)
        
        if len(data.shape) == 2:
            return data[indices_y, indices_x]
        else:
            result = np.zeros_like(data)
            for c in range(data.shape[2]):
                result[:, :, c] = data[:, :, c][indices_y, indices_x]
            return result
    
    def perspective_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random perspective transformation.
        
        Args:
            data: Input image data
            
        Returns:
            Perspective-transformed data
        """
        h, w = data.shape[:2]
        
        # Generate random perspective points
        scale = self.config.perspective_scale
        
        # Source points (corners)
        src = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Destination points (with random perturbation)
        dst = src + np.random.uniform(-scale * min(h, w), scale * min(h, w), src.shape)
        
        # Simple approximation using affine transform
        # For full perspective, would need cv2.getPerspectiveTransform
        
        # Calculate affine matrix (simplified)
        from scipy.ndimage import affine_transform
        
        # Use simple scaling/shearing as approximation
        matrix = np.eye(2)
        matrix[0, 0] = 1 + np.random.uniform(-scale, scale)
        matrix[1, 1] = 1 + np.random.uniform(-scale, scale)
        matrix[0, 1] = np.random.uniform(-scale, scale)
        matrix[1, 0] = np.random.uniform(-scale, scale)
        
        if len(data.shape) == 2:
            return affine_transform(data, matrix, mode='reflect')
        else:
            result = np.zeros_like(data)
            for c in range(data.shape[2]):
                result[:, :, c] = affine_transform(data[:, :, c], matrix, mode='reflect')
            return result
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random geometric augmentation."""
        if np.random.random() < self.config.apply_probability:
            aug_type = np.random.choice(['elastic', 'perspective'])
            
            if aug_type == 'elastic':
                data = self.elastic_transform(data)
            elif aug_type == 'perspective':
                data = self.perspective_transform(data)
        
        return data


class MixupAugmentation:
    """Mixup and CutMix augmentations for geospatial data."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def mixup(self, data1: np.ndarray, data2: np.ndarray,
             label1: np.ndarray, label2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Mixup augmentation.
        
        Args:
            data1: First sample
            data2: Second sample
            label1: First label
            label2: Second label
            
        Returns:
            Tuple of (mixed data, mixed label)
        """
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        mixed_data = lam * data1 + (1 - lam) * data2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_data, mixed_label
    
    def cutmix(self, data1: np.ndarray, data2: np.ndarray,
              label1: np.ndarray, label2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix augmentation.
        
        Args:
            data1: First sample (H x W x C)
            data2: Second sample
            label1: First label
            label2: Second label
            
        Returns:
            Tuple of (mixed data, mixed label)
        """
        h, w = data1.shape[:2]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        
        # Calculate cut size
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cut
        mixed_data = data1.copy()
        if len(data1.shape) == 2:
            mixed_data[y1:y2, x1:x2] = data2[y1:y2, x1:x2]
        else:
            mixed_data[y1:y2, x1:x2, :] = data2[y1:y2, x1:x2, :]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (h * w)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_data, mixed_label
    
    def geospatial_cutmix(self, data1: np.ndarray, data2: np.ndarray,
                         label1: np.ndarray, label2: np.ndarray,
                         mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix with geospatially-aware masking.
        
        Uses geological boundaries or other masks to guide the cut.
        
        Args:
            data1: First sample
            data2: Second sample
            label1: First label
            label2: Second label
            mask: Optional geological boundary mask
            
        Returns:
            Tuple of (mixed data, mixed label)
        """
        if mask is None:
            return self.cutmix(data1, data2, label1, label2)
        
        h, w = data1.shape[:2]
        
        # Use mask to guide mixing
        mixed_data = data1.copy()
        
        # Threshold mask
        binary_mask = mask > np.median(mask)
        
        if len(data1.shape) == 2:
            mixed_data[binary_mask] = data2[binary_mask]
        else:
            for c in range(data1.shape[2]):
                mixed_data[:, :, c][binary_mask] = data2[:, :, c][binary_mask]
        
        # Calculate effective lambda
        lam = 1 - np.mean(binary_mask)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_data, mixed_label


class GeospatialAugmentationPipeline:
    """
    Complete augmentation pipeline for geospatial data.
    
    Combines multiple augmentation techniques with configurable probabilities.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        """
        Initialize the augmentation pipeline.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        
        # Initialize augmentation modules
        self.spectral_aug = SpectralAugmentation(self.config)
        self.spatial_aug = SpatialAugmentation(self.config)
        self.noise_aug = NoiseAugmentation(self.config)
        self.atmospheric_aug = AtmosphericAugmentation(self.config)
        self.geometric_aug = GeometricAugmentation(self.config)
        self.mixup_aug = MixupAugmentation(self.config)
    
    def __call__(self, data: np.ndarray, 
                augmentation_types: List[AugmentationType] = None) -> np.ndarray:
        """
        Apply augmentation pipeline.
        
        Args:
            data: Input data
            augmentation_types: List of augmentation types to apply
                              (None applies all)
            
        Returns:
            Augmented data
        """
        if augmentation_types is None:
            augmentation_types = [
                AugmentationType.SPECTRAL,
                AugmentationType.SPATIAL,
                AugmentationType.NOISE,
                AugmentationType.ATMOSPHERIC,
                AugmentationType.GEOMETRIC
            ]
        
        result = data.copy()
        
        for aug_type in augmentation_types:
            if aug_type == AugmentationType.SPECTRAL:
                result = self.spectral_aug(result)
            elif aug_type == AugmentationType.SPATIAL:
                result = self.spatial_aug(result)
            elif aug_type == AugmentationType.NOISE:
                result = self.noise_aug(result)
            elif aug_type == AugmentationType.ATMOSPHERIC:
                result = self.atmospheric_aug(result)
            elif aug_type == AugmentationType.GEOMETRIC:
                result = self.geometric_aug(result)
        
        return result
    
    def augment_batch(self, batch: np.ndarray, 
                     labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment a batch of data.
        
        Args:
            batch: Batch of data (N x H x W x C) or (N x features)
            labels: Optional labels
            
        Returns:
            Tuple of (augmented batch, labels)
        """
        augmented = np.zeros_like(batch)
        
        for i in range(len(batch)):
            augmented[i] = self(batch[i])
        
        return augmented, labels


class AugmentedDataset(Dataset):
    """
    PyTorch Dataset wrapper with augmentation.
    """
    
    def __init__(self, dataset: Dataset, 
                augmentation_pipeline: GeospatialAugmentationPipeline = None,
                augment_probability: float = 0.5):
        """
        Initialize augmented dataset.
        
        Args:
            dataset: Base dataset
            augmentation_pipeline: Augmentation pipeline
            augment_probability: Probability of applying augmentation
        """
        self.dataset = dataset
        self.augmentation_pipeline = augmentation_pipeline or GeospatialAugmentationPipeline()
        self.augment_probability = augment_probability
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # Apply augmentation
        if np.random.random() < self.augment_probability:
            data = self.augmentation_pipeline(data)
        
        # Convert back to tensor
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        
        return data, label


# Convenience functions
def create_augmentation_pipeline(config: Dict[str, Any] = None) -> GeospatialAugmentationPipeline:
    """Create an augmentation pipeline with optional config."""
    if config:
        aug_config = AugmentationConfig(**config)
    else:
        aug_config = AugmentationConfig()
    
    return GeospatialAugmentationPipeline(aug_config)


def augment_geospatial_data(data: np.ndarray, 
                           augmentation_types: List[str] = None) -> np.ndarray:
    """
    Apply augmentation to geospatial data.
    
    Args:
        data: Input data
        augmentation_types: List of augmentation type names
        
    Returns:
        Augmented data
    """
    pipeline = GeospatialAugmentationPipeline()
    
    if augmentation_types:
        types = [AugmentationType(t) for t in augmentation_types]
    else:
        types = None
    
    return pipeline(data, types)
