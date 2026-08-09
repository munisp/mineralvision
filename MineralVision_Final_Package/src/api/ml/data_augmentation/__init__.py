"""
Data Augmentation for MineralVision

Geospatial-specific data augmentation techniques.
"""

from .geospatial_augmentation import (
    GeospatialAugmentationPipeline,
    AugmentationConfig,
    AugmentationType,
    SpectralAugmentation,
    SpatialAugmentation,
    NoiseAugmentation,
    AtmosphericAugmentation,
    GeometricAugmentation,
    MixupAugmentation,
    AugmentedDataset,
    create_augmentation_pipeline,
    augment_geospatial_data,
)

__all__ = [
    "GeospatialAugmentationPipeline",
    "AugmentationConfig",
    "AugmentationType",
    "SpectralAugmentation",
    "SpatialAugmentation",
    "NoiseAugmentation",
    "AtmosphericAugmentation",
    "GeometricAugmentation",
    "MixupAugmentation",
    "AugmentedDataset",
    "create_augmentation_pipeline",
    "augment_geospatial_data",
]
