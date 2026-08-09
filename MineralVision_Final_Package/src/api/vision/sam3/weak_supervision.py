"""
Weak Supervision for SAM3 Self-Training

Generates pseudo-labels from existing data sources without
human annotation, enabling automated training data generation.

Sources:
- Geophysics grids: Thresholds, contours, anomaly picking
- Drillcore interval logs: Lithology codes to masks
- Soil lab data: Horizon depths to masks
- Satellite indices: NDVI, mineral indices to masks
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SupervisionSource(str, Enum):
    """Types of weak supervision sources."""
    GEOPHYSICS_THRESHOLD = "geophysics_threshold"
    GEOPHYSICS_CONTOUR = "geophysics_contour"
    INTERVAL_LOG = "interval_log"
    SOIL_HORIZON = "soil_horizon"
    SPECTRAL_INDEX = "spectral_index"
    EXISTING_POLYGON = "existing_polygon"
    HEURISTIC = "heuristic"


@dataclass
class PseudoLabel:
    """A pseudo-label generated from weak supervision."""
    source: SupervisionSource
    concept: str
    mask: np.ndarray
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "concept": self.concept,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "mask_shape": list(self.mask.shape)
        }


@dataclass
class WeakSupervisionConfig:
    """Configuration for weak supervision generation."""
    min_confidence: float = 0.6
    min_mask_area: int = 100
    max_mask_area: Optional[int] = None
    merge_overlapping: bool = True
    overlap_threshold: float = 0.5


class GeophysicsWeakSupervision:
    """
    Generate pseudo-labels from geophysics grids.
    
    Methods:
    - Threshold-based: Values above/below threshold
    - Contour-based: Closed contours at specific values
    - Anomaly-based: Statistical outliers
    - Gradient-based: High gradient regions
    """
    
    def __init__(self, config: Optional[WeakSupervisionConfig] = None):
        self.config = config or WeakSupervisionConfig()
    
    def from_threshold(
        self,
        grid: np.ndarray,
        threshold: float,
        above: bool = True,
        concept: str = "anomaly"
    ) -> List[PseudoLabel]:
        """Generate masks from threshold."""
        if above:
            mask = grid > threshold
        else:
            mask = grid < threshold
        
        # Calculate confidence based on how far values are from threshold
        if mask.any():
            values = grid[mask]
            if above:
                confidence = min(1.0, np.mean(values - threshold) / (np.std(grid) + 1e-6))
            else:
                confidence = min(1.0, np.mean(threshold - values) / (np.std(grid) + 1e-6))
        else:
            confidence = 0.0
        
        if confidence < self.config.min_confidence:
            return []
        
        return [PseudoLabel(
            source=SupervisionSource.GEOPHYSICS_THRESHOLD,
            concept=concept,
            mask=mask.astype(np.uint8),
            confidence=confidence,
            metadata={"threshold": threshold, "above": above}
        )]
    
    def from_percentile(
        self,
        grid: np.ndarray,
        percentile: float,
        above: bool = True,
        concept: str = "anomaly"
    ) -> List[PseudoLabel]:
        """Generate masks from percentile threshold."""
        threshold = np.percentile(grid[~np.isnan(grid)], percentile)
        return self.from_threshold(grid, threshold, above, concept)
    
    def from_contours(
        self,
        grid: np.ndarray,
        levels: List[float],
        concept: str = "contour_region"
    ) -> List[PseudoLabel]:
        """Generate masks from closed contours."""
        labels = []
        
        for i, level in enumerate(levels[:-1]):
            next_level = levels[i + 1]
            mask = (grid >= level) & (grid < next_level)
            
            if mask.sum() >= self.config.min_mask_area:
                labels.append(PseudoLabel(
                    source=SupervisionSource.GEOPHYSICS_CONTOUR,
                    concept=f"{concept}_{i}",
                    mask=mask.astype(np.uint8),
                    confidence=0.8,
                    metadata={"level_min": level, "level_max": next_level}
                ))
        
        return labels
    
    def from_anomaly_detection(
        self,
        grid: np.ndarray,
        n_sigma: float = 2.0,
        concept: str = "statistical_anomaly"
    ) -> List[PseudoLabel]:
        """Generate masks from statistical anomalies."""
        valid = ~np.isnan(grid)
        mean = np.mean(grid[valid])
        std = np.std(grid[valid])
        
        # High anomalies
        high_mask = grid > (mean + n_sigma * std)
        # Low anomalies
        low_mask = grid < (mean - n_sigma * std)
        
        labels = []
        
        if high_mask.sum() >= self.config.min_mask_area:
            labels.append(PseudoLabel(
                source=SupervisionSource.GEOPHYSICS_THRESHOLD,
                concept=f"{concept}_high",
                mask=high_mask.astype(np.uint8),
                confidence=0.85,
                metadata={"type": "high", "n_sigma": n_sigma}
            ))
        
        if low_mask.sum() >= self.config.min_mask_area:
            labels.append(PseudoLabel(
                source=SupervisionSource.GEOPHYSICS_THRESHOLD,
                concept=f"{concept}_low",
                mask=low_mask.astype(np.uint8),
                confidence=0.85,
                metadata={"type": "low", "n_sigma": n_sigma}
            ))
        
        return labels
    
    def from_gradient(
        self,
        grid: np.ndarray,
        percentile: float = 90,
        concept: str = "high_gradient"
    ) -> List[PseudoLabel]:
        """Generate masks from high gradient regions."""
        # Calculate gradient magnitude
        gy, gx = np.gradient(grid)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        threshold = np.percentile(gradient_mag[~np.isnan(gradient_mag)], percentile)
        mask = gradient_mag > threshold
        
        if mask.sum() < self.config.min_mask_area:
            return []
        
        return [PseudoLabel(
            source=SupervisionSource.GEOPHYSICS_THRESHOLD,
            concept=concept,
            mask=mask.astype(np.uint8),
            confidence=0.75,
            metadata={"gradient_threshold": threshold}
        )]


class IntervalLogWeakSupervision:
    """
    Generate pseudo-labels from drillcore interval logs.
    
    Maps lithology codes and intervals to image masks
    based on depth-to-pixel alignment.
    """
    
    def __init__(self, config: Optional[WeakSupervisionConfig] = None):
        self.config = config or WeakSupervisionConfig()
        
        # Default lithology to concept mapping
        self.lithology_mapping = {
            "QV": "quartz_vein",
            "QTZ": "quartz_vein",
            "SUL": "sulfide_zone",
            "PY": "pyrite",
            "ASP": "arsenopyrite",
            "SIL": "silicification",
            "SER": "sericite_alteration",
            "PEG": "pegmatite_zone",
            "SPD": "spodumene",
            "LEP": "lepidolite",
            "CARB": "carbonatite",
            "FEN": "fenitization",
            "LAT": "laterite",
            "SAP": "saprolite",
            "OX": "oxide_zone"
        }
    
    def from_interval_log(
        self,
        intervals: List[Dict[str, Any]],
        image_height: int,
        total_depth_m: float,
        start_depth_m: float = 0.0
    ) -> List[PseudoLabel]:
        """
        Generate masks from interval log.
        
        Args:
            intervals: List of {"from_m": float, "to_m": float, "lithology": str}
            image_height: Height of core tray image in pixels
            total_depth_m: Total depth represented in image
            start_depth_m: Starting depth of image
        """
        labels = []
        pixels_per_meter = image_height / total_depth_m
        
        for interval in intervals:
            from_m = interval.get("from_m", 0)
            to_m = interval.get("to_m", 0)
            lithology = interval.get("lithology", "")
            
            # Skip if outside image range
            if to_m < start_depth_m or from_m > start_depth_m + total_depth_m:
                continue
            
            # Clip to image range
            from_m = max(from_m, start_depth_m)
            to_m = min(to_m, start_depth_m + total_depth_m)
            
            # Convert to pixels
            from_px = int((from_m - start_depth_m) * pixels_per_meter)
            to_px = int((to_m - start_depth_m) * pixels_per_meter)
            
            # Get concept from lithology
            concept = self.lithology_mapping.get(
                lithology.upper(),
                f"lithology_{lithology.lower()}"
            )
            
            # Create mask (full width, interval height)
            # Note: Actual width would come from image
            mask = np.zeros((image_height, 1), dtype=np.uint8)
            mask[from_px:to_px, :] = 1
            
            if mask.sum() >= self.config.min_mask_area:
                labels.append(PseudoLabel(
                    source=SupervisionSource.INTERVAL_LOG,
                    concept=concept,
                    mask=mask,
                    confidence=0.9,  # High confidence from logged data
                    metadata={
                        "from_m": from_m,
                        "to_m": to_m,
                        "lithology": lithology
                    }
                ))
        
        return labels
    
    def from_assay_data(
        self,
        assays: List[Dict[str, Any]],
        image_height: int,
        total_depth_m: float,
        element: str = "Au",
        threshold: float = 0.5,
        start_depth_m: float = 0.0
    ) -> List[PseudoLabel]:
        """
        Generate masks from assay data.
        
        Args:
            assays: List of {"from_m": float, "to_m": float, "Au_ppm": float, ...}
            element: Element to threshold (e.g., "Au", "Li", "REE")
            threshold: Grade threshold
        """
        labels = []
        pixels_per_meter = image_height / total_depth_m
        
        element_key = f"{element}_ppm"
        
        for assay in assays:
            grade = assay.get(element_key, 0)
            
            if grade < threshold:
                continue
            
            from_m = assay.get("from_m", 0)
            to_m = assay.get("to_m", 0)
            
            # Clip to image range
            if to_m < start_depth_m or from_m > start_depth_m + total_depth_m:
                continue
            
            from_m = max(from_m, start_depth_m)
            to_m = min(to_m, start_depth_m + total_depth_m)
            
            from_px = int((from_m - start_depth_m) * pixels_per_meter)
            to_px = int((to_m - start_depth_m) * pixels_per_meter)
            
            # Confidence based on grade above threshold
            confidence = min(1.0, 0.7 + 0.3 * (grade / (threshold * 10)))
            
            mask = np.zeros((image_height, 1), dtype=np.uint8)
            mask[from_px:to_px, :] = 1
            
            concept = f"high_{element.lower()}_zone"
            
            if mask.sum() >= self.config.min_mask_area:
                labels.append(PseudoLabel(
                    source=SupervisionSource.INTERVAL_LOG,
                    concept=concept,
                    mask=mask,
                    confidence=confidence,
                    metadata={
                        "from_m": from_m,
                        "to_m": to_m,
                        "element": element,
                        "grade": grade,
                        "threshold": threshold
                    }
                ))
        
        return labels


class SoilHorizonWeakSupervision:
    """
    Generate pseudo-labels from soil lab data.
    
    Maps horizon depths and soil properties to image masks.
    """
    
    def __init__(self, config: Optional[WeakSupervisionConfig] = None):
        self.config = config or WeakSupervisionConfig()
    
    def from_horizon_depths(
        self,
        horizons: List[Dict[str, Any]],
        image_height: int,
        total_depth_cm: float
    ) -> List[PseudoLabel]:
        """
        Generate masks from soil horizon depths.
        
        Args:
            horizons: List of {"horizon": str, "top_cm": float, "bottom_cm": float}
            image_height: Height of soil pit image
            total_depth_cm: Total depth visible in image
        """
        labels = []
        pixels_per_cm = image_height / total_depth_cm
        
        horizon_concepts = {
            "A": "soil_horizon_a",
            "A1": "soil_horizon_a",
            "A2": "soil_horizon_a",
            "B": "soil_horizon_b",
            "B1": "soil_horizon_b",
            "B2": "soil_horizon_b",
            "Bt": "soil_horizon_b",
            "C": "soil_horizon_c",
            "R": "bedrock",
            "O": "organic_layer"
        }
        
        for horizon in horizons:
            horizon_name = horizon.get("horizon", "")
            top_cm = horizon.get("top_cm", 0)
            bottom_cm = horizon.get("bottom_cm", 0)
            
            # Clip to image range
            top_cm = max(0, top_cm)
            bottom_cm = min(total_depth_cm, bottom_cm)
            
            if bottom_cm <= top_cm:
                continue
            
            top_px = int(top_cm * pixels_per_cm)
            bottom_px = int(bottom_cm * pixels_per_cm)
            
            concept = horizon_concepts.get(
                horizon_name.upper(),
                f"soil_horizon_{horizon_name.lower()}"
            )
            
            mask = np.zeros((image_height, 1), dtype=np.uint8)
            mask[top_px:bottom_px, :] = 1
            
            if mask.sum() >= self.config.min_mask_area:
                labels.append(PseudoLabel(
                    source=SupervisionSource.SOIL_HORIZON,
                    concept=concept,
                    mask=mask,
                    confidence=0.85,
                    metadata={
                        "horizon": horizon_name,
                        "top_cm": top_cm,
                        "bottom_cm": bottom_cm
                    }
                ))
        
        return labels
    
    def from_soil_properties(
        self,
        properties: Dict[str, Any],
        image_shape: Tuple[int, int]
    ) -> List[PseudoLabel]:
        """
        Generate masks from soil property indicators.
        
        Uses color/texture heuristics based on soil properties.
        """
        labels = []
        height, width = image_shape
        
        # Example: High clay content indicator
        clay_percent = properties.get("clay_percent", 0)
        if clay_percent > 40:
            # High clay soils often have distinct color
            labels.append(PseudoLabel(
                source=SupervisionSource.HEURISTIC,
                concept="clay_rich_zone",
                mask=np.ones((height, width), dtype=np.uint8),
                confidence=0.6,
                metadata={"clay_percent": clay_percent}
            ))
        
        # Waterlogging indicator
        if properties.get("drainage_class") in ["poor", "very_poor"]:
            labels.append(PseudoLabel(
                source=SupervisionSource.HEURISTIC,
                concept="waterlogging",
                mask=np.ones((height, width), dtype=np.uint8),
                confidence=0.65,
                metadata={"drainage_class": properties.get("drainage_class")}
            ))
        
        return labels


class SpectralIndexWeakSupervision:
    """
    Generate pseudo-labels from spectral indices.
    
    For satellite/UAV imagery with multiple bands.
    """
    
    def __init__(self, config: Optional[WeakSupervisionConfig] = None):
        self.config = config or WeakSupervisionConfig()
    
    def from_ndvi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        vegetation_threshold: float = 0.3,
        bare_soil_threshold: float = 0.1
    ) -> List[PseudoLabel]:
        """Generate masks from NDVI."""
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        labels = []
        
        # Vegetation mask
        veg_mask = ndvi > vegetation_threshold
        if veg_mask.sum() >= self.config.min_mask_area:
            labels.append(PseudoLabel(
                source=SupervisionSource.SPECTRAL_INDEX,
                concept="vegetation",
                mask=veg_mask.astype(np.uint8),
                confidence=0.8,
                metadata={"index": "ndvi", "threshold": vegetation_threshold}
            ))
        
        # Bare soil mask
        soil_mask = (ndvi > -0.1) & (ndvi < bare_soil_threshold)
        if soil_mask.sum() >= self.config.min_mask_area:
            labels.append(PseudoLabel(
                source=SupervisionSource.SPECTRAL_INDEX,
                concept="bare_soil",
                mask=soil_mask.astype(np.uint8),
                confidence=0.75,
                metadata={"index": "ndvi", "threshold": bare_soil_threshold}
            ))
        
        return labels
    
    def from_iron_oxide_index(
        self,
        red: np.ndarray,
        blue: np.ndarray,
        threshold: float = 1.5
    ) -> List[PseudoLabel]:
        """Generate masks from iron oxide index (red/blue ratio)."""
        iron_index = red / (blue + 1e-6)
        
        mask = iron_index > threshold
        
        if mask.sum() < self.config.min_mask_area:
            return []
        
        return [PseudoLabel(
            source=SupervisionSource.SPECTRAL_INDEX,
            concept="iron_oxide_staining",
            mask=mask.astype(np.uint8),
            confidence=0.7,
            metadata={"index": "iron_oxide", "threshold": threshold}
        )]
    
    def from_clay_index(
        self,
        swir1: np.ndarray,
        swir2: np.ndarray,
        threshold: float = 1.2
    ) -> List[PseudoLabel]:
        """Generate masks from clay mineral index."""
        clay_index = swir1 / (swir2 + 1e-6)
        
        mask = clay_index > threshold
        
        if mask.sum() < self.config.min_mask_area:
            return []
        
        return [PseudoLabel(
            source=SupervisionSource.SPECTRAL_INDEX,
            concept="clay_alteration",
            mask=mask.astype(np.uint8),
            confidence=0.7,
            metadata={"index": "clay", "threshold": threshold}
        )]


class WeakSupervisionPipeline:
    """
    Complete weak supervision pipeline for generating training data.
    
    Combines multiple weak supervision sources and handles
    label aggregation and quality filtering.
    """
    
    def __init__(self, config: Optional[WeakSupervisionConfig] = None):
        self.config = config or WeakSupervisionConfig()
        
        self.geophysics = GeophysicsWeakSupervision(config)
        self.interval_log = IntervalLogWeakSupervision(config)
        self.soil_horizon = SoilHorizonWeakSupervision(config)
        self.spectral = SpectralIndexWeakSupervision(config)
        
        self._generated_labels: List[PseudoLabel] = []
    
    def generate_from_geophysics(
        self,
        grid: np.ndarray,
        methods: List[str] = ["threshold", "anomaly"],
        **kwargs
    ) -> List[PseudoLabel]:
        """Generate labels from geophysics grid."""
        labels = []
        
        if "threshold" in methods:
            threshold = kwargs.get("threshold", np.percentile(grid[~np.isnan(grid)], 90))
            labels.extend(self.geophysics.from_threshold(
                grid, threshold, concept=kwargs.get("concept", "geophysics_anomaly")
            ))
        
        if "anomaly" in methods:
            labels.extend(self.geophysics.from_anomaly_detection(
                grid, n_sigma=kwargs.get("n_sigma", 2.0)
            ))
        
        if "gradient" in methods:
            labels.extend(self.geophysics.from_gradient(grid))
        
        self._generated_labels.extend(labels)
        return labels
    
    def generate_from_intervals(
        self,
        intervals: List[Dict[str, Any]],
        image_height: int,
        total_depth_m: float,
        **kwargs
    ) -> List[PseudoLabel]:
        """Generate labels from interval log."""
        labels = self.interval_log.from_interval_log(
            intervals, image_height, total_depth_m,
            start_depth_m=kwargs.get("start_depth_m", 0.0)
        )
        
        self._generated_labels.extend(labels)
        return labels
    
    def generate_from_assays(
        self,
        assays: List[Dict[str, Any]],
        image_height: int,
        total_depth_m: float,
        element: str = "Au",
        threshold: float = 0.5,
        **kwargs
    ) -> List[PseudoLabel]:
        """Generate labels from assay data."""
        labels = self.interval_log.from_assay_data(
            assays, image_height, total_depth_m,
            element=element, threshold=threshold
        )
        
        self._generated_labels.extend(labels)
        return labels
    
    def generate_from_soil_horizons(
        self,
        horizons: List[Dict[str, Any]],
        image_height: int,
        total_depth_cm: float
    ) -> List[PseudoLabel]:
        """Generate labels from soil horizon data."""
        labels = self.soil_horizon.from_horizon_depths(
            horizons, image_height, total_depth_cm
        )
        
        self._generated_labels.extend(labels)
        return labels
    
    def generate_from_spectral(
        self,
        bands: Dict[str, np.ndarray],
        indices: List[str] = ["ndvi", "iron_oxide"]
    ) -> List[PseudoLabel]:
        """Generate labels from spectral indices."""
        labels = []
        
        if "ndvi" in indices and "nir" in bands and "red" in bands:
            labels.extend(self.spectral.from_ndvi(bands["nir"], bands["red"]))
        
        if "iron_oxide" in indices and "red" in bands and "blue" in bands:
            labels.extend(self.spectral.from_iron_oxide_index(bands["red"], bands["blue"]))
        
        if "clay" in indices and "swir1" in bands and "swir2" in bands:
            labels.extend(self.spectral.from_clay_index(bands["swir1"], bands["swir2"]))
        
        self._generated_labels.extend(labels)
        return labels
    
    def get_all_labels(self) -> List[PseudoLabel]:
        """Get all generated labels."""
        return self._generated_labels
    
    def filter_by_confidence(self, min_confidence: float = 0.7) -> List[PseudoLabel]:
        """Filter labels by minimum confidence."""
        return [l for l in self._generated_labels if l.confidence >= min_confidence]
    
    def export_for_training(self, output_dir: str) -> Dict[str, Any]:
        """Export labels in SAM3 training format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "created_at": datetime.now().isoformat(),
            "total_labels": len(self._generated_labels),
            "samples": []
        }
        
        for i, label in enumerate(self._generated_labels):
            # Save mask
            mask_file = output_path / f"mask_{i:06d}.npy"
            np.save(mask_file, label.mask)
            
            # Add to manifest
            manifest["samples"].append({
                "mask_file": str(mask_file),
                "concept": label.concept,
                "source": label.source.value,
                "confidence": label.confidence,
                "metadata": label.metadata
            })
        
        # Save manifest
        manifest_file = output_path / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
    
    def clear(self) -> None:
        """Clear all generated labels."""
        self._generated_labels = []


def create_weak_supervision_pipeline(
    min_confidence: float = 0.6
) -> WeakSupervisionPipeline:
    """Factory function to create weak supervision pipeline."""
    config = WeakSupervisionConfig(min_confidence=min_confidence)
    return WeakSupervisionPipeline(config)
