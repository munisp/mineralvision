"""
Data Preparation Utilities for SAM3 Fine-Tuning

Provides tools for:
- Converting geology data formats to SAM3 training format
- Interactive labeling with SAM3 assistance
- Data augmentation for geology imagery
- Dataset versioning and management
"""

import logging
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class LabeledSample:
    """A labeled sample for training."""
    image_id: str
    image_path: str
    mask_path: str
    concept: str
    text_prompt: str
    modality: str
    source: str = "manual"
    confidence: float = 1.0
    annotator: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LabeledSample":
        return cls(**d)


@dataclass
class DatasetManifest:
    """Manifest for a training dataset."""
    name: str
    version: str
    modality: str
    concepts: List[str]
    sample_count: int
    created_at: str
    samples: List[LabeledSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["samples"] = [s.to_dict() for s in self.samples]
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetManifest":
        samples = [LabeledSample.from_dict(s) for s in d.pop("samples", [])]
        manifest = cls(**d)
        manifest.samples = samples
        return manifest
    
    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "DatasetManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class GeologyDataConverter:
    """
    Convert geology data formats to SAM3 training format.
    
    Supports:
    - Drillcore tray photos with interval annotations
    - Thin section images with mineral phase masks
    - UAV orthomosaics with polygon annotations
    - Geophysics grids with contour-based masks
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize converter.
        
        Args:
            output_dir: Directory for converted data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
    
    def convert_drillcore_intervals(
        self,
        image_path: str,
        intervals: List[Dict[str, Any]],
        core_length_cm: float,
        concept: str = "mineralized_interval"
    ) -> List[LabeledSample]:
        """
        Convert drillcore interval annotations to masks.
        
        Args:
            image_path: Path to core tray photo
            intervals: List of intervals with from_cm, to_cm, label
            core_length_cm: Total core length in image
            concept: Concept name for labeling
            
        Returns:
            List of labeled samples
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Cannot convert drillcore intervals.")
            return []
        
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            samples = []
            
            for i, interval in enumerate(intervals):
                from_cm = interval.get("from_cm", 0)
                to_cm = interval.get("to_cm", core_length_cm)
                label = interval.get("label", concept)
                
                # Calculate pixel positions (assuming horizontal core)
                from_px = int((from_cm / core_length_cm) * width)
                to_px = int((to_cm / core_length_cm) * width)
                
                # Create mask
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([from_px, 0, to_px, height], fill=255)
                
                # Save files
                image_id = f"{Path(image_path).stem}_interval_{i}"
                image_out = self.output_dir / "images" / f"{image_id}.png"
                mask_out = self.output_dir / "masks" / f"{image_id}_mask.png"
                
                image.save(image_out)
                mask.save(mask_out)
                
                samples.append(LabeledSample(
                    image_id=image_id,
                    image_path=str(image_out),
                    mask_path=str(mask_out),
                    concept=label,
                    text_prompt=label,
                    modality="drillcore",
                    metadata={
                        "from_cm": from_cm,
                        "to_cm": to_cm,
                        "core_length_cm": core_length_cm
                    }
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to convert drillcore intervals: {e}")
            return []
    
    def convert_polygon_annotations(
        self,
        image_path: str,
        polygons: List[Dict[str, Any]],
        concept: str = "geological_feature"
    ) -> List[LabeledSample]:
        """
        Convert polygon annotations to masks.
        
        Args:
            image_path: Path to image
            polygons: List of polygons with points and label
            concept: Default concept name
            
        Returns:
            List of labeled samples
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Cannot convert polygon annotations.")
            return []
        
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            samples = []
            
            for i, polygon in enumerate(polygons):
                points = polygon.get("points", [])
                label = polygon.get("label", concept)
                
                if len(points) < 3:
                    continue
                
                # Create mask
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask)
                
                # Convert points to tuples
                point_tuples = [(p["x"], p["y"]) for p in points]
                draw.polygon(point_tuples, fill=255)
                
                # Save files
                image_id = f"{Path(image_path).stem}_polygon_{i}"
                image_out = self.output_dir / "images" / f"{image_id}.png"
                mask_out = self.output_dir / "masks" / f"{image_id}_mask.png"
                
                image.save(image_out)
                mask.save(mask_out)
                
                samples.append(LabeledSample(
                    image_id=image_id,
                    image_path=str(image_out),
                    mask_path=str(mask_out),
                    concept=label,
                    text_prompt=label,
                    modality="uav_ortho",
                    metadata={"polygon_points": len(points)}
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to convert polygon annotations: {e}")
            return []
    
    def convert_geophysics_contours(
        self,
        grid_path: str,
        threshold: float,
        concept: str = "anomaly",
        above_threshold: bool = True
    ) -> List[LabeledSample]:
        """
        Convert geophysics grid to mask based on threshold.
        
        Args:
            grid_path: Path to geophysics grid (GeoTIFF or numpy)
            threshold: Value threshold for masking
            concept: Concept name
            above_threshold: If True, mask values above threshold
            
        Returns:
            List of labeled samples
        """
        try:
            # Load grid
            if grid_path.endswith(".npy"):
                grid = np.load(grid_path)
            elif PIL_AVAILABLE:
                grid = np.array(Image.open(grid_path))
            else:
                logger.warning("Cannot load grid without PIL or numpy file.")
                return []
            
            # Normalize to 0-255 for visualization
            grid_min, grid_max = grid.min(), grid.max()
            if grid_max > grid_min:
                grid_norm = ((grid - grid_min) / (grid_max - grid_min) * 255).astype(np.uint8)
            else:
                grid_norm = np.zeros_like(grid, dtype=np.uint8)
            
            # Create mask
            if above_threshold:
                mask = (grid > threshold).astype(np.uint8) * 255
            else:
                mask = (grid < threshold).astype(np.uint8) * 255
            
            # Save files
            image_id = f"{Path(grid_path).stem}_contour"
            image_out = self.output_dir / "images" / f"{image_id}.png"
            mask_out = self.output_dir / "masks" / f"{image_id}_mask.png"
            
            if PIL_AVAILABLE:
                Image.fromarray(grid_norm).save(image_out)
                Image.fromarray(mask).save(mask_out)
            
            return [LabeledSample(
                image_id=image_id,
                image_path=str(image_out),
                mask_path=str(mask_out),
                concept=concept,
                text_prompt=concept,
                modality="geophysics",
                metadata={
                    "threshold": threshold,
                    "above_threshold": above_threshold,
                    "grid_min": float(grid_min),
                    "grid_max": float(grid_max)
                }
            )]
        except Exception as e:
            logger.error(f"Failed to convert geophysics contours: {e}")
            return []
    
    def convert_thin_section_phases(
        self,
        image_path: str,
        phase_masks: Dict[str, str]
    ) -> List[LabeledSample]:
        """
        Convert thin section mineral phase masks.
        
        Args:
            image_path: Path to thin section image
            phase_masks: Dict mapping mineral name to mask path
            
        Returns:
            List of labeled samples
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Cannot convert thin section phases.")
            return []
        
        try:
            image = Image.open(image_path)
            samples = []
            
            for mineral, mask_path in phase_masks.items():
                mask = Image.open(mask_path).convert("L")
                
                # Save files
                image_id = f"{Path(image_path).stem}_{mineral}"
                image_out = self.output_dir / "images" / f"{image_id}.png"
                mask_out = self.output_dir / "masks" / f"{image_id}_mask.png"
                
                image.save(image_out)
                mask.save(mask_out)
                
                samples.append(LabeledSample(
                    image_id=image_id,
                    image_path=str(image_out),
                    mask_path=str(mask_out),
                    concept=mineral,
                    text_prompt=mineral,
                    modality="thin_section",
                    metadata={"original_mask": mask_path}
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to convert thin section phases: {e}")
            return []


class GeologyDataAugmenter:
    """
    Data augmentation for geology imagery.
    
    Includes geology-specific augmentations that preserve
    meaningful features while increasing dataset diversity.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize augmenter with optional random seed."""
        self.rng = np.random.default_rng(seed)
    
    def augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        augmentations: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations to image and mask.
        
        Args:
            image: Input image array
            mask: Input mask array
            augmentations: List of augmentation names
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        aug_image = image.copy()
        aug_mask = mask.copy()
        
        for aug in augmentations:
            if aug == "rotation":
                aug_image, aug_mask = self._rotate(aug_image, aug_mask)
            elif aug == "flip":
                aug_image, aug_mask = self._flip(aug_image, aug_mask)
            elif aug == "color_jitter":
                aug_image = self._color_jitter(aug_image)
            elif aug == "scale":
                aug_image, aug_mask = self._scale(aug_image, aug_mask)
            elif aug == "noise":
                aug_image = self._add_noise(aug_image)
            elif aug == "blur":
                aug_image = self._blur(aug_image)
        
        return aug_image, aug_mask
    
    def _rotate(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and mask by 90 degree increments."""
        k = self.rng.integers(0, 4)
        return np.rot90(image, k), np.rot90(mask, k)
    
    def _flip(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flip image and mask horizontally or vertically."""
        if self.rng.random() > 0.5:
            return np.fliplr(image), np.fliplr(mask)
        else:
            return np.flipud(image), np.flipud(mask)
    
    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jitter to image."""
        if len(image.shape) < 3:
            return image
        
        # Brightness
        brightness = self.rng.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Contrast
        contrast = self.rng.uniform(0.8, 1.2)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return image
    
    def _scale(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale image and mask."""
        if not CV2_AVAILABLE:
            return image, mask
        
        scale = self.rng.uniform(0.8, 1.2)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Crop or pad to original size
        if new_h > h:
            start_h = (new_h - h) // 2
            image = image[start_h:start_h + h, :]
            mask = mask[start_h:start_h + h, :]
        elif new_h < h:
            pad_h = (h - new_h) // 2
            image = np.pad(image, ((pad_h, h - new_h - pad_h), (0, 0), (0, 0)), mode="constant")
            mask = np.pad(mask, ((pad_h, h - new_h - pad_h), (0, 0)), mode="constant")
        
        if new_w > w:
            start_w = (new_w - w) // 2
            image = image[:, start_w:start_w + w]
            mask = mask[:, start_w:start_w + w]
        elif new_w < w:
            pad_w = (w - new_w) // 2
            if len(image.shape) == 3:
                image = np.pad(image, ((0, 0), (pad_w, w - new_w - pad_w), (0, 0)), mode="constant")
            else:
                image = np.pad(image, ((0, 0), (pad_w, w - new_w - pad_w)), mode="constant")
            mask = np.pad(mask, ((0, 0), (pad_w, w - new_w - pad_w)), mode="constant")
        
        return image, mask
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = self.rng.normal(0, 10, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image."""
        if not PIL_AVAILABLE:
            return image
        
        pil_image = Image.fromarray(image)
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=1))
        return np.array(blurred)
    
    def generate_augmented_dataset(
        self,
        samples: List[LabeledSample],
        augmentations: List[str],
        multiplier: int = 5,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[LabeledSample]:
        """
        Generate augmented dataset from samples.
        
        Args:
            samples: Original labeled samples
            augmentations: List of augmentation names
            multiplier: Number of augmented versions per sample
            output_dir: Directory for augmented data
            
        Returns:
            List of augmented samples
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Cannot generate augmented dataset.")
            return samples
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "images").mkdir(exist_ok=True)
            (output_dir / "masks").mkdir(exist_ok=True)
        
        augmented_samples = []
        
        for sample in samples:
            try:
                image = np.array(Image.open(sample.image_path))
                mask = np.array(Image.open(sample.mask_path))
                
                for i in range(multiplier):
                    aug_image, aug_mask = self.augment(image, mask, augmentations)
                    
                    if output_dir:
                        aug_id = f"{sample.image_id}_aug_{i}"
                        image_out = output_dir / "images" / f"{aug_id}.png"
                        mask_out = output_dir / "masks" / f"{aug_id}_mask.png"
                        
                        Image.fromarray(aug_image).save(image_out)
                        Image.fromarray(aug_mask).save(mask_out)
                        
                        augmented_samples.append(LabeledSample(
                            image_id=aug_id,
                            image_path=str(image_out),
                            mask_path=str(mask_out),
                            concept=sample.concept,
                            text_prompt=sample.text_prompt,
                            modality=sample.modality,
                            source="augmented",
                            metadata={
                                "original_id": sample.image_id,
                                "augmentations": augmentations
                            }
                        ))
            except Exception as e:
                logger.warning(f"Failed to augment sample {sample.image_id}: {e}")
        
        return augmented_samples


class InteractiveLabelingSession:
    """
    Interactive labeling session using SAM3 for assistance.
    
    Workflow:
    1. User provides point/box prompts
    2. SAM3 generates mask proposals
    3. User corrects masks
    4. Corrected masks saved as training data
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        modality: str = "drillcore"
    ):
        """
        Initialize labeling session.
        
        Args:
            output_dir: Directory for labeled data
            modality: Imaging modality
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modality = modality
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.samples: List[LabeledSample] = []
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "corrections").mkdir(exist_ok=True)
    
    def add_sample(
        self,
        image_path: str,
        mask: np.ndarray,
        concept: str,
        text_prompt: str,
        annotator: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledSample:
        """
        Add a labeled sample to the session.
        
        Args:
            image_path: Path to source image
            mask: Segmentation mask array
            concept: Concept being labeled
            text_prompt: Text prompt for this concept
            annotator: Name of annotator
            confidence: Confidence in label (0-1)
            metadata: Additional metadata
            
        Returns:
            Created LabeledSample
        """
        # Generate unique ID
        image_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()[:8]
        sample_id = f"{self.session_id}_{concept}_{image_hash}"
        
        # Copy image
        image_out = self.output_dir / "images" / f"{sample_id}.png"
        if PIL_AVAILABLE:
            Image.open(image_path).save(image_out)
        else:
            shutil.copy(image_path, image_out)
        
        # Save mask
        mask_out = self.output_dir / "masks" / f"{sample_id}_mask.png"
        if PIL_AVAILABLE:
            Image.fromarray(mask.astype(np.uint8)).save(mask_out)
        
        sample = LabeledSample(
            image_id=sample_id,
            image_path=str(image_out),
            mask_path=str(mask_out),
            concept=concept,
            text_prompt=text_prompt,
            modality=self.modality,
            source="interactive",
            confidence=confidence,
            annotator=annotator,
            metadata=metadata or {}
        )
        
        self.samples.append(sample)
        return sample
    
    def save_session(self) -> str:
        """
        Save labeling session to manifest file.
        
        Returns:
            Path to manifest file
        """
        manifest = DatasetManifest(
            name=f"labeling_session_{self.session_id}",
            version="1.0",
            modality=self.modality,
            concepts=list(set(s.concept for s in self.samples)),
            sample_count=len(self.samples),
            created_at=datetime.now().isoformat(),
            samples=self.samples
        )
        
        manifest_path = self.output_dir / f"manifest_{self.session_id}.json"
        manifest.save(manifest_path)
        
        logger.info(f"Saved {len(self.samples)} samples to {manifest_path}")
        return str(manifest_path)
    
    def load_session(self, manifest_path: Union[str, Path]) -> None:
        """Load existing labeling session."""
        manifest = DatasetManifest.load(manifest_path)
        self.samples = manifest.samples
        self.modality = manifest.modality
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")


class DatasetVersionManager:
    """
    Manage versioned training datasets.
    
    Features:
    - Version tracking with semantic versioning
    - Dataset merging and splitting
    - Reproducible dataset snapshots
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize version manager.
        
        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_version(
        self,
        name: str,
        version: str,
        samples: List[LabeledSample],
        modality: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new dataset version.
        
        Args:
            name: Dataset name
            version: Semantic version (e.g., "1.0.0")
            samples: List of labeled samples
            modality: Imaging modality
            metadata: Additional metadata
            
        Returns:
            Path to versioned dataset
        """
        version_dir = self.base_dir / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (version_dir / "images").mkdir(exist_ok=True)
        (version_dir / "masks").mkdir(exist_ok=True)
        
        # Copy samples to version directory
        versioned_samples = []
        for sample in samples:
            # Copy image
            image_dest = version_dir / "images" / Path(sample.image_path).name
            if Path(sample.image_path).exists():
                shutil.copy(sample.image_path, image_dest)
            
            # Copy mask
            mask_dest = version_dir / "masks" / Path(sample.mask_path).name
            if Path(sample.mask_path).exists():
                shutil.copy(sample.mask_path, mask_dest)
            
            versioned_samples.append(LabeledSample(
                image_id=sample.image_id,
                image_path=str(image_dest),
                mask_path=str(mask_dest),
                concept=sample.concept,
                text_prompt=sample.text_prompt,
                modality=sample.modality,
                source=sample.source,
                confidence=sample.confidence,
                annotator=sample.annotator,
                metadata=sample.metadata
            ))
        
        # Create manifest
        manifest = DatasetManifest(
            name=name,
            version=version,
            modality=modality,
            concepts=list(set(s.concept for s in versioned_samples)),
            sample_count=len(versioned_samples),
            created_at=datetime.now().isoformat(),
            samples=versioned_samples,
            metadata=metadata or {}
        )
        
        manifest.save(version_dir / "manifest.json")
        
        logger.info(f"Created dataset {name} v{version} with {len(versioned_samples)} samples")
        return str(version_dir)
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a dataset."""
        dataset_dir = self.base_dir / name
        if not dataset_dir.exists():
            return []
        return sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    
    def load_version(self, name: str, version: str) -> Optional[DatasetManifest]:
        """Load a specific dataset version."""
        manifest_path = self.base_dir / name / version / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"Dataset {name} v{version} not found")
            return None
        return DatasetManifest.load(manifest_path)
    
    def merge_datasets(
        self,
        datasets: List[Tuple[str, str]],
        output_name: str,
        output_version: str
    ) -> str:
        """
        Merge multiple datasets into one.
        
        Args:
            datasets: List of (name, version) tuples
            output_name: Name for merged dataset
            output_version: Version for merged dataset
            
        Returns:
            Path to merged dataset
        """
        all_samples = []
        modalities = set()
        
        for name, version in datasets:
            manifest = self.load_version(name, version)
            if manifest:
                all_samples.extend(manifest.samples)
                modalities.add(manifest.modality)
        
        # Use most common modality
        modality = max(modalities, key=lambda m: sum(1 for s in all_samples if s.modality == m))
        
        return self.create_version(
            name=output_name,
            version=output_version,
            samples=all_samples,
            modality=modality,
            metadata={"merged_from": datasets}
        )


def prepare_training_data(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    modality: str = "drillcore",
    augment: bool = True,
    augmentation_multiplier: int = 5
) -> DatasetManifest:
    """
    Prepare training data from source directory.
    
    Args:
        source_dir: Directory with images and annotations
        output_dir: Directory for prepared data
        modality: Imaging modality
        augment: Whether to apply augmentation
        augmentation_multiplier: Number of augmented versions
        
    Returns:
        DatasetManifest for prepared data
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    converter = GeologyDataConverter(output_dir)
    samples = []
    
    # Look for image-mask pairs
    image_dir = source_dir / "images"
    mask_dir = source_dir / "masks"
    
    if image_dir.exists() and mask_dir.exists():
        for image_path in image_dir.glob("*"):
            if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif"]:
                continue
            
            mask_path = mask_dir / f"{image_path.stem}_mask{image_path.suffix}"
            if not mask_path.exists():
                mask_path = mask_dir / image_path.name
            
            if mask_path.exists():
                samples.append(LabeledSample(
                    image_id=image_path.stem,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    concept="unknown",
                    text_prompt="geological feature",
                    modality=modality
                ))
    
    # Apply augmentation
    if augment and samples:
        augmenter = GeologyDataAugmenter()
        augmented = augmenter.generate_augmented_dataset(
            samples,
            augmentations=["rotation", "flip", "color_jitter"],
            multiplier=augmentation_multiplier,
            output_dir=output_dir / "augmented"
        )
        samples.extend(augmented)
    
    # Create manifest
    manifest = DatasetManifest(
        name=output_dir.name,
        version="1.0",
        modality=modality,
        concepts=list(set(s.concept for s in samples)),
        sample_count=len(samples),
        created_at=datetime.now().isoformat(),
        samples=samples
    )
    
    manifest.save(output_dir / "manifest.json")
    
    return manifest
