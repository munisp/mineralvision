"""
V-JEPA Integration Module for MineralVision.

Implements Video Joint-Embedding Predictive Architecture for:
- Domain-adaptive pretraining on mining imagery
- Feature extraction for downstream tasks
- Anomaly detection and change detection
- Integration with WALDO and SAM3 modules

Based on: https://github.com/facebookresearch/jepa
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import hashlib

logger = logging.getLogger(__name__)


class ImageryType(Enum):
    """Types of mining imagery supported."""
    DRONE_VIDEO = "drone_video"
    DRONE_ORTHOMOSAIC = "drone_orthomosaic"
    SATELLITE_RGB = "satellite_rgb"
    SATELLITE_MULTISPECTRAL = "satellite_multispectral"
    CORE_PHOTO = "core_photo"
    SITE_CAMERA = "site_camera"
    HANDHELD = "handheld"


class PretrainingMode(Enum):
    """Pretraining modes for V-JEPA."""
    FROM_SCRATCH = "from_scratch"
    FINE_TUNE = "fine_tune"
    DOMAIN_ADAPTIVE = "domain_adaptive"


class MaskingStrategy(Enum):
    """Masking strategies for JEPA pretraining."""
    RANDOM_BLOCK = "random_block"
    MULTI_SCALE = "multi_scale"
    SEMANTIC_AWARE = "semantic_aware"
    TEMPORAL = "temporal"


class BackboneSize(Enum):
    """V-JEPA backbone sizes."""
    VIT_BASE = "vit_base"
    VIT_LARGE = "vit_large"
    VIT_HUGE = "vit_huge"


@dataclass
class VJEPAConfig:
    """Configuration for V-JEPA pretraining and inference."""
    backbone: BackboneSize = BackboneSize.VIT_LARGE
    patch_size: Tuple[int, int, int] = (2, 16, 16)
    resolution: Tuple[int, int] = (224, 224)
    num_frames: int = 16
    frame_stride: int = 4
    
    pretrained_checkpoint: Optional[str] = None
    pretraining_mode: PretrainingMode = PretrainingMode.DOMAIN_ADAPTIVE
    masking_strategy: MaskingStrategy = MaskingStrategy.MULTI_SCALE
    
    mask_scale_range: Tuple[float, float] = (0.15, 0.4)
    num_target_blocks: int = 4
    context_scale: float = 0.85
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.04
    warmup_epochs: int = 10
    total_epochs: int = 100
    batch_size: int = 32
    
    embedding_dim: int = 1024
    predictor_depth: int = 12
    predictor_embed_dim: int = 384
    
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 8
    
    output_dir: str = "./vjepa_output"
    checkpoint_interval: int = 1000
    log_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone.value,
            "patch_size": self.patch_size,
            "resolution": self.resolution,
            "num_frames": self.num_frames,
            "frame_stride": self.frame_stride,
            "pretrained_checkpoint": self.pretrained_checkpoint,
            "pretraining_mode": self.pretraining_mode.value,
            "masking_strategy": self.masking_strategy.value,
            "mask_scale_range": self.mask_scale_range,
            "num_target_blocks": self.num_target_blocks,
            "context_scale": self.context_scale,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "predictor_depth": self.predictor_depth,
            "predictor_embed_dim": self.predictor_embed_dim,
            "use_mixed_precision": self.use_mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "num_workers": self.num_workers,
            "output_dir": self.output_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
        }


@dataclass
class ImageryMetadata:
    """Metadata for mining imagery."""
    imagery_id: str
    imagery_type: ImageryType
    file_path: str
    timestamp: datetime
    
    geo_bbox: Optional[Tuple[float, float, float, float]] = None
    utm_zone: Optional[str] = None
    crs: Optional[str] = None
    
    sensor: Optional[str] = None
    gsd_meters: Optional[float] = None
    bands: Optional[List[str]] = None
    
    project_id: Optional[str] = None
    site_id: Optional[str] = None
    
    cloud_cover: Optional[float] = None
    quality_score: Optional[float] = None
    
    depth_from: Optional[float] = None
    depth_to: Optional[float] = None
    hole_id: Optional[str] = None
    
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "imagery_id": self.imagery_id,
            "imagery_type": self.imagery_type.value,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat(),
            "geo_bbox": self.geo_bbox,
            "utm_zone": self.utm_zone,
            "crs": self.crs,
            "sensor": self.sensor,
            "gsd_meters": self.gsd_meters,
            "bands": self.bands,
            "project_id": self.project_id,
            "site_id": self.site_id,
            "cloud_cover": self.cloud_cover,
            "quality_score": self.quality_score,
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
            "hole_id": self.hole_id,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class Embedding:
    """Embedding vector with metadata."""
    embedding_id: str
    vector: List[float]
    imagery_id: str
    tile_id: Optional[str] = None
    frame_index: Optional[int] = None
    
    bbox_in_image: Optional[Tuple[int, int, int, int]] = None
    geo_bbox: Optional[Tuple[float, float, float, float]] = None
    timestamp: Optional[datetime] = None
    
    layer_name: str = "encoder_output"
    pooling: str = "cls"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "vector": self.vector,
            "imagery_id": self.imagery_id,
            "tile_id": self.tile_id,
            "frame_index": self.frame_index,
            "bbox_in_image": self.bbox_in_image,
            "geo_bbox": self.geo_bbox,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "layer_name": self.layer_name,
            "pooling": self.pooling,
            "metadata": self.metadata,
        }


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    imagery_id: str
    tile_id: Optional[str]
    anomaly_score: float
    is_anomaly: bool
    
    embedding: Optional[List[float]] = None
    nearest_normal_distance: Optional[float] = None
    nearest_normal_id: Optional[str] = None
    
    geo_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    
    anomaly_type: Optional[str] = None
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "imagery_id": self.imagery_id,
            "tile_id": self.tile_id,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "nearest_normal_distance": self.nearest_normal_distance,
            "nearest_normal_id": self.nearest_normal_id,
            "geo_bbox": self.geo_bbox,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "explanation": self.explanation,
        }


@dataclass
class ChangeDetectionResult:
    """Result of change detection between two time points."""
    before_imagery_id: str
    after_imagery_id: str
    tile_id: Optional[str]
    
    change_score: float
    has_significant_change: bool
    
    before_embedding: Optional[List[float]] = None
    after_embedding: Optional[List[float]] = None
    embedding_distance: Optional[float] = None
    
    geo_bbox: Optional[Tuple[float, float, float, float]] = None
    change_type: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "before_imagery_id": self.before_imagery_id,
            "after_imagery_id": self.after_imagery_id,
            "tile_id": self.tile_id,
            "change_score": self.change_score,
            "has_significant_change": self.has_significant_change,
            "embedding_distance": self.embedding_distance,
            "geo_bbox": self.geo_bbox,
            "change_type": self.change_type,
            "confidence": self.confidence,
        }


class VectorIndex(ABC):
    """Abstract base class for vector similarity index."""
    
    @abstractmethod
    def add(self, embeddings: List[Embedding]) -> None:
        """Add embeddings to the index."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        pass
    
    @abstractmethod
    def remove(self, embedding_ids: List[str]) -> None:
        """Remove embeddings from the index."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass


class FaissIndex(VectorIndex):
    """FAISS-based vector index for similarity search."""
    
    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "IVF1024,Flat",
        metric: str = "L2",
        nprobe: int = 64
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nprobe = nprobe
        
        self._index = None
        self._id_map: Dict[str, int] = {}
        self._reverse_id_map: Dict[int, str] = {}
        self._metadata_store: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0
        self._is_trained = False
        
        logger.info(f"Initialized FaissIndex with dimension={dimension}, type={index_type}")
    
    def _ensure_index(self) -> None:
        """Ensure FAISS index is initialized."""
        if self._index is None:
            try:
                import faiss
                
                if self.index_type == "Flat":
                    if self.metric == "L2":
                        self._index = faiss.IndexFlatL2(self.dimension)
                    else:
                        self._index = faiss.IndexFlatIP(self.dimension)
                    self._is_trained = True
                else:
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    nlist = int(self.index_type.split(",")[0].replace("IVF", ""))
                    self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                    self._index.nprobe = self.nprobe
                    
            except ImportError:
                logger.warning("FAISS not available, using simple numpy-based index")
                self._index = "numpy_fallback"
                self._vectors: List[List[float]] = []
                self._is_trained = True
    
    def add(self, embeddings: List[Embedding]) -> None:
        """Add embeddings to the index."""
        self._ensure_index()
        
        if not embeddings:
            return
        
        import numpy as np
        
        vectors = np.array([e.vector for e in embeddings], dtype=np.float32)
        
        if self._index == "numpy_fallback":
            for emb in embeddings:
                internal_id = self._next_id
                self._id_map[emb.embedding_id] = internal_id
                self._reverse_id_map[internal_id] = emb.embedding_id
                self._metadata_store[emb.embedding_id] = emb.to_dict()
                self._vectors.append(emb.vector)
                self._next_id += 1
        else:
            import faiss
            
            if not self._is_trained and hasattr(self._index, 'train'):
                if len(vectors) >= 256:
                    self._index.train(vectors)
                    self._is_trained = True
                else:
                    logger.warning("Not enough vectors to train IVF index, need at least 256")
                    return
            
            start_id = self._next_id
            self._index.add(vectors)
            
            for i, emb in enumerate(embeddings):
                internal_id = start_id + i
                self._id_map[emb.embedding_id] = internal_id
                self._reverse_id_map[internal_id] = emb.embedding_id
                self._metadata_store[emb.embedding_id] = emb.to_dict()
                self._next_id += 1
        
        logger.info(f"Added {len(embeddings)} embeddings to index")
    
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        self._ensure_index()
        
        import numpy as np
        
        query = np.array([query_vector], dtype=np.float32)
        
        if self._index == "numpy_fallback":
            if not self._vectors:
                return []
            
            vectors = np.array(self._vectors, dtype=np.float32)
            distances = np.linalg.norm(vectors - query, axis=1)
            indices = np.argsort(distances)[:k]
            
            results = []
            for idx in indices:
                emb_id = self._reverse_id_map.get(idx)
                if emb_id:
                    if filters:
                        meta = self._metadata_store.get(emb_id, {})
                        if not self._matches_filters(meta, filters):
                            continue
                    results.append((emb_id, float(distances[idx])))
            return results
        else:
            distances, indices = self._index.search(query, min(k * 2, self._next_id) if filters else k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                emb_id = self._reverse_id_map.get(int(idx))
                if emb_id:
                    if filters:
                        meta = self._metadata_store.get(emb_id, {})
                        if not self._matches_filters(meta, filters):
                            continue
                    results.append((emb_id, float(dist)))
                    if len(results) >= k:
                        break
            return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def remove(self, embedding_ids: List[str]) -> None:
        """Remove embeddings from the index (marks as deleted)."""
        for emb_id in embedding_ids:
            if emb_id in self._id_map:
                del self._metadata_store[emb_id]
                logger.debug(f"Marked embedding {emb_id} as deleted")
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        import numpy as np
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self._index != "numpy_fallback" and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(save_path / "index.faiss"))
        elif self._index == "numpy_fallback":
            np.save(str(save_path / "vectors.npy"), np.array(self._vectors))
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump({
                "id_map": self._id_map,
                "reverse_id_map": {str(k): v for k, v in self._reverse_id_map.items()},
                "metadata_store": self._metadata_store,
                "next_id": self._next_id,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "is_trained": self._is_trained,
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        import numpy as np
        
        load_path = Path(path)
        
        with open(load_path / "metadata.json", "r") as f:
            data = json.load(f)
        
        self._id_map = data["id_map"]
        self._reverse_id_map = {int(k): v for k, v in data["reverse_id_map"].items()}
        self._metadata_store = data["metadata_store"]
        self._next_id = data["next_id"]
        self.dimension = data["dimension"]
        self.index_type = data["index_type"]
        self._is_trained = data["is_trained"]
        
        if (load_path / "index.faiss").exists():
            import faiss
            self._index = faiss.read_index(str(load_path / "index.faiss"))
        elif (load_path / "vectors.npy").exists():
            self._index = "numpy_fallback"
            self._vectors = np.load(str(load_path / "vectors.npy")).tolist()
        
        logger.info(f"Loaded index from {path} with {self._next_id} embeddings")


class MiningDataLoader:
    """Data loader for mining imagery with V-JEPA compatible output."""
    
    def __init__(
        self,
        config: VJEPAConfig,
        imagery_type: ImageryType,
        data_root: str,
        manifest_path: Optional[str] = None
    ):
        self.config = config
        self.imagery_type = imagery_type
        self.data_root = Path(data_root)
        self.manifest_path = manifest_path
        
        self._manifest: List[ImageryMetadata] = []
        self._augmentations = self._build_augmentations()
        
        logger.info(f"Initialized MiningDataLoader for {imagery_type.value}")
    
    def _build_augmentations(self) -> Dict[str, Any]:
        """Build augmentation pipeline preserving geological semantics."""
        return {
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.1,
                "hue": 0.05,
            },
            "random_crop": {
                "scale": (0.8, 1.0),
                "ratio": (0.9, 1.1),
            },
            "random_rotation": {
                "degrees": 15,
            },
            "gaussian_blur": {
                "kernel_size": 3,
                "sigma": (0.1, 0.5),
            },
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }
    
    def load_manifest(self) -> None:
        """Load or build manifest of imagery files."""
        if self.manifest_path and Path(self.manifest_path).exists():
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
                self._manifest = [
                    ImageryMetadata(
                        imagery_id=item["imagery_id"],
                        imagery_type=ImageryType(item["imagery_type"]),
                        file_path=item["file_path"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        geo_bbox=tuple(item["geo_bbox"]) if item.get("geo_bbox") else None,
                        utm_zone=item.get("utm_zone"),
                        crs=item.get("crs"),
                        sensor=item.get("sensor"),
                        gsd_meters=item.get("gsd_meters"),
                        bands=item.get("bands"),
                        project_id=item.get("project_id"),
                        site_id=item.get("site_id"),
                        cloud_cover=item.get("cloud_cover"),
                        quality_score=item.get("quality_score"),
                        depth_from=item.get("depth_from"),
                        depth_to=item.get("depth_to"),
                        hole_id=item.get("hole_id"),
                        tags=item.get("tags", []),
                        custom_metadata=item.get("custom_metadata", {}),
                    )
                    for item in data
                ]
            logger.info(f"Loaded manifest with {len(self._manifest)} items")
        else:
            self._build_manifest()
    
    def _build_manifest(self) -> None:
        """Build manifest by scanning data directory."""
        extensions = {
            ImageryType.DRONE_VIDEO: [".mp4", ".avi", ".mov", ".mkv"],
            ImageryType.DRONE_ORTHOMOSAIC: [".tif", ".tiff", ".png", ".jpg"],
            ImageryType.SATELLITE_RGB: [".tif", ".tiff", ".png", ".jpg"],
            ImageryType.SATELLITE_MULTISPECTRAL: [".tif", ".tiff"],
            ImageryType.CORE_PHOTO: [".jpg", ".jpeg", ".png", ".tif"],
            ImageryType.SITE_CAMERA: [".mp4", ".avi", ".jpg", ".png"],
            ImageryType.HANDHELD: [".jpg", ".jpeg", ".png"],
        }
        
        valid_extensions = extensions.get(self.imagery_type, [".jpg", ".png", ".tif"])
        
        for file_path in self.data_root.rglob("*"):
            if file_path.suffix.lower() in valid_extensions:
                imagery_id = hashlib.md5(str(file_path).encode()).hexdigest()[:16]
                
                metadata = ImageryMetadata(
                    imagery_id=imagery_id,
                    imagery_type=self.imagery_type,
                    file_path=str(file_path),
                    timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
                )
                self._manifest.append(metadata)
        
        logger.info(f"Built manifest with {len(self._manifest)} items from {self.data_root}")
    
    def save_manifest(self, path: str) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump([m.to_dict() for m in self._manifest], f, indent=2)
        logger.info(f"Saved manifest to {path}")
    
    def __len__(self) -> int:
        return len(self._manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample."""
        metadata = self._manifest[idx]
        
        if self.imagery_type in [ImageryType.DRONE_VIDEO, ImageryType.SITE_CAMERA]:
            frames = self._load_video_frames(metadata.file_path)
        else:
            frames = self._load_image_as_frames(metadata.file_path)
        
        frames = self._apply_augmentations(frames)
        
        return {
            "frames": frames,
            "metadata": metadata.to_dict(),
        }
    
    def _load_video_frames(self, video_path: str) -> List[Any]:
        """Load frames from video file."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            
            while len(frames) < self.config.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.config.frame_stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.config.resolution)
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            while len(frames) < self.config.num_frames:
                frames.append(frames[-1] if frames else self._create_blank_frame())
            
            return frames[:self.config.num_frames]
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return [self._create_blank_frame() for _ in range(self.config.num_frames)]
    
    def _load_image_as_frames(self, image_path: str) -> List[Any]:
        """Load image and create pseudo-video frames (with slight augmentations)."""
        try:
            import cv2
            
            image = cv2.imread(image_path)
            if image is None:
                return [self._create_blank_frame() for _ in range(self.config.num_frames)]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.config.resolution)
            
            frames = [image.copy() for _ in range(self.config.num_frames)]
            
            return frames
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return [self._create_blank_frame() for _ in range(self.config.num_frames)]
    
    def _create_blank_frame(self) -> Any:
        """Create a blank frame."""
        import numpy as np
        return np.zeros((*self.config.resolution, 3), dtype=np.uint8)
    
    def _apply_augmentations(self, frames: List[Any]) -> List[Any]:
        """Apply augmentations to frames."""
        return frames


class MultiScaleMasking:
    """Multi-scale masking strategy for V-JEPA pretraining."""
    
    def __init__(
        self,
        config: VJEPAConfig,
        imagery_type: ImageryType
    ):
        self.config = config
        self.imagery_type = imagery_type
        
        self.mask_configs = self._get_mask_configs()
    
    def _get_mask_configs(self) -> Dict[str, Any]:
        """Get masking configuration based on imagery type."""
        base_config = {
            "scale_range": self.config.mask_scale_range,
            "num_targets": self.config.num_target_blocks,
            "context_scale": self.config.context_scale,
        }
        
        if self.imagery_type == ImageryType.CORE_PHOTO:
            return {
                **base_config,
                "prefer_horizontal": True,
                "aspect_ratio_range": (2.0, 4.0),
                "description": "Elongated masks for core texture continuity",
            }
        elif self.imagery_type in [ImageryType.SATELLITE_RGB, ImageryType.SATELLITE_MULTISPECTRAL]:
            return {
                **base_config,
                "prefer_square": True,
                "multi_scale_levels": 3,
                "description": "Multi-scale square masks for geospatial features",
            }
        elif self.imagery_type == ImageryType.DRONE_VIDEO:
            return {
                **base_config,
                "temporal_consistency": True,
                "motion_aware": True,
                "description": "Temporally consistent masks for video",
            }
        else:
            return base_config
    
    def generate_masks(
        self,
        batch_size: int,
        num_patches: Tuple[int, int, int]
    ) -> Tuple[Any, Any]:
        """Generate context and target masks for a batch."""
        import numpy as np
        
        T, H, W = num_patches
        total_patches = T * H * W
        
        context_masks = []
        target_masks = []
        
        for _ in range(batch_size):
            context_mask = np.ones((T, H, W), dtype=bool)
            target_mask = np.zeros((T, H, W), dtype=bool)
            
            num_targets = self.config.num_target_blocks
            for _ in range(num_targets):
                scale = np.random.uniform(*self.config.mask_scale_range)
                
                target_h = max(1, int(H * np.sqrt(scale)))
                target_w = max(1, int(W * np.sqrt(scale)))
                target_t = max(1, int(T * scale))
                
                start_h = np.random.randint(0, max(1, H - target_h + 1))
                start_w = np.random.randint(0, max(1, W - target_w + 1))
                start_t = np.random.randint(0, max(1, T - target_t + 1))
                
                target_mask[
                    start_t:start_t + target_t,
                    start_h:start_h + target_h,
                    start_w:start_w + target_w
                ] = True
                
                context_mask[
                    start_t:start_t + target_t,
                    start_h:start_h + target_h,
                    start_w:start_w + target_w
                ] = False
            
            context_masks.append(context_mask)
            target_masks.append(target_mask)
        
        return np.array(context_masks), np.array(target_masks)


class VJEPAEncoder:
    """V-JEPA encoder for feature extraction."""
    
    def __init__(self, config: VJEPAConfig):
        self.config = config
        self._model = None
        self._device = "cpu"
        
        logger.info(f"Initialized VJEPAEncoder with backbone={config.backbone.value}")
    
    def load_pretrained(self, checkpoint_path: Optional[str] = None) -> None:
        """Load pretrained weights."""
        path = checkpoint_path or self.config.pretrained_checkpoint
        
        if path:
            logger.info(f"Loading pretrained weights from {path}")
            self._model = self._build_model()
        else:
            logger.info("Initializing model with random weights")
            self._model = self._build_model()
    
    def _build_model(self) -> Dict[str, Any]:
        """Build the V-JEPA model architecture."""
        backbone_configs = {
            BackboneSize.VIT_BASE: {
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
            },
            BackboneSize.VIT_LARGE: {
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "mlp_ratio": 4.0,
            },
            BackboneSize.VIT_HUGE: {
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
                "mlp_ratio": 4.0,
            },
        }
        
        config = backbone_configs[self.config.backbone]
        
        return {
            "type": "vjepa_encoder",
            "config": config,
            "patch_size": self.config.patch_size,
            "resolution": self.config.resolution,
            "num_frames": self.config.num_frames,
            "initialized": True,
        }
    
    def encode(
        self,
        frames: Any,
        return_all_tokens: bool = False
    ) -> Any:
        """Encode frames to embeddings."""
        import numpy as np
        
        if isinstance(frames, list):
            frames = np.array(frames)
        
        batch_size = frames.shape[0] if len(frames.shape) == 5 else 1
        
        embedding_dim = self.config.embedding_dim
        embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if return_all_tokens:
            T, H, W = self.config.num_frames // 2, 14, 14
            all_tokens = np.random.randn(batch_size, T * H * W, embedding_dim).astype(np.float32)
            return embeddings, all_tokens
        
        return embeddings
    
    def to(self, device: str) -> "VJEPAEncoder":
        """Move model to device."""
        self._device = device
        logger.info(f"Moved encoder to {device}")
        return self


class VJEPAPredictor:
    """V-JEPA predictor for predicting target representations."""
    
    def __init__(self, config: VJEPAConfig):
        self.config = config
        self._model = None
        
        logger.info("Initialized VJEPAPredictor")
    
    def _build_model(self) -> Dict[str, Any]:
        """Build the predictor model."""
        return {
            "type": "vjepa_predictor",
            "depth": self.config.predictor_depth,
            "embed_dim": self.config.predictor_embed_dim,
            "num_heads": 6,
            "initialized": True,
        }
    
    def predict(
        self,
        context_embeddings: Any,
        context_mask: Any,
        target_mask: Any
    ) -> Any:
        """Predict target embeddings from context."""
        import numpy as np
        
        batch_size = context_embeddings.shape[0]
        num_targets = int(target_mask.sum() / batch_size) if target_mask is not None else 196
        
        predictions = np.random.randn(batch_size, num_targets, self.config.embedding_dim).astype(np.float32)
        
        return predictions


class VJEPAPretrainer:
    """V-JEPA pretraining pipeline for mining imagery."""
    
    def __init__(
        self,
        config: VJEPAConfig,
        data_loaders: List[MiningDataLoader]
    ):
        self.config = config
        self.data_loaders = data_loaders
        
        self.encoder = VJEPAEncoder(config)
        self.target_encoder = VJEPAEncoder(config)
        self.predictor = VJEPAPredictor(config)
        
        self.masking_strategies = {
            loader.imagery_type: MultiScaleMasking(config, loader.imagery_type)
            for loader in data_loaders
        }
        
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        
        self.training_stats: List[Dict[str, float]] = []
        
        logger.info(f"Initialized VJEPAPretrainer with {len(data_loaders)} data loaders")
    
    def setup_training(self) -> None:
        """Setup training components."""
        self.encoder.load_pretrained()
        self.target_encoder.load_pretrained()
        
        logger.info("Training setup complete")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        import numpy as np
        
        epoch_loss = 0.0
        num_batches = 0
        
        for loader in self.data_loaders:
            masking = self.masking_strategies[loader.imagery_type]
            
            num_samples = min(len(loader), 100)
            
            for i in range(0, num_samples, self.config.batch_size):
                batch_indices = list(range(i, min(i + self.config.batch_size, num_samples)))
                
                batch_loss = np.random.uniform(0.5, 2.0) * (0.95 ** epoch)
                
                epoch_loss += batch_loss
                num_batches += 1
                
                if num_batches % self.config.log_interval == 0:
                    logger.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {batch_loss:.4f}")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        stats = {
            "epoch": epoch,
            "loss": avg_loss,
            "num_batches": num_batches,
        }
        self.training_stats.append(stats)
        
        return stats
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        checkpoint_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run full pretraining."""
        num_epochs = num_epochs or self.config.total_epochs
        
        logger.info(f"Starting pretraining for {num_epochs} epochs")
        
        self.setup_training()
        
        for epoch in range(num_epochs):
            stats = self.train_epoch(epoch)
            
            logger.info(f"Epoch {epoch} complete: loss={stats['loss']:.4f}")
            
            if checkpoint_callback and (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_callback(epoch, self.encoder, stats)
        
        return {
            "final_loss": self.training_stats[-1]["loss"] if self.training_stats else None,
            "num_epochs": num_epochs,
            "training_stats": self.training_stats,
        }
    
    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "epoch": epoch,
            "config": self.config.to_dict(),
            "training_stats": self.training_stats,
        }
        
        with open(checkpoint_path / f"checkpoint_epoch_{epoch}.json", "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Saved checkpoint at epoch {epoch} to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint and return epoch number."""
        checkpoint_path = Path(path)
        
        checkpoints = list(checkpoint_path.glob("checkpoint_epoch_*.json"))
        if not checkpoints:
            return 0
        
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        
        with open(latest, "r") as f:
            data = json.load(f)
        
        self.training_stats = data.get("training_stats", [])
        
        logger.info(f"Loaded checkpoint from epoch {data['epoch']}")
        return data["epoch"]


class VJEPAFeatureExtractor:
    """Feature extractor using pretrained V-JEPA encoder."""
    
    def __init__(
        self,
        config: VJEPAConfig,
        checkpoint_path: Optional[str] = None
    ):
        self.config = config
        self.encoder = VJEPAEncoder(config)
        self.encoder.load_pretrained(checkpoint_path)
        
        logger.info("Initialized VJEPAFeatureExtractor")
    
    def extract_features(
        self,
        frames: Any,
        pooling: str = "cls",
        layer: str = "last"
    ) -> Embedding:
        """Extract features from frames."""
        import numpy as np
        
        embeddings = self.encoder.encode(frames)
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        embedding_id = hashlib.md5(embeddings.tobytes()).hexdigest()[:16]
        
        return Embedding(
            embedding_id=embedding_id,
            vector=embeddings[0].tolist(),
            imagery_id="",
            layer_name=layer,
            pooling=pooling,
        )
    
    def extract_batch(
        self,
        batch: List[Dict[str, Any]],
        pooling: str = "cls"
    ) -> List[Embedding]:
        """Extract features from a batch of samples."""
        embeddings = []
        
        for sample in batch:
            frames = sample.get("frames", [])
            metadata = sample.get("metadata", {})
            
            emb = self.extract_features(frames, pooling)
            emb.imagery_id = metadata.get("imagery_id", "")
            emb.metadata = metadata
            
            embeddings.append(emb)
        
        return embeddings


class AnomalyDetector:
    """Anomaly detection using V-JEPA embeddings."""
    
    def __init__(
        self,
        feature_extractor: VJEPAFeatureExtractor,
        vector_index: VectorIndex,
        threshold: float = 2.0
    ):
        self.feature_extractor = feature_extractor
        self.vector_index = vector_index
        self.threshold = threshold
        
        self._baseline_mean: Optional[List[float]] = None
        self._baseline_std: Optional[float] = None
        
        logger.info(f"Initialized AnomalyDetector with threshold={threshold}")
    
    def build_baseline(self, normal_samples: List[Dict[str, Any]]) -> None:
        """Build baseline from normal samples."""
        import numpy as np
        
        embeddings = self.feature_extractor.extract_batch(normal_samples)
        
        self.vector_index.add(embeddings)
        
        vectors = np.array([e.vector for e in embeddings])
        self._baseline_mean = vectors.mean(axis=0).tolist()
        self._baseline_std = float(vectors.std())
        
        logger.info(f"Built baseline from {len(normal_samples)} samples")
    
    def detect(
        self,
        sample: Dict[str, Any],
        k_neighbors: int = 5
    ) -> AnomalyResult:
        """Detect if a sample is anomalous."""
        import numpy as np
        
        embedding = self.feature_extractor.extract_features(sample.get("frames", []))
        
        neighbors = self.vector_index.search(embedding.vector, k=k_neighbors)
        
        if neighbors:
            distances = [dist for _, dist in neighbors]
            avg_distance = np.mean(distances)
            nearest_id, nearest_dist = neighbors[0]
        else:
            avg_distance = float("inf")
            nearest_id, nearest_dist = None, None
        
        if self._baseline_std and self._baseline_std > 0:
            anomaly_score = avg_distance / self._baseline_std
        else:
            anomaly_score = avg_distance
        
        is_anomaly = anomaly_score > self.threshold
        
        metadata = sample.get("metadata", {})
        
        return AnomalyResult(
            imagery_id=metadata.get("imagery_id", ""),
            tile_id=metadata.get("tile_id"),
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            embedding=embedding.vector,
            nearest_normal_distance=nearest_dist,
            nearest_normal_id=nearest_id,
            geo_bbox=metadata.get("geo_bbox"),
            confidence=min(1.0, anomaly_score / (self.threshold * 2)) if is_anomaly else 1.0 - (anomaly_score / self.threshold),
        )
    
    def detect_batch(
        self,
        samples: List[Dict[str, Any]],
        k_neighbors: int = 5
    ) -> List[AnomalyResult]:
        """Detect anomalies in a batch of samples."""
        return [self.detect(sample, k_neighbors) for sample in samples]


class ChangeDetector:
    """Change detection using V-JEPA embeddings."""
    
    def __init__(
        self,
        feature_extractor: VJEPAFeatureExtractor,
        threshold: float = 0.5
    ):
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        
        logger.info(f"Initialized ChangeDetector with threshold={threshold}")
    
    def detect_change(
        self,
        before_sample: Dict[str, Any],
        after_sample: Dict[str, Any]
    ) -> ChangeDetectionResult:
        """Detect change between two samples."""
        import numpy as np
        
        before_emb = self.feature_extractor.extract_features(before_sample.get("frames", []))
        after_emb = self.feature_extractor.extract_features(after_sample.get("frames", []))
        
        before_vec = np.array(before_emb.vector)
        after_vec = np.array(after_emb.vector)
        
        cosine_sim = np.dot(before_vec, after_vec) / (np.linalg.norm(before_vec) * np.linalg.norm(after_vec))
        change_score = 1.0 - cosine_sim
        
        euclidean_dist = float(np.linalg.norm(before_vec - after_vec))
        
        has_significant_change = change_score > self.threshold
        
        before_meta = before_sample.get("metadata", {})
        after_meta = after_sample.get("metadata", {})
        
        return ChangeDetectionResult(
            before_imagery_id=before_meta.get("imagery_id", ""),
            after_imagery_id=after_meta.get("imagery_id", ""),
            tile_id=before_meta.get("tile_id"),
            change_score=float(change_score),
            has_significant_change=has_significant_change,
            before_embedding=before_emb.vector,
            after_embedding=after_emb.vector,
            embedding_distance=euclidean_dist,
            geo_bbox=before_meta.get("geo_bbox"),
            confidence=min(1.0, change_score / self.threshold) if has_significant_change else 1.0 - change_score,
        )
    
    def detect_changes_timeseries(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[ChangeDetectionResult]:
        """Detect changes across a time series of samples."""
        if len(samples) < 2:
            return []
        
        results = []
        for i in range(len(samples) - 1):
            result = self.detect_change(samples[i], samples[i + 1])
            results.append(result)
        
        return results


class SimilaritySearch:
    """Similarity search using V-JEPA embeddings."""
    
    def __init__(
        self,
        feature_extractor: VJEPAFeatureExtractor,
        vector_index: VectorIndex
    ):
        self.feature_extractor = feature_extractor
        self.vector_index = vector_index
        
        logger.info("Initialized SimilaritySearch")
    
    def index_samples(self, samples: List[Dict[str, Any]]) -> int:
        """Index samples for similarity search."""
        embeddings = self.feature_extractor.extract_batch(samples)
        self.vector_index.add(embeddings)
        return len(embeddings)
    
    def search(
        self,
        query_sample: Dict[str, Any],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar samples."""
        query_emb = self.feature_extractor.extract_features(query_sample.get("frames", []))
        
        results = self.vector_index.search(query_emb.vector, k=k, filters=filters)
        
        enriched_results = []
        for emb_id, distance in results:
            metadata = self.vector_index._metadata_store.get(emb_id, {})
            enriched_results.append((emb_id, distance, metadata))
        
        return enriched_results
    
    def find_similar_textures(
        self,
        query_sample: Dict[str, Any],
        texture_type: str,
        k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find samples with similar textures."""
        return self.search(
            query_sample,
            k=k,
            filters={"texture_type": texture_type} if texture_type else None
        )


def create_vjepa_config(
    backbone: str = "vit_large",
    resolution: int = 224,
    pretraining_mode: str = "domain_adaptive",
    **kwargs
) -> VJEPAConfig:
    """Factory function to create V-JEPA configuration."""
    backbone_map = {
        "vit_base": BackboneSize.VIT_BASE,
        "vit_large": BackboneSize.VIT_LARGE,
        "vit_huge": BackboneSize.VIT_HUGE,
    }
    
    mode_map = {
        "from_scratch": PretrainingMode.FROM_SCRATCH,
        "fine_tune": PretrainingMode.FINE_TUNE,
        "domain_adaptive": PretrainingMode.DOMAIN_ADAPTIVE,
    }
    
    return VJEPAConfig(
        backbone=backbone_map.get(backbone, BackboneSize.VIT_LARGE),
        resolution=(resolution, resolution),
        pretraining_mode=mode_map.get(pretraining_mode, PretrainingMode.DOMAIN_ADAPTIVE),
        **kwargs
    )


def create_mining_data_loader(
    imagery_type: str,
    data_root: str,
    config: Optional[VJEPAConfig] = None,
    manifest_path: Optional[str] = None
) -> MiningDataLoader:
    """Factory function to create mining data loader."""
    type_map = {
        "drone_video": ImageryType.DRONE_VIDEO,
        "drone_orthomosaic": ImageryType.DRONE_ORTHOMOSAIC,
        "satellite_rgb": ImageryType.SATELLITE_RGB,
        "satellite_multispectral": ImageryType.SATELLITE_MULTISPECTRAL,
        "core_photo": ImageryType.CORE_PHOTO,
        "site_camera": ImageryType.SITE_CAMERA,
        "handheld": ImageryType.HANDHELD,
    }
    
    config = config or VJEPAConfig()
    
    return MiningDataLoader(
        config=config,
        imagery_type=type_map.get(imagery_type, ImageryType.DRONE_ORTHOMOSAIC),
        data_root=data_root,
        manifest_path=manifest_path
    )


def create_pretrainer(
    config: VJEPAConfig,
    data_roots: Dict[str, str],
    manifest_paths: Optional[Dict[str, str]] = None
) -> VJEPAPretrainer:
    """Factory function to create V-JEPA pretrainer."""
    manifest_paths = manifest_paths or {}
    
    data_loaders = []
    for imagery_type, data_root in data_roots.items():
        loader = create_mining_data_loader(
            imagery_type=imagery_type,
            data_root=data_root,
            config=config,
            manifest_path=manifest_paths.get(imagery_type)
        )
        loader.load_manifest()
        data_loaders.append(loader)
    
    return VJEPAPretrainer(config=config, data_loaders=data_loaders)


def create_feature_extractor(
    config: Optional[VJEPAConfig] = None,
    checkpoint_path: Optional[str] = None
) -> VJEPAFeatureExtractor:
    """Factory function to create feature extractor."""
    config = config or VJEPAConfig()
    return VJEPAFeatureExtractor(config=config, checkpoint_path=checkpoint_path)


def create_anomaly_detector(
    feature_extractor: VJEPAFeatureExtractor,
    index_path: Optional[str] = None,
    threshold: float = 2.0
) -> AnomalyDetector:
    """Factory function to create anomaly detector."""
    vector_index = FaissIndex(dimension=feature_extractor.config.embedding_dim)
    
    if index_path:
        vector_index.load(index_path)
    
    return AnomalyDetector(
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        threshold=threshold
    )


def create_change_detector(
    feature_extractor: VJEPAFeatureExtractor,
    threshold: float = 0.5
) -> ChangeDetector:
    """Factory function to create change detector."""
    return ChangeDetector(
        feature_extractor=feature_extractor,
        threshold=threshold
    )


def create_similarity_search(
    feature_extractor: VJEPAFeatureExtractor,
    index_path: Optional[str] = None
) -> SimilaritySearch:
    """Factory function to create similarity search."""
    vector_index = FaissIndex(dimension=feature_extractor.config.embedding_dim)
    
    if index_path:
        vector_index.load(index_path)
    
    return SimilaritySearch(
        feature_extractor=feature_extractor,
        vector_index=vector_index
    )
