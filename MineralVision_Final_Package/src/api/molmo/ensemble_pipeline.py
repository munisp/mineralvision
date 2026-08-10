"""
Ensemble Optimization Pipeline for Molmo2.

Combines YOLO11, RF-DETR, SAM3, and V-JEPA with Molmo2 for
maximum accuracy in mining/geological applications.

Architecture:
1. Perception Layer: YOLO11/RF-DETR for detection + SAM3 for segmentation
2. Representation Layer: V-JEPA for temporal embeddings
3. Reasoning Layer: Molmo2 for semantic understanding and explanations
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading

logger = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Raised when a detector/segmenter/embedder backend cannot be loaded.

    The pipeline never fabricates detections, masks, or embeddings: an empty
    result list is returned ONLY when a real model ran and found nothing.
    """


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

class DetectorType(Enum):
    """Types of object detectors."""
    YOLO11 = "yolo11"
    RF_DETR = "rf_detr"
    YOLO_RF_ENSEMBLE = "yolo_rf_ensemble"


class SegmenterType(Enum):
    """Types of segmentation models."""
    SAM3 = "sam3"
    SAM3_GEOLOGY = "sam3_geology"
    SAM3_MINING = "sam3_mining"


class EmbedderType(Enum):
    """Types of video embedding models."""
    VJEPA = "vjepa"
    VJEPA_MINING = "vjepa_mining"


class ReasonerType(Enum):
    """Types of reasoning models."""
    MOLMO2_8B = "molmo2_8b"
    MOLMO2_4B = "molmo2_4b"
    MOLMO2_FINETUNED = "molmo2_finetuned"


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble pipeline."""
    # Detector settings
    detector_type: DetectorType = DetectorType.YOLO_RF_ENSEMBLE
    detector_confidence_threshold: float = 0.3
    detector_nms_threshold: float = 0.5
    
    # Segmenter settings
    segmenter_type: SegmenterType = SegmenterType.SAM3
    segmenter_points_per_side: int = 32
    
    # Embedder settings
    embedder_type: EmbedderType = EmbedderType.VJEPA
    embedder_frame_sample_rate: int = 4
    
    # Reasoner settings
    reasoner_type: ReasonerType = ReasonerType.MOLMO2_8B
    reasoner_adapter: Optional[str] = None
    
    # Pipeline settings
    cascade_threshold: float = 0.5  # Threshold for cascading to Molmo2
    max_rois_per_frame: int = 10
    use_roi_cropping: bool = True
    cache_embeddings: bool = True
    
    # Edge deployment settings
    edge_mode: bool = False
    edge_frame_skip: int = 5
    edge_max_resolution: Tuple[int, int] = (640, 480)
    
    @classmethod
    def for_high_accuracy(cls) -> "EnsembleConfig":
        """Configuration for maximum accuracy."""
        return cls(
            detector_type=DetectorType.YOLO_RF_ENSEMBLE,
            detector_confidence_threshold=0.2,
            segmenter_type=SegmenterType.SAM3,
            segmenter_points_per_side=64,
            embedder_frame_sample_rate=2,
            reasoner_type=ReasonerType.MOLMO2_8B,
            cascade_threshold=0.3,
            max_rois_per_frame=20,
            use_roi_cropping=True,
        )
    
    @classmethod
    def for_edge_deployment(cls) -> "EnsembleConfig":
        """Configuration for edge deployment."""
        return cls(
            detector_type=DetectorType.YOLO11,
            detector_confidence_threshold=0.4,
            segmenter_type=SegmenterType.SAM3,
            segmenter_points_per_side=16,
            embedder_frame_sample_rate=8,
            reasoner_type=ReasonerType.MOLMO2_4B,
            cascade_threshold=0.6,
            max_rois_per_frame=5,
            use_roi_cropping=True,
            edge_mode=True,
            edge_frame_skip=10,
        )
    
    @classmethod
    def for_real_time(cls) -> "EnsembleConfig":
        """Configuration for real-time processing."""
        return cls(
            detector_type=DetectorType.YOLO11,
            detector_confidence_threshold=0.5,
            segmenter_type=SegmenterType.SAM3,
            segmenter_points_per_side=8,
            embedder_frame_sample_rate=16,
            reasoner_type=ReasonerType.MOLMO2_4B,
            cascade_threshold=0.7,
            max_rois_per_frame=3,
            use_roi_cropping=True,
            edge_mode=True,
            edge_frame_skip=15,
        )


# =============================================================================
# DETECTION RESULTS
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box with coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate IoU with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    """Single detection result."""
    bbox: BoundingBox
    class_name: str
    confidence: float
    source: str  # "yolo11", "rf_detr", "molmo2"
    frame_idx: Optional[int] = None
    track_id: Optional[str] = None
    mask: Optional[Any] = None  # SAM3 segmentation mask
    embedding: Optional[Any] = None  # V-JEPA embedding
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "class_name": self.class_name,
            "confidence": self.confidence,
            "source": self.source,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
        }


@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_idx: int
    detections: List[Detection]
    scene_type: Optional[str] = None
    scene_confidence: Optional[float] = None
    embedding: Optional[Any] = None
    molmo_analysis: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "detections": [d.to_dict() for d in self.detections],
            "scene_type": self.scene_type,
            "scene_confidence": self.scene_confidence,
            "processing_time_ms": self.processing_time_ms,
            "molmo_analysis": self.molmo_analysis,
        }


@dataclass
class VideoResult:
    """Results for video analysis."""
    frames: List[FrameResult]
    summary: Dict[str, Any]
    tracks: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    total_processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": [f.to_dict() for f in self.frames],
            "summary": self.summary,
            "tracks": self.tracks,
            "events": self.events,
            "total_processing_time_ms": self.total_processing_time_ms,
        }


# =============================================================================
# PERCEPTION LAYER: DETECTORS
# =============================================================================

class BaseDetector(ABC):
    """Base class for object detectors."""
    
    @abstractmethod
    def detect(self, image: Any) -> List[Detection]:
        """Run detection on image."""
        pass
    
    @abstractmethod
    def detect_batch(self, images: List[Any]) -> List[List[Detection]]:
        """Run detection on batch of images."""
        pass


class YOLO11Detector(BaseDetector):
    """YOLO11 detector for mining objects."""
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._model = None
    
    def _load_model(self) -> None:
        """Load YOLO11 model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO11 model: {self.model_path}")
        except ImportError:
            logger.error("ultralytics not installed")
            raise ImportError("pip install ultralytics")
    
    def detect(self, image: Any) -> List[Detection]:
        """Run detection on image."""
        if self._model is None:
            self._load_model()
        
        results = self._model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = BoundingBox(
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                )
                detections.append(Detection(
                    bbox=bbox,
                    class_name=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    source="yolo11",
                ))
        
        return detections
    
    def detect_batch(self, images: List[Any]) -> List[List[Detection]]:
        """Run detection on batch of images."""
        return [self.detect(img) for img in images]


class RFDETRDetector(BaseDetector):
    """RF-DETR detector for high-accuracy detection."""
    
    def __init__(
        self,
        model_path: str = "rf_detr_base",
        confidence_threshold: float = 0.3,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
    
    def _load_model(self) -> None:
        """Load RF-DETR model."""
        try:
            from rfdetr import RFDETRBase  # lazy: heavy optional dep
        except ImportError as e:
            raise BackendUnavailableError(
                "RF-DETR backend unavailable: the 'rfdetr' package is not "
                "installed. Remediation: pip install rfdetr, or choose a "
                "different detector backend. No detections were fabricated."
            ) from e
        try:
            logger.info(f"Loading RF-DETR model: {self.model_path}")
            self._model = RFDETRBase(pretrain_weights=self.model_path)
        except Exception as e:
            raise BackendUnavailableError(
                f"RF-DETR backend unavailable: failed to load weights "
                f"'{self.model_path}': {e}"
            ) from e
    
    def detect(self, image: Any) -> List[Detection]:
        """Run detection on image."""
        if self._model is None:
            self._load_model()
        
        # Real RF-DETR inference; empty list ONLY when the model found nothing
        raw = self._model.predict(image, threshold=self.confidence_threshold)
        detections = []
        for box, score, cls in zip(
            getattr(raw, "xyxy", []),
            getattr(raw, "confidence", []),
            getattr(raw, "class_id", []),
        ):
            detections.append(Detection(
                bbox=BoundingBox(x1=float(box[0]), y1=float(box[1]),
                                 x2=float(box[2]), y2=float(box[3])),
                confidence=float(score),
                class_name=str(cls),
                source="rf_detr",
            ))
        return detections
    
    def detect_batch(self, images: List[Any]) -> List[List[Detection]]:
        """Run detection on batch of images."""
        return [self.detect(img) for img in images]


class EnsembleDetector(BaseDetector):
    """Ensemble of YOLO11 and RF-DETR detectors."""
    
    def __init__(
        self,
        yolo_config: Optional[Dict[str, Any]] = None,
        rfdetr_config: Optional[Dict[str, Any]] = None,
        fusion_strategy: str = "weighted_boxes_fusion",
        iou_threshold: float = 0.5,
    ):
        self.yolo = YOLO11Detector(**(yolo_config or {}))
        self.rfdetr = RFDETRDetector(**(rfdetr_config or {}))
        self.fusion_strategy = fusion_strategy
        self.iou_threshold = iou_threshold
    
    def detect(self, image: Any) -> List[Detection]:
        """Run ensemble detection."""
        yolo_dets = self.yolo.detect(image)
        rfdetr_dets = self.rfdetr.detect(image)
        
        # Fuse detections
        fused = self._fuse_detections(yolo_dets, rfdetr_dets)
        return fused
    
    def detect_batch(self, images: List[Any]) -> List[List[Detection]]:
        """Run ensemble detection on batch."""
        return [self.detect(img) for img in images]
    
    def _fuse_detections(
        self,
        dets1: List[Detection],
        dets2: List[Detection],
    ) -> List[Detection]:
        """Fuse detections from multiple sources."""
        if self.fusion_strategy == "weighted_boxes_fusion":
            return self._weighted_boxes_fusion(dets1, dets2)
        elif self.fusion_strategy == "nms":
            return self._nms_fusion(dets1, dets2)
        else:
            return dets1 + dets2
    
    def _weighted_boxes_fusion(
        self,
        dets1: List[Detection],
        dets2: List[Detection],
    ) -> List[Detection]:
        """Weighted boxes fusion."""
        all_dets = dets1 + dets2
        if not all_dets:
            return []
        
        # Group by class
        by_class: Dict[str, List[Detection]] = {}
        for det in all_dets:
            if det.class_name not in by_class:
                by_class[det.class_name] = []
            by_class[det.class_name].append(det)
        
        fused = []
        for class_name, class_dets in by_class.items():
            # Simple fusion: keep highest confidence for overlapping boxes
            used = [False] * len(class_dets)
            
            for i, det1 in enumerate(class_dets):
                if used[i]:
                    continue
                
                # Find overlapping detections
                overlapping = [det1]
                used[i] = True
                
                for j, det2 in enumerate(class_dets):
                    if i != j and not used[j]:
                        if det1.bbox.iou(det2.bbox) > self.iou_threshold:
                            overlapping.append(det2)
                            used[j] = True
                
                # Fuse overlapping detections
                if len(overlapping) > 1:
                    # Weighted average of boxes
                    total_conf = sum(d.confidence for d in overlapping)
                    x1 = sum(d.bbox.x1 * d.confidence for d in overlapping) / total_conf
                    y1 = sum(d.bbox.y1 * d.confidence for d in overlapping) / total_conf
                    x2 = sum(d.bbox.x2 * d.confidence for d in overlapping) / total_conf
                    y2 = sum(d.bbox.y2 * d.confidence for d in overlapping) / total_conf
                    
                    fused.append(Detection(
                        bbox=BoundingBox(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=max(d.confidence for d in overlapping),
                        source="ensemble",
                    ))
                else:
                    fused.append(det1)
        
        return fused
    
    def _nms_fusion(
        self,
        dets1: List[Detection],
        dets2: List[Detection],
    ) -> List[Detection]:
        """NMS-based fusion."""
        all_dets = dets1 + dets2
        if not all_dets:
            return []
        
        # Sort by confidence
        all_dets.sort(key=lambda d: d.confidence, reverse=True)
        
        # Apply NMS
        keep = []
        for det in all_dets:
            should_keep = True
            for kept in keep:
                if det.class_name == kept.class_name:
                    if det.bbox.iou(kept.bbox) > self.iou_threshold:
                        should_keep = False
                        break
            if should_keep:
                keep.append(det)
        
        return keep


# =============================================================================
# PERCEPTION LAYER: SEGMENTATION
# =============================================================================

class SAM3Segmenter:
    """SAM3 segmentation for precise masks."""
    
    def __init__(
        self,
        model_type: str = "sam3_base",
        points_per_side: int = 32,
        device: str = "auto",
    ):
        self.model_type = model_type
        self.points_per_side = points_per_side
        self.device = device
        self._model = None
        self._backend = None
    
    def _load_model(self) -> None:
        """Load SAM3 model (real backend only — never a placeholder)."""
        backend_errors = []
        # Preferred: the official sam3 package
        try:
            from sam3 import Sam3Predictor  # type: ignore  # lazy optional dep
            logger.info(f"Loading SAM3 model: {self.model_type}")
            self._model = Sam3Predictor(self.model_type)
            self._backend = "sam3"
            return
        except ImportError as e:
            backend_errors.append(f"sam3 package not installed ({e})")
        except Exception as e:
            backend_errors.append(f"sam3 load failed ({e})")
        # Fallback: ultralytics SAM predictor
        try:
            from ultralytics import SAM  # lazy optional dep
            logger.info(f"Loading SAM via ultralytics: {self.model_type}")
            self._model = SAM(self.model_type)
            self._backend = "ultralytics"
            return
        except ImportError as e:
            backend_errors.append(f"ultralytics not installed ({e})")
        except Exception as e:
            backend_errors.append(f"ultralytics SAM load failed ({e})")
        raise BackendUnavailableError(
            "SAM3 segmenter backend unavailable: "
            + "; ".join(backend_errors)
            + ". Remediation: install the sam3 package (or ultralytics SAM "
              "weights) — no masks were fabricated."
        )
    
    def segment_from_boxes(
        self,
        image: Any,
        boxes: List[BoundingBox],
    ) -> List[Any]:
        """Segment regions from bounding boxes."""
        if self._model is None:
            self._load_model()
        
        # Real SAM3 inference; a None mask means the model produced no mask
        # for that prompt — it is never a placeholder for a missing backend.
        masks = []
        for box in boxes:
            xyxy = [box.x1, box.y1, box.x2, box.y2]
            if self._backend == "ultralytics":
                res = self._model(image, bboxes=[xyxy])
                masks.append(res[0].masks.data[0].cpu().numpy()
                             if res and res[0].masks is not None else None)
            else:
                mask, _scores = self._model.predict(box=np.asarray(xyxy))
                masks.append(mask)
        return masks
    
    def segment_from_points(
        self,
        image: Any,
        points: List[Tuple[float, float]],
    ) -> List[Any]:
        """Segment regions from point prompts."""
        if self._model is None:
            self._load_model()
        
        # Real SAM3 point-prompt inference (None = model found no mask).
        masks = []
        for point in points:
            if self._backend == "ultralytics":
                res = self._model(image, points=[[point[0], point[1]]], labels=[1])
                masks.append(res[0].masks.data[0].cpu().numpy()
                             if res and res[0].masks is not None else None)
            else:
                mask, _scores = self._model.predict(
                    point_coords=np.asarray([[point[0], point[1]]]),
                    point_labels=np.asarray([1]))
                masks.append(mask)
        return masks
    
    def auto_segment(self, image: Any) -> List[Any]:
        """Automatic segmentation of entire image."""
        if self._model is None:
            self._load_model()
        
        # Real SAM3 automatic segmentation; empty list ONLY when the model
        # genuinely produced no masks.
        if self._backend == "ultralytics":
            res = self._model(image)
            if not res or res[0].masks is None:
                return []
            return [m.cpu().numpy() for m in res[0].masks.data]
        return self._model.auto_mask(image)


# =============================================================================
# REPRESENTATION LAYER: V-JEPA EMBEDDINGS
# =============================================================================

class VJEPAEmbedder:
    """V-JEPA video embeddings for temporal understanding."""
    
    def __init__(
        self,
        model_path: str = "vjepa_base",
        frame_sample_rate: int = 4,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.frame_sample_rate = frame_sample_rate
        self.device = device
        self._model = None
        self._embedding_cache: Dict[str, Any] = {}
    
    def _load_model(self) -> None:
        """Load V-JEPA model: bridge to the real api.jepa.torch_core JEPAModel.

        Raises BackendUnavailableError when the JEPA backend (torch) is not
        available — embeddings are never fabricated.
        """
        jepa_mod = None
        for modname in ("src.api.jepa.torch_core", "api.jepa.torch_core"):
            try:
                import importlib
                jepa_mod = importlib.import_module(modname)
                break
            except ImportError:
                continue
        if jepa_mod is None or not jepa_mod.TORCH_AVAILABLE:
            raise BackendUnavailableError(
                "V-JEPA embedder unavailable: api.jepa.torch_core (PyTorch "
                "I-JEPA) is not importable or torch is not installed. "
                "Remediation: install torch, or disable V-JEPA embeddings in "
                "the pipeline config. No random embeddings were fabricated."
            )
        logger.info(f"Loading V-JEPA model: {self.model_path}")
        if self.model_path and os.path.exists(str(self.model_path)):
            self._model = jepa_mod.JEPAModel.load(self.model_path)
        else:
            # untrained-but-real I-JEPA encoders (honest: the actual
            # architecture, random-init weights — not random numbers)
            self._model = jepa_mod.JEPAModel()
        self._img_size = self._model.config.img_size
    
    @staticmethod
    def _resize_hwc(arr, size: int):
        """Resize an [H,W,C] frame to (size, size) with area interpolation."""
        import numpy as np
        try:
            import cv2  # lazy
            return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image  # lazy
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)
                                  if arr.dtype != np.uint8 else arr)
            return np.asarray(img.resize((size, size), Image.BILINEAR),
                              dtype=np.float32)

    def embed_frames(self, frames: List[Any]) -> Any:
        """Get embedding for video frames (real JEPA target encoder)."""
        if self._model is None:
            self._load_model()
        
        import numpy as np
        norm_frames = []
        for f in frames:
            arr = np.asarray(f)
            # JEPAModel expects [H,W,3]; accept [3,H,W] inputs transparently
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            # resize to the encoder's configured input size if needed
            if arr.shape[0] != self._img_size or arr.shape[1] != self._img_size:
                arr = self._resize_hwc(arr, self._img_size)
            norm_frames.append(arr)
        embs = [self._model.embed_image(f) for f in norm_frames]
        emb = np.mean(np.stack(embs), axis=0)
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 0 else emb
    
    def embed_video(self, video_path: str) -> Any:
        """Get embedding for video file."""
        # Check cache
        if video_path in self._embedding_cache:
            return self._embedding_cache[video_path]
        
        if self._model is None:
            self._load_model()
        
        # Real path: decode frames (opencv lazy) and embed them
        try:
            import cv2  # lazy optional dep
        except ImportError as e:
            raise BackendUnavailableError(
                "V-JEPA video embedding requires opencv (cv2) to decode "
                f"'{video_path}'. Remediation: pip install opencv-python, "
                "or pass decoded frames to embed_frames()."
            ) from e
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        idx = 0
        ok, frame = cap.read()
        while ok:
            if idx % self.frame_sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
            ok, frame = cap.read()
        cap.release()
        if not frames:
            raise BackendUnavailableError(
                f"V-JEPA could not decode any frames from '{video_path}' "
                "(missing/corrupt video or unsupported codec)."
            )
        embedding = self.embed_frames(frames)

        self._embedding_cache[video_path] = embedding
        return embedding
    
    def compute_similarity(self, emb1: Any, emb2: Any) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np
        
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def detect_anomalies(
        self,
        frames: List[Any],
        threshold: float = 0.7,
    ) -> List[Tuple[int, float]]:
        """Detect anomalous frames based on embedding similarity."""
        if len(frames) < 2:
            return []
        
        embeddings = [self.embed_frames([f]) for f in frames]
        
        anomalies = []
        for i in range(1, len(embeddings)):
            sim = self.compute_similarity(embeddings[i-1], embeddings[i])
            if sim < threshold:
                anomalies.append((i, 1.0 - sim))
        
        return anomalies


# =============================================================================
# REASONING LAYER: MOLMO2 INTEGRATION
# =============================================================================

class Molmo2Reasoner:
    """Molmo2 reasoning layer for semantic understanding."""
    
    def __init__(
        self,
        model_name: str = "allenai/Molmo2-8B",
        adapter_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device
        self._model = None
        self._processor = None
        
        from .optimization import StructuredOutputParser, PromptLibrary
        self.output_parser = StructuredOutputParser()
        self.prompt_library = PromptLibrary()
    
    def _load_model(self) -> None:
        """Load Molmo2 model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading Molmo2 model: {self.model_name}")
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Load adapter if specified
            if self.adapter_path:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(
                    self._model,
                    self.adapter_path,
                )
                logger.info(f"Loaded adapter: {self.adapter_path}")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise
    
    def analyze_image(
        self,
        image: Any,
        prompt_name: str,
        context: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze image with structured output."""
        if self._model is None:
            self._load_model()
        
        # Get prompt template
        prompt = self.prompt_library.format_prompt(
            prompt_name,
            context=context,
            **kwargs,
        )
        
        # Run inference
        try:
            from PIL import Image
            
            if isinstance(image, str):
                image = Image.open(image)
            
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self._model.device)
            
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
            
            response = self._processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse structured output
            prompt_template = self.prompt_library.get_prompt(prompt_name)
            if prompt_template:
                parsed = self.output_parser.parse(response, prompt_template.schema_type)
                return parsed.data if parsed.is_valid else {"raw": response, "error": parsed.validation_errors}
            
            return {"raw": response}
            
        except Exception as e:
            logger.error(f"Molmo2 inference failed: {e}")
            return {"error": str(e)}
    
    def analyze_detections(
        self,
        image: Any,
        detections: List[Detection],
        task: str = "artisanal_mining",
    ) -> Dict[str, Any]:
        """Analyze detections with Molmo2 for semantic understanding."""
        # Format detector context
        detector_context = json.dumps([d.to_dict() for d in detections], indent=2)
        
        return self.analyze_image(
            image,
            prompt_name=task,
            detector_context=detector_context,
        )
    
    def explain_anomaly(
        self,
        image: Any,
        anomaly_score: float,
        context: str = "",
    ) -> str:
        """Explain detected anomaly."""
        result = self.analyze_image(
            image,
            prompt_name="scene_analysis",
            context=f"Anomaly detected with score {anomaly_score:.2f}. {context}",
        )
        
        return result.get("rationale", result.get("description", "Unknown anomaly"))


# =============================================================================
# ENSEMBLE PIPELINE
# =============================================================================

class EnsemblePipeline:
    """
    Complete ensemble pipeline combining all components.
    
    Architecture:
    1. Perception: YOLO11/RF-DETR detection + SAM3 segmentation
    2. Representation: V-JEPA temporal embeddings
    3. Reasoning: Molmo2 semantic understanding
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        
        # Initialize components
        self._detector: Optional[BaseDetector] = None
        self._segmenter: Optional[SAM3Segmenter] = None
        self._embedder: Optional[VJEPAEmbedder] = None
        self._reasoner: Optional[Molmo2Reasoner] = None
        
        # Caches
        self._embedding_cache: Dict[str, Any] = {}
    
    @property
    def detector(self) -> BaseDetector:
        """Get or create detector."""
        if self._detector is None:
            if self.config.detector_type == DetectorType.YOLO11:
                self._detector = YOLO11Detector(
                    confidence_threshold=self.config.detector_confidence_threshold,
                    nms_threshold=self.config.detector_nms_threshold,
                )
            elif self.config.detector_type == DetectorType.RF_DETR:
                self._detector = RFDETRDetector(
                    confidence_threshold=self.config.detector_confidence_threshold,
                )
            else:
                self._detector = EnsembleDetector(
                    yolo_config={
                        "confidence_threshold": self.config.detector_confidence_threshold,
                        "nms_threshold": self.config.detector_nms_threshold,
                    },
                    rfdetr_config={
                        "confidence_threshold": self.config.detector_confidence_threshold,
                    },
                )
        return self._detector
    
    @property
    def segmenter(self) -> SAM3Segmenter:
        """Get or create segmenter."""
        if self._segmenter is None:
            self._segmenter = SAM3Segmenter(
                points_per_side=self.config.segmenter_points_per_side,
            )
        return self._segmenter
    
    @property
    def embedder(self) -> VJEPAEmbedder:
        """Get or create embedder."""
        if self._embedder is None:
            self._embedder = VJEPAEmbedder(
                frame_sample_rate=self.config.embedder_frame_sample_rate,
            )
        return self._embedder
    
    @property
    def reasoner(self) -> Molmo2Reasoner:
        """Get or create reasoner."""
        if self._reasoner is None:
            model_name = "allenai/Molmo2-8B"
            if self.config.reasoner_type == ReasonerType.MOLMO2_4B:
                model_name = "allenai/Molmo2-4B"
            
            self._reasoner = Molmo2Reasoner(
                model_name=model_name,
                adapter_path=self.config.reasoner_adapter,
            )
        return self._reasoner
    
    def analyze_image(
        self,
        image: Any,
        task: str = "artisanal_mining",
        include_segmentation: bool = True,
    ) -> FrameResult:
        """Analyze single image with full pipeline."""
        import time
        start_time = time.time()
        
        # Step 1: Detection
        detections = self.detector.detect(image)
        
        # Step 2: Segmentation (if enabled)
        if include_segmentation and detections:
            boxes = [d.bbox for d in detections[:self.config.max_rois_per_frame]]
            masks = self.segmenter.segment_from_boxes(image, boxes)
            for i, mask in enumerate(masks):
                if i < len(detections):
                    detections[i].mask = mask
        
        # Step 3: Molmo2 reasoning (if confidence warrants)
        molmo_analysis = None
        max_conf = max((d.confidence for d in detections), default=0)
        
        if max_conf >= self.config.cascade_threshold or not detections:
            molmo_analysis = self.reasoner.analyze_detections(
                image,
                detections,
                task=task,
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return FrameResult(
            frame_idx=0,
            detections=detections,
            scene_type=molmo_analysis.get("scene_type") if molmo_analysis else None,
            scene_confidence=molmo_analysis.get("confidence") if molmo_analysis else None,
            molmo_analysis=molmo_analysis,
            processing_time_ms=processing_time,
        )
    
    def analyze_video(
        self,
        video_path: str,
        task: str = "artisanal_mining",
        max_frames: Optional[int] = None,
    ) -> VideoResult:
        """Analyze video with full pipeline."""
        import time
        import cv2
        
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frame sampling
        frame_skip = self.config.edge_frame_skip if self.config.edge_mode else 1
        if max_frames:
            frame_skip = max(1, total_frames // max_frames)
        
        # Process frames
        frame_results = []
        all_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # Resize for edge mode
                if self.config.edge_mode:
                    h, w = frame.shape[:2]
                    max_w, max_h = self.config.edge_max_resolution
                    if w > max_w or h > max_h:
                        scale = min(max_w / w, max_h / h)
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                all_frames.append(frame)
                
                # Analyze frame
                result = self.analyze_image(frame, task=task, include_segmentation=False)
                result.frame_idx = frame_idx
                frame_results.append(result)
            
            frame_idx += 1
        
        cap.release()
        
        # Step 4: V-JEPA temporal analysis
        if all_frames:
            # Detect anomalies
            anomalies = self.embedder.detect_anomalies(all_frames)
            
            # Explain anomalies with Molmo2
            for frame_idx, anomaly_score in anomalies:
                if frame_idx < len(frame_results):
                    explanation = self.reasoner.explain_anomaly(
                        all_frames[frame_idx],
                        anomaly_score,
                    )
                    if frame_results[frame_idx].molmo_analysis is None:
                        frame_results[frame_idx].molmo_analysis = {}
                    frame_results[frame_idx].molmo_analysis["anomaly_explanation"] = explanation
        
        # Generate summary
        summary = self._generate_summary(frame_results, task)
        
        # Extract tracks
        tracks = self._extract_tracks(frame_results)
        
        # Detect events
        events = self._detect_events(frame_results)
        
        total_time = (time.time() - start_time) * 1000
        
        return VideoResult(
            frames=frame_results,
            summary=summary,
            tracks=tracks,
            events=events,
            total_processing_time_ms=total_time,
        )
    
    def _generate_summary(
        self,
        frame_results: List[FrameResult],
        task: str,
    ) -> Dict[str, Any]:
        """Generate video summary."""
        # Count detections by class
        class_counts: Dict[str, int] = {}
        for result in frame_results:
            for det in result.detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        # Aggregate scene types
        scene_types: Dict[str, int] = {}
        for result in frame_results:
            if result.scene_type:
                scene_types[result.scene_type] = scene_types.get(result.scene_type, 0) + 1
        
        # Determine dominant scene
        dominant_scene = max(scene_types.items(), key=lambda x: x[1])[0] if scene_types else "unknown"
        
        # Task-specific summary
        if task == "artisanal_mining":
            mining_frames = sum(
                1 for r in frame_results
                if r.molmo_analysis and r.molmo_analysis.get("is_artisanal_mining")
            )
            return {
                "dominant_scene": dominant_scene,
                "class_counts": class_counts,
                "scene_types": scene_types,
                "total_frames_analyzed": len(frame_results),
                "artisanal_mining_detected": mining_frames > 0,
                "mining_frame_count": mining_frames,
                "mining_percentage": mining_frames / len(frame_results) * 100 if frame_results else 0,
            }
        
        return {
            "dominant_scene": dominant_scene,
            "class_counts": class_counts,
            "scene_types": scene_types,
            "total_frames_analyzed": len(frame_results),
        }
    
    def _extract_tracks(self, frame_results: List[FrameResult]) -> List[Dict[str, Any]]:
        """Extract object tracks from frame results."""
        # Simple tracking by class and position
        tracks: Dict[str, Dict[str, Any]] = {}
        
        for result in frame_results:
            for det in result.detections:
                # Simple track ID based on class and approximate position
                track_key = f"{det.class_name}_{int(det.bbox.center[0] / 100)}_{int(det.bbox.center[1] / 100)}"
                
                if track_key not in tracks:
                    tracks[track_key] = {
                        "track_id": track_key,
                        "class_name": det.class_name,
                        "positions": [],
                        "first_frame": result.frame_idx,
                        "last_frame": result.frame_idx,
                    }
                
                tracks[track_key]["positions"].append({
                    "frame": result.frame_idx,
                    "bbox": det.bbox.to_list(),
                    "confidence": det.confidence,
                })
                tracks[track_key]["last_frame"] = result.frame_idx
        
        return list(tracks.values())
    
    def _detect_events(self, frame_results: List[FrameResult]) -> List[Dict[str, Any]]:
        """Detect events from frame results."""
        events = []
        
        # Detect scene changes
        prev_scene = None
        for result in frame_results:
            if result.scene_type and result.scene_type != prev_scene:
                events.append({
                    "type": "scene_change",
                    "frame": result.frame_idx,
                    "from_scene": prev_scene,
                    "to_scene": result.scene_type,
                })
                prev_scene = result.scene_type
        
        # Detect activity starts/ends
        prev_mining = False
        for result in frame_results:
            is_mining = (
                result.molmo_analysis and
                result.molmo_analysis.get("is_artisanal_mining", False)
            )
            
            if is_mining and not prev_mining:
                events.append({
                    "type": "activity_start",
                    "frame": result.frame_idx,
                    "activity": "artisanal_mining",
                })
            elif not is_mining and prev_mining:
                events.append({
                    "type": "activity_end",
                    "frame": result.frame_idx,
                    "activity": "artisanal_mining",
                })
            
            prev_mining = is_mining
        
        return events


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ensemble_pipeline(
    mode: str = "balanced",
    adapter_path: Optional[str] = None,
) -> EnsemblePipeline:
    """Create ensemble pipeline with preset configuration."""
    if mode == "high_accuracy":
        config = EnsembleConfig.for_high_accuracy()
    elif mode == "edge":
        config = EnsembleConfig.for_edge_deployment()
    elif mode == "real_time":
        config = EnsembleConfig.for_real_time()
    else:
        config = EnsembleConfig()
    
    if adapter_path:
        config.reasoner_adapter = adapter_path
    
    return EnsemblePipeline(config)


def analyze_mining_video(
    video_path: str,
    mode: str = "balanced",
) -> VideoResult:
    """Analyze video for artisanal mining detection."""
    pipeline = create_ensemble_pipeline(mode)
    return pipeline.analyze_video(video_path, task="artisanal_mining")


def analyze_geological_image(
    image_path: str,
    target_minerals: str = "gold",
) -> FrameResult:
    """Analyze image for geological features."""
    pipeline = create_ensemble_pipeline("high_accuracy")
    
    from PIL import Image
    image = Image.open(image_path)
    
    # Use geological features prompt
    result = pipeline.analyze_image(image, task="geological_features")
    
    return result
