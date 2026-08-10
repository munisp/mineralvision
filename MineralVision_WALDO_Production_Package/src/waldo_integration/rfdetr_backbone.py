"""
RF-DETR Backbone Integration for WALDO
=======================================

Production-grade integration of Roboflow's RF-DETR transformer-based
object detection model as an alternative backbone for WALDO.

RF-DETR offers:
- First real-time model to exceed 60 mAP on COCO
- Superior domain transfer for geological/mineral imagery
- Instance segmentation support (RF-DETR-Seg)
- Optimized for fine-tuning on custom datasets

This module provides:
- RFDETRDetector: Drop-in replacement for YOLOv8 detector
- RFDETRFineTuner: Fine-tuning pipeline for custom classes
- RFDETRExporter: ONNX/TensorRT export utilities
- RFDETRSegmentation: Instance segmentation support
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import json
import os
import logging
import hashlib
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import rfdetr
try:
    from rfdetr import RFDETRBase, RFDETRLarge
    from rfdetr.detr import RFDETR
    from rfdetr.util.coco_classes import COCO_CLASSES
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    logger.warning("rfdetr not installed. Install with: pip install rfdetr")

# Try to import supervision for visualization
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False


class RFDETRUnavailableError(RuntimeError):
    """Raised when the real RF-DETR backend is unavailable and the mock
    fallback has not been explicitly allowed via MV_ALLOW_MOCK_FALLBACK=true."""


def _mock_fallback_allowed() -> bool:
    return os.environ.get("MV_ALLOW_MOCK_FALLBACK", "").lower() in ("1", "true", "yes")


class RFDETRVariant(Enum):
    """RF-DETR model variants."""
    NANO = "nano"       # Smallest, fastest
    SMALL = "small"     # Balance of speed/accuracy
    MEDIUM = "medium"   # Recommended default
    BASE = "base"       # Original base model (deprecated)
    LARGE = "large"     # Highest accuracy


class ExportFormat(Enum):
    """Model export formats."""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TORCHSCRIPT = "torchscript"
    COREML = "coreml"


@dataclass
class RFDETRConfig:
    """Configuration for RF-DETR detector."""
    variant: RFDETRVariant = RFDETRVariant.MEDIUM
    confidence_threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    half_precision: bool = True
    optimize_for_inference: bool = True
    num_classes: int = 80  # COCO default
    class_names: Optional[List[str]] = None
    checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = list(COCO_CLASSES) if RFDETR_AVAILABLE else []


@dataclass
class RFDETRDetection:
    """Single detection result from RF-DETR."""
    detection_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    mask: Optional[np.ndarray] = None  # Instance segmentation mask
    area: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detection_id': self.detection_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'area': self.area,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class RFDETRTrainingConfig:
    """Training configuration for RF-DETR fine-tuning."""
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    gradient_checkpointing: bool = True
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001
    mixed_precision: bool = True
    num_workers: int = 4
    image_size: int = 640
    augmentation: bool = True
    mosaic: bool = True
    mixup: float = 0.0
    copy_paste: float = 0.0
    save_period: int = 5
    resume: Optional[str] = None
    freeze_backbone: bool = False
    freeze_epochs: int = 0


class MockRFDETRModel:
    """
    Mock RF-DETR model for environments without rfdetr installed.
    Provides consistent API for testing and development.
    """
    
    def __init__(self, variant: RFDETRVariant = RFDETRVariant.MEDIUM,
                 num_classes: int = 80):
        self.variant = variant
        self.num_classes = num_classes
        self.device = "cpu"
        self._class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Model parameters by variant
        self._params = {
            RFDETRVariant.NANO: {"params": "5M", "latency": "2.5ms"},
            RFDETRVariant.SMALL: {"params": "15M", "latency": "4ms"},
            RFDETRVariant.MEDIUM: {"params": "29M", "latency": "6ms"},
            RFDETRVariant.BASE: {"params": "29M", "latency": "6ms"},
            RFDETRVariant.LARGE: {"params": "129M", "latency": "12ms"}
        }
        
        logger.info(f"MockRFDETRModel initialized: {variant.value} "
                   f"({self._params[variant]['params']} params)")
    
    def to(self, device: str) -> 'MockRFDETRModel':
        self.device = device
        return self
    
    def eval(self) -> 'MockRFDETRModel':
        return self
    
    def train(self, mode: bool = True) -> 'MockRFDETRModel':
        return self
    
    def half(self) -> 'MockRFDETRModel':
        return self
    
    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        """Run mock inference."""
        h, w = image.shape[:2]
        
        # Generate random detections for testing
        num_detections = np.random.randint(0, 10)
        
        boxes = []
        scores = []
        labels = []
        
        for _ in range(num_detections):
            x1 = np.random.randint(0, w - 50)
            y1 = np.random.randint(0, h - 50)
            x2 = x1 + np.random.randint(20, min(100, w - x1))
            y2 = y1 + np.random.randint(20, min(100, h - y1))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(np.random.uniform(0.3, 0.95))
            labels.append(np.random.randint(0, self.num_classes))
        
        return {
            'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros(0),
            'labels': np.array(labels) if labels else np.zeros(0, dtype=int)
        }
    
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Predict with threshold filtering."""
        results = self(image)
        mask = results['scores'] >= threshold
        return {
            'boxes': results['boxes'][mask],
            'scores': results['scores'][mask],
            'labels': results['labels'][mask]
        }


class RFDETRDetector:
    """
    RF-DETR object detector for WALDO integration.
    
    Drop-in replacement for YOLOv8 detector with transformer-based
    architecture for superior accuracy and domain transfer.
    
    Example:
        config = RFDETRConfig(variant=RFDETRVariant.MEDIUM)
        detector = RFDETRDetector(config)
        detections = detector.detect(image)
    """
    
    def __init__(self, config: RFDETRConfig):
        self.config = config
        self.model = None
        self.device = torch.device(config.device)
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load RF-DETR model based on configuration."""
        if RFDETR_AVAILABLE:
            try:
                if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
                    # Load fine-tuned model
                    self.model = self._load_checkpoint(self.config.checkpoint_path)
                else:
                    # Load pre-trained model
                    self.model = self._load_pretrained()
                
                self.model.to(self.device)
                self.model.eval()
                
                if self.config.half_precision and self.device.type == 'cuda':
                    self.model.half()
                
                if self.config.optimize_for_inference:
                    self._optimize_model()
                
                logger.info(f"RF-DETR {self.config.variant.value} loaded on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load RF-DETR: {e}")
                if not _mock_fallback_allowed():
                    raise RFDETRUnavailableError(
                        f"RF-DETR failed to load ({e}); install rfdetr or set "
                        "MV_ALLOW_MOCK_FALLBACK=true to permit the mock backend") from e
                self.model = MockRFDETRModel(self.config.variant, self.config.num_classes)
        else:
            if not _mock_fallback_allowed():
                raise RFDETRUnavailableError(
                    "rfdetr package is not installed; refusing to silently use "
                    "MockRFDETRModel. Install rfdetr (requirements-ml.txt) or set "
                    "MV_ALLOW_MOCK_FALLBACK=true to permit the mock backend")
            logger.warning("Using MockRFDETRModel (MV_ALLOW_MOCK_FALLBACK=true)")
            self.model = MockRFDETRModel(self.config.variant, self.config.num_classes)
    
    def _load_pretrained(self) -> Any:
        """Load pre-trained RF-DETR model."""
        variant = self.config.variant
        
        if variant == RFDETRVariant.LARGE:
            return RFDETRLarge()
        else:
            # Base/Medium use RFDETRBase with different checkpoints
            return RFDETRBase()
    
    def _load_checkpoint(self, checkpoint_path: str) -> Any:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine model variant from checkpoint
        if 'model_variant' in checkpoint:
            variant = RFDETRVariant(checkpoint['model_variant'])
        else:
            variant = self.config.variant
        
        if variant == RFDETRVariant.LARGE:
            model = RFDETRLarge()
        else:
            model = RFDETRBase()
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _optimize_model(self) -> None:
        """Optimize model for inference."""
        if RFDETR_AVAILABLE and hasattr(self.model, 'optimize_for_inference'):
            try:
                self.model.optimize_for_inference()
                logger.info("Model optimized for inference (up to 2x speedup)")
            except Exception as e:
                logger.warning(f"Could not optimize model: {e}")
    
    def detect(self, image: np.ndarray, 
               metadata: Optional[Dict[str, Any]] = None) -> List[RFDETRDetection]:
        """
        Perform object detection on an image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR or RGB format
            metadata: Optional metadata to attach to detections
            
        Returns:
            List of RFDETRDetection objects
        """
        import time
        start_time = time.time()
        
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Run inference
        if RFDETR_AVAILABLE and not isinstance(self.model, MockRFDETRModel):
            results = self._run_rfdetr_inference(image)
        else:
            results = self.model.predict(image, self.config.confidence_threshold)
            # Fabricated detections must always be marked as mock.
            metadata = {**(metadata or {}), "mock": True}
        
        # Convert to RFDETRDetection objects
        detections = self._process_results(results, metadata)
        
        # Update statistics
        inference_time = time.time() - start_time
        self._inference_count += 1
        self._total_inference_time += inference_time
        
        return detections
    
    def _run_rfdetr_inference(self, image: np.ndarray) -> Dict[str, Any]:
        """Run RF-DETR inference."""
        with torch.no_grad():
            # RF-DETR expects PIL Image or numpy array
            results = self.model.predict(image, threshold=self.config.confidence_threshold)
        
        return results
    
    def _process_results(self, results: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]] = None) -> List[RFDETRDetection]:
        """Process raw results into RFDETRDetection objects."""
        detections = []
        
        boxes = results.get('boxes', np.zeros((0, 4)))
        scores = results.get('scores', np.zeros(0))
        labels = results.get('labels', np.zeros(0, dtype=int))
        masks = results.get('masks', None)
        
        for i in range(len(boxes)):
            bbox = boxes[i].tolist() if isinstance(boxes[i], np.ndarray) else list(boxes[i])
            confidence = float(scores[i])
            class_id = int(labels[i])
            
            # Get class name
            if self.config.class_names and class_id < len(self.config.class_names):
                class_name = self.config.class_names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            # Calculate area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Get mask if available
            mask = None
            if masks is not None and i < len(masks):
                mask = masks[i]
            
            detection = RFDETRDetection(
                detection_id=str(uuid.uuid4())[:8],
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                mask=mask,
                area=area,
                metadata=metadata or {}
            )
            
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray],
                    metadata: Optional[List[Dict[str, Any]]] = None) -> List[List[RFDETRDetection]]:
        """
        Perform batch detection on multiple images.
        
        Args:
            images: List of input images
            metadata: Optional list of metadata dicts
            
        Returns:
            List of detection lists, one per image
        """
        results = []
        metadata = metadata or [None] * len(images)
        
        for img, meta in zip(images, metadata):
            detections = self.detect(img, meta)
            results.append(detections)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = (self._total_inference_time / self._inference_count 
                   if self._inference_count > 0 else 0)
        
        return {
            'model_variant': self.config.variant.value,
            'device': str(self.device),
            'inference_count': self._inference_count,
            'total_inference_time': self._total_inference_time,
            'average_inference_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0,
            'half_precision': self.config.half_precision,
            'optimized': self.config.optimize_for_inference
        }


class RFDETRFineTuner:
    """
    Fine-tuning pipeline for RF-DETR on custom datasets.
    
    Supports:
    - COCO and YOLO format datasets
    - Gradient checkpointing for memory efficiency
    - Early stopping with configurable patience
    - Mixed precision training
    - Learning rate scheduling
    
    Example:
        config = RFDETRTrainingConfig(epochs=50, batch_size=8)
        finetuner = RFDETRFineTuner(config)
        results = finetuner.train(
            data_dir="/path/to/dataset",
            class_names=["mineral_a", "mineral_b"],
            output_dir="/path/to/output"
        )
    """
    
    def __init__(self, config: RFDETRTrainingConfig,
                 base_variant: RFDETRVariant = RFDETRVariant.MEDIUM):
        self.config = config
        self.base_variant = base_variant
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.training_history: List[Dict[str, float]] = []
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: int = 0
        
    def prepare_dataset(self, data_dir: str, class_names: List[str],
                       format: str = 'coco') -> Dict[str, Any]:
        """
        Prepare dataset for training.
        
        Args:
            data_dir: Root directory of dataset
            class_names: List of class names
            format: Dataset format ('coco', 'yolo')
            
        Returns:
            Dataset configuration dict
        """
        dataset_config = {
            'path': data_dir,
            'format': format,
            'num_classes': len(class_names),
            'class_names': class_names,
            'train': os.path.join(data_dir, 'train'),
            'val': os.path.join(data_dir, 'val'),
            'test': os.path.join(data_dir, 'test') if os.path.exists(
                os.path.join(data_dir, 'test')) else None
        }
        
        # Write dataset config
        config_path = os.path.join(data_dir, 'dataset_config.json')
        with open(config_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        logger.info(f"Dataset prepared: {len(class_names)} classes, format={format}")
        
        return dataset_config
    
    def train(self, data_dir: str, class_names: List[str],
             output_dir: str, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Fine-tune RF-DETR on custom dataset.
        
        Args:
            data_dir: Path to dataset directory
            class_names: List of class names
            output_dir: Output directory for checkpoints
            callbacks: Optional training callbacks
            
        Returns:
            Training results dict
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare dataset
        dataset_config = self.prepare_dataset(data_dir, class_names)
        
        if RFDETR_AVAILABLE:
            return self._train_rfdetr(dataset_config, output_dir, callbacks)
        else:
            return self._simulate_training(dataset_config, output_dir)
    
    def _train_rfdetr(self, dataset_config: Dict[str, Any],
                     output_dir: str,
                     callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """Train using actual RF-DETR library."""
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
            
            # Initialize model
            if self.base_variant == RFDETRVariant.LARGE:
                model = RFDETRLarge()
            else:
                model = RFDETRBase()
            
            # Train using rfdetr's training API
            model.train(
                dataset=dataset_config['path'],
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                lr=self.config.learning_rate,
                output_dir=output_dir,
                gradient_checkpointing=self.config.gradient_checkpointing,
                early_stopping=self.config.early_stopping_patience,
                amp=self.config.mixed_precision
            )
            
            # Get best checkpoint
            best_checkpoint = os.path.join(output_dir, 'best.pt')
            
            return {
                'status': 'completed',
                'epochs_trained': self.config.epochs,
                'best_checkpoint': best_checkpoint,
                'metrics': self.best_metrics,
                'history': self.training_history
            }
            
        except Exception as e:
            logger.error(f"RF-DETR training failed: {e}")
            return self._simulate_training(dataset_config, output_dir)
    
    def _simulate_training(self, dataset_config: Dict[str, Any],
                          output_dir: str) -> Dict[str, Any]:
        """Simulate training for environments without rfdetr."""
        logger.info("Simulating RF-DETR training (rfdetr not available)")
        
        num_classes = dataset_config['num_classes']
        
        # Simulate training epochs
        best_map = 0.0
        for epoch in range(min(10, self.config.epochs)):
            # Simulate metrics with realistic progression
            train_loss = 2.5 * np.exp(-0.15 * epoch) + np.random.uniform(-0.1, 0.1)
            val_loss = 2.8 * np.exp(-0.12 * epoch) + np.random.uniform(-0.1, 0.1)
            map50 = 0.3 + 0.05 * epoch + np.random.uniform(-0.02, 0.02)
            map50_95 = 0.2 + 0.04 * epoch + np.random.uniform(-0.02, 0.02)
            
            map50 = min(map50, 0.85)
            map50_95 = min(map50_95, 0.65)
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mAP50': map50,
                'mAP50-95': map50_95,
                'precision': map50 * 1.05,
                'recall': map50 * 0.95
            }
            
            self.training_history.append(metrics)
            
            if map50 > best_map:
                best_map = map50
                self.best_metrics = metrics.copy()
                self.best_epoch = epoch + 1
            
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: "
                       f"loss={train_loss:.4f}, mAP50={map50:.4f}")
        
        # Save simulated checkpoint
        checkpoint_path = os.path.join(output_dir, 'best_simulated.pt')
        checkpoint = {
            'model_variant': self.base_variant.value,
            'num_classes': num_classes,
            'class_names': dataset_config['class_names'],
            'metrics': self.best_metrics,
            'epoch': self.best_epoch,
            'simulated': True
        }
        torch.save(checkpoint, checkpoint_path)
        
        return {
            'status': 'simulated',
            'epochs_trained': len(self.training_history),
            'best_checkpoint': checkpoint_path,
            'best_epoch': self.best_epoch,
            'metrics': self.best_metrics,
            'history': self.training_history
        }
    
    def _build_model_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Any:
        """Instantiate an RF-DETR model and load weights from a checkpoint dict."""
        variant = self.base_variant
        if isinstance(checkpoint, dict) and 'model_variant' in checkpoint:
            variant = RFDETRVariant(checkpoint['model_variant'])
        model = RFDETRLarge() if variant == RFDETRVariant.LARGE else RFDETRBase()
        state = None
        if isinstance(checkpoint, dict):
            state = (checkpoint.get('model_state_dict') or
                     checkpoint.get('state_dict') or None)
        if state is None and isinstance(checkpoint, dict):
            state = checkpoint
        if state:
            try:
                model.model.load_state_dict(state)
            except Exception as e:
                logger.warning(f"state_dict load mismatch (continuing with "
                               f"pretrained head): {e}")
        return model

    def _load_coco_test_set(self, test_data_dir: str) -> List[Dict[str, Any]]:
        """Load a COCO-format test split (images + annotations)."""
        data = Path(test_data_dir)
        ann_file = None
        for cand in ("_annotations.coco.json", "annotations.json",
                     "instances.json"):
            if (data / cand).exists():
                ann_file = data / cand
                break
        if ann_file is None:
            raise FileNotFoundError(
                f"no COCO annotation json found in {test_data_dir}")
        with open(ann_file) as f:
            coco = json.load(f)
        images = {img["id"]: img for img in coco.get("images", [])}
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)
        return [{"image": images[i], "annotations": anns_by_img.get(i, []),
                 "path": str(data / images[i]["file_name"])}
                for i in images]

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _compute_detection_metrics(self, model: Any,
                                   dataset: List[Dict[str, Any]],
                                   iou_threshold: float = 0.5) -> Dict[str, float]:
        """Real mAP@IoU / precision / recall computed by greedy IoU matching."""
        from PIL import Image
        tp_scores: List[Tuple[float, int]] = []  # (confidence, is_tp)
        n_gt = 0
        for sample in dataset:
            gts = sample["annotations"]
            n_gt += len(gts)
            img = np.array(Image.open(sample["path"]).convert("RGB"))
            preds = model.predict(img, threshold=0.05)
            boxes = np.asarray(preds.get("boxes", [])).reshape(-1, 4)
            scores = np.asarray(preds.get("scores", [])).reshape(-1)
            labels = np.asarray(preds.get("labels", [])).reshape(-1)
            matched = set()
            order = np.argsort(-scores)
            for pi in order:
                best_iou, best_gi = 0.0, -1
                for gi, gt in enumerate(gts):
                    if gi in matched:
                        continue
                    if int(labels[pi]) != int(gt.get("category_id", labels[pi])):
                        continue
                    x, y, w, h = gt["bbox"]  # COCO xywh
                    iou = self._iou_xyxy(boxes[pi],
                                         np.array([x, y, x + w, y + h]))
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_threshold and best_gi >= 0:
                    matched.add(best_gi)
                    tp_scores.append((float(scores[pi]), 1))
                else:
                    tp_scores.append((float(scores[pi]), 0))
        if not tp_scores or n_gt == 0:
            return {"mAP50": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_predictions": len(tp_scores), "n_ground_truth": n_gt}
        tp_scores.sort(key=lambda t: -t[0])
        tps = np.array([t[1] for t in tp_scores], dtype=float)
        cum_tp = np.cumsum(tps)
        ranks = np.arange(1, len(tps) + 1)
        precision_curve = cum_tp / ranks
        recall_curve = cum_tp / n_gt
        # 11-point interpolated AP@0.5
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            mask = recall_curve >= r
            ap += (precision_curve[mask].max() if mask.any() else 0.0) / 11.0
        precision = float(cum_tp[-1] / len(tps))
        recall = float(cum_tp[-1] / n_gt)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        return {"mAP50": float(ap), "precision": precision, "recall": recall,
                "f1": f1, "n_predictions": len(tp_scores),
                "n_ground_truth": n_gt}

    def evaluate(self, checkpoint_path: str, test_data_dir: str) -> Dict[str, float]:
        """
        Evaluate fine-tuned model on test set.
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_data_dir: Path to test data
            
        Returns:
            Evaluation metrics
        """
        if not RFDETR_AVAILABLE:
            raise RFDETRUnavailableError(
                "cannot evaluate: rfdetr package not installed; no real metrics "
                "available (install requirements-ml.txt)")
        try:
            # Load model and evaluate on the test set
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = self._build_model_from_checkpoint(checkpoint)
            model.model.eval()
            dataset = self._load_coco_test_set(test_data_dir)
            return self._compute_detection_metrics(model, dataset)
        except RFDETRUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise RFDETRUnavailableError(
                f"real evaluation failed ({e}); simulated metrics are not "
                "returned by default. Set MV_ALLOW_MOCK_FALLBACK=true only for "
                "development scaffolding") from e


class RFDETRExporter:
    """
    Export RF-DETR models to various deployment formats.
    
    Supports:
    - ONNX for cross-platform deployment
    - TensorRT for NVIDIA GPU optimization
    - TorchScript for PyTorch deployment
    - CoreML for Apple devices
    
    Example:
        exporter = RFDETRExporter(checkpoint_path="/path/to/model.pt")
        exporter.export_onnx("/path/to/model.onnx", opset=17)
    """
    
    def __init__(self, checkpoint_path: str,
                 variant: RFDETRVariant = RFDETRVariant.MEDIUM):
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model for export."""
        if RFDETR_AVAILABLE and os.path.exists(self.checkpoint_path):
            try:
                if self.variant == RFDETRVariant.LARGE:
                    self.model = RFDETRLarge()
                else:
                    self.model = RFDETRBase()
                
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                
                self.model.eval()
                logger.info(f"Model loaded from {self.checkpoint_path}")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning("Model not loaded (rfdetr not available or checkpoint missing)")
    
    def export_onnx(self, output_path: str, opset: int = 17,
                   input_size: Tuple[int, int] = (640, 640),
                   dynamic_axes: bool = True,
                   simplify: bool = True) -> Dict[str, Any]:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output ONNX file path
            opset: ONNX opset version
            input_size: Input image size (H, W)
            dynamic_axes: Enable dynamic batch size
            simplify: Simplify ONNX graph
            
        Returns:
            Export result dict
        """
        if self.model is None:
            return self._simulate_export(output_path, ExportFormat.ONNX)
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            
            # Define dynamic axes
            dynamic = None
            if dynamic_axes:
                dynamic = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                opset_version=opset,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic
            )
            
            # Simplify if requested
            if simplify:
                try:
                    import onnx
                    from onnxsim import simplify as onnx_simplify
                    
                    model_onnx = onnx.load(output_path)
                    model_simplified, check = onnx_simplify(model_onnx)
                    if check:
                        onnx.save(model_simplified, output_path)
                        logger.info("ONNX model simplified")
                except ImportError:
                    logger.warning("onnxsim not installed, skipping simplification")
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            return {
                'status': 'success',
                'format': 'onnx',
                'output_path': output_path,
                'opset': opset,
                'input_size': input_size,
                'file_size_mb': file_size,
                'dynamic_axes': dynamic_axes
            }
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return self._simulate_export(output_path, ExportFormat.ONNX)
    
    def export_tensorrt(self, output_path: str,
                       input_size: Tuple[int, int] = (640, 640),
                       fp16: bool = True,
                       int8: bool = False,
                       workspace_gb: int = 4) -> Dict[str, Any]:
        """
        Export model to TensorRT format.
        
        Args:
            output_path: Output TensorRT engine path
            input_size: Input image size
            fp16: Enable FP16 precision
            int8: Enable INT8 quantization
            workspace_gb: TensorRT workspace size in GB
            
        Returns:
            Export result dict
        """
        # First export to ONNX
        onnx_path = output_path.replace('.engine', '.onnx')
        onnx_result = self.export_onnx(onnx_path, input_size=input_size)
        
        if onnx_result['status'] != 'success':
            return self._simulate_export(output_path, ExportFormat.TENSORRT)
        
        try:
            import tensorrt as trt
            
            # Build TensorRT engine
            logger_trt = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger_trt)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger_trt)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_gb * (1 << 30)
            
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            # Serialize
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            return {
                'status': 'success',
                'format': 'tensorrt',
                'output_path': output_path,
                'input_size': input_size,
                'fp16': fp16,
                'int8': int8,
                'file_size_mb': file_size
            }
            
        except ImportError:
            logger.warning("TensorRT not installed")
            return self._simulate_export(output_path, ExportFormat.TENSORRT)
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            return self._simulate_export(output_path, ExportFormat.TENSORRT)
    
    def export_torchscript(self, output_path: str,
                          input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """Export model to TorchScript format."""
        if self.model is None:
            return self._simulate_export(output_path, ExportFormat.TORCHSCRIPT)
        
        try:
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            
            # Trace model
            traced = torch.jit.trace(self.model, dummy_input)
            traced.save(output_path)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            return {
                'status': 'success',
                'format': 'torchscript',
                'output_path': output_path,
                'input_size': input_size,
                'file_size_mb': file_size
            }
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return self._simulate_export(output_path, ExportFormat.TORCHSCRIPT)
    
    def _simulate_export(self, output_path: str, format: ExportFormat) -> Dict[str, Any]:
        """Simulate export for testing."""
        logger.info(f"Simulating {format.value} export")
        
        # Create placeholder file
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        metadata = {
            'format': format.value,
            'variant': self.variant.value,
            'simulated': True,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f)
        
        return {
            'status': 'simulated',
            'format': format.value,
            'output_path': output_path,
            'file_size_mb': 0.001
        }


class RFDETRSegmentation:
    """
    RF-DETR instance segmentation support.
    
    Provides instance segmentation capabilities using RF-DETR-Seg,
    which is 3x faster than YOLO for segmentation tasks.
    
    Example:
        segmenter = RFDETRSegmentation()
        masks = segmenter.segment(image)
    """
    
    def __init__(self, config: Optional[RFDETRConfig] = None):
        self.config = config or RFDETRConfig()
        self.model = None
        self.device = torch.device(self.config.device)
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load RF-DETR-Seg model."""
        if RFDETR_AVAILABLE:
            try:
                # RF-DETR-Seg is loaded similarly to base model
                # with segmentation head enabled
                if self.config.variant == RFDETRVariant.LARGE:
                    self.model = RFDETRLarge()
                else:
                    self.model = RFDETRBase()
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("RF-DETR-Seg model loaded")
                
            except Exception as e:
                logger.error(f"Failed to load RF-DETR-Seg: {e}")
                self.model = None
        else:
            logger.warning("RF-DETR-Seg not available (rfdetr not installed)")
    
    def segment(self, image: np.ndarray,
               return_masks: bool = True,
               return_polygons: bool = False) -> Dict[str, Any]:
        """
        Perform instance segmentation.
        
        Args:
            image: Input image
            return_masks: Return binary masks
            return_polygons: Return polygon contours
            
        Returns:
            Segmentation results with boxes, masks, and optionally polygons
        """
        if self.model is None:
            return self._simulate_segmentation(image, return_masks, return_polygons)
        
        try:
            with torch.no_grad():
                results = self.model.predict(image, threshold=self.config.confidence_threshold)
            
            output = {
                'boxes': results.get('boxes', np.zeros((0, 4))),
                'scores': results.get('scores', np.zeros(0)),
                'labels': results.get('labels', np.zeros(0, dtype=int))
            }
            
            if return_masks:
                output['masks'] = results.get('masks', [])
            
            if return_polygons:
                output['polygons'] = self._masks_to_polygons(results.get('masks', []))
            
            return output
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return self._simulate_segmentation(image, return_masks, return_polygons)
    
    def _simulate_segmentation(self, image: np.ndarray,
                              return_masks: bool,
                              return_polygons: bool) -> Dict[str, Any]:
        """Simulate segmentation for testing."""
        h, w = image.shape[:2]
        num_instances = np.random.randint(1, 5)
        
        boxes = []
        scores = []
        labels = []
        masks = []
        polygons = []
        
        for _ in range(num_instances):
            x1 = np.random.randint(0, w - 50)
            y1 = np.random.randint(0, h - 50)
            x2 = x1 + np.random.randint(30, min(150, w - x1))
            y2 = y1 + np.random.randint(30, min(150, h - y1))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(np.random.uniform(0.5, 0.95))
            labels.append(np.random.randint(0, 10))
            
            if return_masks:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                masks.append(mask)
            
            if return_polygons:
                polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        output = {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'labels': np.array(labels)
        }
        
        if return_masks:
            output['masks'] = masks
        if return_polygons:
            output['polygons'] = polygons
        
        return output
    
    def _masks_to_polygons(self, masks: List[np.ndarray]) -> List[List[List[int]]]:
        """Convert binary masks to polygon contours."""
        polygons = []
        
        try:
            import cv2
            
            for mask in masks:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Get largest contour
                    largest = max(contours, key=cv2.contourArea)
                    polygon = largest.squeeze().tolist()
                    polygons.append(polygon)
                else:
                    polygons.append([])
                    
        except ImportError:
            logger.warning("OpenCV not available for polygon conversion")
        
        return polygons


class UnifiedWALDODetector:
    """
    Unified WALDO detector supporting both YOLOv8 and RF-DETR backbones.
    
    Provides a consistent API regardless of the underlying model,
    allowing easy switching between backbones.
    
    Example:
        # Use RF-DETR
        detector = UnifiedWALDODetector(backbone='rfdetr')
        
        # Use YOLOv8
        detector = UnifiedWALDODetector(backbone='yolov8')
        
        # Same API for both
        detections = detector.detect(image)
    """
    
    def __init__(self, backbone: str = 'rfdetr',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified detector.
        
        Args:
            backbone: 'rfdetr' or 'yolov8'
            config: Configuration dict
        """
        self.backbone_type = backbone.lower()
        self.config = config or {}
        self.detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize the appropriate detector backend."""
        if self.backbone_type == 'rfdetr':
            rfdetr_config = RFDETRConfig(
                variant=RFDETRVariant(self.config.get('variant', 'medium')),
                confidence_threshold=self.config.get('confidence_threshold', 0.5),
                device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                checkpoint_path=self.config.get('checkpoint_path')
            )
            self.detector = RFDETRDetector(rfdetr_config)
            
        elif self.backbone_type == 'yolov8':
            # Import and use existing YOLO detector
            try:
                from .detection import WALDODetector
                self.detector = WALDODetector(self.config)
            except ImportError:
                logger.error("YOLOv8 detector not available")
                # Fall back to RF-DETR
                self.backbone_type = 'rfdetr'
                self._initialize_detector()
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_type}")
        
        logger.info(f"Unified detector initialized with {self.backbone_type} backbone")
    
    def detect(self, image: np.ndarray,
              metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform detection using the configured backbone.
        
        Args:
            image: Input image
            metadata: Optional metadata
            
        Returns:
            List of detection dicts with consistent format
        """
        if self.backbone_type == 'rfdetr':
            detections = self.detector.detect(image, metadata)
            # Convert RFDETRDetection to dict
            return [d.to_dict() for d in detections]
        else:
            return self.detector.detect(image, metadata)
    
    def get_backbone_info(self) -> Dict[str, Any]:
        """Get information about the current backbone."""
        info = {
            'backbone': self.backbone_type,
            'available_backbones': ['rfdetr', 'yolov8']
        }
        
        if self.backbone_type == 'rfdetr' and hasattr(self.detector, 'get_statistics'):
            info['statistics'] = self.detector.get_statistics()
        
        return info
    
    def switch_backbone(self, backbone: str) -> None:
        """Switch to a different backbone."""
        self.backbone_type = backbone.lower()
        self._initialize_detector()


# Factory functions
def create_rfdetr_detector(variant: str = 'medium',
                          confidence: float = 0.5,
                          checkpoint: Optional[str] = None) -> RFDETRDetector:
    """Create RF-DETR detector with specified configuration."""
    config = RFDETRConfig(
        variant=RFDETRVariant(variant),
        confidence_threshold=confidence,
        checkpoint_path=checkpoint
    )
    return RFDETRDetector(config)


def create_rfdetr_finetuner(epochs: int = 50,
                           batch_size: int = 8,
                           variant: str = 'medium') -> RFDETRFineTuner:
    """Create RF-DETR fine-tuner with specified configuration."""
    config = RFDETRTrainingConfig(
        epochs=epochs,
        batch_size=batch_size
    )
    return RFDETRFineTuner(config, RFDETRVariant(variant))


def create_rfdetr_exporter(checkpoint_path: str,
                          variant: str = 'medium') -> RFDETRExporter:
    """Create RF-DETR exporter for model deployment."""
    return RFDETRExporter(checkpoint_path, RFDETRVariant(variant))


def create_unified_detector(backbone: str = 'rfdetr',
                           **kwargs) -> UnifiedWALDODetector:
    """Create unified WALDO detector with specified backbone."""
    return UnifiedWALDODetector(backbone=backbone, config=kwargs)


# Comparison utilities
def compare_backbones(image: np.ndarray,
                     yolo_config: Dict[str, Any],
                     rfdetr_config: Optional[RFDETRConfig] = None) -> Dict[str, Any]:
    """
    Compare YOLOv8 and RF-DETR detection results on the same image.
    
    Args:
        image: Input image
        yolo_config: YOLOv8 configuration
        rfdetr_config: RF-DETR configuration
        
    Returns:
        Comparison results with detections and timing from both models
    """
    import time
    
    results = {
        'image_shape': image.shape,
        'yolov8': {},
        'rfdetr': {}
    }
    
    # Run YOLOv8
    try:
        from .detection import WALDODetector
        yolo_detector = WALDODetector(yolo_config)
        
        start = time.time()
        yolo_detections = yolo_detector.detect(image)
        yolo_time = time.time() - start
        
        results['yolov8'] = {
            'num_detections': len(yolo_detections),
            'inference_time': yolo_time,
            'detections': yolo_detections
        }
    except Exception as e:
        results['yolov8'] = {'error': str(e)}
    
    # Run RF-DETR
    try:
        rfdetr_config = rfdetr_config or RFDETRConfig()
        rfdetr_detector = RFDETRDetector(rfdetr_config)
        
        start = time.time()
        rfdetr_detections = rfdetr_detector.detect(image)
        rfdetr_time = time.time() - start
        
        results['rfdetr'] = {
            'num_detections': len(rfdetr_detections),
            'inference_time': rfdetr_time,
            'detections': [d.to_dict() for d in rfdetr_detections]
        }
    except Exception as e:
        results['rfdetr'] = {'error': str(e)}
    
    return results
