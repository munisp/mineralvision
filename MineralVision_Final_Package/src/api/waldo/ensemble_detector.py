"""
YOLO11 + RF-DETR Ensemble Detector for MineralVision.

Combines the strengths of YOLO11 (speed, good at common objects) and
RF-DETR (accuracy, better at small/occluded objects) for superior detection.

Ensemble Strategies:
- Weighted Box Fusion (WBF): Best for balanced accuracy
- Soft-NMS: Good for recall-focused applications
- Consensus: High precision, requires both models to agree
- Union: Maximum recall, keeps all detections

Usage:
    from api.waldo.ensemble_detector import (
        EnsembleWALDODetector,
        create_ensemble_detector,
    )
    
    # Create ensemble
    ensemble = create_ensemble_detector(
        yolo_model="yolo11m.pt",
        rfdetr_variant="medium",
        fusion_strategy="wbf",
    )
    
    # Run detection
    detections = ensemble.detect(image)

Based on:
- YOLO11: https://docs.ultralytics.com/models/yolo11/
- RF-DETR: https://github.com/roboflow/rf-detr
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Box fusion strategies for ensemble."""
    WBF = "wbf"                    # Weighted Box Fusion (balanced)
    SOFT_NMS = "soft_nms"          # Soft Non-Maximum Suppression (recall)
    NMS = "nms"                    # Standard NMS (fast)
    CONSENSUS = "consensus"        # Both models must agree (precision)
    UNION = "union"                # Keep all detections (max recall)
    CONDITIONAL = "conditional"    # YOLO first, RF-DETR on uncertain


class EnsembleMode(Enum):
    """Ensemble operation modes."""
    PARALLEL = "parallel"          # Run both models simultaneously
    SEQUENTIAL = "sequential"      # Run models one after another
    CONDITIONAL = "conditional"    # Run RF-DETR only when needed


@dataclass
class DetectionResult:
    """Unified detection result from ensemble."""
    detection_id: str
    bbox: List[float]              # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str
    source: str                    # "yolo11", "rfdetr", "ensemble"
    contributing_models: List[str] = field(default_factory=list)
    original_scores: Dict[str, float] = field(default_factory=dict)
    iou_with_match: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "source": self.source,
            "contributing_models": self.contributing_models,
            "original_scores": self.original_scores,
        }


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detector."""
    # Model configs
    yolo_model: str = "yolo11m.pt"
    yolo_confidence: float = 0.25
    rfdetr_variant: str = "medium"
    rfdetr_confidence: float = 0.5
    
    # Fusion config
    fusion_strategy: FusionStrategy = FusionStrategy.WBF
    ensemble_mode: EnsembleMode = EnsembleMode.PARALLEL
    
    # WBF parameters
    wbf_iou_threshold: float = 0.55
    wbf_skip_box_threshold: float = 0.0
    wbf_conf_type: str = "avg"     # "avg", "max", "box_and_model_avg"
    
    # NMS parameters
    nms_iou_threshold: float = 0.45
    soft_nms_sigma: float = 0.5
    
    # Model weights (for weighted fusion)
    yolo_weight: float = 1.0
    rfdetr_weight: float = 1.0
    
    # Consensus parameters
    consensus_iou_threshold: float = 0.5
    consensus_min_models: int = 2
    
    # Conditional parameters
    conditional_uncertainty_threshold: float = 0.6
    conditional_max_detections: int = 50
    
    # Calibration (score adjustment)
    yolo_score_gamma: float = 1.0   # score^gamma
    yolo_score_alpha: float = 1.0   # score * alpha
    rfdetr_score_gamma: float = 1.0
    rfdetr_score_alpha: float = 1.0
    
    # Hardware
    device: str = "auto"
    parallel_workers: int = 2
    
    # Class mapping
    class_names: Optional[List[str]] = None


class BoxFusion:
    """Box fusion algorithms for ensemble detection."""
    
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def weighted_box_fusion(
        boxes_list: List[List[List[float]]],
        scores_list: List[List[float]],
        labels_list: List[List[int]],
        weights: Optional[List[float]] = None,
        iou_threshold: float = 0.55,
        skip_box_threshold: float = 0.0,
        conf_type: str = "avg",
    ) -> Tuple[List[List[float]], List[float], List[int], List[List[int]]]:
        """
        Weighted Box Fusion algorithm.
        
        Args:
            boxes_list: List of box lists from each model [[x1,y1,x2,y2], ...]
            scores_list: List of score lists from each model
            labels_list: List of label lists from each model
            weights: Model weights for fusion
            iou_threshold: IoU threshold for matching boxes
            skip_box_threshold: Skip boxes below this confidence
            conf_type: How to combine confidences ("avg", "max", "box_and_model_avg")
            
        Returns:
            Tuple of (fused_boxes, fused_scores, fused_labels, contributing_models)
        """
        if weights is None:
            weights = [1.0] * len(boxes_list)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Collect all boxes with model index
        all_boxes = []
        for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
            for box, score, label in zip(boxes, scores, labels):
                if score >= skip_box_threshold:
                    all_boxes.append({
                        "box": box,
                        "score": score * weights[model_idx],
                        "original_score": score,
                        "label": label,
                        "model_idx": model_idx,
                        "weight": weights[model_idx],
                    })
        
        if not all_boxes:
            return [], [], [], []
        
        # Sort by score descending
        all_boxes.sort(key=lambda x: x["score"], reverse=True)
        
        # Group by class
        class_boxes = {}
        for box_info in all_boxes:
            label = box_info["label"]
            if label not in class_boxes:
                class_boxes[label] = []
            class_boxes[label].append(box_info)
        
        # Fuse boxes per class
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        contributing_models = []
        
        for label, boxes in class_boxes.items():
            clusters = []
            
            for box_info in boxes:
                matched = False
                
                for cluster in clusters:
                    # Check IoU with cluster representative
                    cluster_box = cluster["fused_box"]
                    iou = BoxFusion.compute_iou(box_info["box"], cluster_box)
                    
                    if iou >= iou_threshold:
                        cluster["members"].append(box_info)
                        matched = True
                        break
                
                if not matched:
                    clusters.append({
                        "fused_box": box_info["box"].copy(),
                        "members": [box_info],
                    })
            
            # Compute fused boxes for each cluster
            for cluster in clusters:
                members = cluster["members"]
                
                # Weighted average of box coordinates
                total_weight = sum(m["weight"] * m["original_score"] for m in members)
                if total_weight > 0:
                    fused_box = [0.0, 0.0, 0.0, 0.0]
                    for m in members:
                        w = m["weight"] * m["original_score"] / total_weight
                        for i in range(4):
                            fused_box[i] += m["box"][i] * w
                else:
                    fused_box = members[0]["box"]
                
                # Compute fused confidence
                if conf_type == "max":
                    fused_score = max(m["original_score"] for m in members)
                elif conf_type == "box_and_model_avg":
                    # Average across models, weighted by number of boxes
                    model_scores = {}
                    for m in members:
                        idx = m["model_idx"]
                        if idx not in model_scores:
                            model_scores[idx] = []
                        model_scores[idx].append(m["original_score"])
                    fused_score = np.mean([np.mean(s) for s in model_scores.values()])
                else:  # avg
                    fused_score = np.mean([m["original_score"] for m in members])
                
                # Boost score if multiple models agree
                n_models = len(set(m["model_idx"] for m in members))
                if n_models > 1:
                    fused_score = min(1.0, fused_score * (1 + 0.1 * (n_models - 1)))
                
                fused_boxes.append(fused_box)
                fused_scores.append(float(fused_score))
                fused_labels.append(label)
                contributing_models.append(list(set(m["model_idx"] for m in members)))
        
        return fused_boxes, fused_scores, fused_labels, contributing_models
    
    @staticmethod
    def soft_nms(
        boxes: List[List[float]],
        scores: List[float],
        labels: List[int],
        model_indices: List[int],
        iou_threshold: float = 0.45,
        sigma: float = 0.5,
        score_threshold: float = 0.001,
    ) -> Tuple[List[List[float]], List[float], List[int], List[List[int]]]:
        """
        Soft Non-Maximum Suppression.
        
        Instead of removing overlapping boxes, reduces their scores based on IoU.
        """
        if not boxes:
            return [], [], [], []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        model_indices = np.array(model_indices)
        
        # Process per class
        unique_labels = np.unique(labels)
        
        final_boxes = []
        final_scores = []
        final_labels = []
        final_models = []
        
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask].copy()
            class_models = model_indices[mask]
            
            indices = list(range(len(class_boxes)))
            
            while indices:
                # Find max score
                max_idx = max(indices, key=lambda i: class_scores[i])
                
                if class_scores[max_idx] < score_threshold:
                    break
                
                final_boxes.append(class_boxes[max_idx].tolist())
                final_scores.append(float(class_scores[max_idx]))
                final_labels.append(int(label))
                final_models.append([int(class_models[max_idx])])
                
                indices.remove(max_idx)
                
                # Decay scores of overlapping boxes
                for i in indices:
                    iou = BoxFusion.compute_iou(
                        class_boxes[max_idx].tolist(),
                        class_boxes[i].tolist()
                    )
                    if iou > 0:
                        # Gaussian decay
                        class_scores[i] *= np.exp(-(iou ** 2) / sigma)
        
        return final_boxes, final_scores, final_labels, final_models
    
    @staticmethod
    def standard_nms(
        boxes: List[List[float]],
        scores: List[float],
        labels: List[int],
        model_indices: List[int],
        iou_threshold: float = 0.45,
    ) -> Tuple[List[List[float]], List[float], List[int], List[List[int]]]:
        """Standard Non-Maximum Suppression."""
        if not boxes:
            return [], [], [], []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        model_indices = np.array(model_indices)
        
        unique_labels = np.unique(labels)
        
        final_boxes = []
        final_scores = []
        final_labels = []
        final_models = []
        
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            class_models = model_indices[mask]
            
            # Sort by score
            order = np.argsort(-class_scores)
            
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                
                if len(order) == 1:
                    break
                
                # Compute IoU with remaining boxes
                ious = np.array([
                    BoxFusion.compute_iou(
                        class_boxes[i].tolist(),
                        class_boxes[j].tolist()
                    )
                    for j in order[1:]
                ])
                
                # Keep boxes with IoU below threshold
                order = order[1:][ious < iou_threshold]
            
            for i in keep:
                final_boxes.append(class_boxes[i].tolist())
                final_scores.append(float(class_scores[i]))
                final_labels.append(int(label))
                final_models.append([int(class_models[i])])
        
        return final_boxes, final_scores, final_labels, final_models
    
    @staticmethod
    def consensus_fusion(
        boxes_list: List[List[List[float]]],
        scores_list: List[List[float]],
        labels_list: List[List[int]],
        iou_threshold: float = 0.5,
        min_models: int = 2,
    ) -> Tuple[List[List[float]], List[float], List[int], List[List[int]]]:
        """
        Consensus fusion - only keep detections where multiple models agree.
        """
        if len(boxes_list) < min_models:
            return [], [], [], []
        
        # Use first model as reference
        ref_boxes = boxes_list[0]
        ref_scores = scores_list[0]
        ref_labels = labels_list[0]
        
        final_boxes = []
        final_scores = []
        final_labels = []
        final_models = []
        
        for i, (box, score, label) in enumerate(zip(ref_boxes, ref_scores, ref_labels)):
            agreeing_models = [0]
            agreeing_scores = [score]
            agreeing_boxes = [box]
            
            # Check other models
            for model_idx in range(1, len(boxes_list)):
                for j, (other_box, other_score, other_label) in enumerate(
                    zip(boxes_list[model_idx], scores_list[model_idx], labels_list[model_idx])
                ):
                    if other_label == label:
                        iou = BoxFusion.compute_iou(box, other_box)
                        if iou >= iou_threshold:
                            agreeing_models.append(model_idx)
                            agreeing_scores.append(other_score)
                            agreeing_boxes.append(other_box)
                            break
            
            if len(agreeing_models) >= min_models:
                # Average the boxes and scores
                avg_box = np.mean(agreeing_boxes, axis=0).tolist()
                avg_score = float(np.mean(agreeing_scores))
                
                final_boxes.append(avg_box)
                final_scores.append(avg_score)
                final_labels.append(label)
                final_models.append(agreeing_models)
        
        return final_boxes, final_scores, final_labels, final_models


class EnsembleWALDODetector:
    """
    Ensemble detector combining YOLO11 and RF-DETR.
    
    Leverages complementary strengths:
    - YOLO11: Fast, good at common objects, efficient
    - RF-DETR: Accurate, better at small/occluded objects, transformer-based
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.yolo_detector = None
        self.rfdetr_detector = None
        self.device = self._get_device()
        
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._yolo_time = 0.0
        self._rfdetr_time = 0.0
        
        self._initialize_detectors()
    
    def _get_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _initialize_detectors(self) -> None:
        """Initialize both detectors."""
        # Initialize YOLO11
        try:
            from ultralytics import YOLO
            self.yolo_detector = YOLO(self.config.yolo_model)
            logger.info(f"YOLO11 loaded: {self.config.yolo_model}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO11: {e}")
            self.yolo_detector = None
        
        # Initialize RF-DETR
        try:
            # Import from existing rfdetr_backbone module
            import sys
            import os as _os
            _waldo_src = _os.getenv(
                "WALDO_PACKAGE_SRC",
                _os.path.normpath(_os.path.join(
                    _os.path.dirname(__file__), "..", "..", "..", "..",
                    "MineralVision_WALDO_Production_Package", "src")),
            )
            if _waldo_src not in sys.path:
                sys.path.insert(0, _waldo_src)
            from waldo_integration.rfdetr_backbone import RFDETRDetector, RFDETRConfig, RFDETRVariant
            
            rfdetr_config = RFDETRConfig(
                variant=RFDETRVariant(self.config.rfdetr_variant),
                confidence_threshold=self.config.rfdetr_confidence,
                device=self.device,
            )
            self.rfdetr_detector = RFDETRDetector(rfdetr_config)
            logger.info(f"RF-DETR loaded: {self.config.rfdetr_variant}")
        except Exception as e:
            logger.warning(f"Failed to load RF-DETR: {e}")
            self.rfdetr_detector = None
    
    def _calibrate_score(
        self,
        score: float,
        gamma: float,
        alpha: float,
    ) -> float:
        """Apply score calibration."""
        return min(1.0, max(0.0, (score ** gamma) * alpha))
    
    def _run_yolo(self, image: np.ndarray) -> Tuple[List[List[float]], List[float], List[int]]:
        """Run YOLO11 inference."""
        if self.yolo_detector is None:
            return [], [], []
        
        start = time.time()
        results = self.yolo_detector(image, conf=self.config.yolo_confidence)
        self._yolo_time += time.time() - start
        
        boxes = []
        scores = []
        labels = []
        
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()
                score = float(box.conf[0])
                label = int(box.cls[0])
                
                # Apply calibration
                score = self._calibrate_score(
                    score,
                    self.config.yolo_score_gamma,
                    self.config.yolo_score_alpha,
                )
                
                boxes.append(bbox)
                scores.append(score)
                labels.append(label)
        
        return boxes, scores, labels
    
    def _run_rfdetr(self, image: np.ndarray) -> Tuple[List[List[float]], List[float], List[int]]:
        """Run RF-DETR inference."""
        if self.rfdetr_detector is None:
            return [], [], []
        
        start = time.time()
        detections = self.rfdetr_detector.detect(image)
        self._rfdetr_time += time.time() - start
        
        boxes = []
        scores = []
        labels = []
        
        for det in detections:
            score = self._calibrate_score(
                det.confidence,
                self.config.rfdetr_score_gamma,
                self.config.rfdetr_score_alpha,
            )
            
            boxes.append(det.bbox)
            scores.append(score)
            labels.append(det.class_id)
        
        return boxes, scores, labels
    
    def _should_run_rfdetr(
        self,
        yolo_boxes: List[List[float]],
        yolo_scores: List[float],
    ) -> bool:
        """Determine if RF-DETR should run (for conditional mode)."""
        if not yolo_boxes:
            return True
        
        # Run RF-DETR if:
        # 1. Many low-confidence detections
        low_conf_count = sum(1 for s in yolo_scores if s < self.config.conditional_uncertainty_threshold)
        if low_conf_count > len(yolo_scores) * 0.3:
            return True
        
        # 2. Too many detections (crowded scene)
        if len(yolo_boxes) > self.config.conditional_max_detections:
            return True
        
        # 3. Average confidence is low
        avg_conf = np.mean(yolo_scores) if yolo_scores else 0
        if avg_conf < self.config.conditional_uncertainty_threshold:
            return True
        
        return False
    
    def detect(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DetectionResult]:
        """
        Run ensemble detection on an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            metadata: Optional metadata to attach to detections
            
        Returns:
            List of DetectionResult objects
        """
        start_time = time.time()
        
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")
        
        # Run detectors based on mode
        if self.config.ensemble_mode == EnsembleMode.PARALLEL:
            yolo_results, rfdetr_results = self._run_parallel(image)
        elif self.config.ensemble_mode == EnsembleMode.CONDITIONAL:
            yolo_results, rfdetr_results = self._run_conditional(image)
        else:  # SEQUENTIAL
            yolo_results, rfdetr_results = self._run_sequential(image)
        
        # Fuse results
        detections = self._fuse_results(yolo_results, rfdetr_results, metadata)
        
        # Update statistics
        self._inference_count += 1
        self._total_inference_time += time.time() - start_time
        
        return detections
    
    def _run_parallel(
        self,
        image: np.ndarray,
    ) -> Tuple[Tuple, Tuple]:
        """Run both detectors in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            yolo_future = executor.submit(self._run_yolo, image)
            rfdetr_future = executor.submit(self._run_rfdetr, image)
            
            yolo_results = yolo_future.result()
            rfdetr_results = rfdetr_future.result()
        
        return yolo_results, rfdetr_results
    
    def _run_sequential(
        self,
        image: np.ndarray,
    ) -> Tuple[Tuple, Tuple]:
        """Run detectors sequentially."""
        yolo_results = self._run_yolo(image)
        rfdetr_results = self._run_rfdetr(image)
        return yolo_results, rfdetr_results
    
    def _run_conditional(
        self,
        image: np.ndarray,
    ) -> Tuple[Tuple, Tuple]:
        """Run YOLO first, RF-DETR only if needed."""
        yolo_results = self._run_yolo(image)
        
        if self._should_run_rfdetr(yolo_results[0], yolo_results[1]):
            rfdetr_results = self._run_rfdetr(image)
        else:
            rfdetr_results = ([], [], [])
        
        return yolo_results, rfdetr_results
    
    def _fuse_results(
        self,
        yolo_results: Tuple[List, List, List],
        rfdetr_results: Tuple[List, List, List],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DetectionResult]:
        """Fuse results from both detectors."""
        yolo_boxes, yolo_scores, yolo_labels = yolo_results
        rfdetr_boxes, rfdetr_scores, rfdetr_labels = rfdetr_results
        
        # Apply fusion strategy
        strategy = self.config.fusion_strategy
        
        if strategy == FusionStrategy.WBF:
            fused_boxes, fused_scores, fused_labels, contributing = BoxFusion.weighted_box_fusion(
                [yolo_boxes, rfdetr_boxes],
                [yolo_scores, rfdetr_scores],
                [yolo_labels, rfdetr_labels],
                weights=[self.config.yolo_weight, self.config.rfdetr_weight],
                iou_threshold=self.config.wbf_iou_threshold,
                skip_box_threshold=self.config.wbf_skip_box_threshold,
                conf_type=self.config.wbf_conf_type,
            )
        
        elif strategy == FusionStrategy.SOFT_NMS:
            # Combine all boxes
            all_boxes = yolo_boxes + rfdetr_boxes
            all_scores = yolo_scores + rfdetr_scores
            all_labels = yolo_labels + rfdetr_labels
            model_indices = [0] * len(yolo_boxes) + [1] * len(rfdetr_boxes)
            
            fused_boxes, fused_scores, fused_labels, contributing = BoxFusion.soft_nms(
                all_boxes, all_scores, all_labels, model_indices,
                iou_threshold=self.config.nms_iou_threshold,
                sigma=self.config.soft_nms_sigma,
            )
        
        elif strategy == FusionStrategy.NMS:
            all_boxes = yolo_boxes + rfdetr_boxes
            all_scores = yolo_scores + rfdetr_scores
            all_labels = yolo_labels + rfdetr_labels
            model_indices = [0] * len(yolo_boxes) + [1] * len(rfdetr_boxes)
            
            fused_boxes, fused_scores, fused_labels, contributing = BoxFusion.standard_nms(
                all_boxes, all_scores, all_labels, model_indices,
                iou_threshold=self.config.nms_iou_threshold,
            )
        
        elif strategy == FusionStrategy.CONSENSUS:
            fused_boxes, fused_scores, fused_labels, contributing = BoxFusion.consensus_fusion(
                [yolo_boxes, rfdetr_boxes],
                [yolo_scores, rfdetr_scores],
                [yolo_labels, rfdetr_labels],
                iou_threshold=self.config.consensus_iou_threshold,
                min_models=self.config.consensus_min_models,
            )
        
        else:  # UNION
            fused_boxes = yolo_boxes + rfdetr_boxes
            fused_scores = yolo_scores + rfdetr_scores
            fused_labels = yolo_labels + rfdetr_labels
            contributing = [[0]] * len(yolo_boxes) + [[1]] * len(rfdetr_boxes)
        
        # Convert to DetectionResult objects
        model_names = ["yolo11", "rfdetr"]
        detections = []
        
        for i, (box, score, label, models) in enumerate(
            zip(fused_boxes, fused_scores, fused_labels, contributing)
        ):
            # Get class name
            if self.config.class_names and label < len(self.config.class_names):
                class_name = self.config.class_names[label]
            else:
                class_name = f"class_{label}"
            
            # Determine source
            if len(models) > 1:
                source = "ensemble"
            else:
                source = model_names[models[0]]
            
            detection = DetectionResult(
                detection_id=str(uuid.uuid4())[:8],
                bbox=box,
                confidence=score,
                class_id=label,
                class_name=class_name,
                source=source,
                contributing_models=[model_names[m] for m in models],
                metadata=metadata or {},
            )
            
            detections.append(detection)
        
        return detections
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        avg_time = (self._total_inference_time / self._inference_count
                   if self._inference_count > 0 else 0)
        
        return {
            "fusion_strategy": self.config.fusion_strategy.value,
            "ensemble_mode": self.config.ensemble_mode.value,
            "inference_count": self._inference_count,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "yolo_time": self._yolo_time,
            "rfdetr_time": self._rfdetr_time,
            "yolo_available": self.yolo_detector is not None,
            "rfdetr_available": self.rfdetr_detector is not None,
        }
    
    def benchmark(
        self,
        image: np.ndarray,
        n_runs: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark ensemble performance."""
        # Warmup
        for _ in range(10):
            self.detect(image)
        
        # Reset stats
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._yolo_time = 0.0
        self._rfdetr_time = 0.0
        
        # Benchmark
        for _ in range(n_runs):
            self.detect(image)
        
        stats = self.get_statistics()
        stats["n_runs"] = n_runs
        
        return stats


# Factory functions

def create_ensemble_detector(
    yolo_model: str = "yolo11m.pt",
    rfdetr_variant: str = "medium",
    fusion_strategy: str = "wbf",
    ensemble_mode: str = "parallel",
    yolo_weight: float = 1.0,
    rfdetr_weight: float = 1.0,
    class_names: Optional[List[str]] = None,
) -> EnsembleWALDODetector:
    """
    Create an ensemble detector combining YOLO11 and RF-DETR.
    
    Args:
        yolo_model: YOLO11 model path or variant
        rfdetr_variant: RF-DETR variant (nano, small, medium, large)
        fusion_strategy: Box fusion strategy (wbf, soft_nms, nms, consensus, union)
        ensemble_mode: Execution mode (parallel, sequential, conditional)
        yolo_weight: Weight for YOLO11 in fusion
        rfdetr_weight: Weight for RF-DETR in fusion
        class_names: List of class names
        
    Returns:
        Configured EnsembleWALDODetector
    """
    config = EnsembleConfig(
        yolo_model=yolo_model,
        rfdetr_variant=rfdetr_variant,
        fusion_strategy=FusionStrategy(fusion_strategy),
        ensemble_mode=EnsembleMode(ensemble_mode),
        yolo_weight=yolo_weight,
        rfdetr_weight=rfdetr_weight,
        class_names=class_names,
    )
    
    return EnsembleWALDODetector(config)


def create_precision_ensemble(
    yolo_model: str = "yolo11m.pt",
    rfdetr_variant: str = "medium",
) -> EnsembleWALDODetector:
    """Create a precision-focused ensemble (consensus mode)."""
    config = EnsembleConfig(
        yolo_model=yolo_model,
        rfdetr_variant=rfdetr_variant,
        fusion_strategy=FusionStrategy.CONSENSUS,
        ensemble_mode=EnsembleMode.PARALLEL,
        consensus_min_models=2,
    )
    return EnsembleWALDODetector(config)


def create_recall_ensemble(
    yolo_model: str = "yolo11m.pt",
    rfdetr_variant: str = "medium",
) -> EnsembleWALDODetector:
    """Create a recall-focused ensemble (union + soft-nms)."""
    config = EnsembleConfig(
        yolo_model=yolo_model,
        rfdetr_variant=rfdetr_variant,
        fusion_strategy=FusionStrategy.SOFT_NMS,
        ensemble_mode=EnsembleMode.PARALLEL,
        soft_nms_sigma=0.5,
    )
    return EnsembleWALDODetector(config)


def create_fast_ensemble(
    yolo_model: str = "yolo11n.pt",
    rfdetr_variant: str = "small",
) -> EnsembleWALDODetector:
    """Create a speed-optimized ensemble (conditional mode)."""
    config = EnsembleConfig(
        yolo_model=yolo_model,
        rfdetr_variant=rfdetr_variant,
        fusion_strategy=FusionStrategy.WBF,
        ensemble_mode=EnsembleMode.CONDITIONAL,
        conditional_uncertainty_threshold=0.7,
    )
    return EnsembleWALDODetector(config)
