"""
Mining-Specific Evaluation Metrics for Molmo2.

Provides comprehensive evaluation metrics tailored to
mining/geological exploration applications.

Metrics:
- Object detection: mAP, site-level F1, false positive rate
- Tracking: MOTA, MOTP, ID switches
- Change detection: AUC, per-change-type F1
- Geological features: per-class F1, IoU
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
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75])
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    site_level_threshold: float = 0.5  # Threshold for site-level detection
    change_magnitude_bins: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 1.0])
    track_match_threshold: float = 0.5
    
    # Mining-specific settings
    artisanal_mining_classes: List[str] = field(default_factory=lambda: [
        "excavation", "panning", "sluicing", "processing", "camp", "equipment"
    ])
    geological_classes: List[str] = field(default_factory=lambda: [
        "outcrop", "fault", "fold", "vein", "alteration_zone", "gossan", "contact"
    ])
    environmental_classes: List[str] = field(default_factory=lambda: [
        "vegetation_loss", "water_pollution", "erosion", "tailings"
    ])


# =============================================================================
# GROUND TRUTH DATA STRUCTURES
# =============================================================================

@dataclass
class GroundTruthBox:
    """Ground truth bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    is_difficult: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class GroundTruthFrame:
    """Ground truth for a single frame."""
    frame_id: str
    boxes: List[GroundTruthBox]
    scene_type: Optional[str] = None
    is_artisanal_mining: Optional[bool] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthVideo:
    """Ground truth for a video."""
    video_id: str
    frames: List[GroundTruthFrame]
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthSite:
    """Ground truth for a mining site."""
    site_id: str
    has_artisanal_mining: bool
    mining_type: Optional[str] = None
    severity: Optional[str] = None
    environmental_impact: Optional[str] = None
    images: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthChange:
    """Ground truth for change detection."""
    change_id: str
    before_image: str
    after_image: str
    has_change: bool
    change_type: Optional[str] = None
    change_magnitude: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PREDICTION DATA STRUCTURES
# =============================================================================

@dataclass
class PredictionBox:
    """Predicted bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class PredictionFrame:
    """Predictions for a single frame."""
    frame_id: str
    boxes: List[PredictionBox]
    scene_type: Optional[str] = None
    scene_confidence: Optional[float] = None
    is_artisanal_mining: Optional[bool] = None
    mining_confidence: Optional[float] = None


@dataclass
class PredictionSite:
    """Site-level prediction."""
    site_id: str
    is_artisanal_mining: bool
    confidence: float
    mining_type: Optional[str] = None
    environmental_impact: Optional[str] = None


@dataclass
class PredictionChange:
    """Change detection prediction."""
    change_id: str
    has_change: bool
    confidence: float
    change_type: Optional[str] = None
    change_magnitude: float = 0.0


# =============================================================================
# CORE METRICS
# =============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
) -> np.ndarray:
    """Compute IoU matrix between predictions and ground truth."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt))
    
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    return iou_matrix


# =============================================================================
# OBJECT DETECTION METRICS
# =============================================================================

@dataclass
class DetectionMetrics:
    """Object detection evaluation metrics."""
    precision: float
    recall: float
    f1: float
    ap: float  # Average Precision
    map: float  # Mean Average Precision
    per_class_ap: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ap": self.ap,
            "map": self.map,
            "per_class_ap": self.per_class_ap,
            "per_class_f1": self.per_class_f1,
        }


class DetectionEvaluator:
    """Evaluator for object detection."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        predictions: List[PredictionFrame],
        ground_truth: List[GroundTruthFrame],
    ) -> DetectionMetrics:
        """Evaluate detection predictions."""
        # Match predictions to ground truth
        all_classes = set()
        for gt in ground_truth:
            for box in gt.boxes:
                all_classes.add(box.class_name)
        
        # Per-class evaluation
        per_class_ap = {}
        per_class_f1 = {}
        
        for class_name in all_classes:
            ap, f1 = self._evaluate_class(predictions, ground_truth, class_name)
            per_class_ap[class_name] = ap
            per_class_f1[class_name] = f1
        
        # Aggregate metrics
        map_score = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        avg_f1 = np.mean(list(per_class_f1.values())) if per_class_f1 else 0.0
        
        # Overall precision/recall
        precision, recall = self._compute_overall_pr(predictions, ground_truth)
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
            ap=map_score,
            map=map_score,
            per_class_ap=per_class_ap,
            per_class_f1=per_class_f1,
        )
    
    def _evaluate_class(
        self,
        predictions: List[PredictionFrame],
        ground_truth: List[GroundTruthFrame],
        class_name: str,
    ) -> Tuple[float, float]:
        """Evaluate single class."""
        # Collect all predictions and GT for this class
        all_preds = []
        all_gt = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = [b for b in pred.boxes if b.class_name == class_name]
            gt_boxes = [b for b in gt.boxes if b.class_name == class_name]
            
            all_preds.append(pred_boxes)
            all_gt.append(gt_boxes)
        
        # Compute AP
        ap = self._compute_ap(all_preds, all_gt)
        
        # Compute F1 at default threshold
        tp, fp, fn = self._count_matches(all_preds, all_gt, self.config.iou_thresholds[0])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ap, f1
    
    def _compute_ap(
        self,
        predictions: List[List[PredictionBox]],
        ground_truth: List[List[GroundTruthBox]],
    ) -> float:
        """Compute Average Precision."""
        # Flatten and sort by confidence
        all_preds = []
        for frame_idx, preds in enumerate(predictions):
            for pred in preds:
                all_preds.append((frame_idx, pred))
        
        all_preds.sort(key=lambda x: x[1].confidence, reverse=True)
        
        # Track which GT boxes have been matched
        gt_matched = [[False] * len(gt) for gt in ground_truth]
        
        # Compute precision-recall curve
        tp = 0
        fp = 0
        total_gt = sum(len(gt) for gt in ground_truth)
        
        precisions = []
        recalls = []
        
        for frame_idx, pred in all_preds:
            gt_boxes = ground_truth[frame_idx]
            
            # Find best matching GT box
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[frame_idx][gt_idx]:
                    continue
                
                iou = compute_iou(pred.to_array(), gt.to_array())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.config.iou_thresholds[0] and best_gt_idx >= 0:
                tp += 1
                gt_matched[frame_idx][best_gt_idx] = True
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / total_gt if total_gt > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation
        if not recalls:
            return 0.0
        
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p = max([p for p, r in zip(precisions, recalls) if r >= t], default=0)
            ap += p / 11
        
        return ap
    
    def _count_matches(
        self,
        predictions: List[List[PredictionBox]],
        ground_truth: List[List[GroundTruthBox]],
        iou_threshold: float,
    ) -> Tuple[int, int, int]:
        """Count TP, FP, FN."""
        tp = 0
        fp = 0
        fn = 0
        
        for preds, gts in zip(predictions, ground_truth):
            gt_matched = [False] * len(gts)
            
            for pred in preds:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = compute_iou(pred.to_array(), gt.to_array())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1
            
            fn += sum(1 for m in gt_matched if not m)
        
        return tp, fp, fn
    
    def _compute_overall_pr(
        self,
        predictions: List[PredictionFrame],
        ground_truth: List[GroundTruthFrame],
    ) -> Tuple[float, float]:
        """Compute overall precision and recall."""
        all_preds = [[b for b in p.boxes] for p in predictions]
        all_gt = [[b for b in g.boxes] for g in ground_truth]
        
        tp, fp, fn = self._count_matches(all_preds, all_gt, self.config.iou_thresholds[0])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall


# =============================================================================
# SITE-LEVEL DETECTION METRICS
# =============================================================================

@dataclass
class SiteLevelMetrics:
    """Site-level detection metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    auc: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "auc": self.auc,
        }


class SiteLevelEvaluator:
    """Evaluator for site-level artisanal mining detection."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        predictions: List[PredictionSite],
        ground_truth: List[GroundTruthSite],
    ) -> SiteLevelMetrics:
        """Evaluate site-level predictions."""
        # Match predictions to ground truth by site_id
        pred_dict = {p.site_id: p for p in predictions}
        
        y_true = []
        y_pred = []
        y_scores = []
        
        for gt in ground_truth:
            pred = pred_dict.get(gt.site_id)
            if pred:
                y_true.append(1 if gt.has_artisanal_mining else 0)
                y_pred.append(1 if pred.is_artisanal_mining else 0)
                y_scores.append(pred.confidence)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        # Compute metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Compute AUC
        auc = self._compute_auc(y_true, y_scores)
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        return SiteLevelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            auc=auc,
            confusion_matrix=confusion_matrix,
        )
    
    def _compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-ROC."""
        if len(np.unique(y_true)) < 2:
            return 0.5
        
        # Sort by scores
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Compute TPR and FPR at each threshold
        tpr = []
        fpr = []
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        tp = 0
        fp = 0
        
        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr.append(tp / n_pos if n_pos > 0 else 0)
            fpr.append(fp / n_neg if n_neg > 0 else 0)
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        return auc


# =============================================================================
# TRACKING METRICS (MOTA, MOTP)
# =============================================================================

@dataclass
class TrackingMetrics:
    """Multi-object tracking metrics."""
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision
    id_switches: int
    fragmentations: int
    mostly_tracked: int
    mostly_lost: int
    false_positives: int
    false_negatives: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mota": self.mota,
            "motp": self.motp,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "mostly_tracked": self.mostly_tracked,
            "mostly_lost": self.mostly_lost,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


class TrackingEvaluator:
    """Evaluator for multi-object tracking."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        pred_tracks: List[Dict[str, Any]],
        gt_tracks: List[Dict[str, Any]],
        frames: List[Dict[str, Any]],
    ) -> TrackingMetrics:
        """Evaluate tracking predictions."""
        # Initialize counters
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_iou = 0.0
        total_matches = 0
        
        # Track ID mapping
        gt_to_pred_mapping: Dict[str, str] = {}
        
        for frame in frames:
            frame_id = frame.get("frame_id")
            
            # Get predictions and GT for this frame
            pred_boxes = self._get_frame_boxes(pred_tracks, frame_id)
            gt_boxes = self._get_frame_boxes(gt_tracks, frame_id)
            
            total_gt += len(gt_boxes)
            
            # Match predictions to GT
            matched_gt = set()
            matched_pred = set()
            
            for gt_id, gt_box in gt_boxes.items():
                best_iou = 0.0
                best_pred_id = None
                
                for pred_id, pred_box in pred_boxes.items():
                    if pred_id in matched_pred:
                        continue
                    
                    iou = compute_iou(
                        np.array(gt_box["bbox"]),
                        np.array(pred_box["bbox"]),
                    )
                    
                    if iou > best_iou and iou >= self.config.track_match_threshold:
                        best_iou = iou
                        best_pred_id = pred_id
                
                if best_pred_id:
                    matched_gt.add(gt_id)
                    matched_pred.add(best_pred_id)
                    total_iou += best_iou
                    total_matches += 1
                    
                    # Check for ID switch
                    if gt_id in gt_to_pred_mapping:
                        if gt_to_pred_mapping[gt_id] != best_pred_id:
                            total_id_switches += 1
                    gt_to_pred_mapping[gt_id] = best_pred_id
            
            # Count FP and FN
            total_fp += len(pred_boxes) - len(matched_pred)
            total_fn += len(gt_boxes) - len(matched_gt)
        
        # Compute MOTA and MOTP
        mota = 1 - (total_fn + total_fp + total_id_switches) / total_gt if total_gt > 0 else 0.0
        motp = total_iou / total_matches if total_matches > 0 else 0.0
        
        # Count mostly tracked/lost
        mostly_tracked, mostly_lost = self._count_track_quality(pred_tracks, gt_tracks)
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            id_switches=total_id_switches,
            fragmentations=0,  # Would need more detailed tracking
            mostly_tracked=mostly_tracked,
            mostly_lost=mostly_lost,
            false_positives=total_fp,
            false_negatives=total_fn,
        )
    
    def _get_frame_boxes(
        self,
        tracks: List[Dict[str, Any]],
        frame_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Get boxes for a specific frame from tracks."""
        boxes = {}
        for track in tracks:
            track_id = track.get("track_id")
            for pos in track.get("positions", []):
                if str(pos.get("frame")) == str(frame_id):
                    boxes[track_id] = {
                        "bbox": pos.get("bbox"),
                        "confidence": pos.get("confidence", 1.0),
                    }
        return boxes
    
    def _count_track_quality(
        self,
        pred_tracks: List[Dict[str, Any]],
        gt_tracks: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Count mostly tracked and mostly lost tracks."""
        mostly_tracked = 0
        mostly_lost = 0
        
        for gt_track in gt_tracks:
            gt_frames = set(p.get("frame") for p in gt_track.get("positions", []))
            
            # Find best matching predicted track
            best_overlap = 0.0
            for pred_track in pred_tracks:
                pred_frames = set(p.get("frame") for p in pred_track.get("positions", []))
                overlap = len(gt_frames & pred_frames) / len(gt_frames) if gt_frames else 0
                best_overlap = max(best_overlap, overlap)
            
            if best_overlap >= 0.8:
                mostly_tracked += 1
            elif best_overlap <= 0.2:
                mostly_lost += 1
        
        return mostly_tracked, mostly_lost


# =============================================================================
# CHANGE DETECTION METRICS
# =============================================================================

@dataclass
class ChangeDetectionMetrics:
    """Change detection evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    per_type_f1: Dict[str, float]
    magnitude_correlation: float
    magnitude_mae: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "per_type_f1": self.per_type_f1,
            "magnitude_correlation": self.magnitude_correlation,
            "magnitude_mae": self.magnitude_mae,
        }


class ChangeDetectionEvaluator:
    """Evaluator for change detection."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        predictions: List[PredictionChange],
        ground_truth: List[GroundTruthChange],
    ) -> ChangeDetectionMetrics:
        """Evaluate change detection predictions."""
        # Match by change_id
        pred_dict = {p.change_id: p for p in predictions}
        
        y_true = []
        y_pred = []
        y_scores = []
        
        gt_magnitudes = []
        pred_magnitudes = []
        
        change_types: Dict[str, List[Tuple[int, int]]] = {}
        
        for gt in ground_truth:
            pred = pred_dict.get(gt.change_id)
            if pred:
                y_true.append(1 if gt.has_change else 0)
                y_pred.append(1 if pred.has_change else 0)
                y_scores.append(pred.confidence)
                
                if gt.has_change:
                    gt_magnitudes.append(gt.change_magnitude)
                    pred_magnitudes.append(pred.change_magnitude)
                
                # Track per-type performance
                if gt.change_type:
                    if gt.change_type not in change_types:
                        change_types[gt.change_type] = []
                    change_types[gt.change_type].append((
                        1 if gt.has_change else 0,
                        1 if pred.has_change else 0,
                    ))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        # Binary metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC
        auc = self._compute_auc(y_true, y_scores)
        
        # Per-type F1
        per_type_f1 = {}
        for change_type, pairs in change_types.items():
            type_tp = sum(1 for gt, pred in pairs if gt == 1 and pred == 1)
            type_fp = sum(1 for gt, pred in pairs if gt == 0 and pred == 1)
            type_fn = sum(1 for gt, pred in pairs if gt == 1 and pred == 0)
            
            type_precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
            type_recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
            per_type_f1[change_type] = (
                2 * type_precision * type_recall / (type_precision + type_recall)
                if (type_precision + type_recall) > 0 else 0.0
            )
        
        # Magnitude metrics
        magnitude_correlation = 0.0
        magnitude_mae = 0.0
        
        if gt_magnitudes and pred_magnitudes:
            gt_mag = np.array(gt_magnitudes)
            pred_mag = np.array(pred_magnitudes)
            
            if len(gt_mag) > 1:
                magnitude_correlation = float(np.corrcoef(gt_mag, pred_mag)[0, 1])
            magnitude_mae = float(np.mean(np.abs(gt_mag - pred_mag)))
        
        return ChangeDetectionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            per_type_f1=per_type_f1,
            magnitude_correlation=magnitude_correlation,
            magnitude_mae=magnitude_mae,
        )
    
    def _compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-ROC."""
        if len(np.unique(y_true)) < 2:
            return 0.5
        
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        tpr = []
        fpr = []
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        tp = 0
        fp = 0
        
        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr.append(tp / n_pos if n_pos > 0 else 0)
            fpr.append(fp / n_neg if n_neg > 0 else 0)
        
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        return auc


# =============================================================================
# GEOLOGICAL FEATURE METRICS
# =============================================================================

@dataclass
class GeologicalMetrics:
    """Geological feature evaluation metrics."""
    per_class_f1: Dict[str, float]
    per_class_iou: Dict[str, float]
    mean_f1: float
    mean_iou: float
    mineralization_accuracy: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_class_f1": self.per_class_f1,
            "per_class_iou": self.per_class_iou,
            "mean_f1": self.mean_f1,
            "mean_iou": self.mean_iou,
            "mineralization_accuracy": self.mineralization_accuracy,
        }


class GeologicalEvaluator:
    """Evaluator for geological feature detection."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        predictions: List[PredictionFrame],
        ground_truth: List[GroundTruthFrame],
    ) -> GeologicalMetrics:
        """Evaluate geological feature predictions."""
        # Per-class metrics
        per_class_f1 = {}
        per_class_iou = {}
        
        for class_name in self.config.geological_classes:
            f1, iou = self._evaluate_class(predictions, ground_truth, class_name)
            per_class_f1[class_name] = f1
            per_class_iou[class_name] = iou
        
        # Mean metrics
        mean_f1 = np.mean(list(per_class_f1.values())) if per_class_f1 else 0.0
        mean_iou = np.mean(list(per_class_iou.values())) if per_class_iou else 0.0
        
        # Mineralization potential accuracy
        mineralization_accuracy = self._evaluate_mineralization(predictions, ground_truth)
        
        return GeologicalMetrics(
            per_class_f1=per_class_f1,
            per_class_iou=per_class_iou,
            mean_f1=float(mean_f1),
            mean_iou=float(mean_iou),
            mineralization_accuracy=mineralization_accuracy,
        )
    
    def _evaluate_class(
        self,
        predictions: List[PredictionFrame],
        ground_truth: List[GroundTruthFrame],
        class_name: str,
    ) -> Tuple[float, float]:
        """Evaluate single geological class."""
        tp = 0
        fp = 0
        fn = 0
        total_iou = 0.0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = [b for b in pred.boxes if b.class_name == class_name]
            gt_boxes = [b for b in gt.boxes if b.class_name == class_name]
            
            gt_matched = [False] * len(gt_boxes)
            
            for pred_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = compute_iou(pred_box.to_array(), gt_box.to_array())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.config.iou_thresholds[0] and best_gt_idx >= 0:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                    total_iou += best_iou
                else:
                    fp += 1
            
            fn += sum(1 for m in gt_matched if not m)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = total_iou / tp if tp > 0 else 0.0
        
        return f1, mean_iou
    
    def _evaluate_mineralization(
        self,
        predictions: List[PredictionFrame],
        ground_truth: List[GroundTruthFrame],
    ) -> float:
        """Evaluate mineralization potential predictions."""
        correct = 0
        total = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_potential = pred.scene_type  # Assuming scene_type contains potential
            gt_potential = gt.attributes.get("mineralization_potential")
            
            if gt_potential:
                total += 1
                if pred_potential == gt_potential:
                    correct += 1
        
        return correct / total if total > 0 else 0.0


# =============================================================================
# COMPREHENSIVE EVALUATOR
# =============================================================================

@dataclass
class ComprehensiveMetrics:
    """All evaluation metrics combined."""
    detection: Optional[DetectionMetrics] = None
    site_level: Optional[SiteLevelMetrics] = None
    tracking: Optional[TrackingMetrics] = None
    change_detection: Optional[ChangeDetectionMetrics] = None
    geological: Optional[GeologicalMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.detection:
            result["detection"] = self.detection.to_dict()
        if self.site_level:
            result["site_level"] = self.site_level.to_dict()
        if self.tracking:
            result["tracking"] = self.tracking.to_dict()
        if self.change_detection:
            result["change_detection"] = self.change_detection.to_dict()
        if self.geological:
            result["geological"] = self.geological.to_dict()
        return result
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 50, "EVALUATION SUMMARY", "=" * 50]
        
        if self.detection:
            lines.append(f"\nObject Detection:")
            lines.append(f"  mAP: {self.detection.map:.3f}")
            lines.append(f"  F1: {self.detection.f1:.3f}")
        
        if self.site_level:
            lines.append(f"\nSite-Level Detection:")
            lines.append(f"  F1: {self.site_level.f1:.3f}")
            lines.append(f"  AUC: {self.site_level.auc:.3f}")
            lines.append(f"  FPR: {self.site_level.false_positive_rate:.3f}")
        
        if self.tracking:
            lines.append(f"\nTracking:")
            lines.append(f"  MOTA: {self.tracking.mota:.3f}")
            lines.append(f"  MOTP: {self.tracking.motp:.3f}")
            lines.append(f"  ID Switches: {self.tracking.id_switches}")
        
        if self.change_detection:
            lines.append(f"\nChange Detection:")
            lines.append(f"  F1: {self.change_detection.f1:.3f}")
            lines.append(f"  AUC: {self.change_detection.auc:.3f}")
        
        if self.geological:
            lines.append(f"\nGeological Features:")
            lines.append(f"  Mean F1: {self.geological.mean_f1:.3f}")
            lines.append(f"  Mean IoU: {self.geological.mean_iou:.3f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all metrics."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        
        self.detection_evaluator = DetectionEvaluator(config)
        self.site_evaluator = SiteLevelEvaluator(config)
        self.tracking_evaluator = TrackingEvaluator(config)
        self.change_evaluator = ChangeDetectionEvaluator(config)
        self.geological_evaluator = GeologicalEvaluator(config)
    
    def evaluate_all(
        self,
        frame_predictions: Optional[List[PredictionFrame]] = None,
        frame_ground_truth: Optional[List[GroundTruthFrame]] = None,
        site_predictions: Optional[List[PredictionSite]] = None,
        site_ground_truth: Optional[List[GroundTruthSite]] = None,
        pred_tracks: Optional[List[Dict[str, Any]]] = None,
        gt_tracks: Optional[List[Dict[str, Any]]] = None,
        frames: Optional[List[Dict[str, Any]]] = None,
        change_predictions: Optional[List[PredictionChange]] = None,
        change_ground_truth: Optional[List[GroundTruthChange]] = None,
    ) -> ComprehensiveMetrics:
        """Run all applicable evaluations."""
        metrics = ComprehensiveMetrics()
        
        # Detection metrics
        if frame_predictions and frame_ground_truth:
            metrics.detection = self.detection_evaluator.evaluate(
                frame_predictions, frame_ground_truth
            )
            metrics.geological = self.geological_evaluator.evaluate(
                frame_predictions, frame_ground_truth
            )
        
        # Site-level metrics
        if site_predictions and site_ground_truth:
            metrics.site_level = self.site_evaluator.evaluate(
                site_predictions, site_ground_truth
            )
        
        # Tracking metrics
        if pred_tracks and gt_tracks and frames:
            metrics.tracking = self.tracking_evaluator.evaluate(
                pred_tracks, gt_tracks, frames
            )
        
        # Change detection metrics
        if change_predictions and change_ground_truth:
            metrics.change_detection = self.change_evaluator.evaluate(
                change_predictions, change_ground_truth
            )
        
        return metrics
    
    def save_results(self, metrics: ComprehensiveMetrics, output_path: str) -> None:
        """Save evaluation results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
    
    def load_ground_truth(self, json_path: str) -> Dict[str, Any]:
        """Load ground truth from JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_artisanal_mining(
    predictions: List[PredictionSite],
    ground_truth: List[GroundTruthSite],
) -> SiteLevelMetrics:
    """Evaluate artisanal mining detection."""
    evaluator = SiteLevelEvaluator()
    return evaluator.evaluate(predictions, ground_truth)


def evaluate_geological_features(
    predictions: List[PredictionFrame],
    ground_truth: List[GroundTruthFrame],
) -> GeologicalMetrics:
    """Evaluate geological feature detection."""
    evaluator = GeologicalEvaluator()
    return evaluator.evaluate(predictions, ground_truth)


def evaluate_change_detection(
    predictions: List[PredictionChange],
    ground_truth: List[GroundTruthChange],
) -> ChangeDetectionMetrics:
    """Evaluate change detection."""
    evaluator = ChangeDetectionEvaluator()
    return evaluator.evaluate(predictions, ground_truth)


def create_evaluation_report(
    metrics: ComprehensiveMetrics,
    output_path: str,
) -> None:
    """Create detailed evaluation report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics.to_dict(),
        "summary": metrics.summary(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Created evaluation report: {output_path}")
