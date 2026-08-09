"""
Evaluation Harness Module
=========================

Production-grade evaluation with:
- mAP50/mAP50-95 computation
- Per-class precision/recall
- Confusion matrix generation
- Latency/throughput benchmarks
- Comprehensive reporting
"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    image_id: str = ""


@dataclass
class GroundTruth:
    """Single ground truth annotation."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    image_id: str = ""
    is_difficult: bool = False
    is_crowd: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    max_detections: int = 100
    area_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'all': (0, float('inf')),
        'small': (0, 32**2),
        'medium': (32**2, 96**2),
        'large': (96**2, float('inf'))
    })


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    class_id: int
    class_name: str
    ap50: float = 0.0
    ap50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    num_ground_truth: int = 0
    num_predictions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    per_class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    pr_curves: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    latency_stats: Dict[str, float] = field(default_factory=dict)
    throughput_fps: float = 0.0
    total_images: int = 0
    total_ground_truth: int = 0
    total_predictions: int = 0
    evaluation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'mAP50': self.mAP50,
            'mAP50_95': self.mAP50_95,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'per_class_metrics': {k: v.to_dict() for k, v in self.per_class_metrics.items()},
            'class_names': self.class_names,
            'pr_curves': self.pr_curves,
            'latency_stats': self.latency_stats,
            'throughput_fps': self.throughput_fps,
            'total_images': self.total_images,
            'total_ground_truth': self.total_ground_truth,
            'total_predictions': self.total_predictions,
            'evaluation_time': self.evaluation_time,
            'timestamp': self.timestamp
        }
        
        if self.confusion_matrix is not None:
            d['confusion_matrix'] = self.confusion_matrix.tolist()
        
        return d
    
    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"mAP@50:     {self.mAP50:.4f}",
            f"mAP@50-95:  {self.mAP50_95:.4f}",
            f"Precision:  {self.precision:.4f}",
            f"Recall:     {self.recall:.4f}",
            f"F1 Score:   {self.f1_score:.4f}",
            "-" * 60,
            f"Total Images:       {self.total_images}",
            f"Total Ground Truth: {self.total_ground_truth}",
            f"Total Predictions:  {self.total_predictions}",
            "-" * 60,
            "Per-Class AP@50:",
        ]
        
        for class_name, metrics in sorted(self.per_class_metrics.items()):
            lines.append(f"  {class_name}: {metrics.ap50:.4f}")
        
        if self.latency_stats:
            lines.extend([
                "-" * 60,
                "Latency Statistics:",
                f"  Mean:   {self.latency_stats.get('mean', 0):.2f} ms",
                f"  Std:    {self.latency_stats.get('std', 0):.2f} ms",
                f"  P50:    {self.latency_stats.get('p50', 0):.2f} ms",
                f"  P95:    {self.latency_stats.get('p95', 0):.2f} ms",
                f"  P99:    {self.latency_stats.get('p99', 0):.2f} ms",
                f"Throughput: {self.throughput_fps:.2f} FPS"
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class IOUCalculator:
    """IoU calculation utilities."""
    
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = IOUCalculator.compute_iou(
                    boxes1[i].tolist(), boxes2[j].tolist()
                )
        
        return iou_matrix
    
    @staticmethod
    def box_area(box: List[float]) -> float:
        """Compute box area."""
        return (box[2] - box[0]) * (box[3] - box[1])


class PrecisionRecallCalculator:
    """Precision-Recall curve calculation."""
    
    @staticmethod
    def compute_pr_curve(detections: List[Detection],
                        ground_truths: List[GroundTruth],
                        iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.
        
        Args:
            detections: List of detections sorted by confidence (descending)
            ground_truths: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall, confidence_thresholds)
        """
        if len(detections) == 0:
            return np.array([1.0]), np.array([0.0]), np.array([1.0])
        
        if len(ground_truths) == 0:
            return np.array([0.0]), np.array([0.0]), np.array([detections[0].confidence])
        
        # Sort detections by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        # Track matched ground truths
        gt_matched = [False] * len(ground_truths)
        
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        confidences = np.array([d.confidence for d in detections])
        
        for i, det in enumerate(detections):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truths):
                if gt_matched[j]:
                    continue
                if det.class_id != gt.class_id:
                    continue
                
                iou = IOUCalculator.compute_iou(det.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / len(ground_truths)
        
        return precision, recall, confidences
    
    @staticmethod
    def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
        """
        Compute Average Precision using 101-point interpolation.
        
        Args:
            precision: Precision values
            recall: Recall values
            
        Returns:
            Average Precision
        """
        # Add sentinel values
        recall = np.concatenate([[0.0], recall, [1.0]])
        precision = np.concatenate([[1.0], precision, [0.0]])
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # 101-point interpolation
        recall_thresholds = np.linspace(0, 1, 101)
        ap = 0.0
        
        for t in recall_thresholds:
            prec_at_recall = precision[recall >= t]
            if len(prec_at_recall) > 0:
                ap += prec_at_recall.max()
        
        return ap / 101


class ConfusionMatrixCalculator:
    """Confusion matrix calculation."""
    
    @staticmethod
    def compute(detections: List[Detection],
               ground_truths: List[GroundTruth],
               class_names: List[str],
               iou_threshold: float = 0.5,
               confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            detections: List of detections
            ground_truths: List of ground truth annotations
            class_names: List of class names
            iou_threshold: IoU threshold for matching
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Confusion matrix [num_classes + 1, num_classes + 1]
            Last row/col is background (false positives/negatives)
        """
        num_classes = len(class_names)
        # +1 for background class
        matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
        
        # Filter detections by confidence
        detections = [d for d in detections if d.confidence >= confidence_threshold]
        
        # Group by image
        det_by_image = defaultdict(list)
        gt_by_image = defaultdict(list)
        
        for det in detections:
            det_by_image[det.image_id].append(det)
        for gt in ground_truths:
            gt_by_image[gt.image_id].append(gt)
        
        all_images = set(det_by_image.keys()) | set(gt_by_image.keys())
        
        for image_id in all_images:
            img_dets = det_by_image[image_id]
            img_gts = gt_by_image[image_id]
            
            gt_matched = [False] * len(img_gts)
            
            # Sort detections by confidence
            img_dets = sorted(img_dets, key=lambda x: x.confidence, reverse=True)
            
            for det in img_dets:
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt in enumerate(img_gts):
                    if gt_matched[j]:
                        continue
                    
                    iou = IOUCalculator.compute_iou(det.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt = img_gts[best_gt_idx]
                    gt_matched[best_gt_idx] = True
                    # Predicted class vs actual class
                    matrix[gt.class_id, det.class_id] += 1
                else:
                    # False positive (background -> predicted class)
                    matrix[num_classes, det.class_id] += 1
            
            # Count unmatched ground truths (false negatives)
            for j, gt in enumerate(img_gts):
                if not gt_matched[j]:
                    # Actual class -> background
                    matrix[gt.class_id, num_classes] += 1
        
        return matrix


class LatencyBenchmark:
    """Latency and throughput benchmarking."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start benchmark."""
        self.latencies = []
        self.start_time = time.time()
    
    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self.latencies.append(latency_ms)
    
    def stop(self):
        """Stop benchmark."""
        self.end_time = time.time()
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        
        stats = {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p90': float(np.percentile(latencies, 90)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'count': len(latencies)
        }
        
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            stats['total_time_s'] = total_time
            stats['throughput_fps'] = len(latencies) / total_time if total_time > 0 else 0
        
        return stats


class EvaluationHarness:
    """
    Complete evaluation harness for object detection models.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.iou_calculator = IOUCalculator()
        self.pr_calculator = PrecisionRecallCalculator()
        self.cm_calculator = ConfusionMatrixCalculator()
        self.benchmark = LatencyBenchmark()
    
    def evaluate(self, detections: List[Detection],
                ground_truths: List[GroundTruth],
                class_names: List[str],
                latencies: Optional[List[float]] = None) -> EvaluationResults:
        """
        Run complete evaluation.
        
        Args:
            detections: List of all detections
            ground_truths: List of all ground truth annotations
            class_names: List of class names
            latencies: Optional list of inference latencies in ms
            
        Returns:
            EvaluationResults
        """
        start_time = time.time()
        
        results = EvaluationResults(
            class_names=class_names,
            total_images=len(set(d.image_id for d in detections) | 
                           set(g.image_id for g in ground_truths)),
            total_ground_truth=len(ground_truths),
            total_predictions=len(detections)
        )
        
        # Group by class
        det_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for det in detections:
            det_by_class[det.class_id].append(det)
        for gt in ground_truths:
            gt_by_class[gt.class_id].append(gt)
        
        # Compute per-class metrics
        all_ap50 = []
        all_ap50_95 = []
        all_precision = []
        all_recall = []
        
        for class_id, class_name in enumerate(class_names):
            class_dets = det_by_class.get(class_id, [])
            class_gts = gt_by_class.get(class_id, [])
            
            # Compute AP at different IoU thresholds
            aps = []
            for iou_thresh in self.config.iou_thresholds:
                precision, recall, _ = self.pr_calculator.compute_pr_curve(
                    class_dets, class_gts, iou_thresh
                )
                ap = self.pr_calculator.compute_ap(precision, recall)
                aps.append(ap)
            
            ap50 = aps[0] if aps else 0.0  # IoU=0.5
            ap50_95 = np.mean(aps) if aps else 0.0
            
            # Compute precision/recall at IoU=0.5
            precision, recall, confidences = self.pr_calculator.compute_pr_curve(
                class_dets, class_gts, 0.5
            )
            
            # Get best F1 point
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_precision = precision[best_idx] if len(precision) > 0 else 0.0
            best_recall = recall[best_idx] if len(recall) > 0 else 0.0
            best_f1 = f1_scores[best_idx] if len(f1_scores) > 0 else 0.0
            
            # Count TP, FP, FN at best threshold
            tp = int(best_recall * len(class_gts)) if len(class_gts) > 0 else 0
            fp = int(tp / best_precision - tp) if best_precision > 0 else len(class_dets)
            fn = len(class_gts) - tp
            
            class_metrics = ClassMetrics(
                class_id=class_id,
                class_name=class_name,
                ap50=ap50,
                ap50_95=ap50_95,
                precision=best_precision,
                recall=best_recall,
                f1_score=best_f1,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                num_ground_truth=len(class_gts),
                num_predictions=len(class_dets)
            )
            
            results.per_class_metrics[class_name] = class_metrics
            
            # Store PR curve
            results.pr_curves[class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'confidence': confidences.tolist()
            }
            
            if len(class_gts) > 0:
                all_ap50.append(ap50)
                all_ap50_95.append(ap50_95)
                all_precision.append(best_precision)
                all_recall.append(best_recall)
        
        # Compute overall metrics
        results.mAP50 = np.mean(all_ap50) if all_ap50 else 0.0
        results.mAP50_95 = np.mean(all_ap50_95) if all_ap50_95 else 0.0
        results.precision = np.mean(all_precision) if all_precision else 0.0
        results.recall = np.mean(all_recall) if all_recall else 0.0
        results.f1_score = (2 * results.precision * results.recall / 
                          (results.precision + results.recall + 1e-10))
        
        # Compute confusion matrix
        results.confusion_matrix = self.cm_calculator.compute(
            detections, ground_truths, class_names
        )
        
        # Latency stats
        if latencies:
            self.benchmark.latencies = latencies
            results.latency_stats = self.benchmark.get_stats()
            results.throughput_fps = results.latency_stats.get('throughput_fps', 0)
        
        results.evaluation_time = time.time() - start_time
        
        return results
    
    def evaluate_model(self, model: Any, dataset: Any,
                      class_names: List[str],
                      device: str = 'cuda',
                      batch_size: int = 1) -> EvaluationResults:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Detection model with predict() method
            dataset: Dataset with images and annotations
            class_names: List of class names
            device: Computation device
            batch_size: Batch size for inference
            
        Returns:
            EvaluationResults
        """
        import torch
        
        all_detections = []
        all_ground_truths = []
        latencies = []
        
        model.eval()
        
        for i, (image, annotations) in enumerate(dataset):
            image_id = str(i)
            
            # Convert annotations to GroundTruth
            for ann in annotations:
                gt = GroundTruth(
                    bbox=ann['bbox'],
                    class_id=ann['class_id'],
                    class_name=class_names[ann['class_id']],
                    image_id=image_id
                )
                all_ground_truths.append(gt)
            
            # Run inference
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            image = image.unsqueeze(0).to(device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model(image)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # Parse outputs
            if hasattr(outputs, 'boxes'):
                # YOLO-style output
                boxes = outputs.boxes
                for j in range(len(boxes)):
                    det = Detection(
                        bbox=boxes.xyxy[j].cpu().tolist(),
                        confidence=float(boxes.conf[j]),
                        class_id=int(boxes.cls[j]),
                        class_name=class_names[int(boxes.cls[j])],
                        image_id=image_id
                    )
                    all_detections.append(det)
            elif isinstance(outputs, dict):
                # Dict output
                boxes = outputs.get('boxes', outputs.get('bboxes', []))
                scores = outputs.get('scores', outputs.get('confidences', []))
                labels = outputs.get('labels', outputs.get('class_ids', []))
                
                for j in range(len(boxes)):
                    det = Detection(
                        bbox=boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],
                        confidence=float(scores[j]),
                        class_id=int(labels[j]),
                        class_name=class_names[int(labels[j])],
                        image_id=image_id
                    )
                    all_detections.append(det)
        
        return self.evaluate(all_detections, all_ground_truths, class_names, latencies)
    
    def load_coco_annotations(self, annotations_file: str,
                             images_dir: str) -> Tuple[List[GroundTruth], List[str]]:
        """
        Load ground truth from COCO format annotations.
        
        Args:
            annotations_file: Path to COCO annotations JSON
            images_dir: Path to images directory
            
        Returns:
            Tuple of (ground_truths, class_names)
        """
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping
        cat_id_to_idx = {}
        class_names = []
        for i, cat in enumerate(coco_data['categories']):
            cat_id_to_idx[cat['id']] = i
            class_names.append(cat['name'])
        
        # Build image id to filename mapping
        img_id_to_file = {}
        for img in coco_data['images']:
            img_id_to_file[img['id']] = img['file_name']
        
        # Load annotations
        ground_truths = []
        for ann in coco_data['annotations']:
            # COCO bbox is [x, y, width, height]
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]
            
            gt = GroundTruth(
                bbox=bbox,
                class_id=cat_id_to_idx[ann['category_id']],
                class_name=class_names[cat_id_to_idx[ann['category_id']]],
                image_id=str(ann['image_id']),
                is_crowd=ann.get('iscrowd', 0) == 1
            )
            ground_truths.append(gt)
        
        return ground_truths, class_names
    
    def load_yolo_annotations(self, labels_dir: str,
                             class_names: List[str],
                             image_size: Tuple[int, int] = (640, 640)) -> List[GroundTruth]:
        """
        Load ground truth from YOLO format annotations.
        
        Args:
            labels_dir: Path to labels directory
            class_names: List of class names
            image_size: Image size for denormalization
            
        Returns:
            List of ground truths
        """
        ground_truths = []
        labels_path = Path(labels_dir)
        
        for label_file in labels_path.glob('*.txt'):
            image_id = label_file.stem
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_size[0]
                    y_center = float(parts[2]) * image_size[1]
                    width = float(parts[3]) * image_size[0]
                    height = float(parts[4]) * image_size[1]
                    
                    bbox = [
                        x_center - width / 2,
                        y_center - height / 2,
                        x_center + width / 2,
                        y_center + height / 2
                    ]
                    
                    gt = GroundTruth(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                        image_id=image_id
                    )
                    ground_truths.append(gt)
        
        return ground_truths
    
    def generate_report(self, results: EvaluationResults,
                       output_dir: str,
                       include_plots: bool = True) -> str:
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
            include_plots: Whether to generate plots
            
        Returns:
            Path to report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results.save(str(output_path / 'results.json'))
        
        # Generate text report
        report_path = output_path / 'report.txt'
        with open(report_path, 'w') as f:
            f.write(results.summary())
        
        # Generate plots if requested
        if include_plots:
            self._generate_plots(results, output_path)
        
        return str(report_path)
    
    def _generate_plots(self, results: EvaluationResults, output_path: Path):
        """Generate evaluation plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # PR curves
            fig, ax = plt.subplots(figsize=(10, 8))
            for class_name, pr_data in results.pr_curves.items():
                ax.plot(pr_data['recall'], pr_data['precision'], label=class_name)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves')
            ax.legend(loc='lower left')
            ax.grid(True)
            fig.savefig(output_path / 'pr_curves.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Confusion matrix
            if results.confusion_matrix is not None:
                fig, ax = plt.subplots(figsize=(12, 10))
                cm = results.confusion_matrix
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                labels = results.class_names + ['Background']
                ax.set(xticks=np.arange(cm.shape[1]),
                      yticks=np.arange(cm.shape[0]),
                      xticklabels=labels,
                      yticklabels=labels,
                      title='Confusion Matrix',
                      ylabel='True label',
                      xlabel='Predicted label')
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha='center', va='center',
                               color='white' if cm[i, j] > thresh else 'black')
                
                fig.savefig(output_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Per-class AP bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            class_names = list(results.per_class_metrics.keys())
            ap50_values = [results.per_class_metrics[c].ap50 for c in class_names]
            
            x = np.arange(len(class_names))
            ax.bar(x, ap50_values, color='steelblue')
            ax.set_xlabel('Class')
            ax.set_ylabel('AP@50')
            ax.set_title('Per-Class Average Precision (IoU=0.5)')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.axhline(y=results.mAP50, color='r', linestyle='--', label=f'mAP@50: {results.mAP50:.3f}')
            ax.legend()
            ax.grid(True, axis='y')
            
            fig.savefig(output_path / 'per_class_ap.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Generated plots in {output_path}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")


def create_evaluation_harness(config: Optional[Dict] = None) -> EvaluationHarness:
    """Factory function to create evaluation harness."""
    if config:
        eval_config = EvaluationConfig(
            iou_thresholds=config.get('iou_thresholds', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
            max_detections=config.get('max_detections', 100)
        )
    else:
        eval_config = EvaluationConfig()
    
    return EvaluationHarness(eval_config)
