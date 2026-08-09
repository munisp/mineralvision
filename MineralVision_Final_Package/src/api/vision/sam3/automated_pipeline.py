"""
Automated SAM3 Pipeline for MineralVision

Fully automated inference, training, and monitoring pipeline
that operates without human-in-the-loop for routine operations.

Features:
- Event-driven inference via Kafka
- Automated drift detection and monitoring
- Self-training with weak supervision
- Confidence-based quality gates
- Canary deployment for new adapters
"""

import logging
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import queue
import time

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class EventType(str, Enum):
    """Types of events in the automated pipeline."""
    NEW_IMAGE = "new_image"
    INFERENCE_COMPLETE = "inference_complete"
    DRIFT_DETECTED = "drift_detected"
    TRAINING_TRIGGERED = "training_triggered"
    ADAPTER_PROMOTED = "adapter_promoted"
    QUALITY_ALERT = "quality_alert"


class DataSource(str, Enum):
    """Source types for incoming data."""
    UAV_FLIGHT = "uav_flight"
    CORE_PHOTO = "core_photo"
    SOIL_PIT = "soil_pit"
    THIN_SECTION = "thin_section"
    GEOPHYSICS_GRID = "geophysics_grid"
    SATELLITE = "satellite"


@dataclass
class PromptPack:
    """Collection of prompts for a specific project/modality."""
    name: str
    modality: str
    prompts: List[Dict[str, Any]]
    confidence_threshold: float = 0.7
    min_mask_area: int = 100
    max_mask_area: Optional[int] = None
    post_processing: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptPack":
        return cls(**d)


@dataclass
class InferenceEvent:
    """Event representing an inference request."""
    event_id: str
    source: DataSource
    image_path: str
    prompt_pack: str
    project_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InferenceEvent":
        d = d.copy()
        d["source"] = DataSource(d["source"])
        return cls(**d)


@dataclass
class InferenceResult:
    """Result from automated inference."""
    event_id: str
    image_path: str
    masks: List[Dict[str, Any]]
    confidence_scores: List[float]
    prompts_used: List[str]
    processing_time_ms: float
    adapter_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftMetrics:
    """Metrics for detecting model/data drift."""
    timestamp: str
    image_brightness_mean: float
    image_brightness_std: float
    mask_area_mean: float
    mask_area_std: float
    mask_count_mean: float
    confidence_mean: float
    prompt_failure_rate: float
    latency_mean_ms: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromptPackRegistry:
    """Registry for managing prompt packs per project/modality."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._packs: Dict[str, PromptPack] = {}
        self._load_default_packs()
    
    def _load_default_packs(self) -> None:
        """Load default prompt packs for each modality."""
        # Soil assessment prompt pack
        self._packs["soil_default"] = PromptPack(
            name="soil_default",
            modality="soil",
            prompts=[
                {"concept": "soil_horizon_a", "text": "topsoil layer", "priority": 1},
                {"concept": "soil_horizon_b", "text": "subsoil layer", "priority": 2},
                {"concept": "laterite", "text": "laterite hardpan", "priority": 3},
                {"concept": "erosion_gully", "text": "erosion gully", "priority": 4},
                {"concept": "waterlogging", "text": "waterlogged soil", "priority": 5},
            ],
            confidence_threshold=0.6,
            min_mask_area=500
        )
        
        # Gold exploration prompt pack
        self._packs["gold_default"] = PromptPack(
            name="gold_default",
            modality="drillcore",
            prompts=[
                {"concept": "visible_gold", "text": "visible gold", "priority": 1},
                {"concept": "gold_bearing_vein", "text": "quartz vein", "priority": 2},
                {"concept": "sulfide_zone", "text": "sulfide minerals", "priority": 3},
                {"concept": "arsenopyrite_gold", "text": "arsenopyrite", "priority": 4},
                {"concept": "silicification", "text": "silicified rock", "priority": 5},
            ],
            confidence_threshold=0.7,
            min_mask_area=50
        )
        
        # Lithium exploration prompt pack
        self._packs["lithium_default"] = PromptPack(
            name="lithium_default",
            modality="drillcore",
            prompts=[
                {"concept": "spodumene", "text": "spodumene crystal", "priority": 1},
                {"concept": "lepidolite", "text": "purple mica", "priority": 2},
                {"concept": "pegmatite_zone", "text": "pegmatite", "priority": 3},
                {"concept": "greisen", "text": "greisen alteration", "priority": 4},
            ],
            confidence_threshold=0.65,
            min_mask_area=100
        )
        
        # Rare earth prompt pack
        self._packs["ree_default"] = PromptPack(
            name="ree_default",
            modality="drillcore",
            prompts=[
                {"concept": "carbonatite", "text": "carbonatite", "priority": 1},
                {"concept": "bastnaesite", "text": "REE carbonate mineral", "priority": 2},
                {"concept": "monazite_core", "text": "monazite", "priority": 3},
                {"concept": "fenitization", "text": "fenite alteration", "priority": 4},
            ],
            confidence_threshold=0.6,
            min_mask_area=100
        )
        
        # UAV/Satellite prompt pack
        self._packs["uav_default"] = PromptPack(
            name="uav_default",
            modality="uav_ortho",
            prompts=[
                {"concept": "gossan", "text": "oxidized outcrop", "priority": 1},
                {"concept": "lineament", "text": "geological lineament", "priority": 2},
                {"concept": "artisanal_working", "text": "artisanal mining pit", "priority": 3},
                {"concept": "quartz_blow", "text": "quartz outcrop", "priority": 4},
            ],
            confidence_threshold=0.65,
            min_mask_area=1000
        )
    
    def get_pack(self, name: str) -> Optional[PromptPack]:
        """Get a prompt pack by name."""
        return self._packs.get(name)
    
    def register_pack(self, pack: PromptPack) -> None:
        """Register a new prompt pack."""
        self._packs[pack.name] = pack
    
    def list_packs(self) -> List[str]:
        """List all available prompt packs."""
        return list(self._packs.keys())


class AutomatedInferencePipeline:
    """
    Automated inference pipeline for SAM3.
    
    Processes images without human intervention using
    predefined prompt packs and quality gates.
    """
    
    def __init__(
        self,
        prompt_registry: Optional[PromptPackRegistry] = None,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.7,
        enable_kafka: bool = True,
        kafka_bootstrap_servers: str = "localhost:9092"
    ):
        self.prompt_registry = prompt_registry or PromptPackRegistry()
        self.output_dir = Path(output_dir) if output_dir else Path("./sam3_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        
        # Kafka setup
        self.enable_kafka = enable_kafka and KAFKA_AVAILABLE
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.producer = None
        self.consumer = None
        
        # Processing queue for non-Kafka mode
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Metrics collection
        self._metrics_buffer: List[Dict[str, Any]] = []
        self._metrics_window_size = 100
        
        # Segmenter (lazy initialization)
        self._segmenter = None
    
    def _get_segmenter(self):
        """Get or create SAM3 segmenter."""
        if self._segmenter is None:
            from .sam3_segmenter import create_sam3_segmenter
            self._segmenter = create_sam3_segmenter()
        return self._segmenter
    
    def start(self) -> None:
        """Start the automated pipeline."""
        self._running = True
        
        if self.enable_kafka:
            self._start_kafka()
        else:
            self._start_local_worker()
        
        logger.info("Automated inference pipeline started")
    
    def stop(self) -> None:
        """Stop the automated pipeline."""
        self._running = False
        
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        logger.info("Automated inference pipeline stopped")
    
    def _start_kafka(self) -> None:
        """Start Kafka consumer/producer."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, falling back to local queue")
            self._start_local_worker()
            return
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            self.consumer = KafkaConsumer(
                'sam3.inference.requests',
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id='sam3-inference-group',
                auto_offset_reset='latest'
            )
            
            self._worker_thread = threading.Thread(target=self._kafka_worker)
            self._worker_thread.daemon = True
            self._worker_thread.start()
        except Exception as e:
            logger.error(f"Failed to start Kafka: {e}")
            self._start_local_worker()
    
    def _start_local_worker(self) -> None:
        """Start local queue worker."""
        self._worker_thread = threading.Thread(target=self._local_worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()
    
    def _kafka_worker(self) -> None:
        """Process messages from Kafka."""
        while self._running:
            try:
                for message in self.consumer:
                    if not self._running:
                        break
                    event = InferenceEvent.from_dict(message.value)
                    result = self._process_event(event)
                    
                    if self.producer:
                        self.producer.send(
                            'sam3.inference.results',
                            value=result.to_dict()
                        )
            except Exception as e:
                logger.error(f"Kafka worker error: {e}")
                time.sleep(1)
    
    def _local_worker(self) -> None:
        """Process events from local queue."""
        while self._running:
            try:
                event = self._queue.get(timeout=1)
                self._process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Local worker error: {e}")
    
    def submit(self, event: InferenceEvent) -> None:
        """Submit an event for processing."""
        if self.enable_kafka and self.producer:
            self.producer.send('sam3.inference.requests', value=event.to_dict())
        else:
            self._queue.put(event)
    
    def _process_event(self, event: InferenceEvent) -> InferenceResult:
        """Process a single inference event."""
        start_time = time.time()
        
        # Get prompt pack
        pack = self.prompt_registry.get_pack(event.prompt_pack)
        if not pack:
            pack = self.prompt_registry.get_pack(f"{event.source.value}_default")
        if not pack:
            pack = self.prompt_registry.get_pack("soil_default")
        
        segmenter = self._get_segmenter()
        
        all_masks = []
        all_scores = []
        prompts_used = []
        
        # Run inference for each prompt
        for prompt_config in pack.prompts:
            try:
                result = segmenter.segment_by_text(
                    image=event.image_path,
                    text_prompt=prompt_config["text"],
                    concept=prompt_config.get("concept")
                )
                
                # Filter by confidence threshold
                for i, score in enumerate(result.scores):
                    if score >= pack.confidence_threshold:
                        mask_info = {
                            "concept": prompt_config.get("concept", prompt_config["text"]),
                            "score": score,
                            "prompt": prompt_config["text"],
                            "mask_index": len(all_masks)
                        }
                        all_masks.append(mask_info)
                        all_scores.append(score)
                        prompts_used.append(prompt_config["text"])
            except Exception as e:
                logger.warning(f"Prompt failed: {prompt_config['text']}: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create result
        result = InferenceResult(
            event_id=event.event_id,
            image_path=event.image_path,
            masks=all_masks,
            confidence_scores=all_scores,
            prompts_used=list(set(prompts_used)),
            processing_time_ms=processing_time,
            metadata={
                "prompt_pack": pack.name,
                "source": event.source.value,
                "project_id": event.project_id
            }
        )
        
        # Collect metrics
        self._collect_metrics(event, result)
        
        # Save result
        self._save_result(result)
        
        return result
    
    def _collect_metrics(self, event: InferenceEvent, result: InferenceResult) -> None:
        """Collect metrics for drift detection."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "mask_count": len(result.masks),
            "confidence_mean": np.mean(result.confidence_scores) if result.confidence_scores else 0,
            "latency_ms": result.processing_time_ms,
            "prompt_failure_rate": 1 - (len(result.masks) / max(len(result.prompts_used), 1))
        }
        
        self._metrics_buffer.append(metrics)
        
        # Keep buffer size limited
        if len(self._metrics_buffer) > self._metrics_window_size:
            self._metrics_buffer = self._metrics_buffer[-self._metrics_window_size:]
    
    def _save_result(self, result: InferenceResult) -> None:
        """Save inference result to disk."""
        result_dir = self.output_dir / "results"
        result_dir.mkdir(exist_ok=True)
        
        result_file = result_dir / f"{result.event_id}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def process_batch(
        self,
        image_paths: List[str],
        source: DataSource,
        prompt_pack: str,
        project_id: Optional[str] = None
    ) -> List[InferenceResult]:
        """Process a batch of images synchronously."""
        results = []
        
        for i, image_path in enumerate(image_paths):
            event = InferenceEvent(
                event_id=f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}",
                source=source,
                image_path=image_path,
                prompt_pack=prompt_pack,
                project_id=project_id
            )
            result = self._process_event(event)
            results.append(result)
        
        return results


class DriftDetector:
    """
    Detects data and model drift in the automated pipeline.
    
    Monitors:
    - Image distribution changes
    - Mask area/count distributions
    - Confidence score distributions
    - Latency changes
    """
    
    def __init__(
        self,
        baseline_window: int = 1000,
        detection_window: int = 100,
        drift_threshold: float = 2.0
    ):
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.drift_threshold = drift_threshold
        
        self._baseline_metrics: List[Dict[str, float]] = []
        self._current_metrics: List[Dict[str, float]] = []
        self._baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
        self._drift_alerts: List[Dict[str, Any]] = []
    
    def add_metrics(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Add new metrics and check for drift.
        
        Returns drift alert if detected, None otherwise.
        """
        self._current_metrics.append(metrics)
        
        # Build baseline if not enough data
        if len(self._baseline_metrics) < self.baseline_window:
            self._baseline_metrics.append(metrics)
            if len(self._baseline_metrics) == self.baseline_window:
                self._compute_baseline_stats()
            return None
        
        # Keep current window limited
        if len(self._current_metrics) > self.detection_window:
            self._current_metrics = self._current_metrics[-self.detection_window:]
        
        # Check for drift
        if len(self._current_metrics) >= self.detection_window:
            drift_alert = self._check_drift()
            if drift_alert:
                self._drift_alerts.append(drift_alert)
                return drift_alert
        
        return None
    
    def _compute_baseline_stats(self) -> None:
        """Compute baseline statistics."""
        self._baseline_stats = {}
        
        metric_keys = ["mask_count", "confidence_mean", "latency_ms", "prompt_failure_rate"]
        
        for key in metric_keys:
            values = [m.get(key, 0) for m in self._baseline_metrics]
            self._baseline_stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values) + 1e-6
            }
    
    def _check_drift(self) -> Optional[Dict[str, Any]]:
        """Check if current metrics indicate drift."""
        if not self._baseline_stats:
            return None
        
        drift_detected = False
        drift_details = {}
        
        for key, baseline in self._baseline_stats.items():
            current_values = [m.get(key, 0) for m in self._current_metrics]
            current_mean = np.mean(current_values)
            
            # Z-score based drift detection
            z_score = abs(current_mean - baseline["mean"]) / baseline["std"]
            
            if z_score > self.drift_threshold:
                drift_detected = True
                drift_details[key] = {
                    "baseline_mean": baseline["mean"],
                    "current_mean": current_mean,
                    "z_score": z_score
                }
        
        if drift_detected:
            return {
                "timestamp": datetime.now().isoformat(),
                "type": "drift_detected",
                "details": drift_details,
                "severity": "high" if any(d["z_score"] > 3 for d in drift_details.values()) else "medium"
            }
        
        return None
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        return {
            "baseline_samples": len(self._baseline_metrics),
            "current_window_samples": len(self._current_metrics),
            "baseline_established": self._baseline_stats is not None,
            "total_drift_alerts": len(self._drift_alerts),
            "recent_alerts": self._drift_alerts[-5:] if self._drift_alerts else []
        }


class SelfTrainingPipeline:
    """
    Self-training pipeline for continuous model improvement.
    
    Uses weak supervision and high-confidence predictions
    to generate training data without human labeling.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        min_samples_for_training: int = 100,
        training_interval_hours: int = 24
    ):
        self.confidence_threshold = confidence_threshold
        self.min_samples_for_training = min_samples_for_training
        self.training_interval_hours = training_interval_hours
        
        self._high_confidence_samples: List[Dict[str, Any]] = []
        self._last_training_time: Optional[datetime] = None
        self._training_history: List[Dict[str, Any]] = []
    
    def add_inference_result(self, result: InferenceResult) -> None:
        """Add inference result for potential self-training."""
        # Filter high-confidence predictions
        high_conf_masks = [
            m for m, s in zip(result.masks, result.confidence_scores)
            if s >= self.confidence_threshold
        ]
        
        if high_conf_masks:
            self._high_confidence_samples.append({
                "image_path": result.image_path,
                "masks": high_conf_masks,
                "scores": [s for s in result.confidence_scores if s >= self.confidence_threshold],
                "timestamp": result.timestamp
            })
    
    def should_trigger_training(self) -> bool:
        """Check if training should be triggered."""
        # Check sample count
        if len(self._high_confidence_samples) < self.min_samples_for_training:
            return False
        
        # Check time interval
        if self._last_training_time:
            hours_since_last = (datetime.now() - self._last_training_time).total_seconds() / 3600
            if hours_since_last < self.training_interval_hours:
                return False
        
        return True
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Prepare training data from high-confidence samples."""
        training_data = {
            "samples": self._high_confidence_samples.copy(),
            "sample_count": len(self._high_confidence_samples),
            "prepared_at": datetime.now().isoformat(),
            "confidence_threshold": self.confidence_threshold
        }
        
        return training_data
    
    def trigger_training(self, training_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Trigger self-training with accumulated samples."""
        if not self.should_trigger_training():
            return {"status": "skipped", "reason": "conditions not met"}
        
        training_data = self.prepare_training_data()
        
        # Record training attempt
        training_record = {
            "triggered_at": datetime.now().isoformat(),
            "sample_count": training_data["sample_count"],
            "status": "started"
        }
        
        try:
            if training_callback:
                result = training_callback(training_data)
                training_record["status"] = "completed"
                training_record["result"] = result
            else:
                training_record["status"] = "no_callback"
            
            # Clear samples after successful training
            self._high_confidence_samples = []
            self._last_training_time = datetime.now()
        except Exception as e:
            training_record["status"] = "failed"
            training_record["error"] = str(e)
        
        self._training_history.append(training_record)
        return training_record
    
    def get_status(self) -> Dict[str, Any]:
        """Get self-training pipeline status."""
        return {
            "accumulated_samples": len(self._high_confidence_samples),
            "min_samples_required": self.min_samples_for_training,
            "last_training": self._last_training_time.isoformat() if self._last_training_time else None,
            "training_history_count": len(self._training_history),
            "ready_for_training": self.should_trigger_training()
        }


class QualityGate:
    """
    Quality gate for automated adapter promotion.
    
    Evaluates adapters against benchmark holdout sets
    before allowing promotion to production.
    """
    
    def __init__(
        self,
        min_iou_threshold: float = 0.7,
        min_samples: int = 50,
        max_regression_percent: float = 5.0
    ):
        self.min_iou_threshold = min_iou_threshold
        self.min_samples = min_samples
        self.max_regression_percent = max_regression_percent
        
        self._benchmark_results: Dict[str, List[Dict[str, float]]] = {}
        self._promotion_history: List[Dict[str, Any]] = []
    
    def add_benchmark_result(
        self,
        adapter_id: str,
        iou: float,
        dice: float,
        latency_ms: float
    ) -> None:
        """Add benchmark evaluation result."""
        if adapter_id not in self._benchmark_results:
            self._benchmark_results[adapter_id] = []
        
        self._benchmark_results[adapter_id].append({
            "iou": iou,
            "dice": dice,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        })
    
    def evaluate_adapter(
        self,
        adapter_id: str,
        baseline_adapter_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate adapter for promotion."""
        if adapter_id not in self._benchmark_results:
            return {
                "passed": False,
                "reason": "no_benchmark_results",
                "adapter_id": adapter_id
            }
        
        results = self._benchmark_results[adapter_id]
        
        if len(results) < self.min_samples:
            return {
                "passed": False,
                "reason": "insufficient_samples",
                "samples": len(results),
                "required": self.min_samples
            }
        
        # Calculate metrics
        avg_iou = np.mean([r["iou"] for r in results])
        avg_dice = np.mean([r["dice"] for r in results])
        avg_latency = np.mean([r["latency_ms"] for r in results])
        
        # Check minimum threshold
        if avg_iou < self.min_iou_threshold:
            return {
                "passed": False,
                "reason": "below_threshold",
                "iou": avg_iou,
                "threshold": self.min_iou_threshold
            }
        
        # Check regression against baseline
        if baseline_adapter_id and baseline_adapter_id in self._benchmark_results:
            baseline_results = self._benchmark_results[baseline_adapter_id]
            baseline_iou = np.mean([r["iou"] for r in baseline_results])
            
            regression_percent = ((baseline_iou - avg_iou) / baseline_iou) * 100
            
            if regression_percent > self.max_regression_percent:
                return {
                    "passed": False,
                    "reason": "regression",
                    "regression_percent": regression_percent,
                    "max_allowed": self.max_regression_percent
                }
        
        return {
            "passed": True,
            "adapter_id": adapter_id,
            "metrics": {
                "iou": avg_iou,
                "dice": avg_dice,
                "latency_ms": avg_latency
            },
            "sample_count": len(results)
        }
    
    def approve_promotion(self, adapter_id: str, evaluation: Dict[str, Any]) -> bool:
        """Record promotion approval."""
        if not evaluation.get("passed"):
            return False
        
        self._promotion_history.append({
            "adapter_id": adapter_id,
            "approved_at": datetime.now().isoformat(),
            "evaluation": evaluation
        })
        
        return True


class AutomatedSAM3System:
    """
    Complete automated SAM3 system integrating all components.
    
    Provides:
    - Automated inference pipeline
    - Drift detection and monitoring
    - Self-training with weak supervision
    - Quality gates for adapter promotion
    """
    
    def __init__(
        self,
        output_dir: str = "./sam3_automated",
        enable_kafka: bool = False,
        kafka_servers: str = "localhost:9092"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.prompt_registry = PromptPackRegistry()
        
        self.inference_pipeline = AutomatedInferencePipeline(
            prompt_registry=self.prompt_registry,
            output_dir=str(self.output_dir / "inference"),
            enable_kafka=enable_kafka,
            kafka_bootstrap_servers=kafka_servers
        )
        
        self.drift_detector = DriftDetector()
        self.self_training = SelfTrainingPipeline()
        self.quality_gate = QualityGate()
        
        self._running = False
    
    def start(self) -> None:
        """Start the automated system."""
        self._running = True
        self.inference_pipeline.start()
        logger.info("Automated SAM3 system started")
    
    def stop(self) -> None:
        """Stop the automated system."""
        self._running = False
        self.inference_pipeline.stop()
        logger.info("Automated SAM3 system stopped")
    
    def process_image(
        self,
        image_path: str,
        source: DataSource,
        prompt_pack: str,
        project_id: Optional[str] = None
    ) -> InferenceResult:
        """Process a single image through the automated pipeline."""
        event = InferenceEvent(
            event_id=f"auto_{hashlib.md5(image_path.encode()).hexdigest()[:8]}_{int(time.time())}",
            source=source,
            image_path=image_path,
            prompt_pack=prompt_pack,
            project_id=project_id
        )
        
        result = self.inference_pipeline._process_event(event)
        
        # Feed to drift detector
        metrics = {
            "mask_count": len(result.masks),
            "confidence_mean": np.mean(result.confidence_scores) if result.confidence_scores else 0,
            "latency_ms": result.processing_time_ms,
            "prompt_failure_rate": 0
        }
        drift_alert = self.drift_detector.add_metrics(metrics)
        
        if drift_alert:
            logger.warning(f"Drift detected: {drift_alert}")
        
        # Feed to self-training
        self.self_training.add_inference_result(result)
        
        return result
    
    def process_batch(
        self,
        image_paths: List[str],
        source: DataSource,
        prompt_pack: str,
        project_id: Optional[str] = None
    ) -> List[InferenceResult]:
        """Process a batch of images."""
        results = []
        for path in image_paths:
            result = self.process_image(path, source, prompt_pack, project_id)
            results.append(result)
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            "running": self._running,
            "drift_detection": self.drift_detector.get_drift_summary(),
            "self_training": self.self_training.get_status(),
            "prompt_packs": self.prompt_registry.list_packs()
        }
    
    def trigger_self_training(self) -> Dict[str, Any]:
        """Manually trigger self-training if conditions are met."""
        return self.self_training.trigger_training()


def create_automated_system(
    output_dir: str = "./sam3_automated",
    enable_kafka: bool = False
) -> AutomatedSAM3System:
    """Factory function to create automated SAM3 system."""
    return AutomatedSAM3System(
        output_dir=output_dir,
        enable_kafka=enable_kafka
    )
