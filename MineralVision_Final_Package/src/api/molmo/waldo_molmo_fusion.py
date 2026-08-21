"""
WALDO-Molmo Fusion Module for MineralVision.

Combines WALDO's object detection (YOLO11/RF-DETR) with Molmo2-8B's
video understanding for enhanced artisanal mining detection.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .molmo_integration import (
    Molmo2Client,
    Molmo2Config,
    VideoAnalysisResult,
    PointingResult,
    TrackingResult,
    AnalysisType,
    BoundingBox,
    create_molmo_client,
)

logger = logging.getLogger(__name__)


class DetectionSource(Enum):
    """Source of detection."""
    WALDO_YOLO = "waldo_yolo"
    WALDO_RFDETR = "waldo_rfdetr"
    MOLMO_POINTING = "molmo_pointing"
    MOLMO_TRACKING = "molmo_tracking"
    FUSED = "fused"


class ActivityType(Enum):
    """Types of artisanal mining activity."""
    EXCAVATION = "excavation"
    PANNING = "panning"
    SLUICING = "sluicing"
    PROCESSING = "processing"
    TRANSPORT = "transport"
    CAMP = "camp"
    UNKNOWN = "unknown"


@dataclass
class WALDODetection:
    """Detection result from WALDO (YOLO11/RF-DETR)."""
    bbox: BoundingBox
    class_name: str
    confidence: float
    source: DetectionSource
    frame_index: int
    track_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "class_name": self.class_name,
            "confidence": self.confidence,
            "source": self.source.value,
            "frame_index": self.frame_index,
            "track_id": self.track_id,
        }


@dataclass
class MolmoUnderstanding:
    """Understanding result from Molmo2."""
    description: str
    activity_type: ActivityType
    confidence: float
    context: str
    environmental_impact: Optional[str] = None
    safety_concerns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "activity_type": self.activity_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "environmental_impact": self.environmental_impact,
            "safety_concerns": self.safety_concerns,
        }


@dataclass
class FusedDetectionResult:
    """Fused detection combining WALDO and Molmo2."""
    detection_id: str
    waldo_detection: Optional[WALDODetection]
    molmo_understanding: Optional[MolmoUnderstanding]
    fused_confidence: float
    fused_class: str
    bbox: BoundingBox
    frame_index: int
    timestamp: float
    is_artisanal_mining: bool
    activity_type: ActivityType
    description: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "waldo_detection": self.waldo_detection.to_dict() if self.waldo_detection else None,
            "molmo_understanding": self.molmo_understanding.to_dict() if self.molmo_understanding else None,
            "fused_confidence": self.fused_confidence,
            "fused_class": self.fused_class,
            "bbox": self.bbox.to_dict(),
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "is_artisanal_mining": self.is_artisanal_mining,
            "activity_type": self.activity_type.value,
            "description": self.description,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass
class FusionAnalysisResult:
    """Complete result from WALDO-Molmo fusion analysis."""
    video_path: str
    total_frames: int
    frames_analyzed: int
    duration_seconds: float
    fused_detections: List[FusedDetectionResult]
    artisanal_mining_detected: bool
    artisanal_mining_confidence: float
    activity_summary: Dict[ActivityType, int]
    environmental_assessment: str
    safety_assessment: str
    recommendations: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "frames_analyzed": self.frames_analyzed,
            "duration_seconds": self.duration_seconds,
            "fused_detections": [d.to_dict() for d in self.fused_detections],
            "artisanal_mining_detected": self.artisanal_mining_detected,
            "artisanal_mining_confidence": self.artisanal_mining_confidence,
            "activity_summary": {k.value: v for k, v in self.activity_summary.items()},
            "environmental_assessment": self.environmental_assessment,
            "safety_assessment": self.safety_assessment,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }


class WALDOMolmoFusion:
    """
    Fusion system combining WALDO detection with Molmo2 understanding.
    
    WALDO provides fast, accurate object detection using YOLO11/RF-DETR.
    Molmo2 provides semantic understanding, context, and reasoning.
    
    The fusion combines:
    - WALDO's precise bounding boxes and object classification
    - Molmo2's scene understanding and activity interpretation
    - Cross-validation for higher confidence detections
    """
    
    def __init__(
        self,
        molmo_client: Optional[Molmo2Client] = None,
        molmo_config: Optional[Molmo2Config] = None,
        waldo_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        fusion_weight_waldo: float = 0.6,
        fusion_weight_molmo: float = 0.4,
    ):
        self.molmo_client = molmo_client or create_molmo_client(molmo_config)
        self.waldo_model_path = waldo_model_path
        self.confidence_threshold = confidence_threshold
        self.fusion_weight_waldo = fusion_weight_waldo
        self.fusion_weight_molmo = fusion_weight_molmo
        
        self._waldo_model = None
        self._lock = threading.Lock()
        
        # Artisanal mining class mappings
        self.artisanal_classes = {
            "person", "people", "worker", "miner",
            "excavation", "pit", "hole",
            "sluice", "pan", "equipment",
            "tent", "camp", "shelter",
            "vehicle", "motorcycle", "bicycle",
        }
    
    def _load_waldo_model(self):
        """
        Load the canonical WALDO detector.

        Imports detector primitives from the canonical
        MineralVision_WALDO_Production_Package via the WALDO_PACKAGE_SRC
        env var (or a relative-path fallback). When unavailable, runs in
        Molmo-only mode — never fabricates detections.
        """
        if self._waldo_model is not None:
            return

        try:
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
            from waldo_integration.detection import WALDODetector

            model_path = self.waldo_model_path or os.environ.get("WALDO_MODEL_PATH")
            if not model_path:
                logger.info("WALDO model artifact is not configured; using Molmo-only mode")
                return
            if not _os.path.isfile(model_path):
                logger.warning("Configured WALDO model artifact does not exist; using Molmo-only mode")
                return
            self._waldo_model = WALDODetector({
                'model_path': model_path,
                'architecture': 'yolo11',
                'confidence_threshold': self.confidence_threshold,
                'device': 'cpu',
            })
            logger.info(f"Canonical WALDO detector loaded: {self._waldo_model.model_path}")
        except Exception as e:
            logger.warning(f"Canonical WALDO detector unavailable ({e}), using Molmo-only mode")
            self._waldo_model = None
    
    def _run_waldo_detection(
        self,
        frame,
        frame_index: int,
    ) -> List[WALDODetection]:
        """Run canonical WALDO detection on a frame."""
        if self._waldo_model is None:
            return []

        import numpy as np
        raw = self._waldo_model.detect(
            np.asarray(frame), metadata={'frame_id': frame_index}
        )
        detections = []

        for det in raw:
            conf = float(det['confidence'])
            if conf >= self.confidence_threshold:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class_name']
                detections.append(WALDODetection(
                    bbox=BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf, label=class_name
                    ),
                    class_name=class_name,
                    confidence=conf,
                    source=DetectionSource.WALDO_YOLO,
                    frame_index=frame_index,
                ))

        return detections
    
    def _get_molmo_understanding(
        self,
        frame,
        waldo_detections: List[WALDODetection],
    ) -> MolmoUnderstanding:
        """Get Molmo2 understanding of the frame."""
        # Build context from WALDO detections
        detection_context = ""
        if waldo_detections:
            detection_context = "Detected objects: " + ", ".join(
                f"{d.class_name} ({d.confidence:.2f})" for d in waldo_detections[:10]
            )
        
        prompt = f"""Analyze this image for artisanal/small-scale mining activity.
        
        {detection_context}
        
        Determine:
        1. Is this artisanal mining? (yes/no/uncertain)
        2. What type of activity? (excavation/panning/sluicing/processing/transport/camp)
        3. Environmental impact visible?
        4. Safety concerns?
        5. Brief description of the scene.
        
        Be specific and concise."""
        
        result = self.molmo_client.analyze_image(
            frame,
            prompt,
            AnalysisType.IMAGE_QA,
        )
        
        # Parse response
        response_lower = result.response.lower()
        
        # Determine activity type
        activity_type = ActivityType.UNKNOWN
        for at in ActivityType:
            if at.value in response_lower:
                activity_type = at
                break
        
        # Extract safety concerns
        safety_concerns = []
        safety_keywords = ["danger", "hazard", "unsafe", "risk", "concern", "warning"]
        for keyword in safety_keywords:
            if keyword in response_lower:
                # Extract sentence containing keyword
                sentences = result.response.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        safety_concerns.append(sentence.strip())
        
        # Determine environmental impact
        environmental_impact = None
        env_keywords = ["erosion", "pollution", "deforestation", "contamination", "damage"]
        for keyword in env_keywords:
            if keyword in response_lower:
                environmental_impact = f"Potential {keyword} detected"
                break
        
        return MolmoUnderstanding(
            description=result.response,
            activity_type=activity_type,
            confidence=result.confidence,
            context=detection_context,
            environmental_impact=environmental_impact,
            safety_concerns=safety_concerns,
        )
    
    def _fuse_detections(
        self,
        waldo_detections: List[WALDODetection],
        molmo_understanding: MolmoUnderstanding,
        frame_index: int,
        timestamp: float,
    ) -> List[FusedDetectionResult]:
        """Fuse WALDO detections with Molmo understanding."""
        import uuid
        
        fused_results = []
        
        # Process each WALDO detection
        for waldo_det in waldo_detections:
            # Calculate fused confidence
            fused_conf = (
                self.fusion_weight_waldo * waldo_det.confidence +
                self.fusion_weight_molmo * molmo_understanding.confidence
            )
            
            # Determine if artisanal mining
            is_artisanal = (
                waldo_det.class_name.lower() in self.artisanal_classes or
                molmo_understanding.activity_type != ActivityType.UNKNOWN
            )
            
            # Generate recommendations
            recommendations = []
            if is_artisanal:
                recommendations.append("Flag for detailed review")
                if molmo_understanding.environmental_impact:
                    recommendations.append("Assess environmental impact")
                if molmo_understanding.safety_concerns:
                    recommendations.append("Review safety concerns")
            
            fused_results.append(FusedDetectionResult(
                detection_id=str(uuid.uuid4())[:8],
                waldo_detection=waldo_det,
                molmo_understanding=molmo_understanding,
                fused_confidence=fused_conf,
                fused_class=waldo_det.class_name,
                bbox=waldo_det.bbox,
                frame_index=frame_index,
                timestamp=timestamp,
                is_artisanal_mining=is_artisanal,
                activity_type=molmo_understanding.activity_type,
                description=molmo_understanding.description[:200],
                recommendations=recommendations,
            ))
        
        # If no WALDO detections but Molmo found activity
        if not waldo_detections and molmo_understanding.activity_type != ActivityType.UNKNOWN:
            fused_results.append(FusedDetectionResult(
                detection_id=str(uuid.uuid4())[:8],
                waldo_detection=None,
                molmo_understanding=molmo_understanding,
                fused_confidence=molmo_understanding.confidence * 0.8,
                fused_class="artisanal_activity",
                bbox=BoundingBox(0, 0, 1, 1, molmo_understanding.confidence),
                frame_index=frame_index,
                timestamp=timestamp,
                is_artisanal_mining=True,
                activity_type=molmo_understanding.activity_type,
                description=molmo_understanding.description[:200],
                recommendations=["Verify with higher resolution imagery"],
            ))
        
        return fused_results
    
    def analyze_video(
        self,
        video_path: str,
        sample_rate: int = 30,
    ) -> FusionAnalysisResult:
        """
        Analyze video using WALDO-Molmo fusion.
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame
        
        Returns:
            FusionAnalysisResult with complete analysis
        """
        import time
        import cv2
        from PIL import Image
        
        start_time = time.time()
        
        # Load WALDO model
        self._load_waldo_model()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        all_fused_detections = []
        activity_counts = {at: 0 for at in ActivityType}
        frames_analyzed = 0
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % sample_rate == 0:
                frames_analyzed += 1
                timestamp = frame_index / fps
                
                # Convert to RGB for Molmo
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Run WALDO detection
                waldo_detections = self._run_waldo_detection(frame, frame_index)
                
                # Get Molmo understanding
                molmo_understanding = self._get_molmo_understanding(
                    pil_frame, waldo_detections
                )
                
                # Fuse results
                fused = self._fuse_detections(
                    waldo_detections,
                    molmo_understanding,
                    frame_index,
                    timestamp,
                )
                
                all_fused_detections.extend(fused)
                
                # Count activities
                for det in fused:
                    activity_counts[det.activity_type] += 1
            
            frame_index += 1
        
        cap.release()
        
        # Determine overall artisanal mining detection
        artisanal_detections = [d for d in all_fused_detections if d.is_artisanal_mining]
        artisanal_detected = len(artisanal_detections) > 0
        artisanal_confidence = (
            sum(d.fused_confidence for d in artisanal_detections) / len(artisanal_detections)
            if artisanal_detections else 0.0
        )
        
        # Generate assessments
        environmental_assessment = self._generate_environmental_assessment(all_fused_detections)
        safety_assessment = self._generate_safety_assessment(all_fused_detections)
        recommendations = self._generate_recommendations(
            artisanal_detected, activity_counts, all_fused_detections
        )
        
        processing_time = time.time() - start_time
        
        return FusionAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            frames_analyzed=frames_analyzed,
            duration_seconds=duration,
            fused_detections=all_fused_detections,
            artisanal_mining_detected=artisanal_detected,
            artisanal_mining_confidence=artisanal_confidence,
            activity_summary=activity_counts,
            environmental_assessment=environmental_assessment,
            safety_assessment=safety_assessment,
            recommendations=recommendations,
            processing_time=processing_time,
            metadata={
                "sample_rate": sample_rate,
                "waldo_available": self._waldo_model is not None,
            },
        )
    
    def analyze_image(
        self,
        image_path: str,
    ) -> FusedDetectionResult:
        """Analyze a single image using WALDO-Molmo fusion."""
        import cv2
        from PIL import Image
        
        self._load_waldo_model()
        
        # Load image
        frame = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        
        # Run WALDO detection
        waldo_detections = self._run_waldo_detection(frame, 0)
        
        # Get Molmo understanding
        molmo_understanding = self._get_molmo_understanding(pil_frame, waldo_detections)
        
        # Fuse results
        fused = self._fuse_detections(waldo_detections, molmo_understanding, 0, 0.0)
        
        if fused:
            return fused[0]
        else:
            import uuid
            return FusedDetectionResult(
                detection_id=str(uuid.uuid4())[:8],
                waldo_detection=None,
                molmo_understanding=molmo_understanding,
                fused_confidence=molmo_understanding.confidence,
                fused_class="scene",
                bbox=BoundingBox(0, 0, 1, 1, 0.5),
                frame_index=0,
                timestamp=0.0,
                is_artisanal_mining=False,
                activity_type=molmo_understanding.activity_type,
                description=molmo_understanding.description,
            )
    
    def _generate_environmental_assessment(
        self,
        detections: List[FusedDetectionResult],
    ) -> str:
        """Generate environmental impact assessment."""
        impacts = []
        for det in detections:
            if det.molmo_understanding and det.molmo_understanding.environmental_impact:
                impacts.append(det.molmo_understanding.environmental_impact)
        
        if impacts:
            return f"Environmental concerns detected: {'; '.join(set(impacts))}"
        return "No significant environmental concerns detected"
    
    def _generate_safety_assessment(
        self,
        detections: List[FusedDetectionResult],
    ) -> str:
        """Generate safety assessment."""
        concerns = []
        for det in detections:
            if det.molmo_understanding and det.molmo_understanding.safety_concerns:
                concerns.extend(det.molmo_understanding.safety_concerns)
        
        if concerns:
            return f"Safety concerns: {'; '.join(set(concerns)[:5])}"
        return "No immediate safety concerns identified"
    
    def _generate_recommendations(
        self,
        artisanal_detected: bool,
        activity_counts: Dict[ActivityType, int],
        detections: List[FusedDetectionResult],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if artisanal_detected:
            recommendations.append("Artisanal mining activity detected - initiate detailed assessment")
            recommendations.append("Consider ground verification of detected sites")
            recommendations.append("Document for regulatory reporting if required")
        
        if activity_counts.get(ActivityType.EXCAVATION, 0) > 5:
            recommendations.append("Multiple excavation sites detected - assess cumulative impact")
        
        if activity_counts.get(ActivityType.PROCESSING, 0) > 0:
            recommendations.append("Processing activity detected - check for mercury/chemical use")
        
        # Check for high-confidence detections
        high_conf = [d for d in detections if d.fused_confidence > 0.8]
        if high_conf:
            recommendations.append(f"{len(high_conf)} high-confidence detections require priority review")
        
        return recommendations
