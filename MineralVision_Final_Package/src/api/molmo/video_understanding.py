"""
Video Understanding Pipeline for MineralVision.

Provides comprehensive video analysis capabilities using Molmo2-8B
for mining exploration, site monitoring, and geological analysis.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor

from .molmo_integration import (
    Molmo2Client,
    Molmo2Config,
    VideoAnalysisResult,
    PointingResult,
    TrackingResult,
    AnalysisType,
    create_molmo_client,
)

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of temporal events detected in video."""
    ACTIVITY_START = "activity_start"
    ACTIVITY_END = "activity_end"
    OBJECT_APPEAR = "object_appear"
    OBJECT_DISAPPEAR = "object_disappear"
    MOVEMENT_DETECTED = "movement_detected"
    CHANGE_DETECTED = "change_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    EQUIPMENT_OPERATION = "equipment_operation"
    PERSON_DETECTED = "person_detected"
    VEHICLE_DETECTED = "vehicle_detected"


class SceneType(Enum):
    """Types of scenes in mining/exploration context."""
    OPEN_PIT = "open_pit"
    UNDERGROUND = "underground"
    PROCESSING_PLANT = "processing_plant"
    EXPLORATION_SITE = "exploration_site"
    DRILL_SITE = "drill_site"
    TAILINGS = "tailings"
    STOCKPILE = "stockpile"
    HAUL_ROAD = "haul_road"
    CAMP = "camp"
    VEGETATION = "vegetation"
    WATER_BODY = "water_body"
    UNKNOWN = "unknown"


@dataclass
class FrameAnalysis:
    """Analysis result for a single video frame."""
    frame_index: int
    timestamp: float
    scene_type: SceneType
    description: str
    objects_detected: List[Dict[str, Any]]
    pointing_results: List[PointingResult]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "scene_type": self.scene_type.value,
            "description": self.description,
            "objects_detected": self.objects_detected,
            "pointing_results": [p.to_dict() for p in self.pointing_results],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class TemporalEvent:
    """A temporal event detected in video."""
    event_type: EventType
    start_frame: int
    end_frame: Optional[int]
    start_time: float
    end_time: Optional[float]
    description: str
    location: Optional[Tuple[float, float]]
    confidence: float
    related_objects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_frames(self) -> Optional[int]:
        if self.end_frame is not None:
            return self.end_frame - self.start_frame
        return None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
            "location": self.location,
            "confidence": self.confidence,
            "related_objects": self.related_objects,
            "duration_frames": self.duration_frames,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class VideoUnderstandingResult:
    """Complete result from video understanding pipeline."""
    video_path: str
    duration_seconds: float
    fps: float
    total_frames: int
    frames_analyzed: int
    scene_classification: Dict[SceneType, float]
    frame_analyses: List[FrameAnalysis]
    temporal_events: List[TemporalEvent]
    tracking_results: List[TrackingResult]
    summary: str
    recommendations: List[str]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frames_analyzed": self.frames_analyzed,
            "scene_classification": {k.value: v for k, v in self.scene_classification.items()},
            "frame_analyses": [f.to_dict() for f in self.frame_analyses],
            "temporal_events": [e.to_dict() for e in self.temporal_events],
            "tracking_results": [t.to_dict() for t in self.tracking_results],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }


class VideoUnderstandingPipeline:
    """
    Comprehensive video understanding pipeline for mining exploration.
    
    Combines Molmo2-8B's capabilities with domain-specific analysis
    for geological and mining applications.
    """
    
    def __init__(
        self,
        client: Optional[Molmo2Client] = None,
        config: Optional[Molmo2Config] = None,
    ):
        self.client = client or create_molmo_client(config)
        self.config = config or Molmo2Config()
        self._lock = threading.Lock()
    
    def analyze_video(
        self,
        video_path: str,
        analysis_focus: Optional[str] = None,
        detect_events: bool = True,
        track_objects: bool = True,
        classify_scenes: bool = True,
    ) -> VideoUnderstandingResult:
        """
        Perform comprehensive video analysis.
        
        Args:
            video_path: Path to video file
            analysis_focus: Optional focus area (e.g., "artisanal mining", "equipment")
            detect_events: Whether to detect temporal events
            track_objects: Whether to track objects across frames
            classify_scenes: Whether to classify scene types
        
        Returns:
            VideoUnderstandingResult with complete analysis
        """
        import time
        import cv2
        
        start_time = time.time()
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Build analysis prompt
        base_prompt = self._build_analysis_prompt(analysis_focus)
        
        # Perform main video analysis
        main_result = self.client.analyze_video(
            video_path,
            base_prompt,
            AnalysisType.VIDEO_QA,
        )
        
        # Classify scenes if requested
        scene_classification = {}
        if classify_scenes:
            scene_classification = self._classify_scenes(video_path)
        
        # Detect temporal events if requested
        temporal_events = []
        if detect_events:
            temporal_events = self._detect_temporal_events(video_path, fps)
        
        # Track objects if requested
        tracking_results = []
        if track_objects:
            tracking_results = self._track_objects(video_path)
        
        # Analyze key frames
        frame_analyses = self._analyze_key_frames(video_path, fps)
        
        # Generate summary and recommendations
        summary = self._generate_summary(main_result, temporal_events, tracking_results)
        recommendations = self._generate_recommendations(
            scene_classification, temporal_events, tracking_results
        )
        
        processing_time = time.time() - start_time
        
        return VideoUnderstandingResult(
            video_path=video_path,
            duration_seconds=duration,
            fps=fps,
            total_frames=total_frames,
            frames_analyzed=main_result.frames_analyzed,
            scene_classification=scene_classification,
            frame_analyses=frame_analyses,
            temporal_events=temporal_events,
            tracking_results=tracking_results,
            summary=summary,
            recommendations=recommendations,
            confidence=main_result.confidence,
            processing_time=processing_time,
            metadata={
                "analysis_focus": analysis_focus,
                "model": self.config.model_name,
            },
        )
    
    def _build_analysis_prompt(self, focus: Optional[str] = None) -> str:
        """Build analysis prompt based on focus area."""
        base_prompt = """Analyze this video from a mining/geological exploration perspective.
        
        Identify and describe:
        1. Scene type (open pit, exploration site, processing, etc.)
        2. Geological features visible (rock types, formations, structures)
        3. Mining activity or equipment present
        4. Environmental conditions
        5. Any changes or events occurring over time
        6. Potential safety concerns
        7. Areas of geological interest
        """
        
        if focus:
            base_prompt += f"\n\nPay special attention to: {focus}"
        
        return base_prompt
    
    def _classify_scenes(self, video_path: str) -> Dict[SceneType, float]:
        """Classify scene types in the video."""
        prompt = """Classify the scenes in this video. For each scene type, 
        provide a confidence score (0-1):
        - open_pit: Open pit mining operation
        - underground: Underground mining
        - processing_plant: Mineral processing facility
        - exploration_site: Exploration/prospecting area
        - drill_site: Drilling operation
        - tailings: Tailings storage
        - stockpile: Ore/waste stockpile
        - haul_road: Mining haul roads
        - camp: Mining camp/facilities
        - vegetation: Natural vegetation
        - water_body: Rivers, lakes, ponds
        
        Format: scene_type: confidence"""
        
        result = self.client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        # Parse classification from response
        classification = {}
        for scene_type in SceneType:
            if scene_type.value in result.response.lower():
                # Extract confidence if mentioned
                import re
                pattern = rf'{scene_type.value}[:\s]+(\d+\.?\d*)'
                match = re.search(pattern, result.response.lower())
                if match:
                    classification[scene_type] = float(match.group(1))
                else:
                    classification[scene_type] = 0.5  # Default confidence
        
        return classification
    
    def _detect_temporal_events(self, video_path: str, fps: float) -> List[TemporalEvent]:
        """Detect temporal events in the video."""
        prompt = """Identify all temporal events in this video:
        - When does activity start/stop?
        - When do objects appear/disappear?
        - When is movement detected?
        - When do significant changes occur?
        - When is equipment operating?
        - When are people or vehicles visible?
        
        For each event, report:
        - Event type
        - Start frame/time
        - End frame/time (if applicable)
        - Description
        - Location in frame (if applicable)"""
        
        result = self.client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        events = []
        
        # Parse events from response
        lines = result.response.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Detect event types
            event_type = None
            if 'start' in line_lower and 'activity' in line_lower:
                event_type = EventType.ACTIVITY_START
            elif 'end' in line_lower and 'activity' in line_lower:
                event_type = EventType.ACTIVITY_END
            elif 'appear' in line_lower:
                event_type = EventType.OBJECT_APPEAR
            elif 'disappear' in line_lower:
                event_type = EventType.OBJECT_DISAPPEAR
            elif 'movement' in line_lower or 'moving' in line_lower:
                event_type = EventType.MOVEMENT_DETECTED
            elif 'change' in line_lower:
                event_type = EventType.CHANGE_DETECTED
            elif 'equipment' in line_lower or 'machine' in line_lower:
                event_type = EventType.EQUIPMENT_OPERATION
            elif 'person' in line_lower or 'people' in line_lower or 'worker' in line_lower:
                event_type = EventType.PERSON_DETECTED
            elif 'vehicle' in line_lower or 'truck' in line_lower:
                event_type = EventType.VEHICLE_DETECTED
            
            if event_type:
                # Extract frame/time information
                import re
                frame_match = re.search(r'frame\s*(\d+)', line_lower)
                time_match = re.search(r'(\d+\.?\d*)\s*s(?:ec)?', line_lower)
                
                start_frame = int(frame_match.group(1)) if frame_match else 0
                start_time = float(time_match.group(1)) if time_match else start_frame / fps
                
                events.append(TemporalEvent(
                    event_type=event_type,
                    start_frame=start_frame,
                    end_frame=None,
                    start_time=start_time,
                    end_time=None,
                    description=line.strip(),
                    location=None,
                    confidence=0.7,
                ))
        
        return events
    
    def _track_objects(self, video_path: str) -> List[TrackingResult]:
        """Track objects across video frames."""
        prompt = """Track all significant objects in this video:
        - Mining equipment (excavators, trucks, drills)
        - People/workers
        - Vehicles
        - Moving materials
        
        For each tracked object, report its position in each frame where visible."""
        
        result = self.client.analyze_video(video_path, prompt, AnalysisType.TRACKING)
        return result.tracking_results
    
    def _analyze_key_frames(self, video_path: str, fps: float) -> List[FrameAnalysis]:
        """Analyze key frames from the video."""
        import cv2
        from PIL import Image
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select key frames (beginning, middle, end, plus some in between)
        key_frame_indices = [
            0,
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4,
            total_frames - 1,
        ]
        
        frame_analyses = []
        
        for idx in key_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Analyze frame
                result = self.client.analyze_image(
                    pil_image,
                    "Describe this frame from a mining/geological perspective. "
                    "Identify scene type, objects, and any notable features.",
                    AnalysisType.IMAGE_QA,
                )
                
                # Determine scene type from response
                scene_type = SceneType.UNKNOWN
                response_lower = result.response.lower()
                for st in SceneType:
                    if st.value.replace('_', ' ') in response_lower:
                        scene_type = st
                        break
                
                frame_analyses.append(FrameAnalysis(
                    frame_index=idx,
                    timestamp=idx / fps,
                    scene_type=scene_type,
                    description=result.response,
                    objects_detected=[],
                    pointing_results=result.pointing_results,
                    confidence=result.confidence,
                ))
        
        cap.release()
        return frame_analyses
    
    def _generate_summary(
        self,
        main_result: VideoAnalysisResult,
        events: List[TemporalEvent],
        tracks: List[TrackingResult],
    ) -> str:
        """Generate a summary of the video analysis."""
        summary_parts = [main_result.response]
        
        if events:
            event_summary = f"\n\nDetected {len(events)} temporal events:"
            for event in events[:5]:  # Top 5 events
                event_summary += f"\n- {event.event_type.value}: {event.description[:100]}"
            summary_parts.append(event_summary)
        
        if tracks:
            track_summary = f"\n\nTracked {len(tracks)} objects across frames."
            summary_parts.append(track_summary)
        
        return "\n".join(summary_parts)
    
    def _generate_recommendations(
        self,
        scene_classification: Dict[SceneType, float],
        events: List[TemporalEvent],
        tracks: List[TrackingResult],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Scene-based recommendations
        if scene_classification.get(SceneType.EXPLORATION_SITE, 0) > 0.5:
            recommendations.append(
                "Exploration site detected - consider detailed geological mapping"
            )
        
        if scene_classification.get(SceneType.TAILINGS, 0) > 0.5:
            recommendations.append(
                "Tailings area detected - monitor for environmental compliance"
            )
        
        # Event-based recommendations
        person_events = [e for e in events if e.event_type == EventType.PERSON_DETECTED]
        if person_events:
            recommendations.append(
                f"Personnel activity detected ({len(person_events)} instances) - "
                "verify safety protocols"
            )
        
        equipment_events = [e for e in events if e.event_type == EventType.EQUIPMENT_OPERATION]
        if equipment_events:
            recommendations.append(
                f"Equipment operation detected ({len(equipment_events)} instances) - "
                "log for maintenance tracking"
            )
        
        # Track-based recommendations
        if len(tracks) > 5:
            recommendations.append(
                "High activity level detected - consider more frequent monitoring"
            )
        
        return recommendations
    
    def analyze_for_artisanal_mining(self, video_path: str) -> VideoUnderstandingResult:
        """Specialized analysis for artisanal mining detection."""
        return self.analyze_video(
            video_path,
            analysis_focus="artisanal and small-scale mining activity, informal operations, "
                          "hand tools, sluices, small excavations, temporary camps",
            detect_events=True,
            track_objects=True,
            classify_scenes=True,
        )
    
    def analyze_for_geological_features(self, video_path: str) -> VideoUnderstandingResult:
        """Specialized analysis for geological feature detection."""
        return self.analyze_video(
            video_path,
            analysis_focus="geological features, rock types, formations, structures, "
                          "alteration zones, mineralization indicators, outcrops",
            detect_events=False,
            track_objects=False,
            classify_scenes=True,
        )
    
    def analyze_for_environmental_monitoring(self, video_path: str) -> VideoUnderstandingResult:
        """Specialized analysis for environmental monitoring."""
        return self.analyze_video(
            video_path,
            analysis_focus="environmental conditions, vegetation health, water quality, "
                          "erosion, dust, pollution indicators, wildlife",
            detect_events=True,
            track_objects=False,
            classify_scenes=True,
        )
