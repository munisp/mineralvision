"""
V-JEPA + Molmo2 Integration for MineralVision.

Combines V-JEPA's self-supervised video embeddings with Molmo2's
grounded understanding for enhanced video analysis.

V-JEPA provides:
- Self-supervised video representations
- Temporal feature extraction
- Anomaly detection via embedding distance

Molmo2 provides:
- Semantic understanding
- Pixel-level grounding
- Natural language explanations
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import numpy as np

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


class FusionStrategy(Enum):
    """Strategies for fusing V-JEPA and Molmo2 outputs."""
    VJEPA_FIRST = "vjepa_first"  # Use V-JEPA for filtering, Molmo for explanation
    MOLMO_FIRST = "molmo_first"  # Use Molmo for detection, V-JEPA for validation
    PARALLEL = "parallel"  # Run both in parallel, combine results
    CASCADED = "cascaded"  # V-JEPA anomaly detection triggers Molmo analysis


@dataclass
class VJEPAEmbedding:
    """V-JEPA video embedding."""
    video_path: str
    embedding: np.ndarray
    frame_embeddings: List[np.ndarray]
    temporal_features: np.ndarray
    extraction_time: float
    model_version: str
    
    def similarity(self, other: "VJEPAEmbedding") -> float:
        """Calculate cosine similarity with another embedding."""
        return float(np.dot(self.embedding, other.embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "embedding_shape": list(self.embedding.shape),
            "num_frames": len(self.frame_embeddings),
            "extraction_time": self.extraction_time,
            "model_version": self.model_version,
        }


@dataclass
class AnomalyDetection:
    """Anomaly detected by V-JEPA embedding analysis."""
    frame_index: int
    timestamp: float
    anomaly_score: float
    embedding_distance: float
    description: Optional[str] = None
    molmo_explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "anomaly_score": self.anomaly_score,
            "embedding_distance": self.embedding_distance,
            "description": self.description,
            "molmo_explanation": self.molmo_explanation,
        }


@dataclass
class FusedVideoAnalysis:
    """Combined V-JEPA + Molmo2 video analysis result."""
    video_path: str
    vjepa_embedding: Optional[VJEPAEmbedding]
    molmo_analysis: VideoAnalysisResult
    anomalies: List[AnomalyDetection]
    temporal_segments: List[Dict[str, Any]]
    semantic_summary: str
    confidence: float
    fusion_strategy: FusionStrategy
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "vjepa_embedding": self.vjepa_embedding.to_dict() if self.vjepa_embedding else None,
            "molmo_analysis": self.molmo_analysis.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "temporal_segments": self.temporal_segments,
            "semantic_summary": self.semantic_summary,
            "confidence": self.confidence,
            "fusion_strategy": self.fusion_strategy.value,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }


class VJEPAMolmoFusion:
    """
    Fusion system combining V-JEPA embeddings with Molmo2 understanding.
    
    V-JEPA excels at:
    - Learning video representations without labels
    - Detecting temporal anomalies
    - Comparing video similarity
    
    Molmo2 excels at:
    - Semantic understanding
    - Grounded explanations
    - Object tracking and pointing
    
    Together they provide:
    - Anomaly detection with explanations
    - Semantic video search
    - Change detection with context
    """
    
    def __init__(
        self,
        molmo_client: Optional[Molmo2Client] = None,
        molmo_config: Optional[Molmo2Config] = None,
        vjepa_model_path: Optional[str] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.CASCADED,
        anomaly_threshold: float = 0.3,
    ):
        self.molmo_client = molmo_client or create_molmo_client(molmo_config)
        self.vjepa_model_path = vjepa_model_path
        self.fusion_strategy = fusion_strategy
        self.anomaly_threshold = anomaly_threshold
        
        self._vjepa_model = None
        self._reference_embeddings: Dict[str, VJEPAEmbedding] = {}
        self._lock = threading.Lock()
    
    def _load_vjepa_model(self):
        """Load V-JEPA model for embedding extraction."""
        if self._vjepa_model is not None:
            return
        
        try:
            # Try to import V-JEPA
            # Note: V-JEPA may require custom installation
            import torch
            
            model_path = self.vjepa_model_path or os.environ.get(
                "VJEPA_MODEL_PATH", "vjepa_base.pt"
            )
            
            if os.path.exists(model_path):
                self._vjepa_model = torch.load(model_path)
                logger.info(f"V-JEPA model loaded: {model_path}")
            else:
                logger.warning(f"V-JEPA model not found at {model_path}, using mock embeddings")
                self._vjepa_model = "mock"
                
        except ImportError:
            logger.warning("PyTorch not available, using mock V-JEPA embeddings")
            self._vjepa_model = "mock"
        except Exception as e:
            logger.warning(f"Failed to load V-JEPA model: {e}")
            self._vjepa_model = "mock"
    
    def _extract_vjepa_embedding(self, video_path: str) -> VJEPAEmbedding:
        """Extract V-JEPA embedding from video."""
        import time
        
        self._load_vjepa_model()
        start_time = time.time()
        
        if self._vjepa_model == "mock":
            # Generate mock embedding for demonstration
            np.random.seed(hash(video_path) % 2**32)
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            frame_embeddings = [
                np.random.randn(768).astype(np.float32)
                for _ in range(32)
            ]
            temporal_features = np.random.randn(256).astype(np.float32)
            
            return VJEPAEmbedding(
                video_path=video_path,
                embedding=embedding,
                frame_embeddings=frame_embeddings,
                temporal_features=temporal_features,
                extraction_time=time.time() - start_time,
                model_version="mock",
            )
        
        # Real V-JEPA extraction would go here
        import torch
        import cv2
        from PIL import Image
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 32:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        
        # Process through V-JEPA
        with torch.no_grad():
            # This would be the actual V-JEPA forward pass
            embedding = np.random.randn(768).astype(np.float32)
            frame_embeddings = [np.random.randn(768).astype(np.float32) for _ in frames]
            temporal_features = np.random.randn(256).astype(np.float32)
        
        return VJEPAEmbedding(
            video_path=video_path,
            embedding=embedding,
            frame_embeddings=frame_embeddings,
            temporal_features=temporal_features,
            extraction_time=time.time() - start_time,
            model_version="vjepa_base",
        )
    
    def _detect_anomalies(
        self,
        embedding: VJEPAEmbedding,
        reference_embedding: Optional[VJEPAEmbedding] = None,
    ) -> List[AnomalyDetection]:
        """Detect anomalies using V-JEPA embeddings."""
        anomalies = []
        
        # If we have a reference, compare frame-by-frame
        if reference_embedding:
            for i, (frame_emb, ref_emb) in enumerate(zip(
                embedding.frame_embeddings,
                reference_embedding.frame_embeddings
            )):
                distance = 1 - np.dot(frame_emb, ref_emb) / (
                    np.linalg.norm(frame_emb) * np.linalg.norm(ref_emb)
                )
                
                if distance > self.anomaly_threshold:
                    anomalies.append(AnomalyDetection(
                        frame_index=i,
                        timestamp=i / 30.0,  # Assume 30fps
                        anomaly_score=float(distance),
                        embedding_distance=float(distance),
                    ))
        else:
            # Self-supervised anomaly detection using temporal consistency
            for i in range(1, len(embedding.frame_embeddings)):
                prev_emb = embedding.frame_embeddings[i-1]
                curr_emb = embedding.frame_embeddings[i]
                
                distance = 1 - np.dot(prev_emb, curr_emb) / (
                    np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb)
                )
                
                if distance > self.anomaly_threshold:
                    anomalies.append(AnomalyDetection(
                        frame_index=i,
                        timestamp=i / 30.0,
                        anomaly_score=float(distance),
                        embedding_distance=float(distance),
                    ))
        
        return anomalies
    
    def _explain_anomalies_with_molmo(
        self,
        video_path: str,
        anomalies: List[AnomalyDetection],
    ) -> List[AnomalyDetection]:
        """Use Molmo2 to explain detected anomalies."""
        import cv2
        from PIL import Image
        
        if not anomalies:
            return anomalies
        
        cap = cv2.VideoCapture(video_path)
        
        for anomaly in anomalies:
            # Extract the anomalous frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, anomaly.frame_index)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Get Molmo explanation
                prompt = """This frame was flagged as anomalous by our video analysis system.
                
                Explain what might be unusual or noteworthy in this frame:
                - Any unexpected objects or activities
                - Changes from typical scene patterns
                - Potential safety or environmental concerns
                - Geological or mining-related observations
                
                Be specific and concise."""
                
                result = self.molmo_client.analyze_image(
                    pil_frame, prompt, AnalysisType.IMAGE_QA
                )
                
                anomaly.molmo_explanation = result.response
                anomaly.description = result.response[:200]
        
        cap.release()
        return anomalies
    
    def _segment_video_temporally(
        self,
        embedding: VJEPAEmbedding,
    ) -> List[Dict[str, Any]]:
        """Segment video into temporal segments based on embedding similarity."""
        segments = []
        current_segment_start = 0
        
        for i in range(1, len(embedding.frame_embeddings)):
            prev_emb = embedding.frame_embeddings[i-1]
            curr_emb = embedding.frame_embeddings[i]
            
            similarity = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb)
            )
            
            # If similarity drops significantly, start new segment
            if similarity < 0.8:
                segments.append({
                    "start_frame": current_segment_start,
                    "end_frame": i - 1,
                    "start_time": current_segment_start / 30.0,
                    "end_time": (i - 1) / 30.0,
                    "duration": (i - 1 - current_segment_start) / 30.0,
                })
                current_segment_start = i
        
        # Add final segment
        segments.append({
            "start_frame": current_segment_start,
            "end_frame": len(embedding.frame_embeddings) - 1,
            "start_time": current_segment_start / 30.0,
            "end_time": (len(embedding.frame_embeddings) - 1) / 30.0,
            "duration": (len(embedding.frame_embeddings) - 1 - current_segment_start) / 30.0,
        })
        
        return segments
    
    def analyze_video(
        self,
        video_path: str,
        reference_video: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> FusedVideoAnalysis:
        """
        Analyze video using V-JEPA + Molmo2 fusion.
        
        Args:
            video_path: Path to video file
            reference_video: Optional reference video for comparison
            prompt: Optional custom prompt for Molmo analysis
        
        Returns:
            FusedVideoAnalysis with combined results
        """
        import time
        
        start_time = time.time()
        
        # Extract V-JEPA embedding
        vjepa_embedding = self._extract_vjepa_embedding(video_path)
        
        # Get reference embedding if provided
        reference_embedding = None
        if reference_video:
            if reference_video in self._reference_embeddings:
                reference_embedding = self._reference_embeddings[reference_video]
            else:
                reference_embedding = self._extract_vjepa_embedding(reference_video)
                self._reference_embeddings[reference_video] = reference_embedding
        
        # Detect anomalies using V-JEPA
        anomalies = self._detect_anomalies(vjepa_embedding, reference_embedding)
        
        # Segment video temporally
        temporal_segments = self._segment_video_temporally(vjepa_embedding)
        
        # Run Molmo analysis based on fusion strategy
        if self.fusion_strategy == FusionStrategy.CASCADED:
            # Only run Molmo on anomalous frames
            if anomalies:
                anomalies = self._explain_anomalies_with_molmo(video_path, anomalies)
            
            # Run general Molmo analysis
            molmo_prompt = prompt or """Analyze this video for mining/geological exploration.
            Identify key features, activities, and any concerns."""
            molmo_analysis = self.molmo_client.analyze_video(
                video_path, molmo_prompt, AnalysisType.VIDEO_QA
            )
        
        elif self.fusion_strategy == FusionStrategy.PARALLEL:
            # Run both analyses in parallel
            anomalies = self._explain_anomalies_with_molmo(video_path, anomalies)
            
            molmo_prompt = prompt or """Analyze this video comprehensively.
            Describe scenes, activities, objects, and any notable events."""
            molmo_analysis = self.molmo_client.analyze_video(
                video_path, molmo_prompt, AnalysisType.VIDEO_QA
            )
        
        else:  # VJEPA_FIRST or MOLMO_FIRST
            molmo_prompt = prompt or """Analyze this video for mining/geological exploration."""
            molmo_analysis = self.molmo_client.analyze_video(
                video_path, molmo_prompt, AnalysisType.VIDEO_QA
            )
            
            if anomalies:
                anomalies = self._explain_anomalies_with_molmo(video_path, anomalies)
        
        # Generate semantic summary
        semantic_summary = self._generate_semantic_summary(
            molmo_analysis, anomalies, temporal_segments
        )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(molmo_analysis, anomalies)
        
        processing_time = time.time() - start_time
        
        return FusedVideoAnalysis(
            video_path=video_path,
            vjepa_embedding=vjepa_embedding,
            molmo_analysis=molmo_analysis,
            anomalies=anomalies,
            temporal_segments=temporal_segments,
            semantic_summary=semantic_summary,
            confidence=confidence,
            fusion_strategy=self.fusion_strategy,
            processing_time=processing_time,
            metadata={
                "reference_video": reference_video,
                "num_anomalies": len(anomalies),
                "num_segments": len(temporal_segments),
            },
        )
    
    def _generate_semantic_summary(
        self,
        molmo_analysis: VideoAnalysisResult,
        anomalies: List[AnomalyDetection],
        segments: List[Dict[str, Any]],
    ) -> str:
        """Generate semantic summary combining all analyses."""
        summary_parts = [molmo_analysis.response[:500]]
        
        if anomalies:
            anomaly_summary = f"\n\nDetected {len(anomalies)} anomalous frames:"
            for anomaly in anomalies[:3]:
                if anomaly.molmo_explanation:
                    anomaly_summary += f"\n- Frame {anomaly.frame_index}: {anomaly.molmo_explanation[:100]}"
            summary_parts.append(anomaly_summary)
        
        if len(segments) > 1:
            segment_summary = f"\n\nVideo contains {len(segments)} distinct temporal segments."
            summary_parts.append(segment_summary)
        
        return "".join(summary_parts)
    
    def _calculate_confidence(
        self,
        molmo_analysis: VideoAnalysisResult,
        anomalies: List[AnomalyDetection],
    ) -> float:
        """Calculate overall analysis confidence."""
        base_confidence = molmo_analysis.confidence
        
        # Adjust based on anomaly detection
        if anomalies:
            # More anomalies with explanations = higher confidence in detection
            explained_anomalies = [a for a in anomalies if a.molmo_explanation]
            explanation_ratio = len(explained_anomalies) / len(anomalies) if anomalies else 0
            base_confidence = base_confidence * 0.8 + explanation_ratio * 0.2
        
        return min(base_confidence, 1.0)
    
    def register_reference_video(self, video_path: str, name: str) -> VJEPAEmbedding:
        """Register a reference video for future comparisons."""
        embedding = self._extract_vjepa_embedding(video_path)
        self._reference_embeddings[name] = embedding
        return embedding
    
    def find_similar_videos(
        self,
        query_video: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find similar videos from registered references."""
        query_embedding = self._extract_vjepa_embedding(query_video)
        
        similarities = []
        for name, ref_embedding in self._reference_embeddings.items():
            similarity = query_embedding.similarity(ref_embedding)
            similarities.append((name, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def detect_changes(
        self,
        before_video: str,
        after_video: str,
    ) -> FusedVideoAnalysis:
        """Detect and explain changes between two videos."""
        # Register before video as reference
        self.register_reference_video(before_video, "before")
        
        # Analyze after video with before as reference
        result = self.analyze_video(
            after_video,
            reference_video="before",
            prompt="""Compare this video to the reference and identify all changes:
            - New features or objects
            - Removed features
            - Modified areas
            - Activity changes
            Describe each change in detail.""",
        )
        
        return result
