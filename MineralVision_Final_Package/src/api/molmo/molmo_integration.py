"""
Molmo2-8B Integration for MineralVision.

Provides core integration with Allen Institute's Molmo2-8B multimodal model
for video understanding, pixel-level grounding, and object tracking.

Features:
- Video and multi-image understanding
- Pixel-level pointing and grounding
- Object tracking across video frames
- Temporal event detection
- Integration with Ollama for local deployment
- HuggingFace Transformers support
"""

import os
import io
import json
import base64
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, BinaryIO
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class Molmo2Backend(Enum):
    """Supported Molmo2 backends."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VLLM = "vllm"
    API = "api"


class AnalysisType(Enum):
    """Types of analysis supported by Molmo2."""
    IMAGE_QA = "image_qa"
    VIDEO_QA = "video_qa"
    POINTING = "pointing"
    TRACKING = "tracking"
    CAPTIONING = "captioning"
    MULTI_IMAGE = "multi_image"
    CHANGE_DETECTION = "change_detection"


@dataclass
class Molmo2Config:
    """Configuration for Molmo2 client."""
    
    backend: Molmo2Backend = Molmo2Backend.HUGGINGFACE
    model_name: str = "allenai/Molmo2-8B"
    device: str = "cuda"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Ollama-specific settings
    ollama_host: str = "http://localhost:11434"
    
    # HuggingFace-specific settings
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Video processing settings
    max_frames: int = 32
    frame_sample_rate: int = 1
    video_resolution: Tuple[int, int] = (384, 384)
    
    # Performance settings
    batch_size: int = 1
    num_workers: int = 4
    cache_embeddings: bool = True
    
    @classmethod
    def from_env(cls) -> "Molmo2Config":
        """Create config from environment variables."""
        backend_str = os.environ.get("MOLMO_BACKEND", "huggingface")
        backend = Molmo2Backend(backend_str)
        
        return cls(
            backend=backend,
            model_name=os.environ.get("MOLMO_MODEL", "allenai/Molmo2-8B"),
            device=os.environ.get("MOLMO_DEVICE", "cuda"),
            max_new_tokens=int(os.environ.get("MOLMO_MAX_TOKENS", "1024")),
            temperature=float(os.environ.get("MOLMO_TEMPERATURE", "0.7")),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    
    @classmethod
    def for_edge_deployment(cls) -> "Molmo2Config":
        """Create config optimized for edge deployment."""
        return cls(
            backend=Molmo2Backend.OLLAMA,
            model_name="molmo2:8b",
            device="cuda",
            max_new_tokens=512,
            max_frames=16,
            load_in_4bit=True,
        )
    
    @classmethod
    def for_high_accuracy(cls) -> "Molmo2Config":
        """Create config optimized for accuracy."""
        return cls(
            backend=Molmo2Backend.HUGGINGFACE,
            model_name="allenai/Molmo2-8B",
            device="cuda",
            max_new_tokens=2048,
            temperature=0.3,
            max_frames=64,
            video_resolution=(512, 512),
        )


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = ""
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "label": self.label,
        }


@dataclass
class PointingResult:
    """Result from pixel-level pointing."""
    x: float
    y: float
    confidence: float
    description: str
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "description": self.description,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
        }


@dataclass
class TrackingResult:
    """Result from object tracking across frames."""
    track_id: str
    object_class: str
    positions: List[Tuple[int, float, float]]  # (frame_idx, x, y)
    bounding_boxes: List[Tuple[int, BoundingBox]]  # (frame_idx, bbox)
    confidence: float
    first_frame: int
    last_frame: int
    
    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory as list of (x, y) positions."""
        return [(x, y) for _, x, y in sorted(self.positions)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_class": self.object_class,
            "positions": self.positions,
            "bounding_boxes": [(idx, bbox.to_dict()) for idx, bbox in self.bounding_boxes],
            "confidence": self.confidence,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_frames": self.duration_frames,
        }


@dataclass
class VideoAnalysisResult:
    """Result from video analysis."""
    analysis_type: AnalysisType
    response: str
    confidence: float
    frames_analyzed: int
    duration_seconds: float
    pointing_results: List[PointingResult] = field(default_factory=list)
    tracking_results: List[TrackingResult] = field(default_factory=list)
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    temporal_events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type.value,
            "response": self.response,
            "confidence": self.confidence,
            "frames_analyzed": self.frames_analyzed,
            "duration_seconds": self.duration_seconds,
            "pointing_results": [p.to_dict() for p in self.pointing_results],
            "tracking_results": [t.to_dict() for t in self.tracking_results],
            "detected_objects": self.detected_objects,
            "temporal_events": self.temporal_events,
            "metadata": self.metadata,
        }


class Molmo2Client(ABC):
    """Abstract base class for Molmo2 clients."""
    
    def __init__(self, config: Molmo2Config):
        self.config = config
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._stats = {
            "images_processed": 0,
            "videos_processed": 0,
            "tokens_generated": 0,
            "total_inference_time": 0.0,
        }
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def analyze_image(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.IMAGE_QA,
    ) -> VideoAnalysisResult:
        """Analyze a single image."""
        pass
    
    @abstractmethod
    def analyze_video(
        self,
        video_path: str,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.VIDEO_QA,
    ) -> VideoAnalysisResult:
        """Analyze a video."""
        pass
    
    @abstractmethod
    def analyze_multi_image(
        self,
        images: List[Union[str, bytes, "PIL.Image.Image"]],
        prompt: str,
    ) -> VideoAnalysisResult:
        """Analyze multiple images together."""
        pass
    
    @abstractmethod
    def point_to_object(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],
        object_description: str,
    ) -> List[PointingResult]:
        """Point to specific objects in an image."""
        pass
    
    @abstractmethod
    def track_object(
        self,
        video_path: str,
        object_description: str,
    ) -> List[TrackingResult]:
        """Track objects across video frames."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return self._stats.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the model is loaded and healthy."""
        return {
            "healthy": self._model is not None,
            "backend": self.config.backend.value,
            "model": self.config.model_name,
            "device": self.config.device,
        }


class HuggingFaceMolmo2Client(Molmo2Client):
    """Molmo2 client using HuggingFace Transformers."""
    
    def _load_model(self) -> None:
        """Load model from HuggingFace."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading Molmo2 model: {self.config.model_name}")
            
            # Determine torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Load model with quantization if specified
            load_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch_dtype,
                "device_map": "auto" if self.config.device == "cuda" else None,
            }
            
            if self.config.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )
            
            if self.config.device != "cuda" and not self.config.load_in_8bit and not self.config.load_in_4bit:
                self._model = self._model.to(self.config.device)
            
            logger.info("Molmo2 model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependencies for HuggingFace backend: {e}")
            raise ImportError(
                "HuggingFace backend requires: pip install transformers torch"
            )
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
    
    def _load_image(self, image: Union[str, bytes, "PIL.Image.Image"]) -> "PIL.Image.Image":
        """Load image from various sources."""
        from PIL import Image
        
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                import requests
                response = requests.get(image)
                return Image.open(io.BytesIO(response.content))
            else:
                return Image.open(image)
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        else:
            return image
    
    def _extract_video_frames(self, video_path: str) -> List["PIL.Image.Image"]:
        """Extract frames from video."""
        try:
            import cv2
            from PIL import Image
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to sample
            sample_indices = list(range(0, total_frames, self.config.frame_sample_rate))
            if len(sample_indices) > self.config.max_frames:
                step = len(sample_indices) // self.config.max_frames
                sample_indices = sample_indices[::step][:self.config.max_frames]
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image = pil_image.resize(self.config.video_resolution)
                    frames.append(pil_image)
            
            cap.release()
            return frames
            
        except ImportError:
            raise ImportError("Video processing requires: pip install opencv-python")
    
    def analyze_image(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.IMAGE_QA,
    ) -> VideoAnalysisResult:
        """Analyze a single image."""
        import time
        
        self._ensure_model_loaded()
        start_time = time.time()
        
        pil_image = self._load_image(image)
        
        # Prepare inputs
        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with self._lock:
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                )
        
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse pointing results if applicable
        pointing_results = []
        if analysis_type == AnalysisType.POINTING:
            pointing_results = self._parse_pointing_response(response)
        
        inference_time = time.time() - start_time
        self._stats["images_processed"] += 1
        self._stats["total_inference_time"] += inference_time
        
        return VideoAnalysisResult(
            analysis_type=analysis_type,
            response=response,
            confidence=0.9,
            frames_analyzed=1,
            duration_seconds=inference_time,
            pointing_results=pointing_results,
            metadata={
                "model": self.config.model_name,
                "inference_time": inference_time,
            },
        )
    
    def analyze_video(
        self,
        video_path: str,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.VIDEO_QA,
    ) -> VideoAnalysisResult:
        """Analyze a video."""
        import time
        
        self._ensure_model_loaded()
        start_time = time.time()
        
        # Extract frames
        frames = self._extract_video_frames(video_path)
        
        # Prepare inputs with multiple frames
        inputs = self._processor(
            text=prompt,
            images=frames,
            return_tensors="pt",
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with self._lock:
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                )
        
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse tracking results if applicable
        tracking_results = []
        if analysis_type == AnalysisType.TRACKING:
            tracking_results = self._parse_tracking_response(response, len(frames))
        
        inference_time = time.time() - start_time
        self._stats["videos_processed"] += 1
        self._stats["total_inference_time"] += inference_time
        
        return VideoAnalysisResult(
            analysis_type=analysis_type,
            response=response,
            confidence=0.85,
            frames_analyzed=len(frames),
            duration_seconds=inference_time,
            tracking_results=tracking_results,
            metadata={
                "model": self.config.model_name,
                "video_path": video_path,
                "frames_extracted": len(frames),
                "inference_time": inference_time,
            },
        )
    
    def analyze_multi_image(
        self,
        images: List[Union[str, bytes, "PIL.Image.Image"]],
        prompt: str,
    ) -> VideoAnalysisResult:
        """Analyze multiple images together."""
        import time
        
        self._ensure_model_loaded()
        start_time = time.time()
        
        pil_images = [self._load_image(img) for img in images]
        
        inputs = self._processor(
            text=prompt,
            images=pil_images,
            return_tensors="pt",
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with self._lock:
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                )
        
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        self._stats["images_processed"] += len(images)
        self._stats["total_inference_time"] += inference_time
        
        return VideoAnalysisResult(
            analysis_type=AnalysisType.MULTI_IMAGE,
            response=response,
            confidence=0.88,
            frames_analyzed=len(images),
            duration_seconds=inference_time,
            metadata={
                "model": self.config.model_name,
                "num_images": len(images),
                "inference_time": inference_time,
            },
        )
    
    def point_to_object(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],
        object_description: str,
    ) -> List[PointingResult]:
        """Point to specific objects in an image."""
        prompt = f"Point to the {object_description} in this image. Provide the exact pixel coordinates."
        
        result = self.analyze_image(
            image,
            prompt,
            analysis_type=AnalysisType.POINTING,
        )
        
        return result.pointing_results
    
    def track_object(
        self,
        video_path: str,
        object_description: str,
    ) -> List[TrackingResult]:
        """Track objects across video frames."""
        prompt = f"Track the {object_description} across all frames. Report position in each frame."
        
        result = self.analyze_video(
            video_path,
            prompt,
            analysis_type=AnalysisType.TRACKING,
        )
        
        return result.tracking_results
    
    def _parse_pointing_response(self, response: str) -> List[PointingResult]:
        """Parse pointing coordinates from model response."""
        import re
        
        results = []
        
        # Look for coordinate patterns like (x, y) or x=123, y=456
        coord_patterns = [
            r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
            r'x[=:]\s*(\d+(?:\.\d+)?)[,\s]+y[=:]\s*(\d+(?:\.\d+)?)',
            r'point[:\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in coord_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    x, y = float(match[0]), float(match[1])
                    results.append(PointingResult(
                        x=x,
                        y=y,
                        confidence=0.85,
                        description=response[:100],
                    ))
                except (ValueError, IndexError):
                    continue
        
        return results
    
    def _parse_tracking_response(self, response: str, num_frames: int) -> List[TrackingResult]:
        """Parse tracking data from model response."""
        import re
        import uuid
        
        results = []
        
        # Look for frame-position patterns
        frame_pattern = r'frame\s*(\d+)[:\s]+\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
        matches = re.findall(frame_pattern, response, re.IGNORECASE)
        
        if matches:
            positions = [(int(m[0]), float(m[1]), float(m[2])) for m in matches]
            
            results.append(TrackingResult(
                track_id=str(uuid.uuid4())[:8],
                object_class="tracked_object",
                positions=positions,
                bounding_boxes=[],
                confidence=0.8,
                first_frame=min(p[0] for p in positions),
                last_frame=max(p[0] for p in positions),
            ))
        
        return results


class OllamaMolmo2Client(Molmo2Client):
    """Molmo2 client using Ollama for local deployment."""
    
    def _load_model(self) -> None:
        """Verify Ollama connection and model availability."""
        try:
            import requests
            
            # Check Ollama is running
            response = requests.get(f"{self.config.ollama_host}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if self.config.model_name not in model_names:
                logger.warning(
                    f"Model {self.config.model_name} not found in Ollama. "
                    f"Available: {model_names}. Will attempt to pull."
                )
            
            self._model = True  # Mark as loaded
            logger.info(f"Ollama backend ready at {self.config.ollama_host}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.config.ollama_host}")
    
    def _ensure_model_loaded(self) -> None:
        """Ensure Ollama connection is established."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
    
    def _encode_image(self, image: Union[str, bytes, "PIL.Image.Image"]) -> str:
        """Encode image to base64 for Ollama API."""
        from PIL import Image
        
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode()
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
    
    def analyze_image(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.IMAGE_QA,
    ) -> VideoAnalysisResult:
        """Analyze image using Ollama."""
        import time
        import requests
        
        self._ensure_model_loaded()
        start_time = time.time()
        
        image_b64 = self._encode_image(image)
        
        response = requests.post(
            f"{self.config.ollama_host}/api/generate",
            json={
                "model": self.config.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_new_tokens,
                },
            },
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        inference_time = time.time() - start_time
        self._stats["images_processed"] += 1
        self._stats["total_inference_time"] += inference_time
        
        return VideoAnalysisResult(
            analysis_type=analysis_type,
            response=response_text,
            confidence=0.85,
            frames_analyzed=1,
            duration_seconds=inference_time,
            metadata={
                "model": self.config.model_name,
                "backend": "ollama",
                "inference_time": inference_time,
            },
        )
    
    def analyze_video(
        self,
        video_path: str,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.VIDEO_QA,
    ) -> VideoAnalysisResult:
        """Analyze video using Ollama (frame-by-frame)."""
        import time
        import cv2
        from PIL import Image
        
        self._ensure_model_loaded()
        start_time = time.time()
        
        # Extract key frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        
        while len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.config.frame_sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_idx += 1
        
        cap.release()
        
        # Analyze frames (Ollama may not support multi-image, so we summarize)
        frame_descriptions = []
        for i, frame in enumerate(frames[:8]):  # Limit for efficiency
            result = self.analyze_image(
                frame,
                f"Frame {i}: {prompt}",
                AnalysisType.IMAGE_QA,
            )
            frame_descriptions.append(f"Frame {i}: {result.response[:200]}")
        
        # Combine descriptions
        combined_response = "\n".join(frame_descriptions)
        
        inference_time = time.time() - start_time
        self._stats["videos_processed"] += 1
        
        return VideoAnalysisResult(
            analysis_type=analysis_type,
            response=combined_response,
            confidence=0.75,
            frames_analyzed=len(frames),
            duration_seconds=inference_time,
            metadata={
                "model": self.config.model_name,
                "backend": "ollama",
                "frames_analyzed": len(frames),
            },
        )
    
    def analyze_multi_image(
        self,
        images: List[Union[str, bytes, "PIL.Image.Image"]],
        prompt: str,
    ) -> VideoAnalysisResult:
        """Analyze multiple images using Ollama."""
        import time
        
        self._ensure_model_loaded()
        start_time = time.time()
        
        # Analyze each image and combine
        results = []
        for i, image in enumerate(images):
            result = self.analyze_image(
                image,
                f"Image {i+1}: {prompt}",
                AnalysisType.IMAGE_QA,
            )
            results.append(f"Image {i+1}: {result.response}")
        
        combined_response = "\n".join(results)
        inference_time = time.time() - start_time
        
        return VideoAnalysisResult(
            analysis_type=AnalysisType.MULTI_IMAGE,
            response=combined_response,
            confidence=0.8,
            frames_analyzed=len(images),
            duration_seconds=inference_time,
            metadata={
                "model": self.config.model_name,
                "backend": "ollama",
                "num_images": len(images),
            },
        )
    
    def point_to_object(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],
        object_description: str,
    ) -> List[PointingResult]:
        """Point to objects using Ollama."""
        prompt = f"Point to the {object_description}. Provide exact pixel coordinates as (x, y)."
        result = self.analyze_image(image, prompt, AnalysisType.POINTING)
        
        # Parse coordinates from response
        import re
        coords = re.findall(r'\((\d+),\s*(\d+)\)', result.response)
        
        return [
            PointingResult(
                x=float(x),
                y=float(y),
                confidence=0.7,
                description=object_description,
            )
            for x, y in coords
        ]
    
    def track_object(
        self,
        video_path: str,
        object_description: str,
    ) -> List[TrackingResult]:
        """Track objects using Ollama (limited support)."""
        prompt = f"Track the {object_description} and report its position in each frame."
        result = self.analyze_video(video_path, prompt, AnalysisType.TRACKING)
        
        # Ollama has limited tracking support, return basic result
        return []


def create_molmo_client(config: Optional[Molmo2Config] = None) -> Molmo2Client:
    """Factory function to create appropriate Molmo2 client."""
    if config is None:
        config = Molmo2Config.from_env()
    
    if config.backend == Molmo2Backend.HUGGINGFACE:
        return HuggingFaceMolmo2Client(config)
    elif config.backend == Molmo2Backend.OLLAMA:
        return OllamaMolmo2Client(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")


# Convenience functions for common operations
def analyze_mining_site_image(
    image_path: str,
    client: Optional[Molmo2Client] = None,
) -> VideoAnalysisResult:
    """Analyze a mining site image for geological features and activity."""
    if client is None:
        client = create_molmo_client()
    
    prompt = """Analyze this mining/exploration site image. Identify:
    1. Geological features (rock types, formations, structures)
    2. Mining activity indicators (equipment, excavations, tailings)
    3. Environmental conditions (vegetation, water, erosion)
    4. Potential mineral indicators (color anomalies, alteration zones)
    5. Safety concerns or hazards
    Provide detailed observations with locations."""
    
    return client.analyze_image(image_path, prompt)


def detect_artisanal_mining(
    video_path: str,
    client: Optional[Molmo2Client] = None,
) -> VideoAnalysisResult:
    """Detect artisanal mining activity in drone video."""
    if client is None:
        client = create_molmo_client()
    
    prompt = """Analyze this drone footage for artisanal mining activity. Look for:
    1. Small-scale excavations or pits
    2. People working with hand tools
    3. Informal processing equipment (sluices, pans)
    4. Temporary structures or camps
    5. Environmental disturbance patterns
    Track any moving objects and report their positions."""
    
    return client.analyze_video(video_path, prompt, AnalysisType.TRACKING)


def compare_site_changes(
    before_image: str,
    after_image: str,
    client: Optional[Molmo2Client] = None,
) -> VideoAnalysisResult:
    """Compare two images to detect site changes over time."""
    if client is None:
        client = create_molmo_client()
    
    prompt = """Compare these two images of the same location taken at different times.
    Identify all changes including:
    1. New excavations or earthworks
    2. Vegetation changes
    3. New structures or equipment
    4. Water body changes
    5. Access road modifications
    Describe each change with its location and significance."""
    
    return client.analyze_multi_image([before_image, after_image], prompt)
