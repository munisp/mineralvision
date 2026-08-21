"""
V-JEPA Integration with WALDO and SAM3 Modules.

Provides:
- JEPA-enhanced feature backbone for WALDO detection
- JEPA-guided prompts for SAM3 segmentation
- Feature distillation from JEPA to detection/segmentation models
- Unified inference pipeline combining all models

Decontamination note: this module previously fabricated detection boxes,
segmentation masks, attention points, and distillation losses with
``numpy.random``. Those paths are now LOUD: they either call a real
service (WALDO_SERVICE_URL / SAM3_SERVICE_URL) or raise
:class:`WaldoIntegrationUnavailable`. No random detections, masks, or
losses are ever produced.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from .vjepa_integration import (
    VJEPAConfig,
    VJEPAFeatureExtractor,
    Embedding,
    FaissIndex,
    JEPAUnavailableError,
    create_feature_extractor,
)

logger = logging.getLogger(__name__)


class WaldoIntegrationUnavailable(RuntimeError):
    """Raised when detection/segmentation is requested but no real WALDO
    detector or SAM3 segmenter (local model or HTTP service) is available.

    Deliberately LOUD: callers must never receive fabricated boxes, masks,
    or attention points.
    """


class IntegrationMode(Enum):
    """Mode of integration with downstream models."""
    FEATURE_EXTRACTION = "feature_extraction"
    BACKBONE_REPLACEMENT = "backbone_replacement"
    TEACHER_DISTILLATION = "teacher_distillation"
    PROMPT_GENERATION = "prompt_generation"
    ENSEMBLE = "ensemble"


class DetectionTarget(Enum):
    """Mining-specific detection targets."""
    EQUIPMENT = "equipment"
    VEHICLE = "vehicle"
    PERSONNEL = "personnel"
    GEOLOGICAL_FEATURE = "geological_feature"
    ALTERATION_ZONE = "alteration_zone"
    STRUCTURE = "structure"
    ARTISANAL_ACTIVITY = "artisanal_activity"
    SAFETY_HAZARD = "safety_hazard"
    STOCKPILE = "stockpile"
    INFRASTRUCTURE = "infrastructure"


class SegmentationTarget(Enum):
    """Mining-specific segmentation targets."""
    LITHOLOGY = "lithology"
    ALTERATION = "alteration"
    MINERALIZATION = "mineralization"
    VEIN = "vein"
    CONTACT = "contact"
    FAULT = "fault"
    REGOLITH = "regolith"
    VEGETATION = "vegetation"
    WATER = "water"
    DISTURBED_GROUND = "disturbed_ground"


@dataclass
class DetectionResult:
    """Result of object detection."""
    detection_id: str
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    
    target_type: Optional[DetectionTarget] = None
    jepa_embedding: Optional[List[float]] = None
    jepa_similarity_score: Optional[float] = None
    
    geo_location: Optional[Tuple[float, float]] = None
    timestamp: Optional[datetime] = None
    
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "target_type": self.target_type.value if self.target_type else None,
            "jepa_similarity_score": self.jepa_similarity_score,
            "geo_location": self.geo_location,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "attributes": self.attributes,
        }


@dataclass
class SegmentationResult:
    """Result of image segmentation."""
    segment_id: str
    class_name: str
    confidence: float
    mask: Any
    
    target_type: Optional[SegmentationTarget] = None
    area_pixels: int = 0
    area_percent: float = 0.0
    
    jepa_embedding: Optional[List[float]] = None
    boundary_embedding: Optional[List[float]] = None
    
    geo_polygon: Optional[List[Tuple[float, float]]] = None
    
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "target_type": self.target_type.value if self.target_type else None,
            "area_pixels": self.area_pixels,
            "area_percent": self.area_percent,
            "geo_polygon": self.geo_polygon,
            "attributes": self.attributes,
        }


@dataclass
class JEPAPrompt:
    """JEPA-generated prompt for SAM3."""
    prompt_type: str
    coordinates: Optional[List[Tuple[int, int]]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    text_prompt: Optional[str] = None
    
    embedding: Optional[List[float]] = None
    confidence: float = 0.0
    
    source: str = "jepa_similarity"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_type": self.prompt_type,
            "coordinates": self.coordinates,
            "bbox": self.bbox,
            "text_prompt": self.text_prompt,
            "confidence": self.confidence,
            "source": self.source,
        }


class WALDOJEPAIntegration:
    """Integration of V-JEPA with WALDO detection system."""
    
    def __init__(
        self,
        jepa_extractor: VJEPAFeatureExtractor,
        waldo_model_path: Optional[str] = None,
        integration_mode: IntegrationMode = IntegrationMode.FEATURE_EXTRACTION,
        detection_targets: Optional[List[DetectionTarget]] = None
    ):
        self.jepa_extractor = jepa_extractor
        self.waldo_model_path = waldo_model_path
        self.integration_mode = integration_mode
        self.detection_targets = detection_targets or list(DetectionTarget)
        
        self._waldo_model = None
        self._target_embeddings: Dict[str, List[float]] = {}
        self._embedding_index = FaissIndex(dimension=jepa_extractor.config.embedding_dim)
        
        logger.info(f"Initialized WALDOJEPAIntegration with mode={integration_mode.value}")
    
    def load_waldo_model(self) -> None:
        """
        Load the canonical WALDO RF-DETR detector.

        Imports detector primitives from the canonical
        MineralVision_WALDO_Production_Package via the WALDO_PACKAGE_SRC
        env var (or a relative-path fallback). When the heavy ML stack is
        unavailable, the detector stays None and detection raises
        WaldoIntegrationUnavailable (or uses WALDO_SERVICE_URL) — never
        fake detections.
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
            from waldo_integration.rfdetr_backbone import (
                RFDETRDetector, RFDETRConfig, RFDETRVariant,
            )

            config = RFDETRConfig(
                variant=RFDETRVariant.MEDIUM,
                confidence_threshold=0.5,
                checkpoint_path=self.waldo_model_path,
            )
            self._waldo_model = RFDETRDetector(config)
            logger.info("Canonical WALDO RF-DETR detector loaded")
        except Exception as e:
            logger.warning(
                f"Canonical WALDO detector unavailable ({e}); detection will "
                "use WALDO_SERVICE_URL if set, otherwise raise "
                "WaldoIntegrationUnavailable (no fake detections)"
            )
            self._waldo_model = None
    
    def register_target_examples(
        self,
        target_type: DetectionTarget,
        example_images: List[Any]
    ) -> None:
        """Register example images for a detection target."""
        embeddings = []
        
        for i, image in enumerate(example_images):
            sample = {"frames": [image]}
            emb = self.jepa_extractor.extract_features(sample.get("frames", []))
            emb.imagery_id = f"{target_type.value}_example_{i}"
            emb.metadata = {"target_type": target_type.value}
            embeddings.append(emb)
        
        self._embedding_index.add(embeddings)
        
        import numpy as np
        vectors = np.array([e.vector for e in embeddings])
        mean_embedding = vectors.mean(axis=0).tolist()
        self._target_embeddings[target_type.value] = mean_embedding
        
        logger.info(f"Registered {len(example_images)} examples for {target_type.value}")
    
    def detect(
        self,
        image: Any,
        confidence_threshold: float = 0.5,
        use_jepa_refinement: bool = True
    ) -> List[DetectionResult]:
        """Run detection with JEPA enhancement."""
        import numpy as np
        import hashlib
        
        if self._waldo_model is None:
            self.load_waldo_model()
        
        raw_detections = self._run_waldo_detection(image, confidence_threshold)
        
        if not use_jepa_refinement:
            return raw_detections
        
        refined_detections = []
        
        for det in raw_detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if isinstance(image, np.ndarray):
                crop = image[y1:y2, x1:x2]
            else:
                crop = image
            
            sample = {"frames": [crop]}
            emb = self.jepa_extractor.extract_features(sample.get("frames", []))
            
            det.jepa_embedding = emb.vector
            
            if self._target_embeddings:
                best_similarity = 0.0
                best_target = None
                
                for target_type, target_emb in self._target_embeddings.items():
                    similarity = self._cosine_similarity(emb.vector, target_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = target_type
                
                det.jepa_similarity_score = best_similarity
                
                if best_target:
                    try:
                        det.target_type = DetectionTarget(best_target)
                    except ValueError:
                        pass
            
            refined_detections.append(det)
        
        return refined_detections
    
    def _run_waldo_detection_service(
        self,
        image: Any,
        confidence_threshold: float
    ) -> List[DetectionResult]:
        """Run detection via the real WALDO HTTP service (compose service).

        POSTs the image to ``$WALDO_SERVICE_URL/detect`` and parses real
        detections from the response. Raises on any failure — never
        fabricates boxes.
        """
        import base64
        import numpy as np

        service_url = os.environ["WALDO_SERVICE_URL"].rstrip("/")
        service_token = os.environ.get("WALDO_API_TOKEN", "")
        if not service_token:
            raise WaldoIntegrationUnavailable(
                "WALDO_API_TOKEN is required when WALDO_SERVICE_URL is configured; refusing unauthenticated inference"
            )

        try:
            import httpx
        except ImportError as e:
            raise WaldoIntegrationUnavailable(
                f"httpx is required to call the WALDO service at {service_url}: {e}"
            )

        arr = np.asarray(image)
        payload = {
            "image": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "confidence_threshold": confidence_threshold,
        }

        try:
            response = httpx.post(
                f"{service_url}/detect",
                json=payload,
                headers={"X-Waldo-Service-Token": service_token},
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as e:
            raise WaldoIntegrationUnavailable(
                f"WALDO service at {service_url}/detect failed: {e}. "
                "No detections will be fabricated."
            )

        detections = []
        for i, det in enumerate(body.get("detections", [])):
            conf = float(det.get("confidence", 0.0))
            if conf < confidence_threshold:
                continue
            bbox = det["bbox"]
            detections.append(DetectionResult(
                detection_id=str(det.get("detection_id") or f"waldo_svc_{i}"),
                class_name=str(det.get("class_name", "unknown")),
                confidence=conf,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                attributes={"source": "waldo_service", "service_url": service_url},
            ))
        return detections

    def _run_waldo_detection(
        self,
        image: Any,
        confidence_threshold: float
    ) -> List[DetectionResult]:
        """
        Run a real WALDO detector on an image.

        Order of backends:
        1. Canonical in-process RF-DETR detector (when loadable).
        2. WALDO HTTP service when ``WALDO_SERVICE_URL`` is configured.
        3. Otherwise raise :class:`WaldoIntegrationUnavailable`.

        Fabricated/random detections have been removed; an empty list is
        only ever returned when a real detector genuinely found nothing.
        """
        import numpy as np

        if self._waldo_model is not None:
            canonical_detections = self._waldo_model.detect(np.asarray(image))
        elif os.environ.get("WALDO_SERVICE_URL"):
            return self._run_waldo_detection_service(image, confidence_threshold)
        else:
            raise WaldoIntegrationUnavailable(
                "No WALDO detector available: canonical RF-DETR could not be "
                "loaded and WALDO_SERVICE_URL is not set. Detection is "
                "unavailable — no fake boxes will be produced. Configure the "
                "waldo compose service or provide a loadable checkpoint."
            )

        detections = []
        for det in canonical_detections:
            if det.confidence < confidence_threshold:
                continue
            x1, y1, x2, y2 = det.bbox
            detections.append(DetectionResult(
                detection_id=det.detection_id,
                class_name=det.class_name,
                confidence=float(det.confidence),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                attributes={"source": "waldo_rfdetr_canonical", **(det.metadata or {})},
            ))

        return detections
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def find_similar_detections(
        self,
        query_detection: DetectionResult,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find similar detections in the index."""
        if query_detection.jepa_embedding is None:
            return []
        
        return self._embedding_index.search(query_detection.jepa_embedding, k=k)


class SAM3JEPAIntegration:
    """Integration of V-JEPA with SAM3 segmentation system."""
    
    def __init__(
        self,
        jepa_extractor: VJEPAFeatureExtractor,
        sam3_model_path: Optional[str] = None,
        integration_mode: IntegrationMode = IntegrationMode.PROMPT_GENERATION,
        segmentation_targets: Optional[List[SegmentationTarget]] = None
    ):
        self.jepa_extractor = jepa_extractor
        self.sam3_model_path = sam3_model_path
        self.integration_mode = integration_mode
        self.segmentation_targets = segmentation_targets or list(SegmentationTarget)
        
        self._sam3_model = None
        self._target_embeddings: Dict[str, List[float]] = {}
        self._boundary_embeddings: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized SAM3JEPAIntegration with mode={integration_mode.value}")
    
    def load_sam3_model(self) -> None:
        """Attempt to load a real SAM-family segmentation model.

        Tries the ultralytics SAM backend when a checkpoint is provided.
        When no real model can be loaded, ``self._sam3_model`` stays None
        and segmentation either uses SAM3_SERVICE_URL or raises
        :class:`WaldoIntegrationUnavailable` — the old behavior of
        pretending a model was loaded (a bare dict) has been removed.
        """
        if self._sam3_model is not None:
            return

        if not self.sam3_model_path:
            logger.warning(
                "No SAM3 checkpoint configured; segmentation will use "
                "SAM3_SERVICE_URL if set, otherwise raise "
                "WaldoIntegrationUnavailable"
            )
            return

        try:
            from ultralytics import SAM
            self._sam3_model = SAM(self.sam3_model_path)
            logger.info(f"Loaded real SAM model from {self.sam3_model_path}")
        except Exception as e:
            logger.warning(
                f"Could not load real SAM model from {self.sam3_model_path} "
                f"({e}); segmentation will use SAM3_SERVICE_URL if set, "
                "otherwise raise WaldoIntegrationUnavailable"
            )
            self._sam3_model = None
    
    def register_target_examples(
        self,
        target_type: SegmentationTarget,
        example_images: List[Any],
        example_masks: Optional[List[Any]] = None
    ) -> None:
        """Register example images and masks for a segmentation target."""
        import numpy as np
        
        embeddings = []
        
        for i, image in enumerate(example_images):
            sample = {"frames": [image]}
            emb = self.jepa_extractor.extract_features(sample.get("frames", []))
            embeddings.append(emb.vector)
        
        mean_embedding = np.array(embeddings).mean(axis=0).tolist()
        self._target_embeddings[target_type.value] = mean_embedding
        
        if example_masks:
            boundary_embeddings = []
            for mask in example_masks:
                boundary_emb = self._extract_boundary_features(mask)
                if boundary_emb:
                    boundary_embeddings.append(boundary_emb)
            
            if boundary_embeddings:
                mean_boundary = np.array(boundary_embeddings).mean(axis=0).tolist()
                self._boundary_embeddings[target_type.value] = mean_boundary
        
        logger.info(f"Registered {len(example_images)} examples for {target_type.value}")
    
    def _extract_boundary_features(self, mask: Any) -> Optional[List[float]]:
        """Extract real geometric features from mask boundaries.

        Computes a deterministic 8-dim descriptor (area, perimeter,
        centroid, bbox extents, fill ratio) from the mask via a numpy
        morphological gradient. The previous implementation returned
        ``np.random.randn(256)`` and has been removed.
        """
        import numpy as np

        m = np.asarray(mask)
        if m.size == 0 or m.ndim != 2:
            return None

        binary = (m > 0).astype(np.float32)
        area = float(binary.sum())
        if area == 0.0:
            return None

        # Morphological gradient: boundary = mask - eroded(mask).
        eroded = np.minimum.reduce([
            binary,
            np.roll(binary, 1, axis=0),
            np.roll(binary, -1, axis=0),
            np.roll(binary, 1, axis=1),
            np.roll(binary, -1, axis=1),
        ])
        boundary = binary - eroded
        perimeter = float(boundary.sum())

        ys, xs = np.nonzero(binary)
        h, w = binary.shape
        features = [
            area / (h * w),
            perimeter / (2.0 * (h + w)),
            float(xs.mean()) / w,
            float(ys.mean()) / h,
            float(xs.max() - xs.min() + 1) / w,
            float(ys.max() - ys.min() + 1) / h,
            area / (perimeter + 1e-8),
            perimeter / (area + 1e-8),
        ]
        return features
    
    def generate_prompts(
        self,
        image: Any,
        target_types: Optional[List[SegmentationTarget]] = None,
        num_prompts: int = 5
    ) -> List[JEPAPrompt]:
        """Generate JEPA-guided prompts for SAM3."""
        import numpy as np
        
        target_types = target_types or self.segmentation_targets
        
        sample = {"frames": [image]}
        image_emb = self.jepa_extractor.extract_features(sample.get("frames", []))
        
        prompts = []
        
        for target_type in target_types:
            if target_type.value not in self._target_embeddings:
                continue
            
            target_emb = self._target_embeddings[target_type.value]
            similarity = self._cosine_similarity(image_emb.vector, target_emb)
            
            if similarity > 0.3:
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                else:
                    h, w = 224, 224
                
                attention_points = self._generate_attention_points(
                    image_emb.vector,
                    target_emb,
                    (h, w),
                    num_points=min(3, num_prompts)
                )
                
                for point in attention_points:
                    prompt = JEPAPrompt(
                        prompt_type="point",
                        coordinates=[point],
                        embedding=image_emb.vector,
                        confidence=similarity,
                        source=f"jepa_similarity_{target_type.value}",
                    )
                    prompts.append(prompt)
                
                bbox = self._generate_attention_bbox(
                    image_emb.vector,
                    target_emb,
                    (h, w)
                )
                
                if bbox:
                    prompt = JEPAPrompt(
                        prompt_type="bbox",
                        bbox=bbox,
                        embedding=image_emb.vector,
                        confidence=similarity,
                        source=f"jepa_similarity_{target_type.value}",
                    )
                    prompts.append(prompt)
        
        prompts.sort(key=lambda p: p.confidence, reverse=True)
        return prompts[:num_prompts]
    
    def _generate_attention_points(
        self,
        image_emb: List[float],
        target_emb: List[float],
        image_size: Tuple[int, int],
        num_points: int = 3
    ) -> List[Tuple[int, int]]:
        """Generate attention points based on embedding similarity.

        A global (pooled) embedding carries no spatial information, so
        there is no honest way to localize attention points from it. The
        previous implementation returned random pixel coordinates and has
        been removed. Raises until a real spatial attention/saliency map
        (e.g. from token-level torch_core embeddings) is wired in.
        """
        raise WaldoIntegrationUnavailable(
            "JEPA-guided attention point generation requires a real spatial "
            "attention map (token-level embeddings), which is not available "
            "from a pooled embedding. Random attention points have been "
            "removed — no fabricated prompts will be produced."
        )

    def _generate_attention_bbox(
        self,
        image_emb: List[float],
        target_emb: List[float],
        image_size: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Generate attention bounding box based on embedding similarity.

        See _generate_attention_points: pooled embeddings cannot localize,
        and the previous random-box implementation has been removed.
        """
        raise WaldoIntegrationUnavailable(
            "JEPA-guided attention bbox generation requires a real spatial "
            "attention map (token-level embeddings), which is not available "
            "from a pooled embedding. Random bounding boxes have been "
            "removed — no fabricated prompts will be produced."
        )
    
    def segment(
        self,
        image: Any,
        prompts: Optional[List[JEPAPrompt]] = None,
        auto_generate_prompts: bool = True
    ) -> List[SegmentationResult]:
        """Run segmentation with JEPA-guided prompts."""
        import numpy as np
        import hashlib
        
        if self._sam3_model is None:
            self.load_sam3_model()
        
        if prompts is None and auto_generate_prompts:
            prompts = self.generate_prompts(image)
        
        results = []
        
        for i, prompt in enumerate(prompts or []):
            mask = self._run_sam3_segmentation(image, prompt)
            
            if mask is not None:
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                else:
                    h, w = 224, 224
                
                area_pixels = int(np.sum(mask)) if isinstance(mask, np.ndarray) else 0
                area_percent = area_pixels / (h * w) * 100 if h * w > 0 else 0
                
                segment_id = hashlib.md5(f"seg_{i}_{prompt.prompt_type}".encode()).hexdigest()[:12]
                
                result = SegmentationResult(
                    segment_id=segment_id,
                    class_name=prompt.source.split("_")[-1] if prompt.source else "unknown",
                    confidence=prompt.confidence,
                    mask=mask,
                    area_pixels=area_pixels,
                    area_percent=area_percent,
                    jepa_embedding=prompt.embedding,
                )
                
                results.append(result)
        
        return results
    
    def _run_sam3_segmentation_service(
        self,
        image: Any,
        prompt: JEPAPrompt
    ) -> Any:
        """Run segmentation via the real SAM3 HTTP service.

        POSTs the image and prompt to ``$SAM3_SERVICE_URL/segment`` and
        decodes the returned mask. Raises on any failure — never
        fabricates masks.
        """
        import base64
        import numpy as np

        service_url = os.environ["SAM3_SERVICE_URL"].rstrip("/")

        try:
            import httpx
        except ImportError as e:
            raise WaldoIntegrationUnavailable(
                f"httpx is required to call the SAM3 service at {service_url}: {e}"
            )

        arr = np.asarray(image)
        payload = {
            "image": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "prompt": prompt.to_dict(),
        }

        try:
            response = httpx.post(f"{service_url}/segment", json=payload, timeout=60.0)
            response.raise_for_status()
            body = response.json()
        except Exception as e:
            raise WaldoIntegrationUnavailable(
                f"SAM3 service at {service_url}/segment failed: {e}. "
                "No segmentation mask will be fabricated."
            )

        mask_b64 = body.get("mask")
        mask_shape = body.get("mask_shape")
        if not mask_b64 or not mask_shape:
            raise WaldoIntegrationUnavailable(
                f"SAM3 service at {service_url}/segment returned no mask. "
                "No segmentation mask will be fabricated."
            )

        mask = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape(mask_shape)
        return mask

    def _run_sam3_segmentation(
        self,
        image: Any,
        prompt: JEPAPrompt
    ) -> Optional[Any]:
        """Run real segmentation with a prompt.

        Order of backends:
        1. A real loaded SAM model (ultralytics), when available.
        2. SAM3 HTTP service when ``SAM3_SERVICE_URL`` is configured.
        3. Otherwise raise :class:`WaldoIntegrationUnavailable`.

        The previous implementation drew circles of random radius around
        prompt points and called them masks; that has been removed.
        """
        import numpy as np

        if self._sam3_model is not None:
            predict_kwargs: Dict[str, Any] = {}
            if prompt.prompt_type == "point" and prompt.coordinates:
                predict_kwargs["points"] = [list(p) for p in prompt.coordinates]
                predict_kwargs["labels"] = [1] * len(prompt.coordinates)
            elif prompt.prompt_type == "bbox" and prompt.bbox:
                predict_kwargs["bboxes"] = [list(prompt.bbox)]
            result = self._sam3_model.predict(np.asarray(image), **predict_kwargs)
            masks = getattr(result[0], "masks", None)
            if masks is None or masks.data is None or len(masks.data) == 0:
                return None
            mask = masks.data[0].cpu().numpy().astype(np.uint8)
            return mask

        if os.environ.get("SAM3_SERVICE_URL"):
            return self._run_sam3_segmentation_service(image, prompt)

        raise WaldoIntegrationUnavailable(
            "No SAM3 segmenter available: no real SAM model could be loaded "
            "and SAM3_SERVICE_URL is not set. Segmentation is unavailable — "
            "no fabricated masks will be produced."
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))


class UnifiedMiningVisionPipeline:
    """Unified pipeline combining JEPA, WALDO, and SAM3 for mining vision tasks."""
    
    def __init__(
        self,
        jepa_config: Optional[VJEPAConfig] = None,
        jepa_checkpoint: Optional[str] = None,
        waldo_model_path: Optional[str] = None,
        sam3_model_path: Optional[str] = None
    ):
        self.jepa_config = jepa_config or VJEPAConfig()
        
        self.jepa_extractor = create_feature_extractor(
            config=self.jepa_config,
            checkpoint_path=jepa_checkpoint
        )
        
        self.waldo_integration = WALDOJEPAIntegration(
            jepa_extractor=self.jepa_extractor,
            waldo_model_path=waldo_model_path,
            integration_mode=IntegrationMode.FEATURE_EXTRACTION
        )
        
        self.sam3_integration = SAM3JEPAIntegration(
            jepa_extractor=self.jepa_extractor,
            sam3_model_path=sam3_model_path,
            integration_mode=IntegrationMode.PROMPT_GENERATION
        )
        
        self._embedding_index = FaissIndex(dimension=self.jepa_config.embedding_dim)
        
        logger.info("Initialized UnifiedMiningVisionPipeline")
    
    def process_image(
        self,
        image: Any,
        tasks: Optional[List[str]] = None,
        detection_threshold: float = 0.5,
        segmentation_targets: Optional[List[SegmentationTarget]] = None
    ) -> Dict[str, Any]:
        """Process an image through the unified pipeline."""
        tasks = tasks or ["embedding", "detection", "segmentation"]
        results = {}
        
        if "embedding" in tasks:
            sample = {"frames": [image]}
            embedding = self.jepa_extractor.extract_features(sample.get("frames", []))
            results["embedding"] = embedding.to_dict()
        
        if "detection" in tasks:
            detections = self.waldo_integration.detect(
                image,
                confidence_threshold=detection_threshold,
                use_jepa_refinement=True
            )
            results["detections"] = [d.to_dict() for d in detections]
        
        if "segmentation" in tasks:
            prompts = self.sam3_integration.generate_prompts(
                image,
                target_types=segmentation_targets
            )
            segments = self.sam3_integration.segment(image, prompts)
            results["segments"] = [s.to_dict() for s in segments]
            results["prompts"] = [p.to_dict() for p in prompts]
        
        return results
    
    def process_batch(
        self,
        images: List[Any],
        tasks: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a batch of images."""
        return [self.process_image(img, tasks, **kwargs) for img in images]
    
    def index_images(
        self,
        images: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Index images for similarity search."""
        embeddings = []
        
        for i, image in enumerate(images):
            sample = {"frames": [image]}
            emb = self.jepa_extractor.extract_features(sample.get("frames", []))
            emb.imagery_id = f"indexed_{i}"
            
            if metadata and i < len(metadata):
                emb.metadata = metadata[i]
            
            embeddings.append(emb)
        
        self._embedding_index.add(embeddings)
        return len(embeddings)
    
    def search_similar(
        self,
        query_image: Any,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar images."""
        sample = {"frames": [query_image]}
        query_emb = self.jepa_extractor.extract_features(sample.get("frames", []))
        
        results = self._embedding_index.search(query_emb.vector, k=k, filters=filters)
        
        enriched = []
        for emb_id, distance in results:
            metadata = self._embedding_index._metadata_store.get(emb_id, {})
            enriched.append((emb_id, distance, metadata))
        
        return enriched
    
    def save_state(self, path: str) -> None:
        """Save pipeline state."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self._embedding_index.save(str(save_path / "embedding_index"))
        
        state = {
            "jepa_config": self.jepa_config.to_dict(),
            "waldo_targets": list(self.waldo_integration._target_embeddings.keys()),
            "sam3_targets": list(self.sam3_integration._target_embeddings.keys()),
        }
        
        with open(save_path / "pipeline_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved pipeline state to {path}")
    
    def load_state(self, path: str) -> None:
        """Load pipeline state."""
        load_path = Path(path)
        
        self._embedding_index.load(str(load_path / "embedding_index"))
        
        with open(load_path / "pipeline_state.json", "r") as f:
            state = json.load(f)
        
        logger.info(f"Loaded pipeline state from {path}")


class FeatureDistillation:
    """Distill JEPA features into downstream models."""
    
    def __init__(
        self,
        teacher_extractor: VJEPAFeatureExtractor,
        student_model: Any,
        temperature: float = 4.0,
        alpha: float = 0.5
    ):
        self.teacher = teacher_extractor
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        logger.info(f"Initialized FeatureDistillation with T={temperature}, alpha={alpha}")
    
    def compute_distillation_loss(
        self,
        images: List[Any],
        student_features: Any,
        hard_labels: Optional[Any] = None
    ) -> Dict[str, float]:
        """Compute distillation loss."""
        import numpy as np
        
        teacher_features = []
        for image in images:
            sample = {"frames": [image]}
            emb = self.teacher.extract_features(sample.get("frames", []))
            teacher_features.append(emb.vector)
        
        teacher_features = np.array(teacher_features)
        
        if isinstance(student_features, list):
            student_features = np.array(student_features)
        
        teacher_soft = self._softmax(teacher_features / self.temperature)
        student_soft = self._softmax(student_features / self.temperature)
        
        kl_loss = float(np.mean(np.sum(teacher_soft * np.log(teacher_soft / (student_soft + 1e-8) + 1e-8), axis=1)))
        
        mse_loss = float(np.mean((teacher_features - student_features) ** 2))
        
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * mse_loss
        
        return {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "mse_loss": mse_loss,
        }
    
    def _softmax(self, x: Any) -> Any:
        """Compute softmax."""
        import numpy as np
        
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _extract_student_features(self, batch: List[Any]) -> Any:
        """Extract features from the real student model.

        The student must expose ``extract_features(images)`` or be directly
        callable on a batch of images. Raises honestly otherwise — the
        previous implementation used ``np.random.randn`` as student
        features, which made the distillation loss meaningless.
        """
        import numpy as np

        if self.student is None:
            raise JEPAUnavailableError(
                "Feature distillation requires a real student model; "
                "student_model is None. No random student features will be used."
            )

        if hasattr(self.student, "extract_features"):
            features = self.student.extract_features(batch)
        elif callable(self.student):
            features = self.student(batch)
        else:
            raise JEPAUnavailableError(
                "Feature distillation requires a student model that is "
                "callable or exposes extract_features(images); got "
                f"{type(self.student).__name__}. No random student features "
                "will be used."
            )

        return np.asarray(features, dtype=np.float32)

    def distill_to_detector(
        self,
        detector_backbone: Any,
        training_images: List[Any],
        num_epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """Distill JEPA features to a detection backbone.

        Student features come from the real student model
        (``self.student``); teacher features from the real torch_core-backed
        extractor. Raises :class:`JEPAUnavailableError` if either is
        unavailable — no fabricated losses.
        """
        training_stats = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for i in range(0, len(training_images), 32):
                batch = training_images[i:i + 32]
                
                student_features = self._extract_student_features(batch)
                
                losses = self.compute_distillation_loss(batch, student_features)
                epoch_loss += losses["total_loss"]
            
            avg_loss = epoch_loss / max(1, len(training_images) // 32)
            training_stats.append({"epoch": epoch, "loss": avg_loss})
            
            logger.info(f"Distillation epoch {epoch}: loss={avg_loss:.4f}")
        
        return {
            "num_epochs": num_epochs,
            "final_loss": training_stats[-1]["loss"] if training_stats else None,
            "training_stats": training_stats,
        }


def create_waldo_jepa_integration(
    jepa_checkpoint: Optional[str] = None,
    waldo_model_path: Optional[str] = None,
    detection_targets: Optional[List[str]] = None
) -> WALDOJEPAIntegration:
    """Factory function to create WALDO-JEPA integration."""
    jepa_extractor = create_feature_extractor(checkpoint_path=jepa_checkpoint)
    
    targets = None
    if detection_targets:
        targets = [DetectionTarget(t) for t in detection_targets]
    
    return WALDOJEPAIntegration(
        jepa_extractor=jepa_extractor,
        waldo_model_path=waldo_model_path,
        detection_targets=targets
    )


def create_sam3_jepa_integration(
    jepa_checkpoint: Optional[str] = None,
    sam3_model_path: Optional[str] = None,
    segmentation_targets: Optional[List[str]] = None
) -> SAM3JEPAIntegration:
    """Factory function to create SAM3-JEPA integration."""
    jepa_extractor = create_feature_extractor(checkpoint_path=jepa_checkpoint)
    
    targets = None
    if segmentation_targets:
        targets = [SegmentationTarget(t) for t in segmentation_targets]
    
    return SAM3JEPAIntegration(
        jepa_extractor=jepa_extractor,
        sam3_model_path=sam3_model_path,
        segmentation_targets=targets
    )


def create_unified_pipeline(
    jepa_checkpoint: Optional[str] = None,
    waldo_model_path: Optional[str] = None,
    sam3_model_path: Optional[str] = None
) -> UnifiedMiningVisionPipeline:
    """Factory function to create unified mining vision pipeline."""
    return UnifiedMiningVisionPipeline(
        jepa_checkpoint=jepa_checkpoint,
        waldo_model_path=waldo_model_path,
        sam3_model_path=sam3_model_path
    )
