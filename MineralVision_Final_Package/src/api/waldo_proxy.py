"""
MineralVision WALDO Service Proxy

Provides a proxy endpoint to the WALDO (Water-Air-Land Detection Objects) service
for object detection in aerial/satellite imagery.
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# WALDO service configuration
WALDO_SERVICE_URL = os.getenv("WALDO_SERVICE_URL", "http://localhost:8001")
WALDO_API_KEY = os.getenv("WALDO_API_KEY", "")
WALDO_TIMEOUT = int(os.getenv("WALDO_TIMEOUT", "60"))

router = APIRouter(prefix="/api/waldo", tags=["WALDO Object Detection"])


# Pydantic models
class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_name: str
    class_id: int


class DetectionResult(BaseModel):
    image_id: str
    detections: List[BoundingBox]
    processing_time_ms: float
    model_version: str


class DetectionRequest(BaseModel):
    image_url: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    classes: Optional[List[str]] = None
    max_detections: int = Field(default=100, ge=1, le=1000)


class WALDOStatus(BaseModel):
    status: str
    version: str
    models_loaded: List[str]
    gpu_available: bool


class WALDOProxyClient:
    """Client for communicating with the WALDO service."""
    
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )
        return self._client
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check WALDO service health."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"WALDO health check failed: {e}")
            return {"status": "unavailable", "error": str(e)}
    
    async def detect_objects(
        self,
        image_data: bytes,
        filename: str,
        confidence_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
        max_detections: int = 100
    ) -> DetectionResult:
        """
        Send image to WALDO for object detection.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            confidence_threshold: Minimum confidence for detections
            classes: Optional list of class names to filter
            max_detections: Maximum number of detections to return
            
        Returns:
            DetectionResult with bounding boxes
        """
        try:
            client = await self._get_client()
            
            files = {"image": (filename, image_data)}
            data = {
                "confidence_threshold": str(confidence_threshold),
                "max_detections": str(max_detections)
            }
            if classes:
                data["classes"] = ",".join(classes)
            
            response = await client.post("/detect", files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            return DetectionResult(**result)
            
        except httpx.RequestError as e:
            logger.error(f"WALDO detection request failed: {e}")
            raise HTTPException(status_code=503, detail=f"WALDO service unavailable: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"WALDO detection error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    
    async def detect_from_url(
        self,
        image_url: str,
        confidence_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
        max_detections: int = 100
    ) -> DetectionResult:
        """
        Send image URL to WALDO for object detection.
        
        Args:
            image_url: URL of the image to process
            confidence_threshold: Minimum confidence for detections
            classes: Optional list of class names to filter
            max_detections: Maximum number of detections to return
            
        Returns:
            DetectionResult with bounding boxes
        """
        try:
            client = await self._get_client()
            
            payload = {
                "image_url": image_url,
                "confidence_threshold": confidence_threshold,
                "max_detections": max_detections
            }
            if classes:
                payload["classes"] = classes
            
            response = await client.post("/detect/url", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return DetectionResult(**result)
            
        except httpx.RequestError as e:
            logger.error(f"WALDO URL detection request failed: {e}")
            raise HTTPException(status_code=503, detail=f"WALDO service unavailable: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"WALDO URL detection error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available detection models."""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"WALDO models request failed: {e}")
            raise HTTPException(status_code=503, detail=f"WALDO service unavailable: {e}")
    
    async def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes."""
        try:
            client = await self._get_client()
            response = await client.get("/classes")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"WALDO classes request failed: {e}")
            raise HTTPException(status_code=503, detail=f"WALDO service unavailable: {e}")


# Global client instance
waldo_client = WALDOProxyClient(WALDO_SERVICE_URL, WALDO_API_KEY, WALDO_TIMEOUT)


# API Endpoints
@router.get("/status", response_model=WALDOStatus)
async def get_waldo_status():
    """
    Get WALDO service status.
    
    Returns service health, version, and available models.
    """
    health = await waldo_client.health_check()
    
    if health.get("status") == "unavailable":
        # Return mock status when WALDO is not available
        return WALDOStatus(
            status="unavailable",
            version="unknown",
            models_loaded=[],
            gpu_available=False
        )
    
    return WALDOStatus(
        status=health.get("status", "unknown"),
        version=health.get("version", "unknown"),
        models_loaded=health.get("models", []),
        gpu_available=health.get("gpu", False)
    )


@router.post("/detect", response_model=DetectionResult)
async def detect_objects(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5),
    classes: Optional[str] = Form(default=None),
    max_detections: int = Form(default=100)
):
    """
    Detect objects in an uploaded image.
    
    Args:
        image: Image file to process
        confidence_threshold: Minimum confidence score (0.0-1.0)
        classes: Comma-separated list of class names to filter
        max_detections: Maximum number of detections to return
        
    Returns:
        DetectionResult with bounding boxes and metadata
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    image_data = await image.read()
    
    # Parse classes if provided
    class_list = None
    if classes:
        class_list = [c.strip() for c in classes.split(",")]
    
    # Call WALDO service
    result = await waldo_client.detect_objects(
        image_data=image_data,
        filename=image.filename or "image.jpg",
        confidence_threshold=confidence_threshold,
        classes=class_list,
        max_detections=max_detections
    )
    
    return result


@router.post("/detect/url", response_model=DetectionResult)
async def detect_objects_from_url(request: DetectionRequest):
    """
    Detect objects in an image from URL.
    
    Args:
        request: Detection request with image URL and parameters
        
    Returns:
        DetectionResult with bounding boxes and metadata
    """
    if not request.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    
    result = await waldo_client.detect_from_url(
        image_url=request.image_url,
        confidence_threshold=request.confidence_threshold,
        classes=request.classes,
        max_detections=request.max_detections
    )
    
    return result


@router.get("/models")
async def list_models():
    """
    List available detection models.
    
    Returns list of models with their capabilities and performance metrics.
    """
    try:
        models = await waldo_client.get_available_models()
        return {"models": models}
    except HTTPException:
        # Return default models when WALDO is unavailable
        return {
            "models": [
                {
                    "id": "yolov8-mineral",
                    "name": "YOLOv8 Mineral Detection",
                    "description": "Fine-tuned for mineral exploration imagery",
                    "classes": ["equipment", "vehicle", "structure", "sample_site", "drill_rig"],
                    "status": "unavailable"
                },
                {
                    "id": "yolov8-thermal",
                    "name": "YOLOv8 Thermal Anomaly",
                    "description": "Thermal imagery anomaly detection",
                    "classes": ["heat_source", "thermal_anomaly", "equipment"],
                    "status": "unavailable"
                }
            ]
        }


@router.get("/classes")
async def list_classes():
    """
    List supported object classes.
    
    Returns all object classes that can be detected.
    """
    try:
        classes = await waldo_client.get_supported_classes()
        return {"classes": classes}
    except HTTPException:
        # Return default classes when WALDO is unavailable
        return {
            "classes": [
                "equipment",
                "vehicle",
                "structure",
                "sample_site",
                "drill_rig",
                "person",
                "geological_feature",
                "water_body",
                "vegetation",
                "road",
                "building",
                "heat_source",
                "thermal_anomaly"
            ]
        }


@router.post("/batch")
async def batch_detect(
    images: List[UploadFile] = File(...),
    confidence_threshold: float = Form(default=0.5),
    classes: Optional[str] = Form(default=None),
    max_detections: int = Form(default=100)
):
    """
    Detect objects in multiple images.
    
    Args:
        images: List of image files to process
        confidence_threshold: Minimum confidence score (0.0-1.0)
        classes: Comma-separated list of class names to filter
        max_detections: Maximum number of detections per image
        
    Returns:
        List of DetectionResults
    """
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    class_list = None
    if classes:
        class_list = [c.strip() for c in classes.split(",")]
    
    results = []
    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
            results.append({
                "filename": image.filename,
                "error": "Invalid file type"
            })
            continue
        
        try:
            image_data = await image.read()
            result = await waldo_client.detect_objects(
                image_data=image_data,
                filename=image.filename or "image.jpg",
                confidence_threshold=confidence_threshold,
                classes=class_list,
                max_detections=max_detections
            )
            results.append({
                "filename": image.filename,
                "result": result
            })
        except HTTPException as e:
            results.append({
                "filename": image.filename,
                "error": e.detail
            })
    
    return {"results": results, "total": len(results)}


# Cleanup on shutdown
async def cleanup_waldo_client():
    """Close WALDO client connections."""
    await waldo_client.close()
