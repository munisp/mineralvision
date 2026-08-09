"""
SAM3 API Endpoints for MineralVision

FastAPI endpoints for:
- Image segmentation (text, exemplar, point prompts)
- Video tracking
- Fine-tuning job management
- Model/adapter management
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import base64
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/sam3", tags=["SAM3 Segmentation"])


# Request/Response Models

class TextSegmentationRequest(BaseModel):
    """Request for text-based segmentation."""
    text_prompt: str = Field(..., description="Text description of target to segment")
    concept: Optional[str] = Field(None, description="Domain concept name for vocabulary lookup")
    modality: str = Field("drillcore", description="Imaging modality")
    return_masks: bool = Field(True, description="Whether to return mask data")
    adapter_id: Optional[str] = Field(None, description="Specific adapter to use")


class PointSegmentationRequest(BaseModel):
    """Request for point-based segmentation."""
    points: List[Dict[str, int]] = Field(..., description="List of {x, y} coordinates")
    labels: List[int] = Field(..., description="Labels for points (1=foreground, 0=background)")
    modality: str = Field("drillcore", description="Imaging modality")


class ExemplarSegmentationRequest(BaseModel):
    """Request for exemplar-based segmentation."""
    exemplar_box: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2] on exemplar")
    modality: str = Field("drillcore", description="Imaging modality")


class SegmentationResponse(BaseModel):
    """Response from segmentation endpoint."""
    success: bool
    mask_count: int
    scores: List[float]
    concept: str
    prompt_type: str
    masks_base64: Optional[List[str]] = None
    metadata: Dict[str, Any] = {}


class TrainingJobRequest(BaseModel):
    """Request to start a fine-tuning job."""
    job_name: str = Field(..., description="Name for the training job")
    dataset_path: str = Field(..., description="Path to training dataset")
    modality: str = Field("drillcore", description="Target modality")
    strategy: str = Field("lora", description="Training strategy (lora, full, adapter)")
    concepts: List[str] = Field(default_factory=list, description="Concepts to train")
    
    # Training parameters
    learning_rate: float = Field(1e-4, description="Learning rate")
    batch_size: int = Field(4, description="Batch size")
    num_epochs: int = Field(10, description="Number of epochs")
    lora_rank: int = Field(16, description="LoRA rank")


class TrainingJobResponse(BaseModel):
    """Response for training job."""
    job_id: str
    status: str
    message: str
    config: Dict[str, Any] = {}


class AdapterRegistrationRequest(BaseModel):
    """Request to register an adapter."""
    name: str = Field(..., description="Adapter name")
    version: str = Field(..., description="Semantic version")
    modality: str = Field(..., description="Target modality")
    concepts: List[str] = Field(..., description="Concepts this adapter handles")
    weights_path: str = Field(..., description="Path to adapter weights")
    description: str = Field("", description="Human-readable description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class AdapterResponse(BaseModel):
    """Response for adapter operations."""
    adapter_id: str
    name: str
    version: str
    modality: str
    concepts: List[str]
    status: str
    created_at: str


class ConceptListResponse(BaseModel):
    """Response listing available concepts."""
    modality: str
    concepts: List[Dict[str, Any]]


# Global instances (initialized on startup)
_segmenter = None
_registry = None
_training_jobs: Dict[str, Dict[str, Any]] = {}


def get_segmenter():
    """Get or create SAM3 segmenter instance."""
    global _segmenter
    if _segmenter is None:
        from .sam3_segmenter import create_sam3_segmenter
        _segmenter = create_sam3_segmenter()
    return _segmenter


def get_registry():
    """Get or create model registry instance."""
    global _registry
    if _registry is None:
        from .model_registry import create_model_registry
        _registry = create_model_registry()
    return _registry


# Segmentation Endpoints

@router.post("/segment/text", response_model=SegmentationResponse)
async def segment_by_text(
    request: TextSegmentationRequest,
    image: UploadFile = File(..., description="Image to segment")
):
    """
    Segment image using text prompt.
    
    Supports domain-specific concepts for geology/mining:
    - Drillcore: vein, sulfide_zone, alteration_halo, etc.
    - Thin section: quartz, feldspar, pyrite, etc.
    - UAV/Satellite: lineament, fault_trace, gossan, etc.
    - Geophysics: magnetic_high, gravity_anomaly, etc.
    """
    try:
        segmenter = get_segmenter()
        
        # Load adapter if specified
        if request.adapter_id:
            registry = get_registry()
            adapter = registry.get_adapter(request.adapter_id)
            if adapter and adapter.weights_path:
                segmenter._load_adapter(adapter.weights_path)
        
        # Read image
        image_data = await image.read()
        
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        # Perform segmentation
        result = segmenter.segment_by_text(
            image=tmp_path,
            text_prompt=request.text_prompt,
            concept=request.concept
        )
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        # Encode masks if requested
        masks_base64 = None
        if request.return_masks and result.masks:
            masks_base64 = []
            for mask in result.masks:
                try:
                    from PIL import Image
                    import numpy as np
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    buffer = io.BytesIO()
                    mask_img.save(buffer, format="PNG")
                    masks_base64.append(base64.b64encode(buffer.getvalue()).decode())
                except Exception:
                    pass
        
        return SegmentationResponse(
            success=True,
            mask_count=len(result.masks),
            scores=result.scores,
            concept=result.concept,
            prompt_type=result.prompt_type,
            masks_base64=masks_base64,
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Text segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment/point", response_model=SegmentationResponse)
async def segment_by_point(
    request: PointSegmentationRequest,
    image: UploadFile = File(..., description="Image to segment")
):
    """
    Segment image using point prompts.
    
    Provide foreground (label=1) and background (label=0) points
    to guide segmentation.
    """
    try:
        segmenter = get_segmenter()
        
        # Read image
        image_data = await image.read()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        # Convert points
        points = [(p["x"], p["y"]) for p in request.points]
        
        result = segmenter.segment_by_point(
            image=tmp_path,
            points=points,
            labels=request.labels
        )
        
        Path(tmp_path).unlink(missing_ok=True)
        
        return SegmentationResponse(
            success=True,
            mask_count=len(result.masks),
            scores=result.scores,
            concept=result.concept,
            prompt_type=result.prompt_type,
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Point segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment/exemplar", response_model=SegmentationResponse)
async def segment_by_exemplar(
    request: ExemplarSegmentationRequest,
    image: UploadFile = File(..., description="Target image to segment"),
    exemplar: UploadFile = File(..., description="Exemplar image showing target concept")
):
    """
    Segment image using exemplar image prompt.
    
    Provide an example image showing the target concept,
    and SAM3 will find similar regions in the target image.
    """
    try:
        segmenter = get_segmenter()
        
        # Read images
        image_data = await image.read()
        exemplar_data = await exemplar.read()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
            tmp1.write(image_data)
            image_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
            tmp2.write(exemplar_data)
            exemplar_path = tmp2.name
        
        # Convert box if provided
        exemplar_box = None
        if request.exemplar_box and len(request.exemplar_box) == 4:
            exemplar_box = tuple(request.exemplar_box)
        
        result = segmenter.segment_by_exemplar(
            image=image_path,
            exemplar_image=exemplar_path,
            exemplar_box=exemplar_box
        )
        
        Path(image_path).unlink(missing_ok=True)
        Path(exemplar_path).unlink(missing_ok=True)
        
        return SegmentationResponse(
            success=True,
            mask_count=len(result.masks),
            scores=result.scores,
            concept=result.concept,
            prompt_type=result.prompt_type,
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Exemplar segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Concept Endpoints

@router.get("/concepts/{modality}", response_model=ConceptListResponse)
async def list_concepts(modality: str):
    """
    List available domain concepts for a modality.
    
    Modalities:
    - drillcore: Core tray photos
    - thin_section: Microscopy images
    - uav_ortho: UAV orthomosaics
    - satellite: Satellite imagery
    - geophysics: Geophysics rasters
    """
    try:
        segmenter = get_segmenter()
        
        from .sam3_segmenter import Modality
        try:
            mod = Modality(modality)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid modality: {modality}")
        
        concepts = segmenter.get_concepts_for_modality(mod)
        
        return ConceptListResponse(
            modality=modality,
            concepts=[
                {
                    "name": c.name,
                    "text_prompts": c.text_prompts,
                    "description": c.description,
                    "synonyms": c.synonyms
                }
                for c in concepts
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training Endpoints

@router.post("/training/start", response_model=TrainingJobResponse)
async def start_training_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a fine-tuning job for domain adaptation.
    
    Training strategies:
    - lora: Parameter-efficient LoRA fine-tuning (recommended)
    - full: Full model fine-tuning (requires more resources)
    - adapter: Adapter-based fine-tuning
    """
    try:
        from .fine_tuning import SAM3FineTuner, TrainingConfig, TrainingStrategy, GeologyDatasetConfig
        from .data_preparation import GeologySegmentationDataset
        import uuid
        
        job_id = f"train_{uuid.uuid4().hex[:8]}"
        
        # Create training config
        config = TrainingConfig(
            strategy=TrainingStrategy(request.strategy),
            modality=request.modality,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            lora_rank=request.lora_rank,
            checkpoint_dir=f"./checkpoints/{job_id}"
        )
        
        # Store job info
        _training_jobs[job_id] = {
            "status": "starting",
            "config": config.to_dict(),
            "progress": 0,
            "message": "Initializing training job"
        }
        
        # Start training in background
        async def run_training():
            try:
                _training_jobs[job_id]["status"] = "running"
                _training_jobs[job_id]["message"] = "Training in progress"
                
                # Load dataset
                dataset_config = GeologyDatasetConfig(
                    name=request.job_name,
                    modality=request.modality,
                    concepts=request.concepts,
                    image_dir=f"{request.dataset_path}/images",
                    mask_dir=f"{request.dataset_path}/masks"
                )
                dataset = GeologySegmentationDataset(dataset_config)
                
                # Create fine-tuner and train
                fine_tuner = SAM3FineTuner(config)
                result = fine_tuner.train(dataset)
                
                _training_jobs[job_id]["status"] = "completed"
                _training_jobs[job_id]["result"] = result
                _training_jobs[job_id]["message"] = "Training completed successfully"
            except Exception as e:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["message"] = str(e)
        
        background_tasks.add_task(run_training)
        
        return TrainingJobResponse(
            job_id=job_id,
            status="starting",
            message="Training job started",
            config=config.to_dict()
        )
    except Exception as e:
        logger.error(f"Failed to start training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/{job_id}", response_model=TrainingJobResponse)
async def get_training_status(job_id: str):
    """Get status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _training_jobs[job_id]
    return TrainingJobResponse(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        config=job.get("config", {})
    )


@router.get("/training", response_model=List[TrainingJobResponse])
async def list_training_jobs():
    """List all training jobs."""
    return [
        TrainingJobResponse(
            job_id=job_id,
            status=job["status"],
            message=job["message"],
            config=job.get("config", {})
        )
        for job_id, job in _training_jobs.items()
    ]


# Adapter Management Endpoints

@router.post("/adapters/register", response_model=AdapterResponse)
async def register_adapter(request: AdapterRegistrationRequest):
    """
    Register a fine-tuned adapter in the model registry.
    
    After training, register the adapter to make it available
    for inference.
    """
    try:
        registry = get_registry()
        
        from .model_registry import AdapterType
        
        metadata = registry.register_adapter(
            name=request.name,
            version=request.version,
            adapter_type=AdapterType.LORA,
            modality=request.modality,
            concepts=request.concepts,
            base_model="sam3-base",
            weights_path=request.weights_path,
            description=request.description,
            tags=request.tags
        )
        
        return AdapterResponse(
            adapter_id=metadata.adapter_id,
            name=metadata.name,
            version=metadata.version,
            modality=metadata.modality,
            concepts=metadata.concepts,
            status=metadata.status.value,
            created_at=metadata.created_at
        )
    except Exception as e:
        logger.error(f"Failed to register adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters", response_model=List[AdapterResponse])
async def list_adapters(
    modality: Optional[str] = None,
    status: Optional[str] = None,
    concept: Optional[str] = None
):
    """
    List registered adapters with optional filtering.
    
    Filter by modality, status, or concept to find relevant adapters.
    """
    try:
        registry = get_registry()
        
        from .model_registry import ModelStatus
        
        status_filter = None
        if status:
            try:
                status_filter = ModelStatus(status)
            except ValueError:
                pass
        
        adapters = registry.list_adapters(
            modality=modality,
            status=status_filter,
            concept=concept
        )
        
        return [
            AdapterResponse(
                adapter_id=a.adapter_id,
                name=a.name,
                version=a.version,
                modality=a.modality,
                concepts=a.concepts,
                status=a.status.value,
                created_at=a.created_at
            )
            for a in adapters
        ]
    except Exception as e:
        logger.error(f"Failed to list adapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters/{adapter_id}", response_model=AdapterResponse)
async def get_adapter(adapter_id: str):
    """Get details of a specific adapter."""
    try:
        registry = get_registry()
        adapter = registry.get_adapter(adapter_id)
        
        if not adapter:
            raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
        
        return AdapterResponse(
            adapter_id=adapter.adapter_id,
            name=adapter.name,
            version=adapter.version,
            modality=adapter.modality,
            concepts=adapter.concepts,
            status=adapter.status.value,
            created_at=adapter.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapters/{adapter_id}/promote")
async def promote_adapter(adapter_id: str):
    """Promote an adapter to production status."""
    try:
        registry = get_registry()
        adapter = registry.promote_to_production(adapter_id)
        
        if not adapter:
            raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
        
        return {"message": f"Adapter {adapter_id} promoted to production", "status": adapter.status.value}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/adapters/{adapter_id}")
async def delete_adapter(adapter_id: str):
    """Delete an adapter from the registry."""
    try:
        registry = get_registry()
        success = registry.delete_adapter(adapter_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
        
        return {"message": f"Adapter {adapter_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check

@router.get("/health")
async def health_check():
    """Check SAM3 service health."""
    try:
        from .sam3_segmenter import SAM3_AVAILABLE, TORCH_AVAILABLE
        
        return {
            "status": "healthy",
            "sam3_available": SAM3_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "adapters_registered": len(get_registry()._adapters) if get_registry() else 0
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }
