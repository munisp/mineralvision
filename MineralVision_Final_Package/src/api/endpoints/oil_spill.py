"""API endpoints for reviewable oil-spill image and mask assessments."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..database import AuditLogModel, OilSpillIncidentModel, ProjectModel, get_db
from ..oil_spill.analysis import (
    MaskValidationError,
    assess_mask,
    build_coverage_priority_cells,
    decode_probability_mask,
)
from ..oil_spill.models import ModelInferenceError, ModelNotConfiguredError, model_from_environment
from ..oil_spill.schemas import (
    GeographicBounds,
    MaskAnalysisRequest,
    ObservationSource,
    OilSpillAssessmentResponse,
    ReviewRequest,
    ReviewStatus,
    SearchPlanRequest,
    SearchPlanResponse,
    Severity,
)

router = APIRouter(prefix="/api/oil-spill", tags=["Oil Spill Intelligence"])
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _validate_project(project_id: Optional[str], db: Session) -> None:
    if project_id and not db.query(ProjectModel).filter(ProjectModel.id == project_id).first():
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")


def _serialize_record(record: OilSpillIncidentModel) -> OilSpillAssessmentResponse:
    return OilSpillAssessmentResponse(
        incident_id=record.id,
        source=ObservationSource(record.source),
        model_id=record.model_id,
        model_version=record.model_version,
        review_status=ReviewStatus(record.review_status),
        severity=Severity(record.severity),
        oil_pixel_count=record.oil_pixel_count,
        oil_fraction=record.oil_fraction,
        oil_area_m2=record.oil_area_m2,
        oil_area_hectares=round(record.oil_area_m2 / 10_000, 6) if record.oil_area_m2 is not None else None,
        confidence=record.confidence,
        quality_flags=record.quality_flags or [],
        geometry_geojson=record.geometry_geojson,
        mask_dimensions=[record.image_width_px, record.image_height_px],
        observed_at=record.observed_at,
        created_at=record.created_at,
    )


def _persist_assessment(request: MaskAnalysisRequest, probability_map, db: Session) -> OilSpillAssessmentResponse:
    """Run deterministic assessment logic and persist an auditable incident record."""
    _validate_project(request.project_id, db)
    try:
        assessment = assess_mask(
            probability_map,
            threshold=request.probability_threshold,
            min_component_area_px=request.min_component_area_px,
            ground_sampling_distance_m=request.ground_sampling_distance_m,
            geographic_bounds=request.geographic_bounds,
        )
    except MaskValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    incident_id = str(uuid.uuid4())
    source_metadata = {
        **request.metadata,
        "analysis_parameters": {
            "probability_threshold": request.probability_threshold,
            "min_component_area_px": request.min_component_area_px,
            "candidate_pixels": assessment.candidate_pixels,
            "retained_component_count": assessment.component_count,
            "ground_sampling_distance_m": request.ground_sampling_distance_m,
            "geographic_bounds": request.geographic_bounds.model_dump() if request.geographic_bounds else None,
        },
    }
    record = OilSpillIncidentModel(
        id=incident_id,
        project_id=request.project_id,
        image_id=request.image_id,
        source=request.source.value,
        model_id=request.model_id,
        model_version=request.model_version,
        review_status=ReviewStatus.PENDING_REVIEW.value,
        severity=assessment.severity.value,
        oil_pixel_count=assessment.retained_pixels,
        oil_fraction=round(assessment.retained_pixels / (request.image_width_px * request.image_height_px), 8),
        oil_area_m2=assessment.oil_area_m2,
        confidence=assessment.confidence,
        quality_flags=assessment.quality_flags,
        geometry_geojson=assessment.geometry_geojson,
        image_width_px=request.image_width_px,
        image_height_px=request.image_height_px,
        observed_at=request.observed_at,
        source_metadata=source_metadata,
    )
    db.add(record)
    db.add(
        AuditLogModel(
            id=str(uuid.uuid4()),
            action="oil_spill_assessment_created",
            entity_type="oil_spill_incident",
            entity_id=incident_id,
            details={
                "source": request.source.value,
                "model_id": request.model_id,
                "model_version": request.model_version,
                "review_status": ReviewStatus.PENDING_REVIEW.value,
            },
        )
    )
    db.commit()
    db.refresh(record)
    return _serialize_record(record)


@router.post("/analyze/mask", response_model=OilSpillAssessmentResponse, status_code=201)
async def analyze_mask(request: MaskAnalysisRequest, db: Session = Depends(get_db)):
    """Assess caller-supplied, versioned segmentation evidence and create a review record.

    This endpoint does not independently classify raw imagery. The caller must identify the
    producing model or annotation process through `model_id` and `model_version`.
    """
    try:
        probability_map = decode_probability_mask(
            request.mask_base64,
            expected_width=request.image_width_px,
            expected_height=request.image_height_px,
        )
    except MaskValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _persist_assessment(request, probability_map, db)


@router.post("/analyze/image", response_model=OilSpillAssessmentResponse, status_code=201)
async def analyze_image(
    image: UploadFile = File(...),
    source: ObservationSource = Form(...),
    model_id: Optional[str] = Form(None),
    model_version: Optional[str] = Form(None),
    image_id: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    observed_at: Optional[datetime] = Form(None),
    ground_sampling_distance_m: Optional[float] = Form(None),
    geographic_bounds_json: Optional[str] = Form(None),
    probability_threshold: float = Form(0.5),
    min_component_area_px: int = Form(25),
    metadata_json: str = Form("{}"),
    db: Session = Depends(get_db),
):
    """Run an operator-configured local segmentation model against uploaded imagery.

    The deployment must configure `OIL_SPILL_MODEL_PATH` and model provenance variables.
    The endpoint fails closed when a trusted local model is not configured.
    """
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are accepted")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Image upload is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image upload exceeds the 25 MiB safety limit")

    try:
        import io
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image_width_px, image_height_px = uploaded_image.size
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Upload is not a valid image") from exc

    try:
        bounds = GeographicBounds.model_validate(json.loads(geographic_bounds_json)) if geographic_bounds_json else None
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must decode to an object")
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid geospatial or metadata JSON: {exc}") from exc

    try:
        model = model_from_environment()
    except ModelNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if model_id and model_id != model.descriptor.model_id:
        raise HTTPException(status_code=422, detail="model_id must match the operator-configured model")
    if model_version and model_version != model.descriptor.version:
        raise HTTPException(status_code=422, detail="model_version must match the operator-configured model")

    try:
        probability_map = model.predict_probability(image_bytes)
    except ModelInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        request = MaskAnalysisRequest(
            # This provenance-only value is not used after `probability_map` is generated.
            mask_base64="AA==",
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            source=source,
            model_id=model.descriptor.model_id,
            model_version=model.descriptor.version,
            image_id=image_id or image.filename,
            project_id=project_id,
            observed_at=observed_at,
            ground_sampling_distance_m=ground_sampling_distance_m,
            geographic_bounds=bounds,
            probability_threshold=probability_threshold,
            min_component_area_px=min_component_area_px,
            metadata={**metadata, "image_filename": image.filename},
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return _persist_assessment(request, probability_map, db)


@router.get("/incidents", response_model=List[OilSpillAssessmentResponse])
async def list_incidents(
    project_id: Optional[str] = Query(None),
    review_status: Optional[ReviewStatus] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List persisted oil-spill assessments, optionally filtered by project and review state."""
    query = db.query(OilSpillIncidentModel)
    if project_id:
        query = query.filter(OilSpillIncidentModel.project_id == project_id)
    if review_status:
        query = query.filter(OilSpillIncidentModel.review_status == review_status.value)
    records = query.order_by(OilSpillIncidentModel.created_at.desc()).offset(offset).limit(limit).all()
    return [_serialize_record(record) for record in records]


@router.get("/incidents/{incident_id}", response_model=OilSpillAssessmentResponse)
async def get_incident(incident_id: str, db: Session = Depends(get_db)):
    """Retrieve a single reviewable oil-spill assessment."""
    record = db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Oil-spill incident {incident_id} not found")
    return _serialize_record(record)


@router.patch("/incidents/{incident_id}/review", response_model=OilSpillAssessmentResponse)
async def review_incident(incident_id: str, request: ReviewRequest, db: Session = Depends(get_db)):
    """Record an authorized operator's review without changing the underlying evidence."""
    record = db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Oil-spill incident {incident_id} not found")

    record.review_status = request.status.value
    record.reviewer = request.reviewer
    record.review_note = request.note
    record.reviewed_at = datetime.utcnow()
    db.add(
        AuditLogModel(
            id=str(uuid.uuid4()),
            action="oil_spill_incident_reviewed",
            entity_type="oil_spill_incident",
            entity_id=incident_id,
            details={"status": request.status.value, "reviewer": request.reviewer},
        )
    )
    db.commit()
    db.refresh(record)
    return _serialize_record(record)


@router.post("/incidents/{incident_id}/coverage-plan", response_model=SearchPlanResponse)
async def create_coverage_plan(incident_id: str, request: SearchPlanRequest, db: Session = Depends(get_db)):
    """Produce an advisory priority grid for a later, authorized flight-planning process."""
    record = db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Oil-spill incident {incident_id} not found")
    recommended_area_m2, priority_cells, notes = build_coverage_priority_cells(
        record.geometry_geojson,
        cell_size_m=request.cell_size_m,
        drone_count=request.drone_count,
        buffer_m=request.buffer_m,
    )
    return SearchPlanResponse(
        incident_id=incident_id,
        recommended_search_area_m2=recommended_area_m2,
        priority_cells=priority_cells,
        notes=notes,
    )
