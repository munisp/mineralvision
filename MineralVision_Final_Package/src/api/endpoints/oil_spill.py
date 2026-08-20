"""API endpoints for reviewable oil-spill image and mask assessments."""

from __future__ import annotations

import base64
import io
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..database import (
    AuditLogModel,
    OilSpillEvaluationRunModel,
    OilSpillIncidentEventModel,
    OilSpillIncidentModel,
    OilSpillModelModel,
    ProjectModel,
    get_db,
)
from ..oil_spill.analysis import (
    MaskValidationError,
    assess_mask,
    build_coverage_priority_cells,
    decode_probability_mask,
)
from ..oil_spill.models import ModelInferenceError, ModelNotConfiguredError, model_from_environment
from ..oil_spill.schemas import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    GeographicBounds,
    IncidentEventRequest,
    IncidentEventResponse,
    MaskAnalysisRequest,
    ModelApprovalRequest,
    ModelLifecycleStatus,
    ModelPromotionResponse,
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    OperationsSummaryResponse,
    ObservationSource,
    TemporalConsensusRequest,
    TemporalConsensusResponse,
    OilSpillAssessmentResponse,
    ReviewRequest,
    ReviewStatus,
    SearchPlanRequest,
    SearchPlanResponse,
    Severity,
)
from ..oil_spill.governance import evaluate_promotion_eligibility, fuse_temporal_probabilities

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
    registered_model = db.query(OilSpillModelModel).filter(
        (OilSpillModelModel.model_id == model.descriptor.model_id)
        & (OilSpillModelModel.model_version == model.descriptor.version)
        & (OilSpillModelModel.artifact_sha256 == model.descriptor.artifact_sha256)
    ).first()
    if not registered_model:
        raise HTTPException(
            status_code=503,
            detail="Configured model artifact is not registered with matching provenance; image inference remains disabled.",
        )
    if registered_model.lifecycle_status != ModelLifecycleStatus.APPROVED.value:
        raise HTTPException(
            status_code=409,
            detail="Configured model is not approved for image inference; submit sealed evaluation evidence first.",
        )

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
    _record_incident_event(
        db,
        incident_id=incident_id,
        event_type="review_completed",
        actor=request.reviewer,
        details={"status": request.status.value, "note": request.note},
    )
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
    _record_incident_event(
        db,
        incident_id=incident_id,
        event_type="coverage_plan_created",
        actor="api_coverage_planner",
        details={"cell_size_m": request.cell_size_m, "drone_count": request.drone_count, "buffer_m": request.buffer_m},
    )
    db.commit()
    return SearchPlanResponse(
        incident_id=incident_id,
        recommended_search_area_m2=recommended_area_m2,
        priority_cells=priority_cells,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Model governance, evaluation, export, and operational timeline
# ---------------------------------------------------------------------------

def _model_response(record: OilSpillModelModel) -> ModelRegistrationResponse:
    return ModelRegistrationResponse(
        id=record.id,
        model_id=record.model_id,
        model_version=record.model_version,
        engine=record.engine,
        artifact_sha256=record.artifact_sha256,
        intended_domains=record.intended_domains or [],
        lifecycle_status=ModelLifecycleStatus(record.lifecycle_status),
        created_at=record.created_at,
    )


def _evaluation_response(record: OilSpillEvaluationRunModel, model: OilSpillModelModel) -> EvaluationRunResponse:
    return EvaluationRunResponse(
        id=record.id,
        model_id=model.model_id,
        model_version=model.model_version,
        dataset_fingerprint=record.dataset_fingerprint,
        split=record.split,
        domain=record.domain,
        sample_count=record.sample_count,
        metrics=record.metrics,
        jepa_backbone=record.jepa_backbone,
        created_at=record.created_at,
    )


def _event_response(record: OilSpillIncidentEventModel) -> IncidentEventResponse:
    return IncidentEventResponse(
        id=record.id,
        incident_id=record.incident_id,
        event_type=record.event_type,
        actor=record.actor,
        details=record.details or {},
        created_at=record.created_at,
    )


def _record_incident_event(
    db: Session,
    *,
    incident_id: str,
    event_type: str,
    actor: str,
    details: Dict[str, Any],
) -> OilSpillIncidentEventModel:
    event = OilSpillIncidentEventModel(
        id=str(uuid.uuid4()),
        incident_id=incident_id,
        event_type=event_type,
        actor=actor,
        details=details,
    )
    db.add(event)
    return event


def _promotion_response(record: OilSpillModelModel, db: Session) -> ModelPromotionResponse:
    evaluations = db.query(OilSpillEvaluationRunModel).filter(
        OilSpillEvaluationRunModel.model_registration_id == record.id
    ).all()
    decision = evaluate_promotion_eligibility(
        [
            (evaluation.domain, evaluation.split, evaluation.sample_count, evaluation.metrics)
            for evaluation in evaluations
        ],
        record.intended_domains or [],
    )
    return ModelPromotionResponse(
        model_id=record.model_id,
        model_version=record.model_version,
        eligible=decision.eligible,
        lifecycle_status=ModelLifecycleStatus(record.lifecycle_status),
        reasons=decision.reasons,
    )


@router.post("/models", response_model=ModelRegistrationResponse, status_code=201)
async def register_model(request: ModelRegistrationRequest, db: Session = Depends(get_db)):
    """Register provenance for a candidate local TorchScript or ONNX artifact.

    Registration does not load a checkpoint or enable inference. It establishes the immutable
    identity needed for evaluation and approval before an operator configures the artifact.
    """
    duplicate = db.query(OilSpillModelModel).filter(
        (OilSpillModelModel.model_id == request.model_id)
        & (OilSpillModelModel.model_version == request.model_version)
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail="A model with this id and version is already registered")
    artifact_duplicate = db.query(OilSpillModelModel).filter(
        OilSpillModelModel.artifact_sha256 == request.artifact_sha256
    ).first()
    if artifact_duplicate:
        raise HTTPException(status_code=409, detail="This model artifact hash is already registered")

    record = OilSpillModelModel(
        id=str(uuid.uuid4()),
        model_id=request.model_id,
        model_version=request.model_version,
        engine=request.engine,
        artifact_sha256=request.artifact_sha256.lower(),
        intended_domains=request.intended_domains,
        model_card_url=request.model_card_url,
        notes=request.notes,
        lifecycle_status=ModelLifecycleStatus.CANDIDATE.value,
    )
    db.add(record)
    db.add(
        AuditLogModel(
            id=str(uuid.uuid4()),
            action="oil_spill_model_registered",
            entity_type="oil_spill_model",
            entity_id=record.id,
            details={"model_id": record.model_id, "model_version": record.model_version},
        )
    )
    db.commit()
    db.refresh(record)
    return _model_response(record)


@router.get("/models", response_model=List[ModelRegistrationResponse])
async def list_models(
    lifecycle_status: Optional[ModelLifecycleStatus] = Query(None),
    db: Session = Depends(get_db),
):
    """List registered segmentation models and their governance state."""
    query = db.query(OilSpillModelModel).order_by(OilSpillModelModel.created_at.desc())
    if lifecycle_status:
        query = query.filter(OilSpillModelModel.lifecycle_status == lifecycle_status.value)
    return [_model_response(record) for record in query.all()]


@router.post("/models/{model_id}/{model_version}/evaluations", response_model=EvaluationRunResponse, status_code=201)
async def record_evaluation(
    model_id: str,
    model_version: str,
    request: EvaluationRunRequest,
    db: Session = Depends(get_db),
):
    """Store a reproducible evaluation record; the API never calculates or fabricates metrics."""
    model = db.query(OilSpillModelModel).filter(
        (OilSpillModelModel.model_id == model_id) & (OilSpillModelModel.model_version == model_version)
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Registered model not found")
    record = OilSpillEvaluationRunModel(
        id=str(uuid.uuid4()),
        model_registration_id=model.id,
        dataset_fingerprint=request.dataset_fingerprint,
        split=request.split.value,
        domain=request.domain,
        sample_count=request.sample_count,
        metrics=request.metrics.model_dump(),
        jepa_backbone=request.jepa_backbone,
        reviewer=request.reviewer,
        notes=request.notes,
    )
    db.add(record)
    db.add(
        AuditLogModel(
            id=str(uuid.uuid4()),
            action="oil_spill_model_evaluation_recorded",
            entity_type="oil_spill_model",
            entity_id=model.id,
            details={"evaluation_id": record.id, "split": request.split.value, "domain": request.domain},
        )
    )
    db.commit()
    db.refresh(record)
    return _evaluation_response(record, model)


@router.get("/models/{model_id}/{model_version}/promotion", response_model=ModelPromotionResponse)
async def get_model_promotion_status(model_id: str, model_version: str, db: Session = Depends(get_db)):
    """Evaluate the documented 97%-quality promotion gate without changing model state."""
    model = db.query(OilSpillModelModel).filter(
        (OilSpillModelModel.model_id == model_id) & (OilSpillModelModel.model_version == model_version)
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Registered model not found")
    return _promotion_response(model, db)


@router.post("/models/{model_id}/{model_version}/approve", response_model=ModelPromotionResponse)
async def approve_model(
    model_id: str,
    model_version: str,
    request: ModelApprovalRequest,
    db: Session = Depends(get_db),
):
    """Approve a model only after every intended domain clears the sealed-evaluation gate."""
    model = db.query(OilSpillModelModel).filter(
        (OilSpillModelModel.model_id == model_id) & (OilSpillModelModel.model_version == model_version)
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Registered model not found")
    promotion = _promotion_response(model, db)
    if not promotion.eligible:
        raise HTTPException(status_code=409, detail={"message": "Model is not eligible for approval", "reasons": promotion.reasons})

    model.lifecycle_status = ModelLifecycleStatus.APPROVED.value
    model.approved_by = request.reviewer
    model.approved_at = datetime.utcnow()
    db.add(
        AuditLogModel(
            id=str(uuid.uuid4()),
            action="oil_spill_model_approved",
            entity_type="oil_spill_model",
            entity_id=model.id,
            details={"reviewer": request.reviewer, "note": request.note},
        )
    )
    db.commit()
    return _promotion_response(model, db)


@router.get("/operations/summary", response_model=OperationsSummaryResponse)
async def operations_summary(db: Session = Depends(get_db)):
    """Return compact incident and model-governance counts for operations dashboards."""
    return OperationsSummaryResponse(
        total_incidents=db.query(OilSpillIncidentModel).count(),
        pending_review=db.query(OilSpillIncidentModel).filter(
            OilSpillIncidentModel.review_status == ReviewStatus.PENDING_REVIEW.value
        ).count(),
        confirmed=db.query(OilSpillIncidentModel).filter(
            OilSpillIncidentModel.review_status == ReviewStatus.CONFIRMED.value
        ).count(),
        needs_resurvey=db.query(OilSpillIncidentModel).filter(
            OilSpillIncidentModel.review_status == ReviewStatus.NEEDS_RESURVEY.value
        ).count(),
        high_or_critical=db.query(OilSpillIncidentModel).filter(
            OilSpillIncidentModel.severity.in_([Severity.HIGH.value, Severity.CRITICAL.value])
        ).count(),
        approved_models=db.query(OilSpillModelModel).filter(
            OilSpillModelModel.lifecycle_status == ModelLifecycleStatus.APPROVED.value
        ).count(),
        candidate_models=db.query(OilSpillModelModel).filter(
            OilSpillModelModel.lifecycle_status == ModelLifecycleStatus.CANDIDATE.value
        ).count(),
    )


@router.post("/incidents/{incident_id}/events", response_model=IncidentEventResponse, status_code=201)
async def create_incident_event(
    incident_id: str,
    request: IncidentEventRequest,
    db: Session = Depends(get_db),
):
    """Record a structured, human-attributable operational event without dispatching actions."""
    if not db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first():
        raise HTTPException(status_code=404, detail=f"Oil-spill incident {incident_id} not found")
    event = _record_incident_event(
        db,
        incident_id=incident_id,
        event_type=request.event_type.value,
        actor=request.actor,
        details=request.details,
    )
    db.add(
        AuditLogModel(
            id=str(uuid.uuid4()),
            action="oil_spill_incident_event_created",
            entity_type="oil_spill_incident",
            entity_id=incident_id,
            details={"event_type": request.event_type.value, "actor": request.actor},
        )
    )
    db.commit()
    db.refresh(event)
    return _event_response(event)


@router.get("/incidents/{incident_id}/events", response_model=List[IncidentEventResponse])
async def list_incident_events(incident_id: str, db: Session = Depends(get_db)):
    """List the immutable timeline for an oil-spill assessment."""
    if not db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first():
        raise HTTPException(status_code=404, detail=f"Oil-spill incident {incident_id} not found")
    events = db.query(OilSpillIncidentEventModel).filter(
        OilSpillIncidentEventModel.incident_id == incident_id
    ).order_by(OilSpillIncidentEventModel.created_at.asc()).all()
    return [_event_response(event) for event in events]


@router.get("/incidents/{incident_id}/export.geojson")
async def export_incident_geojson(incident_id: str, db: Session = Depends(get_db)):
    """Export a reviewed incident as interoperable GeoJSON evidence; no external action occurs."""
    record = db.query(OilSpillIncidentModel).filter(OilSpillIncidentModel.id == incident_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Oil-spill incident {incident_id} not found")
    geometry = record.geometry_geojson
    if not geometry:
        raise HTTPException(status_code=409, detail="Incident has no georeferenced geometry to export")
    _record_incident_event(
        db,
        incident_id=incident_id,
        event_type="exported",
        actor="api_export",
        details={"format": "geojson"},
    )
    db.commit()
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": record.id,
                "geometry": geometry,
                "properties": {
                    "source": record.source,
                    "model_id": record.model_id,
                    "model_version": record.model_version,
                    "review_status": record.review_status,
                    "severity": record.severity,
                    "oil_area_m2": record.oil_area_m2,
                    "confidence": record.confidence,
                    "observed_at": record.observed_at.isoformat() if record.observed_at else None,
                    "created_at": record.created_at.isoformat(),
                },
            }
        ],
    }


@router.post("/temporal-consensus", response_model=TemporalConsensusResponse)
async def temporal_consensus(request: TemporalConsensusRequest):
    """Fuse aligned sequence evidence; supplied embeddings must originate from a real JEPA backend.

    The endpoint is intentionally non-persistent: it creates a reviewable intermediate result,
    which a caller may then submit through ``/analyze/mask`` with its own provenance metadata.
    """
    try:
        probability_maps = [
            decode_probability_mask(
                mask_base64,
                expected_width=request.image_width_px,
                expected_height=request.image_height_px,
            )
            for mask_base64 in request.masks_base64
        ]
        result = fuse_temporal_probabilities(
            probability_maps,
            jepa_embeddings=request.jepa_embeddings,
        )
    except (MaskValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray((result.probability_map * 255).round().astype("uint8"), mode="L").save(buffer, format="PNG")
    return TemporalConsensusResponse(
        fused_mask_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        frame_weights=result.frame_weights,
        temporal_stability=result.temporal_stability,
        used_jepa_embeddings=result.used_jepa_embeddings,
        quality_flags=result.quality_flags,
    )
