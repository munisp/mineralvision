"""HTTP layer for the integration_hub."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import get_db
from .logic import (
    ApiKeyModel,
    WebhookRegistry,
    api_key_to_dict,
    build_api_key_dependency,
    create_api_key,
    delivery_to_dict,
    webhook_to_dict,
)
from .models import ApiKeyModel as _ApiKeyModel
from .governed import (
    approve_writeback,
    discover_arcgis_capabilities,
    evidence_to_dict,
    proposal_to_dict,
    register_evidence,
    stage_writeback,
)

router = APIRouter()

# Scoped-key dependencies (mirror the platform's require_role factory pattern).
require_read_key = build_api_key_dependency(get_db, "read")
require_write_key = build_api_key_dependency(get_db, "write")


class SubscribeRequest(BaseModel):
    url: str = Field(..., min_length=1)
    topics: List[str] = Field(..., min_length=1)
    name: str = ""


class PublishRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1)
    scopes: List[str] = Field(..., min_length=1)
    tenant_id: str = Field(default="", max_length=128)


class RegisterEvidenceRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=128)
    source_system: str = Field(..., min_length=1, max_length=64)
    source_ref: str = Field(..., min_length=1, max_length=1024)
    source_version: str = Field(..., min_length=1, max_length=256)
    observed_at: datetime
    geometry: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    model_run: Dict[str, Any] = Field(default_factory=dict)


class ArcGISCapabilityRequest(BaseModel):
    service_metadata: Dict[str, Any] = Field(default_factory=dict)


class StageWritebackRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=128)
    evidence_id: str = Field(..., min_length=1, max_length=40)
    target_system: str = Field(..., min_length=1, max_length=64)
    target_ref: str = Field(..., min_length=1, max_length=1024)
    candidate_payload: Dict[str, Any] = Field(default_factory=dict)
    submitted_by: str = Field(..., min_length=1, max_length=128)
    dry_run: Dict[str, Any] = Field(default_factory=dict)


class ApproveWritebackRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=128)
    reviewer_id: str = Field(..., min_length=1, max_length=128)
    mfa_verified: bool
    review_reason: str = Field(..., min_length=1, max_length=2048)


# ----------------------------------------------------------------- webhooks
@router.post("/webhooks", status_code=201)
def subscribe(req: SubscribeRequest, db: Session = Depends(get_db),
              _key: ApiKeyModel = Depends(require_write_key)):
    """Register a webhook; the signing secret is returned ONCE."""
    registry = WebhookRegistry(db)
    try:
        webhook = registry.subscribe(req.url, req.topics, req.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return webhook_to_dict(webhook, include_secret=True)


@router.get("/webhooks")
def list_webhooks(db: Session = Depends(get_db), _key: ApiKeyModel = Depends(require_read_key)):
    registry = WebhookRegistry(db)
    return {"count": len(registry.list_webhooks()),
            "webhooks": [webhook_to_dict(w) for w in registry.list_webhooks()]}  # secret masked


@router.delete("/webhooks/{webhook_id}")
def unsubscribe(webhook_id: int, db: Session = Depends(get_db),
                _key: ApiKeyModel = Depends(require_write_key)):
    registry = WebhookRegistry(db)
    if not registry.unsubscribe(webhook_id):
        raise HTTPException(status_code=404, detail=f"webhook {webhook_id} not found")
    return {"deleted": webhook_id}


@router.post("/publish")
def publish(req: PublishRequest, db: Session = Depends(get_db),
            _key: ApiKeyModel = Depends(require_write_key)):
    """Fan an event out to subscribers; returns per-webhook delivery results."""
    registry = WebhookRegistry(db)
    deliveries = registry.publish(req.topic, req.payload)
    return {"topic": req.topic, "deliveries": [delivery_to_dict(d) for d in deliveries]}


@router.get("/deliveries")
def list_deliveries(webhook_id: Optional[int] = None, db: Session = Depends(get_db),
                    _key: ApiKeyModel = Depends(require_read_key)):
    registry = WebhookRegistry(db)
    deliveries = registry.list_deliveries(webhook_id=webhook_id)
    return {"count": len(deliveries), "deliveries": [delivery_to_dict(d) for d in deliveries]}


# ----------------------------------------------------------------- API keys
@router.post("/apikeys", status_code=201)
def create_key(req: CreateKeyRequest, db: Session = Depends(get_db),
               _key: ApiKeyModel = Depends(require_write_key)):
    """Create a scoped API key; plaintext returned once, bcrypt hash stored."""
    try:
        return create_api_key(db, req.name, req.scopes, tenant_id=req.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/apikeys")
def list_keys(db: Session = Depends(get_db), _key: ApiKeyModel = Depends(require_read_key)):
    rows = db.query(_ApiKeyModel).order_by(_ApiKeyModel.id).all()
    return {"count": len(rows), "api_keys": [api_key_to_dict(k) for k in rows]}


@router.delete("/apikeys/{key_id}")
def deactivate_key(key_id: str, db: Session = Depends(get_db),
                   _key: ApiKeyModel = Depends(require_write_key)):
    record = db.query(_ApiKeyModel).filter(_ApiKeyModel.key_id == key_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail=f"api key {key_id!r} not found")
    record.active = False
    db.commit()
    return {"deactivated": key_id}


# ------------------------------------------------------- governed integration
def _require_governed_tenant(key: ApiKeyModel, tenant_id: str) -> None:
    """Reject unbound or cross-tenant service keys for governed evidence APIs."""
    if not key.tenant_id or key.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="API key is not bound to this tenant")


@router.post("/evidence", status_code=201)
def create_evidence(
    req: RegisterEvidenceRequest,
    db: Session = Depends(get_db),
    _key: ApiKeyModel = Depends(require_write_key),
):
    """Register incumbent-source evidence with immutable canonical lineage."""
    _require_governed_tenant(_key, req.tenant_id)
    try:
        record = register_evidence(
            db,
            tenant_id=req.tenant_id,
            source_system=req.source_system,
            source_ref=req.source_ref,
            source_version=req.source_version,
            observed_at=req.observed_at,
            geometry=req.geometry,
            payload=req.payload,
            model_run=req.model_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return evidence_to_dict(record)


@router.post("/arcgis/capabilities")
def inspect_arcgis_capabilities(
    req: ArcGISCapabilityRequest,
    _key: ApiKeyModel = Depends(require_read_key),
):
    """Reduce Feature Service metadata to a conservative capability contract.

    No remote URL is fetched here. A deployment-specific connector worker must
    fetch service metadata through an approved, SSRF-safe allow-list.
    """
    return {"capabilities": discover_arcgis_capabilities(req.service_metadata)}


@router.post("/writebacks", status_code=201)
def create_writeback_proposal(
    req: StageWritebackRequest,
    db: Session = Depends(get_db),
    _key: ApiKeyModel = Depends(require_write_key),
):
    """Stage a candidate update; this endpoint never writes to an incumbent."""
    _require_governed_tenant(_key, req.tenant_id)
    try:
        proposal = stage_writeback(
            db,
            tenant_id=req.tenant_id,
            evidence_id=req.evidence_id,
            target_system=req.target_system,
            target_ref=req.target_ref,
            candidate_payload=req.candidate_payload,
            submitted_by=req.submitted_by,
            dry_run=req.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return proposal_to_dict(proposal)


@router.post("/writebacks/{proposal_id}/approve")
def approve_staged_writeback(
    proposal_id: str,
    req: ApproveWritebackRequest,
    db: Session = Depends(get_db),
    _key: ApiKeyModel = Depends(require_write_key),
):
    """Require MFA and a distinct reviewer before a worker may attempt a write."""
    _require_governed_tenant(_key, req.tenant_id)
    try:
        proposal = approve_writeback(
            db,
            tenant_id=req.tenant_id,
            proposal_id=proposal_id,
            reviewer_id=req.reviewer_id,
            mfa_verified=req.mfa_verified,
            review_reason=req.review_reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return proposal_to_dict(proposal)
