"""HTTP layer for the integration_hub."""

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
        return create_api_key(db, req.name, req.scopes)
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
