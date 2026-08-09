"""HTTP layer for indigenous knowledge governance.

RBAC is enforced with the EXISTING platform dependencies from
``src.api.auth_middleware`` (require_auth / require_role); tier policy on
direct reads is applied by logic.py.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...auth_middleware import TokenPayload, require_auth, require_role
from .db import get_db
from .logic import (
    AUDIT_ROLES,
    CREATE_ROLES,
    EXPORT_ROLES,
    AccessTier,
    IndigenousGovernance,
    audit_to_dict,
    record_to_dict,
)

router = APIRouter()


class CreateRecordRequest(BaseModel):
    title: str = Field(..., min_length=1)
    community: str = Field(..., min_length=1)
    tier: AccessTier
    content: str = Field(..., min_length=1)
    consent_reference: str = Field(..., min_length=1)
    attribution: str = Field(..., min_length=1)


@router.post("/records", status_code=201)
def create_record(
    req: CreateRecordRequest,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_role(list(CREATE_ROLES))),
):
    gov = IndigenousGovernance(db)
    record = gov.create_record(
        title=req.title,
        community=req.community,
        tier=req.tier,
        content=req.content,
        consent_reference=req.consent_reference,
        attribution=req.attribution,
        actor=user.username,
    )
    return record_to_dict(record)


@router.get("/records")
def list_records(
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """List records: sacred NEVER listed; restricted requires elevated role."""
    gov = IndigenousGovernance(db)
    records = gov.list_records(actor=user.username, role=user.role)
    return {"count": len(records), "records": [record_to_dict(r) for r in records]}


@router.get("/records/export")
def export_records(
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_role(list(EXPORT_ROLES))),
):
    """Bulk export for authorized roles; sacred tier is always excluded."""
    gov = IndigenousGovernance(db)
    records = gov.export_records(actor=user.username, role=user.role)
    return {"count": len(records), "records": [record_to_dict(r) for r in records]}


@router.get("/records/{record_id}")
def get_record(
    record_id: int,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """Direct id access — the ONLY way to reach a sacred record (role-gated)."""
    gov = IndigenousGovernance(db)
    record = gov.get_record(record_id, actor=user.username, role=user.role)
    return record_to_dict(record)


@router.get("/audit")
def get_audit(
    record_id: Optional[int] = None,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_role(list(AUDIT_ROLES))),
):
    gov = IndigenousGovernance(db)
    rows = gov.audit_trail(record_id=record_id)
    return {"count": len(rows), "audit": [audit_to_dict(r) for r in rows]}
