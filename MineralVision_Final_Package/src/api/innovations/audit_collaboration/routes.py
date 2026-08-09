"""HTTP layer for audit_collaboration."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import get_db
from .logic import (
    AuditCollaboration,
    comment_to_dict,
    event_to_dict,
    snapshot_to_dict,
)

router = APIRouter()


class RecordEventRequest(BaseModel):
    actor: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)


class CommentRequest(BaseModel):
    entity_type: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    parent_id: Optional[int] = None


class SettingsRequest(BaseModel):
    settings: Dict[str, Any]
    actor: str = Field(..., min_length=1)
    note: str = ""


class RevertRequest(BaseModel):
    version: int = Field(..., ge=1)
    actor: str = Field(..., min_length=1)


@router.post("/events", status_code=201)
def record_event(req: RecordEventRequest, db: Session = Depends(get_db)):
    """Append an audit event; the before/after JSON diff is computed server-side."""
    svc = AuditCollaboration(db)
    event = svc.record_event(
        actor=req.actor, action=req.action, entity_type=req.entity_type,
        entity_id=req.entity_id, before=req.before, after=req.after,
    )
    return event_to_dict(event)


@router.get("/events")
def list_events(entity_type: Optional[str] = None, entity_id: Optional[str] = None,
                db: Session = Depends(get_db)):
    svc = AuditCollaboration(db)
    events = svc.list_events(entity_type=entity_type, entity_id=entity_id)
    return {"count": len(events), "events": [event_to_dict(e) for e in events]}


@router.post("/comments", status_code=201)
def add_comment(req: CommentRequest, db: Session = Depends(get_db)):
    svc = AuditCollaboration(db)
    comment = svc.add_comment(
        entity_type=req.entity_type, entity_id=req.entity_id,
        author=req.author, body=req.body, parent_id=req.parent_id,
    )
    return comment_to_dict(comment)


@router.get("/comments")
def get_comments(entity_type: str, entity_id: str, threaded: bool = True,
                 db: Session = Depends(get_db)):
    """Comments for an entity; threaded=true returns depth-first thread order."""
    svc = AuditCollaboration(db)
    if threaded:
        comments = svc.get_threaded_comments(entity_type, entity_id)
    else:
        comments = [comment_to_dict(c) for c in svc.get_comments(entity_type, entity_id)]
    return {"count": len(comments), "comments": comments}


@router.put("/settings/{project_id}", status_code=201)
def set_settings(project_id: str, req: SettingsRequest, db: Session = Depends(get_db)):
    """Store a new immutable settings snapshot for the project."""
    svc = AuditCollaboration(db)
    snapshot = svc.set_settings(project_id, req.settings, req.actor, req.note)
    return snapshot_to_dict(snapshot)


@router.get("/settings/{project_id}")
def get_settings(project_id: str, db: Session = Depends(get_db)):
    svc = AuditCollaboration(db)
    current = svc.get_current_settings(project_id)
    if current is None:
        raise HTTPException(status_code=404, detail=f"no settings for project {project_id!r}")
    history = svc.get_settings_history(project_id)
    return {
        "current": snapshot_to_dict(current),
        "history": [snapshot_to_dict(s) for s in history],
    }


@router.post("/settings/{project_id}/revert", status_code=201)
def revert_settings(project_id: str, req: RevertRequest, db: Session = Depends(get_db)):
    """Revert to a previous settings version (creates a new snapshot)."""
    svc = AuditCollaboration(db)
    snapshot = svc.revert_settings(project_id, req.version, req.actor)
    return snapshot_to_dict(snapshot)
