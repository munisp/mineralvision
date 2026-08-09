"""HTTP layer for field_sync — thin wrappers over logic.py."""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import get_db
from .logic import FieldSync, conflict_to_dict, op_to_dict, state_to_dict

router = APIRouter()


class SyncOpRequest(BaseModel):
    client_op_id: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    op: str
    base_version: int = Field(..., ge=0)
    payload: Dict[str, Any] = Field(default_factory=dict)
    client_ts: Optional[datetime] = None


class SyncBatchRequest(BaseModel):
    ops: List[SyncOpRequest] = Field(..., min_length=1)


def _apply(sync: FieldSync, item: SyncOpRequest):
    try:
        result = sync.apply_op(
            client_op_id=item.client_op_id,
            entity_id=item.entity_id,
            op=item.op,
            base_version=item.base_version,
            payload=item.payload,
            client_ts=item.client_ts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return asdict(result)


@router.post("/ops", status_code=200)
def push_op(req: SyncOpRequest, db: Session = Depends(get_db)):
    """Push a single change op (idempotent via client_op_id)."""
    return _apply(FieldSync(db), req)


@router.post("/ops/batch", status_code=200)
def push_ops(req: SyncBatchRequest, db: Session = Depends(get_db)):
    """Push a batch of change ops in order."""
    sync = FieldSync(db)
    results = [_apply(sync, item) for item in req.ops]
    return {"processed": len(results), "results": results}


@router.get("/state/{entity_id}")
def get_state(entity_id: str, db: Session = Depends(get_db)):
    sync = FieldSync(db)
    state = sync.get_state(entity_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"entity {entity_id!r} not found")
    return state_to_dict(state)


@router.get("/pull")
def pull_since(since: int = 0, entity_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Delta download: applied ops with version > `since`."""
    sync = FieldSync(db)
    ops = sync.pull_since(since=since, entity_id=entity_id)
    return {"since": since, "count": len(ops), "ops": [op_to_dict(o) for o in ops]}


@router.get("/conflicts")
def list_conflicts(entity_id: Optional[str] = None, db: Session = Depends(get_db)):
    sync = FieldSync(db)
    conflicts = sync.list_conflicts(entity_id=entity_id)
    return {"count": len(conflicts), "conflicts": [conflict_to_dict(c) for c in conflicts]}
