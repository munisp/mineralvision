"""HTTP layer for the custody ledger — thin wrappers over logic.py."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import get_db
from .logic import CustodyLedger, entry_to_dict

router = APIRouter()


class AppendRequest(BaseModel):
    entity_id: str = Field(..., min_length=1)
    entity_type: str
    event_type: str
    actor: str = Field(..., min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None


class BatchAppendRequest(BaseModel):
    entries: List[AppendRequest] = Field(..., min_length=1)


@router.post("/entries", status_code=201)
def append_entry(req: AppendRequest, db: Session = Depends(get_db)):
    """Append one custody event (batch_created/dispatch/lab_receipt/results)."""
    ledger = CustodyLedger(db)
    try:
        entry = ledger.append(
            entity_id=req.entity_id,
            entity_type=req.entity_type,
            event_type=req.event_type,
            actor=req.actor,
            payload=req.payload,
            timestamp=req.timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return entry_to_dict(entry)


@router.post("/entries/batch", status_code=201)
def append_entries(req: BatchAppendRequest, db: Session = Depends(get_db)):
    """Append several custody events in order (e.g. dispatch + receipt + results)."""
    ledger = CustodyLedger(db)
    out = []
    for item in req.entries:
        try:
            entry = ledger.append(
                entity_id=item.entity_id,
                entity_type=item.entity_type,
                event_type=item.event_type,
                actor=item.actor,
                payload=item.payload,
                timestamp=item.timestamp,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        out.append(entry_to_dict(entry))
    return {"appended": len(out), "entries": out}


@router.get("/chain/{entity_id}")
def get_chain(entity_id: str, db: Session = Depends(get_db)):
    """Return the full custody chain for an entity (e.g. a sample batch)."""
    ledger = CustodyLedger(db)
    chain = ledger.get_chain(entity_id)
    if not chain:
        raise HTTPException(status_code=404, detail=f"no ledger entries for entity {entity_id!r}")
    return {"entity_id": entity_id, "length": len(chain), "entries": [entry_to_dict(e) for e in chain]}


@router.get("/verify/{entity_id}")
def verify_chain(entity_id: str, db: Session = Depends(get_db)):
    """Walk one entity's chain proving hash/linkage/signature integrity."""
    ledger = CustodyLedger(db)
    result = ledger.verify_chain(entity_id)
    if result.entries_checked == 0:
        raise HTTPException(status_code=404, detail=f"no ledger entries for entity {entity_id!r}")
    return {
        "entity_id": entity_id,
        "valid": result.valid,
        "entries_checked": result.entries_checked,
        "first_invalid_id": result.first_invalid_id,
        "errors": result.errors,
    }


@router.get("/verify")
def verify_all(db: Session = Depends(get_db)):
    """Verify every chain in the ledger plus global sequence monotonicity."""
    ledger = CustodyLedger(db)
    result = ledger.verify_all()
    return {
        "valid": result.valid,
        "entries_checked": result.entries_checked,
        "first_invalid_id": result.first_invalid_id,
        "errors": result.errors,
    }
