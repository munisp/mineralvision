"""HTTP layer for the prospectivity copilot (thin — see logic.py)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database import get_db
from .logic import parse_query, explain_query, execute_query

router = APIRouter(prefix="/innovations/prospectivity_copilot",
                   tags=["prospectivity_copilot"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(50, ge=1, le=500)


@router.post("/parse")
def parse_only(req: QueryRequest):
    """Parse the question and return the AST + explanation (no DB access)."""
    pq = parse_query(req.question)
    return {"parsed": pq.to_dict(), "explanation": explain_query(pq)}


@router.post("/query")
def query(req: QueryRequest, db: Session = Depends(get_db)):
    """Parse, execute over projects/drillholes/samples, echo parsed query."""
    pq = parse_query(req.question)
    out = execute_query(db, pq, limit=req.limit)
    out["parsed"] = pq.to_dict()
    out["intent"] = pq.intent
    return out
