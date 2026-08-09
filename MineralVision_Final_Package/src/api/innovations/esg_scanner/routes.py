"""HTTP layer for the ESG scanner."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .logic import ESGScanner, result_to_dict

router = APIRouter()


class ScanRequest(BaseModel):
    project_data: Dict[str, Any]
    categories: Optional[List[str]] = Field(
        default=None, description="Subset of rule categories; default scans all."
    )


@router.get("/rules")
def list_rules():
    """List all rule packs and their rules."""
    scanner = ESGScanner()
    return {
        category: [
            {
                "id": r.id,
                "description": r.description,
                "check": {"field": r.check_field, "op": r.op, "value": r.value},
                "severity": r.severity,
                "remediation": r.remediation,
                "framework_ref": r.framework_ref,
            }
            for r in rules
        ]
        for category, rules in sorted(scanner.packs.items())
    }


@router.post("/scan")
def scan(req: ScanRequest):
    """Evaluate posted project data against the ESG rule packs → gap list."""
    scanner = ESGScanner()
    try:
        result = scanner.scan(req.project_data, req.categories)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result_to_dict(result)
