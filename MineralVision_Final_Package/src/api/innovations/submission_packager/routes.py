"""HTTP layer for the submission packager."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from .logic import get_template, package_submission, validate_submission
from .templates import TEMPLATES

router = APIRouter()


class SubmissionRequest(BaseModel):
    template: str
    data: Dict[str, Any]


@router.get("/templates")
def list_templates():
    """List jurisdiction templates and their sections."""
    return {
        name: {
            "title": tpl["title"],
            "sections": [
                {
                    "key": s["key"],
                    "filename": s["filename"],
                    "required": s["required"],
                    "description": s["description"],
                }
                for s in tpl["sections"]
            ],
        }
        for name, tpl in sorted(TEMPLATES.items())
    }


@router.post("/validate")
def validate(req: SubmissionRequest):
    """Required-section completeness check; fails (valid=false) on omissions."""
    try:
        result = validate_submission(req.template, req.data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "valid": result.valid,
        "issues": [{"section": i.section, "filename": i.filename, "reason": i.reason} for i in result.issues],
    }


@router.post("/package")
def package(req: SubmissionRequest):
    """Build the deterministic submission ZIP. 422 when validation fails."""
    try:
        result = package_submission(req.template, req.data, strict=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return Response(
        content=result.zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{req.template}_submission.zip"',
            "X-Manifest-Sha256": result.manifest["files"][-1]["sha256"] if result.manifest["files"] else "",
        },
    )


@router.post("/package/preview")
def package_preview(req: SubmissionRequest):
    """Return the manifest JSON without the ZIP payload."""
    try:
        result = package_submission(req.template, req.data, strict=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result.manifest
