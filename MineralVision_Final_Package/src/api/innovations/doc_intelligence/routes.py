"""HTTP layer for doc_intelligence.

Accepts plain text / markdown directly. PDF upload is supported when the
optional ``pypdf`` package is installed; without it a PDF post returns
501 Not Implemented (plain text remains fully supported).
"""

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from .logic import extract_all, extract_text_from_pdf

router = APIRouter()


class ExtractTextRequest(BaseModel):
    text: str
    filename: Optional[str] = None


def _result_payload(text: str, source: str) -> dict:
    result = extract_all(text)
    payload = asdict(result)
    payload["source"] = source
    payload["counts"] = {
        "hole_ids": len(result.hole_ids),
        "intervals": len(result.intervals),
        "commodities": len(result.commodities),
        "dates": len(result.dates),
        "coordinates": len(result.coordinates),
    }
    return payload


@router.post("/extract/text")
def extract_from_text(req: ExtractTextRequest):
    """Extract entities from posted plain text / markdown."""
    return _result_payload(req.text, source=req.filename or "inline-text")


@router.post("/extract/document")
async def extract_from_document(file: UploadFile):
    """Extract entities from an uploaded .txt/.md/.pdf document.

    PDF requires the optional ``pypdf`` dependency → 501 when absent.
    """
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        try:
            text = extract_text_from_pdf(raw)
        except RuntimeError as exc:
            if str(exc) == "pypdf_unavailable":
                raise HTTPException(
                    status_code=501,
                    detail="PDF extraction not configured: install the optional 'pypdf' package. "
                           "Plain text / markdown uploads are supported without it.",
                )
            raise
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="document is not valid UTF-8 text")
    return _result_payload(text, source=file.filename or "upload")
