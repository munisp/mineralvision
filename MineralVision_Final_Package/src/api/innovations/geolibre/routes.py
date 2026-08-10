"""GeoLibre integration routes — /innovations/geolibre.

Honest-degradation contract: JSON project authoring never requires the
``geolibre`` wheel.  ``/map-html`` lazily imports ``geolibre``; when absent it
renders a real, minimal MapLibre HTML page (marked
``generator: mineralvision-fallback``) containing the actual layer
URLs/GeoJSON — never a fabricated mock.  When neither can run it returns 503
with remediation.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

try:  # dual-context import
    from src.api.database import get_db
    from src.api.innovations.geolibre import project_builder as pb
    from src.api.innovations.geolibre import service
except ImportError:  # pragma: no cover
    from api.database import get_db
    from api.innovations.geolibre import project_builder as pb
    from api.innovations.geolibre import service

router = APIRouter(prefix="/innovations/geolibre", tags=["geolibre"])


def _geolibre_pkg() -> Optional[Any]:
    try:
        return importlib.import_module("geolibre")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    document: Dict[str, Any] = Field(..., description="a .geolibre.json project document")


class MapHtmlRequest(BaseModel):
    document: Dict[str, Any]
    prefer_backend: str = Field(
        "auto",
        description="auto|geolibre|fallback — 'geolibre' 503s when the wheel is absent",
    )


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/export")
def export_project(
    project_id: str,
    assay_key: Optional[str] = None,
    basemap: str = "carto-positron",
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Build a .geolibre.json project document from real platform data.

    404 when the project has no drillholes — nothing is fabricated.
    """
    try:
        doc = service.build_project_document(
            db, project_id, assay_key=assay_key, basemap=basemap
        )
    except pb.ProjectBuilderError as exc:
        raise HTTPException(422, str(exc))
    if doc is None:
        raise HTTPException(
            404,
            f"project '{project_id}' has no drillholes; cannot author a "
            "GeoLibre project from empty data",
        )
    return doc


@router.post("/projects/validate")
def validate_project(req: ValidateRequest) -> Dict[str, Any]:
    """Validate/normalize an uploaded .geolibre.json; return describe_project."""
    return pb.describe_project(req.document)


@router.post("/map-html", response_class=HTMLResponse)
def map_html(req: MapHtmlRequest) -> HTMLResponse:
    """Render a project document to an HTML map page.

    Prefers the real ``geolibre`` package (lazy import) when present;
    otherwise uses the honest minimal MapLibre fallback renderer.
    """
    problems = pb.validate_project(req.document)
    if problems:
        raise HTTPException(422, {"detail": "invalid project document",
                                  "problems": problems})
    pkg = _geolibre_pkg()
    if pkg is not None and req.prefer_backend in ("auto", "geolibre"):
        m = pkg.Map(center=req.document["center"], zoom=req.document["zoom"])
        m.to_html  # real client render
        html = m.to_html()
        return HTMLResponse(html)
    if req.prefer_backend == "geolibre":
        raise HTTPException(
            503,
            "geolibre package not installed; install with `pip install geolibre` "
            "or call with prefer_backend='fallback' for the minimal MapLibre renderer",
        )
    return HTMLResponse(service.render_fallback_html(req.document))


@router.get("/capabilities")
def capabilities() -> Dict[str, Any]:
    """Report geolibre availability, supported layer types, basemap catalog."""
    pkg = _geolibre_pkg()
    if pkg is not None:
        try:
            basemaps: Any = pkg.basemap_catalog()
        except Exception:
            basemaps = pb.BASEMAP_CATALOG
        pkg_info = {"available": True,
                    "version": getattr(pkg, "__version__", "unknown")}
    else:
        basemaps = pb.BASEMAP_CATALOG
        pkg_info = {"available": False, "version": None,
                    "remediation": "pip install geolibre"}
    return {
        "module": "geolibre-integration",
        "geolibre_package": pkg_info,
        "format_version": pb.FORMAT_VERSION,
        "generator": pb.GENERATOR,
        "supported_layer_types": list(pb.LAYER_TYPES),
        "color_ramps": list(pb.COLOR_RAMPS),
        "basemap_catalog": basemaps,
        "map_html_backends": ["geolibre (if installed)",
                              "mineralvision-fallback (built-in MapLibre)"],
        "platform_tile_routes": {
            "drillhole_features": service.DRILLHOLE_FEATURE_TILES,
            "raster_tiles": service.RASTER_TILES,
        },
    }
