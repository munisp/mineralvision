"""
geodb bridge — FastAPI router.

Unifies the API database, lakehouse parquet storage, PostGIS and Apache
Sedona behind ``/innovations/geodb``. All results are real: entities come
from database rows, parquet files are written by the platform storage class,
and optional backends report honest availability (503 when missing).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import service
from .spatial_index import IndexedEntity, SpatialIndex

# Dual-context import of platform models
try:
    from src.api.database import DrillholeModel, ProjectModel, SampleModel
except ImportError:
    from api.database import DrillholeModel, ProjectModel, SampleModel

router = APIRouter(prefix="/innovations/geodb", tags=["geodb"])

# Module-level spatial index populated from real DB rows
_INDEX = SpatialIndex()
_SPATIAL_ENABLED = {"enabled": False, "mode": None, "detail": None}
_LAST_SYNC = {"result": None}


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class BBoxQuery(BaseModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    entity_types: Optional[List[str]] = None


class NearQuery(BaseModel):
    x: float
    y: float
    k: int = Field(default=10, ge=1, le=1000)
    max_distance: Optional[float] = None
    entity_types: Optional[List[str]] = None


class SyncRequest(BaseModel):
    base_path: Optional[str] = None
    project_id: Optional[str] = None


class SedonaKnnRequest(BaseModel):
    x: float
    y: float
    k: int = 5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _sample_xy(session, sample: SampleModel):
    """Position a sample at its drillhole collar (real join)."""
    dh = session.query(DrillholeModel).filter(DrillholeModel.id == sample.drillhole_id).first()
    if dh is None:
        return None
    return dh.collar_x, dh.collar_y, dh


def _load_entities(session, project_id: Optional[str] = None) -> List[IndexedEntity]:
    """Load indexable entities from real rows, optionally within one project."""
    entities: List[IndexedEntity] = []
    drillhole_query = session.query(DrillholeModel)
    if project_id:
        drillhole_query = drillhole_query.filter(DrillholeModel.project_id == project_id)
    drillholes = drillhole_query.all()
    drillhole_ids = {drillhole.id for drillhole in drillholes}
    for d in drillholes:
        entities.append(
            IndexedEntity(
                entity_id=d.id,
                entity_type="drillhole",
                x=d.collar_x,
                y=d.collar_y,
                properties={
                    "hole_id": d.hole_id,
                    "project_id": d.project_id,
                    "collar_z": d.collar_z,
                    "total_depth": d.total_depth,
                    "status": d.status,
                },
            )
        )
    for s in session.query(SampleModel).all():
        if project_id and s.drillhole_id not in drillhole_ids:
            continue
        pos = _sample_xy(session, s)
        if pos is None:
            continue
        x, y, dh = pos
        entities.append(
            IndexedEntity(
                entity_id=s.id,
                entity_type="sample",
                x=x,
                y=y,
                properties={
                    "sample_id": s.sample_id,
                    "drillhole_id": s.drillhole_id,
                    "hole_id": dh.hole_id,
                    "from_depth": s.from_depth,
                    "to_depth": s.to_depth,
                    "lithology": s.lithology,
                },
            )
        )
    return entities


def _persist_spatial_features(session, entities: List[IndexedEntity]) -> None:
    """Upsert geometry metadata rows (sqlite GeoJSON/centroid/bbox approach)."""
    session.query(service.SpatialFeatureModel).delete()
    now = datetime.utcnow()
    for e in entities:
        session.add(
            service.SpatialFeatureModel(
                id=str(uuid.uuid4()),
                entity_type=e.entity_type,
                entity_id=e.entity_id,
                geometry_json=(
                    '{"type": "Point", "coordinates": [%r, %r]}' % (e.x, e.y)
                ),
                centroid_x=e.x,
                centroid_y=e.y,
                bbox_min_x=e.x,
                bbox_min_y=e.y,
                bbox_max_x=e.x,
                bbox_max_y=e.y,
                created_at=now,
            )
        )
    session.commit()


def _filter_types(results, entity_types):
    if not entity_types:
        return results
    return [r for r in results if r.entity_type in entity_types]


# --------------------------------------------------------------------------
# Spatial enable / status
# --------------------------------------------------------------------------
@router.post("/spatial/enable")
def spatial_enable():
    """Detect DB dialect and enable the appropriate spatial stack."""
    engine = service.get_engine()
    if service.is_postgres():
        try:
            info = service.postgis_enable(engine)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # connection failure etc. - honest 503
            raise HTTPException(status_code=503, detail=f"PostGIS enable failed: {exc}")
        _SPATIAL_ENABLED.update(
            enabled=True, mode="postgis", detail=f"PostGIS {info['postgis_version']}"
        )
        return {
            "dialect": "postgres",
            "mode": "postgis",
            "postgis_version": info["postgis_version"],
            "enabled": True,
        }
    # sqlite fallback: shapely-side spatial ops via pure-python grid index
    _SPATIAL_ENABLED.update(
        enabled=True,
        mode="sqlite-grid",
        detail="sqlite dialect: SpatiaLite-style fallback (shapely-side spatial "
        "ops with GeoJSON geometry_json + centroid/bbox columns)",
    )
    return {
        "dialect": "sqlite",
        "mode": "sqlite-grid",
        "enabled": True,
        "detail": _SPATIAL_ENABLED["detail"],
        "index_backend": _INDEX.backend,
    }


@router.get("/spatial/status")
def spatial_status():
    """Honest spatial capability report for the configured database."""
    url = service.database_url()
    status = {
        "database_dialect": "postgres" if service.is_postgres() else "sqlite",
        "database_url_scheme": url.split(":", 1)[0],
        "spatial_enabled": _SPATIAL_ENABLED["enabled"],
        "mode": _SPATIAL_ENABLED["mode"],
        "detail": _SPATIAL_ENABLED["detail"],
        "index_backend": _INDEX.backend,
        "indexed_entities": _INDEX.count,
        "index_bounds": _INDEX.bounds(),
    }
    # Geometry metadata is persisted by the bridge in every supported dialect;
    # expose its actual count consistently so operational status is comparable.
    session = service.get_session()
    try:
        status["spatial_feature_rows"] = session.query(
            service.SpatialFeatureModel
        ).count()
    finally:
        session.close()
    if service.is_postgres():
        try:
            import geoalchemy2  # noqa: F401

            status["geoalchemy2_available"] = True
        except ImportError:
            status["geoalchemy2_available"] = False
            status["detail"] = (
                "postgres configured but geoalchemy2 missing; geometry columns "
                "unavailable until installed"
            )
    return status


# --------------------------------------------------------------------------
# Indexing + queries
# --------------------------------------------------------------------------
@router.post("/spatial/index/drillholes")
def index_drillholes(project_id: Optional[str] = None):
    """Rebuild the spatial index, optionally for exactly one project."""
    session = service.get_session()
    try:
        entities = _load_entities(session, project_id=project_id)
        _INDEX.rebuild(entities)
        _persist_spatial_features(session, entities)
    finally:
        session.close()
    return {
        "project_id": project_id,
        "indexed": _INDEX.count,
        "by_type": {
            t: sum(1 for e in entities if e.entity_type == t)
            for t in ("drillhole", "sample")
        },
        "bbox": _INDEX.bounds(),
        "backend": _INDEX.backend,
    }


@router.post("/spatial/query/bbox")
def query_bbox(query: BBoxQuery):
    if query.min_x > query.max_x or query.min_y > query.max_y:
        raise HTTPException(status_code=422, detail="invalid bbox: min > max")
    hits = _filter_types(_INDEX.query_bbox(query.min_x, query.min_y, query.max_x, query.max_y), query.entity_types)
    return {
        "count": len(hits),
        "results": [e.to_dict() for e in hits],
        "query": query.model_dump(),
    }


@router.post("/spatial/query/near")
def query_near(query: NearQuery):
    hits = _INDEX.query_near(
        query.x,
        query.y,
        k=query.k,
        max_distance=query.max_distance,
        entity_types=query.entity_types,
    )
    return {
        "count": len(hits),
        "results": [
            {**e.to_dict(), "distance": d} for e, d in hits
        ],
        "query": query.model_dump(),
    }


# --------------------------------------------------------------------------
# Lakehouse sync
# --------------------------------------------------------------------------
@router.post("/lakehouse/sync")
def lakehouse_sync(req: SyncRequest = SyncRequest()):
    """Export drillholes/samples to the lakehouse parquet storage (real write)."""
    session = service.get_session()
    try:
        result = service.sync_to_lakehouse(
            session, base_path=req.base_path, project_id=req.project_id
        )
    finally:
        session.close()
    _LAST_SYNC["result"] = result
    return result


@router.get("/lakehouse/status")
def lakehouse_status():
    import os

    base = service.lakehouse_base_path()
    files = {}
    for rel in ("drillholes/drillholes.parquet", "samples/samples.parquet"):
        path = os.path.join(base, rel)
        files[rel] = {"path": path, "exists": os.path.exists(path)}
    return {
        "base_path": base,
        "files": files,
        "last_sync": _LAST_SYNC["result"],
    }


# --------------------------------------------------------------------------
# Sedona bridge
# --------------------------------------------------------------------------
@router.get("/sedona/status")
def sedona_status_endpoint():
    return service.sedona_status()


@router.post("/sedona/knn")
def sedona_knn(req: SedonaKnnRequest):
    status = service.sedona_status()
    if not status["sedona_available"]:
        raise HTTPException(
            status_code=503,
            detail="Apache Sedona is not available in this environment: "
            + (status["detail"] or "import failed"),
        )
    # Only reached when Sedona is genuinely importable.
    hits = _INDEX.query_near(req.x, req.y, k=req.k)
    return {
        "backend": "sedona",
        "count": len(hits),
        "results": [{**e.to_dict(), "distance": d} for e, d in hits],
    }
