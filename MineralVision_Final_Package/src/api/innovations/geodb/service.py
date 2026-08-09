"""
geodb bridge service layer.

Unifies the MineralVision API database (SQLAlchemy), the lakehouse parquet
storage, PostGIS (when configured) and Apache Sedona (when importable).

Database sessions are resolved lazily from the ``DATABASE_URL`` environment
variable so tests (and deployments) can point the bridge at any SQLAlchemy
DSN; when unset it falls back to the platform ``database`` module's engine.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine, text
from sqlalchemy.orm import sessionmaker

# Dual-context import of the platform database models (for entity queries).
# NOTE: we deliberately do NOT use the platform Base for our auxiliary table —
# registering on it would create schema drift against the alembic migrations.
try:  # running inside the FastAPI package layout (src/ on sys.path)
    from src.api.database import DrillholeModel, ProjectModel, SampleModel
except ImportError:  # running with MineralVision_Final_Package/src on sys.path
    from api.database import DrillholeModel, ProjectModel, SampleModel

from sqlalchemy.orm import declarative_base

Base = declarative_base()  # module-private metadata for auxiliary tables


# ---------------------------------------------------------------------------
# Geometry metadata table (sqlite fallback: GeoJSON + centroid + bbox columns)
# ---------------------------------------------------------------------------
class SpatialFeatureModel(Base):
    """Spatial metadata for any platform entity.

    On sqlite this acts as the SpatiaLite-style substitute: geometry stored as
    GeoJSON text plus denormalised centroid/bbox columns for fast filtering.
    """

    __tablename__ = "spatial_features"

    id = Column(String(36), primary_key=True)
    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(String(36), nullable=False, index=True)
    geometry_json = Column(Text, nullable=False)  # GeoJSON geometry
    centroid_x = Column(Float, nullable=False)
    centroid_y = Column(Float, nullable=False)
    bbox_min_x = Column(Float, nullable=False)
    bbox_min_y = Column(Float, nullable=False)
    bbox_max_x = Column(Float, nullable=False)
    bbox_max_y = Column(Float, nullable=False)
    created_at = Column(DateTime)


# ---------------------------------------------------------------------------
# Engine / session resolution
# ---------------------------------------------------------------------------
_engine_cache: Dict[str, Any] = {}


def database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./mineralvision.db")


def is_postgres() -> bool:
    return database_url().startswith(("postgres://", "postgresql://"))


def get_engine():
    url = database_url()
    if url not in _engine_cache:
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_engine(url, echo=False, **kwargs)
        Base.metadata.create_all(engine)
        _engine_cache[url] = engine
    return _engine_cache[url]


def get_session():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())()


# ---------------------------------------------------------------------------
# Lakehouse parquet storage (MineralVision_Enhanced, real write)
# ---------------------------------------------------------------------------
def _load_parquet_storage_class():
    """Import the platform ParquetStorage with dual-context fallbacks."""
    try:
        from data_storage.parquet_storage import ParquetConfig, ParquetStorage

        return ParquetStorage, ParquetConfig
    except ImportError:
        pass
    try:
        from MineralVision_Enhanced.lakehouse_architecture.data_storage.parquet_storage import (  # noqa: E501
            ParquetConfig,
            ParquetStorage,
        )

        return ParquetStorage, ParquetConfig
    except ImportError:
        pass
    # last resort: resolve by file location relative to the repo root
    import sys

    here = os.path.abspath(__file__)
    # .../repo/MineralVision_Final_Package/src/api/innovations/geodb/service.py
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here)))))
    candidate = os.path.join(
        os.path.dirname(repo_root), "MineralVision_Enhanced", "lakehouse_architecture"
    )
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    from data_storage.parquet_storage import ParquetConfig, ParquetStorage

    return ParquetStorage, ParquetConfig


def lakehouse_base_path() -> str:
    return os.getenv("GEODB_LAKEHOUSE_PATH", os.path.join(os.getcwd(), "lakehouse_data"))


def sync_to_lakehouse(session, base_path: Optional[str] = None) -> Dict[str, Any]:
    """Export drillholes + samples from the API DB to lakehouse parquet files.

    Uses the real MineralVision_Enhanced ParquetStorage class. Returns paths
    and row counts of the files actually written.
    """
    ParquetStorage, ParquetConfig = _load_parquet_storage_class()
    base = base_path or lakehouse_base_path()
    storage = ParquetStorage(ParquetConfig(base_path=base))

    drillholes = session.query(DrillholeModel).all()
    dh_rows = [
        {
            "id": d.id,
            "hole_id": d.hole_id,
            "project_id": d.project_id,
            "collar_x": d.collar_x,
            "collar_y": d.collar_y,
            "collar_z": d.collar_z,
            "total_depth": d.total_depth,
            "azimuth": d.azimuth,
            "dip": d.dip,
            "status": d.status,
            "assay_count": d.assay_count,
        }
        for d in drillholes
    ]
    samples = session.query(SampleModel).all()
    sample_rows = [
        {
            "id": s.id,
            "sample_id": s.sample_id,
            "drillhole_id": s.drillhole_id,
            "from_depth": s.from_depth,
            "to_depth": s.to_depth,
            "sample_type": s.sample_type,
            "lithology": s.lithology,
            "assay_data": json.dumps(s.assay_data or {}),
        }
        for s in samples
    ]

    storage.write_parquet("drillholes/drillholes.parquet", dh_rows)
    storage.write_parquet("samples/samples.parquet", sample_rows)
    return {
        "base_path": base,
        "drillholes": {
            "path": os.path.join(base, "drillholes", "drillholes.parquet"),
            "row_count": len(dh_rows),
        },
        "samples": {
            "path": os.path.join(base, "samples", "samples.parquet"),
            "row_count": len(sample_rows),
        },
    }


# ---------------------------------------------------------------------------
# PostGIS helpers (lazy, honest)
# ---------------------------------------------------------------------------
def postgis_enable(engine) -> Dict[str, Any]:
    """Enable PostGIS on a Postgres engine. Raises RuntimeError honestly."""
    try:
        import geoalchemy2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Postgres DSN configured but geoalchemy2 is not installed"
        ) from exc
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        version = conn.execute(text("SELECT PostGIS_Version()")).scalar()
        conn.commit()
    return {"postgis_version": version}


def sedona_status() -> Dict[str, Any]:
    """Honestly probe Apache Sedona availability (platform module + real lib)."""
    result: Dict[str, Any] = {
        "sedona_available": False,
        "platform_module_available": False,
        "version": None,
        "detail": None,
    }
    try:
        import sedona  # noqa: F401

        result["sedona_available"] = True
        result["version"] = getattr(sedona, "__version__", None) or getattr(
            sedona, "version", None
        )
    except ImportError as exc:
        result["detail"] = f"apache-sedona not importable: {exc}"
    # platform integration module performs the same probe at import time
    try:
        try:
            from middleware.geospatial import sedona_integration as si
        except ImportError:
            import sys

            here = os.path.abspath(__file__)
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
            )
            candidate = os.path.join(os.path.dirname(repo_root), "MineralVision_Enhanced")
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            from middleware.geospatial import sedona_integration as si
        result["platform_module_available"] = bool(getattr(si, "SEDONA_AVAILABLE", False))
    except Exception as exc:  # module itself may fail; report honestly
        result["platform_module_available"] = False
        if not result["detail"]:
            result["detail"] = f"platform sedona_integration probe failed: {exc}"
    return result
