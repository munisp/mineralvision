"""Integration-hub persistence bootstrap.

The hub maintains isolated connector evidence only when explicitly configured.
Production deployments must supply a PostgreSQL URL; silently creating a local
SQLite database would break tenant lineage, backup, and operational guarantees.
SQLite remains available only for explicitly non-production local/test usage.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()

_engine = None
_SessionLocal = None


def _database_url() -> str:
    url = os.getenv("MV_HUB_DATABASE_URL")
    env = os.getenv("ENV", "development").lower()
    if url:
        if env in {"production", "preproduction", "staging"} and url.startswith("sqlite"):
            raise RuntimeError("MV_HUB_DATABASE_URL must be PostgreSQL outside development/test")
        return url
    if env in {"production", "preproduction", "staging"}:
        raise RuntimeError(
            "MV_HUB_DATABASE_URL is required outside development/test; "
            "refusing standalone SQLite integration evidence storage"
        )
    return "sqlite:///./integration_hub.db"


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = _database_url()
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
            pool_pre_ping=True,
        )
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionLocal


def get_db():
    factory = get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()
