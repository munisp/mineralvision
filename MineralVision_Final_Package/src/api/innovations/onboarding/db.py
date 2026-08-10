"""Database plumbing for the onboarding innovation.

Follows the geodb service pattern: sessions are resolved lazily from the
``MV_ONBOARDING_DATABASE_URL`` (preferred) or ``DATABASE_URL`` environment
variables, falling back to a local sqlite file. The module keeps its OWN
declarative Base — deliberately NOT the platform Base — so the onboarding
tables never create schema drift against the platform alembic migrations.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()  # module-private metadata (see docstring)

_engine = None
_SessionLocal = None


def _resolve_database_url() -> str:
    url = os.getenv("MV_ONBOARDING_DATABASE_URL") or os.getenv("DATABASE_URL")
    if url:
        return url
    return "sqlite:///./onboarding.db"


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = _resolve_database_url()
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
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
