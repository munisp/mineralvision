"""Database plumbing for the custody ledger innovation.

Self-contained SQLAlchemy engine/session factory so the module works
standalone (tests override the dependency or set the env URL).
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine():
    """Lazily create the engine and ensure the schema exists."""
    global _engine, _SessionLocal
    if _engine is None:
        url = os.getenv("MV_LEDGER_DATABASE_URL", "sqlite:///./custody_ledger.db")
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
    """FastAPI dependency yielding a session."""
    factory = get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()
