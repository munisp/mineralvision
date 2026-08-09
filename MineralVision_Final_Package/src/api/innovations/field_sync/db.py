"""Database plumbing for the field_sync innovation."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = os.getenv("MV_SYNC_DATABASE_URL", "sqlite:///./field_sync.db")
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
