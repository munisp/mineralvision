"""SQLAlchemy model-registry table for trained geoai model metadata.

sqlite is fine (default). Engine/DB URL resolved lazily so tests can point
``GEOAI_REGISTRY_DB`` at a temp file.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class GeoAIModelRegistry(Base):
    __tablename__ = "geoai_model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    task: Mapped[str] = mapped_column(String(64), nullable=False)  # segment/change/detect
    backend: Mapped[str] = mapped_column(String(64), nullable=False)  # geoai/samgeo/torch/cpu
    version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    training_chips: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_engine(db_url: Optional[str] = None):
    url = db_url or os.environ.get("GEOAI_REGISTRY_DB", "sqlite:///geoai_registry.sqlite")
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_url: Optional[str] = None):
    return sessionmaker(bind=get_engine(db_url), future=True)()
