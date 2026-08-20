"""Alembic environment for the MineralVision API database.

Targets ``src.api.database.Base.metadata`` and resolves the required PostgreSQL
URL from the DATABASE_URL environment variable. A URL passed via config (``-x dburl=`` or
``config.set_main_option``) takes precedence — used by tests.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `src.api` importable regardless of the invoking cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.database import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required for Alembic migrations")
    if not url.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://")):
        raise RuntimeError("Alembic migrations require a PostgreSQL DATABASE_URL")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a connection)."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
