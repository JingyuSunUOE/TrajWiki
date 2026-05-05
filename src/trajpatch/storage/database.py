"""Database setup utilities for TrajPatch."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from .models import Base, IndexBase


def build_engine(database_path: Path, *, profile: str = "default"):
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine_kwargs = {"future": True}
    if profile == "worker_shard":
        engine_kwargs["poolclass"] = NullPool
    engine = create_engine(f"sqlite:///{database_path}", **engine_kwargs)
    if profile == "worker_shard":
        @event.listens_for(engine, "connect")
        def _configure_worker_shard_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
    return engine


def create_schema(database_path: Path, *, profile: str = "default") -> sessionmaker[Session]:
    engine = build_engine(database_path, profile=profile)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_index_schema(database_path: Path) -> sessionmaker[Session]:
    engine = build_engine(database_path)
    IndexBase.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
