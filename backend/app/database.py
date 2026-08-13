"""SQLAlchemy wiring.

SQLite is the zero-infra dev default; set ``DATABASE_URL`` to a Postgres DSN
(Supabase/Render) in production. The models and session code are identical for
both — only the URL changes.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

_url = settings.DATABASE_URL
_engine_kwargs = {"future": True}

if _url.startswith("sqlite"):
    # SQLite is accessed from multiple threads (request handlers + background
    # audit jobs); disable the per-thread check.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    if ":memory:" in _url:
        # In-memory SQLite must share one connection across threads.
        from sqlalchemy.pool import StaticPool

        _engine_kwargs["poolclass"] = StaticPool
elif _url.startswith("postgresql"):
    _engine_kwargs["pool_pre_ping"] = True

engine = create_engine(_url, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they do not exist (call on app startup)."""
    from . import models  # noqa: F401 — register models on Base

    Base.metadata.create_all(bind=engine)
