"""
novelty_detector/storage/database.py
=======================================
Database Engine & Session Management
--------------------------------------
Provides a single SQLAlchemy engine and a thread-safe ``Session`` factory
(``SessionLocal``) shared across the whole application.

Key patterns used
-----------------
- **Engine singleton** — created once at module import using ``settings.database_url``.
- **``get_session()`` context manager** — ensures sessions are always committed
  or rolled back and closed, even on exceptions.
- **``init_db()``** — creates all tables on first run (idempotent).

Flask integration
-----------------
In the Flask app factory (Phase 3) call ``init_db()`` inside ``create_app()``.
Use ``get_session()`` inside each request handler:

    with get_session() as session:
        records = session.query(DocumentRecord).all()

Thread-safety
-------------
``sessionmaker`` creates a *new* session per call. Each thread/request should
use its own session obtained via ``get_session()``.  Do NOT share sessions
across threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from novelty_detector.config import settings
from novelty_detector.storage.models import Base


# ─────────────────────────────────────────────────────────────────────────────
# Engine (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

def _create_engine() -> Engine:
    """
    Build the SQLAlchemy engine from the configured DATABASE_URL.

    SQLite-specific settings
    ------------------------
    - ``check_same_thread=False`` is required for multi-threaded Flask apps.
    - WAL journal mode improves concurrent read performance.

    PostgreSQL / other
    ------------------
    Connection pooling (``pool_size``, ``max_overflow``) can be added here
    when switching to a production database.
    """
    url = settings.database_url
    is_sqlite = url.startswith("sqlite")

    connect_args = {"check_same_thread": False} if is_sqlite else {}

    engine = create_engine(
        url,
        connect_args=connect_args,
        echo=False,  # Set True to log SQL for debugging
    )

    # Enable WAL mode for SQLite — better concurrent reads for the dashboard.
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _set_wal(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.close()

    logger.info("Database engine created | url={}", url)
    return engine


engine: Engine = _create_engine()

# Session factory — call ``SessionLocal()`` to get a new session.
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,  # explicit commit required
    autoflush=False,   # flush only on commit/explicit flush
    expire_on_commit=False,  # keep objects accessible after commit
)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables that do not already exist.

    This is idempotent — safe to call on every app startup.
    In production with migrations, replace this with ``alembic upgrade head``.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session with automatic
    commit / rollback and guaranteed close.

    Usage
    -----
        with get_session() as session:
            session.add(some_model_instance)
            # session.commit() is called automatically on clean exit

    Raises
    ------
    Any exception raised inside the ``with`` block is re-raised after
    rolling back the transaction and closing the session.
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Session rolled back due to an exception.")
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """
    Verify database connectivity.

    Returns
    -------
    bool
        True if a test query succeeds, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection check passed.")
        return True
    except Exception as exc:
        logger.error("Database connection check failed: {}", exc)
        return False
