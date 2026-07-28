"""Engine, session factory, and schema management (ADR 19.1, 19.2).

SQLite is the default backend for development and single-user installs
(ADR 19.1). PostgreSQL is supported for the mature deployment (ADR 19.2) and is
selected purely by ``persistence.url``; the ORM layer above is identical either
way. ``psycopg`` is an optional extra, so nothing in this module imports it at
module scope.

SQLite needs three pragmas to survive the concurrency ADR 19.2 asks for
(web UI and CLI writing at the same time). They are set per connection in
:func:`_configure_sqlite`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, event, inspect, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.persistence.models import Base

log = get_logger(__name__)

#: How long SQLite waits on a locked database before raising. Long enough that a
#: CLI write does not fail while the web UI holds the write lock, short enough
#: that a genuine deadlock still surfaces.
SQLITE_BUSY_TIMEOUT_MS = 10_000


def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
    """Per-connection pragmas.

    WAL is the important one: without it, readers block writers and the moment
    the web UI and the CLI touch the database together SQLite raises
    "database is locked" (ADR 19.2 wants concurrent web and CLI access).
    ``foreign_keys=ON`` is off by default in SQLite and the retention cascade
    depends on it. ``busy_timeout`` turns instant lock failures into a wait.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        # NORMAL is the recommended durability level under WAL: a crash can lose
        # the last transaction but never corrupts the file.
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


class Database:
    """Owns one engine and hands out short-lived sessions.

    The engine is created once and shared across threads (it holds the
    connection pool); each unit of work takes its own ORM session from
    :meth:`session`. Do not keep a session alive across a request or a graph
    node.
    """

    def __init__(self, url: str | None = None, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.url = url or self.settings.database_url
        self._parsed = make_url(self.url)
        self.dialect = self._parsed.get_backend_name()
        self.is_sqlite = self.dialect == "sqlite"
        self.is_postgres = self.dialect in ("postgresql", "postgres")
        self.engine = self._make_engine()
        self._sessionmaker = sessionmaker(
            bind=self.engine, expire_on_commit=False, class_=OrmSession
        )
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    # -- construction ----------------------------------------------------

    def _make_engine(self) -> Engine:
        from sqlalchemy import create_engine

        kwargs: dict[str, Any] = {"echo": self.settings.persistence.echo_sql, "future": True}
        if self.is_sqlite:
            path = self._parsed.database or ""
            in_memory = path in ("", ":memory:")
            # check_same_thread=False because one shared pool serves the API
            # threadpool, the CLI, and background jobs.
            kwargs["connect_args"] = {"check_same_thread": False}
            if in_memory:
                # A memory database dies with its connection, so every session
                # must reuse the same one.
                kwargs["poolclass"] = StaticPool
            else:
                Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            kwargs["pool_pre_ping"] = True
            kwargs["pool_size"] = 5
            kwargs["max_overflow"] = 10

        try:
            engine = create_engine(self.url, **kwargs)
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                f"database driver for {self.url!r} is not installed. "
                "Install the postgres extra: pip install 'mimir[postgres]'"
            ) from exc

        if self.is_sqlite:
            event.listen(engine, "connect", _configure_sqlite)
        return engine

    # -- schema ----------------------------------------------------------

    def create_all(self) -> None:
        """Create any missing table. Safe to call repeatedly."""
        Base.metadata.create_all(self.engine)

    def migrate(self) -> list[str]:
        """Add columns that exist in the ORM but not yet in the database.

        Deliberately lightweight: MIMIR is a local-first single-binary tool and
        a full Alembic setup is more machinery than a personal install needs.
        This covers the only migration shape that has come up so far, which is a
        new nullable column on an existing table. Anything destructive (dropped
        or retyped columns) is reported and left alone for a human.

        Returns the DDL statements applied.
        """
        applied: list[str] = []
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    continue
                have = {col["name"] for col in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name in have:
                        continue
                    if not column.nullable and column.server_default is None:
                        log.warning(
                            "persistence.migrate.skipped_not_null",
                            table=table.name,
                            column=column.name,
                        )
                        continue
                    ddl = self._add_column_ddl(table.name, column.name, column.type)
                    conn.execute(text(ddl))
                    applied.append(ddl)
                    log.info("persistence.migrate.column_added", table=table.name,
                             column=column.name)
        return applied

    def _add_column_ddl(self, table: str, column: str, type_: Any) -> str:
        compiled = type_.compile(dialect=self.engine.dialect)
        return f'ALTER TABLE "{table}" ADD COLUMN "{column}" {compiled}'

    def ensure_schema(self) -> None:
        """Create then migrate, once per process."""
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.create_all()
            self.migrate()
            self._schema_ready = True

    # -- sessions --------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator[OrmSession]:
        """One unit of work. Commits on success, rolls back on any exception."""
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        with self.engine.begin() as conn:
            yield conn

    # -- diagnostics -----------------------------------------------------

    def pragma(self, name: str) -> Any:
        """Read a SQLite pragma. Returns ``None`` on other backends."""
        if not self.is_sqlite:
            return None
        with self.engine.connect() as conn:
            row = conn.execute(text(f"PRAGMA {name}")).first()
        return row[0] if row else None

    def health(self) -> dict[str, Any]:
        """Backend facts worth showing in ``mimir doctor`` (ADR 20)."""
        info: dict[str, Any] = {
            "url": self._parsed.render_as_string(hide_password=True),
            "dialect": self.dialect,
            "tables": len(Base.metadata.tables),
        }
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            info["reachable"] = True
        except SQLAlchemyError as exc:
            info["reachable"] = False
            info["error"] = str(exc)
            return info
        if self.is_sqlite:
            info["journal_mode"] = self.pragma("journal_mode")
            info["foreign_keys"] = bool(self.pragma("foreign_keys"))
            info["busy_timeout"] = self.pragma("busy_timeout")
        return info

    def dispose(self) -> None:
        self.engine.dispose()


_database: Database | None = None
_database_lock = threading.Lock()


def get_database(settings: Settings | None = None, *, url: str | None = None) -> Database:
    """Process-wide database handle. The engine is built exactly once."""
    global _database
    if _database is None:
        with _database_lock:
            if _database is None:
                db = Database(url, settings=settings)
                db.ensure_schema()
                _database = db
    return _database


def reset_database() -> None:
    """Drop the shared handle. Used by tests and by ``mimir config reload``."""
    global _database
    with _database_lock:
        if _database is not None:
            _database.dispose()
        _database = None


__all__ = [
    "SQLITE_BUSY_TIMEOUT_MS",
    "Database",
    "get_database",
    "reset_database",
]
