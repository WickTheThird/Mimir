"""LangGraph checkpointer selection (ADR 12, 14.1, 19.2).

LangGraph owns its own checkpoint tables and migrates them itself, so MIMIR does
not model them. This module only decides which saver a graph gets:

* SQLite at ``Settings.checkpoint_path`` by default (ADR 19.1).
* PostgreSQL when ``persistence.url`` points at Postgres and the optional
  ``langgraph-checkpoint-postgres`` package is installed (ADR 19.2).
* An in-memory saver with a loud warning if neither is importable, so a missing
  optional dependency degrades the durability of a resume rather than stopping
  the run.

Checkpoint files are kept beside, not inside, the main database. The two have
different write patterns and LangGraph's schema is not ours to migrate.

Verified against langgraph 1.2.x / langgraph-checkpoint-sqlite 3.x, where
``SqliteSaver`` wraps a ``sqlite3.Connection`` and ``AsyncSqliteSaver`` wraps an
``aiosqlite.Connection``. Both ``from_conn_string`` helpers are context managers
that close the connection on exit, which is why the scoped accessors below are
context managers too.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)

#: Same reasoning as the main database: WAL plus a busy timeout is what lets the
#: web UI and the CLI checkpoint the same graph without lock errors.
_SQLITE_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=10000",
    "PRAGMA synchronous=NORMAL",
)



# LangGraph warns on deserialising unregistered types today and will refuse in a
# future version, which would silently break `mimir session resume`. Declaring
# them keeps resume working and is the safer posture besides: the allowlist is
# what stops a tampered checkpoint from instantiating arbitrary classes.
def _mimir_checkpoint_types() -> tuple[type, ...]:
    """The domain types that appear inside a checkpointed investigation.

    Passing the classes themselves rather than module strings keeps the list
    honest: a renamed class fails at import here instead of silently dropping
    out of the allowlist and breaking resume months later.
    """
    from mimir.models.approval import ApprovalDecision, ApprovalRequest
    from mimir.models.command import (
        CommandKind,
        CommandOutcome,
        ExecutionRecord,
        PolicyViolation,
        ProposedCommand,
        RiskAssessment,
        RiskClass,
        TargetContext,
    )
    from mimir.models.evidence import (
        Citation,
        Evidence,
        EvidenceKind,
        Freshness,
        SourceType,
    )
    from mimir.models.session import ChatMessage, MessageRole, Session, SessionStatus
    from mimir.models.specialist import (
        CoordinatorPlan,
        FinalAnswer,
        Hypothesis,
        HypothesisStatus,
        PlannedStep,
        SpecialistName,
        SpecialistReport,
        TaskType,
    )
    from mimir.models.state import (
        EnvironmentContext,
        InvestigationState,
        MemoryProposal,
        WebSource,
    )

    return (
        ApprovalDecision, ApprovalRequest,
        CommandKind, CommandOutcome, ExecutionRecord, PolicyViolation,
        ProposedCommand, RiskAssessment, RiskClass, TargetContext,
        Citation, Evidence, EvidenceKind, Freshness, SourceType,
        ChatMessage, MessageRole, Session, SessionStatus,
        CoordinatorPlan, FinalAnswer, Hypothesis, HypothesisStatus, PlannedStep,
        SpecialistName, SpecialistReport, TaskType,
        EnvironmentContext, InvestigationState, MemoryProposal, WebSource,
    )


def _serde() -> Any:
    """A serialiser that knows about MIMIR's own state types."""
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError:  # pragma: no cover - very old langgraph
        return None
    try:
        return JsonPlusSerializer(allowed_msgpack_modules=_mimir_checkpoint_types())
    except TypeError:
        # Older builds do not accept the argument; the default is permissive,
        # so checkpointing still works and only the warning remains.
        return JsonPlusSerializer()


def _is_postgres(settings: Settings) -> bool:
    url = settings.database_url
    return url.startswith("postgresql") or url.startswith("postgres://")


def _prepare_sqlite_path(settings: Settings) -> Path:
    path = settings.checkpoint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _memory_saver(reason: str) -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    log.warning("checkpoint.memory_fallback", reason=reason)
    return MemorySaver()


def _postgres_saver(settings: Settings) -> Any | None:
    """Return a set-up ``PostgresSaver``, or ``None`` if the extra is absent."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        return None
    # from_conn_string is a context manager in every released version; entering
    # it manually keeps the connection alive for the lifetime of the process,
    # which is what a long-lived saver needs.
    manager = PostgresSaver.from_conn_string(settings.database_url)
    saver = manager.__enter__()
    saver.setup()
    return saver


def get_checkpointer(settings: Settings | None = None) -> Any:
    """A synchronous checkpointer suitable for a long-lived process.

    The returned saver owns its connection and stays usable until the process
    exits. Use :func:`checkpointer_scope` instead when the lifetime is bounded
    and the connection should be closed deterministically.
    """
    settings = settings or get_settings()

    if _is_postgres(settings):
        saver = _postgres_saver(settings)
        if saver is not None:
            return saver
        log.warning("checkpoint.postgres_unavailable", hint="pip install 'mimir[postgres]'")

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        return _memory_saver(f"langgraph-checkpoint-sqlite is not installed ({exc})")

    path = _prepare_sqlite_path(settings)
    # check_same_thread=False because the API threadpool and the CLI both reach
    # the same saver instance.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    for pragma in _SQLITE_PRAGMAS:
        conn.execute(pragma)
    serde = _serde()
    saver = SqliteSaver(conn, serde=serde) if serde is not None else SqliteSaver(conn)
    saver.setup()
    return saver


@contextmanager
def checkpointer_scope(settings: Settings | None = None) -> Iterator[Any]:
    """Synchronous checkpointer with a deterministic close.

    Preferred for one-shot CLI runs and tests.
    """
    settings = settings or get_settings()
    saver = get_checkpointer(settings)
    try:
        yield saver
    finally:
        close_checkpointer(saver)


@asynccontextmanager
async def async_checkpointer(settings: Settings | None = None) -> AsyncIterator[Any]:
    """Async checkpointer for async graphs.

    ``AsyncSqliteSaver`` needs an ``aiosqlite`` connection bound to the running
    event loop, so this one is a context manager rather than a plain accessor.
    """
    settings = settings or get_settings()

    if _is_postgres(settings):
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:
            log.warning("checkpoint.postgres_unavailable", hint="pip install 'mimir[postgres]'")
        else:
            async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
                await saver.setup()
                yield saver
                return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        yield _memory_saver(f"langgraph-checkpoint-sqlite[aio] is not installed ({exc})")
        return

    path = _prepare_sqlite_path(settings)
    serde = _serde()
    # from_conn_string is itself an async context manager. Driving it manually
    # rather than with `async with` means this generator owns exactly one
    # __aexit__ call, and a caller that abandons the generator (or lets the loop
    # finalise it during shutdown) cannot trigger a second concurrent athrow on
    # the inner one. That double-close is what raises "asynchronous generator is
    # already running" and prints a traceback over the operator's answer.
    manager = AsyncSqliteSaver.from_conn_string(str(path))
    saver = await manager.__aenter__()
    closed = False
    try:
        if serde is not None:
            saver.serde = serde
        for pragma in _SQLITE_PRAGMAS:
            await saver.conn.execute(pragma)
        await saver.setup()
        yield saver
    except GeneratorExit:
        # Abandoned mid-yield. Close the saver and let the exit propagate; do
        # not await anything else, since the loop may already be tearing down.
        closed = True
        with suppress(Exception):
            await manager.__aexit__(None, None, None)
        raise
    finally:
        if not closed:
            with suppress(Exception):
                await manager.__aexit__(None, None, None)


def close_checkpointer(saver: Any) -> None:
    """Best-effort close. Savers differ in whether they expose one."""
    conn = getattr(saver, "conn", None)
    close = getattr(conn, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - each driver raises its own close errors
            log.debug("checkpoint.close_failed", error=str(exc))


def checkpoint_backend(settings: Settings | None = None) -> str:
    """Which backend :func:`get_checkpointer` would pick. Cheap, opens nothing."""
    settings = settings or get_settings()
    if _is_postgres(settings):
        try:
            import langgraph.checkpoint.postgres
        except ImportError:
            pass
        else:
            return "postgres"
    try:
        import langgraph.checkpoint.sqlite  # noqa: F401
    except ImportError:
        return "memory"
    return "sqlite"


def thread_config(session_id: str, **extra: Any) -> dict[str, Any]:
    """The LangGraph config that binds a run to a session's thread (ADR 14.1)."""
    return {"configurable": {"thread_id": session_id, **extra}}


__all__ = [
    "async_checkpointer",
    "checkpoint_backend",
    "checkpointer_scope",
    "close_checkpointer",
    "get_checkpointer",
    "thread_config",
]
