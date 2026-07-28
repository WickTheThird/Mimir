"""Persistence layer (ADR 19).

SQLite by default, PostgreSQL when configured. The ORM rows live in
:mod:`~mimir.persistence.models`, the engine in :mod:`~mimir.persistence.db`,
the typed access layer in :mod:`~mimir.persistence.repositories`, and LangGraph
checkpointer selection in :mod:`~mimir.persistence.checkpoint`.
"""

from mimir.persistence.checkpoint import (
    async_checkpointer,
    checkpoint_backend,
    checkpointer_scope,
    get_checkpointer,
    thread_config,
)
from mimir.persistence.db import Database, get_database, reset_database
from mimir.persistence.models import Base
from mimir.persistence.repositories import (
    ApprovalRepository,
    AuditRepository,
    CommandRepository,
    EvalRepository,
    EvidenceRepository,
    ExecutionRepository,
    MessageRepository,
    ModelCallRepository,
    PersistenceService,
    PruneReport,
    SessionRepository,
    get_persistence,
    load_state,
    reset_persistence,
    save_state,
)

__all__ = [
    "ApprovalRepository",
    "AuditRepository",
    "Base",
    "CommandRepository",
    "Database",
    "EvalRepository",
    "EvidenceRepository",
    "ExecutionRepository",
    "MessageRepository",
    "ModelCallRepository",
    "PersistenceService",
    "PruneReport",
    "SessionRepository",
    "async_checkpointer",
    "checkpoint_backend",
    "checkpointer_scope",
    "get_checkpointer",
    "get_database",
    "get_persistence",
    "load_state",
    "reset_database",
    "reset_persistence",
    "save_state",
    "thread_config",
]
