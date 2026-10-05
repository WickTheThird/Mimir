"""SQLAlchemy ORM models for the MIMIR store (ADR 19.3)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Importing the dialect type does not import psycopg, so this stays safe on a
JSONType = JSON().with_variant(JSONB, "postgresql")

_ID = String(64)
_SHORT = String(128)
_MED = String(512)


class Base(DeclarativeBase):
    """Declarative base. ``Base.metadata`` drives schema creation in db.py."""


class SessionRow(Base):
    """An investigation session (ADR 19.3 sessions)."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    title: Mapped[str] = mapped_column(_MED, default="")
    status: Mapped[str] = mapped_column(String(32), index=True, default="active")
    interface: Mapped[str] = mapped_column(String(32), index=True, default="cli")
    user_request: Mapped[str] = mapped_column(Text, default="")
    task_type: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    model_alias: Mapped[str | None] = mapped_column(_SHORT, default=None)

    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    started_at: Mapped[float] = mapped_column(Float, default=0.0)
    completed_at: Mapped[float | None] = mapped_column(Float, default=None)

    iteration: Mapped[int] = mapped_column(Integer, default=0)
    final_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    tags: Mapped[list[str]] = mapped_column(JSONType, default=list)
    # Denormalised lowercase "|tag|tag|" mirror of ``tags``.
    tags_text: Mapped[str] = mapped_column(_MED, index=True, default="")

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class MessageRow(Base):
    """Chat transcript entry. Content is redacted before insert."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session_seq", "session_id", "seq"),)

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(32), index=True, default="user")
    name: Mapped[str | None] = mapped_column(_SHORT, default=None)
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class GraphCheckpointRow(Base):
    """Metadata only (ADR 19.3 graph checkpoints)."""

    __tablename__ = "graph_checkpoints"
    __table_args__ = (
        UniqueConstraint("thread_id", "checkpoint_ns", name="uq_graph_checkpoint_thread"),
    )

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(_ID, index=True)
    checkpoint_ns: Mapped[str] = mapped_column(_SHORT, default="")
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    backend: Mapped[str] = mapped_column(String(32), default="sqlite")
    checkpoint_id: Mapped[str | None] = mapped_column(_ID, default=None)
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class CommandRow(Base):
    """A proposed command plus its policy assessment (ADR 13)."""

    __tablename__ = "commands"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), index=True, default="shell")
    binary: Mapped[str] = mapped_column(_SHORT, index=True, default="")
    display: Mapped[str] = mapped_column(Text, default="")
    argv: Mapped[list[str]] = mapped_column(JSONType, default=list)
    purpose: Mapped[str] = mapped_column(Text, default="")
    expected_effect: Mapped[str] = mapped_column(Text, default="")
    proposed_by: Mapped[str] = mapped_column(_SHORT, default="mimir")
    tool_name: Mapped[str | None] = mapped_column(_SHORT, default=None)
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)

    # Denormalised out of the assessment because the safety review and the audit
    risk: Mapped[str | None] = mapped_column(String(8), index=True, default=None)
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=True)
    forbidden: Mapped[bool] = mapped_column(Boolean, index=True, default=False)

    context_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    assessment_json: Mapped[dict[str, Any] | None] = mapped_column(JSONType, default=None)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class ExecutionRow(Base):
    """What actually ran."""

    __tablename__ = "executions"
    __table_args__ = (Index("ix_executions_session_started", "session_id", "started_at"),)

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, default=None
    )
    command_id: Mapped[str] = mapped_column(_ID, index=True)
    display: Mapped[str] = mapped_column(Text, default="")
    argv: Mapped[list[str]] = mapped_column(JSONType, default=list)
    outcome: Mapped[str] = mapped_column(String(32), index=True, default="success")
    exit_code: Mapped[int | None] = mapped_column(Integer, index=True, default=None)
    risk: Mapped[str] = mapped_column(String(8), index=True, default="R1")
    started_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    duration_s: Mapped[float] = mapped_column(Float, index=True, default=0.0)

    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_ref: Mapped[str | None] = mapped_column(_ID, default=None)

    approval_id: Mapped[str | None] = mapped_column(_ID, index=True, default=None)
    approved_by: Mapped[str | None] = mapped_column(_SHORT, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    context_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    environment_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class ApprovalRow(Base):
    """Approval request and its decision in one row (ADR 13, 15)."""

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, default=None
    )
    command_id: Mapped[str] = mapped_column(_ID, index=True, default="")
    risk: Mapped[str] = mapped_column(String(8), index=True, default="R1")
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending")
    prompt: Mapped[str] = mapped_column(Text, default="")
    requested_by: Mapped[str] = mapped_column(_SHORT, default="mimir")
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    expires_at: Mapped[float | None] = mapped_column(Float, default=None)

    decided_at: Mapped[float | None] = mapped_column(Float, index=True, default=None)
    decided_by: Mapped[str | None] = mapped_column(_SHORT, default=None)
    decision_status: Mapped[str | None] = mapped_column(String(32), index=True, default=None)
    decision_reason: Mapped[str | None] = mapped_column(Text, default=None)
    edited_argv: Mapped[list[str] | None] = mapped_column(JSONType, default=None)

    command_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    assessment_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class EvidenceRow(Base):
    """Evidence item (ADR 11.4, 12)."""

    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint("session_id", "evidence_id", name="uq_evidence_session_item"),
        Index("ix_evidence_session_source", "session_id", "source_type"),
    )

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[str] = mapped_column(_ID, index=True)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    claim: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(32), index=True, default="observed")
    source_type: Mapped[str] = mapped_column(String(32), index=True, default="command_output")
    source_id: Mapped[str] = mapped_column(_MED, index=True, default="")
    excerpt: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, index=True, default=0.6)
    supports: Mapped[bool] = mapped_column(Boolean, index=True, default=True)
    freshness: Mapped[str] = mapped_column(String(16), index=True, default="live")
    collected_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    collected_by: Mapped[str] = mapped_column(_SHORT, default="mimir")
    artifact_ref: Mapped[str | None] = mapped_column(_ID, default=None)
    tags: Mapped[list[str]] = mapped_column(JSONType, default=list)
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSONType, default=None)


class CitationRow(Base):
    """Precise pointer back into a source, one row per citation (ADR 19.3)."""

    __tablename__ = "citations"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_row_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evidence.row_id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)
    source_type: Mapped[str] = mapped_column(String(32), index=True, default="command_output")
    locator: Mapped[str] = mapped_column(Text, default="")
    repo: Mapped[str | None] = mapped_column(_SHORT, default=None)
    path: Mapped[str | None] = mapped_column(_MED, index=True, default=None)
    start_line: Mapped[int | None] = mapped_column(Integer, default=None)
    end_line: Mapped[int | None] = mapped_column(Integer, default=None)
    url: Mapped[str | None] = mapped_column(Text, default=None)
    title: Mapped[str | None] = mapped_column(_MED, default=None)
    retrieved_at: Mapped[float | None] = mapped_column(Float, default=None)


class WebSourceRow(Base):
    """Page consulted during web research (ADR 5.7, 19.3)."""

    __tablename__ = "web_sources"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(_MED, default="")
    query: Mapped[str] = mapped_column(_MED, index=True, default="")
    excerpt: Mapped[str] = mapped_column(Text, default="")
    retrieved_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)


class SkillUsedRow(Base):
    """Skill selected for a session (ADR 19.3 selected skills)."""

    __tablename__ = "skills_used"
    __table_args__ = (UniqueConstraint("session_id", "skill", name="uq_skill_used_session"),)

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    skill: Mapped[str] = mapped_column(_MED, index=True, default="")
    loaded: Mapped[bool] = mapped_column(Boolean, default=False)
    body_chars: Mapped[int] = mapped_column(Integer, default=0)
    selected_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)


class MemoryProposalRow(Base):
    """Candidate knowledge-base note awaiting review (ADR 11.6)."""

    __tablename__ = "memory_proposals"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(_MED, default="")
    category: Mapped[str] = mapped_column(_SHORT, index=True, default="history/investigations")
    body: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list[str]] = mapped_column(JSONType, default=list)
    sources: Mapped[list[str]] = mapped_column(JSONType, default=list)
    verification_status: Mapped[str] = mapped_column(String(32), index=True, default="unverified")
    confidence: Mapped[float] = mapped_column(Float, index=True, default=0.5)
    supersedes: Mapped[str | None] = mapped_column(_ID, default=None)
    approved: Mapped[bool] = mapped_column(Boolean, index=True, default=False)
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)


class AuditEventRow(Base):
    """Append-only record of anything worth answering "who did what" about."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_session_created", "session_id", "created_at"),)

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, default=None
    )
    event_type: Mapped[str] = mapped_column(_SHORT, index=True, default="")
    actor: Mapped[str] = mapped_column(_SHORT, index=True, default="mimir")
    subject: Mapped[str | None] = mapped_column(_MED, index=True, default=None)
    risk: Mapped[str | None] = mapped_column(String(8), index=True, default=None)
    outcome: Mapped[str | None] = mapped_column(String(32), index=True, default=None)
    correlation_id: Mapped[str | None] = mapped_column(_ID, index=True, default=None)
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class ModelCallRow(Base):
    """One LLM round trip: latency, tokens, alias (ADR 20 telemetry)."""

    __tablename__ = "model_calls"
    __table_args__ = (
        Index("ix_model_calls_alias_created", "alias", "created_at"),
        # Identity comes from the router, not from a timestamp.
        Index("uq_model_calls_invocation", "session_id", "invocation_id", unique=True),
    )

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        _ID, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, default=None
    )
    invocation_id: Mapped[str | None] = mapped_column(_ID, default=None)
    alias: Mapped[str] = mapped_column(_SHORT, index=True, default="")
    model: Mapped[str] = mapped_column(_MED, default="")
    runtime: Mapped[str] = mapped_column(String(32), index=True, default="")
    task_class: Mapped[str | None] = mapped_column(_SHORT, index=True, default=None)
    specialist: Mapped[str | None] = mapped_column(_SHORT, index=True, default=None)
    latency_ms: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    total_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    context_size: Mapped[int | None] = mapped_column(Integer, default=None)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    ok: Mapped[bool] = mapped_column(Boolean, index=True, default=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class EvalRunRow(Base):
    """One execution of an evaluation suite (ADR 21)."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    name: Mapped[str] = mapped_column(_MED, default="")
    suite: Mapped[str] = mapped_column(_SHORT, index=True, default="")
    model_alias: Mapped[str | None] = mapped_column(_SHORT, index=True, default=None)
    created_at: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    completed_at: Mapped[float | None] = mapped_column(Float, default=None)
    total: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class EvalResultRow(Base):
    """One case within an evaluation run (ADR 21.1)."""

    __tablename__ = "eval_results"
    __table_args__ = (Index("ix_eval_results_run_case", "run_id", "case_id"),)

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(_SHORT, index=True, default="")
    category: Mapped[str] = mapped_column(_SHORT, index=True, default="")
    passed: Mapped[bool] = mapped_column(Boolean, index=True, default=False)
    score: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    # Eval sessions are pruned on their own schedule, so this is an unenforced
    session_id: Mapped[str | None] = mapped_column(_ID, index=True, default=None)
    expected: Mapped[str] = mapped_column(Text, default="")
    actual: Mapped[str] = mapped_column(Text, default="")
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


# : Child tables of ``sessions``, in delete order (leaves first).
SESSION_CHILD_TABLES: tuple[type[Base], ...] = (
    CitationRow,
    EvidenceRow,
    ExecutionRow,
    ApprovalRow,
    CommandRow,
    MessageRow,
    WebSourceRow,
    SkillUsedRow,
    MemoryProposalRow,
    ModelCallRow,
    AuditEventRow,
    GraphCheckpointRow,
)

__all__ = [
    "SESSION_CHILD_TABLES",
    "ApprovalRow",
    "AuditEventRow",
    "Base",
    "CitationRow",
    "CommandRow",
    "EvalResultRow",
    "EvalRunRow",
    "EvidenceRow",
    "ExecutionRow",
    "GraphCheckpointRow",
    "JSONType",
    "MemoryProposalRow",
    "MessageRow",
    "ModelCallRow",
    "SessionRow",
    "SkillUsedRow",
    "WebSourceRow",
]
