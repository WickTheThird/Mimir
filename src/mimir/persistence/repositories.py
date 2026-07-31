"""Typed repositories over the ORM (ADR 19.3, 20).

Every repository takes a :class:`~mimir.persistence.db.Database` and opens a
short-lived session per call, so a repository instance is safe to hold for the
lifetime of a process or a request.

Two rules are enforced here rather than by convention:

* Nothing sensitive is persisted unredacted. Message content, command output,
  and evidence excerpts go through :func:`mimir.redaction.redact` on the way in
  (ADR 13.4).
* :class:`ExecutionRepository` is append-only. It has no update or delete
  method, because that table is the audit trail.

:class:`PersistenceService` sits on top and provides the pair the CLI and API
both need to resume a session (ADR 14.1): :meth:`~PersistenceService.save_state`
and :meth:`~PersistenceService.load_state`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session as OrmSession

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.approval import ApprovalDecision, ApprovalRequest, ApprovalStatus
from mimir.models.command import (
    CommandOutcome,
    ExecutionRecord,
    ProposedCommand,
    RiskAssessment,
    RiskClass,
)
from mimir.models.evidence import Citation, Evidence, SourceType
from mimir.models.session import ChatMessage, Session, SessionStatus
from mimir.models.state import InvestigationState, MemoryProposal, WebSource
from mimir.persistence.db import Database, get_database
from mimir.persistence.models import (
    SESSION_CHILD_TABLES,
    ApprovalRow,
    AuditEventRow,
    CitationRow,
    CommandRow,
    EvalResultRow,
    EvalRunRow,
    EvidenceRow,
    ExecutionRow,
    GraphCheckpointRow,
    MemoryProposalRow,
    MessageRow,
    ModelCallRow,
    SessionRow,
    SkillUsedRow,
    WebSourceRow,
)
from mimir.redaction import redact, redact_mapping

log = get_logger(__name__)

#: InvestigationState fields that have a column or a child table of their own.
#: Everything else is written to ``sessions.state_json``. Keeping the list in one
#: place is what stops a field from being stored twice or lost entirely.
_STATE_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "user_request",
        "task_type",
        "interface",
        "messages",
        "evidence",
        "commands_planned",
        "commands_executed",
        "model_calls",
        "approvals",
        "approval_decisions",
        "selected_skills",
        "web_sources",
        "memory_proposals",
        "iteration",
        "started_at",
        "completed_at",
        "error",
        "final_confidence",
    }
)


def _tags_text(tags: Iterable[str]) -> str:
    """Pipe-delimited lowercase mirror of a tag list, for portable LIKE search."""
    cleaned = sorted({t.strip().lower() for t in tags if t and t.strip()})
    return "|" + "|".join(cleaned) + "|" if cleaned else ""


# -- conversion helpers ---------------------------------------------------


def session_to_row(session: Session, row: SessionRow | None = None) -> SessionRow:
    row = row or SessionRow(id=session.id)
    row.title = session.title
    row.status = str(session.status)
    row.interface = session.interface
    row.user_request = session.user_request
    row.task_type = session.task_type
    row.model_alias = session.model_alias
    row.created_at = session.created_at
    row.updated_at = session.updated_at
    row.tags = list(session.tags)
    row.tags_text = _tags_text(session.tags)
    row.metadata_json = dict(session.metadata)
    return row


def row_to_session(row: SessionRow) -> Session:
    return Session(
        id=row.id,
        title=row.title,
        status=SessionStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        interface=row.interface,
        user_request=row.user_request,
        task_type=row.task_type,
        model_alias=row.model_alias,
        tags=list(row.tags or []),
        metadata=dict(row.metadata_json or {}),
    )


def message_to_row(
    session_id: str, message: ChatMessage, seq: int = 0, *, redact_enabled: bool = True
) -> MessageRow:
    return MessageRow(
        id=message.id,
        session_id=session_id,
        seq=seq,
        role=str(message.role),
        name=message.name,
        content=redact(message.content, enabled=redact_enabled),
        created_at=message.created_at,
        metadata_json=redact_mapping(dict(message.metadata), enabled=redact_enabled),
    )


def row_to_message(row: MessageRow) -> ChatMessage:
    return ChatMessage(
        id=row.id,
        role=row.role,
        content=row.content,
        name=row.name,
        created_at=row.created_at,
        metadata=dict(row.metadata_json or {}),
    )


def command_to_row(
    session_id: str, command: ProposedCommand, row: CommandRow | None = None
) -> CommandRow:
    row = row or CommandRow(id=command.id)
    row.session_id = session_id
    row.kind = str(command.kind)
    row.binary = command.binary
    row.display = command.display
    row.argv = list(command.argv)
    row.purpose = command.purpose
    row.expected_effect = command.expected_effect
    row.proposed_by = command.proposed_by
    row.tool_name = command.tool_name
    row.created_at = command.created_at
    row.context_json = command.context.model_dump(mode="json")
    assessment = command.assessment
    row.risk = str(assessment.risk) if assessment else None
    row.requires_approval = bool(assessment.requires_approval) if assessment else True
    row.forbidden = bool(assessment.forbidden) if assessment else False
    row.assessment_json = assessment.model_dump(mode="json") if assessment else None
    # stdin and env can carry credentials, so they go through the redactor even
    # though they were authored by MIMIR rather than read from a system.
    row.payload_json = {
        "stdin": redact(command.stdin) if command.stdin else None,
        "cwd": command.cwd,
        "env": redact_mapping(dict(command.env)),
        "timeout_s": command.timeout_s,
        "metadata": redact_mapping(dict(command.metadata)),
    }
    return row


def row_to_command(row: CommandRow) -> ProposedCommand:
    payload = dict(row.payload_json or {})
    return ProposedCommand(
        id=row.id,
        kind=row.kind,
        argv=list(row.argv or []),
        stdin=payload.get("stdin"),
        cwd=payload.get("cwd"),
        env=dict(payload.get("env") or {}),
        timeout_s=payload.get("timeout_s"),
        purpose=row.purpose,
        expected_effect=row.expected_effect,
        context=row.context_json or {},
        proposed_by=row.proposed_by,
        created_at=row.created_at,
        tool_name=row.tool_name,
        metadata=dict(payload.get("metadata") or {}),
        assessment=RiskAssessment(**row.assessment_json) if row.assessment_json else None,
    )


def execution_to_row(record: ExecutionRecord, *, redact_enabled: bool = True) -> ExecutionRow:
    return ExecutionRow(
        id=record.id,
        session_id=record.session_id,
        command_id=record.command_id,
        display=record.display,
        argv=list(record.argv),
        outcome=str(record.outcome),
        exit_code=record.exit_code,
        risk=str(record.risk),
        started_at=record.started_at,
        duration_s=record.duration_s,
        stdout=redact(record.stdout, enabled=redact_enabled),
        stderr=redact(record.stderr, enabled=redact_enabled),
        truncated=record.truncated,
        artifact_ref=record.artifact_ref,
        approval_id=record.approval_id,
        approved_by=record.approved_by,
        error=record.error,
        context_json=record.context.model_dump(mode="json"),
        environment_json=redact_mapping(dict(record.environment), enabled=redact_enabled),
    )


def row_to_execution(row: ExecutionRow) -> ExecutionRecord:
    return ExecutionRecord(
        id=row.id,
        command_id=row.command_id,
        session_id=row.session_id,
        argv=list(row.argv or []),
        outcome=CommandOutcome(row.outcome),
        exit_code=row.exit_code,
        stdout=row.stdout,
        stderr=row.stderr,
        truncated=row.truncated,
        artifact_ref=row.artifact_ref,
        started_at=row.started_at,
        duration_s=row.duration_s,
        risk=RiskClass(row.risk),
        approval_id=row.approval_id,
        approved_by=row.approved_by,
        context=row.context_json or {},
        environment=dict(row.environment_json or {}),
        error=row.error,
    )


def approval_to_row(
    request: ApprovalRequest,
    decision: ApprovalDecision | None = None,
    row: ApprovalRow | None = None,
) -> ApprovalRow:
    row = row or ApprovalRow(id=request.id)
    row.session_id = request.session_id
    row.command_id = request.command.id
    row.risk = str(request.assessment.risk)
    row.status = str(request.status)
    row.prompt = redact(request.prompt)
    row.requested_by = request.requested_by
    row.created_at = request.created_at
    row.expires_at = request.expires_at
    row.command_json = request.command.model_dump(mode="json")
    row.assessment_json = request.assessment.model_dump(mode="json")
    if decision is not None:
        row.decision_status = str(decision.status)
        row.decided_at = decision.decided_at
        row.decided_by = decision.decided_by
        row.decision_reason = decision.reason
        row.edited_argv = list(decision.edited_argv) if decision.edited_argv else None
        # The decision is the authoritative end state of the request.
        row.status = str(decision.status)
    return row


def row_to_approval(row: ApprovalRow) -> ApprovalRequest:
    return ApprovalRequest(
        id=row.id,
        session_id=row.session_id,
        command=ProposedCommand(**row.command_json),
        assessment=RiskAssessment(**row.assessment_json),
        prompt=row.prompt,
        created_at=row.created_at,
        expires_at=row.expires_at,
        requested_by=row.requested_by,
        status=ApprovalStatus(row.status),
    )


def row_to_decision(row: ApprovalRow) -> ApprovalDecision | None:
    if not row.decision_status:
        return None
    return ApprovalDecision(
        approval_id=row.id,
        status=ApprovalStatus(row.decision_status),
        decided_at=row.decided_at or row.created_at,
        decided_by=row.decided_by or "user",
        reason=row.decision_reason,
        edited_argv=list(row.edited_argv) if row.edited_argv else None,
    )


def evidence_to_row(
    session_id: str,
    evidence: Evidence,
    row: EvidenceRow | None = None,
    *,
    redact_enabled: bool = True,
) -> EvidenceRow:
    row = row or EvidenceRow(session_id=session_id, evidence_id=evidence.id)
    row.claim = evidence.claim
    row.kind = str(evidence.kind)
    row.source_type = str(evidence.source_type)
    row.source_id = evidence.source_id
    # Excerpts are slices of command output and fetched pages, which is exactly
    # where a credential leaks in. The id was already computed upstream from the
    # unredacted text, so redacting here does not change identity.
    row.excerpt = redact(evidence.excerpt, enabled=redact_enabled)
    row.confidence = evidence.confidence
    row.supports = evidence.supports
    row.freshness = str(evidence.freshness)
    row.collected_at = evidence.collected_at
    row.collected_by = evidence.collected_by
    row.artifact_ref = evidence.artifact_ref
    row.tags = list(evidence.tags)
    row.structured = (
        redact_mapping(evidence.structured, enabled=redact_enabled)
        if evidence.structured
        else None
    )
    return row


def row_to_evidence(row: EvidenceRow, citations: Sequence[CitationRow] = ()) -> Evidence:
    return Evidence(
        id=row.evidence_id,
        claim=row.claim,
        kind=row.kind,
        source_type=SourceType(row.source_type),
        source_id=row.source_id,
        excerpt=row.excerpt,
        structured=row.structured,
        citations=[row_to_citation(c) for c in citations],
        collected_at=row.collected_at,
        freshness=row.freshness,
        confidence=row.confidence,
        supports=row.supports,
        collected_by=row.collected_by,
        artifact_ref=row.artifact_ref,
        tags=list(row.tags or []),
    )


def citation_to_row(
    session_id: str, evidence_row_id: int, citation: Citation, position: int = 0
) -> CitationRow:
    return CitationRow(
        evidence_row_id=evidence_row_id,
        session_id=session_id,
        position=position,
        source_type=str(citation.source_type),
        locator=citation.locator,
        repo=citation.repo,
        path=citation.path,
        start_line=citation.start_line,
        end_line=citation.end_line,
        url=citation.url,
        title=citation.title,
        retrieved_at=citation.retrieved_at,
    )


def row_to_citation(row: CitationRow) -> Citation:
    return Citation(
        source_type=SourceType(row.source_type),
        locator=row.locator,
        repo=row.repo,
        path=row.path,
        start_line=row.start_line,
        end_line=row.end_line,
        url=row.url,
        retrieved_at=row.retrieved_at,
        title=row.title,
    )


# -- repositories ---------------------------------------------------------


class _Repository:
    """Shared plumbing: a database handle and the redaction switch."""

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_database(self.settings)

    @property
    def _redact(self) -> bool:
        return self.settings.safety.redact_secrets


class SessionRepository(_Repository):
    """Sessions: create, read, list, status changes, and search (ADR 19.3)."""

    def create(self, session: Session) -> Session:
        with self.db.session() as db:
            row = session_to_row(session)
            db.add(row)
        return session

    def upsert(self, session: Session) -> Session:
        with self.db.session() as db:
            row = db.get(SessionRow, session.id)
            if row is None:
                db.add(session_to_row(session))
            else:
                session_to_row(session, row)
        return session

    def get(self, session_id: str) -> Session | None:
        with self.db.session() as db:
            row = db.get(SessionRow, session_id)
            return row_to_session(row) if row else None

    def exists(self, session_id: str) -> bool:
        with self.db.session() as db:
            return db.get(SessionRow, session_id) is not None

    def list(
        self,
        *,
        status: SessionStatus | str | None = None,
        interface: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        stmt = select(SessionRow).order_by(SessionRow.updated_at.desc())
        if status is not None:
            stmt = stmt.where(SessionRow.status == str(status))
        if interface is not None:
            stmt = stmt.where(SessionRow.interface == interface)
        with self.db.session() as db:
            rows = db.execute(stmt.limit(limit).offset(offset)).scalars().all()
            return [row_to_session(r) for r in rows]

    def update_status(
        self, session_id: str, status: SessionStatus | str, *, error: str | None = None
    ) -> bool:
        with self.db.session() as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                return False
            row.status = str(status)
            row.updated_at = time.time()
            if error is not None:
                row.error = error
            return True

    def add_tags(self, session_id: str, tags: Sequence[str]) -> list[str]:
        with self.db.session() as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                return []
            merged = list(dict.fromkeys([*(row.tags or []), *tags]))
            row.tags = merged
            row.tags_text = _tags_text(merged)
            row.updated_at = time.time()
            return merged

    def search(
        self,
        *,
        tag: str | None = None,
        query: str | None = None,
        since: float | None = None,
        until: float | None = None,
        status: SessionStatus | str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """Tag, free-text, time-window, and status search in one call."""
        stmt = select(SessionRow).order_by(SessionRow.created_at.desc())
        if tag:
            stmt = stmt.where(SessionRow.tags_text.like(f"%|{tag.strip().lower()}|%"))
        if query:
            pattern = f"%{query}%"
            stmt = stmt.where(
                SessionRow.user_request.like(pattern) | SessionRow.title.like(pattern)
            )
        if since is not None:
            stmt = stmt.where(SessionRow.created_at >= since)
        if until is not None:
            stmt = stmt.where(SessionRow.created_at <= until)
        if status is not None:
            stmt = stmt.where(SessionRow.status == str(status))
        with self.db.session() as db:
            rows = db.execute(stmt.limit(limit)).scalars().all()
            return [row_to_session(r) for r in rows]

    def record_thread(
        self,
        session_id: str,
        thread_id: str,
        *,
        checkpoint_ns: str = "",
        backend: str = "sqlite",
        checkpoint_id: str | None = None,
    ) -> None:
        """Map a LangGraph thread to a session so resume can find it (ADR 14.1)."""
        now = time.time()
        with self.db.session() as db:
            row = db.execute(
                select(GraphCheckpointRow).where(
                    GraphCheckpointRow.thread_id == thread_id,
                    GraphCheckpointRow.checkpoint_ns == checkpoint_ns,
                )
            ).scalar_one_or_none()
            if row is None:
                db.add(
                    GraphCheckpointRow(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        session_id=session_id,
                        backend=backend,
                        checkpoint_id=checkpoint_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                row.session_id = session_id
                row.backend = backend
                row.checkpoint_id = checkpoint_id
                row.updated_at = now

    def thread_for(self, session_id: str) -> str | None:
        with self.db.session() as db:
            row = db.execute(
                select(GraphCheckpointRow)
                .where(GraphCheckpointRow.session_id == session_id)
                .order_by(GraphCheckpointRow.updated_at.desc())
            ).scalars().first()
            return row.thread_id if row else None


class MessageRepository(_Repository):
    """Chat transcript. Content is redacted on the way in (ADR 13.4)."""

    def append(self, session_id: str, message: ChatMessage) -> ChatMessage:
        with self.db.session() as db:
            seq = self._next_seq(db, session_id)
            db.add(message_to_row(session_id, message, seq, redact_enabled=self._redact))
        return message

    def append_many(self, session_id: str, messages: Sequence[ChatMessage]) -> int:
        if not messages:
            return 0
        with self.db.session() as db:
            seq = self._next_seq(db, session_id)
            for offset, message in enumerate(messages):
                db.add(
                    message_to_row(
                        session_id, message, seq + offset, redact_enabled=self._redact
                    )
                )
        return len(messages)

    def list(self, session_id: str, *, limit: int | None = None) -> list[ChatMessage]:
        stmt = (
            select(MessageRow)
            .where(MessageRow.session_id == session_id)
            .order_by(MessageRow.seq, MessageRow.created_at)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        with self.db.session() as db:
            return [row_to_message(r) for r in db.execute(stmt).scalars().all()]

    def count(self, session_id: str) -> int:
        with self.db.session() as db:
            return int(
                db.execute(
                    select(func.count())
                    .select_from(MessageRow)
                    .where(MessageRow.session_id == session_id)
                ).scalar_one()
            )

    @staticmethod
    def _next_seq(db: OrmSession, session_id: str) -> int:
        current = db.execute(
            select(func.max(MessageRow.seq)).where(MessageRow.session_id == session_id)
        ).scalar()
        return int(current) + 1 if current is not None else 0


class CommandRepository(_Repository):
    """Proposed commands and their policy assessments."""

    def upsert(self, session_id: str, command: ProposedCommand) -> ProposedCommand:
        with self.db.session() as db:
            row = db.get(CommandRow, command.id)
            if row is None:
                db.add(command_to_row(session_id, command))
            else:
                command_to_row(session_id, command, row)
        return command

    def get(self, command_id: str) -> ProposedCommand | None:
        with self.db.session() as db:
            row = db.get(CommandRow, command_id)
            return row_to_command(row) if row else None

    def list_for_session(self, session_id: str) -> list[ProposedCommand]:
        stmt = (
            select(CommandRow)
            .where(CommandRow.session_id == session_id)
            .order_by(CommandRow.created_at)
        )
        with self.db.session() as db:
            return [row_to_command(r) for r in db.execute(stmt).scalars().all()]

    def list_by_risk(
        self, risk: RiskClass | str, *, since: float | None = None, limit: int = 100
    ) -> list[ProposedCommand]:
        stmt = (
            select(CommandRow)
            .where(CommandRow.risk == str(risk))
            .order_by(CommandRow.created_at.desc())
        )
        if since is not None:
            stmt = stmt.where(CommandRow.created_at >= since)
        with self.db.session() as db:
            return [row_to_command(r) for r in db.execute(stmt.limit(limit)).scalars().all()]


class ExecutionRepository(_Repository):
    """The audit trail of everything MIMIR actually ran (ADR 19.3).

    APPEND-ONLY BY DESIGN. There is no ``update`` and no ``delete`` method, and
    that omission is the enforcement mechanism: an audit trail that any caller
    can rewrite proves nothing about what happened. If a record is wrong, append
    a corrected one and an audit event explaining it. The only path that ever
    removes an execution row is retention pruning
    (:meth:`PersistenceService.prune`), which deletes whole sessions, reports
    exactly what it removed, and is off by default.
    """

    def record(self, execution: ExecutionRecord) -> ExecutionRecord:
        """Insert one execution. Re-recording an existing id is a no-op."""
        with self.db.session() as db:
            if db.get(ExecutionRow, execution.id) is not None:
                log.warning("persistence.execution.duplicate_ignored", execution_id=execution.id)
                return execution
            db.add(execution_to_row(execution, redact_enabled=self._redact))
        return execution

    def record_many(self, executions: Sequence[ExecutionRecord]) -> int:
        written = 0
        with self.db.session() as db:
            for execution in executions:
                if db.get(ExecutionRow, execution.id) is not None:
                    continue
                db.add(execution_to_row(execution, redact_enabled=self._redact))
                written += 1
        return written

    def get(self, execution_id: str) -> ExecutionRecord | None:
        with self.db.session() as db:
            row = db.get(ExecutionRow, execution_id)
            return row_to_execution(row) if row else None

    def list_for_session(self, session_id: str) -> list[ExecutionRecord]:
        stmt = (
            select(ExecutionRow)
            .where(ExecutionRow.session_id == session_id)
            .order_by(ExecutionRow.started_at)
        )
        with self.db.session() as db:
            return [row_to_execution(r) for r in db.execute(stmt).scalars().all()]

    def list_by_outcome(
        self,
        outcome: CommandOutcome | str,
        *,
        since: float | None = None,
        limit: int = 100,
    ) -> list[ExecutionRecord]:
        stmt = (
            select(ExecutionRow)
            .where(ExecutionRow.outcome == str(outcome))
            .order_by(ExecutionRow.started_at.desc())
        )
        if since is not None:
            stmt = stmt.where(ExecutionRow.started_at >= since)
        with self.db.session() as db:
            return [row_to_execution(r) for r in db.execute(stmt.limit(limit)).scalars().all()]

    def list_failures(self, *, since: float | None = None, limit: int = 100) -> list[
        ExecutionRecord
    ]:
        stmt = (
            select(ExecutionRow)
            .where(ExecutionRow.exit_code.is_not(None), ExecutionRow.exit_code != 0)
            .order_by(ExecutionRow.started_at.desc())
        )
        if since is not None:
            stmt = stmt.where(ExecutionRow.started_at >= since)
        with self.db.session() as db:
            return [row_to_execution(r) for r in db.execute(stmt.limit(limit)).scalars().all()]

    def count(self, session_id: str | None = None) -> int:
        stmt = select(func.count()).select_from(ExecutionRow)
        if session_id is not None:
            stmt = stmt.where(ExecutionRow.session_id == session_id)
        with self.db.session() as db:
            return int(db.execute(stmt).scalar_one())


class ApprovalRepository(_Repository):
    """Approval requests and their decisions (ADR 13, 15)."""

    def create(self, request: ApprovalRequest) -> ApprovalRequest:
        with self.db.session() as db:
            row = db.get(ApprovalRow, request.id)
            if row is None:
                db.add(approval_to_row(request))
            else:
                approval_to_row(request, row=row)
        return request

    def record_decision(self, decision: ApprovalDecision) -> bool:
        with self.db.session() as db:
            row = db.get(ApprovalRow, decision.approval_id)
            if row is None:
                return False
            row.decision_status = str(decision.status)
            row.status = str(decision.status)
            row.decided_at = decision.decided_at
            row.decided_by = decision.decided_by
            row.decision_reason = decision.reason
            row.edited_argv = list(decision.edited_argv) if decision.edited_argv else None
            return True

    def get(self, approval_id: str) -> ApprovalRequest | None:
        with self.db.session() as db:
            row = db.get(ApprovalRow, approval_id)
            return row_to_approval(row) if row else None

    def decision_for(self, approval_id: str) -> ApprovalDecision | None:
        with self.db.session() as db:
            row = db.get(ApprovalRow, approval_id)
            return row_to_decision(row) if row else None

    def list_for_session(self, session_id: str) -> list[ApprovalRequest]:
        stmt = (
            select(ApprovalRow)
            .where(ApprovalRow.session_id == session_id)
            .order_by(ApprovalRow.created_at)
        )
        with self.db.session() as db:
            return [row_to_approval(r) for r in db.execute(stmt).scalars().all()]

    def pending(self, session_id: str | None = None, *, limit: int = 50) -> list[ApprovalRequest]:
        stmt = (
            select(ApprovalRow)
            .where(ApprovalRow.decision_status.is_(None))
            .order_by(ApprovalRow.created_at)
        )
        if session_id is not None:
            stmt = stmt.where(ApprovalRow.session_id == session_id)
        with self.db.session() as db:
            return [row_to_approval(r) for r in db.execute(stmt.limit(limit)).scalars().all()]

    def wait_times(self, *, since: float | None = None) -> list[float]:
        """Seconds between request and decision, for ADR 20 approval latency."""
        stmt = select(ApprovalRow.created_at, ApprovalRow.decided_at).where(
            ApprovalRow.decided_at.is_not(None)
        )
        if since is not None:
            stmt = stmt.where(ApprovalRow.created_at >= since)
        with self.db.session() as db:
            return [float(decided - created) for created, decided in db.execute(stmt).all()]


class EvidenceRepository(_Repository):
    """Evidence and its citations (ADR 11.4, 12)."""

    def add(self, session_id: str, evidence: Evidence) -> Evidence:
        with self.db.session() as db:
            self._upsert(db, session_id, evidence)
        return evidence

    def add_many(self, session_id: str, items: Sequence[Evidence]) -> int:
        with self.db.session() as db:
            for item in items:
                self._upsert(db, session_id, item)
        return len(items)

    def get(self, session_id: str, evidence_id: str) -> Evidence | None:
        with self.db.session() as db:
            row = self._find(db, session_id, evidence_id)
            if row is None:
                return None
            return row_to_evidence(row, self._citations(db, row.row_id))

    def list_for_session(
        self, session_id: str, *, source_type: SourceType | str | None = None
    ) -> list[Evidence]:
        stmt = (
            select(EvidenceRow)
            .where(EvidenceRow.session_id == session_id)
            .order_by(EvidenceRow.collected_at)
        )
        if source_type is not None:
            stmt = stmt.where(EvidenceRow.source_type == str(source_type))
        with self.db.session() as db:
            rows = db.execute(stmt).scalars().all()
            citations = self._citations_by_evidence(db, [r.row_id for r in rows])
            return [row_to_evidence(r, citations.get(r.row_id, [])) for r in rows]

    def count(self, session_id: str) -> int:
        with self.db.session() as db:
            return int(
                db.execute(
                    select(func.count())
                    .select_from(EvidenceRow)
                    .where(EvidenceRow.session_id == session_id)
                ).scalar_one()
            )

    # -- internals -------------------------------------------------------

    def _upsert(self, db: OrmSession, session_id: str, evidence: Evidence) -> EvidenceRow:
        row = self._find(db, session_id, evidence.id)
        if row is None:
            row = evidence_to_row(session_id, evidence, redact_enabled=self._redact)
            db.add(row)
            db.flush()
        else:
            evidence_to_row(session_id, evidence, row, redact_enabled=self._redact)
            db.execute(delete(CitationRow).where(CitationRow.evidence_row_id == row.row_id))
        for position, citation in enumerate(evidence.citations):
            db.add(citation_to_row(session_id, row.row_id, citation, position))
        return row

    @staticmethod
    def _find(db: OrmSession, session_id: str, evidence_id: str) -> EvidenceRow | None:
        return db.execute(
            select(EvidenceRow).where(
                EvidenceRow.session_id == session_id,
                EvidenceRow.evidence_id == evidence_id,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _citations(db: OrmSession, evidence_row_id: int) -> list[CitationRow]:
        return list(
            db.execute(
                select(CitationRow)
                .where(CitationRow.evidence_row_id == evidence_row_id)
                .order_by(CitationRow.position)
            )
            .scalars()
            .all()
        )

    @staticmethod
    def _citations_by_evidence(
        db: OrmSession, evidence_row_ids: Sequence[int]
    ) -> dict[int, list[CitationRow]]:
        if not evidence_row_ids:
            return {}
        rows = (
            db.execute(
                select(CitationRow)
                .where(CitationRow.evidence_row_id.in_(evidence_row_ids))
                .order_by(CitationRow.position)
            )
            .scalars()
            .all()
        )
        grouped: dict[int, list[CitationRow]] = {}
        for row in rows:
            grouped.setdefault(row.evidence_row_id, []).append(row)
        return grouped


class AuditRepository(_Repository):
    """Append-only "who did what" log. Same rule as executions: no update."""

    def record(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        actor: str = "mimir",
        subject: str | None = None,
        risk: RiskClass | str | None = None,
        outcome: str | None = None,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.db.session() as db:
            db.add(
                AuditEventRow(
                    session_id=session_id,
                    event_type=event_type,
                    actor=actor,
                    subject=subject,
                    risk=str(risk) if risk else None,
                    outcome=outcome,
                    correlation_id=correlation_id,
                    created_at=time.time(),
                    payload_json=redact_mapping(payload or {}, enabled=self._redact),
                )
            )

    def list(
        self,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        stmt = select(AuditEventRow).order_by(AuditEventRow.created_at.desc())
        if session_id is not None:
            stmt = stmt.where(AuditEventRow.session_id == session_id)
        if event_type is not None:
            stmt = stmt.where(AuditEventRow.event_type == event_type)
        if since is not None:
            stmt = stmt.where(AuditEventRow.created_at >= since)
        with self.db.session() as db:
            rows = db.execute(stmt.limit(limit)).scalars().all()
            return [
                {
                    "id": r.row_id,
                    "session_id": r.session_id,
                    "event_type": r.event_type,
                    "actor": r.actor,
                    "subject": r.subject,
                    "risk": r.risk,
                    "outcome": r.outcome,
                    "correlation_id": r.correlation_id,
                    "created_at": r.created_at,
                    "payload": dict(r.payload_json or {}),
                }
                for r in rows
            ]


class ModelCallRepository(_Repository):
    """Model latency, tokens, and alias telemetry (ADR 20)."""

    def record(
        self,
        *,
        alias: str,
        model: str,
        latency_ms: float,
        session_id: str | None = None,
        runtime: str = "",
        task_class: str | None = None,
        specialist: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        context_size: int | None = None,
        tool_calls: int = 0,
        retries: int = 0,
        ok: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.db.session() as db:
            db.add(
                ModelCallRow(
                    session_id=session_id,
                    alias=alias,
                    model=model,
                    runtime=runtime,
                    task_class=task_class,
                    specialist=specialist,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    context_size=context_size,
                    tool_calls=tool_calls,
                    retries=retries,
                    ok=ok,
                    error=error,
                    created_at=time.time(),
                    metadata_json=redact_mapping(metadata or {}, enabled=self._redact),
                )
            )

    def list_for_session(self, session_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(ModelCallRow)
            .where(ModelCallRow.session_id == session_id)
            .order_by(ModelCallRow.created_at)
        )
        with self.db.session() as db:
            return [self._as_dict(r) for r in db.execute(stmt).scalars().all()]

    def summary(self, *, since: float | None = None) -> list[dict[str, Any]]:
        """Per-alias call count, mean latency, and token totals."""
        stmt = select(
            ModelCallRow.alias,
            func.count().label("calls"),
            func.avg(ModelCallRow.latency_ms).label("avg_latency_ms"),
            func.max(ModelCallRow.latency_ms).label("max_latency_ms"),
            func.sum(ModelCallRow.total_tokens).label("total_tokens"),
            func.sum(ModelCallRow.retries).label("retries"),
        ).group_by(ModelCallRow.alias)
        if since is not None:
            stmt = stmt.where(ModelCallRow.created_at >= since)
        with self.db.session() as db:
            return [
                {
                    "alias": alias,
                    "calls": int(calls),
                    "avg_latency_ms": float(avg or 0.0),
                    "max_latency_ms": float(mx or 0.0),
                    "total_tokens": int(tokens or 0),
                    "retries": int(retries or 0),
                }
                for alias, calls, avg, mx, tokens, retries in db.execute(stmt).all()
            ]

    @staticmethod
    def _as_dict(row: ModelCallRow) -> dict[str, Any]:
        return {
            "id": row.row_id,
            "session_id": row.session_id,
            "alias": row.alias,
            "model": row.model,
            "runtime": row.runtime,
            "task_class": row.task_class,
            "specialist": row.specialist,
            "latency_ms": row.latency_ms,
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
            "context_size": row.context_size,
            "tool_calls": row.tool_calls,
            "retries": row.retries,
            "ok": row.ok,
            "error": row.error,
            "created_at": row.created_at,
            "metadata": dict(row.metadata_json or {}),
        }


class EvalRepository(_Repository):
    """Evaluation runs and per-case results (ADR 21)."""

    def create_run(
        self,
        run_id: str,
        *,
        name: str = "",
        suite: str = "",
        model_alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.db.session() as db:
            db.add(
                EvalRunRow(
                    id=run_id,
                    name=name,
                    suite=suite,
                    model_alias=model_alias,
                    created_at=time.time(),
                    metadata_json=metadata or {},
                )
            )
        return run_id

    def record_result(
        self,
        run_id: str,
        *,
        case_id: str,
        category: str = "",
        passed: bool = False,
        score: float = 0.0,
        duration_s: float = 0.0,
        session_id: str | None = None,
        expected: str = "",
        actual: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.db.session() as db:
            db.add(
                EvalResultRow(
                    run_id=run_id,
                    case_id=case_id,
                    category=category,
                    passed=passed,
                    score=score,
                    duration_s=duration_s,
                    session_id=session_id,
                    expected=expected,
                    actual=redact(actual, enabled=self._redact),
                    detail_json=redact_mapping(detail or {}, enabled=self._redact),
                )
            )

    def complete_run(self, run_id: str) -> dict[str, int]:
        """Roll per-case results up onto the run row."""
        with self.db.session() as db:
            row = db.get(EvalRunRow, run_id)
            if row is None:
                return {"total": 0, "passed": 0, "failed": 0}
            total, passed = db.execute(
                select(
                    func.count(),
                    func.sum(case((EvalResultRow.passed.is_(True), 1), else_=0)),
                ).where(EvalResultRow.run_id == run_id)
            ).one()
            row.total = int(total or 0)
            row.passed = int(passed or 0)
            row.failed = row.total - row.passed
            row.completed_at = time.time()
            return {"total": row.total, "passed": row.passed, "failed": row.failed}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.db.session() as db:
            row = db.get(EvalRunRow, run_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "name": row.name,
                "suite": row.suite,
                "model_alias": row.model_alias,
                "created_at": row.created_at,
                "completed_at": row.completed_at,
                "total": row.total,
                "passed": row.passed,
                "failed": row.failed,
                "metadata": dict(row.metadata_json or {}),
            }

    def list_runs(self, *, suite: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        stmt = select(EvalRunRow.id).order_by(EvalRunRow.created_at.desc())
        if suite is not None:
            stmt = stmt.where(EvalRunRow.suite == suite)
        with self.db.session() as db:
            ids = list(db.execute(stmt.limit(limit)).scalars().all())
        runs = [self.get_run(run_id) for run_id in ids]
        return [r for r in runs if r is not None]

    def results_for_run(self, run_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(EvalResultRow)
            .where(EvalResultRow.run_id == run_id)
            .order_by(EvalResultRow.row_id)
        )
        with self.db.session() as db:
            return [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "passed": r.passed,
                    "score": r.score,
                    "duration_s": r.duration_s,
                    "session_id": r.session_id,
                    "expected": r.expected,
                    "actual": r.actual,
                    "detail": dict(r.detail_json or {}),
                }
                for r in db.execute(stmt).scalars().all()
            ]


# -- state persistence + retention ---------------------------------------


@dataclass(slots=True)
class PruneReport:
    """What a retention pass removed. Returned so a caller can report it."""

    enabled: bool
    older_than_days: float | None
    cutoff: float | None
    session_ids: list[str] = field(default_factory=list)
    deleted_by_table: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def sessions_deleted(self) -> int:
        return len(self.session_ids)

    @property
    def rows_deleted(self) -> int:
        return sum(self.deleted_by_table.values())

    def render(self) -> str:
        if not self.enabled:
            return "retention disabled (retention_days is not set); nothing was deleted"
        if not self.session_ids:
            return f"no sessions older than {self.older_than_days} days; nothing was deleted"
        verb = "would delete" if self.dry_run else "deleted"
        lines = [
            f"{verb} {self.sessions_deleted} session(s) older than "
            f"{self.older_than_days} days ({self.rows_deleted} rows total)"
        ]
        lines.extend(
            f"  {table:<20} {count}"
            for table, count in sorted(self.deleted_by_table.items())
            if count
        )
        return "\n".join(lines)


class PersistenceService:
    """Whole-state save and load, plus retention (ADR 12, 14.1, 19.3).

    ``save_state`` and ``load_state`` are the pair the CLI and the API both use
    to resume a session. The split is deliberate: anything with a table of its
    own is written through the repository that owns it, and the leftover state
    fields ride in ``sessions.state_json``.
    """

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_database(self.settings)
        self.sessions = SessionRepository(self.db, self.settings)
        self.messages = MessageRepository(self.db, self.settings)
        self.commands = CommandRepository(self.db, self.settings)
        self.executions = ExecutionRepository(self.db, self.settings)
        self.approvals = ApprovalRepository(self.db, self.settings)
        self.evidence = EvidenceRepository(self.db, self.settings)
        self.audit = AuditRepository(self.db, self.settings)
        self.model_calls = ModelCallRepository(self.db, self.settings)
        self.evals = EvalRepository(self.db, self.settings)

    @property
    def _redact(self) -> bool:
        return self.settings.safety.redact_secrets

    # -- state -----------------------------------------------------------

    def save_state(
        self,
        state: InvestigationState,
        *,
        status: SessionStatus | str | None = None,
        title: str | None = None,
        tags: Sequence[str] | None = None,
        model_alias: str | None = None,
    ) -> None:
        """Persist an entire investigation, creating the session if needed."""
        session_id = state.session_id
        now = time.time()
        resolved_status = str(status) if status is not None else str(self._infer_status(state))

        with self.db.session() as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                row = SessionRow(id=session_id, created_at=state.started_at)
                db.add(row)
            row.title = title if title is not None else (row.title or self._title_for(state))
            row.status = resolved_status
            row.interface = state.interface
            row.user_request = state.user_request
            row.task_type = str(state.task_type) if state.task_type else None
            if model_alias is not None:
                row.model_alias = model_alias
            if tags is not None:
                merged = list(dict.fromkeys([*(row.tags or []), *tags]))
                row.tags = merged
                row.tags_text = _tags_text(merged)
            row.updated_at = now
            row.started_at = state.started_at
            row.completed_at = state.completed_at
            row.iteration = state.iteration
            row.final_confidence = state.final_confidence
            row.error = state.error
            row.metadata_json = redact_mapping(dict(state.metadata), enabled=self._redact)
            row.state_json = state.model_dump(mode="json", exclude=set(_STATE_OWNED_FIELDS))

            self._save_messages(db, session_id, state.messages)
            self._save_commands(db, session_id, state.commands_planned)
            self._save_executions(db, session_id, state.commands_executed)
            self._save_model_calls(db, session_id, state.model_calls)
            self._save_approvals(db, state)
            self._save_evidence(db, session_id, state.evidence)
            self._save_web_sources(db, session_id, state.web_sources)
            self._save_skills(db, session_id, state)
            self._save_memory_proposals(db, session_id, state.memory_proposals)

    def load_state(self, session_id: str) -> InvestigationState | None:
        """Rebuild an :class:`InvestigationState` from the store."""
        with self.db.session() as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                return None

            data: dict[str, Any] = dict(row.state_json or {})
            data.update(
                session_id=row.id,
                user_request=row.user_request,
                task_type=row.task_type,
                interface=row.interface,
                iteration=row.iteration,
                started_at=row.started_at,
                completed_at=row.completed_at,
                error=row.error,
                final_confidence=row.final_confidence,
                metadata=dict(row.metadata_json or {}),
            )

            message_rows = (
                db.execute(
                    select(MessageRow)
                    .where(MessageRow.session_id == session_id)
                    .order_by(MessageRow.seq, MessageRow.created_at)
                )
                .scalars()
                .all()
            )
            data["messages"] = [row_to_message(r).model_dump() for r in message_rows]

            command_rows = (
                db.execute(
                    select(CommandRow)
                    .where(CommandRow.session_id == session_id)
                    .order_by(CommandRow.created_at)
                )
                .scalars()
                .all()
            )
            data["commands_planned"] = [row_to_command(r).model_dump() for r in command_rows]

            execution_rows = (
                db.execute(
                    select(ExecutionRow)
                    .where(ExecutionRow.session_id == session_id)
                    .order_by(ExecutionRow.started_at)
                )
                .scalars()
                .all()
            )
            data["commands_executed"] = [row_to_execution(r).model_dump() for r in execution_rows]

            approval_rows = (
                db.execute(
                    select(ApprovalRow)
                    .where(ApprovalRow.session_id == session_id)
                    .order_by(ApprovalRow.created_at)
                )
                .scalars()
                .all()
            )
            data["approvals"] = [row_to_approval(r).model_dump() for r in approval_rows]
            data["approval_decisions"] = [
                decision.model_dump()
                for decision in (row_to_decision(r) for r in approval_rows)
                if decision is not None
            ]

            evidence_rows = (
                db.execute(
                    select(EvidenceRow)
                    .where(EvidenceRow.session_id == session_id)
                    .order_by(EvidenceRow.collected_at, EvidenceRow.row_id)
                )
                .scalars()
                .all()
            )
            citations = EvidenceRepository._citations_by_evidence(
                db, [r.row_id for r in evidence_rows]
            )
            data["evidence"] = [
                row_to_evidence(r, citations.get(r.row_id, [])).model_dump()
                for r in evidence_rows
            ]

            web_rows = (
                db.execute(
                    select(WebSourceRow)
                    .where(WebSourceRow.session_id == session_id)
                    .order_by(WebSourceRow.retrieved_at, WebSourceRow.row_id)
                )
                .scalars()
                .all()
            )
            data["web_sources"] = [
                {
                    "url": r.url,
                    "title": r.title,
                    "retrieved_at": r.retrieved_at,
                    "excerpt": r.excerpt,
                    "query": r.query,
                }
                for r in web_rows
            ]

            skill_rows = (
                db.execute(
                    select(SkillUsedRow)
                    .where(SkillUsedRow.session_id == session_id)
                    .order_by(SkillUsedRow.selected_at, SkillUsedRow.row_id)
                )
                .scalars()
                .all()
            )
            data["selected_skills"] = [r.skill for r in skill_rows]

            proposal_rows = (
                db.execute(
                    select(MemoryProposalRow)
                    .where(MemoryProposalRow.session_id == session_id)
                    .order_by(MemoryProposalRow.created_at)
                )
                .scalars()
                .all()
            )
            data["memory_proposals"] = [
                {
                    "id": r.id,
                    "title": r.title,
                    "category": r.category,
                    "body": r.body,
                    "tags": list(r.tags or []),
                    "sources": list(r.sources or []),
                    "verification_status": r.verification_status,
                    "confidence": r.confidence,
                    "supersedes": r.supersedes,
                    "created_at": r.created_at,
                    "approved": r.approved,
                }
                for r in proposal_rows
            ]

        return InvestigationState(**data)

    # -- retention -------------------------------------------------------

    def prune(
        self, older_than_days: float | None = None, *, dry_run: bool = False
    ) -> PruneReport:
        """Delete sessions older than the cutoff, and report exactly what went.

        ADR 19.3 leaves retention open, so the default is to keep everything:
        with ``persistence.retention_days`` unset and no explicit argument this
        is a no-op that says so. Nothing is ever deleted silently, and the
        returned :class:`PruneReport` is meant to be shown to the operator.
        """
        days = older_than_days if older_than_days is not None else (
            self.settings.persistence.retention_days
        )
        if days is None:
            return PruneReport(enabled=False, older_than_days=None, cutoff=None, dry_run=dry_run)

        cutoff = time.time() - float(days) * 86400.0
        report = PruneReport(
            enabled=True, older_than_days=float(days), cutoff=cutoff, dry_run=dry_run
        )

        with self.db.session() as db:
            report.session_ids = list(
                db.execute(
                    select(SessionRow.id).where(SessionRow.updated_at < cutoff)
                ).scalars().all()
            )
            if not report.session_ids:
                return report

            # Children are removed explicitly rather than left to the FK cascade
            # so every deletion is counted and reportable.
            for model in SESSION_CHILD_TABLES:
                count = int(
                    db.execute(
                        select(func.count())
                        .select_from(model)
                        .where(model.session_id.in_(report.session_ids))
                    ).scalar_one()
                )
                report.deleted_by_table[model.__tablename__] = count
                if count and not dry_run:
                    db.execute(delete(model).where(model.session_id.in_(report.session_ids)))
            report.deleted_by_table["sessions"] = len(report.session_ids)
            if not dry_run:
                db.execute(delete(SessionRow).where(SessionRow.id.in_(report.session_ids)))

        if not dry_run:
            log.warning(
                "persistence.prune",
                sessions=report.sessions_deleted,
                rows=report.rows_deleted,
                older_than_days=days,
            )
        return report

    # -- internals -------------------------------------------------------

    @staticmethod
    def _infer_status(state: InvestigationState) -> SessionStatus:
        if state.error:
            return SessionStatus.FAILED
        if state.pending_approval() is not None:
            return SessionStatus.WAITING_APPROVAL
        if state.completed_at is not None:
            return SessionStatus.COMPLETED
        return SessionStatus.ACTIVE

    @staticmethod
    def _title_for(state: InvestigationState) -> str:
        text = state.user_request.strip().splitlines()[0] if state.user_request.strip() else ""
        return text[:120]

    def _save_messages(
        self, db: OrmSession, session_id: str, messages: Sequence[ChatMessage]
    ) -> None:
        known = set(
            db.execute(
                select(MessageRow.id).where(MessageRow.session_id == session_id)
            ).scalars().all()
        )
        seq = MessageRepository._next_seq(db, session_id)
        for message in messages:
            if message.id in known:
                continue
            db.add(message_to_row(session_id, message, seq, redact_enabled=self._redact))
            seq += 1

    def _save_commands(
        self, db: OrmSession, session_id: str, commands: Sequence[ProposedCommand]
    ) -> None:
        for command in commands:
            row = db.get(CommandRow, command.id)
            if row is None:
                db.add(command_to_row(session_id, command))
            else:
                command_to_row(session_id, command, row)

    def _save_executions(
        self, db: OrmSession, session_id: str, executions: Sequence[ExecutionRecord]
    ) -> None:
        # Append-only: an execution already on disk is never rewritten, even if
        # the in-memory copy differs.
        for execution in executions:
            if db.get(ExecutionRow, execution.id) is not None:
                continue
            record = execution if execution.session_id else execution.model_copy(
                update={"session_id": session_id}
            )
            db.add(execution_to_row(record, redact_enabled=self._redact))

    def _save_model_calls(
        self, db: OrmSession, session_id: str, calls: Sequence[dict[str, Any]]
    ) -> None:
        """Write model telemetry in the same transaction as the session row.

        The table, the repository method and the router's call log all existed
        independently for a long time and were never joined, so model_calls held
        zero rows across a hundred sessions. Writing it here, beside executions,
        keeps telemetry on the same footing as the audit trail.
        """
        existing = {
            row[0]
            for row in db.execute(
                select(ModelCallRow.invocation_id).where(
                    ModelCallRow.session_id == session_id
                )
            )
        }
        for call in calls:
            payload = dict(call)
            invocation_id = payload.get("invocation_id")
            if invocation_id and invocation_id in existing:
                # save_state can run more than once for a session. Keying on the
                # router-minted id makes re-persisting a no-op rather than a
                # doubling of every latency and token figure.
                continue
            if invocation_id:
                existing.add(invocation_id)
            db.add(
                ModelCallRow(
                    session_id=session_id,
                    invocation_id=invocation_id,
                    alias=payload.get("alias", ""),
                    model=payload.get("model", ""),
                    runtime=payload.get("runtime", ""),
                    task_class=payload.get("task_class") or None,
                    specialist=payload.get("specialist") or None,
                    latency_ms=float(payload.get("latency_ms") or 0.0),
                    prompt_tokens=payload.get("prompt_tokens"),
                    completion_tokens=payload.get("completion_tokens"),
                    total_tokens=payload.get("total_tokens"),
                    context_size=payload.get("context_size"),
                    tool_calls=int(payload.get("tool_calls") or 0),
                    retries=int(payload.get("attempt") or 0),
                    ok=bool(payload.get("ok", True)),
                    error=payload.get("error"),
                    created_at=float(payload.get("started_at") or time.time()),
                    metadata_json=payload.get("metadata") or {},
                )
            )

    def _save_approvals(self, db: OrmSession, state: InvestigationState) -> None:
        decisions = {d.approval_id: d for d in state.approval_decisions}
        for request in state.approvals:
            scoped = request if request.session_id else request.model_copy(
                update={"session_id": state.session_id}
            )
            row = db.get(ApprovalRow, scoped.id)
            if row is None:
                db.add(approval_to_row(scoped, decisions.get(scoped.id)))
            else:
                approval_to_row(scoped, decisions.get(scoped.id), row)

    def _save_evidence(
        self, db: OrmSession, session_id: str, items: Sequence[Evidence]
    ) -> None:
        repo = self.evidence
        for item in items:
            repo._upsert(db, session_id, item)

    def _save_web_sources(
        self, db: OrmSession, session_id: str, sources: Sequence[WebSource]
    ) -> None:
        # Web sources have no stable id, so the session's set is replaced whole.
        db.execute(delete(WebSourceRow).where(WebSourceRow.session_id == session_id))
        for source in sources:
            db.add(
                WebSourceRow(
                    session_id=session_id,
                    url=source.url,
                    title=source.title,
                    query=source.query,
                    excerpt=redact(source.excerpt, enabled=self._redact),
                    retrieved_at=source.retrieved_at,
                )
            )

    def _save_skills(self, db: OrmSession, session_id: str, state: InvestigationState) -> None:
        known = {
            row.skill: row
            for row in db.execute(
                select(SkillUsedRow).where(SkillUsedRow.session_id == session_id)
            ).scalars().all()
        }
        now = time.time()
        for skill in state.selected_skills:
            body = state.loaded_skill_bodies.get(skill, "")
            row = known.get(skill)
            if row is None:
                db.add(
                    SkillUsedRow(
                        session_id=session_id,
                        skill=skill,
                        loaded=bool(body),
                        body_chars=len(body),
                        selected_at=now,
                    )
                )
            else:
                row.loaded = bool(body) or row.loaded
                row.body_chars = len(body) or row.body_chars

    def _save_memory_proposals(
        self, db: OrmSession, session_id: str, proposals: Sequence[MemoryProposal]
    ) -> None:
        for proposal in proposals:
            row = db.get(MemoryProposalRow, proposal.id)
            if row is None:
                row = MemoryProposalRow(id=proposal.id, session_id=session_id)
                db.add(row)
            row.title = proposal.title
            row.category = proposal.category
            row.body = redact(proposal.body, enabled=self._redact)
            row.tags = list(proposal.tags)
            row.sources = list(proposal.sources)
            row.verification_status = proposal.verification_status
            row.confidence = proposal.confidence
            row.supersedes = proposal.supersedes
            row.approved = proposal.approved
            row.created_at = proposal.created_at


_service: PersistenceService | None = None


def get_persistence(settings: Settings | None = None) -> PersistenceService:
    """Process-wide service handle over the shared engine."""
    global _service
    if _service is None:
        _service = PersistenceService(settings=settings)
    return _service


def reset_persistence() -> None:
    global _service
    _service = None


def save_state(state: InvestigationState, **kwargs: Any) -> None:
    get_persistence().save_state(state, **kwargs)


def load_state(session_id: str) -> InvestigationState | None:
    return get_persistence().load_state(session_id)


__all__ = [
    "ApprovalRepository",
    "AuditRepository",
    "CommandRepository",
    "EvalRepository",
    "EvidenceRepository",
    "ExecutionRepository",
    "MessageRepository",
    "ModelCallRepository",
    "PersistenceService",
    "PruneReport",
    "SessionRepository",
    "command_to_row",
    "get_persistence",
    "load_state",
    "message_to_row",
    "reset_persistence",
    "row_to_command",
    "row_to_message",
    "row_to_session",
    "save_state",
    "session_to_row",
]
