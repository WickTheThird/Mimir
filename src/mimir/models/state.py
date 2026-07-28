"""Explicit investigation state (ADR 12).

The ADR is emphatic that a LangGraph investigation must use explicit state
rather than relying on chat history alone. :class:`InvestigationState` is that
state. It is a plain pydantic model so it can be serialised into checkpoints,
persisted, rendered in the web UI, and exported as an evidence package.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from mimir.models.approval import ApprovalDecision, ApprovalRequest
from mimir.models.command import ExecutionRecord, ProposedCommand
from mimir.models.evidence import Evidence, rank_evidence
from mimir.models.session import ChatMessage
from mimir.models.specialist import (
    CoordinatorPlan,
    FinalAnswer,
    Hypothesis,
    HypothesisStatus,
    SpecialistReport,
    TaskType,
)


class EnvironmentContext(BaseModel):
    """Resolved operating context. Populated from the shell, stored profile, or
    the user prompt (ADR 5.1 step 1)."""

    environment: str | None = None
    cluster_context: str | None = None
    namespace: str | None = None
    sdm_resource: str | None = None
    database: str | None = None
    repositories: list[str] = Field(default_factory=list)
    time_range: str | None = None
    cwd: str | None = None
    shell: str | None = None
    kubeconfig: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)

    def merge(self, other: EnvironmentContext) -> EnvironmentContext:
        """Non-empty fields on ``other`` win."""
        data = self.model_dump()
        for key, value in other.model_dump().items():
            if value in (None, [], {}, ""):
                continue
            if key == "repositories":
                data[key] = list(dict.fromkeys([*data.get(key, []), *value]))
            elif key == "extra":
                data[key] = {**data.get(key, {}), **value}
            else:
                data[key] = value
        return EnvironmentContext(**data)

    def render_pairs(self) -> list[tuple[str, str]]:
        pairs = [
            ("environment", self.environment),
            ("context", self.cluster_context),
            ("namespace", self.namespace),
            ("sdm resource", self.sdm_resource),
            ("database", self.database),
            ("repositories", ", ".join(self.repositories) if self.repositories else None),
            ("time range", self.time_range),
        ]
        return [(k, str(v)) for k, v in pairs if v]

    def render_lines(self) -> list[str]:
        return [f"{k:<14} {v}" for k, v in self.render_pairs()]


class WebSource(BaseModel):
    url: str
    title: str = ""
    retrieved_at: float = Field(default_factory=time.time)
    excerpt: str = ""
    query: str = ""


class MemoryProposal(BaseModel):
    """Candidate note produced by the Memory Curator (ADR 11.6)."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}")
    title: str
    category: str = "history/investigations"
    body: str = ""
    tags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    verification_status: str = "unverified"
    confidence: float = 0.5
    supersedes: str | None = None
    created_at: float = Field(default_factory=time.time)
    approved: bool = False


class InvestigationState(BaseModel):
    """The full state carried through the LangGraph run.

    Field names follow the conceptual state listed in ADR 12.
    """

    session_id: str = Field(default_factory=lambda: f"ses_{uuid.uuid4().hex[:12]}")
    user_request: str = ""
    task_type: TaskType | None = None
    interface: str = "cli"

    environment: EnvironmentContext = Field(default_factory=EnvironmentContext)

    messages: list[ChatMessage] = Field(default_factory=list)
    plan: CoordinatorPlan | None = None

    evidence: list[Evidence] = Field(default_factory=list)
    commands_planned: list[ProposedCommand] = Field(default_factory=list)
    commands_executed: list[ExecutionRecord] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
    """artifact_ref -> short description, full bodies live in the artifact store."""

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    rejected_hypotheses: list[Hypothesis] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    approvals: list[ApprovalRequest] = Field(default_factory=list)
    approval_decisions: list[ApprovalDecision] = Field(default_factory=list)

    selected_skills: list[str] = Field(default_factory=list)
    loaded_skill_bodies: dict[str, str] = Field(default_factory=dict)
    web_sources: list[WebSource] = Field(default_factory=list)
    memory_hits: list[str] = Field(default_factory=list)
    memory_proposals: list[MemoryProposal] = Field(default_factory=list)

    reports: list[SpecialistReport] = Field(default_factory=list)
    final_answer: FinalAnswer | None = None
    final_confidence: float = 0.0

    iteration: int = 0
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- mutation helpers -------------------------------------------------

    def add_evidence(self, items: Evidence | list[Evidence]) -> None:
        batch = [items] if isinstance(items, Evidence) else items
        known = {e.id for e in self.evidence}
        for item in batch:
            if item.id not in known:
                self.evidence.append(item)
                known.add(item.id)

    def add_report(self, report: SpecialistReport) -> None:
        self.reports.append(report)
        self.add_evidence(report.evidence)
        for hyp in report.hypotheses:
            self.upsert_hypothesis(hyp)
        for question in report.open_questions:
            if question not in self.pending_questions:
                self.pending_questions.append(question)

    def upsert_hypothesis(self, hypothesis: Hypothesis) -> None:
        for index, existing in enumerate(self.hypotheses):
            if existing.statement.strip().lower() == hypothesis.statement.strip().lower():
                self.hypotheses[index] = hypothesis
                break
        else:
            self.hypotheses.append(hypothesis)
        if hypothesis.status == HypothesisStatus.REJECTED:
            self.reject_hypothesis(hypothesis.id)

    def reject_hypothesis(self, hypothesis_id: str, reason: str | None = None) -> None:
        for index, hyp in enumerate(list(self.hypotheses)):
            if hyp.id == hypothesis_id:
                hyp.status = HypothesisStatus.REJECTED
                if reason:
                    hyp.rejected_reason = reason
                self.rejected_hypotheses.append(hyp)
                self.hypotheses.pop(index)
                return

    def record_execution(self, record: ExecutionRecord) -> None:
        self.commands_executed.append(record)
        if record.artifact_ref:
            self.outputs[record.artifact_ref] = record.display

    def evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidence if e.id == evidence_id), None)

    def ranked_evidence(self, limit: int | None = None) -> list[Evidence]:
        ranked = rank_evidence(self.evidence)
        return ranked[:limit] if limit else ranked

    def pending_approval(self) -> ApprovalRequest | None:
        decided = {d.approval_id for d in self.approval_decisions}
        return next((a for a in self.approvals if a.id not in decided), None)

    def decision_for(self, approval_id: str) -> ApprovalDecision | None:
        return next((d for d in self.approval_decisions if d.approval_id == approval_id), None)

    def command_by_id(self, command_id: str) -> ProposedCommand | None:
        return next((c for c in self.commands_planned if c.id == command_id), None)

    def ranked_hypotheses(self) -> list[Hypothesis]:
        return sorted(self.hypotheses, key=lambda h: -h.likelihood)

    @property
    def duration_s(self) -> float:
        return (self.completed_at or time.time()) - self.started_at
