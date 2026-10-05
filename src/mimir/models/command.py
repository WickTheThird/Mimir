"""Command proposal, risk classification, and execution records (ADR 13)."""

from __future__ import annotations

import shlex
import time
import uuid
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class RiskClass(StrEnum):
    """ADR 13.2."""

    R0 = "R0"  # pure analysis, no execution
    R1 = "R1"  # read-only local
    R2 = "R2"  # elevated inspection (exec, port-forward, db session)
    R3 = "R3"  # reversible mutation
    R4 = "R4"  # high-risk mutation

    @property
    def rank(self) -> int:
        return int(self.value[1])

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, RiskClass):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, RiskClass):
            return self.rank <= other.rank
        return NotImplemented


RISK_DESCRIPTIONS: dict[RiskClass, str] = {
    RiskClass.R0: "Pure analysis. Nothing is executed.",
    RiskClass.R1: "Read-only. Reads local files or fetches remote state without changing it.",
    RiskClass.R2: "Elevated inspection. Opens a session into a live system or reads sensitive "
    "operational output.",
    RiskClass.R3: "Reversible mutation. Changes live state in a way that can be undone.",
    RiskClass.R4: "High-risk mutation. Broad, destructive, or unclear blast radius.",
}


class CommandKind(StrEnum):
    SHELL = "shell"
    KUBECTL = "kubectl"
    SDM = "sdm"
    CONTAINER = "container"
    SQL = "sql"
    HTTP = "http"
    INTERNAL = "internal"


class TargetContext(BaseModel):
    """Everything that must be shown before a sensitive action (ADR 13.3)."""

    cluster_context: str | None = None
    namespace: str | None = None
    sdm_resource: str | None = None
    database: str | None = None
    container: str | None = None
    pod: str | None = None
    repo: str | None = None
    host: str | None = None
    targets: list[str] = Field(default_factory=list)

    def render_pairs(self) -> list[tuple[str, str]]:
        """Label/value pairs."""
        pairs = [
            ("cluster/context", self.cluster_context),
            ("namespace", self.namespace),
            ("sdm resource", self.sdm_resource),
            ("database", self.database),
            ("pod", self.pod),
            ("container", self.container),
            ("repo", self.repo),
            ("host", self.host),
        ]
        out = [(label, str(value)) for label, value in pairs if value]
        if self.targets:
            out.append(("target objects", ", ".join(self.targets)))
        return out

    def render_lines(self) -> list[str]:
        return [f"{label:<16} {value}" for label, value in self.render_pairs()]


class PolicyViolation(BaseModel):
    rule: str
    message: str
    fatal: bool = True


class RiskAssessment(BaseModel):
    """Produced by the deterministic policy engine, not by the model."""

    risk: RiskClass
    reasons: list[str] = Field(default_factory=list)
    requires_approval: bool = True
    forbidden: bool = False
    violations: list[PolicyViolation] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    production_target: bool = False
    reversible: bool = True
    rollback_hint: str | None = None
    classified_at: float = Field(default_factory=time.time)
    classifier: str = "policy"

    @property
    def summary(self) -> str:
        return RISK_DESCRIPTIONS.get(self.risk, "")


class ProposedCommand(BaseModel):
    """A command MIMIR wants to run, before any policy decision."""

    id: str = Field(default_factory=lambda: f"cmd_{uuid.uuid4().hex[:12]}")
    kind: CommandKind = CommandKind.SHELL
    argv: list[str] = Field(default_factory=list)
    """Argument vector. Preferred over a shell string; no shell is spawned."""

    stdin: str | None = None
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = None

    purpose: str = ""
    """Why this command is being run, in one sentence. Shown at approval time."""

    expected_effect: str = ""
    context: TargetContext = Field(default_factory=TargetContext)
    proposed_by: str = "mimir"
    created_at: float = Field(default_factory=time.time)
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Filled in by the policy engine.
    assessment: RiskAssessment | None = None

    @model_validator(mode="after")
    def _require_argv(self) -> Self:
        if not self.argv:
            raise ValueError("ProposedCommand.argv must not be empty")
        return self

    @property
    def display(self) -> str:
        return shlex.join(self.argv)

    @property
    def binary(self) -> str:
        return self.argv[0]

    def render_preview(self) -> str:
        """The pre-execution display required by ADR 13.3."""
        lines = [f"$ {self.display}"]
        ctx = self.context.render_lines()
        if ctx:
            lines.append("")
            lines.extend("  " + line for line in ctx)
        if self.purpose:
            lines.append(f"\n  reason           {self.purpose}")
        if self.expected_effect:
            lines.append(f"  expected effect  {self.expected_effect}")
        if self.assessment:
            lines.append(f"  risk             {self.assessment.risk.value} "
                         f"({self.assessment.summary})")
            for reason in self.assessment.reasons:
                lines.append(f"                   - {reason}")
            if self.assessment.rollback_hint:
                lines.append(f"  rollback         {self.assessment.rollback_hint}")
        return "\n".join(lines)


class CommandOutcome(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DENIED = "denied"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    ERROR = "error"


class ExecutionRecord(BaseModel):
    """What actually happened. Persisted for audit (ADR 19.3) and reuse as evidence."""

    id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    command_id: str
    session_id: str | None = None
    argv: list[str]
    outcome: CommandOutcome
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    artifact_ref: str | None = None
    started_at: float = Field(default_factory=time.time)
    duration_s: float = 0.0
    risk: RiskClass = RiskClass.R1
    approval_id: str | None = None
    approved_by: str | None = None
    context: TargetContext = Field(default_factory=TargetContext)
    environment: dict[str, str] = Field(default_factory=dict)
    """Non-secret environment metadata worth keeping, for example kube context."""

    error: str | None = None

    @property
    def display(self) -> str:
        return shlex.join(self.argv)

    @property
    def ok(self) -> bool:
        return self.outcome == CommandOutcome.SUCCESS

    def combined_output(self, limit: int | None = None) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.rstrip()}")
        text = "\n".join(parts)
        if limit is not None and len(text) > limit:
            return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"
        return text
