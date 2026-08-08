"""Council specialist contracts (ADR 7).

Each specialist has a narrow responsibility, restricted tools, a structured
output schema, and a confidence field. Disagreement between specialists must
stay visible, so reports carry both supporting and contradicting evidence.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field

from mimir.models.evidence import Evidence


class TaskType(StrEnum):
    """Classification produced by the Coordinator (ADR 7.1 S1)."""

    COMMAND_CONSTRUCTION = "command_construction"
    REPOSITORY_EXPLORATION = "repository_exploration"
    FEATURE_VERIFICATION = "feature_verification"
    KUBERNETES_INVESTIGATION = "kubernetes_investigation"
    SDM_CONTAINER_INVESTIGATION = "sdm_container_investigation"
    LOG_DIAGNOSIS = "log_diagnosis"
    WEB_RESEARCH = "web_research"
    MEMORY_LOOKUP = "memory_lookup"
    MUTATION_PLANNING = "mutation_planning"
    DATABASE_INVESTIGATION = "database_investigation"
    EVIDENCE_PACKAGE = "evidence_package"
    GENERAL_QUESTION = "general_question"


class SpecialistName(StrEnum):
    """ADR 7.1 S1 through S10."""

    COORDINATOR = "coordinator"
    REPOSITORY_EXPLORER = "repository_explorer"
    BEHAVIOUR_VERIFIER = "behaviour_verifier"
    KUBERNETES_INVESTIGATOR = "kubernetes_investigator"
    SDM_INVESTIGATOR = "sdm_investigator"
    LOG_ANALYST = "log_analyst"
    WEB_RESEARCHER = "web_researcher"
    MEMORY_CURATOR = "memory_curator"
    SAFETY_REVIEWER = "safety_reviewer"
    SYNTHESIS = "synthesis"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONFIRMED = "confirmed"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class Hypothesis(BaseModel):
    """A candidate explanation, ranked and kept even after rejection (ADR 5.6)."""

    id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:10]}")
    statement: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    next_check: str | None = None
    """The single cheapest command or lookup that would move this either way."""

    proposed_by: str = SpecialistName.LOG_ANALYST.value
    rejected_reason: str | None = None
    created_at: float = Field(default_factory=time.time)


class PlannedStep(BaseModel):
    """One unit of work the Coordinator wants a specialist to perform."""

    specialist: SpecialistName
    objective: str
    skill: str | None = None
    inputs: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""


class CoordinatorPlan(BaseModel):
    """Structured output of the Coordinator specialist (ADR 7.1 S1)."""

    task_type: TaskType = TaskType.GENERAL_QUESTION
    restated_question: str = ""
    steps: list[PlannedStep] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    needs_live_environment: bool = False
    needs_web: bool = False
    needs_repository: bool = False
    missing_context: list[str] = Field(default_factory=list)
    """Questions to ask the user before useful work is possible."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""


class SpecialistReport(BaseModel):
    """Uniform envelope every specialist returns."""

    id: str = Field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:10]}")
    specialist: SpecialistName
    objective: str = ""
    conclusion: str = ""
    detail: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    """Statements that conflict with other evidence. Never silently dropped."""

    proposed_command_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tool_calls: int = 0
    duration_s: float = 0.0
    error: str | None = None
    created_at: float = Field(default_factory=time.time)

    @property
    def failed(self) -> bool:
        return self.error is not None

    def render(self, max_evidence: int = 6) -> str:
        lines = [f"## {self.specialist.value} (confidence {self.confidence:.2f})"]
        if self.objective:
            lines.append(f"objective: {self.objective}")
        if self.conclusion:
            lines.append(f"conclusion: {self.conclusion}")
        if self.detail:
            lines.append(self.detail.strip())
        if self.evidence:
            lines.append("evidence:")
            lines.extend(f"  {e.render()}" for e in self.evidence[:max_evidence])
        if self.contradictions:
            lines.append("contradictions:")
            lines.extend(f"  ! {c}" for c in self.contradictions)
        if self.open_questions:
            lines.append("open questions:")
            lines.extend(f"  ? {q}" for q in self.open_questions)
        if self.error:
            lines.append(f"ERROR: {self.error}")
        return "\n".join(lines)


class FinalAnswer(BaseModel):
    """Structured output of the Synthesis specialist (ADR 7.1 S10)."""

    answer: str

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """An uncalibrated **score**, not a probability.

    Measured over 236 scored cases: AUC 0.558 against correctness, Brier 0.444,
    expected calibration error 0.472. A constant equal to the base rate scores
    better. The model says 0.13 and is right 66% of the time, so this number
    ranks weakly and its magnitude means nothing.

    It is kept because it is a mildly useful feature, and named ``confidence``
    for compatibility, but nothing may present it as a likelihood of being
    correct.
    """

    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    """Calibrated probability of correctness, or None.

    Stays None until a calibration model exists and has been validated on
    grouped cross-validation. ADR-003 invariant 7 forbids presenting model
    confidence as probability unless calibrated; making that a separate,
    nullable field enforces the rule structurally rather than by convention,
    because a field that does not exist cannot be misread.
    """
    observed_facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    proposed_commands: list[str] = Field(default_factory=list)
