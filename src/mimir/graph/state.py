"""LangGraph state channels (ADR 6.2 C2, 12).

:class:`~mimir.models.state.InvestigationState` is the durable record of an
investigation. This module wraps it in the channel shape LangGraph needs so that
specialists can run in parallel without clobbering each other.

The split matters: ``session`` has a single writer at any point in the graph,
while ``reports``, ``evidence``, and ``proposed_commands`` are append-only
channels that several concurrent specialists write to. Merging happens in one
place, in :func:`merge_into_session`, rather than being smeared across nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

from mimir.logging import get_logger
from mimir.models.command import ProposedCommand
from mimir.models.evidence import Evidence
from mimir.models.specialist import PlannedStep, SpecialistReport
from mimir.models.state import InvestigationState, MemoryProposal

log = get_logger(__name__)


def _merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """Append while de-duplicating on the content-derived evidence id."""
    seen = {item.id for item in left}
    out = list(left)
    for item in right:
        if item.id not in seen:
            out.append(item)
            seen.add(item.id)
    return out


def _merge_unique_str(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys([*left, *right]))


class GraphState(TypedDict, total=False):
    """Channels the graph reads and writes."""

    session: InvestigationState
    pending_steps: list[PlannedStep]
    reports: Annotated[list[SpecialistReport], operator.add]
    evidence: Annotated[list[Evidence], _merge_evidence]
    proposed_commands: Annotated[list[ProposedCommand], operator.add]
    memory_proposals: Annotated[list[MemoryProposal], operator.add]
    notes: Annotated[list[str], _merge_unique_str]
    route: str
    round: int
    halt_reason: str


class StepPayload(TypedDict):
    """What a fan-out branch receives through ``Send``."""

    session: InvestigationState
    step: PlannedStep
    round: int


def initial_state(
    session: InvestigationState, *, route: str = "coordinator"
) -> GraphState:
    return GraphState(
        session=session,
        pending_steps=[],
        reports=[],
        evidence=[],
        proposed_commands=[],
        memory_proposals=[],
        notes=[],
        route=route,
        round=0,
        halt_reason="",
    )


ModelT = TypeVar("ModelT", bound=BaseModel)


def _coerce(items: Any, model: type[ModelT]) -> list[ModelT]:
    """Accept either model instances or the dicts a checkpoint may return.

    Resuming a run deserialises through LangGraph's serialiser, which can hand
    back plain dicts for a type it was not told about, or for a checkpoint
    written by an older build. Crashing on that would make a resumable session
    unresumable, so entries are coerced and unparseable ones are dropped with a
    warning rather than taking the whole resume down.
    """
    out: list[ModelT] = []
    for item in items or []:
        if isinstance(item, model):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(model.model_validate(item))
            except ValidationError as exc:
                log.warning(
                    "checkpoint_entry_dropped", model=model.__name__, error=str(exc)[:200]
                )
    return out


def merge_into_session(state: GraphState) -> InvestigationState:
    """Fold the append-only channels back into the durable record.

    Called once at the end of a round so the InvestigationState stays the single
    thing worth persisting, exporting, and showing in the UI.
    """
    session = state["session"]
    if isinstance(session, dict):
        session = InvestigationState.model_validate(session)
        state["session"] = session

    for report in _coerce(state.get("reports"), SpecialistReport):
        if report.id not in {r.id for r in session.reports}:
            session.add_report(report)
    session.add_evidence(_coerce(state.get("evidence"), Evidence))
    known = {c.id for c in session.commands_planned}
    for command in _coerce(state.get("proposed_commands"), ProposedCommand):
        if command.id not in known:
            session.commands_planned.append(command)
            known.add(command.id)
    known_proposals = {p.id for p in session.memory_proposals}
    for proposal in _coerce(state.get("memory_proposals"), MemoryProposal):
        if proposal.id not in known_proposals:
            session.memory_proposals.append(proposal)
    return session


def state_summary(state: GraphState) -> dict[str, Any]:
    """Compact view used for streaming progress to the CLI and web UI."""
    session = state["session"]
    return {
        "session_id": session.session_id,
        "task_type": session.task_type.value if session.task_type else None,
        "round": state.get("round", 0),
        "route": state.get("route", ""),
        "reports": [r.specialist.value for r in state.get("reports", [])],
        "evidence_count": len(state.get("evidence", [])),
        "commands_planned": len(state.get("proposed_commands", [])),
        "hypotheses": len(session.hypotheses),
        "pending_questions": session.pending_questions,
    }
