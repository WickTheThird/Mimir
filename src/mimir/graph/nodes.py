"""Graph nodes (ADR 6.2 C2, 7.2).

The council decision pattern from ADR 7.2 maps onto these nodes:

    resolve_context -> recall_memory -> coordinate -> select_skills
        -> [fan out: specialists in parallel] -> gather
        -> verify -> safety_review -> synthesise -> curate_memory

Every node takes and returns a :class:`~mimir.graph.state.GraphState` slice.
Nodes never execute commands directly; specialists do that through typed tools,
which route through the policy engine.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from mimir.council.specialists import Specialist, build_council
from mimir.graph.state import GraphState, StepPayload, merge_into_session
from mimir.graph.triage import triage
from mimir.llm.base import ModelError
from mimir.llm.router import ModelRouter, StructuredOutputError, get_router
from mimir.logging import get_logger
from mimir.models.evidence import Evidence, EvidenceKind, SourceType
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
from mimir.models.state import EnvironmentContext, InvestigationState, MemoryProposal
from mimir.tools.base import REGISTRY, ToolContext, ToolRegistry
from mimir.verify.claims import (
    attach_resolved_citations,
    check_answer,
    demote_unsupported,
    unsupported_brief,
)
from mimir.verify.grounding import check as grounding_check
from mimir.verify.patterns import claims_retries, demote_unsupported_retry
from mimir.verify.patterns import from_text as retry_from_text
from mimir.verify.grounding import demote_ungrounded
from mimir.verify.sufficiency import check as sufficiency_check
from mimir.verify.sufficiency import (
    classify_currency,
    demote_overreach,
    demote_stale,
)

log = get_logger(__name__)


#: Task types that can be answered without a full council pass. Routing them
#: straight to one specialist is the ADR 7.2 note that two well-scoped steps beat
#: six vague ones, and it keeps command completion fast (ADR R6).
DIRECT_ROUTES: dict[TaskType, SpecialistName] = {
    TaskType.COMMAND_CONSTRUCTION: SpecialistName.KUBERNETES_INVESTIGATOR,
    TaskType.MEMORY_LOOKUP: SpecialistName.MEMORY_CURATOR,
    TaskType.WEB_RESEARCH: SpecialistName.WEB_RESEARCHER,
}


class SkillAccess:
    """Thin, failure-tolerant adapter over the skills subsystem.

    Nodes should not be littered with try/except around skill lookups, and a
    malformed skill on disk must never take an investigation down. Every method
    here degrades to an empty result and logs instead of raising.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self.runner: Any = None
        try:
            from mimir.skills.runner import SkillRunner

            self.runner = SkillRunner(registry)
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_runner_unavailable", error=str(exc))

    def catalogue(self, specialist: SpecialistName | None = None) -> str:
        try:
            return self.registry.catalogue(specialist)
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_catalogue_failed", error=str(exc))
            return ""

    def select(self, request: str, specialist: SpecialistName | None = None) -> list[str]:
        """Skill names for a request.

        registry.select returns SkillSelection wrappers (skill, score, matched,
        explicit), not Skill objects. Unwrapping here keeps the wrapper shape
        out of the graph: reaching for .name on a selection is what silently
        disabled automatic skill selection.
        """
        try:
            selections = self.registry.select(request, specialist=specialist)
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_selection_failed", error=str(exc))
            return []
        names = []
        for selection in selections:
            skill = getattr(selection, "skill", selection)
            name = getattr(skill, "name", None)
            if name:
                names.append(name)
        return names

    def body(self, name: str) -> str:
        if self.runner is None:
            return ""
        try:
            return self.runner.load(name).body
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_load_failed", skill=name, error=str(exc))
            return ""

    def allowed_tools(
        self, name: str, specialist: SpecialistName | None = None
    ) -> list[str] | None:
        if self.runner is None:
            return None
        try:
            skill = self.registry.get(name)
            if skill is None:
                return None
            # The names, not the specs. available_tools() narrows by name and
            # puts what it is given into a set, so handing it ToolSpec objects
            # raised "unhashable type: ToolSpec" on every specialist step that
            # a skill narrowed. ToolPermissions carries a names property for
            # exactly this and the call site reached past it.
            return self.runner.permitted_tools(skill, specialist).names
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_tool_filter_failed", skill=name, error=str(exc))
            return None


class NodeDeps:
    """Everything the nodes need, injected once so nodes stay testable."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        router: ModelRouter | None = None,
        tool_context: ToolContext | None = None,
        skill_registry: Any = None,
        knowledge: Any = None,
    ) -> None:
        self.registry = registry or REGISTRY
        self.router = router or get_router()
        self.tool_context = tool_context or ToolContext()
        self.skills = SkillAccess(skill_registry) if skill_registry is not None else None
        self.knowledge = knowledge
        self.council: dict[SpecialistName, Specialist] = build_council(
            registry=self.registry, router=self.router
        )

    def specialist(self, name: SpecialistName) -> Specialist:
        return self.council[name]


# ---------------------------------------------------------------------------
# 1. Context resolution
# ---------------------------------------------------------------------------


async def resolve_context(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Fill in the operating context from the shell and config (ADR 5.1 step 1).

    Anything the user stated explicitly wins. This only supplies defaults, and it
    never guesses a namespace: an unresolved namespace must surface as a question
    rather than as a silent default, because the wrong namespace is how the wrong
    cluster gets touched.
    """
    session = state["session"]
    settings = deps.tool_context.settings
    discovered = EnvironmentContext(
        cwd=os.getcwd(),
        shell=os.environ.get("SHELL"),
        kubeconfig=os.environ.get("KUBECONFIG") or (
            str(settings.kubernetes.kubeconfig) if settings.kubernetes.kubeconfig else None
        ),
        cluster_context=settings.kubernetes.default_context,
        namespace=settings.kubernetes.default_namespace,
        environment=settings.environment,
    )

    repos = list(session.environment.repositories)
    if not repos:
        for entry in settings.repos.entries:
            repos.append(entry.name)
        if not repos:
            cwd = Path.cwd()
            if (cwd / ".git").is_dir():
                repos.append(cwd.name)
    discovered.repositories = repos

    # The session's own values take precedence over anything discovered here.
    merged = discovered.merge(session.environment)
    session.environment = merged
    deps.tool_context.environment = merged

    return {
        "session": session,
        "notes": [f"context resolved: {', '.join(merged.render_lines()) or 'none'}"],
    }


# ---------------------------------------------------------------------------
# 2. Memory recall
# ---------------------------------------------------------------------------


async def recall_memory(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Retrieve curated knowledge before planning (ADR G5, 11.4).

    Memory informs the plan but never outranks live evidence. Retrieved notes
    enter as evidence with their real source type and freshness, so the trust
    ladder in ADR 11.4 does the ranking rather than recency of retrieval.
    """
    session = state["session"]
    spec = deps.registry.get("search_memory")
    if spec is None:
        return {"notes": ["memory subsystem unavailable; continuing without it"]}

    result = await spec.invoke(
        {"query": session.user_request, "limit": 6}, deps.tool_context
    )
    if not result.ok:
        return {"notes": [f"memory search failed: {result.error}"]}

    hits = result.data.get("results", []) if isinstance(result.data, dict) else []
    session.memory_hits = [str(h.get("document_id") or h.get("path", "")) for h in hits][:10]
    return {
        "session": session,
        "evidence": result.evidence,
        "notes": [f"memory recall: {len(hits)} candidate notes"],
    }


# ---------------------------------------------------------------------------
# 3. Coordination
# ---------------------------------------------------------------------------


async def coordinate(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Classify the request and produce a plan (ADR 7.1 S1)."""
    session = state["session"]
    catalogue = deps.skills.catalogue() if deps.skills is not None else ""

    memory_context = _render_memory_context(state.get("evidence", []))
    extra = []
    if catalogue:
        extra.append(f"Available skills (choose by name, do not invent):\n{catalogue}")
    extra.append(
        "Specialist names you may assign: "
        + ", ".join(s.value for s in SpecialistName if s != SpecialistName.COORDINATOR)
    )

    # Cheapest possible path first. A greeting needs no context resolution, no
    # classification, no specialists and no synthesis; running them cost over a
    # minute and routed "hello" to the Kubernetes investigator.
    verdict = triage(session.user_request)
    if verdict.cheap:
        session.task_type = TaskType.GENERAL_QUESTION
        session.final_answer = FinalAnswer(
            answer=verdict.reply,
            confidence=1.0,
            observed_facts=[],
            inferences=[],
        )
        session.final_confidence = 1.0
        log.info("triaged_without_investigation", kind=verdict.kind.value)
        return {
            "session": session,
            "pending_steps": [],
            "route": "done",
            "notes": [f"triage: {verdict.kind.value}, no investigation needed"],
        }

    coordinator = deps.specialist(SpecialistName.COORDINATOR)
    try:
        plan: CoordinatorPlan = await coordinator.structured_report(
            "Classify this request and plan the investigation.",
            session,
            CoordinatorPlan,
            extra_context="\n\n".join(extra),
            memory_context=memory_context,
        )
    except (ModelError, StructuredOutputError) as exc:
        log.warning("coordinator_failed", error=str(exc))
        plan = _fallback_plan(session)

    plan.steps = _sanitise_steps(plan.steps, session)
    session.plan = plan
    session.task_type = plan.task_type
    session.selected_skills = plan.selected_skills
    for question in plan.missing_context:
        if question not in session.pending_questions:
            session.pending_questions.append(question)

    route = "ask_user" if plan.missing_context and not plan.steps else "select_skills"
    log.info(
        "plan_ready",
        task_type=plan.task_type.value,
        steps=len(plan.steps),
        skills=plan.selected_skills,
        missing_context=len(plan.missing_context),
    )
    return {
        "session": session,
        "pending_steps": plan.steps,
        "route": route,
        "notes": [f"plan: {plan.task_type.value} with {len(plan.steps)} step(s)"],
    }


def _fallback_plan(session: InvestigationState) -> CoordinatorPlan:
    """Used when the model cannot produce a valid plan.

    Failing to plan must not fail the investigation. A single broad repository
    or log step still produces something the operator can use, and the failure
    is reported rather than hidden.
    """
    request = session.user_request.lower()
    if any(word in request for word in ("timeout", "timing out", "latency", "slow", "restart")):
        specialist = SpecialistName.LOG_ANALYST
        task_type = TaskType.LOG_DIAGNOSIS
    elif any(word in request for word in ("kubectl", "pod", "deployment", "namespace", "cluster")):
        specialist = SpecialistName.KUBERNETES_INVESTIGATOR
        task_type = TaskType.KUBERNETES_INVESTIGATION
    else:
        specialist = SpecialistName.REPOSITORY_EXPLORER
        task_type = TaskType.REPOSITORY_EXPLORATION
    return CoordinatorPlan(
        task_type=task_type,
        restated_question=session.user_request,
        steps=[
            PlannedStep(
                specialist=specialist,
                objective=session.user_request,
                rationale="coordinator planning failed; routed by keyword fallback",
            )
        ],
        confidence=0.25,
        notes="fallback plan: the coordinator model did not return a usable plan",
    )


def _sanitise_steps(steps: list[PlannedStep], session: InvestigationState) -> list[PlannedStep]:
    """Drop steps the plan cannot support, and cap the fan-out."""
    cleaned: list[PlannedStep] = []
    for step in steps:
        if step.specialist in (SpecialistName.COORDINATOR, SpecialistName.SYNTHESIS):
            continue
        if not step.objective.strip():
            step.objective = session.user_request
        cleaned.append(step)
    if not cleaned:
        cleaned = _fallback_plan(session).steps
    return cleaned[:5]


async def ask_user(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Terminal node when the request cannot proceed without more input."""
    session = state["session"]
    questions = session.pending_questions or ["What should I investigate?"]
    session.final_answer = FinalAnswer(
        answer=(
            "I need more context before this is safe or useful to investigate:\n"
            + "\n".join(f"  - {q}" for q in questions)
        ),
        confidence=0.0,
        unverified=questions,
        next_steps=["Re-run with the missing context supplied."],
    )
    session.completed_at = time.time()
    return {"session": session, "route": "done", "halt_reason": "missing_context"}


# ---------------------------------------------------------------------------
# 4. Skills
# ---------------------------------------------------------------------------


async def select_skills(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Load the bodies of the selected skills only (ADR 10.2 progressive disclosure).

    Level 1 (name plus description) was already in the coordinator's context as a
    catalogue. This node performs the level-2 load, and only for the skills the
    plan actually chose. References and scripts stay unloaded until a skill body
    asks for them by name.
    """
    session = state["session"]
    if deps.skills is None:
        return {"route": "dispatch"}

    names = list(session.selected_skills)
    if not names:
        names = deps.skills.select(session.user_request)

    bodies: dict[str, str] = {}
    for name in names:
        body = deps.skills.body(name)
        if body:
            bodies[name] = body

    session.selected_skills = list(bodies)
    session.loaded_skill_bodies = bodies
    return {
        "session": session,
        "route": "dispatch",
        "notes": [f"skills loaded: {', '.join(bodies) or 'none'}"],
    }


# ---------------------------------------------------------------------------
# 5. Specialist fan-out
# ---------------------------------------------------------------------------


async def run_specialist_step(payload: StepPayload, deps: NodeDeps) -> dict[str, Any]:
    """One fan-out branch. Runs a single specialist against a single objective."""
    session = payload["session"]
    step = payload["step"]
    specialist = deps.specialist(step.specialist)

    skill_bodies = {}
    if step.skill and step.skill in session.loaded_skill_bodies:
        skill_bodies = {step.skill: session.loaded_skill_bodies[step.skill]}
    elif session.loaded_skill_bodies:
        skill_bodies = session.loaded_skill_bodies

    allowed_tools = None
    if step.skill and deps.skills is not None:
        allowed_tools = deps.skills.allowed_tools(step.skill, step.specialist)

    extra = []
    if step.inputs:
        extra.append(
            "Inputs from the plan:\n"
            + "\n".join(f"  {k}: {v}" for k, v in step.inputs.items())
        )
    prior = _render_prior_findings(session)
    if prior:
        extra.append(prior)

    run = await specialist.run(
        step.objective,
        session,
        ctx=deps.tool_context,
        skill_bodies=skill_bodies,
        allowed_tools=allowed_tools,
        extra_context="\n\n".join(extra),
    )

    proposed = [
        command
        for result in run.tool_results
        for command in result.proposed_commands
    ]
    return {
        "reports": [run.report],
        "evidence": run.report.evidence,
        "proposed_commands": proposed,
    }


# ---------------------------------------------------------------------------
# 6. Gather, verify, review
# ---------------------------------------------------------------------------


async def gather(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Fold parallel results back into the session and decide what comes next."""
    session = merge_into_session(state)
    round_number = state.get("round", 0) + 1

    failures = [r for r in state.get("reports", []) if r.failed]
    if failures:
        for report in failures:
            note = f"{report.specialist.value} failed: {report.error}"
            if note not in session.risks:
                session.risks.append(note)

    needs_verification = session.task_type in (
        TaskType.FEATURE_VERIFICATION,
        TaskType.REPOSITORY_EXPLORATION,
        TaskType.LOG_DIAGNOSIS,
    ) and any(not r.failed for r in state.get("reports", []))

    route = "verify" if needs_verification else "safety_review"
    if round_number >= deps.tool_context.settings.graph.max_specialist_rounds:
        route = "safety_review"

    return {
        "session": session,
        "round": round_number,
        "route": route,
        "notes": [f"round {round_number}: {len(state.get('reports', []))} report(s)"],
    }


async def verify(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Behaviour Verifier pass (ADR 7.2 step 5).

    Its job is to attack the conclusion, not to agree with it. Disagreement is
    recorded, never smoothed away (ADR 7.2).
    """
    session = state["session"]
    reports = [r for r in state.get("reports", []) if not r.failed]
    if not reports:
        return {"route": "safety_review"}

    verifier = deps.specialist(SpecialistName.BEHAVIOUR_VERIFIER)
    context = "\n\n".join(r.render() for r in reports)
    run = await verifier.run(
        "Check whether the conclusions above are actually supported by the evidence. "
        "Look for the evidence that would make them false. Report every contradiction "
        "you find, and state which claims remain unverified.",
        session,
        ctx=deps.tool_context,
        extra_context=f"Specialist findings to check:\n\n{context}",
    )

    for report in reports:
        if report.confidence > 0.5 and run.report.contradictions:
            report.confidence = max(0.2, report.confidence - 0.2)

    return {
        "session": session,
        "reports": [run.report],
        "evidence": run.report.evidence,
        "route": "safety_review",
    }


async def safety_review(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Safety and Command Reviewer pass (ADR 7.1 S9, 7.2 step 6).

    Skipped entirely when nothing was proposed, because reviewing an empty list
    burns a model call for no benefit.
    """
    session = merge_into_session(state)
    commands = session.commands_planned
    if not commands:
        return {"session": session, "route": "synthesise"}

    rendered = "\n\n".join(c.render_preview() for c in commands[:8])
    reviewer = deps.specialist(SpecialistName.SAFETY_REVIEWER)
    run = await reviewer.run(
        "Review these proposed commands. For each one state whether the target is "
        "fully resolved, whether it matches its stated purpose, what its real blast "
        "radius is, and whether a safer read-only command would answer the same "
        "question.",
        session,
        ctx=deps.tool_context,
        extra_context=f"Proposed commands, already classified by policy code:\n\n{rendered}",
    )
    for concern in run.report.contradictions or []:
        if concern not in session.risks:
            session.risks.append(concern)

    return {
        "session": session,
        "reports": [run.report],
        "route": "synthesise",
    }


# ---------------------------------------------------------------------------
# 7. Synthesis and memory curation
# ---------------------------------------------------------------------------


async def synthesise(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Combine specialist output into the final evidence-backed answer (S10)."""
    session = merge_into_session(state)
    reports = state.get("reports", []) or session.reports

    evidence_block = "\n".join(e.render() for e in session.ranked_evidence(limit=30))
    reports_block = "\n\n".join(r.render() for r in reports)
    disagreements = _detect_disagreements(reports)

    extra = [
        f"Specialist reports:\n\n{reports_block}",
        f"Ranked evidence (most trusted first):\n{evidence_block or 'none collected'}",
    ]
    if disagreements:
        extra.append(
            "Detected disagreements between specialists. Report these openly:\n"
            + "\n".join(f"  - {d}" for d in disagreements)
        )
    if session.commands_planned:
        extra.append(
            "Commands proposed but not executed:\n"
            + "\n".join(f"  $ {c.display}" for c in session.commands_planned[:8])
        )
    if session.risks:
        extra.append("Risks and failures recorded this run:\n" + "\n".join(
            f"  - {r}" for r in session.risks
        ))

    synth = deps.specialist(SpecialistName.SYNTHESIS)
    try:
        answer: FinalAnswer = await synth.structured_report(
            "Produce the final answer for the operator.",
            session,
            FinalAnswer,
            extra_context="\n\n".join(extra),
        )
    except (ModelError, StructuredOutputError) as exc:
        log.warning("synthesis_failed", error=str(exc))
        answer = _fallback_answer(session, reports, disagreements, str(exc))

    answer.disagreements = list(dict.fromkeys([*answer.disagreements, *disagreements]))

    # The blanket citation attach that used to live here bolted the top ten
    # evidence citations onto any answer that returned none, regardless of what
    # it claimed. That is citation as ornament: it satisfied a presence check
    # while carrying no relationship to the text. Support is now resolved
    # per claim instead.
    answer = await _enforce_claim_support(answer, session, synth, extra)
    answer = _enforce_sufficiency(answer, session)
    answer = _enforce_grounding(answer, session)
    answer = _enforce_retry_signature(answer, session)

    session.final_answer = answer
    session.final_confidence = _final_confidence(answer, reports, session)
    session.completed_at = time.time()

    return {"session": session, "route": "curate_memory"}


async def _enforce_claim_support(
    answer: FinalAnswer,
    session: InvestigationState,
    synth: Any,
    extra: list[str],
) -> FinalAnswer:
    """Resolve every stated fact against evidence, deterministically.

    One repair attempt, then mechanical demotion. Repairing more than once
    invites the model to keep rewording until something passes, which optimises
    the checker rather than the answer, and costs a model call per attempt.

    The check itself never consults a model. That is the point of the exercise:
    the components of MIMIR that are already deterministic are the only ones
    that do not change their mind between identical runs.
    """
    support = check_answer(answer, session.evidence)
    session.metadata["claim_support"] = {
        "factual_claims": support.total,
        "unsupported_before": support.unsupported_count,
        "dangling_citations_before": len(support.dangling_citations),
        "repair_attempted": False,
    }
    if support.clean:
        session.metadata["claim_support"]["citations_attached"] = (
            attach_resolved_citations(answer, support, session.evidence)
        )
        return answer

    try:
        repaired: FinalAnswer = await synth.structured_report(
            "Revise the final answer so every stated fact resolves to evidence.",
            session,
            FinalAnswer,
            extra_context="\n\n".join([*extra, unsupported_brief(support)]),
        )
        session.metadata["claim_support"]["repair_attempted"] = True
        repaired.disagreements = list(
            dict.fromkeys([*repaired.disagreements, *answer.disagreements])
        )
        answer = repaired
        support = check_answer(answer, session.evidence)
    except (ModelError, StructuredOutputError) as exc:
        log.warning("claim_repair_failed", error=str(exc))

    answer, demoted, dropped = demote_unsupported(answer, support)
    attached = attach_resolved_citations(answer, support, session.evidence)
    session.metadata["claim_support"].update(
        {
            "unsupported_after": support.unsupported_count,
            "demoted": demoted,
            "citations_dropped": dropped,
            "resolved_citations": len(support.resolved_citations),
            "citations_attached": attached,
        }
    )
    log.info(
        "claim_support_enforced",
        session_id=session.session_id,
        summary=support.summary(),
        demoted=demoted,
        dropped=dropped,
    )
    return answer


def _enforce_sufficiency(answer: FinalAnswer, session: InvestigationState) -> FinalAnswer:
    """Refuse definite existence claims that outrun the search behind them.

    The claim gate above asks whether each stated fact resolves to evidence.
    It cannot catch this one, because "there is no billing pod" is a claim
    about the *absence* of evidence and resolves to nothing by construction.
    An answer that asserts absence after every cluster timed out passes the
    claim gate cleanly, which is how the failure survived every model size
    measured.

    Deterministic and never consults a model. Under a failed search the answer
    is demoted to unknown whatever the model concluded, because the operator
    acting on it has no way to tell the two situations apart from the text.
    """
    # The operator's own statement only. Evidence excerpts carry retrieved
    # runbooks, and a runbook about investigating timeouts contains the word
    # "timeout", which classified a healthy case as a failed search and made
    # the gate corrupt a correct answer. Whether a search actually failed is a
    # structural fact: it lives in the executions and the recorded tool
    # errors, not in the prose of reference material.
    observations = session.user_request
    result = sufficiency_check(
        " ".join(
            filter(
                None,
                [
                    answer.answer or "",
                    *(answer.observed_facts or []),
                ],
            )
        ),
        observations=observations,
        risks="\n".join(session.risks),
        executions=len(session.commands_executed),
    )
    session.metadata["sufficiency"] = {
        "retrieval": str(result.retrieval),
        "overreaching": result.overreaching,
        "claims": len(result.absence_claims) + len(result.presence_claims),
    }
    demoted = 0
    if result.overreaching:
        answer, demoted = demote_overreach(answer, result)

    # Staleness is the same question asked of time rather than of reach: is
    # what this rests on good enough to state as current? A note nobody has
    # verified in fourteen months was treated exactly like one verified three
    # days ago on every model measured.
    currency = classify_currency(
        observations=session.user_request,
        freshness=[str(e.freshness) for e in session.evidence],
        stale_after_days=_stale_after_days(),
        executions=len(session.commands_executed),
    )
    answer, stale_demoted = demote_stale(answer, currency)
    session.metadata["sufficiency"].update(
        {"demoted": demoted, "currency": str(currency), "stale_demoted": stale_demoted}
    )
    if demoted or stale_demoted:
        log.info(
            "sufficiency_enforced",
            session_id=session.session_id,
            retrieval=str(result.retrieval),
            currency=str(currency),
            demoted=demoted,
            stale_demoted=stale_demoted,
        )
    return answer


_STALE_AFTER_DAYS_DEFAULT = 180
"""Mirrors config.knowledge.stale_after_days so the fallback is the real
default rather than a number invented at the call site.

The first version of this reached for ``settings.memory.stale_after_days``,
which does not exist. getattr returned the literal written beside it and the
gate ran on a 30-day window with nothing logged. Wrong config paths that fall
back quietly are the same shape as the nine defects already catalogued here:
the working path and the broken path produce identical output.
"""


def _stale_after_days() -> int:
    """The configured freshness window.

    A verification gate that crashes the run it protects has made things worse
    than the bug it was added for, so a settings failure falls back rather than
    raising. It says so in the log, because a silent fallback is what this
    comment is about.
    """
    try:
        from mimir.config import get_settings

        return int(get_settings().knowledge.stale_after_days)
    except Exception as exc:  # noqa: BLE001 - a gate must not break the run
        log.warning("stale_window_unreadable", error=str(exc),
                    using=_STALE_AFTER_DAYS_DEFAULT)
        return _STALE_AFTER_DAYS_DEFAULT


def _enforce_grounding(answer: FinalAnswer, session: InvestigationState) -> FinalAnswer:
    """Did the answer name anything nobody read?

    The agent loop has run this check for some time. The investigation graph,
    which is the path the ops corpus exercises, did not. An answer naming pods
    that appear in no listing scored as an ordinary answer on every model size
    measured.

    What the operator asked is part of the ground truth. Repeating back a
    workload name they supplied is not an invention, and flagging it would
    teach them to ignore the warning.
    """
    observed = "\n".join(
        [
            *(e.excerpt for e in session.evidence[:40]),
            *(r.render() for r in session.reports[:12]),
        ]
    )
    result = grounding_check(
        " ".join(
            filter(None, [answer.answer or "", *(answer.observed_facts or [])])
        ),
        observed,
        asked=session.user_request,
    )
    session.metadata["grounding"] = {
        "checked": result.checked,
        "ungrounded": len(result.ungrounded),
        "rate": result.rate,
    }
    if result.ok:
        return answer
    answer, demoted = demote_ungrounded(answer, result)
    session.metadata["grounding"]["demoted"] = demoted
    log.info(
        "grounding_enforced",
        session_id=session.session_id,
        ungrounded=result.ungrounded[:8],
        demoted=demoted,
    )
    return answer


def _enforce_retry_signature(
    answer: FinalAnswer, session: InvestigationState
) -> FinalAnswer:
    """Withdraw a retry diagnosis the timing does not show.

    Retries with backoff leave the same identifier several times with the gap
    between attempts roughly doubling. Shown a single request and a batch job
    that opened four hundred connections, every model measured still blamed
    retries, which makes the diagnosis uninformative: it appears whether or
    not the pattern is there.

    One-directional. A present signature is consistent with retries causing
    the incident and does not establish it, so a match never raises
    confidence. Only the absence demotes.
    """
    if not claims_retries(answer.answer or ""):
        return answer
    observations = "\n".join(
        [session.user_request, *(e.excerpt for e in session.evidence[:40])]
    )
    evidence = retry_from_text(observations)
    session.metadata["retry_signature"] = {
        "occurrences": evidence.occurrences,
        "gaps": evidence.gaps[:8],
        "storm": evidence.is_storm,
    }
    answer, demoted = demote_unsupported_retry(answer, evidence)
    if demoted:
        session.metadata["retry_signature"]["demoted"] = demoted
        log.info(
            "retry_signature_absent",
            session_id=session.session_id,
            occurrences=evidence.occurrences,
            demoted=demoted,
        )
    return answer


def _fallback_answer(
    session: InvestigationState,
    reports: list[SpecialistReport],
    disagreements: list[str],
    error: str,
) -> FinalAnswer:
    """Never lose the work because the synthesis call failed."""
    observed = [
        e.claim for e in session.ranked_evidence(limit=12) if e.kind == EvidenceKind.OBSERVED
    ]
    lines = [r.conclusion for r in reports if r.conclusion]
    return FinalAnswer(
        answer=(
            "Synthesis could not be completed, so here are the raw specialist findings.\n\n"
            + "\n\n".join(lines)
        ),
        confidence=0.2,
        observed_facts=observed,
        unverified=[f"final synthesis failed: {error}"],
        disagreements=disagreements,
        next_steps=["Re-run synthesis, or read the specialist reports directly."],
    )


def _detect_disagreements(reports: list[SpecialistReport]) -> list[str]:
    """Surface conflicts the model might be tempted to tidy away (ADR 7.2)."""
    out: list[str] = []
    for report in reports:
        for contradiction in report.contradictions:
            out.append(f"{report.specialist.value}: {contradiction}")
    confident = [r for r in reports if r.confidence >= 0.7 and not r.failed]
    doubtful = [r for r in reports if r.confidence <= 0.35 and not r.failed]
    if confident and doubtful:
        out.append(
            f"confidence split: {', '.join(r.specialist.value for r in confident)} are "
            f"confident while {', '.join(r.specialist.value for r in doubtful)} are not"
        )
    return out


def _final_confidence(
    answer: FinalAnswer, reports: list[SpecialistReport], session: InvestigationState
) -> float:
    """Confidence is the weakest link, not the average (ADR 7.1 S10, 21.3)."""
    scores = [r.confidence for r in reports if not r.failed]
    if not scores:
        return 0.15
    confidence = min(min(scores), answer.confidence or 1.0)
    if not session.evidence:
        confidence = min(confidence, 0.25)
    if answer.disagreements:
        confidence -= 0.1
    if answer.unverified:
        confidence -= 0.05 * min(3, len(answer.unverified))
    if any(r.failed for r in reports):
        confidence -= 0.1
    return round(max(0.05, min(0.95, confidence)), 2)


async def curate_memory(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Propose, never promote (ADR 11.6, NG4)."""
    session = state["session"]
    if session.final_confidence < 0.4 or not session.evidence:
        return {"session": session, "route": "done"}

    spec = deps.registry.get("propose_memory_note")
    if spec is None:
        return {"session": session, "route": "done"}

    answer = session.final_answer
    body_lines = [
        f"# {session.user_request}",
        "",
        (answer.answer if answer else ""),
        "",
        "## Evidence",
        *[f"- {e.claim} ({', '.join(c.render() for c in e.citations) or e.source_id})"
          for e in session.ranked_evidence(limit=10)],
    ]
    result = await spec.invoke(
        {
            "title": session.user_request[:120],
            "body": "\n".join(body_lines),
            "category": "history/investigations",
            "confidence": session.final_confidence,
            "sources": [e.source_id for e in session.ranked_evidence(limit=10)],
            "verification_status": "unverified",
        },
        deps.tool_context,
    )
    proposals = []
    if result.ok and isinstance(result.data, dict) and result.data.get("proposal"):
        try:
            proposals = [MemoryProposal.model_validate(result.data["proposal"])]
        except Exception as exc:  # noqa: BLE001
            log.warning("memory_proposal_invalid", error=str(exc))

    return {"session": session, "memory_proposals": proposals, "route": "done"}


async def finalise(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Last node. Merges everything and fires the session-complete hook."""
    session = merge_into_session(state)
    _harvest_executions(session, deps)
    observed = _harvest_model_calls(session, deps)
    if session.completed_at is None:
        session.completed_at = time.time()
    hooks = deps.tool_context.hooks
    if hooks is not None:
        await hooks.on_session_complete(session)
    log.info(
        "investigation_complete",
        session_id=session.session_id,
        confidence=session.final_confidence,
        evidence=len(session.evidence),
        commands=len(session.commands_executed),
        model_calls=len(session.model_calls),
        model_invocations=observed,
        duration_s=round(session.duration_s, 2),
    )
    return {"session": session, "route": "done"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _harvest_executions(session: InvestigationState, deps: NodeDeps) -> None:
    """Pull this session's executed commands into the durable state.

    Typed helpers execute through the shared CommandExecutor and return a
    ToolResult; the ExecutionRecord itself stays in the executor. Collecting it
    here is what makes the audit trail real rather than nominal, and it has to
    happen before the session is persisted.
    """
    executor = getattr(deps.tool_context, "executor", None)
    if executor is None or not hasattr(executor, "history_for"):
        return
    known = {record.id for record in session.commands_executed}
    for record in executor.history_for(session.session_id):
        if record.id not in known:
            session.record_execution(record)
            known.add(record.id)


def _harvest_model_calls(session: InvestigationState, deps: NodeDeps) -> int:
    """Copy this session's model invocations onto the state for persistence.

    Returns the number observed. The router counts invocations at the call site
    (`invocations_attempted`); this counts what was attributable to this
    session. The persistence layer then writes them in the same transaction as
    the session row, so the telemetry invariant

        model invocations observed == model-call records persisted

    can be asserted without depending on the order two writers happen to run in.

    The first attempt wrote directly to the repository from here and failed the
    session_id foreign key on every row, because the session is not on disk yet
    at this point. That produced observed=9, persisted=0 - visible only because
    the invariant was added at the same time as the instrumentation.
    """
    router = getattr(deps, "router", None)
    if router is None:
        return 0

    mine = [
        record
        for record in getattr(router, "call_log", [])
        if record.session_id == session.session_id
    ]
    known = {c.get("invocation_id") for c in session.model_calls}
    for record in mine:
        if record.invocation_id in known:
            continue
        known.add(record.invocation_id)
        session.model_calls.append(
            {
                "invocation_id": record.invocation_id,
                "alias": record.alias,
                "model": record.model,
                "runtime": record.runtime,
                "task_class": record.task_class,
                "specialist": record.specialist,
                "latency_ms": record.latency_ms,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "context_size": record.context_window,
                "tool_calls": record.tool_calls,
                "attempt": record.attempt,
                "ok": record.ok,
                "error": record.error,
                "started_at": record.started_at,
                "metadata": {
                    "digest": record.digest,
                    "purpose": record.purpose,
                    "status": record.status,
                    "error_type": record.error_type,
                    "completed_at": record.completed_at,
                    "context_estimate": record.context_estimate,
                    "trimmed": record.trimmed,
                    "tool_calls_before": record.tool_calls_before,
                    "tool_calls_after": record.tool_calls_after,
                    "finish_reason": record.finish_reason,
                },
            }
        )
    return len(mine)


def _render_memory_context(evidence: list[Evidence], limit: int = 6) -> str:
    notes = [
        e
        for e in evidence
        if e.source_type
        in (SourceType.RUNBOOK, SourceType.HISTORICAL_INCIDENT, SourceType.IMPORTED_MEMORY)
    ][:limit]
    if not notes:
        return ""
    return "\n".join(
        f"  [{e.freshness.value}] {e.claim}: {e.excerpt[:300]}" for e in notes
    )


def _render_prior_findings(session: InvestigationState, limit: int = 8) -> str:
    if not session.reports:
        return ""
    lines = ["Findings from earlier specialists in this investigation:"]
    for report in session.reports[-3:]:
        lines.append(f"  {report.specialist.value}: {report.conclusion[:300]}")
    for evidence in session.ranked_evidence(limit=limit):
        lines.append(f"  evidence: {evidence.claim[:200]}")
    return "\n".join(lines)


def promote_hypotheses(session: InvestigationState, reports: list[SpecialistReport]) -> None:
    """Merge hypotheses from reports, keeping rejected ones visible (ADR 5.6)."""
    for report in reports:
        for hypothesis in report.hypotheses:
            if hypothesis.status == HypothesisStatus.REJECTED:
                session.rejected_hypotheses.append(hypothesis)
            else:
                session.upsert_hypothesis(hypothesis)


__all__ = [
    "Hypothesis",
    "NodeDeps",
    "ask_user",
    "coordinate",
    "curate_memory",
    "finalise",
    "gather",
    "promote_hypotheses",
    "recall_memory",
    "resolve_context",
    "run_specialist_step",
    "safety_review",
    "select_skills",
    "synthesise",
    "verify",
]
