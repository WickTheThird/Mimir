"""Graph nodes (ADR 6.2 C2, 7.2)."""

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
from mimir.config import get_settings
from mimir.decide import Choice, build_decider, decide_async
from mimir.verify.sufficiency import Retrieval, classify_retrieval
from mimir.verify.sufficiency import check as sufficiency_check
from mimir.verify.sufficiency import (
    classify_currency,
    demote_empty,
    demote_overreach,
    demote_stale,
    state_verdict,
)

log = get_logger(__name__)


# : Task types that can be answered without a full council pass.
DIRECT_ROUTES: dict[TaskType, SpecialistName] = {
    TaskType.COMMAND_CONSTRUCTION: SpecialistName.KUBERNETES_INVESTIGATOR,
    TaskType.MEMORY_LOOKUP: SpecialistName.MEMORY_CURATOR,
    TaskType.WEB_RESEARCH: SpecialistName.WEB_RESEARCHER,
}


class SkillAccess:
    """Thin, failure-tolerant adapter over the skills subsystem."""

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
        """Skill names for a request."""
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
            # The names, not the specs.
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
        decider: Any = None,
    ) -> None:
        self.registry = registry or REGISTRY
        self.router = router or get_router()
        self.tool_context = tool_context or ToolContext()
        self.skills = SkillAccess(skill_registry) if skill_registry is not None else None
        self.knowledge = knowledge
        self.council: dict[SpecialistName, Specialist] = build_council(
            registry=self.registry, router=self.router
        )
        # The closed-set decision model (ADR-004 tier 2).
        self.decider: Any = decider if decider is not None else build_decider(
            get_settings(), router=self.router
        )

    def specialist(self, name: SpecialistName) -> Specialist:
        return self.council[name]


# ---------------------------------------------------------------------------


async def resolve_context(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Fill in the operating context from the shell and config (ADR 5.1 step 1)."""
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


async def recall_memory(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Retrieve curated knowledge before planning (ADR G5, 11.4)."""
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


async def resolve_target(
    session: InvestigationState, store: Any, decider: Any
) -> dict[str, Any]:
    """Bind the workload the operator named to where it actually lives."""
    from mimir.agent.request import parse_request

    parsed = parse_request(session.user_request)
    name = parsed.name_contains
    env = session.environment
    # What the operator stated is bound before anything is looked up.
    if parsed.namespace and not env.namespace:
        env.namespace = parsed.namespace
    if parsed.environment and not env.environment:
        env.environment = parsed.environment
    if not name or store is None:
        return {"status": "unnamed" if not name else "no_store"}
    if env.namespace and env.cluster_context:
        return {"status": "scoped_by_operator", "name": name}

    found = store.candidates(name, kinds=("deployment", "statefulset", "daemonset",
                                          "workload", "pod", "namespace"))
    scopes = sorted({(e.context, e.namespace) for e in found if e.namespace or e.context})
    if not scopes:
        return {"status": "unknown", "name": name}
    if len(scopes) == 1:
        context, namespace = scopes[0]
        env.cluster_context = env.cluster_context or context or None
        env.namespace = env.namespace or namespace or None
        return {"status": "bound", "name": name, "scope": f"{context}/{namespace}",
                "source": "store"}

    options = tuple(f"{c}/{n}" for c, n in scopes)[:25] + ("ask the operator",)
    choice = Choice(
        name="target",
        options=options,
        description=(
            f"Which of these places holds the {name} the operator means. Pick one "
            "only if the request or its context says so; otherwise choose to ask."
        ),
    )
    verdicts = await decide_async(
        decider,
        "\n".join([f"Operator request: {session.user_request}",
                    f"Places where a workload named like '{name}' has been seen:",
                    *(f"- {o}" for o in options[:-1])]),
        [choice], session_id=session.session_id,
    )
    verdict = verdicts.get("target")
    if verdict is not None:
        acted = _acted_on(verdict)
        session.metadata.setdefault("decisions", []).append(
            {"field": "target", "options": list(options), "choice": verdict.choice,
             "probability": verdict.probability, "margin": verdict.margin,
             "calibrated": verdict.calibrated, "truncated": verdict.truncated,
             "backend": getattr(decider, "name", "unknown"), "node": "coordinate",
             "acted": acted}
        )
        if acted and verdict.choice != "ask the operator":
            context, _, namespace = verdict.choice.partition("/")
            env.cluster_context = env.cluster_context or context or None
            env.namespace = env.namespace or namespace or None
            return {"status": "bound", "name": name, "scope": verdict.choice,
                    "source": "decider"}
    question = (
        f"'{name}' exists in more than one place: "
        + ", ".join(options[:-1]) + ". Which one do you mean?"
    )
    if question not in session.pending_questions:
        session.pending_questions.append(question)
    return {"status": "ambiguous", "name": name, "candidates": list(options[:-1])}


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
    # Rows that produced confident answers for this shape before (plan step
    from mimir.council.table import preferred_for

    preferred = preferred_for(str(session.task_type or ""), deps.tool_context.settings)
    if preferred:
        extra.append("Specialists that answered this kind of question confidently before: "
                     + ", ".join(preferred[:4]))

    # Cheapest possible path first.
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

    target = await resolve_target(
        session, getattr(deps.tool_context, "entities", None), getattr(deps, "decider", None)
    )
    session.metadata["target"] = target
    if target.get("status") == "ambiguous":
        # A computed ambiguity is a question, not a plan.
        return {
            "session": session,
            "pending_steps": [],
            "route": "ask_user",
            "notes": [f"target ambiguous: {target['name']} in {len(target['candidates'])} places"],
        }
    if target.get("status") == "bound":
        extra.append(f"Target resolved from prior observations: {target['name']} in {target['scope']}.")

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
    """Used when the model cannot produce a valid plan."""
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


async def select_skills(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Load the bodies of the selected skills only (ADR 10.2 progressive disclosure)."""
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


NEXT_CHOICE = Choice(
    name="next",
    options=("conclude", "continue", "ask"),
    description=(
        "What the investigation should do now. conclude: the evidence in hand "
        "answers the question or no further check would change the answer. "
        "continue: a specific further check, suggested by what was just "
        "found, would change the answer. ask: the operator holds a fact "
        "without which no check can proceed."
    ),
)
"""ADR-004 step 2, ADR-003 phase 3."""


def _progress(session: InvestigationState, round_number: int) -> tuple[int, int]:
    """(new evidence this round, total)."""
    # Observed evidence only.
    seen_key = "evidence_seen_by_round"
    history: dict[str, int] = session.metadata.setdefault(seen_key, {})
    total = sum(1 for e in session.evidence if e.kind == EvidenceKind.OBSERVED)
    previous = history.get(str(round_number - 1), 0)
    history[str(round_number)] = total
    return max(0, total - previous), total


def _assess_context(session: InvestigationState, round_number: int, new: int, total: int) -> str:
    findings = []
    for report in session.reports[-6:]:
        head = (report.conclusion or report.detail or "").strip().splitlines()
        findings.append(f"- {report.specialist.value}: {head[0][:240] if head else '(no conclusion)'}"
                        + (f" [failed: {report.error}]" if report.failed else ""))
    hypotheses = [f"- {h.statement[:160]}" for h in session.hypotheses[:5] if getattr(h, "statement", "")]
    return "\n".join(
        [
            f"Question: {session.user_request}",
            f"Round {round_number} finished. New evidence this round: {new}. Total: {total}.",
            "Findings so far:",
            *(findings or ["- none"]),
            *(["Open hypotheses:", *hypotheses] if hypotheses else []),
            *([f"Recorded failures: {len(session.risks)}"] if session.risks else []),
        ]
    )


PREDICTION_CHOICE = Choice(
    name="holds",
    options=("confirmed", "contradicted", "untested"),
    description=(
        "Whether the evidence gathered this round bears on the hypothesis's "
        "stated next check. confirmed: the check came back as the hypothesis "
        "predicts. contradicted: it came back the other way. untested: nothing "
        "this round speaks to it."
    ),
)
"""ADR-003 phase 5, plan step 7."""


async def _score_predictions(
    session: InvestigationState, deps: Any, round_number: int, new: int
) -> list[dict[str, Any]]:
    if new == 0:
        return []
    open_ones = [h for h in session.hypotheses
                 if h.status != HypothesisStatus.REJECTED and h.next_check][:6]
    if not open_ones:
        return []
    observed = [e for e in session.evidence if e.kind == EvidenceKind.OBSERVED]
    recent = observed[-min(len(observed), max(new, 1)):]
    evidence_block = "\n".join(f"- {e.claim[:120]}: {e.excerpt[:200]}" for e in recent[:12])
    residuals: list[dict[str, Any]] = []
    for h in open_ones:
        verdicts = await decide_async(
            getattr(deps, "decider", None),
            "\n".join([f"Hypothesis: {h.statement}", f"Predicted next check: {h.next_check}",
                        "Evidence gathered this round:", evidence_block]),
            [PREDICTION_CHOICE], session_id=session.session_id,
        )
        verdict = verdicts.get("holds")
        if verdict is None:
            continue
        acted = _acted_on(verdict)
        session.metadata.setdefault("decisions", []).append(
            {"field": "holds", "options": list(PREDICTION_CHOICE.options), "choice": verdict.choice,
             "probability": verdict.probability, "margin": verdict.margin,
             "calibrated": verdict.calibrated, "truncated": verdict.truncated,
             "backend": getattr(deps.decider, "name", "unknown"), "node": "assess",
             "round": round_number, "acted": acted, "hypothesis": h.id}
        )
        before = h.likelihood
        if acted and verdict.choice == "confirmed":
            h.likelihood = round(min(0.95, h.likelihood + 0.2), 3)
            h.supporting_evidence_ids += [e.id for e in recent[:3] if e.id not in h.supporting_evidence_ids]
        elif acted and verdict.choice == "contradicted":
            h.likelihood = round(max(0.05, h.likelihood - 0.25), 3)
            h.contradicting_evidence_ids += [e.id for e in recent[:3] if e.id not in h.contradicting_evidence_ids]
            if h.likelihood < 0.15:
                h.status = HypothesisStatus.REJECTED
                h.rejected_reason = "predicted check came back the other way"
        residuals.append({"hypothesis": h.id, "holds": verdict.choice, "acted": acted,
                          "likelihood_before": before, "likelihood_after": h.likelihood})
    return residuals


def _record_round(session: InvestigationState, round_number: int, new: int) -> None:
    """State without the transcript (ADR-003 phase 2, smallest form)."""
    rounds: list[dict[str, Any]] = session.metadata.setdefault("rounds", [])
    recent = session.evidence[-new:] if new else []
    rounds.append({
        "round": round_number,
        "observations": [e.id for e in recent],
        "claims": [{"by": r.specialist.value, "claim": (r.conclusion or "")[:200],
                    "evidence": [e.id for e in r.evidence[:6]], "confidence": r.confidence}
                   for r in session.reports[-6:]],
        "hypotheses": [{"id": h.id, "likelihood": h.likelihood, "status": str(h.status)}
                       for h in session.hypotheses[:10]],
        "open_questions": [q for r in session.reports[-6:] for q in r.open_questions[:2]][:6],
    })


async def assess(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Decide conclude / continue / ask after a round, by policy first."""
    session = state["session"]
    round_number = int(state.get("round", 0))
    max_rounds = deps.tool_context.settings.graph.max_specialist_rounds
    new, total = _progress(session, round_number)

    reason = ""
    choice = "conclude"
    if round_number >= max_rounds:
        reason = f"round cap {max_rounds} reached"
    elif new == 0 and round_number > 0:
        reason = "no new evidence this round"
    elif not session.reports:
        reason = "nothing gathered to assess"
    else:
        verdicts = await decide_async(
            getattr(deps, "decider", None),
            _assess_context(session, round_number, new, total),
            [NEXT_CHOICE],
            session_id=session.session_id,
        )
        verdict = verdicts.get("next")
        if verdict is None:
            reason = "no decision model; single pass"
        else:
            acted = _acted_on(verdict)
            session.metadata.setdefault("decisions", []).append(
                {
                    "field": "next", "options": list(NEXT_CHOICE.options),
                    "choice": verdict.choice, "probability": verdict.probability,
                    "margin": verdict.margin, "calibrated": verdict.calibrated,
                    "truncated": verdict.truncated,
                    "backend": getattr(deps.decider, "name", "unknown"),
                    "node": "assess", "round": round_number, "acted": acted,
                }
            )
            choice = verdict.choice if acted else "conclude"
            reason = "decided" if acted else "verdict below floor; concluding"

    residuals = await _score_predictions(session, deps, round_number, new)
    session.metadata.setdefault("assess", []).append(
        {"round": round_number, "new_evidence": new, "total_evidence": total,
         "choice": choice, "reason": reason, "predictions": residuals}
    )
    _record_round(session, round_number, new)
    log.info("assessed", round=round_number, choice=choice, reason=reason, new=new)

    if choice == "continue":
        route = "replan"
    elif choice == "ask":
        route = "replan_ask"
    else:
        route = state.get("route", "safety_review")
        if route not in ("verify", "safety_review"):
            route = "safety_review"
    return {"session": session, "route": route,
            "notes": [f"assess round {round_number}: {choice} ({reason})"]}


async def replan(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Ask the coordinator for the next steps given what was found."""
    session = state["session"]
    asking = state.get("route") == "replan_ask"
    done = {(r.specialist, r.objective.strip().lower()) for r in session.reports}
    findings = "\n".join(
        f"- {r.specialist.value}: {(r.conclusion or '')[:300]}" for r in session.reports[-8:]
    )
    extra = [
        f"Round {state.get('round', 0)} is complete. Findings so far:\n{findings or '- none'}",
        "Plan only the further checks that these findings make necessary. Do not "
        "repeat a completed objective. If a fact from the operator is required "
        "before any check can proceed, put the question in missing_context and "
        "plan no steps.",
    ]
    if asking:
        extra.append("The assessment concluded a question for the operator is needed.")
    coordinator = deps.specialist(SpecialistName.COORDINATOR)
    try:
        plan: CoordinatorPlan = await coordinator.structured_report(
            "Plan the next round of the investigation.", session, CoordinatorPlan,
            extra_context="\n\n".join(extra),
        )
    except (ModelError, StructuredOutputError) as exc:
        log.warning("replan_failed", error=str(exc))
        return {"session": session, "pending_steps": [], "route": "safety_review",
                "notes": ["replan failed; concluding on what was gathered"]}

    # No fallback step on a replan.
    steps = [
        st for st in (_sanitise_steps(plan.steps, session) if plan.steps else [])
        if (st.specialist, st.objective.strip().lower()) not in done
    ]
    for question in plan.missing_context:
        if question not in session.pending_questions:
            session.pending_questions.append(question)
    if steps:
        route = "select_skills"
    elif plan.missing_context or asking:
        route = "ask_user"
    else:
        route = "safety_review"
    return {"session": session, "pending_steps": steps, "route": route,
            "notes": [f"replan: {len(steps)} new step(s), {len(plan.missing_context)} question(s)"]}


SUFFICIENT_CHOICE = Choice(
    name="sufficient",
    options=("yes", "no"),
    description=(
        "Whether the findings and evidence gathered answer the question that was "
        "asked. yes: an operator could act on this. no: a specific thing is still "
        "unknown and the answer would have to guess it."
    ),
)
CONFLICT_CHOICE = Choice(
    name="conflict",
    options=("consistent", "contradictory", "unrelated"),
    description=(
        "Whether the specialists' conclusions agree. consistent: they support one "
        "account. contradictory: at least two cannot both be true. unrelated: they "
        "answer different questions and neither confirms nor denies the other."
    ),
)
"""ADR-004 step 4."""


def _verify_context(session: InvestigationState, reports: list[SpecialistReport]) -> str:
    lines = [f"Question: {session.user_request}", "", "Specialist conclusions:"]
    for r in reports[:8]:
        lines.append(f"- {r.specialist.value} (confidence {r.confidence:.2f}): "
                     f"{(r.conclusion or r.detail or '').strip()[:400]}")
        for q in r.open_questions[:3]:
            lines.append(f"    open: {q[:160]}")
    lines += ["", "Evidence (most trusted first):"]
    for e in session.ranked_evidence(limit=12):
        lines.append(f"- [{e.source_type}] {e.claim[:120]}: {e.excerpt[:200]}")
    return "\n".join(lines)


async def verify(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Behaviour verification (ADR 7.2 step 5), as two decisions."""
    session = state["session"]
    reports = [r for r in state.get("reports", []) if not r.failed]
    if not reports:
        return {"route": "safety_review"}

    decider = getattr(deps, "decider", None)
    verdicts = await decide_async(
        decider, _verify_context(session, reports),
        [SUFFICIENT_CHOICE, CONFLICT_CHOICE], session_id=session.session_id,
    )
    if verdicts:
        applied: dict[str, Any] = {}
        for field, verdict in verdicts.items():
            acted = _acted_on(verdict)
            session.metadata.setdefault("decisions", []).append(
                {"field": field, "options": list(
                    SUFFICIENT_CHOICE.options if field == "sufficient" else CONFLICT_CHOICE.options),
                 "choice": verdict.choice, "probability": verdict.probability,
                 "margin": verdict.margin, "calibrated": verdict.calibrated,
                 "truncated": verdict.truncated, "backend": getattr(decider, "name", "unknown"),
                 "node": "verify", "acted": acted}
            )
            if acted:
                applied[field] = verdict.choice
        if applied.get("sufficient") == "no":
            for report in reports:
                report.confidence = round(max(0.2, report.confidence - 0.2), 3)
            note = "the evidence gathered does not settle the question asked"
            if note not in session.risks:
                session.risks.append(note)
        if applied.get("conflict") == "contradictory":
            for report in reports:
                report.contradictions.append(
                    "another specialist's conclusion cannot be true at the same time"
                )
                report.confidence = round(max(0.2, report.confidence - 0.2), 3)
        session.metadata["verify"] = {"mode": "decided", **applied}
        return {"session": session, "reports": [], "route": "safety_review"}

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
    session.metadata["verify"] = {"mode": "generative"}
    return {
        "session": session,
        "reports": [run.report],
        "evidence": run.report.evidence,
        "route": "safety_review",
    }


async def safety_review(state: GraphState, deps: NodeDeps) -> dict[str, Any]:
    """Safety and Command Reviewer pass (ADR 7.1 S9, 7.2 step 6)."""
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
    answer = await _enforce_claim_support(answer, session, synth, extra)
    answer = await _enforce_sufficiency(answer, session, deps)
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
    """Resolve every stated fact against evidence, deterministically."""
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


RETRIEVAL_CHOICE = Choice(
    name="retrieval",
    options=("observed", "empty", "failed"),
    description=(
        "What happened when the system went to look. observed: a query ran "
        "and returned data. empty: a query ran to completion and found "
        "nothing. failed: the query could not run, timed out, was refused, "
        "or produced no listing at all."
    ),
)
"""ADR-004 step 1."""


def _listing_noun(request: str) -> str:
    text = request.lower()
    for noun in ("pods", "deployments", "namespaces", "services", "nodes", "jobs", "files", "matches"):
        if noun in text or noun[:-1] in text:
            return noun
    return "items"


def _acted_on(verdict: Any) -> bool:
    """Whether a verdict clears the floor to be acted on."""
    config = get_settings().decisions
    if verdict.margin < config.min_margin:
        return False
    return (not verdict.calibrated) or verdict.probability >= config.min_probability


def _execution_counts(session: InvestigationState) -> tuple[int, int, int]:
    """(failed, empty, observed) from the commands that actually ran."""
    failed = empty = observed = 0
    for record in session.commands_executed:
        code = record.exit_code
        if code is None:
            continue
        if code != 0:
            failed += 1
        elif record.stdout.strip():
            observed += 1
        else:
            empty += 1
    return failed, empty, observed


async def _decide_retrieval(
    session: InvestigationState, deps: Any
) -> Retrieval | None:
    """Ask the decision model, and log the decision whatever it says."""
    decider = getattr(deps, "decider", None)
    context = "\n".join(
        [
            f"Operator request: {session.user_request}",
            *(f"Recorded failure: {r}" for r in session.risks[:8]),
        ]
    )
    verdicts = await decide_async(
        decider, context, [RETRIEVAL_CHOICE], session_id=session.session_id
    )
    verdict = verdicts.get("retrieval")
    if verdict is None:
        return None
    acted = _acted_on(verdict)
    session.metadata.setdefault("decisions", []).append(
        {
            "field": verdict.field,
            "options": list(RETRIEVAL_CHOICE.options),
            "choice": verdict.choice,
            "probability": verdict.probability,
            "margin": verdict.margin,
            "calibrated": verdict.calibrated,
            "truncated": verdict.truncated,
            "backend": getattr(decider, "name", "unknown"),
            "node": "synthesise",
            "acted": acted,
        }
    )
    # A calibrated verdict below the floor is logged and not used: the
    return Retrieval(verdict.choice) if acted else None


async def _enforce_sufficiency(
    answer: FinalAnswer, session: InvestigationState, deps: Any = None
) -> FinalAnswer:
    """Refuse definite existence claims that outrun the search behind them."""
    failed, empty, observed = _execution_counts(session)
    retrieval: Retrieval | None = None
    source = "text"
    if failed or empty or observed:
        retrieval = classify_retrieval(failed=failed, empty=empty, observed=observed)
        source = "executions"
    else:
        # The operator's own explicit words outrank a judged verdict.
        stated = classify_retrieval(observations=session.user_request,
                                    risks="\n".join(session.risks))
        if stated is not Retrieval.UNKNOWN:
            retrieval = stated
            source = "statement"
        else:
            retrieval = await _decide_retrieval(session, deps)
            if retrieval is not None:
                source = "decider"

    observations = session.user_request
    result = sufficiency_check(
        " ".join(
            filter(None, [answer.answer or "", *(answer.observed_facts or [])])
        ),
        observations=observations,
        risks="\n".join(session.risks),
        executions=0,
        retrieval=retrieval,
    )
    session.metadata["sufficiency"] = {
        "retrieval": str(result.retrieval),
        "retrieval_source": source,
        "overreaching": result.overreaching,
        "claims": len(result.absence_claims) + len(result.presence_claims),
    }
    demoted = 0
    if result.overreaching:
        answer, demoted = demote_overreach(answer, result)
    answer, stated_none = demote_empty(
        answer, result.retrieval, noun=_listing_noun(session.user_request)
    )
    demoted += stated_none
    answer, led = state_verdict(answer, result.retrieval, noun=_listing_noun(session.user_request))
    demoted += led

    # Staleness is the same question asked of time rather than of reach: is
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
"""Mirrors config.knowledge.stale_after_days so the fallback is the real default rather than a number invented at the call site."""


def _stale_after_days() -> int:
    """The configured freshness window."""
    try:
        from mimir.config import get_settings

        return int(get_settings().knowledge.stale_after_days)
    except Exception as exc:  # noqa: BLE001 - a gate must not break the run
        log.warning("stale_window_unreadable", error=str(exc),
                    using=_STALE_AFTER_DAYS_DEFAULT)
        return _STALE_AFTER_DAYS_DEFAULT


def _enforce_grounding(answer: FinalAnswer, session: InvestigationState) -> FinalAnswer:
    """Did the answer name anything nobody read?"""
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
    """Withdraw a retry diagnosis the timing does not show."""
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
    # What the organisation keeps from this session (plan step 8).
    try:
        from mimir.knowledge.experience import get_experience_store, write_corpus_draft

        session.metadata["experience"] = get_experience_store(
            deps.tool_context.settings
        ).record(session)
        draft = write_corpus_draft(deps.tool_context.settings.home, session)
        if draft is not None:
            session.metadata["corpus_draft"] = str(draft)
        from mimir.knowledge.skill_drafts import draft_from_session, write_draft

        skill = draft_from_session(session)
        if skill is not None:
            session.metadata["skill_draft"] = str(
                write_draft(deps.tool_context.settings.home, skill)
            )
    except Exception as exc:  # noqa: BLE001 - experience must not fail the session
        log.warning("experience_skipped", error=str(exc))
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


def _harvest_executions(session: InvestigationState, deps: NodeDeps) -> None:
    """Pull this session's executed commands into the durable state."""
    executor = getattr(deps.tool_context, "executor", None)
    if executor is None or not hasattr(executor, "history_for"):
        return
    known = {record.id for record in session.commands_executed}
    for record in executor.history_for(session.session_id):
        if record.id not in known:
            session.record_execution(record)
            known.add(record.id)


def _harvest_model_calls(session: InvestigationState, deps: NodeDeps) -> int:
    """Copy this session's model invocations onto the state for persistence."""
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
