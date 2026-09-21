"""The council (ADR 7).

Each specialist is the same machine with different wiring: a narrow system
prompt, a restricted tool set, a bounded tool-calling loop, and a structured
report. That uniformity is what makes the council auditable rather than a
group-chat gimmick (ADR 7 preamble).

The loop is deliberately small and explicit rather than delegated to a framework
agent, because the ADR requires visibility into tool calls, evidence capture,
and per-specialist context limits.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mimir.config import Settings, get_settings
from mimir.council.prompts import specialist_system_prompt
from mimir.llm.base import GenerationOptions, LLMMessage, ModelError, ToolCall
from mimir.llm.router import ModelRouter, TaskClass, get_router
from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.models.evidence import Evidence, EvidenceKind, SourceType
from mimir.models.specialist import (
    Hypothesis,
    SpecialistName,
    SpecialistReport,
)
from mimir.models.state import InvestigationState
from mimir.tools.base import (
    REGISTRY,
    Capability,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

log = get_logger(__name__)


#: Generic helpers (parallel search, document reader) that every specialist may
#: use. They gather nothing on their own; they fan out to the capability-specific
#: tools, which apply their own restrictions.
_COMMON = (Capability.INTERNAL,)

#: Which capabilities each specialist may touch (ADR 7: "Restricted tool access").
SPECIALIST_CAPABILITIES: dict[SpecialistName, tuple[Capability, ...]] = {
    SpecialistName.COORDINATOR: (Capability.MEMORY, Capability.SKILLS, *_COMMON),
    SpecialistName.REPOSITORY_EXPLORER: (
        Capability.REPOSITORY,
        Capability.MEMORY,
        Capability.SKILLS,
        *_COMMON,
    ),
    SpecialistName.BEHAVIOUR_VERIFIER: (
        Capability.REPOSITORY,
        Capability.KUBERNETES,
        Capability.MEMORY,
        *_COMMON,
    ),
    SpecialistName.KUBERNETES_INVESTIGATOR: (
        Capability.KUBERNETES,
        Capability.LOGS,
        Capability.MEMORY,
        Capability.SKILLS,
        *_COMMON,
    ),
    SpecialistName.SDM_INVESTIGATOR: (
        Capability.SDM,
        Capability.DATABASE,
        Capability.LOGS,
        Capability.MEMORY,
        Capability.SKILLS,
        *_COMMON,
    ),
    SpecialistName.LOG_ANALYST: (
        Capability.LOGS,
        Capability.SANDBOX,
        Capability.KUBERNETES,
        Capability.MEMORY,
        *_COMMON,
    ),
    SpecialistName.WEB_RESEARCHER: (Capability.WEB, Capability.MEMORY, *_COMMON),
    SpecialistName.MEMORY_CURATOR: (Capability.MEMORY, *_COMMON),
    # The safety reviewer reads and reasons. It gets no execution capability at
    # all, so a compromised reviewer cannot run the thing it was asked to judge.
    SpecialistName.SAFETY_REVIEWER: (Capability.MEMORY,),
    SpecialistName.SYNTHESIS: (),
}

#: Per-specialist model routing (ADR 18.4).
SPECIALIST_TASK_CLASS: dict[SpecialistName, str] = {
    SpecialistName.COORDINATOR: TaskClass.CLASSIFICATION,
    SpecialistName.REPOSITORY_EXPLORER: TaskClass.DEEP_INVESTIGATION,
    SpecialistName.BEHAVIOUR_VERIFIER: TaskClass.EVIDENCE_VERIFICATION,
    SpecialistName.KUBERNETES_INVESTIGATOR: TaskClass.FAST_COMMAND,
    SpecialistName.SDM_INVESTIGATOR: TaskClass.FAST_COMMAND,
    SpecialistName.LOG_ANALYST: TaskClass.DEEP_INVESTIGATION,
    SpecialistName.WEB_RESEARCHER: TaskClass.WEB_SYNTHESIS,
    SpecialistName.MEMORY_CURATOR: TaskClass.CLASSIFICATION,
    SpecialistName.SAFETY_REVIEWER: TaskClass.EVIDENCE_VERIFICATION,
    SpecialistName.SYNTHESIS: TaskClass.FINAL_SYNTHESIS,
}

#: Highest risk a specialist's tools may carry. Mutation never happens inside a
#: specialist loop; it goes through prepare/approve/execute (ADR 5.5, 13).
SPECIALIST_MAX_RISK: dict[SpecialistName, RiskClass] = {
    SpecialistName.COORDINATOR: RiskClass.R1,
    SpecialistName.REPOSITORY_EXPLORER: RiskClass.R1,
    SpecialistName.BEHAVIOUR_VERIFIER: RiskClass.R1,
    SpecialistName.KUBERNETES_INVESTIGATOR: RiskClass.R2,
    SpecialistName.SDM_INVESTIGATOR: RiskClass.R2,
    SpecialistName.LOG_ANALYST: RiskClass.R2,
    SpecialistName.WEB_RESEARCHER: RiskClass.R1,
    SpecialistName.MEMORY_CURATOR: RiskClass.R1,
    SpecialistName.SAFETY_REVIEWER: RiskClass.R0,
    SpecialistName.SYNTHESIS: RiskClass.R0,
}


@dataclass(slots=True)
class SpecialistBudget:
    max_tool_calls: int = 8
    max_iterations: int = 6
    max_tool_output_chars: int = 6000
    wall_clock_s: float = 300.0


@dataclass
class SpecialistRun:
    """Everything one specialist turn produced."""

    report: SpecialistReport
    messages: list[LLMMessage] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


class Specialist:
    """A bounded tool-calling agent with a fixed remit."""

    def __init__(
        self,
        name: SpecialistName,
        *,
        registry: ToolRegistry | None = None,
        router: ModelRouter | None = None,
        settings: Settings | None = None,
        budget: SpecialistBudget | None = None,
    ) -> None:
        self.name = name
        self.registry = registry or REGISTRY
        self.router = router or get_router(settings)
        self.settings = settings or get_settings()
        self.budget = budget or SpecialistBudget(
            max_tool_calls=self.settings.graph.max_tool_calls_per_turn
        )

    # -- tools -----------------------------------------------------------

    def available_tools(self, allowed_names: Sequence[str] | None = None) -> list[ToolSpec[Any]]:
        """Tools this specialist may call, optionally narrowed by a skill.

        A skill can only ever narrow the set. It cannot widen it, because the
        intersection is taken against the specialist's own capability list.
        """
        specs = self.registry.select(
            specialist=self.name,
            capabilities=SPECIALIST_CAPABILITIES.get(self.name, ()),
            max_risk=SPECIALIST_MAX_RISK.get(self.name, RiskClass.R1),
        )
        if allowed_names:
            wanted = set(allowed_names)
            specs = [s for s in specs if s.name in wanted]
        return specs

    # -- main loop -------------------------------------------------------

    async def run(
        self,
        objective: str,
        state: InvestigationState,
        *,
        ctx: ToolContext,
        skill_bodies: dict[str, str] | None = None,
        memory_context: str = "",
        allowed_tools: Sequence[str] | None = None,
        extra_context: str = "",
    ) -> SpecialistRun:
        started = time.perf_counter()
        specs = self.available_tools(allowed_tools)
        schemas = self.registry.openai_schemas(specs)
        by_name = {s.name: s for s in specs}

        system = specialist_system_prompt(
            self.name,
            environment_lines=state.environment.render_lines(),
            skill_bodies=skill_bodies,
            memory_context=memory_context,
        )
        user_parts = [
            f"Investigation question: {state.user_request}",
            f"Your objective: {objective}",
        ]
        if extra_context:
            user_parts.append(extra_context)
        if not specs:
            user_parts.append(
                "You have no tools for this turn. Reason from the evidence you were given "
                "and do not claim anything you cannot support from it."
            )

        messages: list[LLMMessage] = [
            LLMMessage.system(system),
            LLMMessage.user("\n\n".join(user_parts)),
        ]

        evidence: list[Evidence] = []
        tool_results: list[ToolResult] = []
        tool_calls_used = 0
        error: str | None = None
        final_text = ""

        # Gathering and concluding are different phases and must not share a
        # budget. The previous version gave the whole iteration budget to
        # gathering, so a specialist that used its tool calls had nothing left
        # to write an answer with, and died reporting that it never concluded.
        #
        # Measured: on qwen3-coder:30b, 16 of 19 corpus failures were this one
        # path. On qwen2.5:7b only 4 of 18, because a weaker model calls fewer
        # tools and falls out of the loop early. Better tool adherence made the
        # bug bite harder, so the constrained-decoding win was feeding it.
        concluded = False
        for _iteration in range(self.budget.max_iterations):
            if time.perf_counter() - started > self.budget.wall_clock_s:
                error = f"specialist exceeded its {self.budget.wall_clock_s:.0f}s budget"
                break

            budget_left = self.budget.max_tool_calls - tool_calls_used
            if budget_left <= 0:
                break

            options = GenerationOptions(tools=schemas if schemas else [])
            try:
                response = await self.router.chat(
                    messages,
                    task_class=SPECIALIST_TASK_CLASS.get(self.name, TaskClass.DEFAULT),
                    options=options,
                    session_id=state.session_id,
                    purpose=f"specialist:{self.name.value}",
                )
            except ModelError as exc:
                error = f"model call failed: {exc.message}"
                break

            final_text = response.content or final_text
            if not response.has_tool_calls:
                concluded = True
                break

            messages.append(response.as_message())
            calls = response.tool_calls[:budget_left]
            for call in calls:
                tool_calls_used += 1
                result = await self._invoke(call, by_name, ctx)
                tool_results.append(result)
                evidence.extend(result.evidence)
                messages.append(
                    LLMMessage.tool_result(
                        call.id,
                        result.render(self.budget.max_tool_output_chars),
                        name=call.name,
                    )
                )

            if len(response.tool_calls) > len(calls):
                messages.append(
                    LLMMessage.user(
                        f"{len(response.tool_calls) - len(calls)} further tool calls were "
                        "dropped because the budget for this turn is exhausted."
                    )
                )

        if not concluded and error is None:
            final_text, error = await self._conclude(
                messages, state, final_text, tool_calls_used
            )

        report = SpecialistReport(
            specialist=self.name,
            objective=objective,
            conclusion=_first_paragraph(final_text),
            detail=final_text,
            evidence=evidence,
            confidence=_estimate_confidence(final_text, evidence, error),
            tool_calls=tool_calls_used,
            duration_s=time.perf_counter() - started,
            error=error,
        )
        log.info(
            "specialist_complete",
            specialist=self.name.value,
            tool_calls=tool_calls_used,
            evidence=len(evidence),
            duration_s=round(report.duration_s, 2),
            error=error,
        )
        return SpecialistRun(report=report, messages=messages, tool_results=tool_results)

    async def _conclude(
        self,
        messages: list[LLMMessage],
        state: Any,
        final_text: str,
        tool_calls_used: int,
    ) -> tuple[str, str | None]:
        """One closing turn with no tools offered, to get an actual answer.

        A turn that cannot call tools is a turn that must answer. Offering
        schemas the budget can no longer honour is what produced the
        non-termination: the model kept calling, every call was dropped for
        lack of budget, and each dropped call cost an iteration until the loop
        died with nothing written.

        This turn is deliberately outside the iteration budget. It makes no
        tool calls, so it cannot loop, and charging it to a budget that
        gathering has already spent is what left nothing to conclude with.

        A failure here is reported rather than swallowed. A specialist that
        gathered evidence and could not write it up is a different and less
        alarming thing than one that never looked, and the two must not
        produce the same empty report.
        """
        messages = [
            *messages,
            LLMMessage.user(
                "Stop here and report. Do not request any more tools. State what "
                "the evidence you already have does and does not show, and say "
                "plainly if it is not enough to answer."
            ),
        ]
        try:
            response = await self.router.chat(
                messages,
                task_class=SPECIALIST_TASK_CLASS.get(self.name, TaskClass.DEFAULT),
                options=GenerationOptions(tools=[]),
                session_id=state.session_id,
                purpose=f"specialist:{self.name.value}:conclude",
            )
        except ModelError as exc:
            return final_text, f"closing call failed: {exc.message}"

        text = response.content or final_text
        if not text.strip():
            return text, (
                f"specialist used {tool_calls_used} tool call(s) and returned no "
                "conclusion when asked to stop and report"
            )
        return text, None

    async def _invoke(
        self, call: ToolCall, by_name: dict[str, ToolSpec[Any]], ctx: ToolContext
    ) -> ToolResult:
        spec = by_name.get(call.name)
        if spec is None:
            # A hallucinated or out-of-remit tool name is a routine local-model
            # failure. Name the allowed set so the next turn can correct itself.
            allowed = ", ".join(sorted(by_name)) or "none"
            return ToolResult.failure(
                call.name,
                f"tool '{call.name}' is not available to the {self.name.value}. "
                f"Available tools: {allowed}",
                code="not_permitted",
            )
        return await spec.invoke(call.arguments, ctx.child(specialist=self.name))

    # -- structured helpers ----------------------------------------------

    async def structured_report(
        self,
        objective: str,
        state: InvestigationState,
        schema: type,
        *,
        extra_context: str = "",
        memory_context: str = "",
    ) -> Any:
        """One-shot structured output with no tools. Used by the Coordinator and
        by Synthesis, which reason over material rather than gathering it."""
        system = specialist_system_prompt(
            self.name,
            environment_lines=state.environment.render_lines(),
            memory_context=memory_context,
        )
        messages = [
            LLMMessage.system(system),
            LLMMessage.user(
                f"Investigation question: {state.user_request}\n\n"
                f"Your objective: {objective}"
                + (f"\n\n{extra_context}" if extra_context else "")
            ),
        ]
        return await self.router.structured(
            messages,
            schema,
            task_class=SPECIALIST_TASK_CLASS.get(self.name, TaskClass.DEFAULT),
            session_id=state.session_id,
            purpose=f"specialist:{self.name.value}:structured",
        )


def _first_paragraph(text: str) -> str:
    for block in (text or "").strip().split("\n\n"):
        cleaned = block.strip()
        if cleaned:
            return cleaned[:600]
    return ""


_HEDGES = (
    "i do not know",
    "i don't know",
    "unverified",
    "cannot confirm",
    "could not find",
    "unable to",
    "no evidence",
    "not reachable",
    "failed",
)


def _estimate_confidence(text: str, evidence: list[Evidence], error: str | None) -> float:
    """A crude prior, refined later by the graph.

    Deliberately pessimistic: an answer with no evidence behind it should not
    present as confident just because the model wrote fluently.
    """
    if error:
        return 0.2
    if not evidence:
        return 0.3
    base = min(0.85, 0.45 + 0.05 * len(evidence))
    lowered = (text or "").lower()
    if any(h in lowered for h in _HEDGES):
        base -= 0.15
    observed = sum(1 for e in evidence if e.kind == EvidenceKind.OBSERVED)
    if observed == 0:
        base -= 0.2
    if any(e.source_type == SourceType.COMMAND_OUTPUT for e in evidence):
        base += 0.05
    if any(not e.supports for e in evidence):
        base -= 0.1
    return max(0.05, min(0.95, base))


def build_council(
    *,
    registry: ToolRegistry | None = None,
    router: ModelRouter | None = None,
    settings: Settings | None = None,
) -> dict[SpecialistName, Specialist]:
    return {
        name: Specialist(name, registry=registry, router=router, settings=settings)
        for name in SpecialistName
    }


def hypothesis_from_text(statement: str, likelihood: float, proposed_by: str) -> Hypothesis:
    return Hypothesis(statement=statement, likelihood=likelihood, proposed_by=proposed_by)
