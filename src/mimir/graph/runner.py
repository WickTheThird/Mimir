"""Investigation runner (ADR 6.1, 14, 15).

One entry point that the CLI, the HTTP API, and the evaluation harness all use,
so the three interfaces cannot drift apart in behaviour. It owns wiring
(registry, router, executor, hooks, checkpointer) and exposes two shapes:

* :meth:`InvestigationRunner.run` for a complete result,
* :meth:`InvestigationRunner.stream` for incremental events.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mimir.config import Settings, get_settings
from mimir.graph.build import build_graph
from mimir.graph.nodes import NodeDeps
from mimir.graph.state import GraphState, initial_state, merge_into_session, state_summary
from mimir.hooks.manager import HookManager, get_hook_manager
from mimir.llm.router import ModelRouter, get_router
from mimir.logging import correlation_context, get_logger
from mimir.models.session import ChatMessage, MessageRole
from mimir.models.specialist import FinalAnswer
from mimir.models.state import EnvironmentContext, InvestigationState
from mimir.safety.approvals import ApprovalBroker, get_approval_broker
from mimir.tools.artifacts import get_artifact_store
from mimir.tools.base import REGISTRY, ToolContext, ToolRegistry, load_all_tools
from mimir.tools.exec import CommandExecutor, get_executor

log = get_logger(__name__)


class EventType(StrEnum):
    STARTED = "started"
    NODE_START = "node_start"
    NODE_END = "node_end"
    PLAN = "plan"
    SPECIALIST = "specialist"
    EVIDENCE = "evidence"
    COMMAND = "command"
    APPROVAL = "approval"
    TOKEN = "token"
    ANSWER = "answer"
    ERROR = "error"
    DONE = "done"


@dataclass(slots=True)
class RunEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "at": self.at, **self.data}


class InvestigationRunner:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: ToolRegistry | None = None,
        router: ModelRouter | None = None,
        executor: CommandExecutor | None = None,
        approvals: ApprovalBroker | None = None,
        hooks: HookManager | None = None,
        skill_registry: Any = None,
        checkpointer: Any = None,
        parallel: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or load_all_tools()
        self.router = router or get_router(self.settings)
        self.approvals = approvals or get_approval_broker(self.settings)
        self.hooks = hooks or get_hook_manager(self.settings)
        self.executor = executor or get_executor(self.settings)
        # The executor needs the hook manager for the mutation lifecycle hooks,
        # and it is constructed before hooks exist in the default path.
        if self.executor.hooks is None:
            self.executor.hooks = self.hooks
        self.skill_registry = skill_registry if skill_registry is not None else _load_skills()
        from mimir.knowledge.entities import get_entity_store

        self.entities = get_entity_store(self.settings)
        self.checkpointer = checkpointer
        self.parallel = (
            self.settings.graph.parallel_specialists if parallel is None else parallel
        )
        self._compiled: Any = None
        self._stack = AsyncExitStack()

    # -- wiring ----------------------------------------------------------

    def tool_context(self, session_id: str | None = None) -> ToolContext:
        return ToolContext(
            settings=self.settings,
            session_id=session_id,
            artifacts=get_artifact_store(self.settings),
            executor=self.executor,
            approvals=self.approvals,
            hooks=self.hooks,
            registry=self.registry,
            entities=self.entities,
        )

    def _deps(self, session_id: str | None) -> NodeDeps:
        return NodeDeps(
            registry=self.registry,
            router=self.router,
            tool_context=self.tool_context(session_id),
            skill_registry=self.skill_registry,
        )

    async def _compile(self, session_id: str | None) -> Any:
        deps = self._deps(session_id)
        graph = build_graph(deps, parallel=self.parallel)
        checkpointer = self.checkpointer
        if checkpointer is None:
            checkpointer = await self._default_checkpointer()
        return graph.compile(checkpointer=checkpointer)

    async def _default_checkpointer(self) -> Any:
        try:
            from mimir.persistence.checkpoint import async_checkpointer

            return await self._stack.enter_async_context(async_checkpointer(self.settings))
        except Exception as exc:  # noqa: BLE001 - durability is a nice-to-have, not a blocker
            log.warning("checkpointer_unavailable", error=str(exc))
            return None

    # -- running ---------------------------------------------------------

    def new_session(
        self,
        question: str,
        *,
        environment: EnvironmentContext | None = None,
        interface: str = "cli",
        session_id: str | None = None,
    ) -> InvestigationState:
        state = InvestigationState(
            user_request=question,
            interface=interface,
            environment=environment or EnvironmentContext(),
        )
        if session_id:
            state.session_id = session_id
        state.messages.append(ChatMessage(role=MessageRole.USER, content=question))
        return state

    async def run(
        self,
        question: str,
        *,
        environment: EnvironmentContext | None = None,
        interface: str = "cli",
        session_id: str | None = None,
        state: InvestigationState | None = None,
    ) -> InvestigationState:
        session = state or self.new_session(
            question, environment=environment, interface=interface, session_id=session_id
        )
        async for event in self.stream(question, state=session):
            if event.type == EventType.ERROR:
                session.error = str(event.data.get("error"))
        return session

    async def stream(
        self,
        question: str,
        *,
        environment: EnvironmentContext | None = None,
        interface: str = "cli",
        session_id: str | None = None,
        state: InvestigationState | None = None,
    ) -> AsyncIterator[RunEvent]:
        session = state or self.new_session(
            question, environment=environment, interface=interface, session_id=session_id
        )
        compiled = await self._compile(session.session_id)
        config = {
            "configurable": {"thread_id": session.session_id},
            "recursion_limit": self.settings.graph.recursion_limit,
        }

        with correlation_context(session_id=session.session_id):
            yield RunEvent(
                EventType.STARTED,
                {"session_id": session.session_id, "question": question},
            )

            # Buffered per stream, not per runner: two concurrent investigations
            # must not see each other's approval prompts.
            approval_events: list[RunEvent] = []
            unsubscribe = self.approvals.add_listener(
                _approval_relay(approval_events, session.session_id)
            )
            graph_state: GraphState = initial_state(session)
            try:
                async for chunk in compiled.astream(
                    graph_state, config=config, stream_mode="updates"
                ):
                    for node_name, update in chunk.items():
                        while approval_events:
                            yield approval_events.pop(0)
                        if not isinstance(update, dict):
                            continue
                        graph_state = _apply(graph_state, update)
                        for event in _events_for(node_name, update, graph_state):
                            yield event
            except Exception as exc:
                log.exception("investigation_failed", session_id=session.session_id)
                session.error = f"{type(exc).__name__}: {exc}"
                session.completed_at = time.time()
                if session.final_answer is None:
                    session.final_answer = FinalAnswer(
                        answer=f"The investigation failed: {session.error}",
                        confidence=0.0,
                    )
                yield RunEvent(EventType.ERROR, {"error": session.error})
            finally:
                unsubscribe()
                while approval_events:
                    yield approval_events.pop(0)

            merged = merge_into_session(graph_state)
            session.__dict__.update(merged.__dict__)
            self._persist(session)
            if session.final_answer:
                session.messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=session.final_answer.answer,
                        metadata={"confidence": session.final_confidence},
                    )
                )
                yield RunEvent(
                    EventType.ANSWER,
                    {
                        "answer": session.final_answer.model_dump(),
                        "confidence": session.final_confidence,
                    },
                )
            yield RunEvent(
                EventType.DONE,
                {
                    "session_id": session.session_id,
                    "duration_s": round(session.duration_s, 2),
                    "evidence": len(session.evidence),
                    "confidence": session.final_confidence,
                },
            )

    def _persist(self, session: InvestigationState) -> None:
        """Write the finished investigation to the audit store.

        Persistence failures are logged and swallowed: losing the ability to
        resume later is bad, but discarding an answer the operator is waiting on
        because a database write failed is worse.
        """
        try:
            from mimir.persistence.repositories import save_state

            save_state(session)
        except Exception as exc:  # noqa: BLE001
            log.warning("session_persist_failed", session_id=session.session_id, error=str(exc))

    # -- resume ----------------------------------------------------------

    async def resume(self, session_id: str) -> InvestigationState | None:
        """Continue a checkpointed run (ADR 14.1 ``mimir session resume``)."""
        compiled = await self._compile(session_id)
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await compiled.aget_state(config)
        if snapshot is None or not snapshot.values:
            return None
        graph_state: GraphState = snapshot.values
        async for chunk in compiled.astream(None, config=config, stream_mode="updates"):
            for update in chunk.values():
                if isinstance(update, dict):
                    graph_state = _apply(graph_state, update)
        return merge_into_session(graph_state)

    async def aclose(self) -> None:
        await self.router.close()
        await self._stack.aclose()


def _apply(state: GraphState, update: dict[str, Any]) -> GraphState:
    """Mirror LangGraph's channel updates onto our local copy for streaming.

    LangGraph owns the authoritative merge; this local copy exists purely so the
    stream can emit meaningful events without another round trip for state.
    """
    merged: GraphState = dict(state)  # type: ignore[assignment]
    for key, value in update.items():
        if key in ("reports", "evidence", "proposed_commands", "memory_proposals", "notes"):
            existing = list(merged.get(key, []))  # type: ignore[arg-type]
            existing.extend(value or [])
            merged[key] = existing  # type: ignore[literal-required]
        else:
            merged[key] = value  # type: ignore[literal-required]
    return merged


def _events_for(node: str, update: dict[str, Any], state: GraphState) -> list[RunEvent]:
    events = [RunEvent(EventType.NODE_END, {"node": node, **state_summary(state)})]
    session = state.get("session")
    if node == "coordinate" and session is not None and session.plan:
        events.append(
            RunEvent(
                EventType.PLAN,
                {
                    "task_type": session.plan.task_type.value,
                    "steps": [
                        {"specialist": s.specialist.value, "objective": s.objective}
                        for s in session.plan.steps
                    ],
                    "skills": session.plan.selected_skills,
                    "missing_context": session.plan.missing_context,
                },
            )
        )
    for report in update.get("reports", []) or []:
        events.append(
            RunEvent(
                EventType.SPECIALIST,
                {
                    "specialist": report.specialist.value,
                    "conclusion": report.conclusion,
                    "confidence": report.confidence,
                    "tool_calls": report.tool_calls,
                    "evidence": len(report.evidence),
                    "error": report.error,
                },
            )
        )
    for evidence in update.get("evidence", []) or []:
        events.append(
            RunEvent(
                EventType.EVIDENCE,
                {
                    "id": evidence.id,
                    "claim": evidence.claim,
                    "source_type": evidence.source_type.value,
                    "citations": [c.render() for c in evidence.citations],
                    "supports": evidence.supports,
                    "kind": evidence.kind.value,
                },
            )
        )
    for command in update.get("proposed_commands", []) or []:
        events.append(
            RunEvent(
                EventType.COMMAND,
                {
                    "id": command.id,
                    "display": command.display,
                    "risk": command.assessment.risk.value if command.assessment else None,
                    "preview": command.render_preview(),
                },
            )
        )
    return events


def _approval_relay(sink: list[RunEvent], session_id: str):
    async def listener(request: Any) -> None:
        # A broker is shared process-wide, so filter to this stream's session.
        if request.session_id and request.session_id != session_id:
            return
        sink.append(
            RunEvent(
                EventType.APPROVAL,
                {
                    "approval_id": request.id,
                    "command": request.command.display,
                    "risk": request.assessment.risk.value,
                    "prompt": request.prompt,
                },
            )
        )

    return listener


def _load_skills() -> Any:
    try:
        from mimir.skills.registry import get_skill_registry

        return get_skill_registry()
    except Exception as exc:  # noqa: BLE001 - skills are optional at runtime
        log.warning("skill_registry_unavailable", error=str(exc))
        return None


_runner: InvestigationRunner | None = None


def get_runner(settings: Settings | None = None) -> InvestigationRunner:
    global _runner
    if _runner is None:
        _runner = InvestigationRunner(settings=settings, registry=REGISTRY)
    return _runner


def reset_runner() -> None:
    global _runner
    _runner = None
