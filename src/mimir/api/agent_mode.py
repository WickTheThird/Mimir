"""MIMIR as the agent behind /v1: its own tool loop answers, Warp just displays."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

ASSISTANT_TOOLS: tuple[str, ...] = (
    "repository_map", "search_repository", "read_file_range", "locate_tests",
    "find_workloads", "get_current_context", "list_workloads", "summarise_pod_health",
    "get_logs", "get_events", "describe_resource", "search_memory",
)
"""Read-only. The loop reads the estate and the repositories; it never mutates."""


def instruction_from(messages: list[Any], *, turns: int = 6) -> str:
    """The last user message as the task, with recent turns as context."""
    users = [m for m in messages if getattr(m, "role", "") == "user"]
    if not users:
        return ""
    task = (users[-1].content or "").strip()
    prior = [m for m in messages if getattr(m, "role", "") in ("user", "assistant")][:-1][-turns:]
    if not prior:
        return task
    context = "\n".join(f"{m.role}: {(m.content or '').strip()[:400]}" for m in prior if (m.content or "").strip())
    return f"Conversation so far:\n{context}\n\nNow: {task}"


def _render_args(arguments: dict[str, Any]) -> str:
    shown = {k: v for k, v in arguments.items() if k not in ("task", "repo", "view") and v not in (None, "", [])}
    return ", ".join(f"{k}={str(v)[:40]!r}" for k, v in list(shown.items())[:4])


async def run_agent(messages: list[Any], runner: Any, settings: Any) -> AsyncIterator[str]:
    """Yield progress lines and the answer as it is produced."""
    from mimir.agent.events import AgentEventType
    from mimir.agent.ops import OpsAgent

    instruction = instruction_from(messages)
    if not instruction:
        yield "No request found in the conversation."
        return
    agent = OpsAgent(
        router=runner.router,
        registry=runner.registry,
        tool_context=runner.tool_context(None),
        settings=settings,
        environment=None,
        tools=tuple(t for t in ASSISTANT_TOOLS if runner.registry.get(t) is not None),
        task_class="fast_command",
        max_steps=settings.api.facade_agent_max_steps,
    )
    budget = settings.api.facade_agent_timeout_s
    started = time.time()
    said_anything = False
    try:
        async for event in agent.run(instruction):
            if time.time() - started > budget:
                yield f"\n[MIMIR: stopped after {budget:.0f}s; what was found so far is above]\n"
                break
            if event.type is AgentEventType.TOOL_START:
                yield f"\n> {event.tool}({_render_args(event.arguments)})\n"
            elif event.type is AgentEventType.TOOL_END:
                summary = (event.result.summary if event.result is not None else "").strip().splitlines()
                head = summary[0][:160] if summary else ("ok" if event.result and event.result.ok else "failed")
                mark = "ok" if (event.result is None or event.result.ok) else "failed"
                yield f"  {mark}: {head}\n"
            elif event.type is AgentEventType.TEXT and event.text:
                said_anything = True
                yield event.text
            elif event.type is AgentEventType.ERROR:
                yield f"\n[MIMIR: {event.error}]\n"
    except Exception as exc:  # noqa: BLE001 - the client must get a reply, not a dropped stream
        log.exception("facade_agent_failed")
        yield f"\n[MIMIR: {type(exc).__name__}: {exc}]\n"
    outcome = getattr(agent, "outcome", None)
    if outcome is not None and outcome.stopped == "repeating":
        yield "\n[MIMIR: stopped repeating the same call; the thing may be named differently]\n"
    if not said_anything:
        yield "\n(no conclusion was written; the tool results above are what was found)\n"


__all__ = ["ASSISTANT_TOOLS", "instruction_from", "run_agent"]
