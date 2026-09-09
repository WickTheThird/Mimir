"""The coding loop.

MIMIR's investigation graph is single pass by design (ADR-003): plan, fan out
to a council, verify, synthesise, stop. That shape is right for answering a
question and wrong for changing code, because changing code is iterative. You
read, you edit, you run the tests, and what the tests say determines the next
edit. There is no way to know the third step at planning time.

So this is a second mode rather than a back edge in the graph. The graph keeps
its property that safety is decided before any model runs; this loop keeps that
property too, by dispatching every call through the same registry, the same
risk classifier and the same approval broker. What it does not keep is the
council, the verification pass and the synthesis node, because paying for a
ten specialist fan-out to add a null check is exactly the waste the triage
module was written to stop.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.agent.events import AgentEvent, AgentEventType
from mimir.agent.prompt import system_prompt
from mimir.config import Settings, get_settings
from mimir.llm.base import ChunkType, GenerationOptions, LLMMessage, ToolCall
from mimir.llm.router import ModelRouter
from mimir.logging import get_logger
from mimir.tools.base import ToolContext, ToolRegistry, ToolResult

log = get_logger(__name__)

CODING_TOOLS: tuple[str, ...] = (
    # read
    "search_repository",
    "read_file_range",
    "locate_tests",
    "inspect_git_history",
    # resolve, rather than guess
    "lsp_definition",
    "lsp_references",
    "lsp_symbols",
    "lsp_diagnostics",
    # change
    "edit_worktree_file",
    "write_worktree_file",
    # verify
    "run_worktree_tests",
    "diff_task_worktree",
)
"""The coding surface, named rather than derived from a capability.

Twelve tools, against the seventy six registered. The council was offered
thirty five at one point and its prompts more than doubled; the tools it did
not need still cost their schema on every call and still invited a worse
choice. A loop that runs many short steps pays that on every step.
"""

_BOUND = ("task", "repo")
"""Arguments the loop supplies and the model never sees.

Every coding tool takes the worktree it operates on. Leaving that to the model
means it is wrong occasionally, and a wrong task name is not a harmless error:
it is an edit to a different worktree. Binding it removes the class."""

_WORKTREE_TOOLS = frozenset({
    "create_task_worktree",
    "diff_task_worktree",
    "discard_task_worktree",
    "edit_worktree_file",
    "run_worktree_tests",
    "write_worktree_file",
})
"""Tools whose ``repo`` means the source checkout, not the worktree.

These take the repository and the task and derive the worktree path from the
pair, so they need the original. Every other tool reads a tree, and the tree it
must read is the worktree: binding all of them to the source checkout would
make a file read back without the edit that was just made to it, which is the
kind of defect that looks like the model hallucinating."""

MAX_RESULT_CHARS = 6000
"""How much of a tool result goes back into context. A repository search can
return more than the context window."""


@dataclass
class TurnOutcome:
    steps: int = 0
    tool_calls: int = 0
    files_changed: set[str] = field(default_factory=set)
    tests_run: int = 0
    stopped: str = "done"
    """done, max_steps, error or interrupted."""


class CodingAgent:
    """One conversation, bound to one task worktree."""

    def __init__(
        self,
        *,
        router: ModelRouter,
        registry: ToolRegistry,
        tool_context: ToolContext,
        task: str,
        repo: str,
        view: str,
        worktree_root: Path,
        settings: Settings | None = None,
        tools: Sequence[str] = CODING_TOOLS,
        max_steps: int = 20,
        task_class: str = "deep_investigation",
    ) -> None:
        self.router = router
        self.registry = registry
        self.ctx = tool_context
        self.task = task
        self.repo = repo
        self.view = view
        """The name the worktree is registered under, for the tools that read it."""

        self.root = Path(worktree_root)
        self.settings = settings or get_settings()
        self.max_steps = max_steps
        self.task_class = task_class
        self.specs = [s for s in (registry.get(n) for n in tools) if s is not None]
        self.messages: list[LLMMessage] = [
            LLMMessage.system(system_prompt(str(self.root)))
        ]
        self.outcome = TurnOutcome()

    # -- tool surface ----------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas with the bound arguments removed.

        Removed from ``required`` as well. A schema that demands a field the
        model is told never to send produces a model that sends it anyway, or
        one that refuses to call the tool at all.
        """
        out = []
        for spec in self.specs:
            schema = spec.openai_schema()
            parameters = schema["function"]["parameters"]
            properties = parameters.get("properties") or {}
            for name in _BOUND:
                properties.pop(name, None)
            required = parameters.get("required")
            if required:
                parameters["required"] = [r for r in required if r not in _BOUND]
            out.append(schema)
        return out

    def _bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self.registry.get(name)
        fields = set(spec.input_model.model_fields) if spec is not None else set()
        bound = dict(arguments)
        if "task" in fields:
            bound["task"] = self.task
        if "repo" in fields:
            bound["repo"] = self.repo if name in _WORKTREE_TOOLS else self.view
        return bound

    # -- the loop --------------------------------------------------------

    async def run(self, instruction: str) -> AsyncIterator[AgentEvent]:
        """Work on ``instruction`` until the model stops calling tools.

        The conversation persists on the instance, so a follow-up turn keeps
        every file already read. That is the whole reason this is a session and
        not a command: re-reading the same four files for a two line follow-up
        is most of what makes a local model feel unusable.
        """
        self.messages.append(LLMMessage.user(instruction))
        self.outcome = TurnOutcome()
        options = GenerationOptions(tools=self.schemas(), temperature=0.0)

        for step in range(1, self.max_steps + 1):
            self.outcome.steps = step
            yield AgentEvent(type=AgentEventType.STEP, step=step)

            text_parts: list[str] = []
            calls: list[ToolCall] = []
            failed = ""

            model = self.router.for_task(self.task_class)
            started = time.time()
            async for chunk in model.stream(self.messages, options):
                if chunk.type is ChunkType.CONTENT:
                    text_parts.append(chunk.text)
                    yield AgentEvent(type=AgentEventType.TEXT, step=step, text=chunk.text)
                elif chunk.type is ChunkType.TOOL_CALL and chunk.tool_call is not None:
                    calls.append(chunk.tool_call)
                elif chunk.type is ChunkType.ERROR:
                    failed = chunk.error or "model error"

            self._record(model, started, text_parts, calls, failed, step)

            if failed:
                self.outcome.stopped = "error"
                yield AgentEvent(type=AgentEventType.ERROR, step=step, error=failed)
                return

            self.messages.append(LLMMessage.assistant("".join(text_parts), calls))

            if not calls:
                self.outcome.stopped = "done"
                yield AgentEvent(
                    type=AgentEventType.DONE, step=step, text="".join(text_parts)
                )
                return

            for call in calls:
                async for event in self._dispatch(call, step):
                    yield event

        self.outcome.stopped = "max_steps"
        yield AgentEvent(
            type=AgentEventType.ERROR,
            step=self.max_steps,
            error=(
                f"stopped after {self.max_steps} steps without finishing. "
                "Say what to do next, or /worktree diff to see what changed so far."
            ),
        )

    async def _dispatch(self, call: ToolCall, step: int) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=AgentEventType.TOOL_START,
            step=step,
            tool=call.name,
            arguments=call.arguments,
        )
        started = time.time()
        result = await self.registry.invoke(
            call.name, self._bind(call.name, call.arguments), self.ctx
        )
        elapsed = time.time() - started

        self.outcome.tool_calls += 1
        if call.name in ("edit_worktree_file", "write_worktree_file") and result.ok:
            self.outcome.files_changed.add(str(call.arguments.get("path", "")))
        if call.name == "run_worktree_tests" and result.ok:
            self.outcome.tests_run += 1

        self.messages.append(
            LLMMessage.tool_result(call.id, self._render(result), name=call.name)
        )
        yield AgentEvent(
            type=AgentEventType.TOOL_END,
            step=step,
            tool=call.name,
            arguments=call.arguments,
            result=result,
            elapsed_s=elapsed,
        )

    @staticmethod
    def _render(result: ToolResult) -> str:
        text = result.render(max_chars=MAX_RESULT_CHARS)
        return text or ("ok" if result.ok else "failed")

    def _record(
        self,
        model: Any,
        started: float,
        text_parts: list[str],
        calls: list[ToolCall],
        failed: str,
        step: int,
    ) -> None:
        """Log the invocation the way ``ModelRouter.chat`` does.

        Streaming bypasses the router's own call site, and a mode that invokes
        the model without recording it breaks the invariant that model
        invocations observed equals model call records persisted. That
        invariant is the only reason the empty ``model_calls`` table was ever
        found, so a new caller does not get to opt out of it.
        """
        from mimir.llm.base import ModelCallRecord

        self.router.invocations_attempted += 1
        content = "".join(text_parts)
        self.router.call_log.append(
            ModelCallRecord(
                alias=model.alias,
                model=model.model,
                started_at=started,
                latency_s=time.time() - started,
                prompt_tokens=0,
                completion_tokens=0,
                tool_calls=len(calls),
                finish_reason="error" if failed else ("tool_calls" if calls else "stop"),
                session_id=self.ctx.session_id,
                purpose=f"coding:{self.task}",
                error=failed or None,
                attempt=0,
                runtime=getattr(getattr(model, "profile", None), "runtime", ""),
                digest=self.router.digest_for(model.alias),
                task_class=self.task_class,
                context_window=model.context_window,
                tool_calls_before=self.outcome.tool_calls,
            )
        )
        log.debug(
            "coding_step",
            step=step,
            tools=[c.name for c in calls],
            text_chars=len(content),
        )


def format_arguments(tool: str, arguments: dict[str, Any]) -> str:
    """A one-line argument display for the terminal.

    Long values (a file's new contents, a replacement span) are summarised
    rather than printed: the diff is shown separately and printing it twice
    pushes everything else off the screen.
    """
    bits = []
    for key, value in arguments.items():
        if key in _BOUND:
            continue
        if key in ("content", "old_string", "new_string"):
            lines = str(value).count("\n") + 1
            bits.append(f"{key}={lines} lines")
            continue
        rendered = value if isinstance(value, str) else json.dumps(value)
        rendered = str(rendered)
        if len(rendered) > 60:
            rendered = rendered[:57] + "..."
        bits.append(rendered if key == "path" else f"{key}={rendered}")
    return " ".join(bits)


__all__ = ["CODING_TOOLS", "CodingAgent", "TurnOutcome", "format_arguments"]
