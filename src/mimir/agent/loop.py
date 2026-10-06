"""The coding loop."""

from __future__ import annotations

import json
import re
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
    # read; the map first, because a lookup beats a search when the name is known
    "repository_map",
    "search_repository",
    "read_file_range",
    "locate_tests",
    # resolve, rather than guess
    "lsp_definition",
    "lsp_references",
    "lsp_diagnostics",
    # change
    "insert_worktree_lines",
    "edit_worktree_file",
    "write_worktree_file",
    # verify
    "run_worktree_tests",
    "diff_task_worktree",
    # what was already learned about this code
    "search_memory",
)
"""The coding surface, named rather than derived from a capability."""

_BOUND = ("task", "repo")
"""Arguments the loop supplies and the model never sees."""

def _is_worktree_tool(fields: set[str]) -> bool:
    """Whether ``repo`` means the source checkout rather than the worktree."""
    return {"task", "repo"} <= fields

_TOOL_TEXT = re.compile(
    r"(<function\s*=|<tool_call>|</function>|<\|tool\|>|"
    r'"(?:name|tool_name)"\s*:\s*"[a-z_]+"\s*,\s*"(?:arguments|parameters)")',
    re.IGNORECASE,
)
"""Text that is an attempted tool call rather than an answer."""

MAX_RESULT_CHARS = 6000
"""How much of a tool result goes back into context."""


@dataclass
class TurnOutcome:
    steps: int = 0
    tool_calls: int = 0
    files_changed: set[str] = field(default_factory=set)
    tests_run: int = 0
    stopped: str = "done"
    """done, max_steps, error or interrupted."""

    grounding: Any = None
    """Result of the deterministic name check over the final answer."""

    corrections: int = 0
    """Times the model wrote a tool call as prose and was told to try again."""

    repeats: int = 0
    """Calls that repeated one already made this turn, exactly."""


IDEMPOTENT_TOOLS: frozenset[str] = frozenset({
    "read_file_range", "search_repository", "repository_map", "locate_tests",
    "lsp_definition", "lsp_references", "lsp_diagnostics", "search_memory",
    "find_workloads", "list_workloads", "get_current_context", "list_namespaces",
    "describe_resource", "summarise_pod_health", "get_events", "get_logs", "get_resource_usage",
    "get_rollout_status", "trace_feature", "trace_symbol",
})
"""Tools whose result cannot change within one turn; repeating one is never progress."""


class AgentLoop:
    """Call the model, run what it asks for, feed the result back, repeat."""

    label = "agent"

    def __init__(
        self,
        *,
        router: ModelRouter,
        registry: ToolRegistry,
        tool_context: ToolContext,
        tools: Sequence[str],
        system: str,
        settings: Settings | None = None,
        max_steps: int = 20,
        task_class: str = "deep_investigation",
        constrained: bool = True,
        temperature: float = 0.0,
    ) -> None:
        self.router = router
        self.registry = registry
        self.ctx = tool_context
        self.settings = settings or get_settings()
        self.max_steps = max_steps
        self.task_class = task_class
        self.constrained = constrained
        """Decode against a schema instead of trusting the tool-call channel."""

        self.temperature = temperature
        """Zero for a single run."""

        self.specs = [s for s in (registry.get(n) for n in tools) if s is not None]
        self.system = system
        self.messages: list[LLMMessage] = [LLMMessage.system(system)]
        self.outcome = TurnOutcome()
        self.instruction = ""
        self.glossary: Any = None
        """Set by the caller. Absent is fine; the loop just works harder."""

        self.observed: list[str] = []
        """Every tool result of this turn, as the ground truth for grounding."""

        self.seen: dict[str, str] = {}
        self.seen_results: set[tuple[str, str]] = set()
        """Calls already made this turn, so a repeat is recognisable."""

        self.failures: dict[str, int] = {}
        """How many times each tool has failed this turn."""

    # -- binding ---------------------------------------------------------

    def hidden(self) -> tuple[str, ...]:
        """Arguments removed from the schemas because the loop supplies them."""
        return ()

    def bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

    def note_success(self, name: str, result: ToolResult) -> None:
        """A tool succeeded. Subclasses use this to learn from what it found."""

    def specs_now(self) -> list[Any]:
        """Which tools are offered at this point in the turn."""
        return self.specs

    def note_instruction(self, instruction: str) -> None:
        """Called at the start of every turn, for scope the operator stated."""

    def system_for(self, instruction: str) -> str:
        """The system message for this turn. Rebuilt, never appended to."""
        return self.system

    def _fields(self, name: str) -> set[str]:
        spec = self.registry.get(name)
        return set(spec.input_model.model_fields) if spec is not None else set()

    # -- tool surface ----------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas with the bound arguments removed."""
        hidden = self.hidden()
        out = []
        for spec in self.specs:
            schema = spec.openai_schema()
            parameters = schema["function"]["parameters"]
            properties = parameters.get("properties") or {}
            for name in hidden:
                properties.pop(name, None)
            required = parameters.get("required")
            if required:
                parameters["required"] = [r for r in required if r not in hidden]
            out.append(schema)
        return out

    # -- the loop --------------------------------------------------------

    async def run(self, instruction: str) -> AsyncIterator[AgentEvent]:
        """Work on ``instruction`` until the model stops calling tools."""
        # The hint goes in the system message, which is rebuilt rather than
        hint = self._hint(instruction)
        base = self.system_for(instruction)
        self.messages[0] = LLMMessage.system(f"{base}\n\n{hint}" if hint else base)
        self.messages.append(LLMMessage.user(instruction))
        self.outcome = TurnOutcome()
        self.instruction = instruction
        self.observed = []
        self.seen: dict[str, str] = {}
        self.seen_results: set[tuple[str, str]] = set()
        self.failures = {}
        self.note_instruction(instruction)
        options = GenerationOptions(tools=self.schemas(), temperature=0.0)
        corrections = 0

        for step in range(1, self.max_steps + 1):
            self.outcome.steps = step
            yield AgentEvent(type=AgentEventType.STEP, step=step)

            text_parts: list[str] = []
            calls: list[ToolCall] = []
            failed = ""

            model = self.router.for_task(self.task_class)
            started = time.time()
            if self.constrained:
                text_parts, calls, failed = await self._constrained_step(step)
                if text_parts:
                    yield AgentEvent(
                        type=AgentEventType.TEXT, step=step, text=text_parts[0]
                    )
            else:
                async for chunk in model.stream(self.messages, options):
                    if chunk.type is ChunkType.CONTENT:
                        text_parts.append(chunk.text)
                        yield AgentEvent(
                            type=AgentEventType.TEXT, step=step, text=chunk.text
                        )
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
                answer = "".join(text_parts)
                if _TOOL_TEXT.search(answer) and corrections < 2:
                    corrections += 1
                    self.outcome.corrections = corrections
                    self.messages.append(
                        LLMMessage.user(
                            "That was written as text, so nothing ran. Make the "
                            "call through the tool interface instead of writing "
                            "it in your reply."
                        )
                    )
                    continue

                self.outcome.stopped = "done"
                self.outcome.grounding = self._grounding(answer)
                self._learn()
                yield AgentEvent(type=AgentEventType.DONE, step=step, text=answer)
                return

            repeated = 0
            for call in calls:
                async for event in self._dispatch(call, step):
                    yield event
                    if event.type is AgentEventType.TOOL_END and event.extra_repeat:
                        repeated += 1

            # A model that runs the same call again has stopped making
            if repeated and repeated == len(calls):
                self.outcome.repeats += 1
                # Re-reading cannot produce anything new; one all-repeat step is enough.
                limit = 1 if all(c.name in IDEMPOTENT_TOOLS for c in calls) else 3
                if self.outcome.repeats >= limit:
                    self.outcome.stopped = "repeating"
                    answer = self.observed[-1] if self.observed else ""
                    self.outcome.grounding = self._grounding(answer)
                    self._learn()
                    yield AgentEvent(
                        type=AgentEventType.DONE, step=step,
                        text="Stopped: the same call was repeated with the same result.",
                    )
                    return

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
        arguments = self.bind(call.name, call.arguments)
        signature = f"{call.name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        repeat = signature in self.seen
        result = await self.registry.invoke(call.name, arguments, self.ctx)
        elapsed = time.time() - started
        # Same tool, same result is a repeat even when the arguments were spelled differently.
        read_only = call.name in IDEMPOTENT_TOOLS
        result_key = (call.name, result.summary or "")
        if read_only and result.ok and result_key in self.seen_results:
            repeat = True
        self.seen[signature] = result.summary or ""
        if result.ok:
            self.seen_results.add(result_key)

        self.outcome.tool_calls += 1
        if result.ok:
            self.note_success(call.name, result)
        else:
            self.failures[call.name] = self.failures.get(call.name, 0) + 1
        if call.name in ("edit_worktree_file", "write_worktree_file") and result.ok:
            self.outcome.files_changed.add(str(call.arguments.get("path", "")))
        if call.name == "run_worktree_tests" and result.ok:
            self.outcome.tests_run += 1

        if repeat and read_only:
            # Do not feed the same content twice; it is already in the conversation.
            rendered = (f"[{call.name} returned exactly what it returned before: {result.summary}. "
                        "That content is already above. Answer from it now, or read something else.]")
        else:
            rendered = self._render(result)
            if repeat:
                rendered += (
                    "\n[this is the same call you already made, with the same result. "
                    "Do something different, or finish.]"
                )
        # The model sees the trimmed render; the grounding check sees
        self.observed.append(f"{rendered}\n{_all_text(result)}")
        self.messages.append(
            LLMMessage.tool_result(call.id, rendered, name=call.name)
        )
        event = AgentEvent(
            type=AgentEventType.TOOL_END,
            step=step,
            tool=call.name,
            arguments=call.arguments,
            result=result,
            elapsed_s=elapsed,
        )
        event.extra_repeat = repeat
        yield event

    async def _constrained_step(self, step: int) -> tuple[list[str], list[ToolCall], str]:
        """One decoded step whose shape the runtime guarantees."""
        from mimir.agent.constrained import build_schema, parse_step
        from mimir.llm.base import ModelError

        schema = build_schema(self.specs_now(), hidden=self.hidden())
        budget = 0
        for attempt in range(2):
            try:
                content, reason = await self.router.constrained(
                    self.messages,
                    schema,
                    task_class=self.task_class,
                    session_id=self.ctx.session_id,
                    purpose=f"{self.label}:{self.task_class}",
                    temperature=self.temperature,
                    max_tokens=budget,
                    tool_calls_before=self.outcome.tool_calls,
                )
            except ModelError as exc:
                return [], [], exc.message

            # A call cut off mid-argument is not an answer.
            if reason == "length" and attempt == 0:
                budget = 16_384
                log.info("constrained_output_truncated", step=step, retry_budget=budget)
                continue
            if reason == "length":
                return [], [], (
                    "the model's output was cut off mid-call even at the larger "
                    "budget; the change it was making is too large for one step"
                )
            break

        decoded = parse_step(content)
        call = decoded.as_tool_call()
        return ([decoded.say] if decoded.say else []), ([call] if call else []), ""

    def _hint(self, instruction: str) -> str:
        if self.glossary is None:
            return ""
        try:
            return self.glossary.hint(instruction)
        except Exception:  # noqa: BLE001 - a hint must never fail a turn
            return ""

    def _learn(self) -> None:
        """Associate the operator's words with names this turn actually saw."""
        if self.glossary is None:
            return
        from mimir.verify.grounding import identifiers

        names = identifiers("\n".join(self.observed))
        try:
            self.glossary.learn(self.instruction, names, scope=self.label)
        except Exception:  # noqa: BLE001
            log.debug("glossary_learn_failed")

    def _grounding(self, answer: str) -> Any:
        """Check the answer's identifiers against what was actually read."""
        from mimir.verify.grounding import check

        return check(answer, "\n".join(self.observed), asked=self.instruction)

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
        """Log the invocation the way ``ModelRouter.chat`` does."""
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
                purpose=f"{self.label}:{self.task_class}",
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


class CodingAgent(AgentLoop):
    """One conversation, bound to one task worktree."""

    label = "coding"

    def __init__(
        self,
        *,
        task: str,
        repo: str,
        view: str,
        worktree_root: Path,
        tools: Sequence[str] = CODING_TOOLS,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.repo = repo
        self.view = view
        """The name the worktree is registered under, for the tools that read it."""

        self.root = Path(worktree_root)
        super().__init__(tools=tools, system=system_prompt(str(self.root)), **kwargs)

    def hidden(self) -> tuple[str, ...]:
        return _BOUND

    def specs_now(self) -> list[Any]:
        """Withdraw a tool that has failed the same way twice."""
        if self.failures.get("edit_worktree_file", 0) < 2:
            return self.specs
        return [s for s in self.specs if s.name != "edit_worktree_file"]

    def bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fields = self._fields(name)
        bound = dict(arguments)
        if "task" in fields:
            bound["task"] = self.task
        if "repo" in fields:
            bound["repo"] = self.repo if _is_worktree_tool(fields) else self.view
        return bound


def _all_text(result: ToolResult) -> str:
    """Every string a tool result carries, for grounding only."""
    parts: list[str] = [result.summary or ""]

    def walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(result.data)
    for evidence in result.evidence:
        parts.append(getattr(evidence, "excerpt", "") or "")
    return "\n".join(p for p in parts if p)


def format_arguments(tool: str, arguments: dict[str, Any]) -> str:
    """A one-line argument display for the terminal."""
    bits = []
    for key, value in arguments.items():
        if key in _BOUND or value in (None, "", [], {}):
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


__all__ = [
    "CODING_TOOLS",
    "AgentLoop",
    "CodingAgent",
    "TurnOutcome",
    "format_arguments",
]
