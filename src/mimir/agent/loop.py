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
    # read
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
"""The coding surface, named rather than derived from a capability.

Twelve tools, against the seventy six registered.

insert_worktree_lines was added and two lookup tools removed to pay for it,
because adherence falls with schema volume and the surface must not grow. Both
edit failures in the first real coding runs were about reproducing existing
text, first its line-number gutter and then its indentation; inserting at a
line number needs neither. inspect_git_history and lsp_symbols went, being the
two whose questions read_file_range and search_repository already answer. The council was offered
thirty five at one point and its prompts more than doubled.

The cost argument for keeping this small turned out to be wrong and is worth
recording as wrong: the runtime caches the prefix, so the schema is paid once
per conversation rather than on every step. Measured at eighteen times cheaper
after the first step.

The reason that survives measurement is adherence, not cost. Past a schema
volume this model stops emitting tool calls at all and writes them into its
prose instead, and unneeded tools still invite a worse choice.
"""

_BOUND = ("task", "repo")
"""Arguments the loop supplies and the model never sees.

Every coding tool takes the worktree it operates on. Leaving that to the model
means it is wrong occasionally, and a wrong task name is not a harmless error:
it is an edit to a different worktree. Binding it removes the class."""

def _is_worktree_tool(fields: set[str]) -> bool:
    """Whether ``repo`` means the source checkout rather than the worktree.

    Derived from the tool's own arguments rather than from a list. A tool that
    takes both a task and a repo derives the worktree path from the pair, so it
    needs the original checkout; anything else reads a tree, and the tree it
    must read is the worktree.

    This was a hand-maintained frozenset until a new worktree tool was added
    and not put in it. Every call it made resolved against the worktree as if
    that were the source repo, so it reported that the worktree did not exist
    while every other tool was working on it happily.
    """
    return {"task", "repo"} <= fields

_TOOL_TEXT = re.compile(
    r"(<function\s*=|<tool_call>|</function>|<\|tool\|>|"
    r'"(?:name|tool_name)"\s*:\s*"[a-z_]+"\s*,\s*"(?:arguments|parameters)")',
    re.IGNORECASE,
)
"""Text that is an attempted tool call rather than an answer.

Local models drop out of the tool-call channel and write the call into their
prose instead, complete with closing tags. The loop used to see a turn with no
tool calls and conclude the work was finished, so the run ended after one step
having done nothing, and reported success. Detecting the shape is deterministic
and the correction costs one extra step."""

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

    grounding: Any = None
    """Result of the deterministic name check over the final answer."""

    corrections: int = 0
    """Times the model wrote a tool call as prose and was told to try again."""

    repeats: int = 0
    """Calls that repeated one already made this turn, exactly."""


class AgentLoop:
    """Call the model, run what it asks for, feed the result back, repeat.

    The surface and the argument binding are parameters, because the loop is
    the same whether the work is editing a repository or reading a cluster.
    What differs is which tools exist and which of their arguments the operator
    already decided.
    """

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
    ) -> None:
        self.router = router
        self.registry = registry
        self.ctx = tool_context
        self.settings = settings or get_settings()
        self.max_steps = max_steps
        self.task_class = task_class
        self.constrained = constrained
        """Decode against a schema instead of trusting the tool-call channel."""

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
        """Calls already made this turn, so a repeat is recognisable."""

    # -- binding ---------------------------------------------------------

    def hidden(self) -> tuple[str, ...]:
        """Arguments removed from the schemas because the loop supplies them."""
        return ()

    def bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

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
        """Tool schemas with the bound arguments removed.

        Removed from ``required`` as well. A schema that demands a field the
        model is told never to send produces a model that sends it anyway, or
        one that refuses to call the tool at all.
        """
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
        """Work on ``instruction`` until the model stops calling tools.

        The conversation persists on the instance, so a follow-up turn keeps
        every file already read. That is the whole reason this is a session and
        not a command: re-reading the same four files for a two line follow-up
        is most of what makes a local model feel unusable.
        """
        # The hint goes in the system message, which is rebuilt rather than
        # appended to, so it never grows across turns.
        #
        # It took three attempts to land here and the reason is worth keeping.
        # Put before the instruction, 162 characters of preamble made
        # qwen3-coder:30b stop emitting tool calls entirely and write them into
        # its prose instead, at temperature zero, reproducibly. Moved after the
        # instruction it worked in a six-way probe, and then failed in the real
        # loop with a 75 character hint that differed only in wording. Length
        # was never the whole story and neither was position: the user turn is
        # simply not a stable place to put anything but the request. Every
        # system-prompt variant in that probe worked, long and short alike.
        #
        # The general lesson, which cost most of an afternoon: with a local
        # model, added prompt text can cost protocol adherence rather than just
        # tokens, and it fails silently, because a turn with no tool calls
        # looks exactly like a turn that finished.
        hint = self._hint(instruction)
        base = self.system_for(instruction)
        self.messages[0] = LLMMessage.system(f"{base}\n\n{hint}" if hint else base)
        self.messages.append(LLMMessage.user(instruction))
        self.outcome = TurnOutcome()
        self.instruction = instruction
        self.observed = []
        self.seen: dict[str, str] = {}
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
            # progress, and under a constrained decoder it cannot wander into
            # prose to signal that. Three identical calls end the turn rather
            # than burning the step budget on the same answer.
            if repeated and repeated == len(calls):
                self.outcome.repeats += 1
                if self.outcome.repeats >= 3:
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
        self.seen[signature] = result.summary or ""

        self.outcome.tool_calls += 1
        if call.name in ("edit_worktree_file", "write_worktree_file") and result.ok:
            self.outcome.files_changed.add(str(call.arguments.get("path", "")))
        if call.name == "run_worktree_tests" and result.ok:
            self.outcome.tests_run += 1

        rendered = self._render(result)
        if repeat:
            rendered += (
                "\n[this is the same call you already made, with the same result. "
                "Do something different, or finish.]"
            )
        # The model sees the trimmed render; the grounding check sees
        # everything the tool returned. Checking an answer against a truncated
        # copy of its own evidence flags what was quoted from the part that got
        # cut, and a check that cries wolf is one people switch off.
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
        """One decoded step whose shape the runtime guarantees.

        Nothing is recorded here because ModelRouter.constrained records its own
        invocation, which keeps every path through this loop counted the same
        way.
        """
        from mimir.agent.constrained import build_schema, parse_step
        from mimir.llm.base import ModelError

        schema = build_schema(self.specs, hidden=self.hidden())
        try:
            content = await self.router.constrained(
                self.messages,
                schema,
                task_class=self.task_class,
                session_id=self.ctx.session_id,
                purpose=f"{self.label}:{self.task_class}",
                tool_calls_before=self.outcome.tool_calls,
            )
        except ModelError as exc:
            return [], [], exc.message

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
        """Associate the operator's words with names this turn actually saw.

        Only names that were observed, so a turn that resolved nothing teaches
        nothing. Learning from the model's own text instead would record the
        invented names alongside the real ones.
        """
        if self.glossary is None:
            return
        from mimir.verify.grounding import identifiers

        names = identifiers("\n".join(self.observed))
        try:
            self.glossary.learn(self.instruction, names, scope=self.label)
        except Exception:  # noqa: BLE001
            log.debug("glossary_learn_failed")

    def _grounding(self, answer: str) -> Any:
        """Check the answer's identifiers against what was actually read.

        Runs on every turn rather than on request. A check you have to ask for
        is a check that is not running when it matters, and the failure it
        catches - a plausible list of names that were never observed - is
        invisible to the person reading the answer.
        """
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
    """A one-line argument display for the terminal.

    Long values (a file's new contents, a replacement span) are summarised
    rather than printed: the diff is shown separately and printing it twice
    pushes everything else off the screen.
    """
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
