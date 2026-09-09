"""What a coding turn emits.

The loop yields events rather than printing, for the same reason the graph
runner does: the terminal renderer, a future side panel, the session log and a
test all want the same sequence and none of them should have to parse text to
get it.

:class:`TimelineEntry` is the reason this module is separate from the loop. The
question worth answering later is not "what did the model say" but "what did it
look for, and which pieces did it keep". That is derivable from the tool call
and its result without a model, so it is derived here once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mimir.tools.base import ToolResult


class AgentEventType(StrEnum):
    STEP = "step"
    """A new model turn begins. Carries the step number."""

    TEXT = "text"
    """A fragment of assistant text, as it streams."""

    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ERROR = "error"
    DONE = "done"


@dataclass(slots=True)
class AgentEvent:
    type: AgentEventType
    step: int = 0
    text: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def timeline(self) -> TimelineEntry | None:
        if self.type is not AgentEventType.TOOL_END:
            return None
        return TimelineEntry.of(self.step, self.tool, self.arguments, self.result)


# The argument that says what a call was actually looking for. Falling back to
# a serialised argument dict makes the timeline unreadable, so a tool with no
# entry here contributes its name alone.
_SUBJECT: dict[str, tuple[str, ...]] = {
    "search_repository": ("query",),
    "read_file_range": ("path",),
    "locate_tests": ("subject",),
    "inspect_git_history": ("path", "query"),
    "lsp_definition": ("symbol",),
    "lsp_references": ("symbol",),
    "lsp_hover": ("symbol",),
    "lsp_symbols": ("path",),
    "lsp_diagnostics": ("path",),
    "edit_worktree_file": ("path",),
    "write_worktree_file": ("path",),
    "run_worktree_tests": ("command",),
    "diff_task_worktree": ("task",),
}

_CONSTRAINT: dict[str, tuple[str, ...]] = {
    "search_repository": ("globs",),
    "read_file_range": ("start_line", "end_line"),
    "locate_tests": ("languages",),
    "lsp_definition": ("path",),
    "lsp_references": ("path",),
}

_VERB: dict[str, str] = {
    "search_repository": "searched for",
    "read_file_range": "read",
    "locate_tests": "located tests for",
    "inspect_git_history": "checked the history of",
    "lsp_definition": "resolved",
    "lsp_references": "found callers of",
    "lsp_hover": "checked the type of",
    "lsp_symbols": "listed the symbols in",
    "lsp_diagnostics": "checked for errors in",
    "edit_worktree_file": "edited",
    "write_worktree_file": "wrote",
    "run_worktree_tests": "ran",
    "diff_task_worktree": "reviewed",
}


@dataclass(slots=True)
class TimelineEntry:
    """One step of how an answer was reached, derived not narrated.

    A model asked to explain its own reasoning produces a plausible account of
    reasoning, which is not the same artefact and is not checkable. This is
    built from the calls that actually happened.
    """

    step: int
    tool: str
    verb: str
    subject: str
    found: str
    ok: bool
    kept: bool
    """Whether the call produced evidence. A search returning nothing is part of
    the story, and hiding it makes the path look straighter than it was."""

    constraint: str = ""
    """What narrowed the call: a glob, a line range, a language filter.

    A search shown as "searched for X -> 0 matches" reads as a broken tool. The
    same search shown with the glob that excluded everything reads as what it
    was. Dropping the qualifier turns a model mistake into an apparent defect,
    and someone then goes looking for the defect."""

    @classmethod
    def of(
        cls,
        step: int,
        tool: str,
        arguments: dict[str, Any],
        result: ToolResult | None,
    ) -> TimelineEntry:
        subject = ""
        for key in _SUBJECT.get(tool, ()):
            value = arguments.get(key)
            if value:
                subject = str(value)
                break
        constraint = " ".join(
            f"{key}={arguments[key]}"
            for key in _CONSTRAINT.get(tool, ())
            if arguments.get(key) not in (None, "", [], {})
        )
        found = ""
        if result is not None:
            found = (result.summary or result.error or "").strip().splitlines()[:1]
            found = found[0] if found else ""
        return cls(
            step=step,
            tool=tool,
            verb=_VERB.get(tool, tool.replace("_", " ")),
            subject=subject,
            found=found[:160],
            constraint=constraint[:80],
            ok=bool(result is None or result.ok),
            kept=bool(result is not None and result.evidence),
        )


__all__ = ["AgentEvent", "AgentEventType", "TimelineEntry"]
