"""What a coding turn emits."""

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
    extra_repeat: bool = False
    """Whether this call had already been made, identically, this turn."""

    @property
    def timeline(self) -> TimelineEntry | None:
        if self.type is not AgentEventType.TOOL_END:
            return None
        return TimelineEntry.of(self.step, self.tool, self.arguments, self.result)


# The argument that says what a call was actually looking for.
_SUBJECT: dict[str, tuple[str, ...]] = {
    "find_workloads": ("name_contains",),
    "list_workloads": ("context",),
    "summarise_pod_health": ("context",),
    "get_logs": ("target",),
    "get_events": ("context",),
    "describe_resource": ("name",),
    "get_rollout_status": ("name",),
    "get_resource_usage": ("context",),
    "search_memory": ("query",),
    "find_similar_incidents": ("query",),
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
    "find_workloads": ("name_contains", "context_contains", "namespace_contains"),
    "list_workloads": ("name_contains", "selector", "namespace"),
    "summarise_pod_health": ("service", "selector", "namespace"),
    "get_logs": ("namespace", "tail", "since", "grep"),
    "get_events": ("namespace", "only_warnings"),
    "describe_resource": ("namespace",),
}

_VERB: dict[str, str] = {
    "find_workloads": "searched every namespace for",
    "list_workloads": "listed workloads in",
    "summarise_pod_health": "checked pod health in",
    "get_logs": "read logs from",
    "get_events": "read events in",
    "describe_resource": "described",
    "get_rollout_status": "checked the rollout of",
    "get_resource_usage": "measured usage in",
    "get_current_context": "checked the current context",
    "search_memory": "searched memory for",
    "find_similar_incidents": "looked for past incidents like",
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
    """One step of how an answer was reached, derived not narrated."""

    step: int
    tool: str
    verb: str
    subject: str
    found: str
    ok: bool
    kept: bool
    """Whether the call produced evidence."""

    constraint: str = ""
    """What narrowed the call: a glob, a line range, a language filter."""

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
