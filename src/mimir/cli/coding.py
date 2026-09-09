"""Terminal surface for the coding loop.

The investigation view prints four one-line events and then nothing for
minutes. That is tolerable for a batch answer and not for work you are
supervising: you cannot tell a slow step from a stuck one, and by the time
output appears the decision that mattered is already made.

So this renders as it happens. Text streams, each tool call appears when it
starts rather than when it finishes, and an edit shows its diff inline.
"""

from __future__ import annotations

import asyncio
import difflib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from mimir.agent.events import AgentEvent, AgentEventType, TimelineEntry
from mimir.agent.loop import CodingAgent, format_arguments
from mimir.tools.base import ToolResult

MARK = "⏺"
INDENT = "  "

_READ = "cyan"
_WRITE = "yellow"
_RUN = "magenta"

_STYLES = {
    "edit_worktree_file": _WRITE,
    "write_worktree_file": _WRITE,
    "run_worktree_tests": _RUN,
}


def _tool_style(tool: str) -> str:
    return _STYLES.get(tool, _READ)


def render_tool_start(console: Console, tool: str, arguments: dict[str, Any]) -> None:
    console.print(
        Text.assemble(
            (INDENT, ""),
            (f"{MARK} ", _tool_style(tool)),
            (tool, f"bold {_tool_style(tool)}"),
            ("  ", ""),
            (format_arguments(tool, arguments), "dim"),
        )
    )


def render_tool_end(console: Console, event: AgentEvent) -> None:
    result = event.result
    if result is None:
        return
    if not result.ok:
        console.print(
            Text(f"{INDENT}  -> {result.error or 'failed'}", style="red")
        )
        return
    summary = (result.summary or "ok").strip().splitlines()
    console.print(Text(f"{INDENT}  -> {summary[0] if summary else 'ok'}", style="dim"))
    for line in summary[1:4]:
        console.print(Text(f"{INDENT}     {line}", style="dim"))

    if event.tool == "edit_worktree_file":
        render_edit_diff(console, result)


def render_edit_diff(console: Console, result: ToolResult) -> None:
    """Show what the edit changed, at the line numbers it changed.

    The tool result already carries both sides, so this is a display of what
    happened rather than a re-read of the file that might disagree with it.
    """
    old = str(result.data.get("old_string", ""))
    new = str(result.data.get("new_string", ""))
    start = int(result.data.get("line", 1) or 1)
    if not old and not new:
        return

    console.print()
    old_no = new_no = start
    for line in difflib.unified_diff(
        old.splitlines(), new.splitlines(), n=0, lineterm=""
    ):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("-"):
            console.print(Text(f"{INDENT}  {old_no:>5}  {line}", style="red"))
            old_no += 1
        elif line.startswith("+"):
            console.print(Text(f"{INDENT}  {new_no:>5}  {line}", style="green"))
            new_no += 1
    console.print()


def render_timeline(console: Console, entries: Sequence[TimelineEntry]) -> None:
    """How the answer was reached, as a list of what was looked for and found."""
    if not entries:
        console.print(Text("nothing was looked up yet", style="dim"))
        return
    console.print()
    for entry in entries:
        mark = "x" if not entry.ok else ("*" if entry.kept else "-")
        console.print(
            Text.assemble(
                (f" {mark} ", "red" if not entry.ok else ("cyan" if entry.kept else "dim")),
                (f"{entry.verb} ", "dim"),
                (entry.subject or entry.tool, "bold"),
                (f"  {entry.constraint}" if entry.constraint else "", "dim"),
            )
        )
        if entry.found:
            console.print(Text(f"     {entry.found}", style="dim"))
    console.print()


class CodingSession:
    """One task worktree, one conversation, one renderer."""

    def __init__(
        self,
        console: Console,
        agent: CodingAgent,
        *,
        task: str,
        repo: str,
        root: Path,
    ) -> None:
        self.console = console
        self.agent = agent
        self.task = task
        self.repo = repo
        self.root = root
        self.timeline: list[TimelineEntry] = []

    @property
    def prompt(self) -> str:
        return f"mimir({self.task})> "

    async def turn(self, instruction: str) -> None:
        """Run one instruction, rendering as it goes.

        Ctrl-C cancels the turn rather than the process. A long tool call
        already running is allowed to finish, because killing a test run
        mid-write leaves the worktree in a state nobody asked for.
        """
        console = self.console
        console.print()
        line_open = False
        try:
            async for event in self.agent.run(instruction):
                if event.type is AgentEventType.TEXT:
                    if not line_open:
                        console.print(INDENT, end="")
                        line_open = True
                    console.print(
                        Text(event.text.replace("\n", "\n" + INDENT)), end=""
                    )
                    continue

                if line_open:
                    console.print()
                    line_open = False

                if event.type is AgentEventType.TOOL_START:
                    console.print()
                    render_tool_start(console, event.tool, event.arguments)
                elif event.type is AgentEventType.TOOL_END:
                    render_tool_end(console, event)
                    entry = event.timeline
                    if entry is not None:
                        self.timeline.append(entry)
                elif event.type is AgentEventType.ERROR:
                    console.print(Text(f"{INDENT}{event.error}", style="bold red"))
                elif event.type is AgentEventType.DONE:
                    pass
        except asyncio.CancelledError:
            console.print(Text(f"\n{INDENT}stopped", style="yellow"))
            raise
        except KeyboardInterrupt:
            console.print(Text(f"\n{INDENT}stopped", style="yellow"))
        finally:
            if line_open:
                console.print()

        self._summarise()

    def _summarise(self) -> None:
        outcome = self.agent.outcome
        if outcome.stopped == "error":
            return
        bits = []
        if outcome.files_changed:
            bits.append(
                f"{len(outcome.files_changed)} file"
                f"{'s' if len(outcome.files_changed) > 1 else ''} changed"
            )
        if outcome.tests_run:
            bits.append(f"tests run {outcome.tests_run}x")
        bits.append(f"{outcome.steps} steps, {outcome.tool_calls} tool calls")
        self.console.print()
        self.console.print(Text(f"{INDENT}{', '.join(bits)}", style="dim"))
        if outcome.files_changed:
            self.console.print(
                Text(f"{INDENT}/diff to review, /why for how it got there", style="dim")
            )
        self.console.print()


def _warn_if_dirty(console: Console, root: Path) -> None:
    """Say when the new worktree is missing work that is in the checkout.

    A worktree branches from HEAD, so uncommitted files are simply not in it.
    Asked to change one of them, the model finds nothing, casts around, and
    edits the nearest plausible file instead. That happened on the first real
    run of this loop. The cause is two commands away from obvious and the
    symptom looks like the model inventing a filename, so it is said out loud
    rather than left to be rediscovered.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    changed = [line for line in out.splitlines() if line.strip()]
    if not changed:
        return
    console.print(
        Text(
            f"note: {len(changed)} uncommitted file(s) in the checkout are not in "
            "this worktree, which branches from HEAD. Commit them first if the "
            "task needs them.",
            style="yellow",
        )
    )


def start_coding_session(
    console: Console,
    runner: Any,
    task: str,
    repo: str | None = None,
) -> CodingSession:
    """Open or reopen a task worktree and bind an agent to it.

    Reopening is the common case and must not be destructive: a worktree that
    already exists is picked up with its changes intact, because the second
    thing anyone does after a coding turn is start another one.
    """
    from mimir.agent.loop import CodingAgent
    from mimir.tools.repo import get_repository_directory
    from mimir.worktree import WorktreeError, WorktreeManager

    directory = get_repository_directory(runner.settings)
    resolved = directory.resolve(repo)
    manager = WorktreeManager(runner.settings.home)

    try:
        worktree = manager.find(resolved.root, task)
        console.print(
            Text(f"reopened {worktree.branch} at {worktree.root}", style="dim")
        )
    except WorktreeError:
        worktree = manager.create(resolved.root, task)
        console.print(
            Text(f"created {worktree.branch} at {worktree.root}", style="dim")
        )
        _warn_if_dirty(console, resolved.root)

    view = f"{worktree.name}-worktree"
    directory.register_session(
        view, worktree.root, f"task worktree of {resolved.name}"
    )

    agent = CodingAgent(
        router=runner.router,
        registry=runner.registry,
        tool_context=runner.tool_context(None),
        task=worktree.name,
        repo=resolved.name,
        view=view,
        worktree_root=worktree.root,
        settings=runner.settings,
    )
    return CodingSession(
        console, agent, task=worktree.name, repo=resolved.name, root=worktree.root
    )


__all__ = [
    "CodingSession",
    "render_edit_diff",
    "render_timeline",
    "render_tool_end",
    "render_tool_start",
    "start_coding_session",
]
