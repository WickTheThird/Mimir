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
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from mimir.agent.events import AgentEvent, AgentEventType, TimelineEntry
from mimir.agent.loop import format_arguments
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


def fit(text: str, width: int) -> str:
    """Clip to one line rather than letting it wrap.

    A wrapped diff line is worse than a clipped one: the continuation lands in
    the left gutter where the line numbers are, so it reads as another line of
    code. The same is true of a tool summary, which becomes two lines of which
    the second has no context.
    """
    text = text.replace("\t", "    ").rstrip()
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


class StreamWriter:
    """Writes streamed model text with a hanging indent.

    Rich wraps each ``print`` independently, and a stream arrives in fragments
    that are not lines, so letting it wrap puts the continuation of every
    sentence at column zero and destroys the gutter that separates prose from
    tool calls. Tracking the column here is the only way to wrap text that
    arrives a few characters at a time.
    """

    def __init__(self, console: Console, indent: str = INDENT) -> None:
        self.console = console
        self.indent = indent
        self.width = max(40, console.width - len(indent) - 1)
        self.column = 0
        self.blanks = 0
        self.any_output = False
        self._pending = ""

    def write(self, text: str) -> None:
        self._pending += text
        # Hold back the trailing partial word: it may still grow, and wrapping
        # on a fragment breaks words in half.
        cut = max(self._pending.rfind(" "), self._pending.rfind("\n"))
        if cut < 0:
            return
        chunk, self._pending = self._pending[: cut + 1], self._pending[cut + 1 :]
        self._emit(chunk)

    def close(self) -> None:
        if self._pending:
            self._emit(self._pending)
            self._pending = ""
        if self.column:
            self.console.print()
            self.column = 0

    def _emit(self, chunk: str) -> None:
        for token in re.split(r"(\s+)", chunk):
            if not token:
                continue
            if "\n" in token:
                self._break(token.count("\n"))
            elif token.isspace():
                if self.column:
                    self._raw(" ")
                    self.column += 1
            else:
                self._word(token)

    def _word(self, word: str) -> None:
        # A token longer than the line has no break point of its own. Models
        # emit these constantly: absolute paths, dotted symbols, hashes.
        while len(word) > self.width:
            if self.column:
                self._break(1)
            self._raw(self.indent)
            self._raw(word[: self.width])
            word = word[self.width :]
            self.column = self.width
            self._break(1)
        if self.column and self.column + len(word) > self.width:
            self._break(1)
        if self.column == 0:
            self._raw(self.indent)
        self._raw(word)
        self.column += len(word)
        self.blanks = 0
        self.any_output = True

    def _break(self, count: int) -> None:
        if self.column:
            self.console.print()
            self.column = 0
            self.blanks = 0
            count -= 1
        # A model that emits four newlines should not push the tool call it is
        # about to make off the screen.
        for _ in range(min(count, 1 - self.blanks)):
            self.console.print()
            self.blanks += 1

    def _raw(self, text: str) -> None:
        self.console.print(Text(text), end="", soft_wrap=True, highlight=False)


def render_tool_start(console: Console, tool: str, arguments: dict[str, Any]) -> None:
    room = console.width - len(INDENT) - len(tool) - 4
    console.print(
        Text.assemble(
            (INDENT, ""),
            (f"{MARK} ", _tool_style(tool)),
            (tool, f"bold {_tool_style(tool)}"),
            ("  ", ""),
            (fit(format_arguments(tool, arguments), max(10, room)), "dim"),
        ),
        soft_wrap=True,
    )


def render_tool_end(console: Console, event: AgentEvent) -> None:
    result = event.result
    if result is None:
        return
    lead = f"{INDENT}  -> "
    room = console.width - len(lead)
    if not result.ok:
        console.print(
            Text(lead + fit(result.error or f"{event.tool} failed", room), style="red"),
            soft_wrap=True,
        )
        return
    summary = (result.summary or "ok").strip().splitlines()
    console.print(
        Text(lead + fit(summary[0] if summary else "ok", room), style="dim"),
        soft_wrap=True,
    )
    for line in summary[1:4]:
        console.print(
            Text(f"{INDENT}     " + fit(line, room), style="dim"), soft_wrap=True
        )

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
        room = console.width - len(INDENT) - 9
        if line.startswith("-"):
            console.print(
                Text(f"{INDENT}  {old_no:>5}  {fit(line, room)}", style="red"),
                soft_wrap=True,
            )
            old_no += 1
        elif line.startswith("+"):
            console.print(
                Text(f"{INDENT}  {new_no:>5}  {fit(line, room)}", style="green"),
                soft_wrap=True,
            )
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
        subject = entry.subject or entry.tool
        room = console.width - len(entry.verb) - len(subject) - 6
        console.print(
            Text.assemble(
                (f" {mark} ", "red" if not entry.ok else ("cyan" if entry.kept else "dim")),
                (f"{entry.verb} ", "dim"),
                (fit(subject, max(10, console.width - len(entry.verb) - 6)), "bold"),
                (f"  {fit(entry.constraint, max(0, room))}" if entry.constraint else "", "dim"),
            ),
            soft_wrap=True,
        )
        if entry.found:
            console.print(
                Text("     " + fit(entry.found, console.width - 5), style="dim"),
                soft_wrap=True,
            )
    console.print()


class AgentView:
    """One agent loop, one conversation, one renderer.

    Shared by the coding and operations loops: what differs between them is
    the tool surface and the prompt, not how a turn should look on screen.
    """

    def __init__(
        self,
        console: Console,
        agent: Any,
        *,
        task: str = "",
        repo: str = "",
        root: Path | None = None,
    ) -> None:
        self.console = console
        self.agent = agent
        self.task = task
        self.repo = repo
        self.root = root
        self.timeline: list[TimelineEntry] = []

    @property
    def prompt(self) -> str:
        return f"mimir({self.task})> " if self.task else "mimir> "

    async def turn(self, instruction: str) -> None:
        """Run one instruction, rendering as it goes.

        Ctrl-C cancels the turn rather than the process. A long tool call
        already running is allowed to finish, because killing a test run
        mid-write leaves the worktree in a state nobody asked for.
        """
        console = self.console
        console.print()
        stream = StreamWriter(console)
        try:
            async for event in self.agent.run(instruction):
                if event.type is AgentEventType.TEXT:
                    stream.write(event.text)
                    continue

                stream.close()

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
            stream.close()
            console.print(Text(f"{INDENT}stopped", style="yellow"))
            raise
        except KeyboardInterrupt:
            stream.close()
            console.print(Text(f"{INDENT}stopped", style="yellow"))
        finally:
            stream.close()

        self._summarise()

    def _summarise(self) -> None:
        outcome = self.agent.outcome
        if outcome.stopped == "error":
            return

        # Printed above the counts, not below them, because it changes how the
        # answer should be read and the counts do not.
        grounding = getattr(outcome, "grounding", None)
        if grounding is not None and not grounding.ok:
            self.console.print()
            for line in _wrap(grounding.brief(), self.console.width - len(INDENT)):
                self.console.print(Text(INDENT + line, style="yellow"))

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


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, max(20, width)) or [text]


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


def start_ops_view(console: Console, runner: Any, environment: Any) -> AgentView:
    """A loop that reads the estate, for a request that already names its target."""
    from mimir.agent.ops import OpsAgent

    agent = OpsAgent(
        router=runner.router,
        registry=runner.registry,
        tool_context=runner.tool_context(None),
        settings=runner.settings,
        environment=environment,
        task_class="fast_command",
        max_steps=12,
    )
    agent.ctx.environment = environment
    return AgentView(console, agent)


def start_coding_session(
    console: Console,
    runner: Any,
    task: str,
    repo: str | None = None,
) -> AgentView:
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
    return AgentView(
        console, agent, task=worktree.name, repo=resolved.name, root=worktree.root
    )


__all__ = [
    "AgentView",
    "StreamWriter",
    "fit",
    "render_edit_diff",
    "render_timeline",
    "render_tool_end",
    "render_tool_start",
    "start_coding_session",
    "start_ops_view",
]
