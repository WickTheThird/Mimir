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

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from mimir.agent.events import AgentEvent, AgentEventType, TimelineEntry
from mimir.agent.loop import format_arguments
from mimir.logging import get_logger
from mimir.tools.base import ToolResult

log = get_logger(__name__)

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


class Surface:
    """Where a turn's lines go.

    The renderers used to print straight to the console, which meant the only
    possible layout was one column. Handing them finished lines instead is what
    lets the same code feed a live panel beside a narrower transcript.
    """

    def __init__(self, width: int) -> None:
        self.width = width

    def line(self, text: Text) -> None:
        raise NotImplementedError


class ConsoleSurface(Surface):
    def __init__(self, console: Console, width: int | None = None) -> None:
        super().__init__(width if width is not None else console.width)
        self.console = console

    def line(self, text: Text) -> None:
        self.console.print(text, soft_wrap=True)


class BufferSurface(Surface):
    """Keeps the last ``height`` lines, because a live region cannot scroll."""

    def __init__(self, width: int, height: int = 200) -> None:
        super().__init__(width)
        self.height = height
        self.lines: list[Text] = []

    def line(self, text: Text) -> None:
        self.lines.append(text)
        if len(self.lines) > self.height:
            del self.lines[: len(self.lines) - self.height]

    def tail(self, rows: int) -> list[Text]:
        return self.lines[-rows:] if rows > 0 else []


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

    def __init__(self, surface: Surface, indent: str = INDENT) -> None:
        self.surface = surface
        self.indent = indent
        self.width = max(30, surface.width - len(indent) - 1)
        self.column = 0
        self.blanks = 0
        self.any_output = False
        self._pending = ""
        self._current = Text()

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
            self._flush()

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
            self._flush()
            count -= 1
        # A model that emits four newlines should not push the tool call it is
        # about to make off the screen.
        for _ in range(min(count, 1 - self.blanks)):
            self.surface.line(Text())
            self.blanks += 1

    def _flush(self) -> None:
        self.surface.line(self._current)
        self._current = Text()
        self.column = 0
        self.blanks = 0

    def partial(self) -> Text | None:
        """The line still being written, for a live view to show mid-stream."""
        return self._current if self.column else None

    def _raw(self, text: str) -> None:
        self._current.append(text)


def render_tool_start(surface: Surface, tool: str, arguments: dict[str, Any]) -> None:
    room = surface.width - len(INDENT) - len(tool) - 4
    surface.line(
        Text.assemble(
            (INDENT, ""),
            (f"{MARK} ", _tool_style(tool)),
            (tool, f"bold {_tool_style(tool)}"),
            ("  ", ""),
            (fit(format_arguments(tool, arguments), max(10, room)), "dim"),
        )
    )


def render_tool_end(surface: Surface, event: AgentEvent) -> None:
    result = event.result
    if result is None:
        return
    lead = f"{INDENT}  -> "
    room = surface.width - len(lead)
    if not result.ok:
        surface.line(
            Text(lead + fit(result.error or f"{event.tool} failed", room), style="red")
        )
        return
    summary = (result.summary or "ok").strip().splitlines()
    surface.line(Text(lead + fit(summary[0] if summary else "ok", room), style="dim"))
    for line in summary[1:4]:
        surface.line(Text(f"{INDENT}     " + fit(line, room), style="dim"))

    if event.tool == "edit_worktree_file":
        render_edit_diff(surface, result)


def render_edit_diff(surface: Surface, result: ToolResult) -> None:
    """Show what the edit changed, at the line numbers it changed.

    The tool result already carries both sides, so this is a display of what
    happened rather than a re-read of the file that might disagree with it.
    """
    old = str(result.data.get("old_string", ""))
    new = str(result.data.get("new_string", ""))
    start = int(result.data.get("line", 1) or 1)
    if not old and not new:
        return

    surface.line(Text())
    old_no = new_no = start
    for line in difflib.unified_diff(
        old.splitlines(), new.splitlines(), n=0, lineterm=""
    ):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            continue
        room = surface.width - len(INDENT) - 9
        if line.startswith("-"):
            surface.line(Text(f"{INDENT}  {old_no:>5}  {fit(line, room)}", style="red"))
            old_no += 1
        elif line.startswith("+"):
            surface.line(Text(f"{INDENT}  {new_no:>5}  {fit(line, room)}", style="green"))
            new_no += 1
    surface.line(Text())


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


PANEL_WIDTH = 34
MIN_WIDTH_FOR_PANEL = 104
"""Below this the panel would leave the transcript too narrow to read.

A diff line and a wrapped sentence both need room, and taking a third of a
90 column terminal to show what was looked up makes the thing being looked up
unreadable. Narrow terminals get the transcript and /why."""


class TimelinePanel:
    """The right hand column: what was looked for, and what came back."""

    def __init__(self, width: int = PANEL_WIDTH) -> None:
        self.width = width
        self.entries: list[TimelineEntry] = []

    def add(self, entry: TimelineEntry) -> None:
        self.entries.append(entry)

    def render(self, height: int) -> Panel:
        body = Text()
        # Newest last, and the tail is what is kept, because the step being
        # worked on now is the one worth seeing.
        room = self.width - 4
        shown = self.entries[-max(1, height // 2) :]
        for entry in shown:
            mark = "x" if not entry.ok else ("*" if entry.kept else "-")
            style = "red" if not entry.ok else ("cyan" if entry.kept else "dim")
            body.append(f"{mark} ", style)
            body.append(fit(entry.subject or entry.tool, room - 2) + "\n", "bold")
            if entry.found:
                body.append("  " + fit(entry.found, room - 2) + "\n", "dim")
        if not self.entries:
            body.append("nothing looked up yet", "dim")
        return Panel(
            body,
            title="[dim]trail[/dim]",
            width=self.width,
            height=height,
            border_style="dim",
            padding=(0, 1),
        )


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
        self.source_root: Path | None = None
        """The checkout the worktree came from, which is where its toolchain is."""

        self.panel = True
        """Whether to show the trail beside the transcript while a turn runs."""

        self.panel_view = TimelinePanel()

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
        side = self.panel and console.width >= MIN_WIDTH_FOR_PANEL
        if side:
            await self._turn_with_panel(instruction)
        else:
            await self._turn_inline(instruction)
        self._summarise()

    async def _consume(self, instruction: str, surface, stream, on_change) -> None:
        """The event loop both layouts share, so they cannot drift apart."""
        try:
            async for event in self.agent.run(instruction):
                if event.type is AgentEventType.TEXT:
                    stream.write(event.text)
                    on_change()
                    continue

                stream.close()
                if event.type is AgentEventType.TOOL_START:
                    surface.line(Text())
                    render_tool_start(surface, event.tool, event.arguments)
                elif event.type is AgentEventType.TOOL_END:
                    render_tool_end(surface, event)
                    entry = event.timeline
                    if entry is not None:
                        self.timeline.append(entry)
                        self.panel_view.add(entry)
                elif event.type is AgentEventType.ERROR:
                    surface.line(Text(f"{INDENT}{event.error}", style="bold red"))
                on_change()
        except asyncio.CancelledError:
            stream.close()
            surface.line(Text(f"{INDENT}stopped", style="yellow"))
            on_change()
            raise
        except KeyboardInterrupt:
            stream.close()
            surface.line(Text(f"{INDENT}stopped", style="yellow"))
            on_change()
        finally:
            stream.close()

    async def _turn_inline(self, instruction: str) -> None:
        surface = ConsoleSurface(self.console)
        stream = StreamWriter(surface)
        await self._consume(instruction, surface, stream, lambda: None)

    async def _turn_with_panel(self, instruction: str) -> None:
        """Transcript left, trail right, both updating as the turn runs.

        The live region cannot scroll, so the transcript tails: only the last
        screenful is on show while the turn runs. That is the cost of seeing
        the trail build, and it is why the whole transcript is printed again
        underneath when the turn ends, into the terminal's own scrollback where
        it can be read properly.
        """
        console = self.console
        body_width = console.width - PANEL_WIDTH - 2
        surface = BufferSurface(body_width)
        stream = StreamWriter(surface)
        ceiling = max(8, min(28, console.size.height - 6))

        def frame():
            # The region grows with the work rather than opening at full
            # height. A tall empty box at the top of a turn says the tool is
            # waiting for something, which is the opposite of what is true.
            rows = max(4, min(ceiling, max(len(surface.lines), len(self.timeline) * 2)))
            lines = surface.tail(rows)
            partial = stream.partial()
            if partial is not None:
                lines = [*lines, partial][-rows:]
            left = Group(*lines) if lines else Text("")
            return Columns(
                [left, self.panel_view.render(rows + 2)],
                width=None,
                expand=False,
                padding=(0, 1),
            )

        with Live(
            frame(), console=console, refresh_per_second=8, transient=True
        ) as live:
            await self._consume(instruction, surface, stream, lambda: live.update(frame()))

        for line in surface.lines:
            console.print(line, soft_wrap=True)

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


def _glossary(settings: Any) -> Any:
    """Seeded once per process, and never allowed to block a session."""
    try:
        from mimir.knowledge.glossary import get_glossary

        glossary = get_glossary(settings)
        if not glossary.all(limit=1):
            glossary.seed()
        glossary.prune()
        return glossary
    except Exception:  # noqa: BLE001
        return None


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
    agent.glossary = _glossary(runner.settings)
    return AgentView(console, agent)


async def run_best_of(
    console: Console,
    runner: Any,
    task: str,
    instruction: str,
    *,
    attempts: int = 3,
    repo: str | None = None,
    test_command: str = "",
) -> Any:
    """Try the task several times in separate worktrees and keep the best.

    Each attempt gets its own worktree so they cannot see each other, and the
    losers are discarded. Sampling needs temperature above zero or the attempts
    are one attempt repeated, which is the mistake that invalidated the first
    measurement of this.
    """
    from mimir.agent.select import best_of

    console.print(
        Text(f"trying {attempts} times, keeping whichever the rules prefer", style="dim")
    )
    views: list[AgentView] = []

    def make(index: int) -> AgentView:
        view = start_coding_session(console, runner, f"{task}-{index + 1}", repo)
        view.agent.temperature = 0.0 if attempts == 1 else 0.7
        view.panel = False
        views.append(view)
        return view

    winner, _ = await best_of(
        attempts, make, instruction,
        settings=runner.settings, test_command=test_command, console=console,
    )
    console.print()
    if winner is None:
        console.print(Text("no attempt produced a usable change", style="yellow"))
    else:
        console.print(Text(f"kept {winner.render()}", style="green"))
        console.print(
            Text(f"/worktree diff {winner.task} to review it", style="dim")
        )
    _discard_losers(runner, views, winner)
    return winner


def _discard_losers(runner: Any, views: list[Any], winner: Any) -> None:
    """Delete the worktrees that were not chosen.

    Leaving them would fill the home directory with abandoned attempts, and
    keeping the wrong one around is how the wrong one gets reviewed.
    """
    from mimir.tools.repo import get_repository_directory
    from mimir.worktree import WorktreeManager

    manager = WorktreeManager(runner.settings.home)
    directory = get_repository_directory(runner.settings)
    for view in views:
        if winner is not None and view.task == winner.task:
            continue
        try:
            source = directory.resolve(view.repo).root
            manager.discard(manager.find(source, view.task))
        except Exception:  # noqa: BLE001 - discarding a loser is best effort
            log.debug("worktree_not_discarded", task=view.task)


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
    agent.glossary = _glossary(runner.settings)
    view = AgentView(
        console, agent, task=worktree.name, repo=resolved.name, root=worktree.root
    )
    view.source_root = resolved.root
    return view


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
