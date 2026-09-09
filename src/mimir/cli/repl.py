"""Interactive mode: ``mimir`` with no arguments (ADR 14.1).

A conversational loop over the same runner the one-shot commands use. Session
state persists across turns, so follow-up questions can reference earlier command
output as evidence (ADR 5.1 step 7).
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mimir import __version__
from mimir.cli import render
from mimir.cli.approvals import attach_cli_approvals
from mimir.cli.coding import CodingSession, render_timeline, start_coding_session
from mimir.config import get_settings
from mimir.graph.runner import EventType, InvestigationRunner
from mimir.logging import configure_logging
from mimir.models.state import EnvironmentContext, InvestigationState

SLASH_COMMANDS = {
    "/help": "show this help",
    "/code": "start or reopen a code task, for example /code fix-retry-bounds",
    "/diff": "what the current code task has changed",
    "/why": "how the last answer was reached, step by step",
    "/done": "leave the code task and go back to investigating",
    "/status": "model, tools, language servers, and what is loaded",
    "/tools": "list the typed tools available, grouped by capability",
    "/lsp": "language server status and how to install a missing one",
    "/worktree": "list task worktrees, or /worktree diff <task>",
    "/model": "show the bound models and the context actually served",
    "/context": "show or set the operating context",
    "/ns": "set the namespace, for example /ns payments",
    "/cluster": "set the cluster context",
    "/repo": "set the active repository",
    "/evidence": "list the evidence gathered so far",
    "/commands": "list proposed and executed commands",
    "/hypotheses": "show ranked hypotheses",
    "/skills": "list available skills",
    "/session": "show the current session id and summary",
    "/export": "export this session as an evidence package",
    "/new": "start a fresh session",
    "/quit": "exit",
}

BANNER = """\
MIMIR {version}  local operations investigation

{model}  {tools} tools  {servers}

Ask a question in plain language. /help for commands, /status for what is loaded.
Read-only work runs without asking. Anything that changes state stops for approval.
"""


def _banner_facts(settings: Any) -> dict[str, str]:
    """What is actually loaded, read at startup rather than described.

    The banner used to say only what MIMIR is for. It said nothing about which
    model was bound or which tools existed, so a session that had silently lost
    its language servers looked identical to one that had them.
    """
    try:
        from mimir.tools.base import load_all_tools

        tools = len(load_all_tools().select())
    except Exception:  # noqa: BLE001 - the banner must never block the prompt
        tools = 0
    try:
        from mimir.lsp.servers import available_servers

        found = sorted({s.language for s in available_servers()})
        servers = f"lsp: {', '.join(found)}" if found else "no language servers"
    except Exception:  # noqa: BLE001
        servers = "lsp unavailable"
    profile = settings.models.profiles.get(settings.models.routing.default)
    return {
        "model": profile.model if profile else "no model configured",
        "tools": str(tools),
        "servers": servers,
    }


async def run_repl(console: Console) -> None:
    settings = get_settings()
    # Every other command surface quiets its logging; this one never did, so
    # structured developer output (correlation ids, session ids, shadowed skill
    # counts) was printed straight into a conversational prompt. The log file
    # still receives everything at the configured level.
    configure_logging(
        level="WARNING",
        json_logs=settings.observability.json_logs,
        log_file=settings.observability.log_file,
        force=True,
    )
    runner = InvestigationRunner(settings=settings)
    attach_cli_approvals(runner.approvals, console)

    history_path = settings.home / "history"
    session_prompt = PromptSession(
        history=FileHistory(str(history_path)),
        completer=WordCompleter(list(SLASH_COMMANDS), sentence=True),
    )

    console.print(
        Text(BANNER.format(version=__version__, **_banner_facts(settings)), style="dim")
    )
    environment = EnvironmentContext(
        cluster_context=settings.kubernetes.default_context,
        namespace=settings.kubernetes.default_namespace,
    )
    state: InvestigationState | None = None
    coding: CodingSession | None = None

    while True:
        try:
            prompt = coding.prompt if coding else _prompt_text(environment)
            line = await session_prompt.prompt_async(prompt)
        except (EOFError, KeyboardInterrupt):
            break
        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            action = _handle_slash(console, line, environment, state, coding, runner)
            if action == "quit":
                break
            if action == "new":
                state = None
                console.print(Text("started a new session", style="dim"))
            elif isinstance(action, CodingSession):
                coding = action
            elif action == "done":
                if coding is not None:
                    _close_coding(coding, settings)
                    console.print(
                        Text(f"left {coding.task}; the worktree is kept", style="dim")
                    )
                coding = None
            continue

        if coding is not None:
            await _run_interruptibly(coding.turn(line))
        else:
            state = await _ask(runner, console, line, environment, state)

    if coding is not None:
        _close_coding(coding, settings)
    await runner.aclose()
    console.print(Text("bye", style="dim"))


def _close_coding(coding: CodingSession, settings: Any) -> None:
    """Stop addressing the worktree by name; leave the worktree itself alone.

    Discarding on exit would throw away work because someone typed /done, and
    a task worktree is meant to survive until it is reviewed."""
    from mimir.tools.repo import get_repository_directory

    get_repository_directory(settings).forget_session(f"{coding.task}-worktree")


async def _run_interruptibly(coro: Any) -> None:
    """Run a turn so Ctrl-C stops the turn rather than the session.

    Without this, interrupting a twelve step loop that has gone the wrong way
    means killing the process, which loses the conversation and every file it
    had already read.
    """
    task = asyncio.ensure_future(coro)
    loop = asyncio.get_running_loop()
    # Not every platform has POSIX signal handlers on the loop.
    with contextlib.suppress(NotImplementedError, RuntimeError):
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    finally:
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(signal.SIGINT)


def _prompt_text(environment: EnvironmentContext) -> str:
    bits = [b for b in (environment.cluster_context, environment.namespace) if b]
    scope = "/".join(bits)
    return f"mimir[{scope}]> " if scope else "mimir> "


async def _ask(
    runner: InvestigationRunner,
    console: Console,
    question: str,
    environment: EnvironmentContext,
    previous: InvestigationState | None,
) -> InvestigationState:
    state = runner.new_session(question, environment=environment, interface="cli")
    if previous is not None:
        # Carry forward what was already established so a follow-up does not
        # re-gather the same evidence (ADR 5.1 step 7).
        state.evidence = list(previous.evidence)
        state.commands_executed = list(previous.commands_executed)
        state.outputs = dict(previous.outputs)
        state.hypotheses = list(previous.hypotheses)
        state.environment = previous.environment.merge(environment)

    async for event in runner.stream(question, state=state):
        data = event.data
        if event.type == EventType.PLAN:
            console.print(
                Text.assemble(
                    ("plan  ", "dim"),
                    (str(data.get("task_type")), "bold"),
                    ("  ", ""),
                    (" -> ".join(s["specialist"] for s in data.get("steps", [])), "cyan"),
                )
            )
        elif event.type == EventType.SPECIALIST:
            console.print(
                Text(
                    f"  {data['specialist']} (conf {data['confidence']:.2f}, "
                    f"{data['tool_calls']} tool calls)",
                    style="red" if data.get("error") else "cyan",
                )
            )
        elif event.type == EventType.COMMAND:
            console.print(Text(f"  proposed: {data.get('display')}", style="dim"))
        elif event.type == EventType.ERROR:
            console.print(Text(f"error: {data.get('error')}", style="bold red"))

    if state.final_answer:
        console.print()
        console.print(render.render_answer(state.final_answer, state.final_confidence))
    console.print()
    return state


def _handle_slash(
    console: Console,
    line: str,
    environment: EnvironmentContext,
    state: InvestigationState | None,
    coding: CodingSession | None = None,
    runner: Any = None,
) -> Any:
    parts = line.split(maxsplit=1)
    command = parts[0]
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return "quit"
    if command == "/new":
        return "new"

    if command == "/code":
        if not argument:
            console.print(
                Text("name the task: /code fix-retry-bounds [repo]", style="yellow")
            )
            return None
        parts = argument.split()
        try:
            return start_coding_session(
                console, runner, parts[0], parts[1] if len(parts) > 1 else None
            )
        except Exception as exc:  # noqa: BLE001 - a bad repo name must not end the session
            console.print(Text(str(exc), style="yellow"))
            return None
    if command == "/done":
        return "done"
    if command == "/why":
        if coding is None:
            console.print(Text("only inside a code task for now", style="dim"))
        else:
            render_timeline(console, coding.timeline)
        return None
    if command == "/diff":
        if coding is None:
            console.print(Text("not in a code task; /code <task> starts one", style="dim"))
        else:
            _print_worktrees(console, f"diff {coding.task}")
        return None

    if command == "/status":
        _print_status(console)
    elif command == "/tools":
        _print_tools(console, argument)
    elif command == "/lsp":
        _print_lsp(console)
    elif command == "/model":
        _print_models(console)
    elif command == "/worktree":
        _print_worktrees(console, argument)
    elif command == "/help":
        table = Table(box=None, header_style="dim")
        table.add_column("command")
        table.add_column("what it does")
        for name, description in SLASH_COMMANDS.items():
            table.add_row(name, description)
        console.print(table)
    elif command == "/context":
        if argument:
            environment.cluster_context = argument
        lines = environment.render_lines()
        console.print("\n".join(lines) if lines else Text("no context set", style="dim"))
    elif command == "/ns":
        environment.namespace = argument or None
        console.print(Text(f"namespace: {environment.namespace or 'unset'}", style="dim"))
    elif command == "/cluster":
        environment.cluster_context = argument or None
        console.print(Text(f"context: {environment.cluster_context or 'unset'}", style="dim"))
    elif command == "/repo":
        environment.repositories = [argument] if argument else []
        console.print(Text(f"repository: {argument or 'unset'}", style="dim"))
    elif command == "/evidence":
        if state and state.evidence:
            console.print(render.render_evidence(state.ranked_evidence(), limit=40))
        else:
            console.print(Text("no evidence gathered yet", style="dim"))
    elif command == "/commands":
        if state and (state.commands_planned or state.commands_executed):
            for proposed in state.commands_planned:
                console.print(render.render_command(proposed))
            for record in state.commands_executed:
                console.print(render.render_execution(record))
        else:
            console.print(Text("no commands yet", style="dim"))
    elif command == "/hypotheses":
        table = render.render_hypotheses(state) if state else None
        console.print(table or Text("no hypotheses yet", style="dim"))
    elif command == "/skills":
        from mimir.skills.registry import get_skill_registry

        for skill in get_skill_registry().all():
            console.print(f"  {skill.name:<40} {skill.when_to_use[:70]}")
    elif command == "/session":
        if state:
            console.print(render.render_session_summary(state))
        else:
            console.print(Text("no active session", style="dim"))
    elif command == "/export":
        if not state:
            console.print(Text("no active session", style="dim"))
        else:
            from mimir.export import evidence_package_markdown

            target = Path(argument) if argument else Path(f"{state.session_id}.md")
            target.write_text(evidence_package_markdown(state), encoding="utf-8")
            console.print(Text(f"wrote {target}", style="green"))
    else:
        console.print(Text(f"unknown command {command}; try /help", style="yellow"))
    return None


def main() -> None:
    from mimir.cli.render import make_console

    asyncio.run(run_repl(make_console()))


# ---------------------------------------------------------------------------
# capability introspection
#
# The prompt used to describe what MIMIR is for and nothing about what it
# currently has. A session whose language servers had silently gone missing,
# or whose model was serving a different context than configured, looked
# exactly like a healthy one. These read live state rather than repeating the
# documentation.
# ---------------------------------------------------------------------------


def _print_status(console: Console) -> None:
    from mimir.config import get_settings

    settings = get_settings()
    table = Table(box=None, header_style="dim")
    table.add_column("")
    table.add_column("")

    try:
        from mimir.tools.base import load_all_tools

        registry = load_all_tools()
        specs = registry.all()
        offered = len(registry.select())
        table.add_row("tools", f"{len(specs)} registered, {offered} offered to specialists")
    except Exception as exc:  # noqa: BLE001
        table.add_row("tools", f"unavailable: {exc}")

    profile = settings.models.profiles.get(settings.models.routing.default)
    if profile is not None:
        try:
            from mimir.eval.provenance import resolve_model

            identity = resolve_model(settings.models.routing.default, settings)
            served = identity.served_context or 0
            note = ""
            if identity.context_mismatch:
                note = f"  MISMATCH: serving {served}, configured {identity.context_window}"
            table.add_row(
                "model",
                f"{profile.model} via {profile.runtime}, context "
                f"{served or profile.context_window}{note}",
            )
        except Exception:  # noqa: BLE001
            table.add_row("model", f"{profile.model} via {profile.runtime}")

    try:
        from mimir.lsp.servers import SERVERS, available_servers

        found = sorted({s.language for s in available_servers()})
        table.add_row(
            "language servers",
            f"{', '.join(found)}  ({len({s.language for s in SERVERS}) - len(found)} "
            "language(s) without one)" if found else "none installed",
        )
    except Exception:  # noqa: BLE001
        table.add_row("language servers", "unavailable")

    try:
        from mimir.skills.registry import SkillRegistry

        table.add_row("skills", f"{len(SkillRegistry(settings).all())} loaded")
    except Exception:  # noqa: BLE001
        pass

    table.add_row("home", str(settings.home))
    console.print(table)


def _print_tools(console: Console, capability: str) -> None:
    from mimir.tools.base import load_all_tools

    registry = load_all_tools()
    specs = [s for s in registry.select() if not capability or s.capability.value == capability]
    if not specs:
        known = sorted({s.capability.value for s in registry.all()})
        console.print(Text(f"no tools for {capability!r}. try: {', '.join(known)}", style="yellow"))
        return

    grouped: dict[str, list[Any]] = {}
    for spec in specs:
        grouped.setdefault(spec.capability.value, []).append(spec)

    table = Table(box=None, header_style="dim")
    table.add_column("capability")
    table.add_column("risk")
    table.add_column("tool")
    for cap in sorted(grouped):
        for i, spec in enumerate(sorted(grouped[cap], key=lambda s: s.name)):
            table.add_row(cap if i == 0 else "", spec.risk.value, spec.name)
    console.print(table)
    hidden = len(registry.all()) - len(registry.select())
    if hidden and not capability:
        console.print(
            Text(f"{hidden} tool(s) hidden because an exact replacement exists", style="dim")
        )


def _print_lsp(console: Console) -> None:
    from mimir.lsp.servers import SERVERS

    table = Table(box=None, header_style="dim")
    for column in ("language", "server", "state", "install"):
        table.add_column(column)
    for spec in SERVERS:
        table.add_row(
            spec.language,
            spec.binary,
            Text("ready", style="green") if spec.installed else Text("missing", style="dim"),
            "" if spec.installed else spec.install_hint,
        )
    console.print(table)


def _print_models(console: Console) -> None:
    from mimir.config import get_settings
    from mimir.eval.provenance import resolve_model

    settings = get_settings()
    table = Table(box=None, header_style="dim")
    for column in ("role", "model", "context", "digest"):
        table.add_column(column)
    for role, alias in sorted(settings.models.routing.model_dump().items()):
        if not isinstance(alias, str):
            continue
        profile = settings.models.profiles.get(alias)
        if profile is None:
            continue
        try:
            identity = resolve_model(alias, settings)
            served = identity.served_context or profile.context_window
            context = Text(
                str(served),
                style="red" if identity.context_mismatch else "",
            )
            digest = identity.digest[:12]
        except Exception:  # noqa: BLE001
            context, digest = Text(str(profile.context_window)), ""
        table.add_row(role.replace("_", " "), profile.model, context, digest)
    console.print(table)


def _print_worktrees(console: Console, argument: str) -> None:
    from mimir.config import get_settings
    from mimir.worktree import TaskWorktree, WorktreeError, WorktreeManager

    manager = WorktreeManager(get_settings().home)
    if argument.startswith("diff "):
        name = argument.split(maxsplit=1)[1].strip()
        matches = [w for w in manager.list() if name in str(w["name"])]
        if not matches:
            console.print(Text(f"no task worktree matching {name!r}", style="yellow"))
            return
        entry = matches[0]
        root = Path(str(entry["path"]))
        # Same base as diff_worktree uses: the working tree against its own
        # HEAD. Task worktrees are written to, not committed in.
        worktree = TaskWorktree(
            name=str(entry["name"]), branch=str(entry["branch"]), root=root,
            repo_root=root, base_commit="HEAD", created_at=0.0,
        )
        try:
            console.print(Text(manager.diff(worktree, stat=True).strip()
                               or "no uncommitted changes", style="dim"))
        except WorktreeError as exc:
            console.print(Text(str(exc), style="yellow"))
        return

    found = manager.list()
    if not found:
        console.print(Text("no task worktrees", style="dim"))
        return
    table = Table(box=None, header_style="dim")
    for column in ("task", "branch", "state"):
        table.add_column(column)
    for entry in found:
        table.add_row(
            str(entry["name"]),
            str(entry["branch"]),
            Text("uncommitted changes", style="yellow") if entry["dirty"] else "clean",
        )
    console.print(table)
