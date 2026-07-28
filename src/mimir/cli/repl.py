"""Interactive mode: ``mimir`` with no arguments (ADR 14.1).

A conversational loop over the same runner the one-shot commands use. Session
state persists across turns, so follow-up questions can reference earlier command
output as evidence (ADR 5.1 step 7).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mimir import __version__
from mimir.cli import render
from mimir.cli.approvals import attach_cli_approvals
from mimir.config import get_settings
from mimir.graph.runner import EventType, InvestigationRunner
from mimir.models.state import EnvironmentContext, InvestigationState

SLASH_COMMANDS = {
    "/help": "show this help",
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

Ask a question in plain language. Type /help for commands, /quit to exit.
Read-only work runs without asking. Anything that changes state stops for approval.
"""


async def run_repl(console: Console) -> None:
    settings = get_settings()
    runner = InvestigationRunner(settings=settings)
    attach_cli_approvals(runner.approvals, console)

    history_path = settings.home / "history"
    session_prompt = PromptSession(
        history=FileHistory(str(history_path)),
        completer=WordCompleter(list(SLASH_COMMANDS), sentence=True),
    )

    console.print(Text(BANNER.format(version=__version__), style="dim"))
    environment = EnvironmentContext(
        cluster_context=settings.kubernetes.default_context,
        namespace=settings.kubernetes.default_namespace,
    )
    state: InvestigationState | None = None

    while True:
        try:
            line = await session_prompt.prompt_async(_prompt_text(environment))
        except (EOFError, KeyboardInterrupt):
            break
        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            action = _handle_slash(console, line, environment, state)
            if action == "quit":
                break
            if action == "new":
                state = None
                console.print(Text("started a new session", style="dim"))
            continue

        state = await _ask(runner, console, line, environment, state)

    await runner.aclose()
    console.print(Text("bye", style="dim"))


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
) -> str | None:
    parts = line.split(maxsplit=1)
    command = parts[0]
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return "quit"
    if command == "/new":
        return "new"

    if command == "/help":
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
