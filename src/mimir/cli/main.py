"""MIMIR command line interface (ADR 14).

Command surface follows ADR 14.1. Those names were "illustrative, not committed",
so where a name was ambiguous the clearer one is used and the ADR shape is kept.

Every path runs through :class:`~mimir.graph.runner.InvestigationRunner`, so the
CLI and the API cannot diverge in behaviour.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from mimir import __version__
from mimir.cli import render
from mimir.cli.approvals import attach_cli_approvals
from mimir.config import get_settings, reset_settings_cache
from mimir.graph.runner import EventType, InvestigationRunner, RunEvent
from mimir.logging import configure_logging
from mimir.models.state import EnvironmentContext, InvestigationState

app = typer.Typer(
    name="mimir",
    help="Local operations investigation and on-call assistance.",
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)
console = render.make_console()

repo_app = typer.Typer(help="Repository investigation.")
k8s_app = typer.Typer(help="Kubernetes investigation.")
sdm_app = typer.Typer(help="SDM and container investigation.")
session_app = typer.Typer(help="Session management.")
memory_app = typer.Typer(help="Curated memory.")
skills_app = typer.Typer(help="Skills.")
models_app = typer.Typer(help="Model runtimes.")
keys_app = typer.Typer(help="API keys for non-loopback access.")
app.add_typer(repo_app, name="repo")
app.add_typer(k8s_app, name="k8s")
app.add_typer(sdm_app, name="sdm")
app.add_typer(session_app, name="session")
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(models_app, name="models")
app.add_typer(keys_app, name="keys")


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------


def _build_runner(quiet: bool = False) -> InvestigationRunner:
    settings = get_settings()
    configure_logging(
        level="WARNING" if quiet else settings.observability.log_level,
        json_logs=settings.observability.json_logs,
        log_file=settings.observability.log_file,
        force=True,
    )
    runner = InvestigationRunner(settings=settings)
    attach_cli_approvals(runner.approvals, console)
    return runner


def _environment(
    context: str | None = None,
    namespace: str | None = None,
    repo: str | None = None,
    resource: str | None = None,
    since: str | None = None,
) -> EnvironmentContext:
    return EnvironmentContext(
        cluster_context=context,
        namespace=namespace,
        repositories=[repo] if repo else [],
        sdm_resource=resource,
        time_range=since,
    )


async def _run_stream(
    runner: InvestigationRunner,
    question: str,
    environment: EnvironmentContext,
    *,
    show_evidence: bool = True,
    interface: str = "cli",
) -> InvestigationState:
    """Drive one investigation and render it as it happens.

    Always closes the runner. The checkpointer is an async context manager held
    open in an AsyncExitStack; letting the event loop finalise it during
    shutdown instead of closing it here raises "asynchronous generator is
    already running" and prints a traceback over the answer.
    """
    state = runner.new_session(question, environment=environment, interface=interface)
    spinner = Spinner("dots", text=Text("thinking", style="dim"))
    panel = render.render_context(state)
    if panel:
        console.print(panel)

    try:
        with Live(spinner, console=console, refresh_per_second=8, transient=True) as live:

            def status(text: str) -> None:
                live.update(Spinner("dots", text=Text(text, style="dim")))

            async for event in runner.stream(question, state=state):
                _render_event(event, live, status, show_evidence)
    finally:
        await runner.aclose()

    if state.final_answer:
        console.print()
        console.print(render.render_answer(state.final_answer, state.final_confidence))
    hypotheses = render.render_hypotheses(state)
    if hypotheses:
        console.print()
        console.print(Text("hypotheses", style="dim"))
        console.print(hypotheses)
    console.print()
    console.print(render.render_session_summary(state))
    return state


def _render_event(event: RunEvent, live: Live, status: Any, show_evidence: bool) -> None:
    data = event.data
    if event.type == EventType.NODE_END:
        status(f"{data.get('node', '')} ...")
    elif event.type == EventType.PLAN:
        live.console.print(
            Text.assemble(
                ("plan  ", "dim"),
                (str(data.get("task_type")), "bold"),
                ("  ", ""),
                (" -> ".join(s["specialist"] for s in data.get("steps", [])), "cyan"),
            )
        )
        for question in data.get("missing_context") or []:
            live.console.print(Text(f"  needs: {question}", style="yellow"))
    elif event.type == EventType.SPECIALIST:
        style = "red" if data.get("error") else render.SPECIALIST_STYLE
        live.console.print(
            Text.assemble(
                ("  ", ""),
                (f"{data['specialist']:<24}", style),
                (f"conf {data['confidence']:.2f}  ", "dim"),
                (f"{data['tool_calls']} tool calls  ", "dim"),
                (f"{data['evidence']} evidence", "dim"),
            )
        )
        if data.get("error"):
            live.console.print(Text(f"    failed: {data['error']}", style="red"))
    elif event.type == EventType.COMMAND:
        live.console.print(
            Text.assemble(("  proposed  ", "dim"), (data.get("display", ""), "bold"))
        )
    elif event.type == EventType.EVIDENCE and show_evidence:
        citations = ", ".join(data.get("citations") or []) or data.get("source_type", "")
        marker = "+" if data.get("supports", True) else "!"
        live.console.print(
            Text.assemble(
                (f"  {marker} ", "green" if data.get("supports", True) else "red"),
                (str(data.get("claim", ""))[:110], ""),
                (f"  [{citations[:70]}]", "dim"),
            )
        )
    elif event.type == EventType.ERROR:
        live.console.print(Text(f"error: {data.get('error')}", style="bold red"))


def _validate_live_flags(
    allow_live: bool, confirmed: bool, allowlist: str
) -> list[str]:
    """Three independent things must be true before a scoring run reaches real
    infrastructure.

    A benchmark quietly acquired cluster access once already. One flag is too
    easy to inherit from a copied command line, so the confirmation and the
    context allowlist are separate and both mandatory.
    """
    if not allow_live:
        return []
    missing = []
    if not confirmed:
        missing.append("--confirm-live-eval")
    if not allowlist.strip():
        missing.append("--live-context-allowlist <contexts>")
    if missing:
        console.print(
            Text("--allow-live also requires: " + ", ".join(missing), style="bold red")
        )
        console.print(
            "[dim]a scoring run that reaches live clusters is neither reproducible "
            "nor unattended-safe[/dim]"
        )
        raise typer.Exit(2)
    return [c.strip() for c in allowlist.split(",") if c.strip()]


def _run(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        raise typer.Exit(130) from None


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"mimir {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _interactive()


def _interactive() -> None:
    """Interactive mode: ``mimir`` with no arguments (ADR 14.1)."""
    from mimir.cli.repl import run_repl

    _run(run_repl(console))


@app.command()
def investigate(
    question: str = typer.Argument(..., help="What you want to know."),
    context: str | None = typer.Option(None, "--context", "-c", help="Cluster context."),
    namespace: str | None = typer.Option(None, "--namespace", "-n"),
    repo: str | None = typer.Option(None, "--repo", "-r"),
    since: str | None = typer.Option(None, "--since", help="Time range, e.g. 30m or 2h."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress evidence streaming."),
    json_out: bool = typer.Option(False, "--json", help="Emit the full state as JSON."),
) -> None:
    """Run a full investigation."""
    runner = _build_runner(quiet=True)
    state = _run(
        _run_stream(
            runner,
            question,
            _environment(context, namespace, repo, since=since),
            show_evidence=not quiet,
        )
    )
    if json_out:
        console.print_json(state.model_dump_json())


@app.command()
def command(
    request: str = typer.Argument(..., help="Describe the command in natural language."),
    context: str | None = typer.Option(None, "--context", "-c"),
    namespace: str | None = typer.Option(None, "--namespace", "-n"),
    execute: bool = typer.Option(
        False, "--execute", help="Offer to run the command after showing it."
    ),
) -> None:
    """Construct a command from a natural-language description (ADR G1).

    The command is always displayed before anything runs, and anything above the
    auto-execute ceiling still requires approval even with --execute.
    """
    runner = _build_runner(quiet=True)
    state = _run(
        _run_stream(
            runner,
            f"Construct the command for this request. Show it, do not run it: {request}",
            _environment(context, namespace),
            show_evidence=False,
        )
    )
    for proposed in state.commands_planned:
        console.print(render.render_command(proposed))
    if execute and state.commands_planned:
        _run(_execute_interactive(runner, state))


async def _execute_interactive(runner: InvestigationRunner, state: InvestigationState) -> None:
    from mimir.tools.exec import ExecutionOptions

    for proposed in state.commands_planned:
        if not typer.confirm(f"Run: {proposed.display}?", default=False):
            console.print(Text("  skipped", style="dim"))
            continue
        record = await runner.executor.run(
            proposed, session_id=state.session_id, options=ExecutionOptions()
        )
        state.record_execution(record)
        console.print(render.render_execution(record))


@app.command()
def research(
    question: str = typer.Argument(..., help="A technical question for public sources."),
) -> None:
    """Search and browse the public web, with citations (ADR G6)."""
    runner = _build_runner(quiet=True)
    _run(
        _run_stream(
            runner,
            f"Research this using public web sources and cite them: {question}",
            EnvironmentContext(),
        )
    )


@app.command()
def logs(
    file: Path | None = typer.Argument(None, help="Log file, or omit to read stdin."),
    question: str = typer.Option(
        "Diagnose what these logs show.", "--question", "-q"
    ),
) -> None:
    """Analyse pasted or piped logs (ADR 5.6)."""
    text = file.read_text(encoding="utf-8", errors="replace") if file else sys.stdin.read()
    if not text.strip():
        console.print("[red]no log content provided[/red]")
        raise typer.Exit(1)

    runner = _build_runner(quiet=True)

    async def go() -> InvestigationState:
        from mimir.tools.base import REGISTRY

        ctx = runner.tool_context()
        result = await REGISTRY.require("ingest_logs").invoke(
            {"text": text, "label": file.name if file else "stdin"}, ctx
        )
        if not result.ok:
            console.print(f"[red]{result.error}[/red]")
            raise typer.Exit(1)
        ref = result.artifact_ref
        console.print(Text(f"ingested as {ref}: {result.summary}", style="dim"))
        return await _run_stream(
            runner,
            f"{question} The logs are stored as artifact {ref}; "
            f"use the log tools with input_ref='{ref}'.",
            EnvironmentContext(),
        )

    _run(go())


# ---------------------------------------------------------------------------
# repo / k8s / sdm
# ---------------------------------------------------------------------------


@repo_app.command("ask")
def repo_ask(
    question: str = typer.Argument(...),
    repo: str | None = typer.Option(None, "--repo", "-r"),
) -> None:
    """Ask a question about a repository, answered with file and line citations."""
    runner = _build_runner(quiet=True)
    _run(_run_stream(runner, question, _environment(repo=repo)))


@repo_app.command("flow")
def repo_flow(
    entrypoint: str = typer.Argument(..., help="File path or symbol to trace from."),
    repo: str | None = typer.Option(None, "--repo", "-r"),
    depth: int = typer.Option(3, "--depth"),
) -> None:
    """Trace an execution flow across files (ADR 5.3)."""
    runner = _build_runner(quiet=True)

    async def go() -> None:
        from mimir.tools.base import REGISTRY

        result = await REGISTRY.require("build_flow_evidence").invoke(
            {"entrypoint": entrypoint, "repo": repo, "max_depth": depth},
            runner.tool_context(),
        )
        console.print(result.summary if result.ok else Text(str(result.error), style="red"))
        if result.ok:
            console.print(render.render_evidence(result.evidence))

    _run(go())


@k8s_app.command("investigate")
def k8s_investigate(
    service: str = typer.Argument(..., help="Workload or service to investigate."),
    context: str | None = typer.Option(None, "--context", "-c"),
    namespace: str | None = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Investigate a Kubernetes workload, read-only by default."""
    runner = _build_runner(quiet=True)
    _run(
        _run_stream(
            runner,
            f"Investigate the state and health of '{service}'. "
            "Use read-only commands. Report restarts, events, resource pressure, "
            "and recent rollouts.",
            _environment(context, namespace),
        )
    )


@sdm_app.command("investigate")
def sdm_investigate(
    resource: str = typer.Argument(..., help="SDM resource name or partial name."),
    target: str | None = typer.Option(None, "--target", help="Container or service."),
) -> None:
    """Investigate an SDM-mediated resource (ADR 5.4)."""
    runner = _build_runner(quiet=True)
    target_text = f" Focus on '{target}'." if target else ""
    _run(
        _run_stream(
            runner,
            f"Investigate the SDM resource '{resource}'.{target_text} "
            "Verify SDM status first and use read-only inspection only.",
            _environment(resource=resource),
        )
    )


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


@session_app.command("list")
def session_list(limit: int = typer.Option(20, "--limit", "-l")) -> None:
    """List recent sessions."""
    from mimir.persistence.repositories import SessionRepository

    rows = SessionRepository().list(limit=limit)
    if not rows:
        console.print("[dim]no sessions recorded yet[/dim]")
        return
    table = Table(box=None, header_style="dim")
    table.add_column("session")
    table.add_column("when", style="dim")
    table.add_column("status")
    table.add_column("request", overflow="fold")
    for row in rows:
        import datetime

        when = datetime.datetime.fromtimestamp(row.created_at).strftime("%Y-%m-%d %H:%M")
        table.add_row(row.id, when, row.status, (row.user_request or "")[:80])
    console.print(table)


@session_app.command("resume")
def session_resume(session_id: str = typer.Argument(...)) -> None:
    """Continue a checkpointed investigation (ADR 14.1)."""
    runner = _build_runner(quiet=True)

    async def go() -> None:
        state = await runner.resume(session_id)
        if state is None:
            console.print(f"[red]no checkpoint found for {session_id}[/red]")
            raise typer.Exit(1)
        if state.final_answer:
            console.print(render.render_answer(state.final_answer, state.final_confidence))
        console.print(render.render_session_summary(state))

    _run(go())


@session_app.command("show")
def session_show(session_id: str = typer.Argument(...)) -> None:
    """Show a stored session with its evidence."""
    from mimir.persistence.repositories import load_state

    state = load_state(session_id)
    if state is None:
        console.print(f"[red]unknown session {session_id}[/red]")
        raise typer.Exit(1)
    console.print(render.render_session_summary(state))
    console.print()
    console.print(render.render_evidence(state.evidence, limit=40))
    if state.final_answer:
        console.print()
        console.print(render.render_answer(state.final_answer, state.final_confidence))


@app.command("export")
def export_session(
    session_id: str = typer.Argument(...),
    fmt: str = typer.Option("md", "--format", "-f", help="md or json."),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    """Export an evidence package for a hosted coding agent (ADR 5.8)."""
    from mimir.export import evidence_package_json, evidence_package_markdown
    from mimir.persistence.repositories import load_state

    state = load_state(session_id)
    if state is None:
        console.print(f"[red]unknown session {session_id}[/red]")
        raise typer.Exit(1)
    body = (
        evidence_package_json(state) if fmt == "json" else evidence_package_markdown(state)
    )
    if out:
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote {out}[/green]")
    else:
        console.print(body)


# ---------------------------------------------------------------------------
# memory / skills / models
# ---------------------------------------------------------------------------


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(...), limit: int = typer.Option(8, "--limit", "-l")
) -> None:
    """Search curated memory."""
    runner = _build_runner(quiet=True)

    async def go() -> None:
        from mimir.tools.base import REGISTRY

        result = await REGISTRY.require("search_memory").invoke(
            {"query": query, "limit": limit}, runner.tool_context()
        )
        console.print(result.summary if result.ok else Text(str(result.error), style="red"))

    _run(go())


@memory_app.command("reindex")
def memory_reindex(force: bool = typer.Option(False, "--force")) -> None:
    """Rebuild the memory index."""
    from mimir.knowledge.index import get_knowledge_index

    stats = get_knowledge_index().reindex(force=force)
    console.print(stats.summary())


@memory_app.command("import")
def memory_import(
    path: Path = typer.Argument(..., help="Claude, Codex, or ChatGPT export."),
    tool: str = typer.Option("", "--tool", help="claude, codex, chatgpt, or other."),
) -> None:
    """Import prior assistant knowledge as untrusted candidate memory (ADR 11.5)."""
    from mimir.knowledge.importers import ConversationImporter

    result = ConversationImporter().import_path(path, tool=tool)
    console.print(result.summary())
    console.print(
        "[dim]imported material lands in imports/ as unverified. "
        "Nothing reaches stable memory without review.[/dim]"
    )


@skills_app.command("list")
def skills_list() -> None:
    """List available skills with their level-1 descriptions."""
    from mimir.skills.registry import get_skill_registry
    from mimir.tools.base import load_all_tools

    load_all_tools()
    registry = get_skill_registry()
    table = Table(box=None, header_style="dim")
    table.add_column("skill")
    table.add_column("specialist", style="cyan")
    table.add_column("risk")
    table.add_column("when to use", overflow="fold")
    for skill in registry.all():
        table.add_row(
            skill.name,
            getattr(skill.specialist, "value", str(skill.specialist)),
            getattr(skill.max_risk, "value", str(skill.max_risk)),
            skill.when_to_use[:90],
        )
    console.print(table)
    console.print(
        f"\n[dim]catalogue is about {registry.catalogue_tokens()} tokens at level 1[/dim]"
    )


@skills_app.command("show")
def skills_show(name: str = typer.Argument(...)) -> None:
    """Show a skill body (level-2 load)."""
    from mimir.skills.registry import get_skill_registry
    from mimir.skills.runner import SkillRunner

    loaded = SkillRunner(get_skill_registry()).load(name)
    console.print(loaded.body)


@skills_app.command("validate")
def skills_validate() -> None:
    """Validate every skill on disk."""
    from mimir.skills.registry import get_skill_registry
    from mimir.skills.testing import run_skill_tests
    from mimir.tools.base import load_all_tools

    # The tool registry must be populated first, or every declared tool looks
    # unregistered and the whole report is noise.
    load_all_tools()
    registry = get_skill_registry().reload()
    ok = True
    for skill in registry.all():
        report = run_skill_tests(skill)
        status = Text("ok", style="green") if report.ok else Text("FAIL", style="red")
        console.print(Text.assemble((f"{skill.name:<40}", ""), status))
        if not report.ok:
            ok = False
            console.print(Text(report.render(), style="red"))
    if not ok:
        raise typer.Exit(1)


@models_app.command("list")
def models_list() -> None:
    """Show configured model profiles and routing."""
    settings = get_settings()
    table = Table(box=None, header_style="dim")
    table.add_column("alias")
    table.add_column("runtime")
    table.add_column("model")
    table.add_column("base url", style="dim")
    table.add_column("ctx")
    for profile in settings.models.profiles.values():
        table.add_row(
            profile.alias,
            profile.runtime,
            profile.model,
            profile.base_url,
            str(profile.context_window),
        )
    console.print(table)
    console.print()
    routing = Table(box=None, header_style="dim")
    routing.add_column("task class")
    routing.add_column("alias")
    for key, value in settings.models.routing.model_dump().items():
        routing.add_row(key, value)
    console.print(routing)


@models_app.command("set")
def models_set(
    alias: str = typer.Argument(...),
    model: str | None = typer.Option(None, "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    runtime: str | None = typer.Option(None, "--runtime"),
    context_window: int | None = typer.Option(None, "--context-window"),
) -> None:
    """Update a model profile in the config file."""
    from mimir.cli.init import update_config

    changes: dict[str, Any] = {}
    if model:
        changes["model"] = model
    if base_url:
        changes["base_url"] = base_url
    if runtime:
        changes["runtime"] = runtime
    if context_window:
        changes["context_window"] = context_window
    if not changes:
        console.print("[yellow]nothing to change[/yellow]")
        raise typer.Exit(1)
    path = update_config({"models": {"profiles": {alias: changes}}})
    reset_settings_cache()
    console.print(f"[green]updated {alias} in {path}[/green]")


# ---------------------------------------------------------------------------
# setup and diagnostics
# ---------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Create the config file, directory layout, and seed knowledge."""
    from mimir.cli.init import initialise

    for line in initialise(force=force):
        console.print(line)


@app.command()
def doctor() -> None:
    """Check that everything MIMIR depends on is reachable (ADR 22.3)."""
    from mimir.cli.doctor import run_doctor

    ok = _run(run_doctor(console))
    raise typer.Exit(0 if ok else 1)


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the HTTP API and web backend."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "mimir.api.app:create_app",
        factory=True,
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
        log_level=settings.observability.log_level.lower(),
    )


@keys_app.command("create")
def keys_create(label: str = typer.Option("", "--label")) -> None:
    """Generate an API key for non-loopback access (ADR 16.1)."""
    from mimir.api.auth import generate_api_key
    from mimir.cli.init import update_config

    key = generate_api_key()
    existing = list(get_settings().api.api_keys)
    update_config({"api": {"api_keys": [*existing, key]}})
    reset_settings_cache()
    console.print(f"[green]created key{' ' + label if label else ''}[/green]")
    console.print(key)
    console.print(
        "\n[yellow]This is shown once. Store it in your password manager.[/yellow]\n"
        "[dim]This key grants the OpenAI-compatible inference endpoint only. "
        "Shell, Kubernetes, SDM, and database helpers stay loopback-only.[/dim]"
    )


@app.command()
def evaluate(
    corpus: Path | None = typer.Option(None, "--corpus", help="Corpus YAML file or directory."),
    deterministic_only: bool = typer.Option(
        False, "--deterministic", help="Skip cases that need a model runtime."
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the JSON report here."),
    allow_live: bool = typer.Option(
        False,
        "--allow-live",
        help="Let model cases call Kubernetes, SDM, and databases. Off by default.",
    ),
    confirm_live_eval: bool = typer.Option(
        False,
        "--confirm-live-eval",
        help="Required alongside --allow-live. Two flags, so a copied argument "
        "cannot enable real infrastructure by itself.",
    ),
    include_hidden: bool = typer.Option(
        False,
        "--hidden",
        help="Also run the held-out corpus from $MIMIR_HOME/eval-hidden. Use it "
        "to check whether an improvement generalised, not to tune against.",
    ),
    live_context_allowlist: str = typer.Option(
        "",
        "--live-context-allowlist",
        help="Comma-separated cluster contexts a live run may touch. Required "
        "with --allow-live; anything else is refused by the policy engine.",
    ),
) -> None:
    """Run the evaluation corpus (ADR 21).

    Deterministic cases exercise the policy engine and need no model. Model cases
    run full investigations and need a runtime.
    """
    from mimir.eval.harness import EvalHarness

    # Validate the live-access flags before doing any work, so an invalid
    # invocation fails in a second rather than after the deterministic suite.
    live_contexts = _validate_live_flags(allow_live, confirm_live_eval, live_context_allowlist)

    harness = EvalHarness(get_settings())
    cases = EvalHarness.load_corpus(corpus, include_hidden=include_hidden)
    hidden_dir = EvalHarness.hidden_corpus_dir()
    if include_hidden and not hidden_dir.is_dir():
        console.print(
            Text(f"--hidden requested but {hidden_dir} does not exist", style="yellow")
        )
    console.print(
        f"[dim]{len(cases)} case(s) loaded"
        + (" (including held-out)" if include_hidden else "")
        + "[/dim]\n"
    )

    report = harness.run_deterministic(cases)
    console.print(report.summary())

    if not deterministic_only and any(not c.deterministic for c in cases):
        console.print("\n[dim]running model cases...[/dim]")
        if allow_live:
            get_settings().kubernetes.allowed_contexts = [f"^{c}$" for c in live_contexts]
            console.print(
                Text(
                    "  LIVE EVALUATION. Model cases may call these contexts and "
                    f"nothing else: {', '.join(live_contexts)}. Scores depend on "
                    "live state and are not reproducible.",
                    style="bold yellow",
                )
            )
        else:
            console.print(
                "[dim]  live tools disabled; Kubernetes, SDM, and database helpers "
                "are excluded so scores are reproducible[/dim]"
            )

        async def _scored() -> Any:
            return await harness.run_model_cases_contained(cases, allow_live=allow_live)

        model_report = _run(_scored())
        console.print(model_report.summary())
        report.results.extend(model_report.results)

    # Persist unconditionally. ADR-002 section 5: a figure nobody can trace to a
    # stored run is aspirational, so every run gets an id.
    run_id = harness.persist(
        report,
        suite="regression" if not corpus else str(corpus),
        corpus_dir=corpus,
        offline=not allow_live,
    )
    if run_id:
        console.print(f"\n[dim]run {run_id} stored; inspect with 'mimir eval runs'[/dim]")

    for result in report.results:
        # Pending cases are already reported in their own section. Printing them
        # as failures too trains the reader to skim past red lines.
        if not result.passed and not result.pending:
            console.print(
                Text(f"  FAIL {result.case_id}: {result.detail}", style="red")
            )

    if out:
        out.write_text(report.to_json(), encoding="utf-8")
        console.print(f"\n[green]wrote {out}[/green]")

    if not report.acceptable:
        console.print(
            Text(
                "\nADR 21.3 gate failed: unapproved mutations or dangerous proposals occurred.",
                style="bold red",
            )
        )
        raise typer.Exit(1)


@app.command()
def tools(
    capability: str | None = typer.Option(None, "--capability", "-c"),
    schema: str | None = typer.Option(None, "--schema", help="Print one tool's JSON schema."),
) -> None:
    """List the typed helper tools available to the council."""
    from mimir.tools.base import Capability, load_all_tools

    registry = load_all_tools()
    if schema:
        spec = registry.get(schema)
        if spec is None:
            console.print(f"[red]unknown tool {schema}[/red]")
            raise typer.Exit(1)
        console.print_json(json.dumps(spec.openai_schema()))
        return

    caps = [Capability(capability)] if capability else None
    table = Table(box=None, header_style="dim")
    table.add_column("tool")
    table.add_column("cap", style="cyan")
    table.add_column("risk")
    table.add_column("description", overflow="fold")
    for spec in registry.select(capabilities=caps):
        table.add_row(
            spec.name,
            spec.capability.value,
            str(render.risk_text(spec.risk)),
            " ".join(spec.description.split())[:90],
        )
    console.print(table)


def run() -> None:
    app()


if __name__ == "__main__":
    run()


eval_app = typer.Typer(help="Evaluation history and model comparison.")
app.add_typer(eval_app, name="eval")


@eval_app.command("runs")
def eval_runs(limit: int = typer.Option(20, "--limit", "-l")) -> None:
    """List stored evaluation runs (ADR-002 section 5)."""
    from mimir.persistence.repositories import EvalRepository

    runs = EvalRepository().list_runs(limit=limit)
    if not runs:
        console.print(
            "[yellow]no evaluation runs stored[/yellow]\n"
            "[dim]until a run exists, MIMIR has no measured accuracy. "
            "Run 'mimir evaluate'.[/dim]"
        )
        return
    table = Table(box=None, header_style="dim")
    table.add_column("run")
    table.add_column("suite")
    table.add_column("model", style="cyan")
    table.add_column("passed")
    table.add_column("unsupported claims")
    for run in runs:
        meta = run.get("metadata") or {}
        rate = meta.get("unsupported_claim_rate")
        table.add_row(
            run["id"],
            run.get("suite", ""),
            run.get("model_alias") or "-",
            f"{run.get('passed', 0)}/{run.get('total', 0)}",
            "-" if rate is None else f"{rate:.3f}",
        )
    console.print(table)


@eval_app.command("matrix")
def eval_matrix(
    roles: str = typer.Option(
        "classification,deep_investigation,final_synthesis",
        "--roles",
        help="Comma-separated task classes to vary.",
    ),
    models: str = typer.Option(..., "--models", help="Comma-separated model aliases."),
    corpus: Path | None = typer.Option(None, "--corpus"),
    limit: int | None = typer.Option(None, "--limit", help="Cap the combinations run."),
) -> None:
    """Benchmark role-to-model assignments, not models in isolation.

    The interesting question is not which model scores highest alone but which
    assignment of models to roles performs best together. A strong planner
    paired with a weak synthesiser can lose to two mediocre models that agree
    on format.

    Combinations grow as len(models) ** len(roles), so --limit exists to keep a
    sweep finishable. What is dropped is reported rather than silently skipped.
    """
    from mimir.eval.matrix import run_matrix

    role_list = [r.strip() for r in roles.split(",") if r.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    total = len(model_list) ** len(role_list)
    console.print(
        f"[dim]{len(role_list)} role(s) x {len(model_list)} model(s) = "
        f"{total} combination(s)[/dim]"
    )
    if limit and total > limit:
        console.print(f"[yellow]capping at {limit}; {total - limit} not run[/yellow]")

    rows = _run(run_matrix(role_list, model_list, corpus=corpus, limit=limit))
    if not rows:
        console.print("[red]no combinations completed[/red]")
        raise typer.Exit(1)

    table = Table(box=None, header_style="dim")
    for role in role_list:
        table.add_column(role, style="cyan")
    table.add_column("passed")
    table.add_column("unsupported")
    table.add_column("mean s")
    for row in rows:
        table.add_row(
            *[row["assignment"][role] for role in role_list],
            f"{row['passed']}/{row['total']}",
            "-" if row["unsupported_claim_rate"] is None
            else f"{row['unsupported_claim_rate']:.3f}",
            f"{row['mean_duration_s']:.1f}",
        )
    console.print(table)
    console.print(
        "\n[dim]ranked by passed, then by unsupported claim rate. "
        "Each row is a stored run; see 'mimir eval runs'.[/dim]"
    )


@eval_app.command("compare")
def eval_compare(
    baseline: str = typer.Argument(..., help="Run id to compare against."),
    candidate: str = typer.Argument(..., help="Run id under test."),
) -> None:
    """Compare two runs across every dimension, not just the pass count.

    Reports confounds first. A candidate that beat the baseline on a different
    corpus, different prompts, or with live infrastructure enabled has not
    beaten it at all, and the headline number would hide that.
    """
    from mimir.eval.provenance import comparable
    from mimir.persistence.repositories import EvalRepository

    repo = EvalRepository()
    left, right = repo.get_run(baseline), repo.get_run(candidate)
    for run_id, run in ((baseline, left), (candidate, right)):
        if run is None:
            console.print(f"[red]unknown run {run_id}[/red]")
            raise typer.Exit(1)

    left_meta = left.get("metadata") or {}
    right_meta = right.get("metadata") or {}
    problems = comparable(left_meta.get("provenance", {}), right_meta.get("provenance", {}))

    if problems:
        console.print(Text("confounds", style="bold yellow"))
        for problem in problems:
            console.print(Text(f"  - {problem}", style="yellow"))
        console.print(
            "[dim]differences beyond the thing under test make this a "
            "coincidence rather than an experiment[/dim]\n"
        )
    else:
        console.print(
            "[green]provenance matches: corpus, prompts, skills, commit, "
            "and offline mode are identical[/green]\n"
        )

    def rate(meta: dict[str, Any]) -> str:
        value = meta.get("unsupported_claim_rate")
        return "-" if value is None else f"{value:.3f}"

    table = Table(box=None, header_style="dim")
    table.add_column("dimension")
    table.add_column(baseline[:16])
    table.add_column(candidate[:16])
    table.add_column("delta")

    passed_l, passed_r = left.get("passed", 0), right.get("passed", 0)
    table.add_row(
        "cases passed",
        f"{passed_l}/{left.get('total', 0)}",
        f"{passed_r}/{right.get('total', 0)}",
        _delta(passed_r - passed_l, higher_is_better=True),
    )

    rl, rr = left_meta.get("unsupported_claim_rate"), right_meta.get("unsupported_claim_rate")
    table.add_row(
        "unsupported claims",
        rate(left_meta),
        rate(right_meta),
        _delta(round((rr - rl), 3), higher_is_better=False)
        if rl is not None and rr is not None
        else "-",
    )
    for label, key in (
        ("unapproved mutations", "unapproved_mutations"),
        ("dangerous proposals", "dangerous_proposals"),
        ("audit gaps", "audit_gaps"),
    ):
        lv, rv = left_meta.get(key, 0), right_meta.get(key, 0)
        style = "red" if rv else "green"
        table.add_row(label, str(lv), Text(str(rv), style=style), "must be 0")
    console.print(table)

    # Failure categories are where the useful answer usually is: which defects
    # disappeared tells you what the change actually fixed.
    left_fail = left_meta.get("failure_breakdown") or {}
    right_fail = right_meta.get("failure_breakdown") or {}
    if left_fail or right_fail:
        console.print()
        failures = Table(box=None, header_style="dim")
        failures.add_column("failure category")
        failures.add_column(baseline[:16])
        failures.add_column(candidate[:16])
        failures.add_column("change")
        for category in sorted(set(left_fail) | set(right_fail)):
            lv, rv = left_fail.get(category, 0), right_fail.get(category, 0)
            failures.add_row(category, str(lv), str(rv), _delta(rv - lv, higher_is_better=False))
        console.print(failures)

    lp = left_meta.get("provenance", {}).get("models", {})
    rp = right_meta.get("provenance", {}).get("models", {})
    if lp or rp:
        console.print()
        models = Table(box=None, header_style="dim")
        models.add_column("role")
        models.add_column(baseline[:16])
        models.add_column(candidate[:16])
        for alias in sorted(set(lp) | set(rp)):
            models.add_row(alias, _model_label(lp.get(alias)), _model_label(rp.get(alias)))
        console.print(models)


def _delta(value: float, *, higher_is_better: bool) -> Text:
    if value == 0:
        return Text("same", style="dim")
    improved = value > 0 if higher_is_better else value < 0
    return Text(f"{value:+g}", style="green" if improved else "red")


def _model_label(identity: dict[str, Any] | None) -> str:
    if not identity:
        return "-"
    bits = [identity.get("name", "?")]
    if identity.get("quantisation"):
        bits.append(identity["quantisation"])
    if identity.get("digest"):
        bits.append(identity["digest"][:12])
    elif not identity.get("resolved"):
        bits.append("(unresolved)")
    return " ".join(bits)
