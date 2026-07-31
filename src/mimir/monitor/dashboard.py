"""The live terminal dashboard.

Layout intent: the top half is what MIMIR is doing, the bottom half is what the
machine is doing about it. The two together answer the question that neither
answers alone, which is whether a long run is progressing or merely consuming.

Unknown values render as a dim ``unavailable`` with the reason. There is no
placeholder that could be mistaken for a measurement.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mimir.config import Settings, get_settings
from mimir.monitor import activity as activity_mod
from mimir.monitor import machine as machine_mod
from mimir.monitor import runtime as runtime_mod

GIB = float(1 << 30)

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_PULSE = "\u25d0\u25d3\u25d1\u25d2"

_HISTORY: dict[str, deque[float]] = {}
_FRAME = 0


def _record(key: str, value: float, *, keep: int = 48) -> deque[float]:
    """Append a sample to an in-process ring buffer.

    History lives in the monitor rather than the database because it describes
    the display's own sampling, not MIMIR's behaviour. Persisting it would
    invite it being mistaken for a measurement the system made.
    """
    series = _HISTORY.setdefault(key, deque(maxlen=keep))
    series.append(value)
    return series


def _sparkline(values: Sequence[float], *, width: int = 24) -> Text:
    """A trend, drawn from real samples.

    Scaled to the observed range rather than to a fixed ceiling, so a flat line
    means genuinely flat rather than "too small to see". A single sample draws
    nothing: one point is not a trend, and rendering it as a full bar would
    imply a maximum that was never observed.
    """
    points = list(values)[-width:]
    if len(points) < 2:
        return Text("collecting", style="dim")
    low, high = min(points), max(points)
    span = high - low
    if span <= 0:
        return Text("\u2581" * len(points), style="dim")
    return Text(
        "".join(_SPARK[min(7, int((v - low) / span * 7.999))] for v in points),
        style="cyan",
    )


def _pulse(active: bool) -> Text:
    """Motion only when something is genuinely happening.

    A spinner that turns while the system is idle is an animation pretending to
    be a status. This returns a static marker unless there is real activity to
    report.
    """
    if not active:
        return Text("\u00b7", style="dim")
    return Text(_PULSE[_FRAME % len(_PULSE)], style="yellow bold")


_MODEL_CASE_COUNT: int | None = None


def _model_case_count() -> int:
    """How many corpus cases actually open a session.

    Cached for the life of the process. The corpus is frozen during a run, and
    re-reading and re-parsing every YAML file once a second to render a
    denominator would make the monitor a measurable load on the machine it is
    supposed to be reporting on.
    """
    global _MODEL_CASE_COUNT
    if _MODEL_CASE_COUNT is None:
        try:
            from mimir.eval.harness import EvalHarness

            # EvalCase.deterministic is the harness's own predicate. Counting
            # anything else here would silently disagree with the thing being
            # measured, which is how a denominator ends up meaning nothing.
            cases = EvalHarness.load_corpus()
            _MODEL_CASE_COUNT = sum(1 for case in cases if not case.deterministic)
        except Exception:  # noqa: BLE001 - a broken corpus must not kill the monitor
            _MODEL_CASE_COUNT = 0
    return _MODEL_CASE_COUNT


def _bytes(value: float) -> str:
    if value >= GIB:
        return f"{value / GIB:.1f} GB"
    if value >= 1 << 20:
        return f"{value / (1 << 20):.0f} MB"
    return f"{value:.0f} B"


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _unavailable(reason: str) -> Text:
    return Text(f"unavailable ({reason})" if reason else "unavailable", style="dim")


def _bar(fraction: float, width: int = 18, warn: float = 0.75, crit: float = 0.9) -> Text:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    style = "green" if fraction < warn else ("yellow" if fraction < crit else "red")
    bar = Text("█" * filled, style=style)
    bar.append("░" * (width - filled), style="dim")
    return bar


_ROLE_ABBREVIATIONS = {
    "classification": "classify",
    "deep_investigation": "deep",
    "evidence_verification": "verify",
    "fast_command": "command",
    "final_synthesis": "synthesis",
    "web_synthesis": "web synth",
    "embedding": "embedding",
    "default": "default",
}


def _role_label(role: str) -> str:
    return _ROLE_ABBREVIATIONS.get(role, role.replace("_", " "))


def _kv(table: Table, key: str, value: RenderableType) -> None:
    table.add_row(Text(key, style="dim"), value)


def _grid() -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(width=13, no_wrap=True)
    table.add_column(overflow="fold")
    return table


def render_work(act: activity_mod.Activity, sample: machine_mod.MachineSample) -> Panel:
    if not act.readable:
        return Panel(
            Text(f"activity store unreadable: {act.error}", style="red"),
            title="work",
            border_style="red",
        )

    body = _grid()
    mimir_procs = [p for p in sample.processes if p.label.startswith(("mimir", "python -m mimir"))]
    if mimir_procs:
        lines = Text()
        for proc in mimir_procs[:3]:
            lines.append(f"{proc.label}", style="bold green")
            lines.append(
                f"  pid {proc.pid}  up {_duration(proc.age_s)}  "
                f"cpu {proc.cpu_percent:.0f}%  rss {_bytes(proc.rss_bytes)}\n",
                style="dim",
            )
        _kv(body, "processes", lines)
    else:
        _kv(body, "processes", Text("no MIMIR process running", style="dim"))

    if act.running:
        lines = Text()
        for session in act.running[:3]:
            lines.append(f"{session.task_type or 'investigation'}", style="bold cyan")
            lines.append(
                f"  {session.id[:16]}  {_duration(session.duration_s)} elapsed\n", style="dim"
            )
        _kv(body, "in flight", lines)
    else:
        _kv(body, "in flight", Text("idle", style="dim"))

    if act.sessions:
        latest = act.sessions[0]
        recent = Text()
        recent.append(f"{latest.task_type or latest.status}", style="")
        recent.append(f"  {_duration(latest.age_s)} ago", style="dim")
        if latest.confidence is not None:
            style = "green" if latest.confidence >= 0.5 else "yellow"
            recent.append(f"  confidence {latest.confidence:.2f}", style=style)
        if latest.error:
            recent.append(f"  {latest.error[:40]}", style="red")
        _kv(body, "last session", recent)

    rate = act.throughput_per_min()
    stats = Text()
    stats.append(f"{act.sessions_last_hour} sessions/hour")
    if rate is not None:
        stats.append(f"   {rate:.1f}/min recent", style="dim")
        stats.append("  ", style="")
        stats.append_text(_sparkline(_record("rate", rate), width=14))
    _kv(body, "throughput", stats)

    audit = Text()
    audit.append(f"{act.evidence_total} evidence   ")
    audit.append(f"{act.executions_total} executions   ")
    audit.append(
        f"{act.model_calls_total} model calls",
        style="" if act.model_calls_total else "yellow",
    )
    if act.approvals_pending:
        audit.append(f"   {act.approvals_pending} approvals pending", style="yellow bold")
    _kv(body, "audit trail", audit)

    if act.audit_gap:
        _kv(body, "", Text(act.audit_gap, style="yellow"))

    return Panel(body, title="work", border_style="cyan")


def render_models(state: runtime_mod.RuntimeState) -> Panel:
    if not state.reachable:
        return Panel(
            Group(
                Text(f"runtime unreachable at {state.endpoint}", style="red bold"),
                Text(state.error or "no response", style="dim"),
                Text("model cases cannot run in this state.", style="dim"),
            ),
            title="models",
            border_style="red",
        )

    table = Table.grid(padding=(0, 1))
    table.add_column(width=11, no_wrap=True)
    table.add_column(width=17, no_wrap=True, overflow="ellipsis")
    table.add_column(width=8, no_wrap=True)
    table.add_column(width=10, no_wrap=True)
    table.add_column(width=6, no_wrap=True)
    table.add_row(
        Text("role", style="dim"),
        Text("model", style="dim"),
        Text("state", style="dim"),
        Text("resident", style="dim"),
        Text("ttl", style="dim"),
    )

    # Keyed by normalised tag for the same reason the probe normalises: a
    # profile saying "nomic-embed-text" must find "nomic-embed-text:latest".
    # Without this the row showed no resident size and the model was listed a
    # second time as unassigned.
    resident = {runtime_mod.normalise_tag(m.name): m for m in state.loaded}
    seen: set[str] = set()
    for binding in state.bindings:
        if binding.model in seen and binding.loaded:
            state_text = Text("shared", style="green dim")
        elif not binding.installed:
            state_text = Text("not pulled", style="red")
        elif binding.loaded:
            state_text = Text("loaded", style="green bold")
        else:
            state_text = Text("cold", style="dim")
        seen.add(binding.model)

        model = resident.get(runtime_mod.normalise_tag(binding.model))
        table.add_row(
            Text(_role_label(binding.role)),
            Text(binding.model, style="bold" if binding.loaded else ""),
            state_text,
            Text(
                f"{_bytes(model.vram_bytes)} {'gpu' if model.on_gpu else 'cpu'}"
                if model
                else "-",
                style="dim",
            ),
            Text(_duration(model.ttl_s) if model and model.ttl_s is not None else "-", style="dim"),
        )

    bound = {runtime_mod.normalise_tag(b.model) for b in state.bindings}
    extras = [m for m in state.loaded if runtime_mod.normalise_tag(m.name) not in bound]
    for model in extras:
        table.add_row(
            Text("(unassigned)", style="dim"),
            Text(model.name, style="dim"),
            Text("loaded", style="yellow"),
            Text(_bytes(model.vram_bytes), style="dim"),
            Text(_duration(model.ttl_s) if model.ttl_s is not None else "-", style="dim"),
        )

    parts: list[RenderableType] = [table]

    if state.pulls:
        pull_table = _grid()
        for pull in state.pulls[:3]:
            line = Text()
            line.append_text(_bar(pull.fraction, width=14))
            line.append(f" {pull.fraction * 100:5.1f}%  ", style="")
            line.append(
                f"{_bytes(pull.downloaded_bytes)} / {_bytes(pull.total_bytes)}", style="dim"
            )
            if pull.stale_s > 120:
                line.append(f"  stalled {_duration(pull.stale_s)}", style="red bold")
            _kv(pull_table, "downloading", line)
        parts.append(pull_table)

    footer = Text()
    footer.append(f"{state.endpoint}", style="dim")
    if state.version:
        footer.append(f"  v{state.version}", style="dim")
    footer.append(f"  {_bytes(state.vram_bytes)} resident", style="dim")
    parts.append(footer)

    return Panel(Group(*parts), title="models", border_style="magenta")


def render_machine(sample: machine_mod.MachineSample) -> Panel:
    body = _grid()

    if sample.cpu_percent.known and sample.cpu_percent.value is not None:
        history = _record("cpu", sample.cpu_percent.value)
        line = Text()
        line.append_text(_bar(sample.cpu_percent.value / 100.0))
        line.append(f" {sample.cpu_percent.value:5.1f}%  ", style="")
        line.append(f"{sample.cpu_count} cores", style="dim")
        _kv(body, "cpu", line)
        # A trend answers what an instantaneous reading cannot: whether load is
        # climbing, flat, or was a spike that has already passed.
        _kv(body, "", _sparkline(history))
    else:
        _kv(body, "cpu", _unavailable(sample.cpu_percent.unavailable))

    if sample.ram_total_bytes:
        line = Text()
        line.append_text(_bar(sample.ram_used_bytes / sample.ram_total_bytes))
        line.append(
            f" {_bytes(sample.ram_used_bytes)} / {_bytes(sample.ram_total_bytes)}", style=""
        )
        if sample.swap_used_bytes > GIB:
            line.append(f"   swap {_bytes(sample.swap_used_bytes)}", style="yellow")
        _kv(body, "memory", line)
        _kv(body, "", _sparkline(_record("ram", sample.ram_used_bytes / GIB)))
    else:
        _kv(body, "memory", _unavailable(sample.ram_percent.unavailable))

    if sample.load:
        line = Text()
        one, five, fifteen = sample.load
        per_core = sample.load_per_core
        style = "green"
        if per_core is not None:
            style = "green" if per_core < 0.7 else ("yellow" if per_core < 1.0 else "red")
        line.append(f"{one:.2f}", style=style)
        line.append(f"  {five:.2f}  {fifteen:.2f}", style="dim")
        if per_core is not None:
            line.append(f"   {per_core:.2f} per core", style="dim")
        _kv(body, "load", line)
    else:
        _kv(body, "load", _unavailable("getloadavg failed"))

    thermal_style = {
        "throttled": "red bold",
        "warning": "yellow bold",
        "no warning recorded": "dim",
    }.get(sample.thermal, "dim")
    thermal = Text(sample.thermal or "unavailable", style=thermal_style)
    if sample.thermal_detail:
        thermal.append(f"  {sample.thermal_detail}", style="dim")
    _kv(body, "thermal", thermal)

    # Die temperature and GPU utilisation need powermetrics, which needs root.
    # Saying so is more useful than omitting the row, because otherwise the
    # absence looks like an oversight and someone goes looking for the bug.
    _kv(
        body,
        "temp / gpu",
        Text("needs sudo powermetrics; not sampled", style="dim"),
    )

    runtime_procs = [
        p for p in sample.processes if "ollama" in p.label or "llama" in p.label
    ]
    if runtime_procs:
        lines = Text()
        for proc in runtime_procs[:3]:
            lines.append(f"{proc.label}", style="")
            lines.append(
                f"  cpu {proc.cpu_percent:.0f}%  rss {_bytes(proc.rss_bytes)}\n", style="dim"
            )
        _kv(body, "runtime", lines)

    return Panel(body, title="machine", border_style="blue")


def render_in_flight(
    flight: activity_mod.InFlightEval, act: activity_mod.Activity
) -> Panel:
    body = _grid()

    header = Text()
    # Motion here is the difference between "slow" and "hung": the pulse turns
    # only while a case has been observed advancing recently.
    header.append_text(_pulse(bool(act.council.active_names) or (flight.idle_s or 0) < 60))
    header.append(" IN FLIGHT", style="yellow bold")
    header.append(f"  pid {flight.pid}  {_duration(flight.elapsed_s)} elapsed", style="dim")
    _kv(body, "run", header)

    progress = Text()
    if flight.total:
        progress.append_text(_bar(flight.fraction, width=14, warn=1.1, crit=1.2))
        progress.append(f" {flight.completed}/{flight.total} model cases", style="bold")
    else:
        progress.append(f"{flight.completed} model cases done", style="bold")
        progress.append("  (corpus size unknown)", style="dim")
    _kv(body, "progress", progress)

    pace = Text()
    if flight.mean_case_s is not None:
        pace.append(f"{flight.mean_case_s:.0f}s per case")
    else:
        pace.append("measuring", style="dim")
    eta = flight.eta_s
    if eta is not None:
        pace.append(f"   eta {_duration(eta)}", style="dim")
    _kv(body, "pace", pace)

    # Per-case durations come from the database, not from the display's own
    # sampling, so the trend survives restarting the monitor mid-run. This is
    # the shape that would have made the A1-A3 decline visible while it was
    # happening rather than three runs later.
    durations = [c.duration_s for c in act.live_cases if not c.running]
    if len(durations) >= 2:
        trend = Text()
        trend.append_text(_sparkline(durations, width=22))
        trend.append(f"  {min(durations):.0f}-{max(durations):.0f}s", style="dim")
        _kv(body, "case times", trend)

    idle = flight.idle_s
    if idle is not None:
        # A case can legitimately take minutes. Silence far past the observed
        # mean is the signal worth surfacing, not silence itself.
        threshold = max(180.0, (flight.mean_case_s or 60.0) * 3)
        style = "red bold" if idle > threshold else "dim"
        note = Text(f"{_duration(idle)} since the last case finished", style=style)
        if idle > threshold:
            note.append("  possibly stalled", style="red bold")
        _kv(body, "last case", note)

    health = act.telemetry
    live = Text()
    live.append(
        health.summary, style="green" if health.complete else "yellow bold"
    )
    _kv(body, "telemetry", live)

    _kv(
        body,
        "note",
        Text("scores are written when the run finishes", style="dim"),
    )
    return Panel(body, title="evaluation", border_style="yellow")


def render_evaluation(act: activity_mod.Activity) -> Panel:
    if act.in_flight is not None:
        return render_in_flight(act.in_flight, act)
    run = act.latest_run
    if run is None:
        return Panel(
            Text("no evaluation runs recorded", style="dim"),
            title="evaluation",
            border_style="dim",
        )

    # Schema v2 nests everything a comparison needs under one root. Older runs
    # are flat, and are shown with their version so a reader can tell they
    # predate the containment and tool-surface fixes rather than guessing from
    # which keys happen to be present.
    provenance = run.metadata.get("provenance") or {}
    version = provenance.get("schema_version", 1)
    evaluation = provenance.get("evaluation") or {}
    source = provenance.get("source") or {}
    if version < 2:  # flat layout, fields sat at the metadata top level
        evaluation = {
            "contaminated": run.metadata.get("contaminated"),
            "contaminated_reason": run.metadata.get("contaminated_reason", ""),
            "external_calls": run.metadata.get("external_calls", 0),
            "enabled_tools_hash": run.metadata.get("enabled_tools_hash", ""),
            "offline": provenance.get("offline", run.metadata.get("offline")),
            "corpus_hash": provenance.get("corpus_hash", ""),
        }
        source = {
            "commit": provenance.get("mimir_commit", ""),
            "dirty": provenance.get("mimir_dirty", False),
        }

    body = _grid()
    header = Text()
    header.append(run.id, style="bold")
    header.append(f"  {run.suite}", style="dim")
    if run.completed_at is None:
        header.append("  IN FLIGHT", style="yellow bold")
    else:
        header.append(f"  finished {_duration(time.time() - run.completed_at)} ago", style="dim")
    _kv(body, "run", header)

    score = Text()
    score.append_text(_bar(run.pass_rate, width=14, warn=0.6, crit=0.4))
    score.append(f" {run.passed}/{run.total}", style="bold")
    score.append(f"  {run.pass_rate * 100:.0f}%", style="dim")
    _kv(body, "passed", score)

    contaminated = evaluation.get("contaminated") or run.contaminated
    external = evaluation.get("external_calls") or 0
    if contaminated:
        reason = evaluation.get("contaminated_reason") or run.contamination_reason
        _kv(body, "status", Text(f"CONTAMINATED: {reason}", style="red bold"))
    elif external:
        _kv(body, "status", Text(f"{external} external network call(s)", style="red bold"))
    elif evaluation.get("offline"):
        contained = Text("offline, 0 external calls", style="green")
        _kv(body, "status", contained)

    tools = evaluation.get("enabled_tools_hash")
    surface = Text()
    if tools:
        surface.append(tools, style="")
        count = len(evaluation.get("enabled_tools") or [])
        if count:
            surface.append(f"   {count} tools", style="dim")
    else:
        # An empty fingerprint is not a match with another empty fingerprint.
        # Comparing them as equal is what silently disabled the capability check.
        surface.append("not recorded; run is not comparable", style="red")
    _kv(body, "tool surface", surface)

    observed = run.metadata.get("model_invocations_observed")
    if observed is not None:
        persisted = run.metadata.get("model_calls_persisted", 0)
        complete = run.metadata.get("telemetry_complete", False)
        line = Text()
        line.append(f"{persisted}/{observed} calls persisted  ",
                    style="green" if complete else "yellow bold")
        line.append(
            "complete" if complete else "INCOMPLETE",
            style="green" if complete else "yellow bold",
        )
        _kv(body, "telemetry", line)

        validity = Text()
        quality = run.metadata.get("valid_for_quality_reporting", True)
        efficiency = run.metadata.get("valid_for_efficiency_comparison", complete)
        validity.append("quality ", style="dim")
        validity.append("yes" if quality else "no", style="green" if quality else "red")
        validity.append("   efficiency ", style="dim")
        validity.append(
            "yes" if efficiency else "no", style="green" if efficiency else "yellow"
        )
        _kv(body, "valid for", validity)

    gates = Text()
    unapproved = run.metadata.get("unapproved_mutations", 0)
    dangerous = run.metadata.get("dangerous_proposals", 0)
    for label, value in (("unapproved", unapproved), ("dangerous", dangerous)):
        style = "green" if not value else "red bold"
        gates.append(f"{label} {value}  ", style=style)
    _kv(body, "hard gates", gates)

    commit = source.get("commit")
    if commit:
        line = Text(str(commit), style="dim")
        if source.get("dirty"):
            line.append("  dirty", style="yellow")
        if source.get("changed_during_run"):
            line.append("  SOURCE CHANGED MID-RUN", style="red bold")
        corpus = evaluation.get("corpus_hash")
        if corpus:
            line.append(f"   corpus {corpus}", style="dim")
        line.append(f"   schema v{version}", style="dim" if version >= 2 else "yellow")
        _kv(body, "provenance", line)

    return Panel(
        body,
        title="evaluation",
        border_style="yellow" if run.completed_at is None else "green",
    )


_SPECIALIST_SHORT = {
    # Chosen to fit the column without truncation. "kubernete" is worse than
    # "k8s": an abbreviation reads as deliberate, a chopped word reads as a bug.
    "kubernetes_investigator": "k8s",
    "repository_explorer": "repo",
    "behaviour_verifier": "behaviour",
    "sdm_investigator": "sdm",
    "log_analyst": "logs",
    "web_researcher": "web",
    "memory_curator": "memory",
    "safety_reviewer": "safety",
    "coordinator": "coordinator",
    "synthesis": "synthesis",
}


def render_council(act: activity_mod.Activity) -> Panel:
    """The council graph, drawn from its own telemetry.

    This is not a picture of the model. Ollama exposes no weights, activations
    or attention, so anything resembling one would be decoration presented as
    data. What is real, and what this draws, is MIMIR's own topology: the
    structure comes from the code, the edge weights come from measured calls,
    latency, tool use and evidence.
    """
    council = act.council
    if not council.nodes:
        return Panel(
            Text(
                "no model calls in the last hour\n"
                "the council graph is drawn from telemetry, so it needs traffic",
                style="dim",
            ),
            title="council flow",
            border_style="dim",
        )

    body = Text()
    total = council.total_latency_ms

    def node_line(node: activity_mod.CouncilNode, prefix: str, width: int = 11) -> None:
        label = _SPECIALIST_SHORT.get(node.specialist, node.specialist)[:width]
        body.append(prefix, style="dim")
        body.append_text(_pulse(node.active))
        body.append(" ", style="")
        body.append(f"{label:<{width}}", style="bold yellow" if node.active else "bold")
        body.append(f"{node.calls:>4}x ", style="dim")
        body.append(f"{node.mean_latency_ms / 1000:5.1f}s ", style="")
        body.append_text(_bar(node.total_latency_ms / total, width=5, warn=0.4, crit=0.6))
        body.append(f" {node.tool_calls:>3}t" if node.tool_calls else "   -", style="dim")
        if node.failed:
            body.append(f" {node.failed}!", style="red")
        elif node.active:
            body.append(" <", style="yellow bold")
        body.append("\n")

    entry, workers, exit_node = council.entry, council.workers, council.exit

    # The decorative "question" header and spacer rows were the first thing to
    # go when the panel ran out of height: they carry no measurement, and losing
    # the evidence row and the telemetry footer to make room for them would be
    # trading data for ornament.
    if entry is not None:
        node_line(entry, "  ")
    for index, node in enumerate(workers):
        body.append("  " + ("\u251c\u2500" if index < len(workers) - 1 else "\u2514\u2500"),
                    style="dim")
        body.append_text(_pulse(node.active))
        body.append(" ", style="")
        label = _SPECIALIST_SHORT.get(node.specialist, node.specialist)[:9]
        body.append(f"{label:<9}", style="bold yellow" if node.active else "")
        body.append(f"{node.calls:>4}x ", style="dim")
        body.append(f"{node.mean_latency_ms / 1000:5.1f}s ", style="")
        body.append_text(_bar(node.total_latency_ms / total, width=5, warn=0.4, crit=0.6))
        body.append(f" {node.tool_calls:>3}t" if node.tool_calls else "   -", style="dim")
        if node.failed:
            body.append(f" {node.failed}!", style="red")
        elif node.active:
            body.append(" <", style="yellow bold")
        body.append("\n")
    if exit_node is not None:
        node_line(exit_node, "  ")

    if council.evidence_sources:
        body.append("  evidence ", style="dim")
        body.append(
            "  ".join(f"{name[:13]} {count}" for name, count in council.evidence_sources[:2]),
            style="dim",
        )
        body.append("\n")

    health = act.telemetry
    footer = Text()
    footer.append(health.summary, style="green" if health.complete else "yellow")
    footer.append(f"   {health.model_calls_total} calls recorded", style="dim")
    idle = [
        _SPECIALIST_SHORT.get(n, n)
        for n in ("web_researcher", "sdm_investigator", "memory_curator")
        if council.by_name(n) is None
    ]
    if idle:
        # A specialist that never runs is either correctly unused for this
        # workload or quietly broken, and the graph is where that shows.
        footer.append(f"   never ran: {', '.join(idle)}", style="dim")

    return Panel(
        Group(body, footer),
        title="council flow",
        border_style="green" if health.complete else "yellow",
    )


def render_telemetry(act: activity_mod.Activity) -> Panel:
    """Measured cost per role over the last hour.

    This panel could not exist before the telemetry repair: model_calls held
    zero rows, so per-role latency and token cost were unknowable and the only
    number available was wall clock for a whole investigation.
    """
    health = act.telemetry
    if not act.roles:
        body = Text(
            "no model calls recorded in the last hour"
            if health.model_calls_total
            else "model_calls is empty; per-call telemetry is not being recorded",
            style="dim" if health.model_calls_total else "yellow",
        )
        return Panel(body, title="telemetry", border_style="dim")

    total_ms = sum(r.total_latency_ms for r in act.roles) or 1.0
    table = Table.grid(padding=(0, 1))
    table.add_column(width=18, no_wrap=True)
    table.add_column(width=4, justify="right")
    table.add_column(width=7, justify="right")
    table.add_column(width=12, no_wrap=True)
    table.add_column(width=14, no_wrap=True)
    table.add_row(
        Text("role", style="dim"), Text("n", style="dim"),
        Text("mean", style="dim"), Text("share", style="dim"),
        Text("tokens", style="dim"),
    )
    for role in act.roles[:7]:
        share = role.total_latency_ms / total_ms
        line = Text()
        line.append_text(_bar(share, width=6, warn=0.5, crit=0.7))
        line.append(f" {share * 100:3.0f}%", style="dim")
        flags = Text()
        if role.failed:
            flags.append(f"  {role.failed} failed", style="red")
        if role.retries:
            flags.append(f"  {role.retries} retries", style="yellow")
        table.add_row(
            Text(_role_label(role.role)),
            Text(str(role.calls)),
            Text(f"{role.mean_latency_ms / 1000:.1f}s"),
            line,
            Text.assemble(
                (f"{role.prompt_tokens // 1000}k/{role.completion_tokens}", ""), flags
            ),
        )

    footer = Text()
    footer.append(health.summary, style="green" if health.complete else "yellow")
    footer.append(f"   {health.model_calls_total} calls recorded", style="dim")
    if health.orphaned_rows:
        footer.append(f"   {health.orphaned_rows} orphaned", style="red")

    return Panel(
        Group(table, footer),
        title="telemetry",
        border_style="green" if health.complete else "yellow",
    )


def render_live_cases(act: activity_mod.Activity, limit: int = 14) -> Panel:
    """Cases of the run in flight, as they complete.

    No pass or fail column. Scoring happens in process and is not written until
    the run ends, so any verdict here would be invented. Confidence, evidence,
    tool calls and duration are measured, and are shown instead.
    """
    cases = act.live_cases
    if not cases:
        return Panel(
            Text("no evaluation in flight", style="dim"),
            title="cases",
            border_style="dim",
        )

    table = Table.grid(padding=(0, 1))
    table.add_column(width=3, justify="right")
    table.add_column(width=36, no_wrap=True, overflow="ellipsis")
    table.add_column(width=22, no_wrap=True, overflow="ellipsis")
    table.add_column(width=6, justify="right")
    table.add_column(width=5, justify="right")
    table.add_column(width=5, justify="right")
    table.add_column(width=7, justify="right")
    table.add_row(
        Text("#", style="dim"), Text("case", style="dim"),
        Text("task type", style="dim"), Text("conf", style="dim"),
        Text("ev", style="dim"), Text("tool", style="dim"),
        Text("time", style="dim"),
    )

    shown = cases[-limit:]
    offset = len(cases) - len(shown)
    for index, case in enumerate(shown, start=offset + 1):
        if case.running:
            marker, style = "▸", "yellow bold"
        elif case.error:
            marker, style = "!", "red"
        else:
            marker, style = " ", ""
        confidence = Text("-", style="dim")
        if case.confidence is not None:
            # Low confidence is correct on a trap case and wrong on a
            # locate-the-symbol case, so this is coloured by magnitude only and
            # never labelled good or bad.
            confidence = Text(
                f"{case.confidence:.2f}",
                style="green" if case.confidence >= 0.5
                else ("yellow" if case.confidence >= 0.2 else "red"),
            )
        table.add_row(
            Text(f"{index}{marker}", style=style),
            Text(case.case_id, style=style or "bold"),
            Text(case.task_type or "-", style="dim"),
            confidence,
            Text(str(case.evidence), style="dim"),
            Text(str(case.tool_calls), style="dim"),
            Text(_duration(case.duration_s), style="dim"),
        )

    done = [c for c in cases if not c.running]
    footer = Text()
    if done:
        mean_tools = sum(c.tool_calls for c in done) / len(done)
        zero_tool = sum(1 for c in done if c.tool_calls == 0)
        footer.append(f"{len(done)} complete   ", style="dim")
        footer.append(f"mean {mean_tools:.1f} tool calls   ", style="dim")
        footer.append(
            f"{zero_tool} answered with no tools",
            style="yellow" if zero_tool else "dim",
        )
    return Panel(Group(table, footer), title="cases in flight", border_style="cyan")


def render_series(act: activity_mod.Activity) -> Panel:
    """Repeats of the same experiment, and where they disagree.

    Only runs sharing a corpus, a commit and a model are grouped. Averaging
    across a corpus change or a commit change would produce the mean of two
    different experiments.
    """
    series = act.series
    if not series.runs:
        return Panel(
            Text(
                "no comparable completed runs yet\n"
                "a series needs two or more runs sharing corpus, commit and model",
                style="dim",
            ),
            title="run series",
            border_style="dim",
        )

    header = Table.grid(padding=(0, 1))
    header.add_column(width=4)
    header.add_column(width=20, no_wrap=True)
    header.add_column(width=9, justify="right")
    header.add_column(width=16)
    header.add_row(
        Text("run", style="dim"), Text("id", style="dim"),
        Text("passed", style="dim"), Text("", style="dim"),
    )
    for run in series.runs:
        header.add_row(
            Text(run.label, style="bold"),
            Text(run.run_id, style="dim"),
            Text(f"{run.passed}/{run.total}"),
            _bar(run.pass_rate, width=14, warn=0.6, crit=0.4),
        )

    stats = Text()
    if len(series.runs) >= 2:
        stats.append(f"mean {series.mean:.1f}   ", style="")
        stats.append(
            f"range {min(series.counts)}-{max(series.counts)} "
            f"(spread {series.spread})   ",
            style="yellow bold" if series.spread >= 3 else "dim",
        )
        stats.append(f"sd {series.stdev:.1f}", style="dim")
        rate = series.stability_rate()
        if rate is not None:
            stats.append(f"   stability {rate * 100:.0f}%", style="dim")
    else:
        stats.append("one run so far; a single run is not a measurement", style="dim")

    parts: list[RenderableType] = [header, stats]

    unstable = act.series.unstable_cases()
    if unstable:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=38, no_wrap=True, overflow="ellipsis")
        for _ in series.runs:
            grid.add_column(width=2, justify="center")
        grid.add_column(width=10)
        grid.add_row(
            Text("unstable case", style="dim"),
            *[Text(r.label.lstrip("#"), style="dim") for r in series.runs],
            Text("majority", style="dim"),
        )
        for case_id, outcomes, majority, agreement in unstable[:8]:
            cells = []
            for outcome in outcomes:
                if outcome is None:
                    cells.append(Text("-", style="dim"))
                else:
                    cells.append(
                        Text("P" if outcome else "F", style="green" if outcome else "red")
                    )
            verdict = Text(
                f"{'P' if majority else 'F'} {agreement * 100:.0f}%",
                style="green" if majority else "red",
            )
            grid.add_row(Text(case_id), *cells, verdict)
        parts.append(grid)
        parts.append(
            Text(
                f"{len(unstable)} case(s) changed outcome between identical runs",
                style="yellow",
            )
        )
    elif len(series.runs) >= 2:
        parts.append(Text("every case agreed across all runs", style="green"))

    footer = Text()
    footer.append(f"corpus {series.corpus_hash}   ", style="dim")
    footer.append(f"commit {series.commit}", style="dim")
    parts.append(footer)

    return Panel(Group(*parts), title="run series", border_style="magenta")


def render_events(lines: list[str], source: Path | None) -> Panel:
    if not lines:
        hint = (
            f"no lines in {source}"
            if source
            else "set logging.log_file in config.yaml, or pass --log"
        )
        return Panel(Text(hint, style="dim"), title="events", border_style="dim")

    body = Text()
    for line in lines:
        style = ""
        lowered = line.lower()
        if "error" in lowered or "traceback" in lowered or "blocked" in lowered:
            style = "red"
        elif "warning" in lowered or "failed" in lowered:
            style = "yellow"
        elif "complete" in lowered:
            style = "green"
        body.append(line[:400] + "\n", style=style)
    return Panel(body, title=f"events  {source}" if source else "events", border_style="dim")


def _header(settings: Settings, interval: float) -> Panel:
    line = Text()
    line.append("MIMIR", style="bold white")
    line.append("  monitor", style="dim")
    line.append(f"   {time.strftime('%H:%M:%S')}", style="dim")
    line.append(f"   home {settings.home}", style="dim")
    line.append(f"   refresh {interval:g}s   q to quit", style="dim")
    return Panel(Align.center(line), border_style="dim", padding=(0, 1))


def build(
    settings: Settings,
    log_path: Path | None,
    interval: float,
    *,
    first: bool = False,
    height: int = 0,
) -> Layout:
    """One adaptive layout.

    There is no view flag. The panels that matter depend on what is happening,
    not on what the operator remembered to type: cases appear when an evaluation
    is in flight, the series appears once repeats exist to compare. A flag would
    make the interesting state the one you have to know to ask for.
    """
    global _FRAME
    _FRAME += 1

    sample = machine_mod.sample(interval=0.2 if first else 0.0)
    state = runtime_mod.probe(settings)
    act = activity_mod.collect(settings)

    evaluating = next(
        (p for p in sample.processes if p.label.startswith("mimir evaluate")), None
    )
    if evaluating is not None:
        act.in_flight = activity_mod.in_flight_eval(
            evaluating.pid, evaluating.started_at, _model_case_count(), settings
        )
        act.live_cases = activity_mod.live_cases(evaluating.started_at, settings)
    act.series = activity_mod.collect_series(settings)
    act.council = activity_mod.collect_council(settings)

    lines = activity_mod.tail_log(log_path, lines=6)
    running = act.in_flight is not None
    has_series = bool(act.series.runs)

    rows: list[Layout] = [Layout(_header(settings, interval), size=3, name="header")]

    upper = Layout(name="upper", size=14)
    upper.split_row(
        Layout(render_work(act, sample), name="work"),
        Layout(render_models(state), name="models"),
    )
    rows.append(upper)

    middle = Layout(name="middle", size=14)
    middle.split_row(
        Layout(render_machine(sample), name="machine"),
        Layout(render_evaluation(act), name="evaluation"),
    )
    rows.append(middle)

    lower = Layout(name="lower", size=14 if has_series else 12)
    lower.split_row(
        Layout(render_council(act), name="council"),
        Layout(
            render_series(act) if has_series else render_events(lines, log_path),
            name="series" if has_series else "events",
        ),
    )
    rows.append(lower)

    # Cases only while a run is in flight. An empty case table on an idle
    # machine is a row of nothing that pushes everything useful off screen.
    if running:
        rows.append(Layout(render_live_cases(act), name="cases"))
    elif has_series:
        rows.append(Layout(render_events(lines, log_path), name="events"))

    layout = Layout()
    layout.split_column(*rows)
    return layout


def run(
    *,
    settings: Settings | None = None,
    log_path: Path | None = None,
    interval: float = 2.0,
    once: bool = False,
    console: Console | None = None,
) -> None:
    """Render once, or loop until interrupted."""
    active = settings or get_settings()
    out = console or Console()
    target = log_path or active.observability.log_file

    if once:
        out.print(build(active, target, interval, first=True))
        return

    with Live(
        build(active, target, interval, first=True),
        console=out,
        refresh_per_second=4,
        screen=True,
    ) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(build(active, target, interval))
        except KeyboardInterrupt:
            return
