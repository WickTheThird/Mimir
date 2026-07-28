"""Terminal rendering (ADR 14.2).

The CLI UX requirements are explicit: stream responses, render commands clearly,
show tool calls, show output, allow approve/reject/edit, show current context,
stay copy friendly, and never hide failures.

That last one shapes this module. Errors are rendered in full, tool failures are
shown rather than swallowed, and a low-confidence answer is labelled as such.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from mimir.models.approval import ApprovalRequest
from mimir.models.command import ExecutionRecord, ProposedCommand, RiskClass
from mimir.models.evidence import Evidence, EvidenceKind
from mimir.models.specialist import FinalAnswer
from mimir.models.state import InvestigationState

RISK_STYLE: dict[str, str] = {
    "R0": "dim",
    "R1": "green",
    "R2": "yellow",
    "R3": "dark_orange",
    "R4": "bold red",
}

SPECIALIST_STYLE = "cyan"


def make_console(no_color: bool = False) -> Console:
    return Console(soft_wrap=False, no_color=no_color, highlight=False)


def risk_text(risk: RiskClass | str | None) -> Text:
    value = risk.value if isinstance(risk, RiskClass) else (risk or "?")
    return Text(value, style=RISK_STYLE.get(value, "white"))


def render_context(state: InvestigationState) -> Panel | None:
    lines = state.environment.render_lines()
    if not lines:
        return None
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    for label, value in state.environment.render_pairs():
        table.add_row(label, value)
    return Panel(table, title="context", border_style="dim", title_align="left")


def render_command(command: ProposedCommand) -> Panel:
    """The ADR 13.3 pre-execution display."""
    body: list[Any] = [Syntax(command.display, "bash", theme="ansi_dark", word_wrap=True)]

    details = Table.grid(padding=(0, 2))
    details.add_column(style="dim", width=16)
    details.add_column()
    for label, value in command.context.render_pairs():
        details.add_row(label, value)
    if command.purpose:
        details.add_row("reason", command.purpose)
    if command.expected_effect:
        details.add_row("expected effect", command.expected_effect)

    assessment = command.assessment
    if assessment:
        details.add_row(
            "risk", Text.assemble(risk_text(assessment.risk), f"  {assessment.summary}")
        )
        for reason in assessment.reasons[:4]:
            details.add_row("", Text(f"- {reason}", style="dim"))
        if assessment.production_target:
            details.add_row("", Text("this target matches a production pattern", style="bold red"))
        if not assessment.reversible:
            details.add_row("", Text("not automatically reversible", style="bold red"))
        details.add_row(
            "rollback", assessment.rollback_hint or "none available; verify manually"
        )
    body.append(details)

    style = RISK_STYLE.get(assessment.risk.value if assessment else "R1", "white")
    return Panel(Group(*body), title="proposed command", border_style=style, title_align="left")


def render_approval(request: ApprovalRequest) -> Panel:
    return Panel(
        Text(request.prompt),
        title=f"approval required [{request.assessment.risk.value}]",
        border_style=RISK_STYLE.get(request.assessment.risk.value, "yellow"),
        title_align="left",
    )


def render_execution(record: ExecutionRecord, max_lines: int = 25) -> Panel:
    header = Text.assemble(
        ("$ ", "dim"),
        (record.display, "bold"),
        ("  ", ""),
        (f"exit {record.exit_code}", "green" if record.ok else "red"),
        ("  ", ""),
        (f"{record.duration_s:.2f}s", "dim"),
    )
    output = record.combined_output()
    lines = output.splitlines()
    shown = lines[:max_lines]
    if len(lines) > max_lines:
        shown.append(f"... {len(lines) - max_lines} more lines (artifact {record.artifact_ref})")
    body = Group(header, Text("\n".join(shown) or "(no output)", style="dim"))
    return Panel(
        body,
        border_style="green" if record.ok else "red",
        title=record.outcome.value,
        title_align="left",
    )


def render_evidence(items: list[Evidence], limit: int = 12) -> Table:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("", width=3)
    table.add_column("claim", overflow="fold")
    table.add_column("source", style="dim", overflow="fold")
    for item in items[:limit]:
        marker = Text("+" if item.supports else "!", style="green" if item.supports else "red")
        kind = {
            EvidenceKind.OBSERVED: "",
            EvidenceKind.INFERRED: " (inferred)",
            EvidenceKind.HYPOTHESIS: " (hypothesis)",
        }[item.kind]
        citation = "; ".join(c.render() for c in item.citations) or item.source_id
        table.add_row(marker, item.claim + kind, citation)
    return table


def render_answer(answer: FinalAnswer, confidence: float) -> Group:
    """Evidence-first layout: conclusion, then what supports it, then what does not.

    The separation of observed from inferred from unverified is the ADR 2 and
    21.3 requirement, so it is structural here rather than left to prose.
    """
    blocks: list[Any] = [Markdown(answer.answer)]

    if answer.observed_facts:
        blocks.append(Rule("observed", style="dim"))
        blocks.append(_bullets(answer.observed_facts, "green"))
    if answer.inferences:
        blocks.append(Rule("inferred", style="dim"))
        blocks.append(_bullets(answer.inferences, "yellow"))
    if answer.unverified:
        blocks.append(Rule("unverified", style="dim"))
        blocks.append(_bullets(answer.unverified, "dark_orange"))
    if answer.disagreements:
        blocks.append(Rule("disagreement between specialists", style="red"))
        blocks.append(_bullets(answer.disagreements, "red"))
    if answer.proposed_commands:
        blocks.append(Rule("proposed commands", style="dim"))
        for command in answer.proposed_commands:
            blocks.append(Syntax(command, "bash", theme="ansi_dark", word_wrap=True))
    if answer.next_steps:
        blocks.append(Rule("next steps", style="dim"))
        blocks.append(_bullets(answer.next_steps, "cyan"))
    if answer.citations:
        blocks.append(Rule("citations", style="dim"))
        blocks.append(_bullets(answer.citations[:12], "dim"))

    label, style = confidence_label(confidence)
    blocks.append(Text(f"\nconfidence: {confidence:.2f} ({label})", style=style))
    return Group(*blocks)


def confidence_label(confidence: float) -> tuple[str, str]:
    if confidence >= 0.75:
        return "well supported", "green"
    if confidence >= 0.5:
        return "reasonable, verify the key claims", "yellow"
    if confidence >= 0.3:
        return "weak, treat as a lead not a conclusion", "dark_orange"
    return "very weak, gather more evidence before acting", "bold red"


def _bullets(items: list[str], style: str) -> Text:
    text = Text()
    for item in items:
        text.append("  - ", style="dim")
        text.append(item.strip() + "\n", style=style)
    return text


def render_session_summary(state: InvestigationState) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", width=18)
    table.add_column()
    table.add_row("session", state.session_id)
    table.add_row("task type", state.task_type.value if state.task_type else "unknown")
    table.add_row("duration", f"{state.duration_s:.1f}s")
    table.add_row("evidence", str(len(state.evidence)))
    table.add_row("commands run", str(len(state.commands_executed)))
    table.add_row("specialists", ", ".join(r.specialist.value for r in state.reports) or "none")
    if state.selected_skills:
        table.add_row("skills", ", ".join(state.selected_skills))
    if state.risks:
        table.add_row("risks", "\n".join(state.risks))
    if state.error:
        table.add_row("error", Text(state.error, style="red"))
    return table


def render_hypotheses(state: InvestigationState) -> Table | None:
    if not state.hypotheses and not state.rejected_hypotheses:
        return None
    table = Table(box=None, show_header=True, header_style="dim")
    table.add_column("likelihood", width=10)
    table.add_column("status", width=12)
    table.add_column("hypothesis", overflow="fold")
    table.add_column("next check", overflow="fold", style="dim")
    for hypothesis in state.ranked_hypotheses():
        table.add_row(
            f"{hypothesis.likelihood:.2f}",
            hypothesis.status.value,
            hypothesis.statement,
            hypothesis.next_check or "",
        )
    for hypothesis in state.rejected_hypotheses:
        table.add_row(
            Text(f"{hypothesis.likelihood:.2f}", style="dim"),
            Text("rejected", style="dim"),
            Text(hypothesis.statement, style="strike dim"),
            Text(hypothesis.rejected_reason or "", style="dim"),
        )
    return table
