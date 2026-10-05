"""Interactive approval prompts for the CLI (ADR 13.3, 14.2)."""

from __future__ import annotations

import asyncio
import shlex

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from mimir.cli.render import render_approval
from mimir.logging import get_logger
from mimir.models.approval import ApprovalRequest, ApprovalStatus
from mimir.safety.approvals import ApprovalBroker

log = get_logger(__name__)


def attach_cli_approvals(broker: ApprovalBroker, console: Console) -> None:
    """Register a listener that prompts on the terminal."""

    async def listener(request: ApprovalRequest) -> None:
        # The prompt blocks on stdin, so it runs in a worker thread to keep the
        decision = await asyncio.to_thread(_prompt, console, request)
        status, argv, reason = decision
        await broker.resolve(
            request.id,
            status,
            decided_by="cli",
            reason=reason,
            edited_argv=argv,
        )

    broker.add_listener(listener)


def _prompt(
    console: Console, request: ApprovalRequest
) -> tuple[ApprovalStatus, list[str] | None, str | None]:
    console.print()
    console.print(render_approval(request))

    if request.assessment.production_target:
        console.print(
            Text(
                "This target matches a production pattern. Read the command again.",
                style="bold red",
            )
        )

    while True:
        choice = Prompt.ask(
            Text.assemble(
                ("approve", "green"),
                (" / ", "dim"),
                ("reject", "red"),
                (" / ", "dim"),
                ("edit", "yellow"),
                (" / ", "dim"),
                ("explain", "cyan"),
            ),
            choices=["a", "r", "e", "x"],
            default="r",
            console=console,
        )
        if choice == "a":
            return ApprovalStatus.APPROVED, None, None
        if choice == "r":
            reason = Prompt.ask("reason (optional)", default="", console=console)
            return ApprovalStatus.REJECTED, None, reason or "rejected by operator"
        if choice == "x":
            _explain(console, request)
            continue

        edited = Prompt.ask(
            "edited command", default=request.command.display, console=console
        )
        try:
            argv = shlex.split(edited)
        except ValueError as exc:
            console.print(Text(f"could not parse: {exc}", style="red"))
            continue
        if not argv:
            console.print(Text("empty command", style="red"))
            continue
        console.print(
            Text(
                "The edited command is re-classified from scratch. "
                "If it turns out riskier than the original it will be refused.",
                style="dim",
            )
        )
        return ApprovalStatus.EDITED, argv, "edited by operator"


def _explain(console: Console, request: ApprovalRequest) -> None:
    assessment = request.assessment
    console.print()
    console.print(Text("why this needs approval", style="bold"))
    for reason in assessment.reasons:
        console.print(Text(f"  - {reason}", style="dim"))
    rules = ", ".join(assessment.matched_rules)
    console.print(Text(f"  matched policy rules: {rules}", style="dim"))
    console.print(
        Text(
            f"  risk {assessment.risk.value}: {assessment.summary}",
            style="dim",
        )
    )
    if assessment.rollback_hint:
        console.print(Text(f"  rollback: {assessment.rollback_hint}", style="dim"))
    else:
        console.print(Text("  no automatic rollback is known for this action", style="yellow"))
    console.print()


class AutoRejectPolicy:
    """Non-interactive fallback."""

    def __init__(self, broker: ApprovalBroker, console: Console | None = None) -> None:
        self.broker = broker
        self.console = console

    def attach(self) -> None:
        async def listener(request: ApprovalRequest) -> None:
            message = (
                "no interactive terminal is attached, so this command was refused "
                "rather than run unattended"
            )
            if self.console:
                self.console.print(Text(f"refused: {request.command.display}", style="yellow"))
                self.console.print(Text(f"  {message}", style="dim"))
            await self.broker.resolve(
                request.id,
                ApprovalStatus.REJECTED,
                decided_by="system",
                reason=message,
            )

        self.broker.add_listener(listener)
