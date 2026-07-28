"""Approval broker (ADR 13.3, 14.2, 15).

An approval is raised by whichever component hit a gated command, and resolved
by whichever interface owns the session. The broker is an in-process async
rendezvous with a persistence hook, so the CLI can prompt inline while the web
UI resolves the same approval over HTTP.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.approval import ApprovalDecision, ApprovalRequest, ApprovalStatus
from mimir.models.command import ProposedCommand, RiskAssessment

log = get_logger(__name__)

ApprovalListener = Callable[[ApprovalRequest], Awaitable[None]]


def render_approval_prompt(
    command: ProposedCommand, assessment: RiskAssessment
) -> str:
    """The mandatory pre-execution display (ADR 13.3)."""
    lines = [
        "APPROVAL REQUIRED",
        "",
        f"  command          {command.display}",
        f"  resolved binary  {command.metadata.get('resolved_binary', command.binary)}",
    ]
    lines.extend("  " + line for line in command.context.render_lines())
    if command.purpose:
        lines.append(f"  reason           {command.purpose}")
    if command.expected_effect:
        lines.append(f"  expected effect  {command.expected_effect}")
    lines.append(f"  risk class       {assessment.risk.value} - {assessment.summary}")
    for reason in assessment.reasons:
        lines.append(f"                   - {reason}")
    if assessment.production_target:
        lines.append("  WARNING          this target matches a production pattern")
    if not assessment.reversible:
        lines.append("  WARNING          this action is not automatically reversible")
    lines.append(
        f"  rollback         {assessment.rollback_hint or 'none available; verify manually'}"
    )
    return "\n".join(lines)


class ApprovalBroker:
    """Tracks pending approvals and blocks callers until a decision arrives."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pending: dict[str, ApprovalRequest] = {}
        self._futures: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._decisions: dict[str, ApprovalDecision] = {}
        self._listeners: list[ApprovalListener] = []
        self._lock = asyncio.Lock()

    # -- listeners --------------------------------------------------------

    def add_listener(self, listener: ApprovalListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    async def _notify(self, request: ApprovalRequest) -> None:
        if not self._listeners:
            return
        # Every listener must run even if one raises; a failed UI notification
        # must not strand the approval.
        await asyncio.gather(
            *(listener(request) for listener in list(self._listeners)),
            return_exceptions=True,
        )

    # -- lifecycle --------------------------------------------------------

    async def create(
        self,
        command: ProposedCommand,
        assessment: RiskAssessment,
        *,
        session_id: str | None = None,
        requested_by: str = "mimir",
        timeout_s: float | None = None,
    ) -> ApprovalRequest:
        timeout = timeout_s if timeout_s is not None else self.settings.safety.approval_timeout_s
        request = ApprovalRequest(
            session_id=session_id,
            command=command,
            assessment=assessment,
            prompt=render_approval_prompt(command, assessment),
            expires_at=time.time() + timeout if timeout else None,
            requested_by=requested_by,
        )
        async with self._lock:
            self._pending[request.id] = request
            self._futures[request.id] = asyncio.get_running_loop().create_future()
        log.info(
            "approval_requested",
            approval_id=request.id,
            session_id=session_id,
            risk=assessment.risk.value,
            command=command.display,
        )
        await self._notify(request)
        return request

    async def wait(
        self, approval_id: str, timeout_s: float | None = None
    ) -> ApprovalDecision:
        """Block until decided, rejected, or expired."""
        future = self._futures.get(approval_id)
        if future is None:
            existing = self._decisions.get(approval_id)
            if existing:
                return existing
            raise KeyError(f"unknown approval: {approval_id}")

        request = self._pending.get(approval_id)
        timeout = timeout_s
        if timeout is None and request and request.expires_at:
            timeout = max(1.0, request.expires_at - time.time())

        started = time.perf_counter()
        try:
            decision = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            decision = ApprovalDecision(
                approval_id=approval_id,
                status=ApprovalStatus.EXPIRED,
                decided_by="system",
                reason="approval window elapsed with no decision",
            )
            await self._finalise(approval_id, decision)
        log.info(
            "approval_resolved",
            approval_id=approval_id,
            status=decision.status.value,
            waited_s=round(time.perf_counter() - started, 2),
        )
        return decision

    async def resolve(
        self,
        approval_id: str,
        status: ApprovalStatus,
        *,
        decided_by: str = "user",
        reason: str | None = None,
        edited_argv: list[str] | None = None,
    ) -> ApprovalDecision:
        decision = ApprovalDecision(
            approval_id=approval_id,
            status=status,
            decided_by=decided_by,
            reason=reason,
            edited_argv=edited_argv,
        )
        await self._finalise(approval_id, decision)
        return decision

    async def approve(self, approval_id: str, **kwargs: object) -> ApprovalDecision:
        return await self.resolve(approval_id, ApprovalStatus.APPROVED, **kwargs)  # type: ignore[arg-type]

    async def reject(self, approval_id: str, **kwargs: object) -> ApprovalDecision:
        return await self.resolve(approval_id, ApprovalStatus.REJECTED, **kwargs)  # type: ignore[arg-type]

    async def _finalise(self, approval_id: str, decision: ApprovalDecision) -> None:
        async with self._lock:
            request = self._pending.pop(approval_id, None)
            future = self._futures.pop(approval_id, None)
            self._decisions[approval_id] = decision
            if request is not None:
                request.status = decision.status
        if future is not None and not future.done():
            future.set_result(decision)

    # -- inspection -------------------------------------------------------

    def pending(self, session_id: str | None = None) -> list[ApprovalRequest]:
        items = [
            r
            for r in self._pending.values()
            if session_id is None or r.session_id == session_id
        ]
        return sorted(items, key=lambda r: r.created_at)

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._pending.get(approval_id)

    def decision(self, approval_id: str) -> ApprovalDecision | None:
        return self._decisions.get(approval_id)

    async def cancel_session(self, session_id: str) -> int:
        cancelled = 0
        for request in self.pending(session_id):
            await self.resolve(
                request.id,
                ApprovalStatus.CANCELLED,
                decided_by="system",
                reason="session cancelled",
            )
            cancelled += 1
        return cancelled


_broker: ApprovalBroker | None = None


def get_approval_broker(settings: Settings | None = None) -> ApprovalBroker:
    global _broker
    if _broker is None:
        _broker = ApprovalBroker(settings)
    return _broker


def reset_approval_broker() -> None:
    global _broker
    _broker = None


class AutoApprovalBroker(ApprovalBroker):
    """Test and headless helper. Never enabled by configuration.

    Used only by the evaluation harness where every gated command is expected to
    be refused or auto-decided deterministically.
    """

    def __init__(
        self, decision: ApprovalStatus = ApprovalStatus.REJECTED, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.auto_status = decision

    async def create(self, *args: object, **kwargs: object) -> ApprovalRequest:  # type: ignore[override]
        request = await super().create(*args, **kwargs)  # type: ignore[arg-type]
        await self.resolve(
            request.id, self.auto_status, decided_by="auto", reason="headless mode"
        )
        return request
