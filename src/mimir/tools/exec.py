"""The one place where a subprocess is spawned (ADR 9, 13)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import time
from dataclasses import dataclass

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.approval import ApprovalStatus
from mimir.models.command import (
    CommandOutcome,
    ExecutionRecord,
    ProposedCommand,
    RiskClass,
)
from mimir.models.evidence import Citation, Evidence, EvidenceKind, Freshness, SourceType
from mimir.redaction import redact, safe_env_snapshot
from mimir.safety.approvals import ApprovalBroker, get_approval_broker
from mimir.safety.policy import PolicyDecision, PolicyEngine, get_policy_engine
from mimir.tools.artifacts import ArtifactStore, get_artifact_store

log = get_logger(__name__)

# : Environment variables always preserved for child processes.
_ESSENTIAL_ENV = ("PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR")


@dataclass(slots=True)
class ExecutionOptions:
    timeout_s: float | None = None
    max_output_bytes: int | None = None
    require_approval: bool | None = None
    """Override policy."""

    approval_timeout_s: float | None = None
    scrub_env: bool = False
    capture_evidence: bool = True
    dry_run: bool = False


class ExecutionDenied(Exception):
    def __init__(self, record: ExecutionRecord, reason: str) -> None:
        super().__init__(reason)
        self.record = record
        self.reason = reason


class CommandExecutor:
    def __init__(
        self,
        settings: Settings | None = None,
        policy: PolicyEngine | None = None,
        approvals: ApprovalBroker | None = None,
        artifacts: ArtifactStore | None = None,
        hooks: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.policy = policy or get_policy_engine(self.settings)
        self.approvals = approvals or get_approval_broker(self.settings)
        self.artifacts = artifacts or get_artifact_store(self.settings)
        self.hooks = hooks
        self._history: list[ExecutionRecord] = []

    # -- public api -------------------------------------------------------

    async def run(
        self,
        command: ProposedCommand,
        *,
        session_id: str | None = None,
        options: ExecutionOptions | None = None,
    ) -> ExecutionRecord:
        opts = options or ExecutionOptions()
        decision = self.policy.evaluate(command)
        assessment = decision.assessment

        resolved = shutil.which(command.binary)
        if resolved:
            command.metadata["resolved_binary"] = resolved

        if decision.denied:
            record = self._record(
                command,
                session_id,
                CommandOutcome.DENIED,
                error=decision.reason,
                stderr=decision.reason,
            )
            log.warning("execution_denied", command=command.display, reason=decision.reason)
            self._history.append(record)
            return record

        if resolved is None:
            record = self._record(
                command,
                session_id,
                CommandOutcome.ERROR,
                error=f"binary not found on PATH: {command.binary}",
            )
            self._history.append(record)
            return record

        approval_id: str | None = None
        approved_by: str | None = None
        needs_approval = decision.needs_approval or bool(opts.require_approval)
        if needs_approval:
            if self.hooks is not None:
                await self.hooks.on_approval_request(command, assessment)  # type: ignore[attr-defined]
            request = await self.approvals.create(
                command,
                assessment,
                session_id=session_id,
                timeout_s=opts.approval_timeout_s,
            )
            approval_id = request.id
            outcome = await self.approvals.wait(request.id, timeout_s=opts.approval_timeout_s)
            if not outcome.allows_execution:
                record = self._record(
                    command,
                    session_id,
                    CommandOutcome.REJECTED
                    if outcome.status == ApprovalStatus.REJECTED
                    else CommandOutcome.SKIPPED,
                    error=outcome.reason or f"approval {outcome.status.value}",
                    approval_id=approval_id,
                )
                log.info(
                    "execution_not_approved",
                    command=command.display,
                    status=outcome.status.value,
                )
                self._history.append(record)
                return record
            approved_by = outcome.decided_by
            if outcome.status == ApprovalStatus.EDITED and outcome.edited_argv:
                # An edited command is a new proposal and is re-classified from
                command = command.model_copy(update={"argv": outcome.edited_argv})
                recheck = self.policy.evaluate(command)
                if recheck.denied:
                    record = self._record(
                        command,
                        session_id,
                        CommandOutcome.DENIED,
                        error=f"edited command denied: {recheck.reason}",
                        approval_id=approval_id,
                    )
                    self._history.append(record)
                    return record
                if recheck.assessment.risk.rank > assessment.risk.rank:
                    record = self._record(
                        command,
                        session_id,
                        CommandOutcome.DENIED,
                        error=(
                            f"edited command raised risk from {assessment.risk.value} to "
                            f"{recheck.assessment.risk.value}; re-propose it explicitly"
                        ),
                        approval_id=approval_id,
                    )
                    self._history.append(record)
                    return record
                assessment = recheck.assessment

        if opts.dry_run:
            record = self._record(
                command,
                session_id,
                CommandOutcome.SKIPPED,
                error="dry run; command was not executed",
                approval_id=approval_id,
            )
            self._history.append(record)
            return record

        if assessment.risk.rank >= RiskClass.R3.rank and self.hooks is not None:
            await self.hooks.before_mutation(command, assessment)  # type: ignore[attr-defined]

        record = await self._spawn(command, session_id, opts, decision, approval_id, approved_by)

        if assessment.risk.rank >= RiskClass.R3.rank and self.hooks is not None:
            await self.hooks.after_mutation(command, record)  # type: ignore[attr-defined]

        self._history.append(record)
        return record

    async def run_many(
        self,
        commands: list[ProposedCommand],
        *,
        session_id: str | None = None,
        options: ExecutionOptions | None = None,
        concurrency: int = 4,
    ) -> list[ExecutionRecord]:
        """Run read-only commands in parallel (K1 parallel search helper, ADR 8)."""
        ceiling = self.policy.auto_execute_ceiling()
        parallel, serial = [], []
        for command in commands:
            assessment = self.policy.classify_only(command)
            (parallel if assessment.risk.rank <= ceiling.rank else serial).append(command)

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def guarded(cmd: ProposedCommand) -> ExecutionRecord:
            async with semaphore:
                return await self.run(cmd, session_id=session_id, options=options)

        results = list(
            await asyncio.gather(*(guarded(c) for c in parallel), return_exceptions=False)
        )
        for command in serial:
            results.append(await self.run(command, session_id=session_id, options=options))
        return results

    # -- internals --------------------------------------------------------

    async def _spawn(
        self,
        command: ProposedCommand,
        session_id: str | None,
        opts: ExecutionOptions,
        decision: PolicyDecision,
        approval_id: str | None,
        approved_by: str | None,
    ) -> ExecutionRecord:
        timeout = (
            opts.timeout_s
            or command.timeout_s
            or self.settings.safety.command_timeout_s
        )
        max_bytes = opts.max_output_bytes or self.settings.safety.max_output_bytes
        env = self._build_env(command, scrub=opts.scrub_env)
        started = time.time()
        clock = time.perf_counter()

        log.info(
            "execution_start",
            command=command.display,
            risk=decision.assessment.risk.value,
            session_id=session_id,
            timeout_s=timeout,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command.argv,
                stdin=asyncio.subprocess.PIPE if command.stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=command.cwd,
                env=env,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return self._record(
                command,
                session_id,
                CommandOutcome.ERROR,
                error=f"failed to start process: {exc}",
                approval_id=approval_id,
                duration_s=time.perf_counter() - clock,
            )

        outcome = CommandOutcome.SUCCESS
        error: str | None = None
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(command.stdin.encode() if command.stdin else None),
                timeout=timeout,
            )
        except TimeoutError:
            outcome = CommandOutcome.TIMEOUT
            error = f"command exceeded {timeout:.0f}s and was terminated"
            stdout_b, stderr_b = b"", b""
            await self._terminate(process)
        except asyncio.CancelledError:
            await self._terminate(process)
            raise

        duration = time.perf_counter() - clock
        stdout = stdout_b.decode("utf-8", "replace")
        stderr = stderr_b.decode("utf-8", "replace")
        full_output = stdout + (f"\n[stderr]\n{stderr}" if stderr.strip() else "")

        truncated = False
        if len(stdout) > max_bytes:
            stdout = stdout[:max_bytes]
            truncated = True
        if len(stderr) > max_bytes:
            stderr = stderr[:max_bytes]
            truncated = True

        should_redact = self.settings.safety.redact_secrets
        stdout = redact(stdout, enabled=should_redact)
        stderr = redact(stderr, enabled=should_redact)

        exit_code = process.returncode
        if outcome == CommandOutcome.SUCCESS and exit_code not in (0, None):
            outcome = CommandOutcome.FAILED
            error = f"exit code {exit_code}"

        artifact_ref = None
        if full_output.strip():
            artifact = self.artifacts.put(
                full_output,
                kind="command_output",
                session_id=session_id,
                metadata={
                    "command": command.display,
                    "exit_code": exit_code,
                    "risk": decision.assessment.risk.value,
                    "context": command.context.model_dump(exclude_none=True),
                },
            )
            artifact_ref = artifact.ref

        record = self._record(
            command,
            session_id,
            outcome,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
            artifact_ref=artifact_ref,
            error=error,
            approval_id=approval_id,
            approved_by=approved_by,
            started_at=started,
            duration_s=duration,
        )
        log.info(
            "execution_end",
            command=command.display,
            outcome=outcome.value,
            exit_code=exit_code,
            duration_s=round(duration, 3),
            artifact_ref=artifact_ref,
        )
        return record

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
        except (TimeoutError, ProcessLookupError):
            with contextlib.suppress(ProcessLookupError):
                process.kill()

    def _build_env(self, command: ProposedCommand, *, scrub: bool) -> dict[str, str]:
        base = dict(os.environ)
        if scrub:
            patterns = self.settings.sandbox.scrub_env_patterns
            base = {
                k: v
                for k, v in base.items()
                if k in _ESSENTIAL_ENV or not any(p in k.upper() for p in patterns)
            }
        if self.settings.kubernetes.kubeconfig:
            base.setdefault("KUBECONFIG", str(self.settings.kubernetes.kubeconfig))
        base.update(command.env)
        # Keep child processes non-interactive.
        base.setdefault("GIT_TERMINAL_PROMPT", "0")
        base.setdefault("PAGER", "cat")
        base.setdefault("PSQL_PAGER", "cat")
        return base

    def _record(
        self,
        command: ProposedCommand,
        session_id: str | None,
        outcome: CommandOutcome,
        *,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        truncated: bool = False,
        artifact_ref: str | None = None,
        error: str | None = None,
        approval_id: str | None = None,
        approved_by: str | None = None,
        started_at: float | None = None,
        duration_s: float = 0.0,
    ) -> ExecutionRecord:
        assessment = command.assessment
        return ExecutionRecord(
            command_id=command.id,
            session_id=session_id,
            argv=command.argv,
            outcome=outcome,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
            artifact_ref=artifact_ref,
            error=error,
            approval_id=approval_id,
            approved_by=approved_by,
            risk=assessment.risk if assessment else RiskClass.R1,
            context=command.context,
            started_at=started_at or time.time(),
            duration_s=duration_s,
            environment=safe_env_snapshot(
                {
                    k: v
                    for k, v in os.environ.items()
                    if k in ("KUBECONFIG", "AWS_PROFILE", "AWS_REGION", "SHELL")
                }
            ),
        )

    # -- evidence ---------------------------------------------------------

    @staticmethod
    def to_evidence(
        record: ExecutionRecord,
        claim: str,
        *,
        collected_by: str = "executor",
        excerpt_limit: int = 1500,
        confidence: float = 0.9,
    ) -> Evidence:
        """Turn a command result into a citable evidence item (ADR 11.4 rank 1)."""
        return Evidence(
            claim=claim,
            kind=EvidenceKind.OBSERVED,
            source_type=SourceType.COMMAND_OUTPUT,
            source_id=record.display,
            excerpt=record.combined_output(excerpt_limit),
            citations=[
                Citation(
                    source_type=SourceType.COMMAND_OUTPUT,
                    locator=record.display,
                    retrieved_at=record.started_at,
                    title=f"exit {record.exit_code}",
                )
            ],
            collected_at=record.started_at,
            freshness=Freshness.LIVE,
            confidence=confidence if record.ok else min(confidence, 0.5),
            supports=record.ok,
            collected_by=collected_by,
            artifact_ref=record.artifact_ref,
            structured={
                "exit_code": record.exit_code,
                "outcome": record.outcome.value,
                "duration_s": round(record.duration_s, 3),
                **record.context.model_dump(exclude_none=True),
            },
        )

    @property
    def history(self) -> list[ExecutionRecord]:
        return list(self._history)

    def history_for(self, session_id: str | None) -> list[ExecutionRecord]:
        """Execution records belonging to one session."""
        if session_id is None:
            return []
        return [record for record in self._history if record.session_id == session_id]


_executor: CommandExecutor | None = None


def get_executor(settings: Settings | None = None) -> CommandExecutor:
    global _executor
    if _executor is None:
        _executor = CommandExecutor(settings)
    return _executor


def reset_executor() -> None:
    global _executor
    _executor = None
