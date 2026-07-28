"""Executor and approval-gate tests (ADR 13).

The property under test throughout: nothing above the auto-execute ceiling runs
without an explicit decision, and an operator edit cannot raise the risk of what
was approved.
"""

from __future__ import annotations

import asyncio

import pytest

from mimir.models.approval import ApprovalStatus
from mimir.models.command import CommandKind, CommandOutcome, ProposedCommand, TargetContext
from mimir.safety.approvals import ApprovalBroker
from mimir.tools.exec import CommandExecutor, ExecutionOptions


@pytest.fixture
def executor(settings, artifacts):
    return CommandExecutor(settings, approvals=ApprovalBroker(settings), artifacts=artifacts)


async def test_read_only_command_runs(executor):
    record = await executor.run(ProposedCommand(argv=["echo", "hello"]))
    assert record.outcome == CommandOutcome.SUCCESS
    assert "hello" in record.stdout
    assert record.artifact_ref, "output should be stored for later analysis"


async def test_denied_command_never_spawns(executor):
    record = await executor.run(ProposedCommand(argv=["rm", "-rf", "/tmp/does-not-exist"]))
    assert record.outcome == CommandOutcome.DENIED
    assert record.exit_code is None


async def test_missing_binary_reports_clearly(executor):
    record = await executor.run(ProposedCommand(argv=["definitely-not-a-real-binary-xyz"]))
    assert record.outcome == CommandOutcome.ERROR
    assert "not found on PATH" in (record.error or "")


async def test_timeout_terminates_the_process(executor):
    record = await executor.run(
        ProposedCommand(argv=["sleep", "30"], timeout_s=1.0),
        options=ExecutionOptions(timeout_s=1.0),
    )
    assert record.outcome == CommandOutcome.TIMEOUT
    assert "exceeded" in (record.error or "")


async def test_gated_command_waits_and_rejects(executor):
    """A rejected approval must leave the command unexecuted."""
    command = ProposedCommand(
        argv=["kubectl", "-n", "payments", "exec", "api-0", "--", "cat", "/etc/x"],
        kind=CommandKind.KUBECTL,
        context=TargetContext(namespace="payments", pod="api-0", cluster_context="staging"),
    )

    async def reject_when_asked(request):
        await executor.approvals.resolve(
            request.id, ApprovalStatus.REJECTED, reason="not now"
        )

    executor.approvals.add_listener(reject_when_asked)
    record = await executor.run(command)
    assert record.outcome == CommandOutcome.REJECTED
    assert record.exit_code is None


async def test_approved_command_executes(executor):
    command = ProposedCommand(
        argv=["echo", "approved"],
        context=TargetContext(host="localhost"),
    )

    async def approve(request):
        await executor.approvals.resolve(request.id, ApprovalStatus.APPROVED)

    executor.approvals.add_listener(approve)
    record = await executor.run(command, options=ExecutionOptions(require_approval=True))
    assert record.outcome == CommandOutcome.SUCCESS
    assert record.approval_id is not None


async def test_edit_cannot_escalate_risk(executor):
    """An operator approving `echo x` must not end up running `kubectl delete`.

    The edited argv is re-classified from scratch, and a higher class than the
    one reviewed is refused rather than silently executed.
    """
    command = ProposedCommand(argv=["echo", "safe"], context=TargetContext(host="localhost"))

    async def approve_with_escalation(request):
        await executor.approvals.resolve(
            request.id,
            ApprovalStatus.EDITED,
            edited_argv=["kubectl", "-n", "prod", "delete", "deploy", "api", "--all"],
        )

    executor.approvals.add_listener(approve_with_escalation)
    record = await executor.run(command, options=ExecutionOptions(require_approval=True))
    assert record.outcome == CommandOutcome.DENIED
    assert "risk" in (record.error or "").lower()


async def test_approval_expiry_does_not_execute(executor):
    command = ProposedCommand(argv=["echo", "x"], context=TargetContext(host="localhost"))
    record = await executor.run(
        command,
        options=ExecutionOptions(require_approval=True, approval_timeout_s=0.05),
    )
    assert record.outcome in (CommandOutcome.SKIPPED, CommandOutcome.REJECTED)
    assert record.exit_code is None


async def test_secrets_are_redacted_before_storage(executor, artifacts):
    record = await executor.run(
        ProposedCommand(argv=["echo", "password=hunter2supersecret"])
    )
    assert "hunter2supersecret" not in record.stdout
    assert "REDACTED" in record.stdout
    stored = artifacts.read(record.artifact_ref)
    assert "hunter2supersecret" not in stored


async def test_dry_run_never_executes(executor):
    record = await executor.run(
        ProposedCommand(argv=["echo", "nope"]), options=ExecutionOptions(dry_run=True)
    )
    assert record.outcome == CommandOutcome.SKIPPED
    assert record.stdout == ""


async def test_parallel_reads_do_not_serialise(executor):
    commands = [ProposedCommand(argv=["echo", str(i)]) for i in range(6)]
    records = await executor.run_many(commands, concurrency=6)
    assert len(records) == 6
    assert all(r.outcome == CommandOutcome.SUCCESS for r in records)


async def test_evidence_from_execution_is_citable(executor):
    record = await executor.run(ProposedCommand(argv=["echo", "ready 1/1"]))
    evidence = CommandExecutor.to_evidence(record, claim="the pod reports ready")
    assert evidence.source_type.value == "command_output"
    assert evidence.citations and "echo" in evidence.citations[0].render()
    assert evidence.freshness.value == "live"
    assert evidence.supports


async def test_concurrent_approvals_do_not_cross_sessions(settings, artifacts):
    """Two sessions must not see each other's approvals."""
    broker = ApprovalBroker(settings)
    seen: dict[str, list[str]] = {"a": [], "b": []}

    async def listener(request):
        seen.setdefault(request.session_id or "?", []).append(request.id)
        await broker.resolve(request.id, ApprovalStatus.REJECTED)

    broker.add_listener(listener)
    executor = CommandExecutor(settings, approvals=broker, artifacts=artifacts)

    await asyncio.gather(
        executor.run(
            ProposedCommand(argv=["echo", "a"], context=TargetContext(host="h")),
            session_id="a",
            options=ExecutionOptions(require_approval=True),
        ),
        executor.run(
            ProposedCommand(argv=["echo", "b"], context=TargetContext(host="h")),
            session_id="b",
            options=ExecutionOptions(require_approval=True),
        ),
    )
    assert len(seen["a"]) == 1
    assert len(seen["b"]) == 1
    assert seen["a"] != seen["b"]
