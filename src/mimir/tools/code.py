"""Code mutation inside a task worktree (ADR-002 section 3).

MIMIR never writes to the operator's working tree. Every tool here operates on
a git worktree created for one task, so an abandoned or wrong change is undone
by deleting a directory rather than by reverse-engineering what was touched.

Risk follows ADR-002 section 4.2, and the escalation that is easiest to miss is
encoded literally: writing a file inside the worktree is R1, but *running* the
repository's tests is R2, because it executes arbitrary code from that
repository including whatever MIMIR just wrote.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from mimir.models.evidence import Citation, Evidence, EvidenceKind, SourceType
from mimir.safety.risk import RiskClass
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool
from mimir.tools.repo import _target_repos
from mimir.worktree import WorktreeError, WorktreeManager, resolve_inside

MAX_BYTES = 400_000
MAX_DIFF = 20_000

_CREDENTIAL_VARS = frozenset({
    "KUBECONFIG", "AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS", "GH_TOKEN",
    "GITHUB_TOKEN", "SDM_ADMIN_TOKEN", "DOCKER_AUTH_CONFIG", "NPM_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SSH_AUTH_SOCK",
})

_CREDENTIAL_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "APIKEY", "API_KEY")
"""Matched against the variable name. Catches the ones nobody thought to list,
which is most of them in any real shell."""


def _manager(ctx: ToolContext) -> WorktreeManager:
    return WorktreeManager(ctx.settings.home)


async def _repo_root(ctx: ToolContext, name: str | None) -> Path:
    repos = await _target_repos(ctx, name)
    if not repos:
        raise ToolError("no repository is configured", code="not_found")
    return Path(repos[0].root).resolve()


def _wrap(exc: WorktreeError) -> ToolError:
    return ToolError(str(exc), code="refused")


class CreateInput(BaseModel):
    task: str = Field(description="Short name for the task, for example 'fix-retry-bounds'.")
    repo: str | None = None
    base: str = Field(default="HEAD", description="Commit or ref to branch from.")


class ListWorktreesInput(BaseModel):
    """No arguments; listing is unconditional."""


class WorktreeRef(BaseModel):
    task: str
    repo: str | None = None


class WriteInput(BaseModel):
    task: str
    path: str = Field(description="Path relative to the worktree root.")
    content: str = Field(description="Full new contents of the file.")
    repo: str | None = None


class TestInput(BaseModel):
    task: str
    command: str = Field(
        description="Test command, for example 'pytest tests/test_claims.py -q'."
    )
    repo: str | None = None
    timeout_s: float = 600.0


@tool(
    "create_task_worktree",
    description=(
        "Create an isolated git worktree and branch for a code task. All edits happen here, "
        "never in the operator's checkout, so the whole task is reversible by discarding it. "
        "Call this before writing any file."
    ),
    capability=Capability.CODE,
    risk=RiskClass.R1,
    tags=("repository", "worktree", "code"),
)
async def create_task_worktree(args: CreateInput, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)

    def work() -> ToolResult:
        try:
            wt = _manager(ctx).create(root, args.task, base=args.base)
        except WorktreeError as exc:
            raise _wrap(exc) from exc
        return ToolResult(
            tool="create_task_worktree",
            summary=(
                f"worktree {wt.branch} at {wt.root} from {wt.base_commit[:12]}"
            ),
            data=wt.as_dict(),
        )

    return await asyncio.to_thread(work)


@tool(
    "list_task_worktrees",
    description=(
        "List the task worktrees that exist, with their branch and whether they have "
        "uncommitted changes."
    ),
    capability=Capability.CODE,
    risk=RiskClass.R0,
    tags=("repository", "worktree"),
)
async def list_task_worktrees(args: ListWorktreesInput, ctx: ToolContext) -> ToolResult:
    def work() -> ToolResult:
        found = _manager(ctx).list()
        return ToolResult(
            tool="list_task_worktrees",
            summary=f"{len(found)} task worktree(s)",
            data={"worktrees": found},
        )

    return await asyncio.to_thread(work)


@tool(
    "write_worktree_file",
    description=(
        "Create or overwrite a file inside a task worktree. Refused if the path resolves "
        "outside the worktree. Use read_file_range first when editing an existing file, "
        "because this replaces the whole file."
    ),
    capability=Capability.CODE,
    risk=RiskClass.R1,
    tags=("repository", "worktree", "code", "write"),
    mutating=True,
)
async def write_worktree_file(args: WriteInput, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)

    def work() -> ToolResult:
        manager = _manager(ctx)
        try:
            wt = manager.find(root, args.task)
            target = resolve_inside(wt.root, args.path)
        except WorktreeError as exc:
            raise _wrap(exc) from exc
        if len(args.content.encode()) > MAX_BYTES:
            raise ToolError(
                f"refusing to write {len(args.content)} bytes; limit is {MAX_BYTES}",
                code="invalid_arguments",
            )
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        lines = args.content.count("\n") + 1
        return ToolResult(
            tool="write_worktree_file",
            summary=f"{'updated' if existed else 'created'} {args.path} ({lines} lines)",
            data={"path": args.path, "worktree": str(wt.root), "created": not existed,
                  "lines": lines},
            evidence=[
                Evidence(
                    claim=f"{args.path} was {'updated' if existed else 'created'} in {wt.branch}",
                    kind=EvidenceKind.OBSERVED,
                    source_type=SourceType.REPOSITORY,
                    source_id=args.path,
                    excerpt=args.content[:400],
                    collected_by="write_worktree_file",
                    citations=[Citation(source_type=SourceType.REPOSITORY,
                                        locator=args.path, path=args.path)],
                )
            ],
        )

    return await asyncio.to_thread(work)


@tool(
    "diff_task_worktree",
    description=(
        "Show what a task worktree changed against the commit it branched from, including "
        "untracked files. This is the review surface for the whole task."
    ),
    capability=Capability.CODE,
    risk=RiskClass.R1,
    tags=("repository", "worktree", "review"),
)
async def diff_task_worktree(args: WorktreeRef, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)

    def work() -> ToolResult:
        manager = _manager(ctx)
        try:
            wt = manager.find(root, args.task)
            body = manager.diff(wt)
            stat = manager.diff(wt, stat=True)
        except WorktreeError as exc:
            raise _wrap(exc) from exc
        truncated = len(body) > MAX_DIFF
        return ToolResult(
            tool="diff_task_worktree",
            summary=(stat.strip().splitlines() or ["no changes"])[-1][:200],
            data={"branch": wt.branch, "base": wt.base_commit,
                  "diff": body[:MAX_DIFF], "stat": stat},
            truncated=truncated,
        )

    return await asyncio.to_thread(work)


@tool(
    "run_worktree_tests",
    description=(
        "Run the repository's tests inside a task worktree. This executes arbitrary code "
        "from the repository, including code MIMIR just wrote, so it is elevated inspection "
        "rather than a read and requires approval."
    ),
    capability=Capability.SANDBOX,
    risk=RiskClass.R2,
    tags=("repository", "worktree", "verification"),
    long_running=True,
)
async def run_worktree_tests(args: TestInput, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)

    def work() -> ToolResult:
        try:
            wt = _manager(ctx).find(root, args.task)
        except WorktreeError as exc:
            raise _wrap(exc) from exc
        # Scrubbed, not stripped. ADR-002 4.1 requires that test code cannot
        # reach the operator's kubeconfig, SDM session or cloud credentials.
        # An earlier version rebuilt PATH from scratch, which also removed the
        # project's own toolchain and made every test command exit 127 - the
        # environment was safe and useless. Credentials are removed by name and
        # by shape; everything else the repository needs to build is kept.
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in _CREDENTIAL_VARS
            and not any(marker in key.upper() for marker in _CREDENTIAL_MARKERS)
        }
        env["HOME"] = str(wt.root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                args.command, shell=True, cwd=str(wt.root), env=env,
                capture_output=True, text=True, timeout=args.timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"tests exceeded {args.timeout_s:.0f}s and were killed", code="timeout"
            ) from None
        tail = (proc.stdout or "")[-4000:] + (proc.stderr or "")[-2000:]
        passed = proc.returncode == 0
        return ToolResult(
            ok=True,
            tool="run_worktree_tests",
            summary=f"{'passed' if passed else 'FAILED'} (exit {proc.returncode}): {args.command}",
            data={"exit_code": proc.returncode, "passed": passed,
                  "output": tail, "command": args.command},
            evidence=[
                Evidence(
                    claim=f"`{args.command}` exited {proc.returncode} in {wt.branch}",
                    kind=EvidenceKind.OBSERVED,
                    source_type=SourceType.COMMAND_OUTPUT,
                    source_id=args.command,
                    excerpt=tail[:600],
                    supports=passed,
                    collected_by="run_worktree_tests",
                )
            ],
        )

    return await asyncio.to_thread(work)


@tool(
    "discard_task_worktree",
    description=(
        "Delete a task worktree and its branch, throwing away every change made in it. "
        "This is how an abandoned or rejected task is undone."
    ),
    capability=Capability.CODE,
    risk=RiskClass.R1,
    tags=("repository", "worktree", "cleanup"),
    mutating=True,
)
async def discard_task_worktree(args: WorktreeRef, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)

    def work() -> ToolResult:
        manager = _manager(ctx)
        try:
            wt = manager.find(root, args.task)
            manager.discard(wt)
        except WorktreeError as exc:
            raise _wrap(exc) from exc
        return ToolResult(
            tool="discard_task_worktree",
            summary=f"discarded {wt.branch} and everything in it",
            data={"branch": wt.branch, "path": str(wt.root)},
        )

    return await asyncio.to_thread(work)


__all__ = [
    "create_task_worktree",
    "diff_task_worktree",
    "discard_task_worktree",
    "list_task_worktrees",
    "run_worktree_tests",
    "write_worktree_file",
]
