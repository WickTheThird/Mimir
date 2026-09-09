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
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from mimir.logging import get_logger
from mimir.models.evidence import Citation, Evidence, EvidenceKind, SourceType
from mimir.safety.risk import RiskClass
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool
from mimir.tools.repo import _target_repos
from mimir.worktree import WorktreeError, WorktreeManager, resolve_inside

log = get_logger(__name__)

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


def _verify(ctx: ToolContext, root: Path, relative: str, *, updated: str,
            original: str | None, baseline=None):
    """Check a written file, and revert it if it cannot be read.

    Every write goes through this. The model is asked to know the language, the
    framework and the codebase at once; a small model gets one of them wrong
    regularly, and the edit then stays, looks plausible in a diff, and is found
    by whoever runs the code. None of the three needs a model to check.
    """
    from mimir.verify.change import verify_change

    try:
        return verify_change(
            root, relative, updated=updated, original=original,
            settings=ctx.settings, baseline=baseline,
        )
    except Exception:  # noqa: BLE001 - a broken checker must not eat the edit
        log.warning("change_verification_failed", path=relative)
        return None


_LINE_PREFIX = re.compile(r"^\s*\d+\s{2}", re.MULTILINE)


def strip_line_numbers(text: str) -> str:
    """Remove the gutter read_file_range puts on every line.

    The reading tool returns "   312  def all(self):" and the editing tool
    demanded byte-exact text, so the model had to strip six-space-padded line
    numbers off every line and reproduce the indentation underneath perfectly.
    A 30B model does not, and the first real coding run spent eleven steps
    failing the same edit and changed nothing.

    Only stripped when every non-empty line carries a prefix, so a genuine line
    of code that happens to begin with digits is left alone.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and all(_LINE_PREFIX.match(line) for line in lines):
        return "\n".join(_LINE_PREFIX.sub("", line) for line in text.splitlines())
    return text


def find_span(haystack: str, needle: str) -> tuple[str, str] | None:
    """Locate ``needle`` in ``haystack``, tolerating how it was transcribed.

    Exact first. Then the same text with the reading tool's line numbers
    removed. Then ignoring leading whitespace on each line, which is where a
    model reproducing an indented block most often differs, and only when that
    identifies exactly one place: an ambiguous loose match is the model not
    knowing which occurrence it means, and resolving it here would edit the
    wrong one.

    Returns the text to replace as it actually appears, and how it was found.
    """
    if haystack.count(needle) == 1:
        return needle, "exact"

    stripped = strip_line_numbers(needle)
    if stripped != needle and haystack.count(stripped) == 1:
        return stripped, "line numbers removed"

    wanted = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not wanted:
        return None
    lines = haystack.splitlines(keepends=True)
    bare = [line.strip() for line in lines]
    matches: list[tuple[int, int]] = []
    for start in range(len(lines)):
        cursor, index = start, 0
        while cursor < len(lines) and index < len(wanted):
            if not bare[cursor]:
                cursor += 1
                continue
            if bare[cursor] != wanted[index]:
                break
            cursor += 1
            index += 1
        if index == len(wanted):
            matches.append((start, cursor))
    if len(matches) == 1:
        start, end = matches[0]
        return "".join(lines[start:end]), "indentation ignored"
    return None


def _reindent(actual: str, replacement: str) -> str:
    """Shift the replacement to the indentation the file actually uses."""
    def leading(text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line[: len(line) - len(line.lstrip())]
        return ""

    have, want = leading(strip_line_numbers(replacement)), leading(actual)
    if have == want:
        return replacement
    out = []
    for line in strip_line_numbers(replacement).splitlines():
        out.append(want + line[len(have):] if line.startswith(have) else line)
    return "\n".join(out)


def _nearby(original: str, wanted: str) -> str:
    """Show what is actually there, so a retry has something to aim at.

    "does not appear" told the model nothing it could act on, and it retried
    the same edit with cosmetic changes four times."""
    first = next((line.strip() for line in strip_line_numbers(wanted).splitlines()
                  if line.strip()), "")
    if not first:
        return "Read the file again."
    for number, line in enumerate(original.splitlines(), start=1):
        if first[:40] and first[:40] in line:
            return (
                f"Line {number} is {line.strip()[:100]!r}, which is close. "
                "Copy the text from the file exactly, without line numbers."
            )
    return "Read the file again: it may have changed since you last saw it."


def _baseline(root: Path, relative: str):
    from mimir.verify.change import baseline_diagnostics

    try:
        return baseline_diagnostics(root, relative)
    except Exception:  # noqa: BLE001
        return None


def _with_report(result: ToolResult, report) -> ToolResult:
    """Fold a verification report into the tool result the model reads."""
    if report is None:
        return result
    result.data["verification"] = {
        "ok": report.ok,
        "reverted": report.reverted,
        "checks": report.checks_run,
        "skipped": report.checks_skipped,
        "violations": [v.render() for v in report.violations],
        "new_diagnostics": report.new_diagnostics[:8],
    }
    if report.ok:
        result.summary = f"{result.summary}. {report.summary()}"
        return result
    detail = "; ".join(report.detail()[:3])
    result.summary = f"{result.summary}. {report.summary()}: {detail}"
    return result


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
        original = (
            target.read_text(encoding="utf-8", errors="replace") if existed else None
        )
        baseline = _baseline(wt.root, args.path) if existed else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        report = _verify(ctx, wt.root, args.path, updated=args.content,
                         original=original, baseline=baseline)
        if report is not None and report.reverted:
            raise ToolError(
                f"{args.path} {report.violations[0].title}. The file was left as it was.",
                code="invalid_change",
            )
        lines = args.content.count("\n") + 1
        return _with_report(ToolResult(
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
        ), report)

    return await asyncio.to_thread(work)


class EditInput(BaseModel):
    task: str
    path: str = Field(description="Path relative to the worktree root.")
    old_string: str = Field(
        description=(
            "The exact text to replace, copied from the file including its indentation. "
            "Must appear exactly once in the file: include enough surrounding lines to "
            "make it unique."
        )
    )
    new_string: str = Field(description="The text to put in its place.")
    repo: str | None = None


@tool(
    "edit_worktree_file",
    description=(
        "Replace one exact span of text in a file inside a task worktree. Prefer this over "
        "write_worktree_file for any change to an existing file: it does not require "
        "reproducing the whole file, so it cannot silently drop the parts you did not "
        "mention. The old text must match exactly, including indentation, and must appear "
        "exactly once."
    ),
    capability=Capability.CODE,
    risk=RiskClass.R1,
    mutating=True,
    tags=("repository", "worktree", "write"),
)
async def edit_worktree_file(args: EditInput, ctx: ToolContext) -> ToolResult:
    """Exact-span replacement.

    The whole-file write is the wrong primitive for editing. A model asked to
    reproduce a 400 line file to change two of them will drop something, and
    the drop is invisible: the file is syntactically fine and the diff looks
    plausible. Requiring the old text makes the model state what it believes is
    there, so a stale belief fails loudly instead of overwriting the file with
    it.

    A match count that is not exactly one is refused rather than resolved by
    picking the first. Ambiguity here is the model not knowing which occurrence
    it means, and guessing on its behalf edits the wrong line.
    """
    root = await _repo_root(ctx, args.repo)

    def work() -> ToolResult:
        manager = _manager(ctx)
        try:
            wt = manager.find(root, args.task)
            target = resolve_inside(wt.root, args.path)
        except WorktreeError as exc:
            raise _wrap(exc) from exc
        if not target.is_file():
            raise ToolError(
                f"{args.path} does not exist in the worktree. "
                "Use write_worktree_file to create it.",
                code="not_found",
            )
        if not args.old_string:
            raise ToolError(
                "old_string must not be empty. To create a file, use write_worktree_file.",
                code="invalid_arguments",
            )

        original = target.read_text(encoding="utf-8", errors="replace")
        occurrences = original.count(args.old_string)
        if occurrences > 1:
            raise ToolError(
                f"the text to replace appears {occurrences} times in {args.path}. "
                "Include more surrounding lines so it identifies one place.",
                code="ambiguous_match",
            )

        located = find_span(original, args.old_string)
        if located is None:
            raise ToolError(
                f"the text to replace does not appear in {args.path}. "
                + _nearby(original, args.old_string),
                code="no_match",
            )
        actual, how = located
        new_text = args.new_string
        if how == "indentation ignored":
            # Re-indent the replacement to the indentation actually in the
            # file, or a matched-but-differently-indented block would be
            # replaced with text at the model's guessed indentation.
            new_text = _reindent(actual, args.new_string)

        updated = original.replace(actual, new_text, 1)
        if len(updated.encode()) > MAX_BYTES:
            raise ToolError(f"result exceeds {MAX_BYTES} bytes", code="too_large")
        baseline = _baseline(wt.root, args.path)
        target.write_text(updated, encoding="utf-8")
        report = _verify(ctx, wt.root, args.path, updated=updated,
                         original=original, baseline=baseline)
        if report is not None and report.reverted:
            raise ToolError(
                f"that edit left {args.path} unparseable "
                f"({report.violations[0].title}). The file was restored.",
                code="invalid_change",
            )

        before = original[: original.index(actual)].count("\n") + 1
        removed = actual.count("\n") + 1
        added = new_text.count("\n") + 1
        return _with_report(ToolResult(
            tool="edit_worktree_file",
            summary=(
                f"{args.path}: {removed} line(s) replaced with {added} at line {before}"
            ),
            data={
                "path": args.path,
                "branch": wt.branch,
                "line": before,
                "old_string": actual,
                "new_string": new_text,
                "matched_by": how,
            },
            evidence=[
                Evidence(
                    claim=f"edited {args.path} at line {before}",
                    kind=EvidenceKind.OBSERVED,
                    source_type=SourceType.REPOSITORY,
                    source_id=args.path,
                    excerpt=new_text[:400],
                    collected_by="edit_worktree_file",
                    citations=[Citation(source_type=SourceType.REPOSITORY,
                                        locator=f"{args.path}:{before}", path=args.path)],
                )
            ],
        ), report)

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
    "edit_worktree_file",
    "list_task_worktrees",
    "run_worktree_tests",
    "write_worktree_file",
]
