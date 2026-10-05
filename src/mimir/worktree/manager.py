"""Creating, inspecting and discarding task worktrees."""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mimir.logging import get_logger

log = get_logger(__name__)

SLUG = re.compile(r"[^a-z0-9-]+")
BRANCH_PREFIX = "mimir/"


class WorktreeError(RuntimeError):
    """A worktree operation failed, or was refused for safety."""


def slugify(name: str) -> str:
    out = SLUG.sub("-", name.strip().lower()).strip("-")
    return (out or "task")[:48]


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``, refusing anything that escapes."""
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    raise WorktreeError(
        f"{relative!r} resolves to {candidate}, outside the task worktree at {root}. "
        "Writing outside the worktree requires explicit approval (ADR-002 3.3)."
    )


@dataclass(frozen=True)
class TaskWorktree:
    name: str
    branch: str
    root: Path
    repo_root: Path
    base_commit: str
    created_at: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "branch": self.branch,
            "path": str(self.root),
            "repository": str(self.repo_root),
            "base_commit": self.base_commit,
        }


def _git(cwd: Path, *args: str, timeout: float = 60.0) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


class WorktreeManager:
    """One manager per MIMIR home. Worktrees live outside every repository."""

    def __init__(self, home: Path):
        self.base = Path(home) / "worktrees"
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, repo_root: Path, name: str) -> Path:
        return self.base / f"{Path(repo_root).name}-{slugify(name)}"

    def create(self, repo_root: Path, name: str, *, base: str = "HEAD") -> TaskWorktree:
        repo_root = Path(repo_root).resolve()
        if not (repo_root / ".git").exists():
            raise WorktreeError(f"{repo_root} is not a git repository")

        slug = slugify(name)
        branch = f"{BRANCH_PREFIX}{slug}"
        target = self.path_for(repo_root, name)
        if target.exists():
            raise WorktreeError(
                f"a worktree for {slug!r} already exists at {target}. "
                "Discard it first, or choose another task name."
            )

        commit = _git(repo_root, "rev-parse", base).strip()
        # An interrupted run leaves the branch behind when its worktree is
        existing = _git(repo_root, "branch", "--list", branch).strip()
        if existing:
            log.info("worktree_reattached", branch=branch, path=str(target))
            _git(repo_root, "worktree", "add", str(target), branch)
        else:
            _git(repo_root, "worktree", "add", "-b", branch, str(target), commit)
        log.info("worktree_created", branch=branch, path=str(target), base=commit[:12])
        return TaskWorktree(
            name=slug, branch=branch, root=target, repo_root=repo_root,
            base_commit=commit, created_at=time.time(),
        )

    def find(self, repo_root: Path, name: str) -> TaskWorktree:
        repo_root = Path(repo_root).resolve()
        target = self.path_for(repo_root, name)
        if not target.is_dir():
            raise WorktreeError(
                f"no task worktree named {slugify(name)!r}. "
                "Create one with create_task_worktree first."
            )
        branch = _git(target, "rev-parse", "--abbrev-ref", "HEAD").strip()
        merge_base = _git(target, "rev-parse", "HEAD")
        return TaskWorktree(
            name=slugify(name), branch=branch, root=target, repo_root=repo_root,
            base_commit=merge_base.strip(), created_at=target.stat().st_mtime,
        )

    def list(self) -> list[dict[str, object]]:
        out = []
        for path in sorted(self.base.iterdir()) if self.base.is_dir() else []:
            if not (path / ".git").exists():
                continue
            try:
                branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip()
                dirty = bool(_git(path, "status", "--porcelain").strip())
            except WorktreeError:
                continue
            out.append({"name": path.name, "path": str(path), "branch": branch,
                        "dirty": dirty})
        return out

    def diff(self, worktree: TaskWorktree, *, stat: bool = False) -> str:
        args = ["diff", worktree.base_commit]
        if stat:
            args.append("--stat")
        tracked = _git(worktree.root, *args)
        untracked = _git(worktree.root, "ls-files", "--others", "--exclude-standard")
        if untracked.strip():
            tracked += "\n# untracked files:\n" + "".join(
                f"#   {line}\n" for line in untracked.splitlines()
            )
        return tracked

    def discard(self, worktree: TaskWorktree) -> None:
        """Remove the worktree and its branch. The reversibility guarantee."""
        root, repo, branch = worktree.root, worktree.repo_root, worktree.branch
        try:
            _git(repo, "worktree", "remove", "--force", str(root))
        except WorktreeError:
            shutil.rmtree(root, ignore_errors=True)
            _git(repo, "worktree", "prune")
        if branch.startswith(BRANCH_PREFIX):
            # A branch that will not delete is not a reason to fail the
            with contextlib.suppress(WorktreeError):
                _git(repo, "branch", "-D", branch)
        log.info("worktree_discarded", branch=branch, path=str(root))
