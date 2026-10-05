"""Task worktrees: the isolation boundary for code mutation (ADR-002 section 3)."""

from mimir.worktree.manager import (
    TaskWorktree,
    WorktreeError,
    WorktreeManager,
    resolve_inside,
)

__all__ = [
    "TaskWorktree",
    "WorktreeError",
    "WorktreeManager",
    "resolve_inside",
]
