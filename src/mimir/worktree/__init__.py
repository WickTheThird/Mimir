"""Task worktrees: the isolation boundary for code mutation (ADR-002 section 3).

MIMIR never writes to the operator's working tree. Every change happens in a
git worktree created for one task, which buys three properties without any
cleverness:

* every change is reversible by discarding the worktree;
* every change is reviewable as a diff against a known base commit;
* an abandoned task leaves the operator's checkout untouched.

The containment is enforced by path resolution, not by convention. A write
whose resolved path escapes the worktree root is refused, and that check runs
after symlink resolution because a symlink pointing outside is exactly how a
containment boundary made of string prefixes gets crossed.
"""

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
