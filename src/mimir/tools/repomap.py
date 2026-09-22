"""The repository map as a tool: a lookup before a search."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mimir.knowledge.repomap import get_repo_map
from mimir.logging import get_logger
from mimir.safety.risk import RiskClass
from mimir.tools.base import Capability, ToolContext, ToolResult, tool
from mimir.tools.code import _repo_root

log = get_logger(__name__)


class MapInput(BaseModel):
    query: str = Field(
        description=(
            "A symbol name to locate, a file path to outline, or one of: "
            "'hot' for the most-changed files, 'summary' for the index status."
        )
    )
    repo: str | None = None
    refresh: bool = Field(default=False, description="Rebuild the index before answering.")


def _ensure_built(ctx: ToolContext, root: Path, refresh: bool) -> Any:
    repo_map = get_repo_map(ctx.settings, root)
    if refresh or repo_map.summary()["files"] == 0:
        repo_map.build(ignore_globs=tuple(ctx.settings.repos.ignore_globs),
                       max_file_bytes=ctx.settings.repos.max_file_bytes)
    return repo_map


@tool(
    "repository_map",
    description=(
        "Where things are in a repository, from an index rather than a search: "
        "definitions of a symbol with file and line, the outline of a file, the "
        "tests that cover a module, and the files that change most. Ask this "
        "before search_repository when you know a name."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R0,
    tags=("repository", "code", "index"),
)
async def repository_map(args: MapInput, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)
    repo_map = _ensure_built(ctx, root, args.refresh)
    q = args.query.strip()
    if q == "summary":
        data = repo_map.summary()
        return ToolResult(ok=True, tool="repository_map", summary=str(data), data=data)
    if q == "hot":
        hot = repo_map.hot(15)
        return ToolResult(ok=True, tool="repository_map",
                          summary="\n".join(f"{n:4}  {p}" for p, n in hot) or "no history",
                          data={"hot": hot})
    if "/" in q or q.endswith((".py", ".go", ".ts", ".rs")):
        outline = repo_map.outline(q)
        tests = repo_map.tests_for([q])
        lines = [f"{s.line:5}  {s.kind:10} {s.parent + '.' if s.parent else ''}{s.name}" for s in outline]
        return ToolResult(ok=True, tool="repository_map",
                          summary=(f"{q}\n" + "\n".join(lines) + (f"\ntests: {', '.join(tests)}" if tests else "\ntests: none found")),
                          data={"outline": [asdict(s) for s in outline], "tests": tests})
    hits = repo_map.symbols(q)
    if not hits:
        return ToolResult(ok=True, tool="repository_map",
                          summary=f"no definition named like '{q}' in the index; try search_repository",
                          data={"symbols": []})
    return ToolResult(
        ok=True, tool="repository_map",
        summary="\n".join(f"{s.path}:{s.line}  {s.kind} {s.parent + '.' if s.parent else ''}{s.name}" for s in hits),
        data={"symbols": [asdict(s) for s in hits]},
    )


def affected_tests(ctx: ToolContext, root: Path, worktree_root: Path) -> list[str]:
    """Tests that cover what the worktree changed, from the map."""
    try:
        changed = subprocess.run(
            ["git", "-C", str(worktree_root), "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return []
    if not changed:
        return []
    repo_map = _ensure_built(ctx, root, False)
    return repo_map.tests_for(changed)


__all__ = ["affected_tests", "repository_map"]
