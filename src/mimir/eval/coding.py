"""Run the coding corpus: build a fixture repository, change it, check the diff."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mimir.logging import get_logger

log = get_logger(__name__)

CORPUS = Path(__file__).parent / "corpus" / "coding.yaml"


@dataclass(slots=True)
class CodeCase:
    id: str
    instruction: str
    files: dict[str, str]
    pair: str = ""
    test_command: str = ""
    expect_diff_contains: list[str] = field(default_factory=list)
    expect_diff_absent: list[str] = field(default_factory=list)
    expect_max_lines_changed: int | None = None
    expect_tests: str = ""
    description: str = ""


@dataclass(slots=True)
class CodeResult:
    case_id: str
    pair: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    diff: str = ""
    lines_changed: int = 0
    tests_passed: bool | None = None
    duration_s: float = 0.0
    outcome: dict[str, Any] = field(default_factory=dict)


def load_corpus(path: Path = CORPUS) -> list[CodeCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [CodeCase(**c) for c in raw.get("cases", [])]


def build_fixture(case: CodeCase, base: Path) -> Path:
    root = base / case.id
    if root.exists():
        shutil.rmtree(root)
    for rel, text in case.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=eval@mimir", "-c", "user.name=mimir",
                    "commit", "-q", "-m", "fixture"], check=True)
    return root


def _lines_changed(diff: str) -> int:
    return sum(1 for l in diff.splitlines()
               if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))


def check(case: CodeCase, diff: str, tests_passed: bool | None) -> list[str]:
    failures: list[str] = []
    for needle in case.expect_diff_contains:
        if needle not in diff:
            failures.append(f"diff missing {needle!r}")
    for needle in case.expect_diff_absent:
        if needle in diff:
            failures.append(f"diff must not contain {needle!r}")
    if case.expect_max_lines_changed is not None:
        n = _lines_changed(diff)
        if n > case.expect_max_lines_changed:
            failures.append(f"{n} lines changed, at most {case.expect_max_lines_changed} expected")
    if case.expect_tests == "pass" and tests_passed is not True:
        failures.append("fixture tests did not pass after the change"
                        if tests_passed is False else "fixture tests were not run")
    return failures


def _run_tests(root: Path, command: str) -> bool | None:
    if not command:
        return None
    import sys

    cmd = command.replace("python ", f"{sys.executable} ", 1) if command.startswith("python ") else command
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(root), capture_output=True, text=True,
                              timeout=120, check=False, env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                                                             "PYTHONPATH": str(root), "HOME": str(root)})
        return proc.returncode == 0
    except subprocess.SubprocessError:
        return None


async def run_case(case: CodeCase, runner: Any, base: Path) -> CodeResult:
    """One case: fixture, worktree, agent, diff, tests, check."""
    from mimir.agent.loop import AgentEventType, CodingAgent
    from mimir.tools.repo import get_repository_directory
    from mimir.worktree import WorktreeManager

    started = time.time()
    root = build_fixture(case, base)
    directory = get_repository_directory(runner.settings)
    directory.register_session(case.id, root, f"coding corpus fixture {case.id}")
    manager = WorktreeManager(runner.settings.home)
    worktree = manager.create(root, f"eval-{case.id}")
    view = f"{worktree.name}-worktree"
    directory.register_session(view, worktree.root, "eval worktree")
    agent = CodingAgent(
        router=runner.router, registry=runner.registry, tool_context=runner.tool_context(None),
        task=worktree.name, repo=case.id, view=view, worktree_root=worktree.root,
        settings=runner.settings,
    )
    try:
        async for _ in agent.run(case.instruction):
            pass
        diff = subprocess.run(["git", "-C", str(worktree.root), "diff"], capture_output=True,
                              text=True, check=False).stdout
        tests_passed = _run_tests(worktree.root, case.test_command)
        failures = check(case, diff, tests_passed)
        outcome = agent.outcome
        return CodeResult(
            case_id=case.id, pair=case.pair, passed=not failures, failures=failures, diff=diff,
            lines_changed=_lines_changed(diff), tests_passed=tests_passed,
            duration_s=round(time.time() - started, 1),
            outcome={"stopped": outcome.stopped, "steps": outcome.steps,
                     "tool_calls": outcome.tool_calls, "files_changed": sorted(outcome.files_changed),
                     "tests_run": outcome.tests_run},
        )
    finally:
        try:
            manager.remove(root, worktree.name)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def pair_consistency(results: list[CodeResult]) -> tuple[float | None, int]:
    pairs: dict[str, list[bool]] = {}
    for r in results:
        if r.pair:
            pairs.setdefault(r.pair, []).append(r.passed)
    full = {k: v for k, v in pairs.items() if len(v) >= 2}
    if not full:
        return None, 0
    return round(sum(all(v) for v in full.values()) / len(full), 3), len(full)


def summarise(results: list[CodeResult], model: str) -> dict[str, Any]:
    pc, pairs = pair_consistency(results)
    return {"model": model, "total": len(results), "passed": sum(r.passed for r in results),
            "pass_rate": round(sum(r.passed for r in results) / len(results), 3) if results else None,
            "pair_consistency": pc, "pairs": pairs,
            "elapsed_s": round(sum(r.duration_s for r in results))}


def to_json(results: list[CodeResult], model: str) -> str:
    return json.dumps({**summarise(results, model), "results": [
        {"case_id": r.case_id, "pair": r.pair, "passed": r.passed, "failures": r.failures,
         "lines_changed": r.lines_changed, "tests_passed": r.tests_passed,
         "duration_s": r.duration_s, "outcome": r.outcome, "diff": r.diff[:6000]}
        for r in results]}, indent=1)


__all__ = ["CodeCase", "CodeResult", "build_fixture", "check", "load_corpus", "run_case",
           "summarise", "to_json"]
