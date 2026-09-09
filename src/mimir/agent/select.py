"""Sample the work more than once, and let the rules choose.

Across twelve stored runs of the 52 case suite, 39 cases always pass, 13 are
flaky and none fails structurally: every case has passed at least once. So
pass@1 of 0.873 becomes 0.944 at three attempts and 0.964 at five, and the gap
is not knowledge the model lacks, it is variance.

Turning that into a real gain needs a selector, and the selector is the part
that must not be a model. A judge that is wrong q of the time gives back
roughly q of what sampling won, which is why this scores a candidate only on
things that were observed: does it parse, does the linter say anything new, do
the tests pass, did it change anything at all.

The cost is lower than it looks. The k attempts share nothing with each other,
but each is one conversation, and a conversation after its first step costs a
tenth of that step because the prefix is cached.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

TEST_TIMEOUT_S = 900.0


@dataclass
class Candidate:
    """One attempt, and what the rules found in it."""

    index: int
    task: str
    root: Path
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    steps: int = 0
    parses: bool = True
    lint_findings: int = 0
    dead_definitions: int = 0
    """Names it introduced that nothing reads, or defined twice."""
    tests_ran: bool = False
    tests_passed: bool = False
    tests_wanted: bool = False
    """Whether a test command was asked for, whether or not it started."""
    error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether this attempt is worth keeping at all."""
        return bool(self.files_changed) and self.parses and not self.error

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_removed

    @property
    def score(self) -> tuple:
        """Ordered by what was verified, not by how it reads.

        Tests first because they are the strongest evidence available, then the
        linter, then whether anything changed at all. Fewer lines breaks a tie:
        between two attempts that both pass everything, the smaller diff is the
        one that did what was asked and no more.
        """
        return (
            self.usable,
            self.dead_definitions == 0,
            self.tests_passed,
            self.tests_ran,
            -self.lint_findings,
            # Deletion before size. One attempt removed 248 lines and added 76
            # while passing syntax, lint, tests and the definition check, and a
            # single combined line count hid it: 324 changed looked like a
            # large edit rather than a file being gutted.
            -self.lines_removed,
            -self.lines_added,
        )

    def render(self) -> str:
        if self.error:
            return f"#{self.index} failed: {self.error[:80]}"
        if not self.files_changed:
            return f"#{self.index} changed nothing"
        bits = [f"{self.files_changed} file(s)", f"+{self.lines_added}/-{self.lines_removed}"]
        if not self.parses:
            bits.append("does not parse")
        if self.lint_findings:
            bits.append(f"{self.lint_findings} lint finding(s)")
        if self.dead_definitions:
            bits.append(f"{self.dead_definitions} unused/duplicate definition(s)")
        if self.tests_ran:
            bits.append("tests pass" if self.tests_passed else "TESTS FAIL")
        elif self.tests_wanted:
            # Silence here would be the selector quietly demoting itself from
            # deciding on tests to deciding on diff size, which is the whole
            # fail-open shape this project exists to refuse.
            bits.append("TESTS DID NOT RUN")
        return f"#{self.index} " + ", ".join(bits)


def _original(root: Path, relative: str) -> str | None:
    """The file as it was before the change, from git."""
    out = _git(root, "show", f"HEAD:{relative}")
    return out or None


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=120, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def changed_files(root: Path) -> list[str]:
    out = _git(root, "status", "--porcelain")
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def _line_counts(root: Path) -> tuple[int, int]:
    """Added and removed, kept apart.

    Summing them into one number made a change that deleted 248 lines and
    added 76 report as "324 lines changed", which reads as a large edit rather
    than as a file being gutted.
    """
    added = removed = 0
    for line in _git(root, "diff", "--numstat").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            added += int(parts[0])
            removed += int(parts[1])
    return added, removed


def inspect(
    candidate: Candidate,
    settings: Any,
    test_command: str = "",
    repo_root: Path | None = None,
) -> Candidate:
    """Score one attempt against the rules. No model is consulted."""
    from mimir.verify import definitions
    from mimir.verify.change import check_syntax
    from mimir.verify.rules import check_rules, load_rules, rule_roots

    files = changed_files(candidate.root)
    candidate.files_changed = len(files)
    candidate.lines_added, candidate.lines_removed = _line_counts(candidate.root)
    if not files:
        return candidate

    rules = load_rules(rule_roots(settings, candidate.root))
    for relative in files:
        path = candidate.root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if check_syntax(relative, text) is not None:
            candidate.parses = False
            candidate.notes.append(f"{relative} does not parse")
        candidate.lint_findings += len(check_rules(rules, relative, text))

        # An attempt that added something nothing uses did not finish the job,
        # and its diff is smaller than one that did, so without this the tie
        # break actively prefers the incomplete answer.
        before = _original(candidate.root, relative)
        for issue in definitions.check(before, text):
            candidate.dead_definitions += 1
            candidate.notes.append(f"{relative}:{issue.line} {issue.message}")

    candidate.lint_findings += _lint_count(candidate.root, files)
    if test_command:
        candidate.tests_wanted = True
        candidate.tests_ran, candidate.tests_passed = _run_tests(
            candidate.root, test_command, repo_root
        )
        if not candidate.tests_ran:
            candidate.notes.append(
                "the test command did not start, so nothing was verified by it"
            )
            log.warning("candidate_tests_did_not_run", task=candidate.task)
    return candidate


def _lint_count(root: Path, files: list[str]) -> int:
    """Findings from the project's own linter, over the changed files only."""
    python = [f for f in files if f.endswith(".py")]
    if not python or not shutil.which("ruff"):
        return 0
    try:
        finished = subprocess.run(
            ["ruff", "check", "--quiet", "--force-exclude", *python],
            cwd=str(root), capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return len([line for line in finished.stdout.splitlines() if line.strip()])


def _run_tests(root: Path, command: str, repo_root: Path | None = None) -> tuple[bool, bool]:
    """Whether the tests ran, and whether they passed.

    The two are separate because a suite that could not start is not a suite
    that failed, and scoring them the same would let an attempt that broke the
    test runner outrank one that merely broke a test.
    """
    # The same interpreter resolution the test tool does. Two places run
    # tests and only one of them resolved a bare python, so a selector given
    # "python -m pytest" scored every candidate as TESTS DID NOT RUN and fell
    # back to diff size without the tests ever executing.
    from mimir.tools.code import _resolve_interpreter

    resolved = _resolve_interpreter(command, repo_root or root)
    try:
        finished = subprocess.run(
            resolved, cwd=str(root), shell=True, capture_output=True,
            text=True, timeout=TEST_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, False
    if finished.returncode in (0, 1):
        return True, finished.returncode == 0
    return False, False


async def best_of(
    k: int,
    make_view: Any,
    instruction: str,
    *,
    settings: Any,
    test_command: str = "",
    console: Any = None,
) -> tuple[Candidate | None, list[Candidate]]:
    """Run the instruction ``k`` times and return the attempt the rules prefer.

    ``make_view`` is called with an index and returns a view whose agent is
    already bound to its own worktree, so the attempts cannot see each other.
    Attempts run one after another rather than together: they share one
    runtime, and running them concurrently makes each slower without making the
    set faster.
    """
    candidates: list[Candidate] = []
    for index in range(k):
        view = make_view(index)
        candidate = Candidate(index=index, task=view.task, root=Path(view.root))
        try:
            await view.turn(instruction)
            candidate.steps = getattr(view.agent.outcome, "steps", 0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad attempt is not a failure
            candidate.error = str(exc)
            log.warning("candidate_failed", index=index, error=str(exc))
        inspect(candidate, settings, test_command, getattr(view, "source_root", None))
        candidates.append(candidate)
        if console is not None:
            console.print(f"  {candidate.render()}")

    usable = [c for c in candidates if c.usable]
    if not usable:
        return None, candidates
    return max(usable, key=lambda c: c.score), candidates


__all__ = ["Candidate", "best_of", "changed_files", "inspect"]
