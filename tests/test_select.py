"""Best-of-k, and the selector that makes it worth anything.

Across twelve stored runs of the 52 case suite, none of the 52 fails
structurally: every case has passed at least once, so pass@1 of 0.873 becomes
0.944 at three attempts. Turning that into a real gain needs a selector that is
right, which is why this one only looks at things that were observed.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from mimir.agent.select import Candidate, best_of, changed_files, inspect


class Settings:
    def __init__(self, home):
        self.home = home


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("def f():\n    return 1\n")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(root), *argv], check=True, capture_output=True)
    return root


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "rules").mkdir(exist_ok=True)
    return Settings(tmp_path)


def _candidate(root, index=0):
    return Candidate(index=index, task=f"t{index}", root=Path(root))


class TestScoring:
    def test_an_attempt_that_changed_nothing_is_not_usable(self, repo, settings):
        c = inspect(_candidate(repo), settings)
        assert c.files_changed == 0
        assert not c.usable

    def test_an_attempt_that_does_not_parse_is_not_usable(self, repo, settings):
        (repo / "src" / "a.py").write_text("def f(:\n")
        c = inspect(_candidate(repo), settings)
        assert c.files_changed == 1
        assert not c.parses
        assert not c.usable

    def test_a_clean_change_is_usable(self, repo, settings):
        (repo / "src" / "a.py").write_text("def f():\n    return 2\n")
        assert inspect(_candidate(repo), settings).usable

    def test_lint_findings_count_against_an_attempt(self, repo, settings):
        import shutil

        if not shutil.which("ruff"):
            pytest.skip("no linter available")
        (repo / "src" / "a.py").write_text("import os\n\n\ndef f():\n    return 1\n")
        assert inspect(_candidate(repo), settings).lint_findings > 0

    def test_passing_tests_outrank_everything_else(self, repo, settings):
        winner = _candidate(repo, 0)
        winner.files_changed, winner.tests_ran, winner.tests_passed = 1, True, True
        winner.lines_changed = 400
        loser = _candidate(repo, 1)
        loser.files_changed, loser.lines_changed = 1, 2
        assert winner.score > loser.score

    def test_the_smaller_diff_wins_a_tie(self):
        """Between two attempts that pass everything, the smaller one did what
        was asked and no more."""
        small, large = _candidate("/x", 0), _candidate("/x", 1)
        for c, lines in ((small, 5), (large, 90)):
            c.files_changed, c.lines_changed = 1, lines
            c.tests_ran = c.tests_passed = True
        assert small.score > large.score

    def test_a_suite_that_could_not_start_is_not_a_suite_that_failed(self):
        """Exit codes other than 0 and 1 mean the runner itself broke."""
        """Or an attempt that broke the test runner outranks one that broke a
        test."""
        broke_runner, broke_test = _candidate("/x", 0), _candidate("/x", 1)
        broke_runner.files_changed = broke_test.files_changed = 1
        broke_test.tests_ran, broke_test.tests_passed = True, False
        assert broke_test.score > broke_runner.score


class TestTestsAreRun:
    def test_a_passing_suite_is_recorded(self, repo, settings):
        (repo / "src" / "a.py").write_text("def f():\n    return 2\n")
        c = inspect(_candidate(repo), settings, test_command="exit 0")
        assert c.tests_ran and c.tests_passed

    def test_a_failing_suite_is_recorded(self, repo, settings):
        (repo / "src" / "a.py").write_text("def f():\n    return 2\n")
        c = inspect(_candidate(repo), settings, test_command="exit 1")
        assert c.tests_ran and not c.tests_passed


class TestSelection:
    class View:
        def __init__(self, index, root, edits):
            self.task = f"t{index}"
            self.root = root
            self.agent = type("A", (), {"outcome": type("O", (), {"steps": 2})()})()
            self._edits = edits
            self._index = index

        async def turn(self, instruction):
            text = self._edits[self._index]
            if text is not None:
                (Path(self.root) / "src" / "a.py").write_text(text)

    def _run(self, repo, settings, edits):
        roots = []
        for index in range(len(edits)):
            root = Path(repo).parent / f"w{index}"
            subprocess.run(["git", "clone", "-q", str(repo), str(root)],
                           check=True, capture_output=True)
            roots.append(root)
        return asyncio.run(best_of(
            len(edits),
            lambda i: TestSelection.View(i, roots[i], edits),
            "do it",
            settings=settings,
        ))

    def test_the_attempt_that_parses_beats_the_one_that_does_not(self, repo, settings):
        winner, all_of = self._run(repo, settings, ["def f(:\n", "def f():\n    return 2\n"])
        assert winner is not None
        assert winner.index == 1
        assert len(all_of) == 2

    def test_an_attempt_that_changed_nothing_never_wins(self, repo, settings):
        winner, _ = self._run(repo, settings, [None, "def f():\n    return 2\n"])
        assert winner is not None and winner.index == 1

    def test_nothing_usable_returns_nothing_rather_than_the_least_bad(
        self, repo, settings
    ):
        """Shipping the least broken of three broken attempts is worse than
        saying none worked."""
        winner, all_of = self._run(repo, settings, ["def f(:\n", "def g(:\n"])
        assert winner is None
        assert len(all_of) == 2

    def test_one_failing_attempt_does_not_lose_the_others(self, repo, settings):
        class Exploding(TestSelection.View):
            async def turn(self, instruction):
                if self._index == 0:
                    raise RuntimeError("model unreachable")
                await super().turn(instruction)

        roots = []
        for index in range(2):
            root = Path(repo).parent / f"x{index}"
            subprocess.run(["git", "clone", "-q", str(repo), str(root)],
                           check=True, capture_output=True)
            roots.append(root)
        edits = [None, "def f():\n    return 3\n"]
        winner, all_of = asyncio.run(best_of(
            2, lambda i: Exploding(i, roots[i], edits), "do it", settings=settings
        ))
        assert winner is not None and winner.index == 1
        assert all_of[0].error


class TestChangedFiles:
    def test_it_sees_both_edits_and_new_files(self, repo):
        (repo / "src" / "a.py").write_text("x = 1\n")
        (repo / "src" / "b.py").write_text("y = 2\n")
        assert set(changed_files(repo)) == {"src/a.py", "src/b.py"}


class TestASkippedCheckIsVisible:
    """A test command that could not start left the render silent, and the
    selector quietly demoted itself from deciding on tests to deciding on diff
    size. That is the fail-open shape this project exists to refuse."""

    def test_a_test_command_that_never_started_is_reported(self, repo, settings):
        (repo / "src" / "a.py").write_text("def f():\n    return 2\n")
        c = inspect(_candidate(repo), settings, test_command="definitely-not-a-command")
        assert c.tests_wanted
        assert not c.tests_ran
        assert "TESTS DID NOT RUN" in c.render()
        assert c.notes

    def test_no_test_command_means_no_complaint(self, repo, settings):
        (repo / "src" / "a.py").write_text("def f():\n    return 2\n")
        c = inspect(_candidate(repo), settings)
        assert not c.tests_wanted
        assert "TESTS" not in c.render()


class TestTheSelectorRunsTestsTheSameWayTheToolDoes:
    """Two places run tests and only one resolved a bare python. A selector
    given "python -m pytest" scored every candidate as TESTS DID NOT RUN and
    fell back to diff size, with the tests never having executed."""

    def test_a_bare_python_is_resolved_against_the_source_checkout(
        self, repo, settings, tmp_path
    ):
        venv = tmp_path / "src_repo" / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("#!/bin/sh\nexit 0\n")
        (venv / "python").chmod(0o755)

        (repo / "src" / "a.py").write_text("def f():\n    return 2\n")
        c = inspect(_candidate(repo), settings, "python -m pytest",
                    repo_root=tmp_path / "src_repo")
        assert c.tests_ran, "the interpreter has to be found for the tests to run"
        assert c.tests_passed

    def test_both_paths_use_one_resolver(self):
        import inspect as _inspect

        from mimir.agent import select

        assert "_resolve_interpreter" in _inspect.getsource(select._run_tests)
