"""One indexed pass per repository: symbols, imports, tests, hot spots."""

import subprocess

import pytest

from mimir.knowledge.repomap import RepoMap


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "billing"
    (root / "src" / "billing").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "billing" / "__init__.py").write_text("")
    (root / "src" / "billing" / "retry.py").write_text(
        "MAX_TRIES = 3\n\ndef backoff(n):\n    return 2 ** n\n\nclass Retrier:\n    def run(self):\n        pass\n"
    )
    (root / "src" / "billing" / "client.py").write_text(
        "from billing.retry import backoff\n\ndef fetch():\n    return backoff(1)\n"
    )
    (root / "tests" / "test_client.py").write_text(
        "from billing.client import fetch\n\ndef test_fetch():\n    assert fetch()\n"
    )
    (root / "main.go").write_text("package main\n\nfunc Serve() {}\n\ntype Server struct{}\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "init"], check=True)
    return root


def test_python_symbols_are_found_with_kind_and_parent(repo, tmp_path):
    m = RepoMap(tmp_path / "cache", repo)
    stats = m.build()
    assert stats["indexed"] >= 4
    names = {(s.name, s.kind, s.parent) for s in m.symbols("backoff") + m.symbols("run") + m.symbols("MAX_TRIES")}
    assert ("backoff", "function", "") in names
    assert ("run", "method", "Retrier") in names
    assert ("MAX_TRIES", "constant", "") in names


def test_other_languages_get_definitions_marked_as_coarse(repo, tmp_path):
    m = RepoMap(tmp_path / "cache", repo); m.build()
    assert {s.kind for s in m.symbols("Serve")} == {"definition"}


def test_imports_resolve_to_repository_paths(repo, tmp_path):
    m = RepoMap(tmp_path / "cache", repo); m.build()
    assert "src/billing/client.py" in m.importers_of("src/billing/retry.py")


def test_tests_for_a_change_follow_the_import_graph(repo, tmp_path):
    """Change retry.py: client imports it, test_client imports client."""
    m = RepoMap(tmp_path / "cache", repo); m.build()
    assert m.tests_for(["src/billing/retry.py"]) == ["tests/test_client.py"]


def test_a_second_build_skips_unchanged_files(repo, tmp_path):
    m = RepoMap(tmp_path / "cache", repo)
    m.build()
    again = m.build()
    assert again["indexed"] == 0 and again["skipped"] == again["files"]


def test_a_changed_file_is_reindexed_and_a_removed_one_forgotten(repo, tmp_path):
    m = RepoMap(tmp_path / "cache", repo); m.build()
    (repo / "src" / "billing" / "retry.py").write_text("def backoff(n):\n    return n\n")
    (repo / "main.go").unlink()
    m.build()
    assert not m.symbols("MAX_TRIES")
    assert not m.symbols("Serve")


def test_hot_spots_come_from_git_history(repo, tmp_path):
    m = RepoMap(tmp_path / "cache", repo); m.build()
    assert dict(m.hot()).get("src/billing/retry.py") == 1
