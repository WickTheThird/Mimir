"""Task worktree tests.

The property under test is containment. Everything else is convenience; if a
write can escape the worktree, ADR-002's reversibility guarantee is void.
"""

from __future__ import annotations

import subprocess

import pytest

from mimir.worktree import WorktreeError, WorktreeManager, resolve_inside
from mimir.worktree.manager import slugify


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "a.txt").write_text("one\n")
    for args in (["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t",
                                "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    return root


class TestContainment:
    def test_a_path_inside_resolves(self, tmp_path):
        assert resolve_inside(tmp_path, "src/a.py").name == "a.py"

    def test_parent_traversal_is_refused(self, tmp_path):
        with pytest.raises(WorktreeError) as exc:
            resolve_inside(tmp_path, "../../../etc/passwd")
        assert "outside the task worktree" in str(exc.value)

    def test_an_absolute_path_is_refused(self, tmp_path):
        with pytest.raises(WorktreeError):
            resolve_inside(tmp_path, "/etc/passwd")

    def test_a_symlink_pointing_out_is_refused(self, tmp_path):
        """A boundary compared as a string prefix is crossed by a symlink.
        Resolution happens before the check for exactly this reason."""
        outside = tmp_path.parent / "outside"
        outside.mkdir(exist_ok=True)
        root = tmp_path / "wt"
        root.mkdir()
        (root / "link").symlink_to(outside)
        with pytest.raises(WorktreeError):
            resolve_inside(root, "link/stolen.txt")

    def test_the_root_itself_is_inside(self, tmp_path):
        assert resolve_inside(tmp_path, ".") == tmp_path.resolve()


class TestLifecycle:
    def test_create_writes_nothing_to_the_operator_checkout(self, repo, tmp_path):
        manager = WorktreeManager(tmp_path / "home")
        wt = manager.create(repo, "my task")

        (wt.root / "new.txt").write_text("added\n")
        assert not (repo / "new.txt").exists(), "the operator's tree must be untouched"
        assert wt.branch == "mimir/my-task"

    def test_diff_reports_tracked_and_untracked_changes(self, repo, tmp_path):
        manager = WorktreeManager(tmp_path / "home")
        wt = manager.create(repo, "t")
        (wt.root / "a.txt").write_text("two\n")
        (wt.root / "brand_new.py").write_text("x = 1\n")

        diff = manager.diff(wt)
        assert "a.txt" in diff, "modified tracked file"
        assert "brand_new.py" in diff, "untracked files are part of the change"

    def test_discard_removes_everything(self, repo, tmp_path):
        manager = WorktreeManager(tmp_path / "home")
        wt = manager.create(repo, "throwaway")
        (wt.root / "junk.txt").write_text("junk\n")

        manager.discard(wt)
        assert not wt.root.exists()
        branches = subprocess.run(
            ["git", "-C", str(repo), "branch", "--list", wt.branch],
            capture_output=True, text=True, check=False,
        ).stdout
        assert wt.branch not in branches, "the task branch goes with the worktree"

    def test_a_duplicate_task_is_refused_rather_than_clobbered(self, repo, tmp_path):
        manager = WorktreeManager(tmp_path / "home")
        manager.create(repo, "dup")
        with pytest.raises(WorktreeError) as exc:
            manager.create(repo, "dup")
        assert "already exists" in str(exc.value)

    def test_a_missing_worktree_says_how_to_make_one(self, repo, tmp_path):
        manager = WorktreeManager(tmp_path / "home")
        with pytest.raises(WorktreeError) as exc:
            manager.find(repo, "never-created")
        assert "create_task_worktree" in str(exc.value)

    def test_a_non_repository_is_refused(self, tmp_path):
        manager = WorktreeManager(tmp_path / "home")
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(WorktreeError):
            manager.create(plain, "t")


class TestSlug:
    def test_names_become_safe_branch_segments(self):
        assert slugify("Fix Retry Bounds!") == "fix-retry-bounds"
        assert slugify("  ") == "task"
        assert "/" not in slugify("a/b/c")


class TestCredentialScrub:
    def test_credentials_are_removed_by_name_and_by_shape(self):
        """Listing every credential variable is impossible; matching the shape
        catches the ones nobody thought of."""
        from mimir.tools.code import _CREDENTIAL_MARKERS, _CREDENTIAL_VARS

        assert "KUBECONFIG" in _CREDENTIAL_VARS
        for name in ("MY_APP_TOKEN", "DB_PASSWORD", "SOME_SECRET", "X_API_KEY"):
            assert any(m in name.upper() for m in _CREDENTIAL_MARKERS), name
        for name in ("PATH", "HOME", "LANG", "PYTHONPATH"):
            assert not any(m in name.upper() for m in _CREDENTIAL_MARKERS), (
                f"{name} is toolchain, not a credential; stripping it made every "
                "test command exit 127"
            )
