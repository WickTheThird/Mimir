"""The coding corpus: well formed, contrastive, and checkable without a model."""

from mimir.eval.coding import CodeCase, build_fixture, check, load_corpus, pair_consistency, CodeResult


def test_the_corpus_loads_and_every_pair_has_two_sides():
    cases = load_corpus()
    assert len(cases) >= 8
    sides = {}
    for c in cases:
        sides.setdefault(c.pair, []).append(c.id)
    assert all(len(v) == 2 for v in sides.values()), sides


def test_pairs_share_an_instruction_or_differ_by_the_stated_fact():
    """Contrastive means the repository differs, not the ask, except where the ask itself is the altered fact and the description says so."""
    cases = {c.id: c for c in load_corpus()}
    assert cases["code-constant-introduce-a"].instruction == cases["code-constant-already-there-b"].instruction
    assert cases["code-constant-introduce-a"].files != cases["code-constant-already-there-b"].files


def test_a_fixture_builds_into_a_committed_git_repository(tmp_path):
    case = load_corpus()[0]
    root = build_fixture(case, tmp_path)
    assert (root / ".git").is_dir()
    assert (root / "pkg" / "client.py").read_text().startswith("def fetch")


def test_check_rewards_the_right_diff_and_refuses_the_wrong_one():
    case = CodeCase(id="x", instruction="i", files={}, expect_diff_contains=["+def read_config"],
                    expect_diff_absent=["-def retry(n)"], expect_tests="pass")
    assert check(case, "+def read_config\n", True) == []
    assert check(case, "-def retry(n)\n", True) == ["diff missing '+def read_config'",
                                                     "diff must not contain '-def retry(n)'"]
    assert "did not pass" in check(case, "+def read_config\n", False)[0]


def test_a_no_change_case_rejects_any_edit():
    case = CodeCase(id="x", instruction="i", files={}, expect_max_lines_changed=0)
    assert check(case, "", None) == []
    assert check(case, "--- a\n+++ b\n+x\n", None) == ["1 lines changed, at most 0 expected"]


def test_pair_consistency_counts_both_sides():
    rs = [CodeResult("a-1", "p", True), CodeResult("a-2", "p", False),
          CodeResult("b-1", "q", True), CodeResult("b-2", "q", True), CodeResult("c", "", True)]
    assert pair_consistency(rs) == (0.5, 2)


def test_an_absent_needle_must_not_match_the_removal_line_of_a_correct_rename():
    """The first rename needle matched '-    return load_config("x")', which every correct rename contains, and failed a diff that was right."""
    case = next(c for c in load_corpus() if c.id == "code-rename-callers-a")
    correct = ('--- a/pkg/app.py\n+++ b/pkg/app.py\n-from pkg.config import load_config\n'
               '+from pkg.config import read_config\n-    return load_config("x")\n+    return read_config("x")\n'
               '--- a/pkg/config.py\n+++ b/pkg/config.py\n-def load_config(path):\n+def read_config(path):\n')
    assert check(case, correct, True) == []
    missed = correct.replace('+    return read_config("x")', '+    return load_config("x")')
    assert check(case, missed, True)


def test_an_aborted_run_leaves_no_worktree_behind(tmp_path, settings):
    """The first version called a method that did not exist, so every eval
    worktree leaked and the next run refused to start."""
    from mimir.worktree import WorktreeManager

    case = load_corpus()[0]
    root = build_fixture(case, tmp_path)
    manager = WorktreeManager(settings.home)
    stale = manager.create(root, f"eval-{case.id}")
    assert stale.root.exists()
    # the runner's pre-create discard must clear it
    manager.discard(manager.find(root, f"eval-{case.id}"))
    assert not stale.root.exists()


def test_a_file_the_change_created_appears_in_the_diff(tmp_path):
    """The agent once wrote src/fetch.js into a Python fixture and the diff was empty."""
    from mimir.eval.coding import worktree_diff

    root = build_fixture(load_corpus()[0], tmp_path)
    (root / "src").mkdir(); (root / "src" / "fetch.js").write_text("export const x = 1;\n")
    diff = worktree_diff(root)
    assert "src/fetch.js" in diff and "+export const x" in diff
