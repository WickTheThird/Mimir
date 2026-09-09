"""Every write is checked before it is allowed to stand.

MIMIR wrote files and hoped. The model was asked to know the language, the
framework and the codebase at once, and when it got one wrong the edit stayed,
looked plausible in a diff, and was found by whoever ran the code.
"""

from __future__ import annotations

import pytest

from mimir.verify.change import check_syntax, verify_change
from mimir.verify.rules import Rule, check_rules, load_rules


class Settings:
    def __init__(self, home):
        self.home = home


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "rules").mkdir()
    return Settings(tmp_path)


class TestSyntax:
    def test_python_that_does_not_parse_is_caught(self):
        violation = check_syntax("a.py", "def f(:\n    return 1\n")
        assert violation is not None
        assert violation.blocking

    def test_valid_python_passes(self):
        assert check_syntax("a.py", "def f():\n    return 1\n") is None

    def test_json_is_checked_too(self):
        assert check_syntax("a.json", '{"a": }') is not None
        assert check_syntax("a.json", '{"a": 1}') is None

    def test_a_language_with_no_parser_here_is_not_failed(self):
        """It can only block on evidence. No parser means no check, not a
        failed one."""
        assert check_syntax("a.rs", "this is not rust {{{") is None


class TestRevert:
    def test_an_unparseable_edit_is_undone(self, tmp_path, settings):
        """There is no reading under which that is an improvement, and leaving
        it breaks every later tool call on the same file."""
        target = tmp_path / "a.py"
        good = "def f():\n    return 1\n"
        target.write_text("def f(:\n")

        report = verify_change(tmp_path, "a.py", updated="def f(:\n",
                               original=good, settings=settings)
        assert report.reverted
        assert target.read_text() == good

    def test_a_new_file_that_does_not_parse_is_removed(self, tmp_path, settings):
        target = tmp_path / "new.py"
        target.write_text("def (:\n")
        report = verify_change(tmp_path, "new.py", updated="def (:\n",
                               original=None, settings=settings)
        assert report.reverted
        assert not target.exists()

    def test_a_rule_violation_is_reported_not_reverted(self, tmp_path, settings):
        """An edit that breaks an invariant may be the first half of a change
        the next step completes. Reverting it would stop the loop working in
        two steps."""
        (settings.home / "rules" / "r.yaml").write_text(
            "- id: no-todo\n  title: no TODO markers\n  forbid: 'TODO'\n"
        )
        target = tmp_path / "a.py"
        text = "# TODO later\nx = 1\n"
        target.write_text(text)
        report = verify_change(tmp_path, "a.py", updated=text, original="x = 1\n",
                               settings=settings)
        assert not report.reverted
        assert report.blocking
        assert target.read_text() == text


class TestRules:
    def test_a_forbidden_pattern_is_found_with_its_line(self):
        rule = Rule(id="r", title="no eval", forbid=r"\beval\(")
        found = check_rules([rule], "a.py", "x = 1\ny = eval(s)\n")
        assert [v.line for v in found] == [2]

    def test_a_required_pattern_is_only_checked_when_triggered(self):
        rule = Rule(
            id="r", title="handlers await", when=r"async def handler",
            require=r"await",
        )
        assert not check_rules([rule], "a.py", "def other():\n    pass\n")
        assert check_rules([rule], "a.py", "async def handler():\n    send()\n")

    def test_a_rule_only_applies_to_the_paths_it_names(self):
        rule = Rule(id="r", title="t", forbid="x", paths=("*.ts",))
        assert not check_rules([rule], "a.py", "x")
        assert check_rules([rule], "a.ts", "x")

    def test_a_rule_that_cannot_compile_is_skipped_loudly(self, tmp_path):
        """A gate nobody knows has stopped working is worse than no gate."""
        (tmp_path / "r.yaml").write_text(
            "- id: bad\n  title: t\n  forbid: '([unclosed'\n"
            "- id: good\n  title: t\n  forbid: 'ok'\n"
        )
        assert [r.id for r in load_rules([tmp_path])] == ["good"]

    def test_a_rule_without_a_pattern_is_not_a_rule(self, tmp_path):
        (tmp_path / "r.yaml").write_text("- id: x\n  title: just a note\n")
        assert load_rules([tmp_path]) == []

    def test_rules_travel_with_the_repository(self, tmp_path):
        """The invariants of a codebase belong with the codebase."""
        from mimir.verify.rules import rule_roots

        roots = rule_roots(Settings(tmp_path / "home"), tmp_path / "repo")
        assert roots[-1] == tmp_path / "repo" / ".mimir" / "rules"


class TestReporting:
    def test_a_clean_change_says_what_it_checked(self, tmp_path, settings):
        report = verify_change(tmp_path, "a.py", updated="x = 1\n",
                               original=None, settings=settings)
        assert report.ok
        assert "syntax" in report.summary()

    def test_what_could_not_be_checked_is_named(self, tmp_path, settings):
        report = verify_change(tmp_path, "a.rs", updated="fn main() {}\n",
                               original=None, settings=settings)
        assert report.checks_skipped


class TestInsertedBlocksLandAtTheRightDepth:
    """The model decides what to insert and where. It should not also have to
    decide how deep, and when it did, three runs in a row produced a file that
    no longer parsed."""

    SOURCE = [
        "class Glossary:\n",
        "    def lookup(self):\n",
        "        rows = query()\n",
        "        return dict(rows)\n",
        "\n",
        "    def all(self):\n",
        "        return []\n",
    ]

    def _insert(self, after, content):
        import ast

        from mimir.tools.code import _at_depth

        body = _at_depth(self.SOURCE, after, content)
        merged = "".join(self.SOURCE[:after]) + "\n" + body + "".join(self.SOURCE[after:])
        ast.parse(merged)
        return body

    def test_a_method_lands_beside_its_siblings_not_inside_the_one_above(self):
        """The line before the insertion point is inside a method body, so
        following it would nest the new method in that body."""
        body = self._insert(4, "def count(self):\n    return 0")
        assert body.startswith("    def count(self):")
        assert "        return 0" in body

    def test_indentation_the_model_supplied_is_corrected(self):
        body = self._insert(4, "        def count(self):\n            return 0")
        assert body.startswith("    def count(self):")

    def test_relative_indentation_inside_the_block_survives(self):
        body = self._insert(4, "def count(self):\n    if True:\n        return 0")
        assert "        if True:" in body
        assert "            return 0" in body

    def test_a_decorated_method_is_treated_as_a_block(self):
        body = self._insert(4, "@property\ndef count(self):\n    return 0")
        assert body.startswith("    @property")

    def test_a_plain_statement_follows_the_line_before_it(self):
        from mimir.tools.code import _at_depth

        body = _at_depth(self.SOURCE, 3, "total = 0")
        assert body.startswith("        total = 0")
