"""Did the change connect what it added."""

from __future__ import annotations

from mimir.verify import definitions

BEFORE = (
    "MAX = 3\n"
    "\n"
    "\n"
    "def hint(limit: int = 2):\n"
    "    return MAX + limit\n"
)


def _check(after, before=BEFORE):
    return definitions.check(before, after)


class TestTheCaseThisWasWrittenFor:
    def test_a_constant_added_and_never_used_is_dead(self):
        after = BEFORE.replace("MAX = 3\n", "MAX = 3\nDEFAULT_HINT_LIMIT = 2\n")
        issues = _check(after)
        assert [i.kind for i in issues] == ["dead"]
        assert issues[0].name == "DEFAULT_HINT_LIMIT"

    def test_the_same_constant_actually_used_is_clean(self):
        after = BEFORE.replace(
            "MAX = 3\n", "MAX = 3\nDEFAULT_HINT_LIMIT = 2\n"
        ).replace("limit: int = 2", "limit: int = DEFAULT_HINT_LIMIT")
        assert _check(after) == []

    def test_a_constant_added_twice_is_a_duplicate(self):
        after = BEFORE.replace(
            "MAX = 3\n",
            "MAX = 3\nDEFAULT_HINT_LIMIT = 2\nDEFAULT_HINT_LIMIT = 2\n",
        ).replace("limit: int = 2", "limit: int = DEFAULT_HINT_LIMIT")
        assert [i.kind for i in _check(after)] == ["duplicate"]


class TestWhatItDoesNotJudge:
    def test_a_new_file_is_not_judged(self):
        """A module of constants written for its importers looks exactly like a module of constants nobody uses."""
        assert definitions.check(None, "A = 1\nB = 2\n") == []

    def test_something_already_dead_is_not_blamed_on_this_change(self):
        before = "UNUSED = 1\n"
        after = "UNUSED = 1\nOTHER = 2\n\n\ndef f():\n    return OTHER\n"
        assert _check(after, before) == []

    def test_an_exported_name_is_not_dead(self):
        before = '__all__ = ["A"]\nA = 1\n'
        after = '__all__ = ["A", "B"]\nA = 1\nB = 2\n'
        assert _check(after, before) == []

    def test_a_dunder_is_left_alone(self):
        assert _check("MAX = 3\n__version__ = '1'\n\n\ndef hint(limit: int = 2):\n"
                      "    return MAX + limit\n") == []

    def test_a_file_that_does_not_parse_yields_nothing(self):
        """The syntax gate owns that failure; this one has no opinion."""
        assert _check("def f(:\n") == []

    def test_a_name_read_anywhere_at_any_depth_counts_as_used(self):
        after = BEFORE.replace(
            "MAX = 3\n", "MAX = 3\nLIMIT = 2\n"
        ).replace("    return MAX + limit", "    if True:\n        return LIMIT")
        assert _check(after) == []
