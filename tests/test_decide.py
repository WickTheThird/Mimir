"""Typed decisions.

A discriminative model scores the answers you allow rather than writing one.
These pin the boundaries, because the risk of adding a model to a system built
on rules is that the model quietly starts deciding things the rules were
deciding correctly.
"""

from __future__ import annotations

import pytest

from mimir.decide import MAX_OPTIONS, Choice, NoDecider, Verdict, clip
from mimir.decide.backends import _from_kev


class TestTheClosedAnswerSet:
    def test_more_options_than_the_model_takes_is_refused(self):
        """Silently truncating drops the option that was correct and returns a
        confident answer from a smaller world."""
        with pytest.raises(ValueError, match="1 to 26"):
            Choice("x", tuple(str(i) for i in range(MAX_OPTIONS + 1)))

    def test_duplicates_are_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            Choice("x", ("a", "a"))

    def test_a_probability_over_an_option_never_offered_is_dropped(self):
        """It is not a decision about this question, whatever it is."""
        field = Choice("tool", ("get_logs", "answer"))
        found = _from_kev(
            {"tool": {"type": "choice",
                      "probabilities": {"get_logs": 0.4, "answer": 0.2, "rm_rf": 0.9}}},
            [field], False)
        assert set(found["tool"].distribution) == {"get_logs", "answer"}
        assert found["tool"].choice == "get_logs"

    def test_what_survives_is_renormalised(self):
        field = Choice("tool", ("a", "b"))
        found = _from_kev(
            {"tool": {"type": "choice", "probabilities": {"a": 0.2, "b": 0.2, "c": 0.6}}},
            [field], False)
        assert sum(found["tool"].distribution.values()) == pytest.approx(1.0)


class TestAbsenceIsNotAVerdict:
    def test_no_decider_answers_nothing(self):
        """A missing decision model must not look like a decision."""
        assert NoDecider().decide("anything", [Choice("x", ("a", "b"))]) == {}
        assert NoDecider().available is False

    def test_an_unreachable_server_yields_nothing_rather_than_raising(self):
        from mimir.decide import KevDecider

        decider = KevDecider("http://127.0.0.1:9")
        assert decider.available is False
        assert decider.decide("x", [Choice("f", ("a", "b"))]) == {}


class TestUncertaintyIsThePoint:
    def test_margin_measures_distance_from_the_runner_up(self):
        """A win by a nose is not a decision, and on a two-way choice the
        argmax is always something."""
        close = Verdict("f", "a", 0.51, {"a": 0.51, "b": 0.49})
        clear = Verdict("f", "a", 0.95, {"a": 0.95, "b": 0.05})
        assert close.margin == pytest.approx(0.02)
        assert clear.margin == pytest.approx(0.90)

    def test_a_single_option_is_certain_by_construction(self):
        assert Verdict("f", "a", 1.0, {"a": 1.0}).margin == 1.0


class TestTruncation:
    def test_a_context_that_fits_is_untouched(self):
        text, cut = clip("short")
        assert text == "short" and not cut

    def test_a_long_context_keeps_the_end_and_says_so(self):
        """The question and the material it is about are at the end; the
        preamble is what can go."""
        from mimir.decide import MAX_CONTEXT_CHARS

        text, cut = clip("A" * 100 + "B" * MAX_CONTEXT_CHARS)
        assert cut
        assert text.endswith("B")
        assert len(text) == MAX_CONTEXT_CHARS

    def test_a_verdict_carries_whether_it_saw_everything(self):
        found = _from_kev(
            {"f": {"type": "choice", "probabilities": {"a": 1.0}}},
            [Choice("f", ("a", "b"))], True)
        assert found["f"].truncated
        assert "truncated" in found["f"].render()


class TestItSitsBelowTheFacts:
    def test_a_scored_belief_cannot_outrank_an_observed_one(self):
        """A model that is right ninety percent of the time must not overrule a
        check that is right every time."""
        from mimir.agent.select import Candidate

        observed, believed = Candidate(0, "a", "/x"), Candidate(1, "b", "/x")
        for c in (observed, believed):
            c.files_changed, c.lines_added = 1, 5
        observed.tests_ran = observed.tests_passed = True
        believed.satisfies = 0.99
        assert observed.score > believed.score

    def test_it_breaks_a_tie_the_facts_could_not(self):
        from mimir.agent.select import Candidate

        complete, incomplete = Candidate(0, "a", "/x"), Candidate(1, "b", "/x")
        for c in (complete, incomplete):
            c.files_changed, c.lines_added = 1, 5
            c.tests_ran = c.tests_passed = True
        complete.satisfies, incomplete.satisfies = 0.95, 0.10
        assert complete.score > incomplete.score

    def test_an_unjudged_candidate_is_not_penalised(self):
        """No decision model means fall back to what came before, not treat
        every candidate as having failed."""
        from mimir.agent.select import Candidate

        judged, unjudged = Candidate(0, "a", "/x"), Candidate(1, "b", "/x")
        for c in (judged, unjudged):
            c.files_changed, c.lines_added = 1, 5
        judged.satisfies = 0.0
        assert unjudged.score > judged.score


class TestTheKevContract:
    """Written against the published API rather than guessed at. The first
    version invented a /decide endpoint and a flat schema; Kev takes a state
    and a map of questions typed noul, choice or score."""

    def test_a_two_option_field_becomes_a_noul_question(self):
        from mimir.decide.backends import _criteria, _is_boolean

        field = Choice("satisfies", ("yes", "no"), "Did it do the job?")
        assert _is_boolean(field)
        assert _criteria(field) == {"true": "yes", "false": "no"}

    def test_a_many_option_field_becomes_a_choice_question(self):
        from mimir.decide.backends import _criteria, _is_boolean

        field = Choice("tool", ("get_logs", "get_events", "answer"))
        assert not _is_boolean(field)
        assert set(_criteria(field)) == {"get_logs", "get_events", "answer"}

    def test_a_noul_probability_becomes_a_distribution(self):
        from mimir.decide.backends import _from_kev

        field = Choice("satisfies", ("yes", "no"))
        found = _from_kev({"satisfies": {"type": "noul", "noul": 0.93}}, [field], False)
        assert found["satisfies"].choice == "yes"
        assert found["satisfies"].probability == pytest.approx(0.93)
        assert found["satisfies"].distribution["no"] == pytest.approx(0.07)

    def test_a_low_noul_answers_the_other_way(self):
        from mimir.decide.backends import _from_kev

        field = Choice("satisfies", ("yes", "no"))
        found = _from_kev({"satisfies": {"type": "noul", "noul": 0.11}}, [field], False)
        assert found["satisfies"].choice == "no"

    def test_a_field_the_server_did_not_answer_is_absent(self):
        """Absent is not a verdict, here as everywhere else."""
        from mimir.decide.backends import _from_kev

        assert _from_kev({}, [Choice("f", ("a", "b"))], False) == {}
