"""You cannot prove absence from a search that did not run."""

from mimir.models.specialist import FinalAnswer
from mimir.verify.sufficiency import (
    Currency,
    classify_currency,
    demote_stale,
    Retrieval,
    check,
    classify_retrieval,
    demote_overreach,
)

COMPLETE = (
    'A search across every namespace of three reachable clusters returned no '
    'pod whose name contains "billing". All three clusters answered '
    "successfully."
)
FAILED = (
    'A search across every namespace of three clusters returned no pod whose '
    'name contains "billing". All three clusters returned connection timeouts '
    "and no listing was produced."
)


def test_a_completed_search_that_found_nothing_is_a_finding():
    assert classify_retrieval(observations=COMPLETE) is Retrieval.EMPTY


def test_a_search_that_could_not_run_is_not_a_finding():
    """The two prompts differ by one fact and must not classify alike."""
    assert classify_retrieval(observations=FAILED) is Retrieval.FAILED


def test_failure_beats_completion_when_both_appear():
    """A partial search cannot establish that the unreached targets are empty.

    Treating some targets answering as a complete search is the exact error
    this module exists to stop.
    """
    blob = "Two clusters answered successfully. The third timed out."
    assert classify_retrieval(observations=blob) is Retrieval.FAILED


def test_silence_about_looking_is_not_evidence_of_looking():
    assert classify_retrieval(observations="The service is important.") is (
        Retrieval.UNKNOWN
    )


def test_tool_errors_reach_the_classifier_through_risks():
    """In production the marker is a recorded tool failure, not prose."""
    assert (
        classify_retrieval(observations="", risks="kubectl: connection refused")
        is Retrieval.FAILED
    )


def test_an_absence_claim_on_a_failed_search_is_overreach():
    result = check("There is no billing pod.", observations=FAILED)
    assert result.retrieval is Retrieval.FAILED
    assert result.absence_claims == ["There is no billing pod."]
    assert result.overreaching


def test_the_same_claim_on_a_completed_search_is_fine():
    result = check("There is no billing pod.", observations=COMPLETE)
    assert result.retrieval is Retrieval.EMPTY
    assert not result.overreaching


def test_a_presence_claim_also_needs_a_search_that_ran():
    result = check("There is a billing pod.", observations=FAILED)
    assert result.overreaching


def test_an_answer_making_no_existence_claim_is_left_alone():
    result = check("Restart counts rose after the deploy.", observations=FAILED)
    assert not result.overreaching


def test_demotion_rewrites_the_prose_not_only_the_bullets():
    """A demoted bullet under intact prose leaves the wrong conclusion in
    the line the operator actually reads."""
    answer = FinalAnswer(
        answer="There is no billing pod.",
        observed_facts=["There is no billing pod in any namespace."],
        unverified=[],
        confidence=0.9,
    )
    answer, moved = demote_overreach(answer, check(answer.answer, observations=FAILED))
    assert moved
    assert "Unknown" in answer.answer
    assert answer.observed_facts == []
    assert any("billing" in u for u in answer.unverified)
    assert answer.confidence <= 0.3


def test_demotion_keeps_the_claim_visible_rather_than_deleting_it():
    """The claim may be true. What is false is presenting it as established."""
    answer = FinalAnswer(
        answer="There is no billing pod.",
        observed_facts=[],
        unverified=[],
        confidence=0.9,
    )
    answer, _ = demote_overreach(answer, check(answer.answer, observations=FAILED))
    assert any("no billing pod" in u for u in answer.unverified)
    assert "did not complete" in " ".join(answer.unverified)


def test_a_sound_answer_passes_through_untouched():
    answer = FinalAnswer(
        answer="There is no billing pod.",
        observed_facts=["Three clusters listed no billing pod."],
        unverified=[],
        confidence=0.9,
    )
    before = answer.answer
    answer, moved = demote_overreach(answer, check(before, observations=COMPLETE))
    assert moved == 0
    assert answer.answer == before
    assert answer.confidence == 0.9


def test_the_gate_reads_the_field_the_real_answer_actually_has():
    """Written after a SimpleNamespace stand-in accepted a field name that
    FinalAnswer does not have, so the unit tests passed and the graph raised
    AttributeError on the first real run."""
    answer = FinalAnswer(answer="There is no billing pod.", confidence=0.9)
    assert not hasattr(answer, "summary")
    answer, moved = demote_overreach(
        answer, check(answer.answer, observations=FAILED)
    )
    assert moved and "Unknown" in answer.answer


VERIFIED = (
    "A stored note says the payments service runs 6 replicas. It was last "
    "verified 3 days ago. No live check has been run."
)
NEVER = (
    "A stored note says the payments service runs 6 replicas. It has never "
    "been verified and was written 14 months ago. No live check has been run."
)


class TestCurrency:
    """A note nobody checked in fourteen months is not a current fact."""

    def test_a_recently_verified_note_is_current(self):
        assert classify_currency(observations=VERIFIED) is Currency.CURRENT

    def test_a_never_verified_fourteen_month_old_note_is_stale(self):
        """The twin of the case above, differing in one fact."""
        assert classify_currency(observations=NEVER) is Currency.STALE

    def test_an_age_in_months_is_always_past_the_window(self):
        assert classify_currency(observations="written 8 months ago") is Currency.STALE

    def test_an_age_in_days_is_compared_against_the_window(self):
        assert classify_currency(observations="checked 2 days ago") is Currency.CURRENT
        assert classify_currency(observations="checked 90 days ago") is Currency.STALE

    def test_a_live_execution_settles_it_regardless_of_the_notes(self):
        assert classify_currency(observations=NEVER, executions=1) is Currency.CURRENT

    def test_structured_freshness_is_preferred_over_reading_prose(self):
        assert classify_currency(freshness=["stale", "unknown"]) is Currency.STALE
        assert classify_currency(freshness=["live"]) is Currency.CURRENT

    def test_saying_nothing_about_age_is_not_a_claim_of_freshness(self):
        assert classify_currency(observations="It runs 6 replicas.") is (
            Currency.UNKNOWN
        )


class TestStaleDemotion:
    def _answer(self, text, facts=()):
        return FinalAnswer(answer=text, observed_facts=list(facts), confidence=0.9)

    def test_a_stale_value_is_reported_with_an_instruction_to_verify(self):
        answer = self._answer("The payments service runs 6 replicas.",
                              ["payments runs 6 replicas"])
        answer, moved = demote_stale(answer, Currency.STALE)
        assert moved
        assert "erify" in answer.answer
        assert answer.observed_facts == []
        assert any("6 replicas" in u for u in answer.unverified)

    def test_the_remembered_value_stays_visible(self):
        """It is the most useful thing available. What changes is its status."""
        answer = self._answer("The payments service runs 6 replicas.")
        answer, _ = demote_stale(answer, Currency.STALE)
        assert "6 replicas" in answer.answer

    def test_an_answer_that_already_says_verify_is_left_alone(self):
        """Appending a second instruction to an answer that gave the right one
        reads as a system that does not understand its own output."""
        text = "The note says 6 replicas. Verify against the live system."
        answer = self._answer(text)
        answer, moved = demote_stale(answer, Currency.STALE)
        assert moved == 0
        assert answer.answer == text

    def test_a_current_value_passes_through(self):
        answer = self._answer("6 replicas.", ["6 replicas"])
        answer, moved = demote_stale(answer, Currency.CURRENT)
        assert moved == 0
        assert answer.observed_facts == ["6 replicas"]


def test_the_freshness_window_comes_from_config_not_from_a_literal():
    """The first version read settings.memory.stale_after_days, which does not
    exist, so getattr returned the default written beside it and the gate ran
    on a 30-day window with nothing logged."""
    from mimir.config import get_settings
    from mimir.graph.nodes import _stale_after_days

    assert _stale_after_days() == get_settings().knowledge.stale_after_days


class TestTheGateDoesNotFireOnEverything:
    """A gate that fires on nearly every case is not a gate."""

    def test_silence_about_looking_does_not_demote(self):
        """47 of the 52 model cases in this corpus classify as unknown,
        because a prompt describing a situation rarely narrates whether a
        search ran. Demoting on unknown fired on almost all of them."""
        result = check("The pod is running.", observations="The pod is important.")
        assert result.retrieval is Retrieval.UNKNOWN
        assert not result.overreaching

    def test_a_routine_presence_claim_survives_an_unnarrated_situation(self):
        answer = FinalAnswer(answer="There are 3 replicas running.", confidence=0.8)
        before = answer.answer
        answer, moved = demote_overreach(
            answer, check(before, observations="Replica counts were checked.")
        )
        assert moved == 0
        assert answer.answer == before

    def test_only_an_explicit_failure_to_look_demotes(self):
        assert check("There is no billing pod.", observations=FAILED).overreaching

    def test_the_corpus_would_not_be_demoted_wholesale(self):
        """The guard that would have caught this before the sweep."""
        from mimir.eval.harness import EvalHarness

        fires = sum(
            1
            for case in EvalHarness.load_corpus()
            if not case.deterministic
            and check("The pod is running.", observations=case.prompt).overreaching
        )
        assert fires <= 5, f"gate fires on {fires} model cases, which is not a gate"


def test_no_gate_fires_on_a_large_fraction_of_the_corpus():
    """The guard that caught the sufficiency gate demoting 47 of 52 cases.

    A gate is meant to be the exception. One that fires on most cases is
    either measuring something other than what it claims, or the corpus is
    uniformly broken in a way that deserves its own investigation. Either
    way it must not reach a sweep unexamined.
    """
    from mimir.eval.harness import EvalHarness
    from mimir.verify.grounding import check as grounding_check
    from mimir.verify.patterns import claims_retries

    cases = [c for c in EvalHarness.load_corpus() if not c.deterministic]
    budget = len(cases) // 5

    overreach = sum(
        1
        for c in cases
        if check("The pod is running.", observations=c.prompt).overreaching
    )
    stale = sum(
        1 for c in cases if classify_currency(observations=c.prompt) is Currency.STALE
    )
    # Echoing back what the operator said must never read as an invention.
    echoed = sum(
        1 for c in cases if not grounding_check(c.prompt, "", asked=c.prompt).ok
    )
    retry = sum(1 for c in cases if claims_retries(c.prompt))

    assert overreach <= budget, f"sufficiency fires on {overreach}/{len(cases)}"
    assert stale <= budget, f"currency fires on {stale}/{len(cases)}"
    assert echoed == 0, f"grounding flags the operator's own words on {echoed}"
    assert retry <= budget, f"retry is eligible on {retry}/{len(cases)}"
