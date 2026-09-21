"""You cannot prove absence from a search that did not run."""

from mimir.models.specialist import FinalAnswer
from mimir.verify.sufficiency import (
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
