"""A retry storm has a shape. One request is not a storm."""

from mimir.models.specialist import FinalAnswer
from mimir.verify.patterns import (
    MIN_ATTEMPTS,
    RetryEvidence,
    claims_retries,
    demote_unsupported_retry,
    from_text,
    from_timestamps,
)

STORM = (
    "A service logs the same request id three times in 90 seconds with gaps "
    "of 1s, 2s and 4s, and a connection pool exhaustion message appears 20 "
    "seconds later."
)
INNOCENT = (
    "A service logs the same request id once, and a connection pool "
    "exhaustion message appears 20 seconds later. Separately, a batch job "
    "opened 400 connections."
)


class TestSignature:
    def test_backoff_is_recognised(self):
        assert from_text(STORM).is_storm

    def test_a_single_request_is_not_a_storm(self):
        """The twin of the case above, differing in one fact."""
        assert not from_text(INNOCENT).is_storm

    def test_every_gap_in_the_list_is_read(self):
        """A non-greedy pattern read "1s, 2s and 4s" as two gaps. The verdict
        was right anyway, which is how a parsing bug survives a passing test."""
        assert from_text(STORM).gaps == [1.0, 2.0, 4.0]

    def test_evenly_spaced_repeats_are_not_backoff(self):
        """A poller is not a retry storm."""
        assert not from_text(
            "the same request id four times with gaps of 5s, 5s and 5s"
        ).is_storm

    def test_two_attempts_are_never_enough(self):
        """A single gap cannot fail to be consistent with doubling, so a
        two-attempt threshold would pass on any pair of lines in any log."""
        assert MIN_ATTEMPTS >= 3
        assert not from_timestamps([0.0, 1.0]).is_storm

    def test_the_production_path_reads_real_timestamps(self):
        assert from_timestamps([100.0, 101.0, 103.0, 107.0]).is_storm

    def test_jitter_does_not_reject_a_real_storm(self):
        assert from_timestamps([0.0, 1.0, 2.6, 6.1]).is_storm

    def test_units_are_converted_before_comparing(self):
        assert from_text(
            "the same request id three times with gaps of 500ms, 1s and 2s"
        ).gaps == [0.5, 1.0, 2.0]


class TestDemotion:
    def _answer(self, text, facts=()):
        return FinalAnswer(answer=text, observed_facts=list(facts), confidence=0.9)

    def test_a_retry_diagnosis_without_the_signature_is_withdrawn(self):
        answer = self._answer(
            "The pool was exhausted by client retries.",
            ["retries exhausted the connection pool"],
        )
        answer, moved = demote_unsupported_retry(answer, from_text(INNOCENT))
        assert moved
        assert "not supported by the timing" in answer.answer
        assert answer.observed_facts == []
        assert answer.confidence <= 0.35

    def test_the_reason_names_which_test_failed(self):
        answer = self._answer("Caused by a retry storm.")
        answer, _ = demote_unsupported_retry(answer, from_text(INNOCENT))
        assert "only 1 occurrence" in answer.answer

    def test_flat_gaps_get_a_different_reason_than_too_few_attempts(self):
        answer = self._answer("Caused by a retry storm.")
        answer, _ = demote_unsupported_retry(
            answer, RetryEvidence(occurrences=4, gaps=[5.0, 5.0, 5.0])
        )
        assert "do not widen" in answer.answer

    def test_a_supported_retry_diagnosis_is_left_alone(self):
        answer = self._answer("Caused by a retry storm.", ["gaps of 1s, 2s, 4s"])
        answer, moved = demote_unsupported_retry(answer, from_text(STORM))
        assert moved == 0
        assert answer.confidence == 0.9

    def test_an_answer_blaming_something_else_is_untouched(self):
        answer = self._answer("A batch job opened 400 connections.")
        answer, moved = demote_unsupported_retry(answer, from_text(INNOCENT))
        assert moved == 0

    def test_the_gate_never_promotes_on_a_match(self):
        """A present signature is consistent with retries causing the
        incident. It does not establish it."""
        answer = self._answer("A batch job opened 400 connections.", ["batch job"])
        answer, moved = demote_unsupported_retry(answer, from_text(STORM))
        assert moved == 0
        assert not claims_retries(answer.answer)
