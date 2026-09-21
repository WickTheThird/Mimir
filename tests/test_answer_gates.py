"""The gates as the graph actually calls them.

The gate functions have unit tests. This file covers the layer between them
and the graph: which text each gate is shown, what lands in metadata, and
whether they interfere with one another when several could fire.
"""

import pytest

from mimir.graph.nodes import (
    _enforce_grounding,
    _enforce_retry_signature,
    _enforce_sufficiency,
)
from mimir.models.evidence import Evidence, Freshness, SourceType
from mimir.models.specialist import FinalAnswer
from mimir.models.state import InvestigationState


def _session(request: str, excerpts=(), risks=(), freshness=Freshness.LIVE):
    session = InvestigationState(user_request=request)
    session.risks = list(risks)
    session.evidence = [
        Evidence(
            claim="observation",
            source_type=SourceType.COMMAND_OUTPUT,
            source_id=f"cmd-{i}",
            excerpt=text,
            freshness=freshness,
        )
        for i, text in enumerate(excerpts)
    ]
    return session


class TestSufficiencyWiring:
    def test_a_tool_failure_recorded_as_a_risk_reaches_the_gate(self):
        """In production the failure is a recorded tool error, not prose in
        the request. The gate must see the risks list."""
        session = _session(
            "is there a billing pod?",
            risks=["kubectl get pods: connection refused on all 3 contexts"],
        )
        answer = _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session
        )
        assert "Unknown" in answer.answer
        assert session.metadata["sufficiency"]["retrieval"] == "failed"

    def test_the_gate_records_what_it_saw_even_when_it_does_not_fire(self):
        """A gate that only writes metadata when it acts cannot be
        distinguished afterwards from one that never ran."""
        session = _session("is there a billing pod?")
        _enforce_sufficiency(
            FinalAnswer(answer="The pod is running.", confidence=0.9), session
        )
        assert session.metadata["sufficiency"]["retrieval"] == "unknown"
        assert session.metadata["sufficiency"]["overreaching"] is False

    def test_an_executed_command_counts_as_having_looked(self):
        session = _session("how many replicas?", excerpts=["replicas: 6"])
        answer = _enforce_sufficiency(
            FinalAnswer(answer="There are 6 replicas running.", confidence=0.8),
            session,
        )
        assert answer.confidence == 0.8


class TestCurrencyWiring:
    def test_stale_evidence_freshness_demotes_without_any_prose_marker(self):
        """The structured verdict is the production signal. Nothing in the
        request says anything about age."""
        session = _session(
            "how many replicas does payments run?",
            excerpts=["payments replicas: 6"],
            freshness=Freshness.STALE,
        )
        answer = _enforce_sufficiency(
            FinalAnswer(answer="Payments runs 6 replicas.", confidence=0.9), session
        )
        assert "erify" in answer.answer
        assert session.metadata["sufficiency"]["currency"] == "stale"

    def test_live_evidence_is_left_alone(self):
        session = _session(
            "how many replicas?",
            excerpts=["payments replicas: 6"],
            freshness=Freshness.LIVE,
        )
        answer = _enforce_sufficiency(
            FinalAnswer(answer="Payments runs 6 replicas.", confidence=0.9), session
        )
        assert answer.confidence == 0.9
        assert "erify" not in answer.answer


class TestGroundingWiring:
    def test_a_name_in_no_evidence_and_no_request_is_demoted(self):
        session = _session("which pods are running?", excerpts=["No resources found."])
        answer = _enforce_grounding(
            FinalAnswer(
                answer="messaging-router-6cf8 is running.",
                observed_facts=["messaging-router-6cf8 is running"],
                confidence=0.9,
            ),
            session,
        )
        assert answer.observed_facts == []
        assert session.metadata["grounding"]["ungrounded"] >= 1

    def test_a_name_present_in_evidence_survives(self):
        session = _session(
            "which pods are running?",
            excerpts=["NAME  READY\nmessaging-router-6cf8  1/1  Running"],
        )
        answer = _enforce_grounding(
            FinalAnswer(
                answer="messaging-router-6cf8 is running.",
                observed_facts=["messaging-router-6cf8 is running"],
                confidence=0.9,
            ),
            session,
        )
        assert answer.observed_facts
        assert answer.confidence == 0.9

    def test_a_name_the_operator_used_is_not_an_invention(self):
        session = _session("find messaging-whatsapp pods in messaging-squad")
        answer = _enforce_grounding(
            FinalAnswer(answer="No messaging-whatsapp pods found.", confidence=0.9),
            session,
        )
        assert answer.confidence == 0.9


class TestRetryWiring:
    def test_an_answer_not_blaming_retries_costs_nothing_to_check(self):
        """The gate returns before assembling observations, so the common
        case does no work."""
        session = _session("why is the pool exhausted?")
        _enforce_retry_signature(
            FinalAnswer(answer="A batch job opened 400 connections.", confidence=0.8),
            session,
        )
        assert "retry_signature" not in session.metadata

    def test_a_retry_claim_without_the_pattern_is_withdrawn(self):
        session = _session(
            "A service logs the same request id once, and a connection pool "
            "exhaustion message appears 20 seconds later."
        )
        answer = _enforce_retry_signature(
            FinalAnswer(answer="The pool was exhausted by retries.", confidence=0.9),
            session,
        )
        assert "not supported by the timing" in answer.answer
        assert session.metadata["retry_signature"]["storm"] is False

    def test_a_retry_claim_with_the_pattern_stands(self):
        session = _session(
            "A service logs the same request id three times in 90 seconds "
            "with gaps of 1s, 2s and 4s."
        )
        answer = _enforce_retry_signature(
            FinalAnswer(answer="The pool was exhausted by retries.", confidence=0.9),
            session,
        )
        assert answer.confidence == 0.9
        assert session.metadata["retry_signature"]["storm"] is True


class TestGatesTogether:
    def test_a_sound_answer_passes_all_three_untouched(self):
        """The gates must not tax the common case. If a correct answer loses
        confidence by passing through them, the sweep will read as a
        regression and the cause will not be obvious."""
        session = _session(
            "which pods are running in messaging-squad?",
            excerpts=["NAME  READY\nmessaging-router-6cf8  1/1  Running"],
        )
        answer = FinalAnswer(
            answer="messaging-router-6cf8 is running.",
            observed_facts=["messaging-router-6cf8 is running"],
            confidence=0.85,
        )
        for gate in (_enforce_sufficiency, _enforce_grounding,
                     _enforce_retry_signature):
            answer = gate(answer, session)
        assert answer.confidence == 0.85
        assert answer.observed_facts == ["messaging-router-6cf8 is running"]
        assert answer.answer == "messaging-router-6cf8 is running."

    def test_two_gates_firing_do_not_erase_each_other(self):
        session = _session(
            "is there a billing pod?",
            risks=["kubectl: connection refused"],
        )
        answer = FinalAnswer(
            answer="There is no billing pod; messaging-router-6cf8 handles it.",
            observed_facts=["messaging-router-6cf8 handles billing"],
            confidence=0.9,
        )
        answer = _enforce_sufficiency(answer, session)
        answer = _enforce_grounding(answer, session)
        # Sufficiency replaces the prose, so the invented name is gone from the
        # text before grounding reads it and there is nothing left to warn
        # about there. The demotion still happened: both claims are in
        # unverified, which is where a claim the evidence cannot carry belongs.
        assert "Unknown" in answer.answer
        assert "messaging-router-6cf8" not in answer.answer
        unverified = " ".join(answer.unverified)
        assert "no billing pod" in unverified
        assert "names nothing that was read" in unverified
        assert answer.confidence <= 0.3
