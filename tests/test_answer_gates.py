"""The gates as the graph actually calls them."""

import pytest

from mimir.decide.base import Verdict

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
    @pytest.mark.asyncio
    async def test_a_tool_failure_recorded_as_a_risk_reaches_the_gate(self):
        """In production the failure is a recorded tool error, not prose in the request."""
        session = _session(
            "is there a billing pod?",
            risks=["kubectl get pods: connection refused on all 3 contexts"],
        )
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session
        )
        assert "Unknown" in answer.answer
        assert session.metadata["sufficiency"]["retrieval"] == "failed"

    @pytest.mark.asyncio
    async def test_the_gate_records_what_it_saw_even_when_it_does_not_fire(self):
        """A gate that only writes metadata when it acts cannot be distinguished afterwards from one that never ran."""
        session = _session("is there a billing pod?")
        await _enforce_sufficiency(
            FinalAnswer(answer="The pod is running.", confidence=0.9), session
        )
        assert session.metadata["sufficiency"]["retrieval"] == "unknown"
        assert session.metadata["sufficiency"]["overreaching"] is False

    @pytest.mark.asyncio
    async def test_an_executed_command_counts_as_having_looked(self):
        session = _session("how many replicas?", excerpts=["replicas: 6"])
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There are 6 replicas running.", confidence=0.8),
            session,
        )
        assert answer.confidence == 0.8


class TestCurrencyWiring:
    @pytest.mark.asyncio
    async def test_stale_evidence_freshness_demotes_without_any_prose_marker(self):
        """The structured verdict is the production signal."""
        session = _session(
            "how many replicas does payments run?",
            excerpts=["payments replicas: 6"],
            freshness=Freshness.STALE,
        )
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="Payments runs 6 replicas.", confidence=0.9), session
        )
        assert "erify" in answer.answer
        assert session.metadata["sufficiency"]["currency"] == "stale"

    @pytest.mark.asyncio
    async def test_live_evidence_is_left_alone(self):
        session = _session(
            "how many replicas?",
            excerpts=["payments replicas: 6"],
            freshness=Freshness.LIVE,
        )
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="Payments runs 6 replicas.", confidence=0.9), session
        )
        assert answer.confidence == 0.9
        assert "erify" not in answer.answer


class TestGroundingWiring:
    @pytest.mark.asyncio
    async def test_a_name_in_no_evidence_and_no_request_is_demoted(self):
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

    @pytest.mark.asyncio
    async def test_a_name_present_in_evidence_survives(self):
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

    @pytest.mark.asyncio
    async def test_a_name_the_operator_used_is_not_an_invention(self):
        session = _session("find messaging-whatsapp pods in messaging-squad")
        answer = _enforce_grounding(
            FinalAnswer(answer="No messaging-whatsapp pods found.", confidence=0.9),
            session,
        )
        assert answer.confidence == 0.9


class TestRetryWiring:
    @pytest.mark.asyncio
    async def test_an_answer_not_blaming_retries_costs_nothing_to_check(self):
        """The gate returns before assembling observations, so the common case does no work."""
        session = _session("why is the pool exhausted?")
        _enforce_retry_signature(
            FinalAnswer(answer="A batch job opened 400 connections.", confidence=0.8),
            session,
        )
        assert "retry_signature" not in session.metadata

    @pytest.mark.asyncio
    async def test_a_retry_claim_without_the_pattern_is_withdrawn(self):
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

    @pytest.mark.asyncio
    async def test_a_retry_claim_with_the_pattern_stands(self):
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
    @pytest.mark.asyncio
    async def test_a_sound_answer_passes_all_three_untouched(self):
        """The gates must not tax the common case."""
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
            answer = await gate(answer, session) if gate is _enforce_sufficiency else gate(answer, session)
        assert answer.confidence == 0.85
        assert answer.observed_facts == ["messaging-router-6cf8 is running"]
        assert answer.answer == "messaging-router-6cf8 is running."

    @pytest.mark.asyncio
    async def test_two_gates_firing_do_not_erase_each_other(self):
        session = _session(
            "is there a billing pod?",
            risks=["kubectl: connection refused"],
        )
        answer = FinalAnswer(
            answer="There is no billing pod; messaging-router-6cf8 handles it.",
            observed_facts=["messaging-router-6cf8 handles billing"],
            confidence=0.9,
        )
        answer = await _enforce_sufficiency(answer, session)
        answer = _enforce_grounding(answer, session)
        # Sufficiency replaces the prose, so the invented name is gone from the
        assert "Unknown" in answer.answer
        assert "messaging-router-6cf8" not in answer.answer
        unverified = " ".join(answer.unverified)
        assert "no billing pod" in unverified
        assert "names nothing that was read" in unverified
        assert answer.confidence <= 0.3


class FakeDecider:
    available = True
    name = "fake"

    def __init__(self, choice, probability=0.9):
        self.choice, self.probability = choice, probability
        self.contexts = []

    def decide(self, context, fields):
        self.contexts.append(context)
        others = [o for o in fields[0].options if o != self.choice]
        dist = {self.choice: self.probability, **{o: (1 - self.probability) / len(others) for o in others}}
        return {fields[0].name: Verdict(field=fields[0].name, choice=self.choice,
                                        probability=self.probability, distribution=dist)}


class Deps:
    def __init__(self, decider=None):
        self.decider = decider


class TestRetrievalDecision:
    """ADR-004 step 1: the first decision out of the generative model."""

    @pytest.mark.asyncio
    async def test_exit_codes_settle_it_and_the_decider_is_not_asked(self):
        from mimir.models.command import CommandOutcome, ExecutionRecord

        session = _session("is there a billing pod?")
        session.commands_executed = [
            ExecutionRecord(command_id="c1", argv=["kubectl", "get", "pods"],
                            outcome=list(CommandOutcome)[0], exit_code=1, stderr="refused")
        ]
        decider = FakeDecider("observed")
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session, Deps(decider)
        )
        assert session.metadata["sufficiency"]["retrieval"] == "failed"
        assert session.metadata["sufficiency"]["retrieval_source"] == "executions"
        assert decider.contexts == []
        assert "Unknown" in answer.answer

    @pytest.mark.asyncio
    async def test_the_decider_replaces_the_regex_when_nothing_ran(self):
        session = _session("The clusters were slow today. Is there a billing pod?")
        decider = FakeDecider("failed")
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session, Deps(decider)
        )
        assert session.metadata["sufficiency"]["retrieval_source"] == "decider"
        assert "Unknown" in answer.answer
        assert "Operator request:" in decider.contexts[0]

    @pytest.mark.asyncio
    async def test_a_decider_verdict_of_empty_keeps_a_negative_answer_and_leads_with_none(self):
        """Nothing demoted, nothing removed; the computed verdict word leads."""
        session = _session("Is there a billing pod?")
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session,
            Deps(FakeDecider("empty")),
        )
        assert answer.answer.startswith("None:")
        assert answer.answer.endswith("There is no billing pod.")
        assert answer.confidence == 0.9
        assert answer.unverified == []

    @pytest.mark.asyncio
    async def test_every_decision_is_logged_with_its_provenance(self):
        session = _session("Is there a billing pod?")
        await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session,
            Deps(FakeDecider("empty", 0.8)),
        )
        (entry,) = session.metadata["decisions"]
        assert entry["field"] == "retrieval"
        assert entry["choice"] == "empty"
        assert entry["probability"] == 0.8
        assert entry["backend"] == "fake"
        assert entry["options"] == ["observed", "empty", "failed"]

    @pytest.mark.asyncio
    async def test_no_decider_falls_back_to_the_text_rule(self):
        session = _session(
            'All three clusters returned connection timeouts and no listing was produced. '
            'Is there a billing pod?'
        )
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session, Deps(None)
        )
        assert session.metadata["sufficiency"]["retrieval_source"] == "statement"
        assert "Unknown" in answer.answer

    @pytest.mark.asyncio
    async def test_the_decider_is_shown_recorded_failures_not_runbooks(self):
        """The regex regression came from feeding retrieved documents to the classifier."""
        session = _session("Is there a billing pod?", excerpts=["Runbook: timeouts happen when..."],
                           risks=["kubernetes_investigator failed: iteration limit"])
        decider = FakeDecider("failed")
        await _enforce_sufficiency(FinalAnswer(answer="ok", confidence=0.5), session, Deps(decider))
        assert "Runbook" not in decider.contexts[0]
        assert "iteration limit" in decider.contexts[0]


class TestOrderOfAuthority:
    @pytest.mark.asyncio
    async def test_an_explicit_statement_beats_a_wrong_decider(self):
        """Kev read 'no listing was produced' as observed at p=0.58 on the first measured run."""
        session = _session(
            "All three clusters returned connection timeouts and no listing was "
            "produced. Is there a billing pod?"
        )
        decider = FakeDecider("observed", 0.58)
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session, Deps(decider)
        )
        assert session.metadata["sufficiency"]["retrieval_source"] == "statement"
        assert decider.contexts == []
        assert "Unknown" in answer.answer

    @pytest.mark.asyncio
    async def test_a_verdict_by_a_nose_is_not_acted_on_even_uncalibrated(self):
        session = _session("Is there a billing pod?")
        decider = FakeDecider("failed", 0.4)  # margin 0.1 on three options
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="There is no billing pod.", confidence=0.9), session, Deps(decider)
        )
        assert session.metadata["decisions"][0]["acted"] is False
        assert answer.answer == "There is no billing pod."

    @pytest.mark.asyncio
    async def test_an_empty_listing_answer_says_none_in_the_graph(self):
        session = _session("A listing returned no pods at all. Which pods are running?")
        answer = await _enforce_sufficiency(
            FinalAnswer(answer="I cannot determine which pods are running.", confidence=0.4),
            session, Deps(None),
        )
        assert answer.answer.startswith("None.")
