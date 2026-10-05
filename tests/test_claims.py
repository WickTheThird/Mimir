"""Claim support tests."""

from __future__ import annotations

from mimir.models.evidence import Citation, Evidence, SourceType
from mimir.verify.claims import (
    ClaimKind,
    check_answer,
    check_claim,
    demote_unsupported,
    subjects_of,
)


def _evidence(claim="", source_id="", excerpt="", supports=True,
              source_type=SourceType.REPOSITORY, path=None):
    citations = [Citation(source_type=source_type, locator=path, path=path)] if path else []
    return Evidence(
        claim=claim, source_id=source_id, excerpt=excerpt, supports=supports,
        source_type=source_type, citations=citations,
    )


class Answer:
    def __init__(self, observed=None, inferences=None, unverified=None,
                 citations=None, confidence=0.8):
        self.observed_facts = observed or []
        self.inferences = inferences or []
        self.unverified = unverified or []
        self.citations = citations or []
        self.confidence = confidence


class TestSubjectExtraction:
    def test_paths_symbols_and_commands_are_subjects(self):
        subjects = subjects_of(
            "The handler in services/auth/handler.py calls verify_token via kubectl"
        )
        assert "services/auth/handler.py" in subjects
        assert "verify_token" in subjects
        assert "kubectl" in subjects

    def test_prose_names_nothing_concrete(self):
        assert subjects_of("the system appears to be working correctly") == []


class TestAntiDecoration:
    def test_a_claim_naming_a_file_needs_that_file_in_the_evidence(self):
        """The failure this whole gate exists to prevent: any citation satisfying any claim."""
        evidence = [_evidence(claim="config loaded", source_id="src/config.py",
                              path="src/config.py")]
        support = check_claim(
            "retries are handled in services/queue/retry.py",
            ClaimKind.OBSERVED, evidence,
        )
        assert not support.supported
        assert "absent from all evidence" in support.reason

    def test_the_same_claim_resolves_when_the_file_is_present(self):
        evidence = [_evidence(claim="retry loop", source_id="services/queue/retry.py",
                              path="services/queue/retry.py")]
        support = check_claim(
            "retries are handled in services/queue/retry.py",
            ClaimKind.OBSERVED, evidence,
        )
        assert support.supported
        assert support.evidence_ids

    def test_a_path_matches_on_its_filename_but_not_on_a_common_directory(self):
        """handler.py should match; the bare directory "auth" should not stand in for the whole path."""
        assert check_claim(
            "see services/auth/handler.py",
            ClaimKind.OBSERVED,
            [_evidence(excerpt="found in handler.py line 20")],
        ).supported
        assert not check_claim(
            "see services/auth/handler.py",
            ClaimKind.OBSERVED,
            [_evidence(excerpt="the auth service is deployed")],
        ).supported


class TestEvidenceEligibility:
    def test_model_knowledge_cannot_support_an_observed_fact(self):
        """Recall is not observation."""
        evidence = [_evidence(claim="kubectl rollout undo reverts a deployment",
                              excerpt="kubectl rollout undo reverts a deployment",
                              source_type=SourceType.MODEL_KNOWLEDGE)]
        support = check_claim("kubectl rollout undo reverts a deployment",
                              ClaimKind.OBSERVED, evidence)
        assert not support.supported
        assert "non-observational" in support.reason

    def test_contradicting_evidence_does_not_support_a_claim(self):
        evidence = [_evidence(claim="pod is healthy", source_id="pod-1",
                              excerpt="pod-1 running", supports=False)]
        assert not check_claim("pod-1 is running", ClaimKind.OBSERVED, evidence).supported

    def test_no_evidence_at_all_is_reported_as_such(self):
        support = check_claim("anything at all", ClaimKind.OBSERVED, [])
        assert not support.supported
        assert "no observational evidence" in support.reason


class TestClaimKinds:
    def test_inferences_are_checked_but_not_counted_as_factual(self):
        answer = Answer(observed=["src/a.py exists"], inferences=["therefore it is slow"])
        support = check_answer(answer, [_evidence(source_id="src/a.py", path="src/a.py")])
        assert support.total == 1, "only observed facts are factual claims"
        assert len(support.claims) == 2

    def test_declared_unverified_claims_are_not_failures(self):
        support = check_answer(Answer(unverified=["might be a DNS issue"]), [])
        assert support.total == 0
        assert support.claims[0].supported


class TestDanglingCitations:
    def test_a_citation_matching_no_evidence_is_dangling(self):
        answer = Answer(citations=["src/ghost.py:10", "src/real.py:4"])
        support = check_answer(answer, [_evidence(source_id="src/real.py",
                                                  path="src/real.py")])
        assert "src/ghost.py:10" in support.dangling_citations
        assert "src/real.py:4" in support.resolved_citations


class TestDemotion:
    def test_unsupported_facts_are_relabelled_not_deleted(self):
        """The claim may be true; what is false is calling it observed."""
        answer = Answer(observed=["src/ghost.py handles retries", "src/real.py exists"])
        support = check_answer(answer, [_evidence(source_id="src/real.py",
                                                  path="src/real.py")])
        answer, demoted, _ = demote_unsupported(answer, support)

        assert demoted == 1
        assert answer.observed_facts == ["src/real.py exists"]
        assert any("ghost.py" in u for u in answer.unverified)
        assert any("no supporting evidence" in u for u in answer.unverified)

    def test_confidence_falls_in_proportion_to_what_was_demoted(self):
        answer = Answer(observed=["a/ghost.py x", "b/ghost.py y"], confidence=0.8)
        support = check_answer(answer, [_evidence(source_id="c/other.py",
                                                  path="c/other.py")])
        answer, demoted, _ = demote_unsupported(answer, support)
        assert demoted == 2
        assert answer.confidence == 0.0, "nothing resolved, so nothing is asserted"

    def test_dangling_citations_are_dropped_and_resolved_ones_kept(self):
        answer = Answer(observed=[], citations=["src/ghost.py", "src/real.py"])
        support = check_answer(answer, [_evidence(source_id="src/real.py",
                                                  path="src/real.py")])
        answer, _, dropped = demote_unsupported(answer, support)
        assert dropped == 1
        assert answer.citations == ["src/real.py"]

    def test_a_clean_answer_is_left_alone(self):
        answer = Answer(observed=["src/real.py exists"], confidence=0.8)
        support = check_answer(answer, [_evidence(source_id="src/real.py",
                                                  path="src/real.py")])
        before = list(answer.observed_facts)
        answer, demoted, dropped = demote_unsupported(answer, support)
        assert (demoted, dropped) == (0, 0)
        assert answer.observed_facts == before
        assert answer.confidence == 0.8


class TestCompletenessGuard:
    def test_needle_coverage_is_reported_alongside_the_claim_rate(self):
        """A gate can always lower the unsupported rate by making answers emptier."""
        from mimir.eval.harness import CaseKind, CaseResult, EvalReport

        report = EvalReport()
        report.results = [
            CaseResult(case_id="a", kind=CaseKind.FEATURE_EXISTENCE, passed=True,
                       needles_expected=4, needles_found=3),
            CaseResult(case_id="b", kind=CaseKind.FEATURE_EXISTENCE, passed=True,
                       needles_expected=2, needles_found=1),
        ]
        assert report.needle_coverage == round(4 / 6, 3)


class TestGroundedCitations:
    """The constructive half of the anti-decoration rule."""

    def test_citations_come_from_the_evidence_that_resolved_the_claim(self):
        from mimir.verify.claims import attach_resolved_citations

        real = _evidence(source_id="src/real.py", path="src/real.py")
        unrelated = _evidence(source_id="src/unrelated.py", path="src/unrelated.py")
        answer = Answer(observed=["src/real.py exists"])
        support = check_answer(answer, [real, unrelated])

        added = attach_resolved_citations(answer, support, [real, unrelated])
        assert added == 1
        assert answer.citations == ["src/real.py"]
        assert "src/unrelated.py" not in answer.citations, (
            "proximity is not a reason to cite"
        )

    def test_nothing_is_attached_for_unsupported_claims(self):
        from mimir.verify.claims import attach_resolved_citations

        real = _evidence(source_id="src/real.py", path="src/real.py")
        answer = Answer(observed=["src/ghost.py handles retries"])
        support = check_answer(answer, [real])
        assert attach_resolved_citations(answer, support, [real]) == 0
        assert answer.citations == []

    def test_existing_citations_are_preserved(self):
        from mimir.verify.claims import attach_resolved_citations

        real = _evidence(source_id="src/real.py", path="src/real.py")
        answer = Answer(observed=["src/real.py exists"], citations=["manual/note.md"])
        support = check_answer(answer, [real])
        attach_resolved_citations(answer, support, [real])
        assert answer.citations[0] == "manual/note.md"
        assert "src/real.py" in answer.citations


class TestScoreVersusProbability:
    """ADR-003 invariant 7, enforced structurally rather than by convention."""

    def test_probability_is_none_until_a_calibration_model_earns_it(self):
        from mimir.models.specialist import FinalAnswer

        answer = FinalAnswer(answer="x", confidence=0.3)
        assert answer.confidence == 0.3, "the raw score is always available"
        assert answer.probability is None, (
            "an uncalibrated score must not appear in a field named probability"
        )

    def test_the_renderer_does_not_call_the_score_a_probability(self):
        import io

        from rich.console import Console

        from mimir.cli.render import render_answer
        from mimir.models.specialist import FinalAnswer

        console = Console(width=100, record=True, file=io.StringIO())
        console.print(render_answer(FinalAnswer(answer="x", confidence=0.13), 0.13))
        text = console.export_text()
        assert "support score" in text
        assert "not a probability" in text


class TestGroundingDemotion:
    """An empty listing names nothing, so nothing may be named."""

    def _answer(self, text, facts=()):
        from mimir.models.specialist import FinalAnswer

        return FinalAnswer(answer=text, observed_facts=list(facts), confidence=0.9)

    def test_a_name_that_appears_in_no_reading_is_demoted(self):
        from mimir.verify.grounding import check, demote_ungrounded

        answer = self._answer(
            "The running pod is messaging-router-6cf8.",
            ["messaging-router-6cf8 is running."],
        )
        result = check(answer.answer, observed="", asked="Which pods are running?")
        answer, moved = demote_ungrounded(answer, result)
        assert moved
        assert answer.observed_facts == []
        assert any("messaging-router" in u for u in answer.unverified)
        assert answer.confidence <= 0.3

    def test_a_name_the_operator_supplied_is_not_an_invention(self):
        """Repeating back a name they gave us is not hallucination, and flagging it teaches the reader to ignore the warning."""
        from mimir.verify.grounding import check, demote_ungrounded

        answer = self._answer("messaging-whatsapp has no pods.")
        result = check(
            answer.answer,
            observed="",
            asked="find messaging-whatsapp pods in messaging-squad",
        )
        answer, moved = demote_ungrounded(answer, result)
        assert moved == 0
        assert answer.confidence == 0.9

    def test_a_name_that_was_actually_read_survives(self):
        from mimir.verify.grounding import check, demote_ungrounded

        answer = self._answer(
            "messaging-router-6cf8 is running.", ["messaging-router-6cf8 is running."]
        )
        result = check(
            answer.answer, observed="NAME READY\nmessaging-router-6cf8 1/1 Running"
        )
        _, moved = demote_ungrounded(answer, result)
        assert moved == 0

    def test_the_warning_reaches_the_prose_not_only_the_bullets(self):
        from mimir.verify.grounding import check, demote_ungrounded

        answer = self._answer("The running pod is messaging-router-6cf8.")
        answer, _ = demote_ungrounded(
            answer, check(answer.answer, observed="", asked="")
        )
        assert "appear in nothing that was read" in answer.answer
