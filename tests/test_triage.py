"""Triage tests.

The asymmetry under test: failing to divert a greeting wastes a minute; wrongly
diverting a real question is a refusal to work. Every rule is biased toward
investigating, and most of these tests check that bias holds.
"""

from __future__ import annotations

import pytest

from mimir.graph.triage import Triage, triage


class TestConversationalInput:
    @pytest.mark.parametrize(
        "text",
        ["hello", "Hello!", "hi", "hey", "yo", "good morning", "Good Evening.", "howdy"],
    )
    def test_greetings_are_diverted(self, text):
        assert triage(text).kind is Triage.GREETING

    @pytest.mark.parametrize("text", ["thanks", "thank you", "cheers", "ok", "nice"])
    def test_acknowledgements_are_diverted(self, text):
        assert triage(text).kind is Triage.ACKNOWLEDGEMENT

    @pytest.mark.parametrize("text", ["bye", "goodbye", "see you", "good night"])
    def test_farewells_are_diverted(self, text):
        assert triage(text).kind is Triage.FAREWELL

    @pytest.mark.parametrize(
        "text", ["how are you", "who are you", "what can you do", "test", "ping"]
    )
    def test_capability_questions_are_answered_from_static_text(self, text):
        assert triage(text).kind is Triage.CAPABILITY

    @pytest.mark.parametrize("text", ["", "   ", "\n", None])
    def test_empty_input_is_diverted(self, text):
        assert triage(text).kind is Triage.EMPTY

    def test_every_diverted_kind_carries_a_reply(self):
        for text in ("hello", "thanks", "bye", "what can you do", ""):
            result = triage(text)
            assert result.cheap and result.reply.strip()


class TestBiasTowardInvestigating:
    """A false positive here is a refusal to work. These are the important ones."""

    @pytest.mark.parametrize(
        "text",
        [
            "hello, why is the api pod restarting?",
            "hi can you check services/auth/handler.py",
            "thanks, now show me the logs",
            "hey what does kubectl get pods return",
            "good morning, the deployment failed overnight",
        ],
    )
    def test_a_pleasantry_before_a_real_question_is_still_a_question(self, text):
        assert not triage(text).cheap

    @pytest.mark.parametrize(
        "text",
        ["logs", "restart", "the migration", "schema", "timeout", "crash", "pod"],
    )
    def test_short_operational_words_are_investigated(self, text):
        """Brevity is not conversation. These are terse requests."""
        assert not triage(text).cheap

    def test_a_long_message_is_never_diverted_however_it_starts(self):
        text = "hi " + "and then something happened with the system " * 3
        assert len(text) > 64
        assert not triage(text).cheap

    def test_anything_naming_a_file_is_investigated(self):
        assert not triage("ok src/main.py").cheap


class TestCorpusIsUntouched:
    def test_no_corpus_case_is_ever_diverted(self):
        """The property that lets triage ship without confounding a running
        experiment: it can only fire on inputs that are not questions, and
        every corpus case is a question.
        """
        from mimir.eval.harness import EvalHarness

        prompts = [c.prompt for c in EvalHarness.load_corpus() if getattr(c, "prompt", "")]
        assert prompts, "corpus should not be empty"
        diverted = [p for p in prompts if triage(p).cheap]
        assert diverted == [], f"triage would change these corpus cases: {diverted}"


class TestContrastivePairs:
    """Each pair differs in one fact and the correct answers differ with it, so
    a model keying on the shape of the question answers both the same way and
    gets exactly one right. That reads as fifty percent accuracy and zero
    percent consistency, and only the second number says which it was."""

    def _corpus(self):
        from mimir.eval.harness import EvalHarness

        return EvalHarness.load_corpus()

    def test_every_pair_has_exactly_two_members(self):
        from collections import Counter

        counts = Counter(c.pair for c in self._corpus() if c.pair)
        assert counts, "the corpus should contain contrastive pairs"
        assert not {p: n for p, n in counts.items() if n != 2}

    def test_the_twins_differ_in_what_they_expect(self):
        """A pair whose members expect the same thing is two copies of one
        case, and tests nothing about reading the evidence."""
        pairs: dict[str, list] = {}
        for case in self._corpus():
            if case.pair:
                pairs.setdefault(case.pair, []).append(case)
        for name, (first, second) in pairs.items():
            assert (
                first.expect_contains != second.expect_contains
                or first.expect_absent != second.expect_absent
                or first.max_confidence != second.max_confidence
            ), f"{name}: both twins expect the same thing"

    def test_every_case_id_is_distinct(self):
        ids = [c.id for c in self._corpus()]
        assert len(ids) == len(set(ids))

    def test_consistency_is_none_rather_than_zero_without_pairs(self):
        """No pairs answered consistently and no pairs to answer are different
        results, and a caller that cannot tell them apart will report the
        second as the first."""
        from mimir.eval.harness import CaseKind, CaseResult, EvalReport

        report = EvalReport()
        report.results = [CaseResult(case_id="a", kind=CaseKind.TARGETING, passed=True)]
        assert report.pair_consistency is None

    def test_a_model_answering_both_twins_alike_scores_zero(self):
        from mimir.eval.harness import CaseKind, CaseResult, EvalReport

        report = EvalReport()
        report.results = [
            CaseResult(case_id="a", kind=CaseKind.TARGETING, passed=True, pair="p"),
            CaseResult(case_id="b", kind=CaseKind.TARGETING, passed=False, pair="p"),
        ]
        assert report.pair_consistency == 0.0

    def test_both_right_scores_one(self):
        from mimir.eval.harness import CaseKind, CaseResult, EvalReport

        report = EvalReport()
        report.results = [
            CaseResult(case_id="a", kind=CaseKind.TARGETING, passed=True, pair="p"),
            CaseResult(case_id="b", kind=CaseKind.TARGETING, passed=True, pair="p"),
        ]
        assert report.pair_consistency == 1.0

    def test_an_incomplete_pair_is_not_counted(self):
        """Half a pair says nothing about consistency."""
        from mimir.eval.harness import CaseKind, CaseResult, EvalReport

        report = EvalReport()
        report.results = [
            CaseResult(case_id="a", kind=CaseKind.TARGETING, passed=True, pair="p")
        ]
        assert report.pair_consistency is None
