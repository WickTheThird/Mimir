"""Term association.

Operators do not name things the way the estate does. Every case here is a
phrasing from a real session.
"""

from __future__ import annotations

import pytest

from mimir.knowledge.glossary import Glossary, terms_of


@pytest.fixture
def glossary(tmp_path):
    return Glossary(tmp_path / "g.db")


def _seed(glossary, names, kind="project"):
    for name in names:
        for term in name.split("-"):
            if len(term) >= 3:
                glossary.record(term, name, kind, source="seeded")


class TestTermExtraction:
    def test_operational_filler_is_not_a_term(self):
        terms = terms_of("show me the last 10 logs from any pods in the dev cluster")
        assert "logs" not in terms
        assert "cluster" not in terms
        assert "dev" not in terms

    def test_a_three_letter_name_is(self):
        """"ch1" is how this operator names a pair of clusters. A rule that
        cannot represent it misses the term they actually use."""
        assert "ch1" in terms_of("a dev cluster with ch1 inside of it")


class TestNearMisses:
    def test_a_typo_resolves(self, glossary):
        """The case this exists for. "whatapp" has never been seen and
        messaging-whatsapp has, and the distance is one character."""
        _seed(glossary, ["messaging-whatsapp", "messaging-router"])
        near = [
            a for a in glossary.lookup("logs for messaging-whatapp please")
            if a.source == "near match"
        ]
        assert [(a.term, a.name) for a in near] == [("whatapp", "messaging-whatsapp")]

    def test_a_coincidence_does_not(self, glossary):
        """A similarity ratio alone cannot separate these: whatapp/whatsapp
        scores 0.93, retry/registry 0.77, backoff/backoffice 0.82. Length is
        what tells a typo from a different word."""
        _seed(glossary, ["messaging-campaign-registry", "messaging-backoffice"])
        assert glossary.lookup("does the retry backoff still fail open") == []

    def test_a_word_meaning_many_things_means_nothing(self, glossary):
        """"messaging" is a segment of six projects here. Offering all of them
        is the list the model would have got anyway."""
        _seed(glossary, [f"messaging-{n}" for n in ("a1", "b2", "c3", "d4", "e5")])
        assert [a.term for a in glossary.lookup("the messaging thing")] == []

    def test_a_whole_segment_resolves_exactly(self, glossary):
        _seed(glossary, ["kannel-sms-gateway"])
        assert [a.name for a in glossary.lookup("kannel is down")] == ["kannel-sms-gateway"]


class TestLearning:
    def test_it_learns_from_what_was_observed(self, glossary):
        assert glossary.learn("check whatapp", ["messaging-whatsapp", "messaging-router"])
        assert [a.name for a in glossary.lookup("whatapp")] == ["messaging-whatsapp"]

    def test_a_turn_that_resolved_nothing_teaches_nothing(self, glossary):
        assert glossary.learn("check whatapp", []) == 0
        assert glossary.lookup("whatapp") == []

    def test_a_term_is_never_associated_with_itself(self, glossary):
        glossary.record("mimir", "mimir", "project")
        assert glossary.all() == []


class TestHint:
    def test_it_is_offered_as_a_lead_not_a_fact(self, glossary):
        """The estate changes. A hint stated as a fact is a stale fact the
        model will defend."""
        _seed(glossary, ["messaging-whatsapp"])
        hint = glossary.hint("logs for whatapp")
        assert "has meant" in hint
        assert "not as a fact" in hint

    def test_nothing_known_means_no_prompt_at_all(self, glossary):
        assert glossary.hint("something entirely unrelated") == ""


class TestLoopIntegration:
    def test_the_hint_rides_on_the_user_turn_after_the_instruction(self, tmp_path):
        """Not the system prompt: it is about these words, and a system prompt
        that grows a section per turn is paid on every step of every later
        turn. And after the instruction, not before: leading with 162
        characters of preamble made qwen3-coder stop emitting tool calls
        entirely, at temperature zero, reproducibly."""
        import asyncio

        from tests.test_agent import _agent, _drain

        glossary = Glossary(tmp_path / "g.db")
        _seed(glossary, ["messaging-whatsapp"])
        agent, _, _ = _agent([("done", [])])
        agent.glossary = glossary
        asyncio.run(_drain(agent, "look at whatapp"))

        system = agent.messages[0]
        user = next(m for m in agent.messages if m.role.value == "user")
        assert "has meant" not in system.content
        assert "messaging-whatsapp" in user.content
        assert user.content.startswith("look at whatapp"), (
            "the operator's words come first or the model stops calling tools"
        )

    def test_a_broken_glossary_never_fails_a_turn(self, tmp_path):
        import asyncio

        from tests.test_agent import _agent, _drain

        class Broken:
            def hint(self, text):
                raise RuntimeError("no")

            def learn(self, *a, **k):
                raise RuntimeError("no")

        agent, _, _ = _agent([("done", [])])
        agent.glossary = Broken()
        events = asyncio.run(_drain(agent, "anything"))
        assert events[-1].type.value == "done"
