"""The verify node as two decisions, no generative call."""

import pytest

from mimir.decide.base import Verdict
from mimir.graph.nodes import verify
from mimir.models.specialist import SpecialistName, SpecialistReport
from mimir.models.state import InvestigationState


class FakeDecider:
    available = True
    name = "fake"

    def __init__(self, sufficient="yes", conflict="consistent", probability=0.9):
        self.answers = {"sufficient": sufficient, "conflict": conflict}
        self.probability = probability

    def decide(self, context, fields):
        out = {}
        for f in fields:
            c = self.answers[f.name]
            others = [o for o in f.options if o != c]
            dist = {c: self.probability, **{o: (1 - self.probability) / len(others) for o in others}}
            out[f.name] = Verdict(field=f.name, choice=c, probability=self.probability, distribution=dist)
        return out


class Deps:
    def __init__(self, decider):
        self.decider = decider
        self.tool_context = None
        self.asked_model = False

    def specialist(self, name):
        self.asked_model = True
        raise AssertionError("generative verifier must not run when a decider answers")


def _state():
    session = InvestigationState(user_request="why is api restarting?")
    reports = [
        SpecialistReport(specialist=SpecialistName.LOG_ANALYST, conclusion="OOM kills", confidence=0.8),
        SpecialistReport(specialist=SpecialistName.KUBERNETES_INVESTIGATOR, conclusion="probe failures", confidence=0.7),
    ]
    return {"session": session, "reports": reports, "route": "verify"}, reports


@pytest.mark.asyncio
async def test_sufficient_and_consistent_changes_nothing_and_asks_no_model():
    state, reports = _state()
    deps = Deps(FakeDecider())
    out = await verify(state, deps)
    assert out["route"] == "safety_review" and out["reports"] == []
    assert [r.confidence for r in reports] == [0.8, 0.7]
    assert state["session"].metadata["verify"] == {"mode": "decided", "sufficient": "yes", "conflict": "consistent"}
    assert not deps.asked_model


@pytest.mark.asyncio
async def test_insufficient_lowers_confidence_and_records_what_is_missing():
    state, reports = _state()
    await verify(state, Deps(FakeDecider(sufficient="no")))
    assert [r.confidence for r in reports] == [0.6, 0.5]
    assert any("does not settle" in r for r in state["session"].risks)


@pytest.mark.asyncio
async def test_a_contradiction_is_recorded_never_smoothed_away():
    state, reports = _state()
    await verify(state, Deps(FakeDecider(conflict="contradictory")))
    assert all(r.contradictions for r in reports)
    assert [r.confidence for r in reports] == [0.6, 0.5]


@pytest.mark.asyncio
async def test_both_decisions_are_logged():
    state, _ = _state()
    await verify(state, Deps(FakeDecider()))
    fields = [d["field"] for d in state["session"].metadata["decisions"]]
    assert sorted(fields) == ["conflict", "sufficient"]


@pytest.mark.asyncio
async def test_a_verdict_below_the_floor_is_logged_and_not_applied():
    state, reports = _state()
    await verify(state, Deps(FakeDecider(sufficient="no", probability=0.55)))  # margin 0.1
    assert [r.confidence for r in reports] == [0.8, 0.7]
    assert all(d["acted"] is False for d in state["session"].metadata["decisions"])


@pytest.mark.asyncio
async def test_no_reports_means_nothing_to_verify():
    session = InvestigationState(user_request="q")
    out = await verify({"session": session, "reports": [], "route": "verify"}, Deps(FakeDecider()))
    assert out == {"route": "safety_review"}
