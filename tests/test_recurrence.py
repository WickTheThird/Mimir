"""The recurrent edge: conclude, continue or ask, by policy first."""

import pytest

from mimir.decide.base import Verdict
from mimir.graph.nodes import assess, replan
from mimir.models.evidence import Evidence, SourceType
from mimir.models.specialist import CoordinatorPlan, PlannedStep, SpecialistName, SpecialistReport
from mimir.models.state import InvestigationState


class FakeDecider:
    available = True
    name = "fake"

    def __init__(self, choice, probability=0.9):
        self.choice, self.probability, self.calls = choice, probability, 0

    def decide(self, context, fields):
        self.calls += 1
        others = [o for o in fields[0].options if o != self.choice]
        dist = {self.choice: self.probability, **{o: (1 - self.probability) / len(others) for o in others}}
        return {fields[0].name: Verdict(field="next", choice=self.choice,
                                        probability=self.probability, distribution=dist)}


class Settings:
    class graph:
        max_specialist_rounds = 3


class Ctx:
    settings = Settings()


class Deps:
    def __init__(self, decider=None, plan=None, fail=False):
        self.decider = decider
        self.tool_context = Ctx()
        self._plan, self._fail = plan, fail

    def specialist(self, name):
        deps = self

        class Coord:
            async def structured_report(self, *a, **k):
                if deps._fail:
                    from mimir.llm.base import ModelError
                    raise ModelError("down")
                return deps._plan or CoordinatorPlan()

        return Coord()


def _session(reports=1, evidence=2):
    s = InvestigationState(user_request="why is api restarting?")
    for i in range(reports):
        s.reports.append(SpecialistReport(specialist=SpecialistName.LOG_ANALYST,
                                          objective=f"objective {i}", conclusion="restarts rose"))
    for i in range(evidence):
        s.evidence.append(Evidence(claim=f"e{i}", source_type=SourceType.COMMAND_OUTPUT,
                                   source_id=f"c{i}", excerpt=f"line {i}"))
    return s


@pytest.mark.asyncio
async def test_no_decision_model_means_one_pass_as_before():
    session = _session()
    out = await assess({"session": session, "round": 1, "route": "verify"}, Deps(None))
    assert out["route"] == "verify"
    assert session.metadata["assess"][0]["reason"] == "no decision model; single pass"


@pytest.mark.asyncio
async def test_the_round_cap_stops_the_loop_before_any_model_is_asked():
    decider = FakeDecider("continue")
    session = _session()
    out = await assess({"session": session, "round": 3, "route": "safety_review"}, Deps(decider))
    assert out["route"] == "safety_review"
    assert decider.calls == 0
    assert "round cap" in session.metadata["assess"][0]["reason"]


@pytest.mark.asyncio
async def test_no_new_evidence_stops_the_loop_before_any_model_is_asked():
    """A round that added nothing must not be repeated: the next round would
    see the same evidence and choose the same thing."""
    decider = FakeDecider("continue")
    session = _session(evidence=2)
    session.metadata["evidence_seen_by_round"] = {"1": 2}
    out = await assess({"session": session, "round": 2, "route": "safety_review"}, Deps(decider))
    assert out["route"] == "safety_review"
    assert decider.calls == 0
    assert "no new evidence" in session.metadata["assess"][0]["reason"]


@pytest.mark.asyncio
async def test_continue_routes_to_replan_and_logs_the_decision():
    session = _session()
    out = await assess({"session": session, "round": 1, "route": "verify"}, Deps(FakeDecider("continue")))
    assert out["route"] == "replan"
    (entry,) = session.metadata["decisions"]
    assert entry["field"] == "next" and entry["choice"] == "continue" and entry["acted"]


@pytest.mark.asyncio
async def test_ask_routes_to_replan_in_asking_mode():
    session = _session()
    out = await assess({"session": session, "round": 1, "route": "verify"}, Deps(FakeDecider("ask")))
    assert out["route"] == "replan_ask"


@pytest.mark.asyncio
async def test_a_calibrated_verdict_below_the_floor_concludes():
    """The model's own uncertainty is the reason to have a calibrated one."""
    session = _session()
    out = await assess({"session": session, "round": 1, "route": "verify"},
                       Deps(FakeDecider("continue", probability=0.5)))
    assert out["route"] == "verify"
    assert session.metadata["decisions"][0]["acted"] is False


@pytest.mark.asyncio
async def test_replan_drops_steps_that_repeat_a_completed_objective():
    """A coordinator that proposes the same check twice cannot spin the loop."""
    session = _session()
    plan = CoordinatorPlan(steps=[
        PlannedStep(specialist=SpecialistName.LOG_ANALYST, objective="objective 0"),
        PlannedStep(specialist=SpecialistName.KUBERNETES_INVESTIGATOR, objective="check restarts"),
    ])
    out = await replan({"session": session, "round": 1, "route": "replan"}, Deps(plan=plan))
    assert [st.objective for st in out["pending_steps"]] == ["check restarts"]
    assert out["route"] == "select_skills"


@pytest.mark.asyncio
async def test_replan_with_only_a_question_asks_the_operator():
    session = _session()
    plan = CoordinatorPlan(missing_context=["which cluster?"])
    out = await replan({"session": session, "round": 1, "route": "replan"}, Deps(plan=plan))
    assert out["route"] == "ask_user"
    assert "which cluster?" in session.pending_questions


@pytest.mark.asyncio
async def test_replan_with_nothing_new_concludes():
    session = _session()
    out = await replan({"session": session, "round": 1, "route": "replan"}, Deps(plan=CoordinatorPlan()))
    assert out["route"] == "safety_review"
    assert out["pending_steps"] == []


@pytest.mark.asyncio
async def test_a_failed_replan_concludes_on_what_was_gathered():
    session = _session()
    out = await replan({"session": session, "round": 1, "route": "replan"}, Deps(fail=True))
    assert out["route"] == "safety_review"


@pytest.mark.asyncio
async def test_progress_is_measured_against_the_previous_round():
    session = _session(evidence=3)
    await assess({"session": session, "round": 1, "route": "verify"}, Deps(None))
    assert session.metadata["assess"][0]["new_evidence"] == 3
    session.evidence.append(Evidence(claim="e9", source_type=SourceType.COMMAND_OUTPUT,
                                     source_id="c9", excerpt="x"))
    await assess({"session": session, "round": 2, "route": "verify"}, Deps(None))
    assert session.metadata["assess"][1]["new_evidence"] == 1
