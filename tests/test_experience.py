"""Steps 7 and 8: predictions that move likelihood, and experience kept."""

import pytest

from mimir.decide.base import Verdict
from mimir.graph.nodes import assess
from mimir.knowledge.experience import ExperienceStore, corpus_draft, repo_lesson, write_corpus_draft
from mimir.models.evidence import Evidence, SourceType
from mimir.models.specialist import FinalAnswer, Hypothesis, HypothesisStatus, SpecialistName, SpecialistReport
from mimir.models.state import InvestigationState


class FakeDecider:
    available = True
    name = "fake"

    def __init__(self, answers):
        self.answers = answers  # field -> choice

    def decide(self, context, fields):
        out = {}
        for f in fields:
            c = self.answers.get(f.name, f.options[0])
            others = [o for o in f.options if o != c]
            dist = {c: 0.9, **{o: 0.1 / len(others) for o in others}}
            out[f.name] = Verdict(field=f.name, choice=c, probability=0.9, distribution=dist)
        return out


class Settings:
    class graph:
        max_specialist_rounds = 3


class Ctx:
    settings = Settings()


class Deps:
    def __init__(self, decider):
        self.decider = decider
        self.tool_context = Ctx()


def _session():
    s = InvestigationState(user_request="why is api restarting?")
    s.reports.append(SpecialistReport(specialist=SpecialistName.LOG_ANALYST, objective="o", conclusion="OOM"))
    for i in range(3):
        s.evidence.append(Evidence(claim=f"e{i}", source_type=SourceType.COMMAND_OUTPUT, source_id=f"kubectl:{i}", excerpt="OOMKilled"))
    s.hypotheses.append(Hypothesis(statement="the pod is OOM killed", likelihood=0.5,
                                   next_check="last terminated reason is OOMKilled"))
    return s


@pytest.mark.asyncio
async def test_a_confirmed_prediction_raises_likelihood_and_links_evidence():
    s = _session()
    await assess({"session": s, "round": 1, "route": "verify"}, Deps(FakeDecider({"next": "conclude", "holds": "confirmed"})))
    h = s.hypotheses[0]
    assert h.likelihood == 0.7 and h.supporting_evidence_ids
    assert s.metadata["assess"][0]["predictions"][0]["holds"] == "confirmed"


@pytest.mark.asyncio
async def test_a_contradicted_prediction_lowers_likelihood_and_rejects_when_low():
    s = _session(); s.hypotheses[0].likelihood = 0.3
    await assess({"session": s, "round": 1, "route": "verify"}, Deps(FakeDecider({"next": "conclude", "holds": "contradicted"})))
    h = s.hypotheses[0]
    assert h.status == HypothesisStatus.REJECTED and h.contradicting_evidence_ids
    assert "other way" in h.rejected_reason


@pytest.mark.asyncio
async def test_no_new_evidence_means_no_prediction_is_scored():
    s = _session(); s.metadata["evidence_seen_by_round"] = {"1": 3}
    await assess({"session": s, "round": 2, "route": "verify"}, Deps(FakeDecider({"holds": "confirmed"})))
    assert s.hypotheses[0].likelihood == 0.5
    assert s.metadata["assess"][0]["predictions"] == []


@pytest.mark.asyncio
async def test_each_round_leaves_a_record_that_is_not_the_transcript():
    s = _session()
    await assess({"session": s, "round": 1, "route": "verify"}, Deps(None))
    (rec,) = s.metadata["rounds"]
    assert rec["round"] == 1 and len(rec["observations"]) == 3
    assert rec["claims"][0]["by"] == "log_analyst" and rec["hypotheses"][0]["likelihood"] == 0.5


def test_routing_statistics_are_recorded_per_session(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    s = _session(); s.task_type = None; s.final_confidence = 0.8
    s.metadata["decisions"] = [{"field": "retrieval", "choice": "empty", "probability": 0.7, "acted": True}]
    out = store.record(s)
    assert out == {"specialists": 1, "decisions": 1, "rounds": 1}
    (row,) = store.routing_for("")
    assert row["tools_cited"] == '["kubectl"]' and row["decisions_acted"] == 1


def test_preferred_specialists_come_from_confident_answers(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    for conf in (0.9, 0.3):
        s = _session(); s.final_confidence = conf; s.session_id = f"s{conf}"
        store.record(s)
    assert store.preferred_specialists("") == ["log_analyst"]


def test_a_confident_evidenced_session_drafts_a_corpus_case(tmp_path):
    s = _session(); s.final_confidence = 0.8
    s.final_answer = FinalAnswer(answer="messaging-router-6cf8 was OOMKilled at 512Mi.", confidence=0.8)
    draft = corpus_draft(s)
    assert draft and "messaging-router-6cf8" in draft["expect_contains"]
    path = write_corpus_draft(tmp_path, s)
    assert path and path.read_text().startswith("cases:")


def test_a_low_confidence_session_drafts_nothing():
    s = _session(); s.final_confidence = 0.4
    s.final_answer = FinalAnswer(answer="messaging-router-6cf8", confidence=0.4)
    assert corpus_draft(s) is None


def test_a_repo_lesson_is_a_memory_note_under_the_repo():
    note = repo_lesson("billing", files_changed=["pkg/client.py"], test_command="pytest -q",
                       tools_used=["repository_map"], stopped="done")
    assert note["category"] == "repos/billing" and "pkg/client.py" in note["body"]
