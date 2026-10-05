"""The Hermes loop, MIMIR's rules: drafts from tasks, promoted only by a person."""

import pytest

from mimir.knowledge.skill_drafts import (
    draft_from_coding, draft_from_session, list_drafts, promote, write_draft,
)
from mimir.models.evidence import Evidence, SourceType
from mimir.models.specialist import FinalAnswer, SpecialistName, SpecialistReport
from mimir.models.state import InvestigationState


def _session(confidence=0.8, tools=True):
    s = InvestigationState(user_request="why is messaging-router restarting in messaging-squad?")
    s.final_confidence = confidence
    s.final_answer = FinalAnswer(answer="messaging-router-6cf8 was OOMKilled at 512Mi.", confidence=confidence)
    s.reports.append(SpecialistReport(specialist=SpecialistName.KUBERNETES_INVESTIGATOR, conclusion="OOM kills on the router"))
    if tools:
        s.evidence.append(Evidence(claim="last terminated OOMKilled", source_type=SourceType.COMMAND_OUTPUT,
                                   source_id="summarise_pod_health:1", excerpt="OOMKilled"))
    return s


def test_a_confident_tool_using_session_drafts_a_skill_with_a_test():
    d = draft_from_session(_session())
    assert d and d["name"].startswith("learned-")
    fm = d["frontmatter"]
    assert fm["specialist"] == "kubernetes_investigator"
    assert "summarise_pod_health" in fm["allowed_tools"]
    assert fm["tests"][0]["assertions"][0].startswith("contains: messaging-router")


def test_low_confidence_or_no_tools_teaches_nothing():
    assert draft_from_session(_session(confidence=0.4)) is None
    assert draft_from_session(_session(tools=False)) is None


def test_a_written_draft_parses_with_the_real_skill_loader(tmp_path):
    from mimir.skills.loader import parse_skill_file

    path = write_draft(tmp_path, draft_from_session(_session()))
    skill = parse_skill_file(path)
    assert skill.name.startswith("learned-") and skill.tests


def test_drafts_live_outside_the_live_roots_until_promoted(tmp_path):
    path = write_draft(tmp_path, draft_from_session(_session()))
    assert "skills-drafts" in str(path)
    live = tmp_path / "knowledge" / "skills"; live.mkdir(parents=True)
    assert list_drafts(tmp_path) == [path]
    dst = promote(tmp_path, path.parent.name, live)
    assert dst.is_file() and list_drafts(tmp_path) == []


def test_promote_refuses_to_overwrite_a_live_skill(tmp_path):
    path = write_draft(tmp_path, draft_from_session(_session()))
    live = tmp_path / "knowledge" / "skills" / path.parent.name; live.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        promote(tmp_path, path.parent.name, live.parent)


def test_a_coding_task_drafts_a_skill_keyed_on_its_files():
    d = draft_from_coding("billing", "rename load_config to read_config", files_changed=["pkg/config.py", "pkg/app.py"],
                          tools_used=["repository_map"], test_command="pytest -q")
    assert d and d["frontmatter"]["tests"][0]["assertions"] == ["contains: config.py", "contains: app.py"]
    assert draft_from_coding("billing", "x", files_changed=[], tools_used=[], test_command="") is None
