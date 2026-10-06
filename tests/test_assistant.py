"""The assistant behind /v1: question kind, conversation, and a clean answer."""

import pathlib

import pytest
import yaml

from mimir.api.assistant import Part, conversation, respond, rule_intent
from mimir.llm.base import LLMMessage

CORPUS = pathlib.Path(__file__).parent.parent / "src/mimir/eval/corpus"
REGIONS = ("ch1", "fr5", "dc2")


@pytest.mark.parametrize("name", ["intents.yaml", "intents_heldout.yaml"])
def test_a_rule_that_fires_is_never_wrong(name):
    for c in yaml.safe_load((CORPUS / name).read_text())["cases"]:
        got = rule_intent(c["text"], c["history"], regions=REGIONS)
        assert got in (None, c["intent"]), (c["text"], got)


def test_an_identifier_sets_the_surface_not_the_kind_of_question():
    assert rule_intent("what's the point of saveNextSignupStep", False) is None
    assert rule_intent("where do we receive FINISH_ONBOARDING", False) == "locate"


def test_a_configured_region_means_the_clusters():
    assert rule_intent("grab the events for messaging-settings-data on fr5", False, regions=REGIONS) == "live"


def test_a_follow_up_is_a_revision_only_when_there_is_something_to_revise():
    assert rule_intent("are you 100% sure?", True) == "revise"
    assert rule_intent("are you 100% sure?", False) != "revise"


def test_the_latest_message_is_kept_whole_and_history_is_budgeted():
    long_thread = "x" * 9000
    msgs = [LLMMessage.user("old question"), LLMMessage.assistant("*Worked for 3s*\n\nold answer"),
            LLMMessage.user(long_thread)]
    q, history = conversation(msgs, budget=12000)
    assert q == long_thread
    assert history == [("user", "old question"), ("assistant", "old answer")]
    q, history = conversation(msgs, budget=9005)
    assert history == []


@pytest.mark.asyncio
async def test_a_revision_uses_no_tools_and_the_answer_is_clean(settings):
    seen = {}

    class Router:
        async def chat(self, messages, **kw):
            seen["msgs"] = messages
            return type("R", (), {"content": "We do not support it yet; Telnyx should run it."})()

    class Reg:
        def get(self, name):
            raise AssertionError(f"no tool should run for a revision, got {name}")

    class Runner:
        router = Router(); registry = Reg()
        def tool_context(self, sid): return None
    Runner.settings = settings
    msgs = [LLMMessage.user("what should I reply to Juan?"),
            LLMMessage.assistant("We do not currently implement Meta's currency migration flow, so..."),
            LLMMessage.user("ok make the reply as small as possible")]
    parts = [p async for p in respond(msgs, Runner(), settings)]
    progress = [p.text for p in parts if p.kind == "progress"]
    (answer,) = [p.text for p in parts if p.kind == "answer"]
    assert progress[0].startswith("Understood as: change, shorten")
    assert answer.startswith("*Worked for ") and answer.endswith("Telnyx should run it.")
    assert not any(line.startswith((">", "[surface")) for line in answer.splitlines())
    assert any("currency migration flow" in getattr(m, "content", "") for m in seen["msgs"])
