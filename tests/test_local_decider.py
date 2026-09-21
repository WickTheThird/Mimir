"""A decider that cannot be calibrated must say so, not report a zero."""

import json

import pytest

from mimir.decide.base import Choice, Verdict
from mimir.decide.local import LocalDecider, _prompt, _schema


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []

    @property
    def has_tool_calls(self):
        return False


class FakeRouter:
    def __init__(self, content='{"outcome": "failed"}'):
        self.content = content
        self.calls = []

    async def chat(self, messages, *, task_class=None, options=None,
                   session_id="", purpose=""):
        self.calls.append({"messages": messages, "options": options})
        return FakeResponse(self.content)


OUTCOME = Choice(
    name="outcome",
    options=("observed", "empty", "failed"),
    description="What happened when the system tried to look",
)


def test_the_schema_closes_the_option_set():
    """The guarantee the call sites need: nothing can come back that was not
    offered. Constrained decoding measured 100% adherence at every size."""
    schema = _schema([OUTCOME])
    assert schema["properties"]["outcome"]["enum"] == ["observed", "empty", "failed"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["outcome"]


def test_every_field_is_decided_in_one_call():
    """Splitting them multiplies the prompt cost by the number of questions
    and lets the answers drift, since each call sees the context fresh."""
    fields = [OUTCOME, Choice(name="sufficient", options=("yes", "no"))]
    schema = _schema(fields)
    assert set(schema["required"]) == {"outcome", "sufficient"}


@pytest.mark.asyncio
async def test_a_verdict_is_returned_uncalibrated():
    router = FakeRouter()
    verdicts = await LocalDecider(router).decide_async("context", [OUTCOME])
    assert verdicts["outcome"].choice == "failed"
    assert verdicts["outcome"].calibrated is False
    assert verdicts["outcome"].confident is False


@pytest.mark.asyncio
async def test_a_threshold_gate_must_not_read_the_uncalibrated_zero():
    """A caller comparing 0.0 against a 0.7 floor would discard every correct
    decision while looking like it was being careful."""
    router = FakeRouter()
    verdict = (await LocalDecider(router).decide_async("c", [OUTCOME]))["outcome"]
    assert verdict.probability == 0.0
    assert not verdict.calibrated
    assert "uncalibrated" in verdict.render()
    assert "p=" not in verdict.render()


@pytest.mark.asyncio
async def test_an_off_menu_answer_is_dropped_not_defaulted():
    """The schema should make this impossible. If it happens the constraint
    was not applied, and a silent default would look like a real decision."""
    verdicts = await LocalDecider(FakeRouter('{"outcome": "maybe"}')).decide_async(
        "c", [OUTCOME]
    )
    assert verdicts == {}


@pytest.mark.asyncio
async def test_unparseable_output_yields_no_verdict():
    assert await LocalDecider(FakeRouter("not json")).decide_async("c", [OUTCOME]) == {}


@pytest.mark.asyncio
async def test_a_model_failure_is_silence_not_a_negative_verdict():
    """Every caller falls back to what it did before. An absent decider must
    never read as a decision against."""
    from mimir.llm.base import ModelError

    class Broken:
        async def chat(self, *a, **k):
            raise ModelError("down")

    assert await LocalDecider(Broken()).decide_async("c", [OUTCOME]) == {}


@pytest.mark.asyncio
async def test_a_clipped_context_is_flagged():
    from mimir.decide.base import MAX_CONTEXT_CHARS

    verdicts = await LocalDecider(FakeRouter()).decide_async(
        "x" * (MAX_CONTEXT_CHARS + 100), [OUTCOME]
    )
    assert verdicts["outcome"].truncated is True


@pytest.mark.asyncio
async def test_the_request_is_constrained_and_deterministic():
    router = FakeRouter()
    await LocalDecider(router).decide_async("c", [OUTCOME])
    options = router.calls[0]["options"]
    assert options.temperature == 0.0
    assert options.response_format["type"] == "json_schema"


def test_the_prompt_names_every_allowed_option():
    text = _prompt("material", [OUTCOME])
    for option in OUTCOME.options:
        assert option in text


@pytest.mark.asyncio
async def test_no_fields_is_not_a_model_call():
    router = FakeRouter()
    assert await LocalDecider(router).decide_async("c", []) == {}
    assert router.calls == []
