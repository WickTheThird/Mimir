"""Gathering and concluding are different phases with different budgets.

On qwen3-coder:30b, 16 of 19 corpus failures were a specialist that spent its
iteration budget calling tools and had nothing left to write an answer with.
"""

import pytest

from mimir.council.specialists import SpecialistBudget
from mimir.llm.base import LLMMessage


class FakeResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self):
        return bool(self.tool_calls)

    def as_message(self):
        return LLMMessage.assistant(self.content or "")


class FakeCall:
    def __init__(self, name="read_pods", cid="c1"):
        self.name = name
        self.id = cid
        self.arguments = {}


class RecordingRouter:
    """Records what tools each turn was offered."""

    def __init__(self, script):
        self.script = list(script)
        self.turns = []

    async def chat(self, messages, *, task_class=None, options=None,
                   session_id="", purpose=""):
        self.turns.append(
            {"purpose": purpose, "tools": len(getattr(options, "tools", []) or [])}
        )
        return self.script.pop(0) if self.script else FakeResponse("done")

    @property
    def offered_tools_on(self):
        return [t["tools"] > 0 for t in self.turns]


@pytest.mark.asyncio
async def test_the_closing_turn_is_offered_no_tools(monkeypatch):
    """A turn that cannot call tools is a turn that must answer."""
    from mimir.council import specialists as mod

    # Two gathering turns spend the two-call budget, then the closing turn.
    router = RecordingRouter(
        [
            FakeResponse("", [FakeCall()]),
            FakeResponse("", [FakeCall()]),
            FakeResponse("Here is what I found."),
        ]
    )
    spec = _specialist(mod, router, monkeypatch, budget=SpecialistBudget(
        max_tool_calls=2, max_iterations=6
    ))
    run = await spec.run("objective", _state(), ctx=None)
    assert router.turns[-1]["purpose"].endswith(":conclude")
    assert router.turns[-1]["tools"] == 0
    assert run.report.error is None
    assert "what I found" in run.report.detail


@pytest.mark.asyncio
async def test_exhausting_the_tool_budget_ends_gathering_immediately(monkeypatch):
    """The old loop kept asking with tools it could not honour, dropped every
    call, and burned one iteration per dropped call."""
    from mimir.council import specialists as mod

    router = RecordingRouter(
        [FakeResponse("", [FakeCall()])] * 10 + [FakeResponse("Report.")]
    )
    spec = _specialist(mod, router, monkeypatch, budget=SpecialistBudget(
        max_tool_calls=1, max_iterations=6
    ))
    await spec.run("objective", _state(), ctx=None)
    # One gathering turn spends the single tool call, then the closing turn.
    assert len(router.turns) == 2
    assert router.offered_tools_on == [True, False]


@pytest.mark.asyncio
async def test_a_specialist_that_answers_early_never_pays_for_a_closing_turn(
    monkeypatch,
):
    from mimir.council import specialists as mod

    router = RecordingRouter([FakeResponse("Immediate answer.")])
    spec = _specialist(mod, router, monkeypatch)
    run = await spec.run("objective", _state(), ctx=None)
    assert len(router.turns) == 1
    assert run.report.error is None


@pytest.mark.asyncio
async def test_an_empty_closing_turn_is_reported_not_swallowed(monkeypatch):
    """A specialist that gathered and could not write up is a different thing
    from one that never looked. They must not produce the same report."""
    from mimir.council import specialists as mod

    router = RecordingRouter(
        [FakeResponse("", [FakeCall()]), FakeResponse("")]
    )
    spec = _specialist(mod, router, monkeypatch, budget=SpecialistBudget(
        max_tool_calls=1, max_iterations=6
    ))
    run = await spec.run("objective", _state(), ctx=None)
    assert run.report.error
    assert "no conclusion" in run.report.error


def _state():
    from mimir.models.state import InvestigationState

    return InvestigationState(user_request="which pods are running?")


def _specialist(mod, router, monkeypatch, budget=None):
    from mimir.models.specialist import SpecialistName

    spec = mod.Specialist(
        name=SpecialistName.LOG_ANALYST,
        router=router,
        registry=_registry(),
        budget=budget or SpecialistBudget(),
    )

    async def _invoke(call, by_name, ctx):
        from mimir.tools.base import ToolResult

        return ToolResult(tool=call.name, ok=True, output="", evidence=[])

    monkeypatch.setattr(spec, "_invoke", _invoke)
    return spec


def _registry():
    class R:
        def select(self, **kwargs):
            return []

        def specs_for(self, *a, **k):
            return []

        def openai_schemas(self, specs):
            return [{"type": "function", "function": {"name": "read_pods"}}]

        def names(self):
            return ["read_pods"]

    return R()
