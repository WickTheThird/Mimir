"""/v1 answered by MIMIR's loop: tool progress and the answer, as content."""

import pytest

from mimir.api.agent_mode import ASSISTANT_TOOLS, instruction_from, run_agent
from mimir.llm.base import LLMMessage


def test_the_last_user_turn_is_the_task_and_prior_turns_are_context():
    msgs = [LLMMessage.user("where is coexistence set up?"), LLMMessage.assistant("I looked in src."),
            LLMMessage.user("no, trace the logic")]
    text = instruction_from(msgs)
    assert text.endswith("Now: no, trace the logic") and "assistant: I looked in src." in text
    assert instruction_from([LLMMessage.user("hi")]) == "hi"
    assert instruction_from([]) == ""


def test_the_tool_set_is_read_only():
    assert not any(t.startswith(("write_", "edit_", "insert_", "prepare_mutation", "exec")) for t in ASSISTANT_TOOLS)
    assert "search_repository" in ASSISTANT_TOOLS and "repository_map" in ASSISTANT_TOOLS


@pytest.mark.asyncio
async def test_progress_and_answer_stream_as_text(monkeypatch, settings):
    from mimir.agent.events import AgentEvent, AgentEventType
    from mimir.tools.base import ToolResult

    class FakeAgent:
        def __init__(self, **kw): self.outcome = type("O", (), {"stopped": "done"})()
        async def run(self, instruction):
            yield AgentEvent(type=AgentEventType.TOOL_START, tool="search_repository", arguments={"query": "coexist"})
            yield AgentEvent(type=AgentEventType.TOOL_END, tool="search_repository",
                             result=ToolResult(ok=True, tool="search_repository", summary="3 matches in 2 files\nmore"))
            yield AgentEvent(type=AgentEventType.TEXT, text="Coexistence is configured in pkg/wa.py.")
    import mimir.agent.ops as ops
    monkeypatch.setattr(ops, "OpsAgent", FakeAgent)

    class Reg:
        def get(self, name): return object()
    class Runner:
        router = None; registry = Reg()
        def tool_context(self, sid): return None
    settings.api.facade_agent_timeout_s = 60
    out = "".join([t async for t in run_agent([LLMMessage.user("why is the api pod restarting in messaging-squad?")], Runner(), settings)])
    assert "[surface: cluster]" in out
    assert "> search_repository(query='coexist')" in out
    assert "ok: 3 matches in 2 files" in out
    assert out.strip().endswith("Coexistence is configured in pkg/wa.py.")


@pytest.mark.asyncio
async def test_a_repeating_loop_says_so_instead_of_going_quiet(monkeypatch, settings):
    from mimir.agent.events import AgentEvent, AgentEventType

    class FakeAgent:
        def __init__(self, **kw): self.outcome = type("O", (), {"stopped": "repeating"})()
        async def run(self, instruction):
            yield AgentEvent(type=AgentEventType.TOOL_START, tool="search_repository", arguments={"query": "x"})
    import mimir.agent.ops as ops
    monkeypatch.setattr(ops, "OpsAgent", FakeAgent)
    class Reg:
        def get(self, name): return object()
    class Runner:
        router = None; registry = Reg()
        def tool_context(self, sid): return None
    out = "".join([t async for t in run_agent([LLMMessage.user("q")], Runner(), settings)])
    assert "stopped repeating" in out and "no conclusion was written" in out


def test_the_surface_is_read_from_the_words_used():
    from mimir.api.agent_mode import classify_surface

    assert classify_surface("in what files is whatsapp coexistence setup? (readonly search through the repo)") == "repository"
    assert classify_surface("why is the api pod restarting in messaging-squad") == "cluster"
    assert classify_surface("does the repo config match the deployment replicas in the cluster") == "both"


def test_the_search_phrase_drops_the_asking_words():
    from mimir.api.agent_mode import search_phrase

    assert search_phrase("can we check in what files is whatsapp coexistence setup? (readonly search through the repo target)") == "whatsapp coexistence target"


@pytest.mark.asyncio
async def test_a_repository_question_searches_first_and_never_offers_cluster_tools(monkeypatch, settings):
    from mimir.agent.events import AgentEvent, AgentEventType
    import mimir.agent.loop as loop_mod
    import mimir.mcp.server as mcp_server

    seen = {}
    class FakeLoop:
        def __init__(self, **kw):
            seen["tools"] = kw["tools"]; seen["system"] = kw["system"]; seen["instruction"] = None
            self.outcome = type("O", (), {"stopped": "done"})()
        async def run(self, instruction):
            seen["instruction"] = instruction
            yield AgentEvent(type=AgentEventType.TEXT, text="It is set up in pkg/wa.py.")
    async def fake_search(query, repo=None, limit=40):
        seen["query"] = (query, repo)
        return {"files": ["pkg/wa.py"], "matches": [{"path": "pkg/wa.py", "line": 3, "text": "coexistence = True"}], "tried": [query]}
    monkeypatch.setattr(loop_mod, "AgentLoop", FakeLoop)
    monkeypatch.setattr(mcp_server, "search_code_impl", fake_search)
    class Reg:
        def get(self, name): return object()
    class Runner:
        router = None; registry = Reg()
        def tool_context(self, sid): return None
    Runner.settings = settings
    out = "".join([t async for t in run_agent([LLMMessage.user("in what files is whatsapp coexistence set up? readonly search through the repo")], Runner(), settings)])
    assert "[surface: repository]" in out and "pkg/wa.py" in out
    assert "find_workloads" not in seen["tools"] and "search_repository" in seen["tools"]
    assert "Repository search already ran" in seen["instruction"] and "coexistence = True" in seen["instruction"]
    assert seen["query"][0] == "whatsapp coexistence"
