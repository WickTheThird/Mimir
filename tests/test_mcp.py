"""The MCP surface runs the real path and executes nothing."""

import pytest

from mimir.mcp import server as mcp_server


def test_the_three_tools_are_registered():
    server = mcp_server.build_server()
    names = set()
    for attr in ("_tool_manager", "tools"):
        mgr = getattr(server, attr, None)
        if mgr is not None:
            listing = getattr(mgr, "list_tools", None) or getattr(mgr, "_tools", None)
            items = listing() if callable(listing) else listing
            names = {getattr(t, "name", t) for t in (items.values() if isinstance(items, dict) else items)}
            break
    assert {"construct_command", "investigate", "code_task"} <= names


class FakeCommand:
    def __init__(self):
        from mimir.models.command import ProposedCommand, TargetContext
        self.inner = ProposedCommand(
            argv=["kubectl", "get", "pods", "-n", "messaging-squad"],
            purpose="list pods", context=TargetContext(),
        )

    def __getattr__(self, item):
        return getattr(self.inner, item)


class FakeState:
    def __init__(self):
        from mimir.models.command import ProposedCommand, TargetContext
        self.commands_planned = [ProposedCommand(
            argv=["kubectl", "get", "pods", "-n", "messaging-squad"],
            purpose="list pods", context=TargetContext(),
        )]
        self.commands_executed = []
        self.session_id = "ses_test"
        self.final_answer = None
        self.error = None
        self.metadata = {}


class FakeRunner:
    def __init__(self):
        from mimir.config import get_settings
        self.settings = get_settings()
        self.calls = []

    async def run(self, question, *, environment=None, interface="cli"):
        self.calls.append((question, interface))
        return FakeState()


@pytest.mark.asyncio
async def test_construct_command_returns_argv_and_risk_and_runs_nothing(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(mcp_server, "_runner", runner)
    out = await mcp_server.construct_command_impl("list pods in messaging-squad")
    (cmd,) = out["commands"]
    assert cmd["argv"][0] == "kubectl"
    assert cmd["risk"]
    assert "Nothing was executed" in out["note"]
    assert runner.calls[0][1] == "mcp"
    assert "do not run it" in runner.calls[0][0]


@pytest.mark.asyncio
async def test_investigate_without_an_answer_says_so_instead_of_returning_empty(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(mcp_server, "_runner", runner)
    out = await mcp_server.investigate_impl("why is api restarting?")
    assert out["answer"] == ""
    assert "no answer" in out["error"]
