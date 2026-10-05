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


def test_query_variants_widen_a_phrase_the_way_a_person_would():
    from mimir.mcp.server import _query_variants

    v = _query_variants("whatsapp coexistence")
    assert v[0] == "whatsapp coexistence"
    assert "whatsapp.*coexistence" in v and "coexistence.*whatsapp" in v and "whatsapp|coexistence" in v
    assert "whatsapp" in v and "coexistence" in v


def test_the_fourth_tool_is_registered():
    server = mcp_server.build_server()
    mgr = getattr(server, "_tool_manager", None)
    listing = getattr(mgr, "list_tools", None) or getattr(mgr, "_tools", None)
    items = listing() if callable(listing) else listing
    names = {getattr(t, "name", t) for t in (items.values() if isinstance(items, dict) else items)}
    assert "search_code" in names


@pytest.mark.asyncio
async def test_search_code_tries_variants_until_one_matches(monkeypatch):
    from mimir.tools.base import ToolResult

    class Spec:
        def __init__(self, hits_on): self.hits_on, self.seen = hits_on, []
        async def invoke(self, args, ctx):
            self.seen.append(args["query"])
            ok = args["query"] == self.hits_on
            return ToolResult(ok=True, tool="search_repository", summary="",
                              data={"matches": [{"path": "pkg/wa.py", "line": 3, "text": "coexistence = True"}] if ok else []})
    search = Spec("whatsapp|coexistence")
    class Reg:
        def get(self, name): return search if name == "search_repository" else None
    class Runner:
        registry = Reg(); settings = None
        def tool_context(self, sid): return None
    monkeypatch.setattr(mcp_server, "_runner", Runner())
    out = await mcp_server.search_code_impl("whatsapp coexistence")
    assert out["matched_with"] == "whatsapp|coexistence" and out["files"] == ["pkg/wa.py"]
    assert search.seen[:3] == ["whatsapp coexistence", "whatsapp.*coexistence", "coexistence.*whatsapp"]


def test_variants_include_a_crude_stem():
    from mimir.mcp.server import _query_variants

    assert "coexist" in _query_variants("whatsapp coexistence")


@pytest.mark.asyncio
async def test_search_code_keeps_widening_past_a_docs_only_hit(monkeypatch):
    from mimir.tools.base import ToolResult

    class Spec:
        def __init__(self): self.seen = []
        async def invoke(self, args, ctx):
            q = args["query"]; self.seen.append(q)
            rows = {"whatsapp coexistence": [{"path": "AGENTS.md", "line": 261, "text": "## WhatsApp Coexistence"}],
                    "coexist": [{"path": "app/coexistence/service.py", "line": 12, "text": "class CoexistenceService"}]}.get(q, [])
            return ToolResult(ok=True, tool="search_repository", summary="", data={"matches": rows})
    search = Spec()
    class Reg:
        def get(self, name): return search if name == "search_repository" else None
    class Runner:
        registry = Reg(); settings = None
        def tool_context(self, sid): return None
    monkeypatch.setattr(mcp_server, "_runner", Runner())
    out = await mcp_server.search_code_impl("whatsapp coexistence")
    assert out["files"][0] == "app/coexistence/service.py"
    assert "AGENTS.md" in out["files"] and "coexist" in search.seen
