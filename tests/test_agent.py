"""Coding loop tests.

The loop is the one place in MIMIR where a model's output changes files, so
these pin the boundaries rather than the prose: what the model is allowed to
see, what it is never asked to supply, and what happens when it is wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mimir.agent.events import AgentEventType, TimelineEntry
from mimir.agent.loop import CODING_TOOLS, CodingAgent
from mimir.llm.base import ChunkType, StreamChunk, ToolCall
from mimir.tools.base import ToolContext, ToolResult, load_all_tools


class ScriptedModel:
    """Replays a fixed sequence of turns, so a test never needs a runtime."""

    alias = "test"
    model = "scripted"
    context_window = 8192

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list] = []

    async def stream(self, messages, options=None):
        self.seen.append(list(messages))
        text, calls = self.turns.pop(0) if self.turns else ("done", [])
        if text:
            yield StreamChunk(type=ChunkType.CONTENT, text=text)
        for call in calls:
            yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call)
        yield StreamChunk(
            type=ChunkType.DONE, finish_reason="tool_calls" if calls else "stop"
        )


class ScriptedRouter:
    def __init__(self, model):
        self.model = model
        self.call_log: list = []
        self.invocations_attempted = 0

    def for_task(self, task_class):
        return self.model

    def digest_for(self, alias):
        return "sha256:test"


def _agent(turns, registry=None, root="/tmp/wt"):
    model = ScriptedModel(turns)
    router = ScriptedRouter(model)
    registry = registry or load_all_tools()
    agent = CodingAgent(
        router=router,
        registry=registry,
        tool_context=ToolContext(registry=registry),
        task="fix-thing",
        repo="billing",
        view="fix-thing-worktree",
        worktree_root=Path(root),
    )
    return agent, model, router


async def _drain(agent, instruction="do the thing"):
    return [event async for event in agent.run(instruction)]


class TestBoundArguments:
    """The worktree is supplied by the loop, never by the model."""

    def test_the_model_is_not_shown_the_task_or_repo_arguments(self):
        agent, _, _ = _agent([])
        for schema in agent.schemas():
            parameters = schema["function"]["parameters"]
            assert "task" not in parameters.get("properties", {})
            assert "repo" not in parameters.get("properties", {})
            assert "task" not in parameters.get("required", [])
            assert "repo" not in parameters.get("required", [])

    def test_a_write_is_bound_to_the_source_repo_and_the_task(self):
        agent, _, _ = _agent([])
        bound = agent._bind("edit_worktree_file", {"path": "a.py"})
        assert bound["task"] == "fix-thing"
        assert bound["repo"] == "billing", "worktree tools derive the path from the repo"

    def test_a_read_is_bound_to_the_worktree_not_the_checkout(self):
        """The failure this prevents: editing a file and then reading back the
        version without the edit, which looks exactly like a hallucination."""
        agent, _, _ = _agent([])
        assert agent._bind("read_file_range", {"path": "a.py"})["repo"] == (
            "fix-thing-worktree"
        )
        assert agent._bind("search_repository", {"query": "x"})["repo"] == (
            "fix-thing-worktree"
        )

    def test_a_tool_without_those_fields_is_left_alone(self):
        agent, _, _ = _agent([])
        assert agent._bind("list_repositories", {"query": "x"}) == {"query": "x"}


class TestTheLoop:
    def test_it_stops_when_the_model_stops_calling_tools(self):
        agent, _, _ = _agent([("all done", [])])
        events = asyncio.run(_drain(agent))
        assert events[-1].type is AgentEventType.DONE
        assert agent.outcome.stopped == "done"
        assert agent.outcome.steps == 1

    def test_a_tool_result_is_fed_back_before_the_next_step(self):
        call = ToolCall(id="c1", name="list_repositories", arguments={})
        agent, model, _ = _agent([("looking", [call]), ("found it", [])])
        asyncio.run(_drain(agent))

        second = model.seen[1]
        assert second[-1].role.value == "tool"
        assert second[-1].tool_call_id == "c1"

    def test_it_refuses_to_run_forever(self):
        call = ToolCall(name="list_repositories", arguments={})
        agent, _, _ = _agent([("x", [call])] * 40)
        agent.max_steps = 3
        events = asyncio.run(_drain(agent))
        assert agent.outcome.stopped == "max_steps"
        assert events[-1].type is AgentEventType.ERROR
        assert "3 steps" in events[-1].error

    def test_the_conversation_survives_the_turn(self):
        """A follow-up that re-reads every file already read is most of what
        makes a local model feel unusable."""
        agent, _, _ = _agent([("one", []), ("two", [])])
        asyncio.run(_drain(agent, "first"))
        asyncio.run(_drain(agent, "second"))
        contents = [m.content for m in agent.messages]
        assert "first" in contents and "second" in contents

    def test_a_model_error_ends_the_turn_rather_than_looping(self):
        class Failing(ScriptedModel):
            async def stream(self, messages, options=None):
                yield StreamChunk(type=ChunkType.ERROR, error="runtime unreachable")

        agent, _, _ = _agent([])
        agent.router.model = Failing([])
        events = asyncio.run(_drain(agent))
        assert events[-1].type is AgentEventType.ERROR
        assert agent.outcome.stopped == "error"


class TestTelemetry:
    def test_every_streamed_step_is_recorded_as_an_invocation(self):
        """The invariant that found the empty model_calls table: invocations
        observed equals records written. A new caller does not get an exemption
        because it streams."""
        call = ToolCall(name="list_repositories", arguments={})
        agent, _, router = _agent([("x", [call]), ("done", [])])
        asyncio.run(_drain(agent))
        assert router.invocations_attempted == 2
        assert len(router.call_log) == router.invocations_attempted
        assert all(r.purpose == "coding:fix-thing" for r in router.call_log)


class TestToolSurface:
    def test_the_surface_is_small_and_every_name_resolves(self):
        registry = load_all_tools()
        missing = [n for n in CODING_TOOLS if registry.get(n) is None]
        assert not missing, f"named but not registered: {missing}"
        assert len(CODING_TOOLS) <= 15, (
            "a loop pays the whole schema on every step; keep the surface small"
        )

    def test_it_can_read_resolve_change_and_verify(self):
        assert "read_file_range" in CODING_TOOLS
        assert "lsp_definition" in CODING_TOOLS
        assert "edit_worktree_file" in CODING_TOOLS
        assert "run_worktree_tests" in CODING_TOOLS


class TestTimeline:
    def test_an_empty_result_carries_the_constraint_that_emptied_it(self):
        """Shown without its glob, a search that matched nothing reads as a
        broken tool, and someone goes looking for the defect."""
        entry = TimelineEntry.of(
            1,
            "search_repository",
            {"query": "deliberately", "globs": ["**/agent/**"]},
            ToolResult(tool="search_repository", summary="0 matches"),
        )
        assert entry.subject == "deliberately"
        assert "**/agent/**" in entry.constraint
        assert entry.verb == "searched for"

    def test_a_failed_call_stays_in_the_timeline(self):
        entry = TimelineEntry.of(
            2,
            "read_file_range",
            {"path": "src/gone.py"},
            ToolResult(tool="read_file_range", ok=False, error="does not exist"),
        )
        assert not entry.ok
        assert entry.found == "does not exist"

    def test_evidence_is_what_marks_a_step_as_kept(self):
        from mimir.models.evidence import Evidence, SourceType

        kept = TimelineEntry.of(
            1, "read_file_range", {"path": "a.py"},
            ToolResult(tool="read_file_range", summary="ok",
                       evidence=[Evidence(claim="x", source_id="a.py",
                                        source_type=SourceType.REPOSITORY)]),
        )
        assert kept.kept


class TestEditTool:
    """The primitive the whole loop depends on."""

    @pytest.fixture
    def worktree(self, tmp_path):
        import subprocess

        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "a.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 1\n")
        for argv in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t",
                     "-c", "user.name=t", "commit", "-qm", "init"]):
            subprocess.run(["git", "-C", str(repo), *argv], check=True,
                           capture_output=True)
        return repo

    def _edit(self, monkeypatch, repo, tmp_path, **args):
        from mimir.tools import code

        registry = load_all_tools()
        manager_home = tmp_path / "home"
        ctx = ToolContext(registry=registry)
        monkeypatch.setattr(code, "_manager",
                            lambda c: __import__("mimir.worktree", fromlist=["x"])
                            .WorktreeManager(manager_home))

        async def repo_root(c, name):
            return repo

        monkeypatch.setattr(code, "_repo_root", repo_root)
        manager = code._manager(ctx)
        manager.create(repo, "t")
        return asyncio.run(
            registry.invoke("edit_worktree_file", {"task": "t", **args}, ctx)
        )

    def test_it_replaces_the_named_span(self, monkeypatch, worktree, tmp_path):
        result = self._edit(monkeypatch, worktree, tmp_path, path="src/a.py",
                            old_string="def f():", new_string="def h():")
        assert result.ok, result.error
        assert "line 1" in result.summary

    def test_an_ambiguous_span_is_refused_rather_than_guessed(
        self, monkeypatch, worktree, tmp_path
    ):
        """Two matches means the model does not know which one it means.
        Editing the first on its behalf edits the wrong line."""
        result = self._edit(monkeypatch, worktree, tmp_path, path="src/a.py",
                            old_string="    return 1", new_string="    return 2")
        assert not result.ok
        assert result.error_code == "ambiguous_match"
        assert "appears 2 times" in result.error

    def test_a_stale_belief_fails_loudly(self, monkeypatch, worktree, tmp_path):
        result = self._edit(monkeypatch, worktree, tmp_path, path="src/a.py",
                            old_string="def never_existed():", new_string="x")
        assert not result.ok
        assert result.error_code == "no_match"

    def test_it_will_not_create_a_file(self, monkeypatch, worktree, tmp_path):
        result = self._edit(monkeypatch, worktree, tmp_path, path="src/new.py",
                            old_string="a", new_string="b")
        assert not result.ok
        assert result.error_code == "not_found"
