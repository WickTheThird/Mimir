"""The map as a tool, and test selection from it."""

import subprocess

import pytest

from mimir.tools.base import ToolContext, load_all_tools


def _repo(tmp_path):
    root = tmp_path / "billing"
    (root / "src" / "billing").mkdir(parents=True); (root / "tests").mkdir()
    (root / "src" / "billing" / "__init__.py").write_text("")
    (root / "src" / "billing" / "retry.py").write_text("def backoff(n):\n    return 2 ** n\n")
    (root / "src" / "billing" / "client.py").write_text("from billing.retry import backoff\n\ndef fetch():\n    return backoff(1)\n")
    (root / "tests" / "test_client.py").write_text("from billing.client import fetch\n\ndef test_fetch():\n    assert fetch()\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], check=True)
    return root


@pytest.fixture
def ctx(tmp_path, settings):
    root = _repo(tmp_path)
    registry = load_all_tools()
    from mimir.tools.repo import get_repository_directory, reset_repository_directory
    reset_repository_directory()
    get_repository_directory(settings).register_session("billing", root, "fixture")
    return ToolContext(settings=settings, registry=registry), root


def test_the_map_tool_is_registered_and_sits_before_search():
    from mimir.agent.loop import CODING_TOOLS
    assert "repository_map" in load_all_tools().names()
    assert CODING_TOOLS.index("repository_map") < CODING_TOOLS.index("search_repository")


@pytest.mark.asyncio
async def test_a_symbol_query_returns_file_and_line(ctx):
    context, _ = ctx
    tool = context.registry.get("repository_map")
    result = await tool.invoke({"query": "backoff", "repo": "billing"}, context)
    assert result.ok and "src/billing/retry.py:1" in result.summary


@pytest.mark.asyncio
async def test_a_path_query_returns_the_outline_and_its_tests(ctx):
    context, _ = ctx
    tool = context.registry.get("repository_map")
    result = await tool.invoke({"query": "src/billing/retry.py", "repo": "billing"}, context)
    assert "backoff" in result.summary and "tests/test_client.py" in result.summary


@pytest.mark.asyncio
async def test_an_unknown_name_points_at_search_rather_than_inventing(ctx):
    context, _ = ctx
    tool = context.registry.get("repository_map")
    result = await tool.invoke({"query": "nonexistent_thing", "repo": "billing"}, context)
    assert result.ok and "search_repository" in result.summary


def test_affected_tests_follow_the_worktree_diff(ctx, tmp_path):
    from mimir.tools.repomap import affected_tests
    from mimir.worktree import WorktreeManager

    context, root = ctx
    wt = WorktreeManager(context.settings.home).create(root, "sel")
    (wt.root / "src" / "billing" / "retry.py").write_text("def backoff(n):\n    return n\n")
    assert affected_tests(context, root, wt.root) == ["tests/test_client.py"]
