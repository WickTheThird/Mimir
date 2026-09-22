"""The MCP surface: how Warp reaches the real MIMIR.

The OpenAI facade is a model gateway. Warp keeps its own agent loop and calls
MIMIR as if it were a model, so from Warp none of the graph, the gates, the
decision layer or the safety engine runs. Every accuracy gain in this
repository is behind the graph, and the facade never enters it.

This is the fix ADR-001 §16.4 preferred and `docs/warp.md` said was unbuilt:
three tools an outer agent invokes on purpose, each running the full path
with every gate, with execution and approval staying local. One agent loop
stays in charge, Warp's, and MIMIR is a set of things it can call that are
right for the reasons the harness measured. Nesting the graph behind the
model endpoint remains the thing to avoid.

Nothing here executes a command or applies a change. `construct_command`
returns an argument vector and its risk class. `code_task` returns a diff
in a task worktree. Both stop where the policy engine says an operator has to
say yes.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

_runner: Any = None


def _get_runner() -> Any:
    """One runner per server process, built on first use."""
    global _runner
    if _runner is None:
        from mimir.config import get_settings
        from mimir.graph.runner import InvestigationRunner

        _runner = InvestigationRunner(settings=get_settings())
    return _runner


def _environment(context: str | None, namespace: str | None, repo: str | None = None) -> Any:
    from mimir.models.state import EnvironmentContext

    return EnvironmentContext(
        cluster_context=context, namespace=namespace, repositories=[repo] if repo else []
    )


def _command_view(command: Any, runner: Any) -> dict[str, Any]:
    """What an operator needs to decide, and nothing that runs it."""
    from mimir.safety.policy import get_policy_engine

    assessment = get_policy_engine(runner.settings).classify_only(command)
    return {
        "display": command.display,
        "argv": list(command.argv),
        "purpose": command.purpose,
        "expected_effect": command.expected_effect,
        "target": dict(command.context.render_pairs()),
        "risk": str(assessment.risk),
        "risk_summary": assessment.summary,
        "requires_approval": assessment.risk not in ("R0", "r0")
        and str(assessment.risk).upper() != "R0",
    }


def _answer_view(state: Any) -> dict[str, Any]:
    answer = state.final_answer
    if answer is None:
        return {"answer": "", "error": state.error or "no answer produced"}
    return {
        "answer": answer.answer,
        "confidence": answer.confidence,
        "observed_facts": list(answer.observed_facts),
        "inferences": list(answer.inferences),
        "unverified": list(answer.unverified),
        "disagreements": list(answer.disagreements),
        "next_steps": list(getattr(answer, "next_steps", []) or []),
        "citations": len(answer.citations),
        "evidence": [
            {"claim": e.claim[:200], "source": e.source_id, "excerpt": e.excerpt[:300]}
            for e in state.ranked_evidence(limit=8)
        ],
        "commands_proposed": [c.display for c in state.commands_planned[:8]],
        "gates": {
            k: state.metadata.get(k)
            for k in ("sufficiency", "grounding", "retry_signature", "claim_support", "assess")
            if k in state.metadata
        },
        "decisions": state.metadata.get("decisions", []),
        "session_id": state.session_id,
    }


async def construct_command_impl(
    request: str, context: str | None = None, namespace: str | None = None
) -> dict[str, Any]:
    runner = _get_runner()
    state = await runner.run(
        f"Construct the command for this request. Show it, do not run it: {request}",
        environment=_environment(context, namespace),
        interface="mcp",
    )
    commands = [_command_view(c, runner) for c in state.commands_planned]
    return {
        "commands": commands,
        "note": (
            "Nothing was executed. Anything above R0 needs the operator's approval "
            "through MIMIR's policy engine before it runs."
        ),
        "session_id": state.session_id,
    }


async def investigate_impl(
    question: str, context: str | None = None, namespace: str | None = None
) -> dict[str, Any]:
    runner = _get_runner()
    state = await runner.run(
        question, environment=_environment(context, namespace), interface="mcp"
    )
    return _answer_view(state)


async def code_task_impl(
    instruction: str,
    repo: str | None = None,
    task: str | None = None,
    test_command: str = "",
    multi_step: bool = False,
) -> dict[str, Any]:
    """The coding loop in a task worktree. Returns the diff, never applies it.

    Mirrors the CLI's start_coding_session without a console. Reopening an
    existing task worktree is the common case and must not be destructive.
    """
    from mimir.agent.loop import AgentEventType, CodingAgent
    from mimir.tools.repo import get_repository_directory
    from mimir.worktree import WorktreeError, WorktreeManager

    runner = _get_runner()
    directory = get_repository_directory(runner.settings)
    resolved = directory.resolve(repo)
    manager = WorktreeManager(runner.settings.home)
    name = task or "mcp-task"
    try:
        worktree = manager.find(resolved.root, name)
    except WorktreeError:
        worktree = manager.create(resolved.root, name)
    view = f"{worktree.name}-worktree"
    directory.register_session(view, worktree.root, f"task worktree of {resolved.name}")
    agent = CodingAgent(
        router=runner.router,
        registry=runner.registry,
        tool_context=runner.tool_context(None),
        task=worktree.name,
        repo=resolved.name,
        view=view,
        worktree_root=worktree.root,
        settings=runner.settings,
    )
    text: list[str] = []
    steps: list[dict[str, Any]] = []
    if multi_step:
        from mimir.agent.plan import plan_steps, run_plan

        plan = await plan_steps(runner.router, instruction)
        plan = await run_plan(agent, plan, worktree_root=worktree.root)
        steps = [{"title": st.title, "status": st.status, "checkpoint": st.checkpoint}
                 for st in plan.steps]
        text.append(plan.render())
    else:
        async for event in agent.run(instruction):
            if event.type is AgentEventType.TEXT:
                text.append(event.text)
    diff = subprocess.run(
        ["git", "-C", str(worktree.root), "diff"], capture_output=True, text=True
    ).stdout
    outcome = agent.outcome
    await _propose_repo_lesson(runner, resolved.name, outcome, test_command)
    return {
        "worktree": str(worktree.root),
        "branch": worktree.branch,
        "diff": diff,
        "summary": "".join(text).strip()[:4000],
        "steps": steps,
        "outcome": {
            "stopped": outcome.stopped,
            "steps": outcome.steps,
            "tool_calls": outcome.tool_calls,
            "files_changed": sorted(outcome.files_changed),
            "tests_run": outcome.tests_run,
            "repeats": outcome.repeats,
        },
        "note": (
            "The change lives in the task worktree only. Nothing was applied to the "
            "operator's checkout, and nothing was pushed."
        ),
    }


async def _propose_repo_lesson(runner: Any, repo: str, outcome: Any, test_command: str) -> None:
    """What this task learned about the repository, proposed as a note.

    Plan step 8. Today ops sessions curate memory and coding tasks leave
    nothing; the next task on the same repository starts from zero. This is
    a proposal under repos/<name>, never a promotion.
    """
    try:
        from mimir.knowledge.experience import repo_lesson

        spec = runner.registry.get("propose_memory_note")
        if spec is None or not outcome.files_changed:
            return
        note = repo_lesson(
            repo, files_changed=sorted(outcome.files_changed), test_command=test_command,
            tools_used=sorted(getattr(outcome, "tools_used", []) or []), stopped=outcome.stopped,
        )
        await spec.invoke(note, runner.tool_context(None))
    except Exception as exc:  # noqa: BLE001 - a lesson must not fail the task
        log.warning("repo_lesson_skipped", error=str(exc))


def build_server() -> Any:
    """The MCP server with the three tools registered."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "mimir",
        instructions=(
            "MIMIR: a local operations and coding assistant. construct_command "
            "builds a reviewed command and its risk class and never runs it. "
            "investigate runs a full evidence-gated investigation. code_task makes "
            "a change in an isolated worktree and returns the diff for review."
        ),
    )

    @server.tool()
    async def construct_command(
        request: str, context: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Build the command for a natural-language request. Shows argv, the
        cluster context and namespace it targets, and its risk class. Never
        executes anything."""
        return await construct_command_impl(request, context, namespace)

    @server.tool()
    async def investigate(
        question: str, context: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Investigate an operational question with evidence. Returns the answer,
        what was observed, what remains unverified, and the commands it would
        run next, none of which have been run."""
        return await investigate_impl(question, context, namespace)

    @server.tool()
    async def code_task(
        instruction: str, repo: str | None = None, task: str | None = None,
        test_command: str = "", multi_step: bool = False,
    ) -> dict[str, Any]:
        """Carry out a coding instruction in an isolated task worktree and return
        the diff and test outcome. multi_step splits it into checkpointed steps.
        Nothing is applied to the working tree."""
        return await code_task_impl(instruction, repo, task, test_command, multi_step)

    return server


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8010) -> None:
    server = build_server()
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host=host, port=port, stateless_http=True)


__all__ = [
    "build_server",
    "code_task_impl",
    "construct_command_impl",
    "investigate_impl",
    "serve",
]
