"""The MCP surface: how Warp reaches the real MIMIR."""

from __future__ import annotations

import asyncio
import re
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
    # Parser first: milliseconds when everything is stated. The graph is the fallback.
    from mimir.agent.command import construct_fast

    fast = construct_fast(
        request, context=context, namespace=namespace,
        entities=getattr(runner, "entities", None),
        kubectl=getattr(runner.settings.kubernetes, "binary", "kubectl") or "kubectl",
    )
    if fast:
        return {
            "commands": [_command_view(c, runner) for c in fast],
            "source": "parser",
            "note": "Nothing was executed. Built from the stated request; anything above R0 "
                    "needs the operator's approval before it runs.",
        }
    state = await runner.run(
        f"Construct the command for this request. Show it, do not run it: {request}",
        environment=_environment(context, namespace),
        interface="mcp",
    )
    commands = [_command_view(c, runner) for c in state.commands_planned]
    ran = [r.argv for r in state.commands_executed]
    return {
        "commands": commands,
        "source": "graph",
        "executed_readonly_probes": [" ".join(a) for a in ran[:10]],
        "note": (
            (f"The investigation ran {len(ran)} read-only probe(s) to construct this; " if ran
             else "Nothing was executed. ")
            + "the proposed command itself was not run, and anything above R0 needs the "
              "operator's approval through MIMIR's policy engine."
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
    """The coding loop in a task worktree."""
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
    from mimir.eval.coding import worktree_diff

    diff = worktree_diff(worktree.root)
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
    """What this task learned about the repository, proposed as a note."""
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
        from mimir.knowledge.skill_drafts import draft_from_coding, write_draft

        skill = draft_from_coding(
            repo, getattr(outcome, "instruction", "") or "", files_changed=sorted(outcome.files_changed),
            tools_used=sorted(getattr(outcome, "tools_used", []) or []), test_command=test_command,
        )
        if skill is not None:
            write_draft(runner.settings.home, skill)
    except Exception as exc:  # noqa: BLE001 - a lesson must not fail the task
        log.warning("repo_lesson_skipped", error=str(exc))


def _query_variants(phrase: str) -> list[str]:
    """A phrase as the model would type it, widened the way a person would grep it."""
    words = [w for w in re.findall(r"[A-Za-z0-9_]+", phrase) if len(w) > 2]
    out = [phrase.strip()] if phrase.strip() else []
    if len(words) >= 2:
        out.append(".*".join(words))
        out.append(".*".join(reversed(words)))
        out.append("|".join(words))
    out += [w for w in words if w.lower() not in out]
    # A crude stem so "coexistence" also finds "coexist" and "coexisting".
    out += [w[:-4] for w in words if w.lower().endswith(("ence", "ance", "tion", "sion")) and len(w) > 7]
    out += [w[:-3] for w in words if w.lower().endswith("ing") and len(w) > 6]
    return list(dict.fromkeys(v for v in out if v))


async def search_code_impl(query: str, repo: str | None = None, limit: int = 40) -> dict[str, Any]:
    """Index first, then text search over widening variants. No model, no loop."""
    runner = _get_runner()
    ctx = runner.tool_context(None)
    out: dict[str, Any] = {"query": query, "symbols": [], "matches": [], "tried": []}
    try:
        from mimir.tools.repomap import repository_map

        for word in [w for w in re.findall(r"[A-Za-z0-9_]+", query) if len(w) > 2][:3]:
            res = await runner.registry.get("repository_map").invoke({"query": word, "repo": repo}, ctx)
            out["symbols"] += (res.data or {}).get("symbols", []) if res.ok else []
    except Exception as exc:  # noqa: BLE001 - the index is a bonus, not a requirement
        out["index_error"] = str(exc)[:200]
    spec = runner.registry.get("search_repository")
    seen: set[tuple[str, int]] = set()
    docs = (".md", ".rst", ".txt")
    for variant in _query_variants(query):
        out["tried"].append(variant)
        res = await spec.invoke({"query": variant, "repo": repo, "max_results": limit, "case_sensitive": False}, ctx)
        rows = (res.data or {}).get("matches") or [] if res.ok else []
        for m in rows:
            if not isinstance(m, dict):
                continue
            key = (str(m.get("path") or m.get("file")), int(m.get("line") or 0))
            if key not in seen:
                seen.add(key)
                out["matches"].append({**m, "matched_with": variant})
                out.setdefault("matched_with", variant)
        # Stop widening once code files, not only docs, have matched.
        if any(not str(m.get("path") or m.get("file")).endswith(docs) for m in out["matches"]):
            break
    out["matches"].sort(key=lambda m: (str(m.get("path") or m.get("file")).endswith(docs),
                                       str(m.get("path") or m.get("file")), int(m.get("line") or 0)))
    out["matches"] = out["matches"][:limit]
    out["files"] = sorted({str(m.get("path") or m.get("file")) for m in out["matches"]},
                          key=lambda f: (f.endswith(docs), f))
    out["note"] = ("No match for the phrase or any widening of it; the thing may be named "
                   "differently. Try a symbol, a config key, or an error string.") if not out["matches"] else ""
    return out


def build_server() -> Any:
    """The MCP server with the three tools registered."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "mimir",
        instructions=(
            "MIMIR: a local operations and coding assistant. construct_command "
            "builds a reviewed command and its risk class and never runs it. "
            "search_code finds where something lives in a repository without a model. "
            "investigate runs a full evidence-gated investigation. code_task makes "
            "a change in an isolated worktree and returns the diff for review."
        ),
    )

    @server.tool()
    async def construct_command(
        request: str, context: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Build the command for a natural-language request."""
        return await construct_command_impl(request, context, namespace)

    @server.tool()
    async def search_code(query: str, repo: str | None = None, limit: int = 40) -> dict[str, Any]:
        """Read-only search of a repository: symbol index first, then text search that
        widens the phrase (all words in order, reversed, any word) until something
        matches. Returns files and matching lines. No model involved; use this
        before investigate for "where is X set up" questions."""
        return await search_code_impl(query, repo, limit)

    @server.tool()
    async def investigate(
        question: str, context: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Investigate an operational question with evidence."""
        return await investigate_impl(question, context, namespace)

    @server.tool()
    async def code_task(
        instruction: str, repo: str | None = None, task: str | None = None,
        test_command: str = "", multi_step: bool = False,
    ) -> dict[str, Any]:
        """Carry out a coding instruction in an isolated task worktree and return the diff and test outcome."""
        return await code_task_impl(instruction, repo, task, test_command, multi_step)

    return server


class KeyRequired:
    """ASGI middleware: the same API keys the facade uses, on the MCP transport."""

    def __init__(self, app: Any, settings: Any) -> None:
        from mimir.api.auth import RateLimiter

        self.app = app
        self.settings = settings
        self.limiter = RateLimiter(settings.api.facade_rate_limit_per_minute)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        from mimir.api.auth import is_loopback

        client = (scope.get("client") or ("", 0))[0] or ""
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        auth = headers.get("authorization", "")
        presented = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-api-key")
        from mimir.api.auth import validate_key

        label = validate_key(self.settings, presented)
        allowed = label is not None or (
            is_loopback(client) and self.settings.api.allow_loopback_without_auth
        )
        if not allowed:
            log.warning("mcp_unauthenticated", address=client, path=scope.get("path"))
            await self._reject(send, 401, b'{"error":"api key required"}',
                               extra=[(b"www-authenticate", b"Bearer")])
            return
        try:
            length = int(headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > self.settings.api.max_request_bytes:
            await self._reject(send, 413, b'{"error":"request too large"}')
            return
        if not self.limiter.check(label or client):
            await self._reject(send, 429, b'{"error":"rate limited"}')
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send: Any, status: int, body: bytes, extra: list | None = None) -> None:
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), *(extra or [])]})
        await send({"type": "http.response.body", "body": body})


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8010) -> None:
    server = build_server()
    if transport == "stdio":
        server.run(transport="stdio")
        return
    import uvicorn

    from mimir.config import get_settings

    settings = get_settings()
    active = [k for k in settings.api.keys if not k.revoked] or settings.api.api_keys
    if host not in ("127.0.0.1", "localhost", "::1") and not active:
        raise SystemExit(
            "refusing to serve MCP on a non-loopback host with no API keys configured; "
            "run `mimir keys create --label warp` first"
        )
    app = KeyRequired(server.streamable_http_app(stateless_http=True), settings)
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = [
    "build_server",
    "code_task_impl",
    "construct_command_impl",
    "investigate_impl",
    "search_code_impl",
    "serve",
]
