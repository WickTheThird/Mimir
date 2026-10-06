"""MIMIR as the agent behind /v1: its own tool loop answers, Warp just displays."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

ASSISTANT_TOOLS: tuple[str, ...] = (
    "repository_map", "search_repository", "read_file_range", "locate_tests",
    "find_workloads", "get_current_context", "list_workloads", "summarise_pod_health",
    "get_logs", "get_events", "describe_resource", "search_memory",
)
"""Read-only. The loop reads the estate and the repositories; it never mutates."""


def instruction_from(messages: list[Any], *, turns: int = 6) -> str:
    """The last user message as the task, with recent turns as context."""
    users = [m for m in messages if getattr(m, "role", "") == "user"]
    if not users:
        return ""
    task = (users[-1].content or "").strip()
    prior = [m for m in messages if getattr(m, "role", "") in ("user", "assistant")][:-1][-turns:]
    if not prior:
        return task
    context = "\n".join(f"{m.role}: {(m.content or '').strip()[:400]}" for m in prior if (m.content or "").strip())
    return f"Conversation so far:\n{context}\n\nNow: {task}"


def _render_args(arguments: dict[str, Any]) -> str:
    shown = {k: v for k, v in arguments.items() if k not in ("task", "repo", "view") and v not in (None, "", [])}
    return ", ".join(f"{k}={str(v)[:40]!r}" for k, v in list(shown.items())[:4])


REPO_TOOLS: tuple[str, ...] = ("trace_feature", "repository_map", "find_symbol", "find_references",
                               "search_repository", "read_file_range", "locate_tests", "search_memory")
CLUSTER_TOOLS: tuple[str, ...] = tuple(t for t in ASSISTANT_TOOLS if t not in REPO_TOOLS) + ("search_memory",)

_REPO_CUES = re.compile(r"\b(repo|repository|repositories|codebase|source|file|files|code|module|function|class|"
                        r"defined|definition|implemented|implementation|set ?up|configured|config|where is|call site|caller)\b", re.I)
_TRACE_CUES = re.compile(r"\b(trace|end to end|end-to-end|flow|how does .{0,40} work|walk me through|implemented|implementation)\b", re.I)
_CLUSTER_CUES = re.compile(r"\b(pod|pods|namespace|cluster|context|deployment|deployments|logs?|restart|restarts|"
                           r"crashloop|oom|events?|rollout|kubectl|node|nodes|replica|replicas)\b", re.I)

REPO_SYSTEM = """\
You are MIMIR answering a question about a code repository, read-only.
A trace or a repository search has already run and its results are in the
request. Start from those: read the ranges that matter with read_file_range,
and name files and lines. For a trace, write it as ordered hops: entry route,
handler, what it calls, where it is stored, what consumes it, its states. Do not guess at file names; only name what a tool returned. If the
search found nothing, say what you searched for and suggest what the thing
might be called instead. Never describe a cluster for a question about code.
"""


def classify_surface(text: str) -> str:
    """repository, cluster or both, from the words the operator used; no model."""
    repo = len(_REPO_CUES.findall(text))
    cluster = len(_CLUSTER_CUES.findall(text))
    if repo and not cluster:
        return "repository"
    if cluster and not repo:
        return "cluster"
    if repo and cluster:
        return "repository" if repo >= 2 * cluster else ("cluster" if cluster >= 2 * repo else "both")
    return "both"


def pick_repo(text: str, runner: Any) -> str | None:
    """A registered repository the request names, or the only one, or None."""
    try:
        from mimir.tools.repo import get_repository_directory

        repos = get_repository_directory(runner.settings).all()
    except Exception:  # noqa: BLE001
        return None
    lowered = text.lower()
    for r in repos:
        if r.name.lower() in lowered:
            return r.name
    return repos[0].name if len(repos) == 1 else None


def repo_words(repo: str | None, runner: Any) -> tuple[str, ...]:
    """The repository's own name and folder name, split into words; they match everything in it."""
    words: list[str] = []
    if repo:
        words += re.findall(r"[A-Za-z0-9]+", repo)
        try:
            from mimir.tools.repo import get_repository_directory

            words += re.findall(r"[A-Za-z0-9]+", get_repository_directory(runner.settings).resolve(repo).root.name)
        except Exception:  # noqa: BLE001
            pass
    return tuple(dict.fromkeys(w.lower() for w in words))


def search_phrase(text: str, *, exclude: tuple[str, ...] = ()) -> str:
    """The part of the request worth searching for: drop the asking words and repo names."""
    drop = {e.lower() for e in exclude}
    words = [w for w in re.findall(r"[A-Za-z0-9_-]+", text)
             if len(w) > 2 and not _REPO_CUES.fullmatch(w) and not _CLUSTER_CUES.fullmatch(w) and w.lower() not in drop]
    stop = {"can", "you", "what", "which", "how", "the", "and", "for", "our", "check", "find", "look", "read", "readonly", "read-only", "search", "through", "this", "that", "please", "show", "where", "setup", "set", "about", "with", "are", "is"}
    kept = [w for w in words if w.lower() not in stop][:4]
    return " ".join(kept)


async def run_agent(messages: list[Any], runner: Any, settings: Any) -> AsyncIterator[str]:
    """Yield progress lines and the answer as it is produced."""
    from mimir.agent.events import AgentEventType
    from mimir.agent.loop import AgentLoop
    from mimir.agent.ops import OpsAgent

    instruction = instruction_from(messages)
    if not instruction:
        yield "No request found in the conversation."
        return
    surface = classify_surface(instruction)
    yield f"[surface: {surface}]\n"
    available = lambda names: tuple(t for t in names if runner.registry.get(t) is not None)  # noqa: E731
    if surface == "repository":
        from mimir.mcp.server import search_code_impl

        repo = pick_repo(instruction, runner)
        phrase = search_phrase(instruction, exclude=repo_words(repo, runner))
        if phrase and _TRACE_CUES.search(instruction) and runner.registry.get("trace_feature") is not None:
            res = await runner.registry.get("trace_feature").invoke({"feature": phrase, "repo": repo}, runner.tool_context(None))
            if res.ok:
                from mimir.tools.trace import answer_markdown

                yield f"> trace_feature({phrase!r}, repo={repo!r}) -> {len(res.data['routes'])} routes, "
                yield f"{len(res.data['handlers'])} handlers, {len(res.data['states'])} states\n"
                body = answer_markdown(res.data)
                overview = await _overview(runner, instruction, res.summary)
                yield "\n" + (overview + "\n\n" if overview else "") + body + "\n"
                return
        found = await search_code_impl(phrase, repo) if phrase else {"files": [], "matches": [], "tried": []}
        yield f"> search_code({phrase!r}, repo={repo!r}) -> {len(found.get('files', []))} file(s)\n"
        for f in found.get("files", [])[:12]:
            yield f"  {f}\n"
        evidence = "\n".join(
            f"- {m.get('path') or m.get('file')}:{m.get('line')}: {str(m.get('text') or m.get('line_text') or '')[:160]}"
            for m in found.get("matches", [])[:25] if isinstance(m, dict)
        ) or "(nothing matched: " + ", ".join(found.get("tried", [])) + ")"
        tried = ", ".join(found.get("tried", [])[:6])
        instruction = (f"{instruction}\n\nRepository search already ran (repo={repo or 'default'}; "
                       f"patterns tried: {tried}). Do not repeat these searches; read the files instead.\n{evidence}")
        agent = AgentLoop(
            router=runner.router, registry=runner.registry, tool_context=runner.tool_context(None),
            settings=settings, tools=available(REPO_TOOLS), system=REPO_SYSTEM,
            task_class="fast_command", max_steps=settings.api.facade_agent_max_steps,
        )
    else:
        agent = OpsAgent(
            router=runner.router, registry=runner.registry, tool_context=runner.tool_context(None),
            settings=settings, environment=None,
            tools=available(CLUSTER_TOOLS if surface == "cluster" else ASSISTANT_TOOLS),
            task_class="fast_command", max_steps=settings.api.facade_agent_max_steps,
        )
    async for line in _drive(agent, instruction, runner, settings):
        yield line


async def _drive(agent: Any, instruction: str, runner: Any, settings: Any) -> AsyncIterator[str]:
    from mimir.agent.events import AgentEventType

    budget = settings.api.facade_agent_timeout_s
    started = time.time()
    pending: list[str] = []
    """Text since the last tool call; interim narration is dropped, the last turn is the answer."""
    try:
        async for event in agent.run(instruction):
            if time.time() - started > budget:
                yield f"\n[MIMIR: stopped after {budget:.0f}s]\n"
                break
            if event.type is AgentEventType.TOOL_START:
                pending.clear()
                yield f"> {event.tool}({_render_args(event.arguments)})\n"
            elif event.type is AgentEventType.TOOL_END:
                summary = (event.result.summary if event.result is not None else "").strip().splitlines()
                head = summary[0][:160] if summary else ("ok" if event.result and event.result.ok else "failed")
                mark = "ok" if (event.result is None or event.result.ok) else "failed"
                yield f"  {mark}: {head}\n"
            elif event.type is AgentEventType.TEXT and event.text:
                pending.append(event.text)
            elif event.type is AgentEventType.ERROR:
                yield f"\n[MIMIR: {event.error}]\n"
    except Exception as exc:  # noqa: BLE001 - the client must get a reply, not a dropped stream
        log.exception("facade_agent_failed")
        yield f"\n[MIMIR: {type(exc).__name__}: {exc}]\n"
    answer = "".join(pending).strip()
    if not answer:
        answer = await _conclude(agent, runner)
    yield "\n" + (answer or "(no conclusion was written; the tool results above are what was found)") + "\n"


async def _overview(runner: Any, question: str, trace_text: str) -> str:
    """Two or three sentences on what the flow does; the cited hops are not the model's to write."""
    from mimir.llm.base import GenerationOptions, LLMMessage, ModelError

    prompt = (
        "Below is a trace extracted from the code. In two or three plain sentences, say what this "
        "feature does from the first request to completion. Do not list files or line numbers, do "
        "not invent anything not in the trace; a cited list follows your sentences.\n\n"
        f"Question: {question}\n\nTrace:\n{trace_text[:6000]}"
    )
    try:
        response = await runner.router.chat(
            [LLMMessage.user(prompt)], task_class="fast_command",
            options=GenerationOptions(tools=[], temperature=0.0, max_tokens=220), purpose="facade:overview")
    except ModelError as exc:
        log.warning("facade_overview_failed", error=exc.message)
        return ""
    text = (response.content or "").strip()
    return text if len(text) < 1200 else ""


async def _conclude(agent: Any, runner: Any) -> str:
    """One closing turn with no tools when the loop stopped without answering."""
    from mimir.llm.base import GenerationOptions, LLMMessage, ModelError

    messages = list(getattr(agent, "messages", []) or [])
    if not messages:
        return ""
    messages.append(LLMMessage.user(
        "Stop calling tools. Answer the original question now from what the tools returned: "
        "name the files and lines, in a short list, and say what each one does."))
    try:
        response = await runner.router.chat(
            messages, task_class=getattr(agent, "task_class", None),
            options=GenerationOptions(tools=[], temperature=0.0, max_tokens=900), purpose="facade:conclude")
    except ModelError as exc:
        log.warning("facade_conclude_failed", error=exc.message)
        return ""
    return (response.content or "").strip()


__all__ = ["repo_words", "ASSISTANT_TOOLS", "CLUSTER_TOOLS", "REPO_TOOLS", "classify_surface", "instruction_from", "pick_repo", "run_agent", "search_phrase"]
