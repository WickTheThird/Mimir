"""MIMIR behind /v1: understand the question, plan the evidence, gather it, check it, answer."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

INTENTS: tuple[str, ...] = ("locate", "trace", "explain", "gap", "scope", "live", "draft", "revise", "chat")
INTENT_HELP = {
    "locate": "where something is in the code",
    "trace": "how a feature flows end to end",
    "explain": "what some code or behaviour does, or why",
    "gap": "whether we support or implement something",
    "scope": "what it would take to build or change something",
    "live": "what is running or failing in the clusters right now",
    "draft": "write a reply or message for someone",
    "revise": "change, shorten, or double-check the previous answer",
    "chat": "anything else",
}


@dataclass
class Part:
    kind: str  # "progress" or "answer"
    text: str


@dataclass
class Turn:
    """What one request is about, decided before any tool runs."""

    question: str
    history: list[tuple[str, str]] = field(default_factory=list)
    intent: str = "chat"
    intent_source: str = "rule"
    tokens: list[str] = field(default_factory=list)
    repo: str | None = None
    evidence: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    steps: int = 0


# --- 1. what kind of question ------------------------------------------------

_R = lambda p: re.compile(p, re.I)  # noqa: E731
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("live", _R(r"\b(pods?|namespaces?|kubectl|logs?|restart(s|ing)?|crashloop\w*|oom\w*|rollouts?|"
                r"replicas?|(in|on) (prod|dev|staging)\b.*\b(running|deployed|up|down))\b")),
    ("scope", _R(r"\b(scope (it|this|that)?\s*out|scope|what (changes|would it take)|how (would|do|should) (we|i) "
                 r"(build|implement|add)|implementation plan|plan (it|this) out|estimate)\b")),
    ("gap", _R(r"\b(do we (support|implement|have|handle)|(is|are) [\w\s-]{1,40} (supported|implemented)\b|"
               r"does (our|the) (code|service|api|portal) (support|handle|do)|(code|capability) gap|not implemented|missing)\b")),
    ("draft", _R(r"\b(what should (the|my|our) (end |final )?(reply|response|answer)|help us with \d|what should i (reply|say|answer|write|send)|(write|draft|give me) (a|the|my)? ?(reply|response|message|answer)|"
                 r"how (should|do) i (reply|respond)|reply (to|for) (him|her|them))\b")),
    ("trace", _R(r"\b(trace|end to end|end-to-end|walk me through|how does .{0,50} (work|flow))\b")),
    ("explain", _R(r"\b(what does|why does|why is|explain|what is the purpose|how come)\b")),
)
REVISE = _R(r"^\s*(?:(?:ok|okay|so|well|and|alright),?\s+)?(make (it|the reply|this|that)|shorter|smaller|rephrase|reword|simplify|again|"
            r"are you (sure|100|certain)|so (then )?what (exactly )?should|(ok|okay|so),? (then|and)|"
            r"or (we|i) could|(well|but) (he|she|they|we|i) (can|could)|you say)")
LOCATE = _R(r"\b(where|which files?|in what files?|find)\b")


def rule_intent(question: str, has_history: bool, *, regions: tuple[str, ...] = ()) -> str | None:
    """High-precision rules; None when the question needs a judgement."""
    q = question.strip()
    scopes = "|".join(re.escape(r) for r in (*regions, "prod", "production", "dev", "staging"))
    if re.search(rf"\b(on|in) ({scopes})\b", q, re.I):
        return "live"  # a configured region or environment is a place things run
    if re.fullmatch(r"(thanks|thank you|thx|cool|great|nice|perfect|ok|okay)[\s!.]*", q, re.I):
        return "chat"
    if has_history and len(q) < 220 and REVISE.search(q):
        return "revise"
    if re.search(r"\bwhat (exactly )?is there (for (us|me) )?to ?do\b", q, re.I):
        return "scope"
    for intent, rx in RULES:
        if rx.search(q):
            if intent == "gap" and re.search(r"\bwhere\b", q, re.I):
                continue  # "where is X implemented" assumes it is; that is locate or trace
            return intent
    # An identifier says where to look, not what is being asked; only a where-word makes it locate.
    if LOCATE.search(q):
        return "locate"
    return None


async def classify(turn: Turn, runner: Any) -> None:
    """Rules first, the decision model for the rest, chat when neither is sure."""
    regions = tuple(getattr(getattr(runner.settings, "kubernetes", None), "regions", None) or ())
    hit = rule_intent(turn.question, bool(turn.history), regions=regions)
    if hit:
        turn.intent, turn.intent_source = hit, "rule"
        return
    from mimir.decide import Choice, build_decider, decide_async
    from mimir.decide.local import LocalDecider

    decider = build_decider(runner.settings, router=runner.router)
    if not getattr(decider, "available", False):
        decider = LocalDecider(runner.router)
    choice = Choice("intent", INTENTS, "What the operator wants: " + "; ".join(f"{k}: {v}" for k, v in INTENT_HELP.items()))
    previous = next((t for r, t in reversed(turn.history) if r == "assistant"), "")
    context = (f"Previous answer (start): {previous[:400]}\n\n" if previous else "") + f"Operator: {turn.question[:2000]}"
    verdicts = await decide_async(decider, context, [choice])
    v = verdicts.get("intent")
    if v and (not v.calibrated or v.margin >= 0.15):
        turn.intent, turn.intent_source = v.choice, getattr(decider, "name", "decider")


# --- 2-3. plan and gather -----------------------------------------------------

async def _tool(runner: Any, name: str, args: dict[str, Any]) -> Any:
    spec = runner.registry.get(name)
    return await spec.invoke(args, runner.tool_context(None)) if spec is not None else None


async def gather(turn: Turn, runner: Any, settings: Any) -> AsyncIterator[Part]:
    from mimir.api.agent_mode import pick_repo, repo_words, search_phrase
    from mimir.tools.trace import answer_markdown, code_tokens

    turn.repo = pick_repo(turn.question, runner)
    turn.tokens = code_tokens(turn.question)
    phrase = search_phrase(turn.question, exclude=repo_words(turn.repo, runner))

    if turn.intent in ("locate", "explain") and turn.tokens:
        for token in turn.tokens[:2]:
            yield Part("progress", f"Following `{token}` through the code")
            res = await _tool(runner, "trace_symbol", {"symbol": token, "repo": turn.repo})
            turn.steps += 1
            if res is not None and res.ok:
                turn.evidence.append(res.summary)
                return
    if turn.intent == "trace" or (turn.intent in ("locate", "explain") and phrase):
        yield Part("progress", f"Tracing `{phrase}` from its routes to its workers")
        res = await _tool(runner, "trace_feature", {"feature": phrase, "repo": turn.repo})
        turn.steps += 1
        if res is not None and res.ok and (res.data["routes"] or res.data["states"]):
            turn.evidence.append(answer_markdown(res.data))
            return
    if turn.intent in ("locate", "explain", "trace", "gap", "scope"):
        names = _gap_names(turn) if turn.intent in ("gap", "scope") else [phrase]
        for name in [n for n in names if n][:6]:
            yield Part("progress", f"Searching the code for `{name}`")
            from mimir.mcp.server import search_code_impl

            found = await search_code_impl(name, turn.repo, limit=20)
            turn.steps += 1
            hits = found.get("matches", [])[:12]
            if hits:
                turn.evidence.append(f"`{name}` appears in:\n" + "\n".join(
                    f"- {m.get('path') or m.get('file')}:{m.get('line')}: {str(m.get('text') or '')[:150]}" for m in hits))
            else:
                turn.evidence.append(f"`{name}`: no match in the repository code (searched: {', '.join(found.get('tried', [])[:4])}).")
    if turn.intent == "scope":
        from mimir.tools.repo import get_repository_directory
        from mimir.tools.trace import repo_shape, shape_markdown

        yield Part("progress", "Reading how this repository is laid out")
        try:
            root = get_repository_directory(runner.settings).resolve(turn.repo).root
            nouns = [w for w in re.findall(r"[A-Za-z]{4,}", phrase)][:4]
            turn.evidence.append("Repository shape:\n" + shape_markdown(repo_shape(root, nouns)))
            turn.steps += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("repo_shape_failed", error=str(exc))
    if turn.intent in ("gap", "scope") and settings.api.facade_web and runner.registry.get("web_search") is not None:
        query = " ".join(_gap_names(turn)[:2]) + " documentation"
        yield Part("progress", f"Checking external documentation for `{query.strip()}`")
        res = await _tool(runner, "web_search", {"query": query, "max_results": 4})
        turn.steps += 1
        if res is not None and res.ok:
            turn.evidence.append("Documentation:\n" + res.summary[:1500])
    if turn.intent == "live":
        async for part in _live(turn, runner, settings):
            yield part


def _gap_names(turn: Turn) -> list[str]:
    """What to look for: identifiers the operator typed, API-shaped words, then the phrase."""
    from mimir.api.agent_mode import repo_words, search_phrase

    api = re.findall(r"\b[a-z]+(?:_[a-z]+){1,}\b", turn.question)
    phrase = search_phrase(turn.question, exclude=())
    return list(dict.fromkeys([*turn.tokens, *api, phrase]))


async def _live(turn: Turn, runner: Any, settings: Any) -> AsyncIterator[Part]:
    """Cluster questions: the read-only ops loop, its trail as progress, its result as evidence."""
    from mimir.agent.events import AgentEventType
    from mimir.agent.ops import OpsAgent
    from mimir.api.agent_mode import CLUSTER_TOOLS

    agent = OpsAgent(
        router=runner.router, registry=runner.registry, tool_context=runner.tool_context(None),
        settings=settings, environment=None,
        tools=tuple(t for t in CLUSTER_TOOLS if runner.registry.get(t) is not None),
        task_class="fast_command", max_steps=settings.api.facade_agent_max_steps,
    )
    async for event in agent.run(turn.question):
        if event.type is AgentEventType.TOOL_START:
            yield Part("progress", f"{_label(event.tool)}")
        elif event.type is AgentEventType.TOOL_END and event.result is not None:
            turn.steps += 1
            head = (event.result.summary or "").strip().splitlines()
            turn.evidence.append(f"{event.tool}: {head[0] if head else ''}\n{(event.result.summary or '')[:1200]}")


_LABELS = {"find_workloads": "Looking for the workload across clusters", "list_workloads": "Listing workloads",
           "summarise_pod_health": "Checking pod health", "get_logs": "Reading logs", "get_events": "Reading events",
           "describe_resource": "Describing the resource", "get_current_context": "Checking the kube context",
           "get_resource_usage": "Checking resource usage", "get_rollout_status": "Checking the rollout"}


def _label(tool: str) -> str:
    return _LABELS.get(tool, tool.replace("_", " ").capitalize())


# --- 4-5. answer and check ----------------------------------------------------

SYSTEM = """\
You are MIMIR, a senior engineer's assistant for the operator's own services. Answer the
operator's latest message in the context of the whole conversation.

Rules:
- Ground every claim about the code or the clusters in the EVIDENCE below. Cite file:line
  exactly as it appears there. Never invent a file, function, route or number.
- If the evidence does not settle something, say so in one line instead of guessing.
- Match the size the conversation asks for. A request to shorten means shorten.
- When asked to draft a reply, write only the reply, ready to paste.
- Plain language. No preamble, no restating the question.
"""


def conversation(messages: list[Any], *, budget: int = 14000) -> tuple[str, list[tuple[str, str]]]:
    """The latest user message in full, and earlier turns newest first until the budget runs out."""
    turns = [(getattr(m, "role", ""), (getattr(m, "content", "") or "").strip()) for m in messages]
    turns = [(r, t) for r, t in turns if r in ("user", "assistant") and t]
    if not turns or turns[-1][0] != "user":
        return "", []
    question, earlier, used = turns[-1][1], [], len(turns[-1][1])
    for role, text in reversed(turns[:-1]):
        text = text if role == "user" else _strip_trail(text)[:3000]
        if used + len(text) > budget:
            break
        earlier.insert(0, (role, text))
        used += len(text)
    return question, earlier


def _strip_trail(text: str) -> str:
    """An earlier MIMIR answer, without its progress header."""
    return re.sub(r"^\*Worked for [^\n]*\n+", "", text)


async def answer(turn: Turn, runner: Any) -> str:
    from mimir.llm.base import GenerationOptions, LLMMessage, ModelError

    msgs = [LLMMessage.system(SYSTEM)]
    for role, text in turn.history:
        msgs.append(LLMMessage.user(text) if role == "user" else LLMMessage.assistant(text))
    evidence = "\n\n".join(turn.evidence) or "(no tools were needed for this message)"
    msgs.append(LLMMessage.user(f"{turn.question}\n\nEVIDENCE (kind of question: {turn.intent}):\n{evidence[:12000]}"))
    try:
        response = await runner.router.chat(msgs, task_class="fast_command",
                                            options=GenerationOptions(tools=[], temperature=0.0, max_tokens=1400),
                                            purpose=f"facade:{turn.intent}")
    except ModelError as exc:
        return f"(the model failed: {exc.message})"
    return (response.content or "").strip()


def check(turn: Turn, text: str) -> str:
    """Name anything the answer cites that the evidence never showed."""
    if turn.intent in ("draft", "revise", "chat") or not turn.evidence:
        return text
    from mimir.verify.grounding import check as grounding

    seen = "\n".join(turn.evidence) + "\n" + "\n".join(t for _, t in turn.history)
    g = grounding(text, seen, asked=turn.question)
    if g.ok:
        return text
    return text + "\n\n_Not found in the evidence, verify before relying on: " + ", ".join(f"`{u}`" for u in g.ungrounded[:6]) + "_"


async def respond(messages: list[Any], runner: Any, settings: Any) -> AsyncIterator[Part]:
    started = time.time()
    question, history = conversation(messages)
    if not question:
        yield Part("answer", "No question found in the conversation.")
        return
    turn = Turn(question=question, history=history)
    await classify(turn, runner)
    yield Part("progress", f"Understood as: {INTENT_HELP[turn.intent]}")
    try:
        async for part in gather(turn, runner, settings):
            yield part
    except Exception as exc:  # noqa: BLE001 - an answer without evidence beats no answer
        log.exception("assistant_gather_failed")
        turn.evidence.append(f"(gathering failed: {type(exc).__name__}: {exc})")
    deterministic = turn.intent in ("locate", "trace") and turn.evidence and turn.evidence[0].startswith(("**", "`"))
    if deterministic:
        yield Part("progress", "Writing a short overview")
        overview = await _overview(turn, runner)
        text = (overview + "\n\n" if overview else "") + "\n\n".join(turn.evidence)
    else:
        yield Part("progress", "Writing the answer")
        text = check(turn, await answer(turn, runner))
    elapsed = time.time() - started
    yield Part("answer", f"*Worked for {elapsed:.0f}s*\n\n{text}")


async def _overview(turn: Turn, runner: Any) -> str:
    from mimir.api.agent_mode import _overview as overview

    return await overview(runner, turn.question, "\n\n".join(turn.evidence))


__all__ = ["INTENTS", "Part", "Turn", "classify", "conversation", "respond", "rule_intent"]
