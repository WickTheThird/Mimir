"""Measurements of the runtime, not of MIMIR.

Three things were asserted during a long session and none of them was measured
properly: that MIMIR pays a full prefill on every step of a loop, that tool call
adherence collapses past a schema size, and that sampling more than once would
help. The first two were assumed from a single observation at temperature zero,
which is exactly the standard this project refuses to accept from anyone else.

So they are probes, replicated, and they report what they did not control. Each
one answers a question that changes what to build next, and each is cheap
enough that there is no excuse for having guessed.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = 600.0


@dataclass
class Sample:
    label: str
    prompt_tokens_sent: int
    prompt_tokens_evaluated: int
    prompt_eval_s: float
    output_tokens: int
    total_s: float
    tool_calls: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cached_fraction(self) -> float:
        """How much of the prompt the runtime did not have to evaluate."""
        if self.prompt_tokens_sent <= 0:
            return 0.0
        skipped = max(0, self.prompt_tokens_sent - self.prompt_tokens_evaluated)
        return round(skipped / self.prompt_tokens_sent, 3)


@dataclass
class ProbeResult:
    name: str
    alias: str
    model: str
    replicates: int
    samples: list[Sample] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    uncontrolled: list[str] = field(default_factory=list)
    """What could have moved the numbers and was not held fixed."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe": self.name,
            "alias": self.alias,
            "model": self.model,
            "replicates": self.replicates,
            "samples": [vars(s) for s in self.samples],
            "notes": self.notes,
            "uncontrolled": self.uncontrolled,
        }


def _profile(settings: Settings, alias: str):
    profile = settings.models.profiles.get(alias)
    if profile is None:
        raise KeyError(f"unknown model alias {alias!r}")
    return profile


def _base(profile) -> str:
    return str(profile.base_url).rstrip("/").removesuffix("/v1")


def _chat(
    profile,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
    num_predict: int = 8,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """One non-streaming call, through Ollama's native API for its timings.

    The OpenAI-compatible surface does not report how much of the prompt was
    actually evaluated, which is the number the caching probe exists to read.
    """
    payload: dict[str, Any] = {
        "model": profile.model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    if tools:
        payload["tools"] = tools
    if schema is not None:
        payload["format"] = schema
    started = time.time()
    response = httpx.post(f"{_base(profile)}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    body["_wall_s"] = time.time() - started
    return body


def _sample(label: str, body: dict[str, Any], sent: int) -> Sample:
    message = body.get("message") or {}
    return Sample(
        label=label,
        prompt_tokens_sent=sent,
        prompt_tokens_evaluated=int(body.get("prompt_eval_count") or 0),
        prompt_eval_s=round((body.get("prompt_eval_duration") or 0) / 1e9, 3),
        output_tokens=int(body.get("eval_count") or 0),
        total_s=round(body.get("_wall_s", 0.0), 3),
        tool_calls=len(message.get("tool_calls") or []),
    )


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Four characters to a token, which is close enough to read a ratio."""
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


# ---------------------------------------------------------------------------
# 1. Does a growing conversation re-evaluate its whole prefix
# ---------------------------------------------------------------------------


def prefix_cache(
    alias: str = "deep",
    *,
    steps: int = 4,
    replicates: int = 3,
    settings: Settings | None = None,
) -> ProbeResult:
    """Whether each step of a loop pays for the whole prompt or only the delta.

    Every cost estimate in this project assumes the former. If the runtime
    reuses the key-value cache across a shared prefix, a twelve step loop costs
    one prefill and eleven small ones, and the case for shortening loops is
    much weaker than the case for keeping them on one conversation.

    Read ``prompt_tokens_evaluated`` against ``prompt_tokens_sent``. A runtime
    with no prefix cache evaluates everything, every time.
    """
    settings = settings or get_settings()
    profile = _profile(settings, alias)
    result = ProbeResult(
        name="prefix_cache", alias=alias, model=profile.model, replicates=replicates
    )
    result.uncontrolled = [
        "other processes using the same runtime",
        "whether the model was already resident",
    ]

    filler = (
        "The queue consumer catches TransientError and re-raises immediately. "
        "Retry policy is referenced in three places. "
    ) * 60

    for replicate in range(replicates):
        # A unique prefix per replicate, or the second replicate reads the
        # first one's cache and every step looks warm. The first measurement of
        # this reported a cold prefill of 0.01s for exactly that reason.
        nonce = f"session {time.time_ns()}-{replicate}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"You are a terse assistant. {nonce}. {filler}"},
            {"role": "user", "content": "Reply with the single word ready."},
        ]
        for step in range(1, steps + 1):
            sent = _estimate_tokens(messages)
            body = _chat(profile, messages)
            sample = _sample(f"r{replicate}-step{step}", body, sent)
            sample.extra["step"] = step
            sample.extra["replicate"] = replicate
            result.samples.append(sample)
            # Grow the conversation the way a tool loop does: append the
            # assistant turn and a short result, keeping the whole prefix.
            messages.append({"role": "assistant",
                             "content": (body.get("message") or {}).get("content", "ok")})
            messages.append({"role": "user", "content": f"Step {step} done. Continue."})

    # Time, not the token count. prompt_eval_count reports the size of the
    # prompt rather than how much of it was computed, so it stays flat while a
    # cache is doing all the work. The duration is the only honest signal here.
    by_step: dict[int, list[Sample]] = {}
    for sample in result.samples:
        by_step.setdefault(int(sample.extra["step"]), []).append(sample)
    for step in sorted(by_step):
        samples = by_step[step]
        seconds = statistics.median(s.prompt_eval_s for s in samples)
        tokens = statistics.median(s.prompt_tokens_evaluated for s in samples)
        result.notes.append(
            f"step {step}: {seconds:.2f}s prefill for {tokens:.0f} prompt tokens "
            f"(median of {len(samples)})"
        )

    cold = [s.prompt_eval_s for s in result.samples if int(s.extra["step"]) == 1]
    warm = [s.prompt_eval_s for s in result.samples if int(s.extra["step"]) > 1]
    if cold and warm:
        first, rest = statistics.median(cold), statistics.median(warm)
        ratio = first / rest if rest > 0 else float("inf")
        result.notes.append(
            f"first step {first:.2f}s, later steps {rest:.2f}s: {ratio:.0f}x cheaper"
        )
        result.notes.append(
            "the prefix is cached, so a long loop costs one prefill and cheap "
            "deltas. Shorten the first prompt, not the loop."
            if ratio >= 3
            else "every step pays a full prefill; shorter loops matter."
        )
    return result


# ---------------------------------------------------------------------------
# 2. Does tool calling collapse past a schema size
# ---------------------------------------------------------------------------

_PROBE_PROMPTS = (
    "get the last 10 log lines from the api pods in namespace payments",
    "list the workloads in namespace payments on the prod cluster",
    "show the warning events in namespace checkout",
    "describe the api deployment in namespace payments",
    "how many pods restarted in namespace payments in the last hour",
    "tail 50 lines from the worker pods in namespace billing",
    "which pods in namespace search are not ready",
    "find any pod with outbound in its name in a cluster with ch1",
    "show me the events for the checkout deployment in namespace orders",
    "what is the current kubectl context",
    "read the logs of deployment/gateway in namespace edge since 30m",
    "list every workload in namespace platform whose name contains cache",
    "describe pod api-7c9f4 in namespace payments",
    "are there crashlooping pods in namespace ingest",
    "get the warning events from the last hour in namespace payments",
)
"""Fifteen distinct prompts, because replicates do not add samples here.

At temperature zero a replicate is the same call twice and returns the same
answer, so the first version of this probe reported fifteen samples per size
and had five. Every count it produced was an exact multiple of three, which is
what gave it away."""


def _sized_schemas(base: list[dict[str, Any]], target_chars: int) -> list[dict[str, Any]]:
    """A tool surface at roughly a given size.

    Grows by cloning real tools rather than adding junk, since the question is
    whether volume alone breaks tool calling and the padding has to resemble
    what a real surface contains. Shrinks by dropping tools from the end, which
    the first version could not do: it only padded, so every target below the
    base size produced the same surface and two rows of the table were the same
    experiment reported twice.
    """
    schemas = [json.loads(json.dumps(s)) for s in base]
    if len(json.dumps(schemas)) > target_chars:
        while len(schemas) > 2 and len(json.dumps(schemas)) > target_chars:
            schemas.pop()
        return schemas
    index = 0
    while len(json.dumps(schemas)) < target_chars and index <= 200:
        clone = json.loads(json.dumps(base[index % len(base)]))
        clone["function"]["name"] = f"{clone['function']['name']}_variant_{index}"
        schemas.append(clone)
        index += 1
    return schemas


def tool_adherence(
    alias: str = "deep",
    *,
    sizes: tuple[int, ...] = (4_000, 7_000, 10_000, 13_000, 16_000, 20_000),
    replicates: int = 1,
    constrained: bool = False,
    settings: Settings | None = None,
) -> ProbeResult:
    """Fraction of prompts that produce a tool call, against schema volume.

    The cliff this was built to confirm was found once, on one prompt, at
    temperature zero, and then designed around. It does not survive: adherence
    degrades from the smallest surface upward rather than falling off an edge,
    and it is already imperfect at four tools.

    Replicates default to one because production runs at temperature zero,
    where a replicate is the same call twice. Samples come from distinct
    prompts instead.
    """
    from mimir.agent.ops import OPS_TOOLS, SYSTEM
    from mimir.tools.base import load_all_tools

    settings = settings or get_settings()
    profile = _profile(settings, alias)
    registry = load_all_tools()
    base_specs = [s for s in (registry.get(name) for name in OPS_TOOLS) if s is not None]
    base = [spec.openai_schema() for spec in base_specs]

    result = ProbeResult(
        name="tool_adherence", alias=alias, model=profile.model, replicates=replicates
    )
    constrained_schema = None
    if constrained:
        from mimir.agent.constrained import build_schema, parse_step

        result.notes.append("decoder: constrained against a schema")
    else:
        result.notes.append("decoder: the runtime's native tool-call channel")
    result.uncontrolled = [
        "the wording of the probe prompts",
        "padding tools are near-duplicates of real ones",
    ]

    # The operations prompt, because these are operations tools and operations
    # prompts. The first version paired the coding system prompt with cluster
    # tools and cluster questions, and measured that mismatch instead: tool
    # calling sat flat at 20% across every size, which is not a cliff, it is a
    # model being told it is editing a repository and then asked about pods.
    system = SYSTEM.format(context="No cluster context is set; use the one named.")
    for size in sizes:
        schemas = _sized_schemas(base, size)
        actual = len(json.dumps(schemas))
        called = 0
        total = 0
        if constrained:
            specs_here = [
                spec for spec in base_specs
                if any(s["function"]["name"] == spec.name for s in schemas)
            ]
            constrained_schema = build_schema(specs_here or base_specs)
        for replicate in range(max(1, replicates)):
            for prompt in _PROBE_PROMPTS:
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}]
                if constrained:
                    body = _chat(profile, messages, schema=constrained_schema,
                                 num_predict=300)
                    step = parse_step((body.get("message") or {}).get("content", ""))
                    body["message"] = {
                        "tool_calls": [] if step.finished else [{"function": {}}]
                    }
                else:
                    body = _chat(profile, messages, tools=schemas, num_predict=64)
                sample = _sample(f"{actual}c-r{replicate}", body, _estimate_tokens([]))
                sample.extra.update({"schema_chars": actual, "tools": len(schemas),
                                     "prompt": prompt})
                result.samples.append(sample)
                called += 1 if sample.tool_calls else 0
                total += 1
        result.notes.append(
            f"{actual:,} chars / {len(schemas)} tools: "
            f"{called}/{total} prompts produced a tool call ({called / total:.0%})"
        )

    by_size = {}
    for sample in result.samples:
        size = int(sample.extra["schema_chars"])
        hit, seen = by_size.get(size, (0, 0))
        by_size[size] = (hit + (1 if sample.tool_calls else 0), seen + 1)
    ordered = sorted(by_size)
    if len(ordered) >= 2:
        first = by_size[ordered[0]][0] / by_size[ordered[0]][1]
        last = by_size[ordered[-1]][0] / by_size[ordered[-1]][1]
        summary = f"smallest surface {first:.0%}, largest {last:.0%}: "
        if first >= 0.99 and last >= 0.99:
            summary += (
                "volume does not affect adherence, because the decoder cannot "
                "emit anything else"
            )
        elif last < first:
            summary += (
                "adherence degrades with volume and is already imperfect at the "
                "smallest size, so this is a slope rather than a cliff and no "
                "tool count makes it reliable"
            )
        else:
            summary += "no clear relationship at this sample size"
        result.notes.append(summary)
    return result


# ---------------------------------------------------------------------------
# 3. Would sampling more than once help, and by how much
# ---------------------------------------------------------------------------


def sampling_headroom(
    *,
    suite_size: int = 52,
    runs: int = 12,
    settings: Settings | None = None,
) -> ProbeResult:
    """What best-of-k could reach, from replicates already on disk.

    No model is called. The arithmetic for best-of-k assumes attempts are
    independent, and that assumption is the whole question: if a case fails for
    a structural reason it fails every time, and sampling it five times buys
    nothing. Twelve stored runs of the same suite answer that directly.

    pass@k here is the ceiling a perfect selector would reach, not a promise. A
    real selector is the deterministic gate, and it reaches this only if it
    never accepts a wrong answer.
    """
    import sqlite3

    settings = settings or get_settings()
    database = settings.home / "mimir.db"
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row

    identifiers = [
        row["id"]
        for row in connection.execute(
            "SELECT id FROM eval_runs WHERE total = ? ORDER BY created_at DESC LIMIT ?",
            (suite_size, runs),
        )
    ]
    outcomes: dict[str, dict[str, bool]] = {}
    for identifier in identifiers:
        for row in connection.execute(
            "SELECT case_id, passed FROM eval_results WHERE run_id = ?", (identifier,)
        ):
            outcomes.setdefault(row["case_id"], {})[identifier] = bool(row["passed"])

    complete = {
        case: [seen[i] for i in identifiers if i in seen]
        for case, seen in outcomes.items()
    }
    complete = {c: v for c, v in complete.items() if len(v) == len(identifiers)}

    result = ProbeResult(
        name="sampling_headroom", alias="", model="from stored runs",
        replicates=len(identifiers),
    )
    result.uncontrolled = [
        "runs span weeks of code changes, so attempts are not exchangeable",
        "pass@k assumes independence between attempts",
    ]
    if not complete:
        result.notes.append("no suite has enough complete runs to measure")
        return result

    total = len(complete)
    always = sum(1 for v in complete.values() if all(v))
    never = sum(1 for v in complete.values() if not any(v))
    result.notes.append(
        f"{total} cases across {len(identifiers)} runs: {always} always pass, "
        f"{never} never pass, {total - always - never} are flaky"
    )
    result.notes.append(
        f"{never} case(s) fail structurally; sampling cannot reach those"
    )
    for k in (1, 2, 3, 5):
        rate = sum(
            1 - (1 - sum(v) / len(v)) ** k for v in complete.values()
        ) / total
        result.notes.append(f"pass@{k} = {rate:.3f} ({rate * total:.1f}/{total})")
    return result


PROBES = {
    "prefix_cache": prefix_cache,
    "sampling_headroom": sampling_headroom,
    "tool_adherence": tool_adherence,
}

__all__ = ["PROBES", "ProbeResult", "Sample", "prefix_cache", "tool_adherence"]
