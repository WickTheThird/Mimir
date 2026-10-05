"""Deterministic command construction: the parser and the risk classifier, no model."""

from __future__ import annotations

from typing import Any

from mimir.agent.request import KIND_NOUNS, parse_request
from mimir.models.command import CommandKind, ProposedCommand, TargetContext

_KIND = {
    "pod": "pod", "pods": "pod", "deployment": "deployment", "deployments": "deployment",
    "statefulset": "statefulset", "statefulsets": "statefulset", "daemonset": "daemonset",
    "daemonsets": "daemonset", "service": "service", "services": "service",
    "job": "job", "jobs": "job", "cronjob": "cronjob", "cronjobs": "cronjob",
}


def _kind(text: str) -> str:
    lowered = text.lower()
    for noun in KIND_NOUNS:
        if f" {noun} " in f" {lowered} " or f" {noun}/" in f" {lowered} ":
            return _KIND.get(noun, "")
    return ""


def _target(kind: str, name: str) -> str:
    return f"{kind}/{name}" if kind and kind != "pod" else name


def construct_fast(
    request: str, *, context: str | None = None, namespace: str | None = None,
    entities: Any = None, kubectl: str = "kubectl",
) -> list[ProposedCommand] | None:
    """A kubectl proposal from the stated request, or None when something is unstated."""
    parsed = parse_request(request)
    ns = namespace or parsed.namespace
    ctx = context or parsed.context
    name = parsed.name_contains
    kind = _kind(request) or ("pod" if parsed.action in ("logs", "usage") else "")
    if not parsed.action or not name:
        return None
    if not ns and entities is not None:
        scopes = sorted({(e.context, e.namespace) for e in entities.candidates(name) if e.namespace})
        if len(scopes) == 1:
            ctx, ns = ctx or scopes[0][0], scopes[0][1]
    if not ns:
        return None

    base = [kubectl] + (["--context", ctx] if ctx else []) + ["-n", ns]
    target = _target(kind or "pod", name)
    if parsed.action == "logs":
        args = ["logs", target, "--tail", str(parsed.tail or 100)]
        if parsed.since:
            args += ["--since", parsed.since]
        if kind and kind != "pod":
            args += ["--all-containers=true"]
        purpose, effect = f"last {parsed.tail or 100} log lines of {target}", "reads logs"
    elif parsed.action == "events":
        args = ["get", "events", "--field-selector", f"involvedObject.name={name}",
                "--sort-by=.lastTimestamp"]
        purpose, effect = f"recent events for {name}", "reads events"
    elif parsed.action == "describe":
        args = ["describe", target]
        purpose, effect = f"describe {target}", "reads the object"
    elif parsed.action == "status":
        args = ["rollout", "status", target] if kind and kind != "pod" else ["get", "pods", "-l", f"app={name}", "-o", "wide"]
        purpose, effect = f"status of {target}", "reads status"
    elif parsed.action == "usage":
        args = ["top", "pods", "-l", f"app={name}"]
        purpose, effect = f"resource usage of {name}", "reads metrics"
    elif parsed.action == "restarts":
        args = ["get", "pods", "-l", f"app={name}", "-o",
                "custom-columns=NAME:.metadata.name,RESTARTS:.status.containerStatuses[*].restartCount,REASON:.status.containerStatuses[*].lastState.terminated.reason"]
        purpose, effect = f"restart counts for {name}", "reads pod status"
    else:
        return None
    return [ProposedCommand(
        kind=CommandKind.KUBECTL, argv=base + args, purpose=purpose, expected_effect=effect,
        proposed_by="mimir:parser", tool_name="construct_fast",
        context=TargetContext(cluster_context=ctx or None, namespace=ns, targets=[target]),
    )]


__all__ = ["construct_fast"]
