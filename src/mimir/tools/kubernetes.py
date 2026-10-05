"""Kubernetes helpers (ADR 9.2, 5.5, 13)."""

from __future__ import annotations

import json
import re
import shlex
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mimir.logging import get_logger
from mimir.models.command import (
    CommandKind,
    CommandOutcome,
    ExecutionRecord,
    ProposedCommand,
    RiskClass,
    TargetContext,
)
from mimir.models.evidence import Evidence
from mimir.tools.artifacts import ArtifactStore, get_artifact_store
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool
from mimir.tools.exec import CommandExecutor, ExecutionOptions, get_executor

log = get_logger(__name__)

#: ADR 5.1 asks for "pods that restarted in the last hour" as a first-class
RESTART_WINDOW_S = 3600.0

# : Container waiting/terminated reasons that mean the pod is not healthy.
UNHEALTHY_WAITING = frozenset(
    {
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "ErrImagePull",
        "CreateContainerConfigError",
        "CreateContainerError",
        "InvalidImageName",
        "RunContainerError",
        "ContainerCreating",
    }
)
UNHEALTHY_TERMINATED = frozenset(
    {"OOMKilled", "Error", "ContainerCannotRun", "DeadlineExceeded", "Evicted"}
)

#: Percent-of-limit thresholds above which ADR G4 wants throttling and memory
CPU_PRESSURE_PCT = 80.0
MEMORY_PRESSURE_PCT = 85.0

DEFAULT_WORKLOAD_KINDS = ("deployment", "statefulset", "daemonset")


# ---------------------------------------------------------------------------

_UNSAFE_TOKEN = re.compile(r"\s|^-")


def _safe_token(value: str, field_name: str) -> str:
    """Reject values that would be read as a flag or split into extra argv items."""
    token = value.strip()
    if not token or _UNSAFE_TOKEN.search(token):
        raise ToolError(
            f"invalid {field_name}: {value!r} must be a single token and may not start with '-'",
            code="invalid_arguments",
        )
    return token


# ---------------------------------------------------------------------------


def _executor(ctx: ToolContext) -> CommandExecutor:
    executor = ctx.executor
    if executor is None:
        executor = get_executor(ctx.settings)
    return executor  # type: ignore[return-value]


def _artifacts(ctx: ToolContext) -> ArtifactStore:
    store = ctx.artifacts
    if store is None:
        store = get_artifact_store(ctx.settings)
    return store  # type: ignore[return-value]


def _kubectl(ctx: ToolContext) -> str:
    if not ctx.settings.kubernetes.enabled:
        raise ToolError("kubernetes helpers are disabled by configuration", code="disabled")
    return ctx.settings.kubernetes.kubectl_path


def _exec_options(ctx: ToolContext, timeout_s: float | None = None) -> ExecutionOptions:
    return ExecutionOptions(timeout_s=timeout_s or ctx.settings.kubernetes.command_timeout_s)


# ---------------------------------------------------------------------------


def _build(
    ctx: ToolContext,
    *,
    args: list[str],
    purpose: str,
    context: str | None,
    namespace: str | None = None,
    targets: list[str] | None = None,
    pod: str | None = None,
    container: str | None = None,
    expected_effect: str = "",
    tool_name: str = "kubernetes",
    timeout_s: float | None = None,
) -> ProposedCommand:
    """Build a kubectl proposal with the target context fully resolved."""
    argv = [_kubectl(ctx)]
    if context:
        argv += ["--context", context]
    if namespace:
        argv += ["-n", namespace]
    argv += args
    return ProposedCommand(
        kind=CommandKind.KUBECTL,
        argv=argv,
        purpose=purpose,
        expected_effect=expected_effect,
        timeout_s=timeout_s or ctx.settings.kubernetes.command_timeout_s,
        tool_name=tool_name,
        proposed_by=str(ctx.specialist) if ctx.specialist else "mimir",
        context=TargetContext(
            cluster_context=context,
            namespace=namespace,
            pod=pod,
            container=container,
            targets=list(targets or []),
        ),
    )


# ---------------------------------------------------------------------------

_FAILURE_RULES: tuple[tuple[re.Pattern[str], str, str, bool], ...] = (
    (
        re.compile(r"context .*(does not exist|was not found)|no context exists", re.I),
        "unknown_context",
        "the requested cluster context does not exist in the kubeconfig",
        False,
    ),
    (
        re.compile(
            r"unable to connect to the server|connection refused|i/o timeout|no route", re.I
        ),
        "cluster_unreachable",
        "the cluster API server could not be reached (VPN down, or wrong context)",
        True,
    ),
    (
        re.compile(
            r"you must be logged in|unauthorized|invalid bearer token|token has expired", re.I
        ),
        "unauthenticated",
        "kubectl is not authenticated against this cluster; refresh the credential and retry",
        False,
    ),
    (
        re.compile(r"forbidden|cannot (list|get|watch|create|delete)", re.I),
        "forbidden",
        "the current identity is not permitted to perform this read",
        False,
    ),
    (
        re.compile(
            r"metrics api not available|metrics\.k8s\.io|server could not find the requested",
            re.I,
        ),
        "metrics_unavailable",
        "the metrics API is unavailable; metrics-server is probably not installed",
        False,
    ),
    (
        re.compile(r"not found|doesn't have a resource type|no resources found", re.I),
        "not_found",
        "the requested object does not exist in this namespace",
        False,
    ),
    (
        re.compile(r"x509|certificate signed by unknown authority", re.I),
        "tls_error",
        "the API server certificate could not be verified",
        False,
    ),
)


def _explain(record: ExecutionRecord) -> tuple[str, str, bool]:
    """Map a failed record onto (code, message, retryable)."""
    if record.outcome == CommandOutcome.DENIED:
        return "policy_denied", record.error or "denied by policy", False
    if record.outcome == CommandOutcome.REJECTED:
        return "approval_rejected", record.error or "the operator rejected this command", False
    if record.outcome == CommandOutcome.SKIPPED:
        return "approval_not_granted", record.error or "the command was not executed", False
    if record.outcome == CommandOutcome.TIMEOUT:
        return "timeout", record.error or "kubectl timed out", True
    if record.outcome == CommandOutcome.ERROR and "binary not found" in (record.error or ""):
        return (
            "kubectl_missing",
            "kubectl was not found on PATH; set kubernetes.kubectl_path in the MIMIR config",
            False,
        )
    haystack = f"{record.stderr}\n{record.error or ''}"
    for pattern, code, message, retryable in _FAILURE_RULES:
        if pattern.search(haystack):
            return code, message, retryable
    return "kubectl_failed", record.error or "kubectl exited non-zero", False


def _require_ok(record: ExecutionRecord, what: str) -> ExecutionRecord:
    if record.ok:
        return record
    code, message, retryable = _explain(record)
    detail = record.stderr.strip().splitlines()[:3]
    suffix = f" | {' '.join(detail)}" if detail else ""
    raise ToolError(
        f"{what} failed: {message} [{record.display}]{suffix}",
        code=code,
        retryable=retryable,
    )


def _parse_json(record: ExecutionRecord, what: str) -> dict[str, Any]:
    text = record.stdout.strip()
    if not text:
        raise ToolError(f"{what}: kubectl returned no output", code="empty_output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        hint = " (output was truncated by the output cap)" if record.truncated else ""
        raise ToolError(
            f"{what}: could not parse kubectl JSON{hint}: {exc}", code="unparsable_output"
        ) from exc
    if not isinstance(data, dict):
        raise ToolError(f"{what}: unexpected kubectl JSON shape", code="unparsable_output")
    return data


def _items(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("items"), list):
        return [i for i in data["items"] if isinstance(i, dict)]
    return [data]


def _evidence(
    ctx: ToolContext, record: ExecutionRecord, claim: str, *, tool_name: str
) -> Evidence:
    return _executor(ctx).to_evidence(record, claim, collected_by=tool_name)


# ---------------------------------------------------------------------------

#: kubeconfig-derived answers change rarely and every helper needs them, so a
_PROBE_TTL_S = 60.0
_probe_cache: dict[str, tuple[float, str]] = {}


def _cache_key(ctx: ToolContext, name: str) -> str:
    kube = ctx.settings.kubernetes
    return f"{name}|{kube.kubectl_path}|{kube.kubeconfig or ''}"


def _cache_get(key: str) -> str | None:
    hit = _probe_cache.get(key)
    if hit and time.time() - hit[0] <= _PROBE_TTL_S:
        return hit[1]
    return None


async def _probe_current_context(ctx: ToolContext) -> tuple[str, ExecutionRecord | None]:
    """Ask kubectl which context is active."""
    key = _cache_key(ctx, "current-context")
    cached = _cache_get(key)
    if cached:
        return cached, None
    command = _build(
        ctx,
        args=["config", "current-context"],
        purpose="resolve the active cluster context before naming it explicitly",
        context=None,
        tool_name="get_current_context",
    )
    record = await _executor(ctx).run(
        command, session_id=ctx.session_id, options=_exec_options(ctx)
    )
    _require_ok(record, "resolving the current kube context")
    value = record.stdout.strip().splitlines()[0].strip() if record.stdout.strip() else ""
    if not value:
        raise ToolError(
            "kubectl reported no current context; pass 'context' explicitly or set "
            "kubernetes.default_context",
            code="unknown_context",
        )
    _probe_cache[key] = (time.time(), value)
    return value, record


async def _resolve_context(ctx: ToolContext, requested: str | None) -> str:
    """Explicit argument, then configured default, then session environment, then the kubeconfig's current context."""
    if requested:
        return _safe_token(requested, "context")
    if ctx.settings.kubernetes.default_context:
        return ctx.settings.kubernetes.default_context
    env_context = getattr(ctx.environment, "cluster_context", None)
    if env_context:
        return str(env_context)
    value, _ = await _probe_current_context(ctx)
    return value


async def _probe_namespace(ctx: ToolContext, context: str) -> str:
    """Namespace bound to the context in the kubeconfig, or ``default``."""
    key = _cache_key(ctx, f"namespace|{context}")
    cached = _cache_get(key)
    if cached:
        return cached
    command = _build(
        ctx,
        args=["config", "view", "--minify", "-o", "json"],
        purpose="resolve the namespace bound to this context",
        context=context,
        tool_name="get_current_context",
    )
    record = await _executor(ctx).run(
        command, session_id=ctx.session_id, options=_exec_options(ctx)
    )
    namespace = "default"
    if record.ok:
        try:
            view = _parse_json(record, "reading the kubeconfig")
        except ToolError:
            view = {}
        entries = view.get("contexts") or []
        if entries and isinstance(entries[0], dict):
            namespace = (entries[0].get("context") or {}).get("namespace") or "default"
    _probe_cache[key] = (time.time(), namespace)
    return namespace


async def _resolve_namespace(ctx: ToolContext, requested: str | None, context: str) -> str:
    if requested:
        return _safe_token(requested, "namespace")
    if ctx.settings.kubernetes.default_namespace:
        return ctx.settings.kubernetes.default_namespace
    env_namespace = getattr(ctx.environment, "namespace", None)
    if env_namespace:
        return str(env_namespace)
    return await _probe_namespace(ctx, context)


async def _scope(ctx: ToolContext, args: _KubeArgs) -> tuple[str, str]:
    context = await _resolve_context(ctx, args.context)
    namespace = await _resolve_namespace(ctx, args.namespace, context)
    return context, namespace


# ---------------------------------------------------------------------------


def _parse_ts(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    return f"{total // 86400}d{(total % 86400) // 3600:02d}h"


def _age(obj: dict[str, Any], now: float) -> tuple[float | None, str]:
    created = _parse_ts((obj.get("metadata") or {}).get("creationTimestamp"))
    if created is None:
        return None, "unknown"
    seconds = max(0.0, now - created)
    return seconds, _human_duration(seconds)


_CPU_UNITS = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0}
_MEM_UNITS = {
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "Ki": 1024.0,
    "Mi": 1024.0**2,
    "Gi": 1024.0**3,
    "Ti": 1024.0**4,
    "Pi": 1024.0**5,
    "Ei": 1024.0**6,
}
_QUANTITY = re.compile(r"^(?P<num>[0-9.]+)(?P<unit>[A-Za-z]*)$")


def _parse_cpu(value: Any) -> float | None:
    """Kubernetes CPU quantity to cores."""
    match = _QUANTITY.match(str(value or "").strip())
    if not match:
        return None
    factor = _CPU_UNITS.get(match.group("unit"))
    if factor is None:
        return None
    return float(match.group("num")) * factor


def _parse_memory(value: Any) -> float | None:
    """Kubernetes memory quantity to bytes."""
    match = _QUANTITY.match(str(value or "").strip())
    if not match:
        return None
    unit = match.group("unit")
    factor = _MEM_UNITS.get(unit) or _MEM_UNITS.get(unit.rstrip("B"))
    if factor is None:
        return None
    return float(match.group("num")) * factor


def _fmt_cpu(cores: float | None) -> str | None:
    if cores is None:
        return None
    return f"{cores:.3f}".rstrip("0").rstrip(".") if cores >= 1 else f"{round(cores * 1000)}m"


def _fmt_memory(byte_count: float | None) -> str | None:
    if byte_count is None:
        return None
    for unit, factor in (("Gi", 1024.0**3), ("Mi", 1024.0**2), ("Ki", 1024.0)):
        if byte_count >= factor:
            return f"{byte_count / factor:.1f}{unit}"
    return f"{int(byte_count)}"


def _pct(used: float | None, ceiling: float | None) -> float | None:
    if used is None or not ceiling:
        return None
    return round(used / ceiling * 100.0, 1)


def _container_resources(container: dict[str, Any]) -> dict[str, Any]:
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    return {
        "name": container.get("name"),
        "image": container.get("image"),
        "cpu_request": requests.get("cpu"),
        "cpu_limit": limits.get("cpu"),
        "memory_request": requests.get("memory"),
        "memory_limit": limits.get("memory"),
    }


# ---------------------------------------------------------------------------


def _pod_view(pod: dict[str, Any], now: float) -> dict[str, Any]:
    """Everything the ADR G4 restart diagnosis needs, from one pod object."""
    meta = pod.get("metadata") or {}
    spec = pod.get("spec") or {}
    status = pod.get("status") or {}
    statuses = [c for c in (status.get("containerStatuses") or []) if isinstance(c, dict)]

    restarts = 0
    waiting: list[dict[str, str]] = []
    terminated: list[dict[str, Any]] = []
    last_restart_at: float | None = None
    for container in statuses:
        restarts += int(container.get("restartCount") or 0)
        wait_state = (container.get("state") or {}).get("waiting") or {}
        if wait_state.get("reason"):
            waiting.append(
                {
                    "container": str(container.get("name")),
                    "reason": str(wait_state["reason"]),
                    "message": str(wait_state.get("message", ""))[:200],
                }
            )
        last_term = (container.get("lastState") or {}).get("terminated") or {}
        if last_term.get("reason"):
            terminated.append(
                {
                    "container": str(container.get("name")),
                    "reason": str(last_term["reason"]),
                    "exit_code": last_term.get("exitCode"),
                    "finished_at": last_term.get("finishedAt"),
                }
            )
        finished = _parse_ts(last_term.get("finishedAt"))
        if finished is not None and (last_restart_at is None or finished > last_restart_at):
            last_restart_at = finished

    conditions = {
        str(c.get("type")): str(c.get("status"))
        for c in (status.get("conditions") or [])
        if isinstance(c, dict)
    }
    ready_containers = sum(1 for c in statuses if c.get("ready"))
    phase = str(status.get("phase") or "Unknown")
    reasons = {w["reason"] for w in waiting} | {str(t["reason"]) for t in terminated}
    unhealthy = (
        phase not in {"Running", "Succeeded"}
        or (statuses and ready_containers < len(statuses))
        or bool(reasons & (UNHEALTHY_WAITING | UNHEALTHY_TERMINATED))
    )
    age_s, age = _age(pod, now)
    return {
        "name": str(meta.get("name")),
        "namespace": str(meta.get("namespace") or ""),
        "phase": phase,
        "node": spec.get("nodeName"),
        "ready": f"{ready_containers}/{len(statuses)}" if statuses else "0/0",
        "ready_condition": conditions.get("Ready", "Unknown"),
        "restarts": restarts,
        "restarted_recently": bool(
            last_restart_at is not None and now - last_restart_at <= RESTART_WINDOW_S
        ),
        "last_restart_age": _human_duration(
            None if last_restart_at is None else now - last_restart_at
        ),
        "waiting": waiting,
        "last_terminated": terminated,
        "oom_killed": any(str(t["reason"]) == "OOMKilled" for t in terminated),
        "age": age,
        "age_s": age_s,
        "labels": meta.get("labels") or {},
        "owner": next(
            (
                f"{o.get('kind')}/{o.get('name')}"
                for o in (meta.get("ownerReferences") or [])
                if isinstance(o, dict)
            ),
            None,
        ),
        "containers": [
            _container_resources(c) for c in (spec.get("containers") or []) if isinstance(c, dict)
        ],
        "unhealthy": bool(unhealthy),
    }


def _selector_matches(selector: dict[str, Any], labels: dict[str, Any]) -> bool:
    if not selector:
        return False
    return all(labels.get(key) == value for key, value in selector.items())


def _workload_view(
    workload: dict[str, Any], pods: list[dict[str, Any]], now: float
) -> dict[str, Any]:
    meta = workload.get("metadata") or {}
    spec = workload.get("spec") or {}
    status = workload.get("status") or {}
    kind = str(workload.get("kind") or "Unknown")
    template_spec = ((spec.get("template") or {}).get("spec")) or {}
    containers = [
        _container_resources(c)
        for c in (template_spec.get("containers") or [])
        if isinstance(c, dict)
    ]

    if kind == "DaemonSet":
        desired = int(status.get("desiredNumberScheduled") or 0)
        ready = int(status.get("numberReady") or 0)
        updated = int(status.get("updatedNumberScheduled") or 0)
    else:
        desired = int(spec.get("replicas") if spec.get("replicas") is not None else 0)
        ready = int(status.get("readyReplicas") or 0)
        updated = int(status.get("updatedReplicas") or 0)

    selector = ((spec.get("selector") or {}).get("matchLabels")) or {}
    matched = [p for p in pods if _selector_matches(selector, p.get("labels") or {})]
    age_s, age = _age(workload, now)
    return {
        "name": str(meta.get("name")),
        "kind": kind,
        "namespace": str(meta.get("namespace") or ""),
        "desired": desired,
        "ready": ready,
        "updated": updated,
        "available": int(status.get("availableReplicas") or status.get("numberAvailable") or 0),
        "fully_ready": desired > 0 and ready >= desired,
        "generation_lag": int(meta.get("generation") or 0)
        - int(status.get("observedGeneration") or 0),
        "images": [c["image"] for c in containers if c.get("image")],
        "containers": containers,
        "age": age,
        "age_s": age_s,
        "pod_count": len(matched),
        "restarts": sum(int(p["restarts"]) for p in matched),
        "restarted_recently": any(p["restarted_recently"] for p in matched),
        "unhealthy_pods": [p["name"] for p in matched if p["unhealthy"]],
        "pods": [p["name"] for p in matched],
    }


# ---------------------------------------------------------------------------

_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:?\d{2}))\s+(.*)$")
_ERROR_LINE = re.compile(
    r"\b(error|err|exception|fatal|panic|fail(ed|ure)?|timeout|timed out|refused|denied|"
    r"unavailable|5\d\d)\b",
    re.I,
)
_NORMALISE: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
        "<uuid>",
    ),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<addr>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<hex>"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<id>"),
    (re.compile(r'"[^"]{0,200}"'), '"<s>"'),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|kb|mb|gb)?\b", re.I), "<n>"),
)


def _normalise_line(line: str) -> str:
    text = line
    for pattern, replacement in _NORMALISE:
        text = pattern.sub(replacement, text)
    return text.strip()[:300]


def _summarise_log_lines(lines: list[str], top_n: int = 5) -> dict[str, Any]:
    """Line count, time span, and the most repeated error shapes."""
    first_ts: float | None = None
    last_ts: float | None = None
    groups: Counter[str] = Counter()
    samples: dict[str, str] = {}
    error_count = 0

    for line in lines:
        match = _LOG_TS.match(line)
        body = line
        if match:
            stamp = _parse_ts(match.group(1))
            body = match.group(2)
            if stamp is not None:
                first_ts = stamp if first_ts is None else min(first_ts, stamp)
                last_ts = stamp if last_ts is None else max(last_ts, stamp)
        if not _ERROR_LINE.search(body):
            continue
        error_count += 1
        key = _normalise_line(body)
        if not key:
            continue
        groups[key] += 1
        samples.setdefault(key, body.strip()[:300])

    span_s = None if first_ts is None or last_ts is None else max(0.0, last_ts - first_ts)
    return {
        "line_count": len(lines),
        "error_line_count": error_count,
        "time_span": {
            "from": None if first_ts is None else datetime.fromtimestamp(first_ts).isoformat(),
            "to": None if last_ts is None else datetime.fromtimestamp(last_ts).isoformat(),
            "duration": _human_duration(span_s),
            "duration_s": span_s,
        },
        "top_errors": [
            {"count": count, "pattern": key, "sample": samples.get(key, "")}
            for key, count in groups.most_common(top_n)
        ],
    }


# ---------------------------------------------------------------------------


class _KubeArgs(BaseModel):
    context: str | None = Field(
        default=None,
        description="Cluster context. Defaults to the configured default, the session "
        "environment, then 'kubectl config current-context'.",
    )
    namespace: str | None = Field(
        default=None, description="Namespace. Always written onto the argv as -n."
    )


class CurrentContextArgs(BaseModel):
    include_contexts: bool = Field(
        default=True, description="Also list every context available in the kubeconfig."
    )


class ListNamespacesArgs(BaseModel):
    context: str | None = None
    name_contains: str | None = Field(
        default=None, description="Case-insensitive substring filter on the namespace name."
    )


class ListWorkloadsArgs(_KubeArgs):
    selector: str | None = Field(default=None, description="Label selector, for example app=api.")
    kinds: list[str] = Field(
        default_factory=lambda: list(DEFAULT_WORKLOAD_KINDS),
        description="Workload kinds to list.",
    )
    name_contains: str | None = None


class DescribeResourceArgs(_KubeArgs):
    kind: str = Field(description="Resource kind, for example deployment, pod, service.")
    name: str = Field(description="Resource name.")
    include_describe_text: bool = Field(
        default=True,
        description="Also run 'kubectl describe' and store the text as an artifact.",
    )


class GetLogsArgs(_KubeArgs):
    target: str = Field(
        description="Log source: a pod name, or kind/name such as deployment/api."
    )
    container: str | None = None
    since: str | None = Field(
        default=None, description="Relative window such as 30m or 2h, or an RFC3339 timestamp."
    )
    tail: int | None = Field(default=None, description="Trailing lines to fetch.")
    previous: bool = Field(
        default=False, description="Read the previous container instance, after a restart."
    )
    grep: str | None = Field(default=None, description="Regular expression filter, applied here.")
    all_containers: bool = False


class GetEventsArgs(_KubeArgs):
    target: str | None = Field(default=None, description="Restrict to one object name.")
    kind: str | None = Field(default=None, description="Restrict to one involved object kind.")
    only_warnings: bool = False
    within_minutes: int | None = Field(
        default=None, description="Drop events last seen before this many minutes ago."
    )
    limit: int = Field(default=25, ge=1, le=200)


class ResourceUsageArgs(_KubeArgs):
    selector: str | None = None
    pod: str | None = Field(default=None, description="Restrict usage to a single pod.")
    include_nodes: bool = False


class RolloutStatusArgs(_KubeArgs):
    kind: str = Field(default="deployment", description="deployment, statefulset, or daemonset.")
    name: str
    timeout_s: int = Field(default=30, ge=1, le=600)


class ExecReadonlyArgs(_KubeArgs):
    pod: str
    container: str | None = None
    command: list[str] = Field(description="argv to run inside the container, no shell.")

    @field_validator("command", mode="before")
    @classmethod
    def _split_command(cls, value: Any) -> Any:
        return shlex.split(value) if isinstance(value, str) else value


class PrepareMutationArgs(_KubeArgs):
    command: list[str] = Field(
        description="kubectl arguments for the mutation, without --context or -n. "
        "A leading 'kubectl' is stripped."
    )
    purpose: str = Field(default="", description="Why this change is being proposed.")
    expected_effect: str = Field(default="", description="What the operator should expect.")

    @field_validator("command", mode="before")
    @classmethod
    def _split_command(cls, value: Any) -> Any:
        return shlex.split(value) if isinstance(value, str) else value


class ExecuteApprovedMutationArgs(BaseModel):
    command_id: str | None = Field(
        default=None, description="Id returned by prepare_mutation."
    )
    approval_id: str | None = Field(
        default=None, description="Approval id, when the approval is still pending."
    )


class PodHealthArgs(_KubeArgs):
    service: str | None = Field(
        default=None, description="Workload or service name; matched as a pod name prefix."
    )
    selector: str | None = None
    include_events: bool = Field(
        default=True, description="Cross-reference namespace warning events."
    )


async def _run_one(
    ctx: ToolContext, command: ProposedCommand, timeout_s: float | None = None
) -> ExecutionRecord:
    return await _executor(ctx).run(
        command, session_id=ctx.session_id, options=_exec_options(ctx, timeout_s)
    )


async def _run_batch(
    ctx: ToolContext, commands: list[ProposedCommand]
) -> dict[str, ExecutionRecord]:
    """Run read-only commands in parallel and key the results by command id."""
    records = await _executor(ctx).run_many(
        commands, session_id=ctx.session_id, options=_exec_options(ctx)
    )
    return {record.command_id: record for record in records}


async def _fetch_pods(
    ctx: ToolContext,
    *,
    context: str,
    namespace: str,
    selector: str | None = None,
    tool_name: str = "kubernetes",
) -> tuple[list[dict[str, Any]], ExecutionRecord]:
    args = ["get", "pods", "-o", "json"]
    if selector:
        args += ["-l", _safe_token(selector, "selector")]
    command = _build(
        ctx,
        args=args,
        purpose="list pods with their restart counts and container states",
        context=context,
        namespace=namespace,
        targets=["pods"],
        tool_name=tool_name,
    )
    record = _require_ok(await _run_one(ctx, command), "listing pods")
    now = time.time()
    pods = [_pod_view(item, now) for item in _items(_parse_json(record, "listing pods"))]
    return pods, record


# ---------------------------------------------------------------------------


def _percent(token: str) -> float | None:
    """Parse a `kubectl top` percentage cell, tolerating a missing value."""
    cleaned = token.rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


@tool(
    "get_current_context",
    description=(
        "Report the active kubectl context, its namespace, and the other contexts available "
        "in the kubeconfig. Run this first so every later command can name its cluster."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "context"),
)
async def get_current_context(args: CurrentContextArgs, ctx: ToolContext) -> ToolResult:
    context, record = await _probe_current_context(ctx)
    namespace = await _resolve_namespace(ctx, None, context)
    evidence: list[Evidence] = []
    if record is not None:
        evidence.append(
            _evidence(
                ctx,
                record,
                f"the active kubectl context is {context}",
                tool_name="get_current_context",
            )
        )

    contexts: list[str] = []
    if args.include_contexts:
        listing = _build(
            ctx,
            args=["config", "get-contexts", "-o", "name"],
            purpose="list the contexts available in the kubeconfig",
            context=None,
            tool_name="get_current_context",
        )
        listed = await _run_one(ctx, listing)
        if listed.ok:
            contexts = [line.strip() for line in listed.stdout.splitlines() if line.strip()]

    kube = ctx.settings.kubernetes
    production = any(
        re.search(pattern, context, re.I)
        for pattern in ctx.settings.safety.production_context_patterns
    )
    allowed = not kube.allowed_contexts or any(
        re.search(pattern, context) for pattern in kube.allowed_contexts
    )
    denied = any(re.search(pattern, context) for pattern in kube.denied_contexts)

    summary = f"context {context}, namespace {namespace}"
    if production:
        summary += " (matches a production pattern; mutations need strong approval)"
    return ToolResult(
        tool="get_current_context",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "available_contexts": contexts,
            "kubeconfig": str(kube.kubeconfig) if kube.kubeconfig else None,
            "looks_like_production": production,
            "allowed_by_policy": allowed and not denied,
        },
        evidence=evidence,
    )


@tool(
    "list_namespaces",
    description="List namespaces in a cluster with their phase and age.",
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "discovery"),
)
async def list_namespaces(args: ListNamespacesArgs, ctx: ToolContext) -> ToolResult:
    context = await _resolve_context(ctx, args.context)
    command = _build(
        ctx,
        args=["get", "namespaces", "-o", "json"],
        purpose="list the namespaces visible in this cluster",
        context=context,
        targets=["namespaces"],
        tool_name="list_namespaces",
    )
    record = _require_ok(await _run_one(ctx, command), "listing namespaces")
    now = time.time()
    rows = []
    for item in _items(_parse_json(record, "listing namespaces")):
        meta = item.get("metadata") or {}
        name = str(meta.get("name"))
        if args.name_contains and args.name_contains.lower() not in name.lower():
            continue
        age_s, age = _age(item, now)
        rows.append(
            {
                "name": name,
                "phase": str((item.get("status") or {}).get("phase") or "Unknown"),
                "age": age,
                "age_s": age_s,
                "protected": name in ctx.settings.safety.protected_namespaces,
            }
        )
    rows.sort(key=lambda r: r["name"])
    return ToolResult(
        tool="list_namespaces",
        summary=f"{len(rows)} namespaces in {context}",
        data={"context": context, "namespaces": rows},
        evidence=[
            _evidence(
                ctx,
                record,
                f"{len(rows)} namespaces exist in context {context}",
                tool_name="list_namespaces",
            )
        ],
    )


@tool(
    "list_workloads",
    description=(
        "List deployments, statefulsets, and daemonsets in a namespace with ready/desired "
        "replicas, images, resource requests and limits, restart counts, and a flag for pods "
        "that restarted in the last hour."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "workloads"),
)
async def list_workloads(args: ListWorkloadsArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    kinds = ",".join(_safe_token(k, "kind") for k in args.kinds) or ",".join(
        DEFAULT_WORKLOAD_KINDS
    )
    selector = _safe_token(args.selector, "selector") if args.selector else None

    workload_args = ["get", kinds, "-o", "json"]
    pod_args = ["get", "pods", "-o", "json"]
    if selector:
        workload_args += ["-l", selector]
        pod_args += ["-l", selector]

    workload_cmd = _build(
        ctx,
        args=workload_args,
        purpose="list workloads with their replica and image state",
        context=context,
        namespace=namespace,
        targets=list(args.kinds),
        tool_name="list_workloads",
    )
    pod_cmd = _build(
        ctx,
        args=pod_args,
        purpose="list pods so restart counts can be attributed to their workload",
        context=context,
        namespace=namespace,
        targets=["pods"],
        tool_name="list_workloads",
    )
    records = await _run_batch(ctx, [workload_cmd, pod_cmd])
    workload_record = _require_ok(records[workload_cmd.id], "listing workloads")
    pod_record = _require_ok(records[pod_cmd.id], "listing pods")

    now = time.time()
    pods = [_pod_view(item, now) for item in _items(_parse_json(pod_record, "listing pods"))]
    workloads = [
        _workload_view(item, pods, now)
        for item in _items(_parse_json(workload_record, "listing workloads"))
    ]
    if args.name_contains:
        needle = args.name_contains.lower()
        workloads = [w for w in workloads if needle in str(w["name"]).lower()]

    degraded = [w for w in workloads if not w["fully_ready"]]
    restarting = [w for w in workloads if w["restarted_recently"]]
    orphan_pods = [p for p in pods if p["unhealthy"] and not p["owner"]]

    summary = (
        f"{len(workloads)} workloads in {context}/{namespace}: "
        f"{len(workloads) - len(degraded)} fully ready, {len(degraded)} degraded, "
        f"{len(restarting)} with a restart in the last hour"
    )
    return ToolResult(
        tool="list_workloads",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "selector": selector,
            "workloads": workloads,
            "degraded": [w["name"] for w in degraded],
            "restarted_recently": [w["name"] for w in restarting],
            "unowned_unhealthy_pods": [p["name"] for p in orphan_pods],
        },
        evidence=[
            _evidence(ctx, workload_record, summary, tool_name="list_workloads"),
            _evidence(
                ctx,
                pod_record,
                f"{len(pods)} pods in {context}/{namespace}, "
                f"{sum(1 for p in pods if p['restarted_recently'])} restarted in the last hour",
                tool_name="list_workloads",
            ),
        ],
    )


@tool(
    "describe_resource",
    description=(
        "Inspect one Kubernetes object. Returns parsed spec and status fields plus an "
        "artifact reference holding the full 'kubectl describe' text."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "inspect"),
)
async def describe_resource(args: DescribeResourceArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    kind = _safe_token(args.kind, "kind")
    name = _safe_token(args.name, "name")
    ref = f"{kind}/{name}"

    get_cmd = _build(
        ctx,
        args=["get", ref, "-o", "json"],
        purpose=f"read the structured state of {ref}",
        context=context,
        namespace=namespace,
        targets=[ref],
        pod=name if kind.lower().startswith("pod") else None,
        tool_name="describe_resource",
    )
    commands = [get_cmd]
    describe_cmd = None
    if args.include_describe_text:
        describe_cmd = _build(
            ctx,
            args=["describe", ref],
            purpose=f"capture the human-readable description of {ref}, including its events",
            context=context,
            namespace=namespace,
            targets=[ref],
            pod=name if kind.lower().startswith("pod") else None,
            tool_name="describe_resource",
        )
        commands.append(describe_cmd)

    records = await _run_batch(ctx, commands)
    get_record = _require_ok(records[get_cmd.id], f"reading {ref}")
    obj = _parse_json(get_record, f"reading {ref}")

    meta = obj.get("metadata") or {}
    spec = obj.get("spec") or {}
    status = obj.get("status") or {}
    now = time.time()
    age_s, age = _age(obj, now)
    conditions = [
        {
            "type": c.get("type"),
            "status": c.get("status"),
            "reason": c.get("reason"),
            "message": str(c.get("message", ""))[:300],
            "age": _human_duration(
                None
                if _parse_ts(c.get("lastTransitionTime")) is None
                else now - float(_parse_ts(c.get("lastTransitionTime")) or now)
            ),
        }
        for c in (status.get("conditions") or [])
        if isinstance(c, dict)
    ]
    template_spec = ((spec.get("template") or {}).get("spec")) or {}
    containers = [
        _container_resources(c)
        for c in (template_spec.get("containers") or spec.get("containers") or [])
        if isinstance(c, dict)
    ]

    artifact_ref = None
    evidence = [
        _evidence(ctx, get_record, f"current state of {ref} in {context}/{namespace}",
                  tool_name="describe_resource")
    ]
    describe_tail: list[str] = []
    if describe_cmd is not None:
        describe_record = records[describe_cmd.id]
        if describe_record.ok and describe_record.stdout.strip():
            artifact = _artifacts(ctx).put(
                describe_record.stdout,
                kind="kubectl_describe",
                session_id=ctx.session_id,
                metadata={"context": context, "namespace": namespace, "resource": ref},
            )
            artifact_ref = artifact.ref
            # The Events block at the bottom of describe is usually the useful part.
            lines = describe_record.stdout.splitlines()
            for index, line in enumerate(lines):
                if line.startswith("Events:"):
                    describe_tail = [entry.rstrip() for entry in lines[index : index + 15]]
                    break
            evidence.append(
                _evidence(
                    ctx,
                    describe_record,
                    f"kubectl describe output for {ref}",
                    tool_name="describe_resource",
                )
            )

    return ToolResult(
        tool="describe_resource",
        summary=(
            f"{obj.get('kind', kind)}/{name} in {context}/{namespace}, age {age}, "
            f"{len(conditions)} status conditions"
        ),
        data={
            "context": context,
            "namespace": namespace,
            "kind": obj.get("kind", kind),
            "name": name,
            "age": age,
            "age_s": age_s,
            "labels": meta.get("labels") or {},
            "annotations": {
                k: v
                for k, v in (meta.get("annotations") or {}).items()
                if not k.startswith("kubectl.kubernetes.io/last-applied")
            },
            "owner_references": meta.get("ownerReferences") or [],
            "replicas": {
                "desired": spec.get("replicas"),
                "ready": status.get("readyReplicas"),
                "available": status.get("availableReplicas"),
                "updated": status.get("updatedReplicas"),
            },
            "containers": containers,
            "conditions": conditions,
            "phase": status.get("phase"),
            "events_excerpt": describe_tail,
        },
        evidence=evidence,
        artifact_ref=artifact_ref,
    )


_DURATION = re.compile(r"^\d+[smhd]$")


class FindWorkloadsArgs(BaseModel):
    """No namespace, and no single context. That is the point of this tool."""

    name_contains: str = Field(description="Substring of the pod name.")
    context_contains: str | None = Field(
        default=None,
        description="Cluster context substrings, space separated. All must match.",
    )
    namespace_contains: str | None = Field(default=None)
    environment: str | None = Field(
        default=None,
        description="dev or prod: only contexts whose name ends in -dev or -prod, fanned out across all of them.",
    )
    regions: list[str] | None = Field(
        default=None,
        description="Region fragments such as ch1, fr5, dc2: only contexts containing one, fanned out across all of them.",
    )
    limit: int = Field(default=40, ge=1, le=200)


def _remember_unreachable(store: Any, context: str) -> None:
    """Mark a context unreachable now, so the next fan-out skips it for a while."""
    try:
        from mimir.knowledge.entities import Entity, entity_id

        ent = store.get(entity_id("context", context))
        attrs = dict(ent.attrs) if ent else {}
        attrs["unreachable_at"] = time.time()
        store.upsert(Entity(entity_id("context", context), "context", context, attrs=attrs, seen_at=time.time()))
        store.commit()
    except Exception:  # noqa: BLE001 - a cache must not fail a tool
        pass


@tool(
    "find_workloads",
    description=(
        "Find pods by name across every namespace and every matching cluster context, "
        "in one call. Use it whenever the operator describes what they want instead of "
        "naming a namespace."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "search"),
)
async def find_workloads(args: FindWorkloadsArgs, ctx: ToolContext) -> ToolResult:
    """Search by substring, deterministically."""
    wanted = _safe_token(_as_fragment(args.name_contains), "name_contains").lower()
    contexts = await _matching_contexts(
        ctx, args.context_contains, environment=args.environment, regions=args.regions
    )
    if not contexts:
        raise ToolError(
            f"no kubectl context matches {args.context_contains!r}",
            code="not_found",
        )

    # Three columns, not the pod objects.
    columns = (
        "NS:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase"
    )
    commands = [
        _build(
            ctx,
            args=["get", "pods", "--all-namespaces", "-o",
                  f"custom-columns={columns}", "--no-headers",
                  f"--request-timeout={int(ctx.settings.kubernetes.fanout_timeout_s)}s"],
            purpose=f"find pods matching {wanted!r} across namespaces",
            context=name,
            tool_name="find_workloads",
            timeout_s=ctx.settings.kubernetes.fanout_timeout_s + 5,
        )
        for name in contexts
    ]
    records = await _run_batch(ctx, commands)

    rows: list[dict[str, Any]] = []
    unreachable: list[str] = []
    namespace_filter = _as_fragment(args.namespace_contains).lower()
    store = getattr(ctx, "entities", None)
    for name, command in zip(contexts, commands, strict=True):
        record = records[command.id]
        if not record.ok:
            unreachable.append(name)
            if store is not None:
                _remember_unreachable(store, name)
            continue
        for line in record.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            namespace, pod_name = parts[0], parts[1]
            phase = parts[2] if len(parts) > 2 else ""
            if wanted not in pod_name.lower():
                continue
            if namespace_filter and namespace_filter not in namespace.lower():
                continue
            rows.append({
                "context": name,
                "namespace": namespace,
                "pod": pod_name,
                "phase": phase,
            })

    # Absence and failure are different answers.
    if unreachable and not rows and len(unreachable) == len(contexts):
        raise ToolError(
            f"every context searched was unreachable: {', '.join(unreachable)}. "
            "This is not evidence that nothing matches.",
            code="unavailable",
        )

    # Stable ordering: the same question returns the same answer, and the first
    rows.sort(key=lambda r: (r["context"], r["namespace"], r["pod"]))
    shown = rows[: args.limit]

    where = f" in {len(contexts)} context(s) matching {args.context_contains!r}" if (
        args.context_contains
    ) else ""
    if not rows:
        summary = f"no pod name contains {wanted!r}{where}"
    else:
        places = sorted({f"{r['context']}/{r['namespace']}" for r in shown})
        summary = (
            f"{len(rows)} pod(s) matching {wanted!r}{where}, in "
            f"{len(places)} namespace(s): {', '.join(places[:4])}"
        )
    if unreachable:
        summary += f" ({len(unreachable)} context(s) unreachable)"

    return ToolResult(
        tool="find_workloads",
        summary=summary,
        data={
            "matches": shown,
            "total": len(rows),
            "contexts_searched": contexts,
            "contexts_unreachable": unreachable,
        },
        truncated=len(rows) > len(shown),
        evidence=[
            _evidence(
                ctx,
                records[commands[0].id],
                f"{len(rows)} pod(s) match {wanted!r} across {len(contexts)} context(s)",
                tool_name="find_workloads",
            )
        ] if rows else [],
    )


def _as_fragment(value: str | None) -> str:
    """Normalise a name fragment the way names are actually written."""
    if not value:
        return ""
    return "-".join(str(value).strip().lower().split())


async def _matching_contexts(
    ctx: ToolContext, fragment: str | None, *, environment: str | None = None,
    regions: list[str] | None = None,
) -> list[str]:
    """Contexts containing every fragment, filtered by environment suffix and region, sorted."""
    fragments = [f for f in (fragment or "").lower().replace(",", " ").split() if f]
    environment = (environment or getattr(ctx.environment, "environment", None) or "").lower()
    environment = {"production": "prod", "development": "dev"}.get(environment, environment)
    regions = [r.lower() for r in (regions or []) if r]
    if not fragments and not environment and not regions:
        return [await _resolve_context(ctx, None)]
    # The operator's regions, when the request names none; the script they
    # trust tries ch1|fr5|dc2 and nothing else.
    if not regions and not fragments:
        regions = [r.lower() for r in ctx.settings.kubernetes.regions]
    listing = _build(
        ctx,
        args=["config", "get-contexts", "-o", "name"],
        purpose="list contexts so the named cluster can be resolved",
        context=None,
        tool_name="find_workloads",
    )
    record = await _run_one(ctx, listing)
    if not record.ok:
        return [await _resolve_context(ctx, None)]
    names = [line.strip() for line in record.stdout.splitlines() if line.strip()]
    kept = [n for n in names if all(f in n.lower() for f in fragments)]
    if environment:
        kept = [n for n in kept if n.lower().endswith(f"-{environment}")]
    if regions:
        kept = [n for n in kept if any(r in n.lower() for r in regions)]
    # A context that failed recently is skipped unless it was named outright.
    store = getattr(ctx, "entities", None)
    if store is not None and not fragments:
        from mimir.knowledge.entities import entity_id

        horizon = time.time() - ctx.settings.kubernetes.skip_unreachable_for_s
        for name in list(kept):
            ent = store.get(entity_id("context", name))
            when = (ent.attrs.get("unreachable_at") if ent else None) or 0
            if when and when > horizon:
                kept.remove(name)
    return sorted(kept)


_LOG_KINDS = frozenset({
    "pod", "po", "pods", "deployment", "deploy", "deployments",
    "statefulset", "sts", "daemonset", "ds", "job", "cronjob", "cj",
    "replicaset", "rs", "service", "svc",
})
"""Kinds kubectl will read logs from. Used to tell kind/name from namespace/pod."""


@tool(
    "get_logs",
    description=(
        "Fetch container logs for a pod or workload. The full body is stored as an artifact; "
        "the result carries only a line count, the time span, and the most repeated error "
        "lines. Supports since, tail, container, previous, and a regex filter."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    long_running=True,
    tags=("kubernetes", "logs"),
)
async def get_logs(args: GetLogsArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    target = _safe_token(args.target, "target")

    # "messaging-squad/messaging-router-abc" is how both operators and models
    if "/" in target:
        kind = target.split("/", 1)[0].lower()
        if kind not in _LOG_KINDS:
            raise ToolError(
                f"target {target!r} is not kind/name: {kind!r} is not a kind "
                "kubectl reads logs from. If it is a namespace, pass it as the "
                "namespace argument and give the pod name alone as target.",
                code="invalid_arguments",
            )
    tail = args.tail if args.tail is not None else ctx.settings.kubernetes.log_tail_lines

    # --timestamps is not optional here: the time span in the summary is derived
    log_args = ["logs", target, "--timestamps=true", f"--tail={int(tail)}"]
    if args.container:
        log_args += ["-c", _safe_token(args.container, "container")]
    elif args.all_containers:
        log_args += ["--all-containers=true", "--prefix=true"]
    if args.since:
        since = _safe_token(args.since, "since")
        log_args.append(
            f"--since={since}" if _DURATION.match(since) else f"--since-time={since}"
        )
    if args.previous:
        log_args.append("--previous=true")

    is_pod_ref = "/" not in target
    command = _build(
        ctx,
        args=log_args,
        purpose=f"read logs for {target}",
        context=context,
        namespace=namespace,
        targets=[target],
        pod=target if is_pod_ref else None,
        container=args.container,
        tool_name="get_logs",
    )
    # The scope belongs in the failure text.
    record = _require_ok(
        await _run_one(ctx, command),
        f"reading logs for {target} in {context}/{namespace}",
    )

    body = record.stdout
    lines = body.splitlines()
    matched: list[str] = []
    if args.grep:
        try:
            pattern = re.compile(args.grep, re.I)
        except re.error as exc:
            raise ToolError(f"invalid grep pattern: {exc}", code="invalid_arguments") from exc
        matched = [line for line in lines if pattern.search(line)]

    stats = _summarise_log_lines(matched if args.grep else lines)
    artifact = _artifacts(ctx).put(
        body,
        kind="kubectl_logs",
        session_id=ctx.session_id,
        metadata={
            "context": context,
            "namespace": namespace,
            "target": target,
            "container": args.container,
            "since": args.since,
            "tail": tail,
            "previous": args.previous,
            "command": record.display,
        },
    )

    summary = (
        f"{stats['line_count']} log lines for {target} in {context}/{namespace} spanning "
        f"{stats['time_span']['duration']}, {stats['error_line_count']} error-shaped lines"
    )
    if args.grep:
        summary += f", {len(matched)} of {len(lines)} lines matched /{args.grep}/"
    return ToolResult(
        tool="get_logs",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "target": target,
            "container": args.container,
            "previous": args.previous,
            "grep": args.grep,
            "grep_matches": len(matched) if args.grep else None,
            "grep_sample": matched[:10] if args.grep else [],
            **stats,
            "artifact_ref": artifact.ref,
        },
        artifact_ref=artifact.ref,
        truncated=record.truncated,
        evidence=[_evidence(ctx, record, summary, tool_name="get_logs")],
    )


@tool(
    "get_events",
    description=(
        "Read namespace events, newest last-seen first, with repeated events grouped and "
        "counted. Optionally restricted to one object or to warnings."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "events"),
)
async def get_events(args: GetEventsArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    event_args = ["get", "events", "-o", "json"]
    selectors = []
    if args.target:
        selectors.append(f"involvedObject.name={_safe_token(args.target, 'target')}")
    if args.kind:
        selectors.append(f"involvedObject.kind={_safe_token(args.kind, 'kind')}")
    if args.only_warnings:
        selectors.append("type=Warning")
    if selectors:
        event_args.append(f"--field-selector={','.join(selectors)}")

    command = _build(
        ctx,
        args=event_args,
        purpose="read recent events for this namespace",
        context=context,
        namespace=namespace,
        targets=[args.target] if args.target else ["events"],
        tool_name="get_events",
    )
    record = _require_ok(await _run_one(ctx, command), "reading events")
    groups = _group_events(
        _items(_parse_json(record, "reading events")),
        within_minutes=args.within_minutes,
        limit=args.limit,
    )
    warnings = [g for g in groups if g["type"] == "Warning"]
    summary = (
        f"{len(groups)} distinct events in {context}/{namespace} "
        f"({sum(int(g['count']) for g in groups)} occurrences, {len(warnings)} warning groups)"
    )
    return ToolResult(
        tool="get_events",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "events": groups,
            "warning_reasons": sorted({str(g["reason"]) for g in warnings}),
        },
        evidence=[_evidence(ctx, record, summary, tool_name="get_events")],
    )


def _event_last_seen(event: dict[str, Any]) -> float | None:
    """Events changed shape between the core and events.k8s.io APIs, so several fields have to be tried before an event is treated as undated."""
    series = event.get("series") or {}
    for candidate in (
        event.get("lastTimestamp"),
        series.get("lastObservedTime"),
        event.get("eventTime"),
        event.get("deprecatedLastTimestamp"),
        event.get("firstTimestamp"),
        (event.get("metadata") or {}).get("creationTimestamp"),
    ):
        stamp = _parse_ts(candidate)
        if stamp is not None:
            return stamp
    return None


def _group_events(
    events: list[dict[str, Any]], *, within_minutes: int | None, limit: int
) -> list[dict[str, Any]]:
    now = time.time()
    cutoff = now - within_minutes * 60 if within_minutes else None
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in events:
        last_seen = _event_last_seen(event)
        if cutoff is not None and (last_seen is None or last_seen < cutoff):
            continue
        involved = event.get("involvedObject") or event.get("regarding") or {}
        obj = f"{involved.get('kind', '?')}/{involved.get('name', '?')}"
        message = str(event.get("message") or event.get("note") or "")
        key = (
            str(event.get("type") or "Normal"),
            str(event.get("reason") or ""),
            obj,
            _normalise_line(message),
        )
        count = int(event.get("count") or (event.get("series") or {}).get("count") or 1)
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                "type": key[0],
                "reason": key[1],
                "object": obj,
                "message": message[:300],
                "count": count,
                "last_seen_s": last_seen,
                "last_seen": _human_duration(None if last_seen is None else now - last_seen),
                "source": (event.get("source") or {}).get("component")
                or (event.get("reportingComponent") or None),
            }
        else:
            entry["count"] = int(entry["count"]) + count
            if last_seen is not None and (
                entry["last_seen_s"] is None or last_seen > float(entry["last_seen_s"])
            ):
                entry["last_seen_s"] = last_seen
                entry["last_seen"] = _human_duration(now - last_seen)
    ordered = sorted(
        grouped.values(),
        key=lambda g: (g["last_seen_s"] is not None, g["last_seen_s"] or 0.0),
        reverse=True,
    )
    return ordered[:limit]


def _parse_top_containers(stdout: str) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    """Parse ``kubectl top pods --containers --no-headers`` into (pod, container)."""
    out: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        out[(parts[0], parts[1])] = (_parse_cpu(parts[2]), _parse_memory(parts[3]))
    return out


def _parse_top_nodes(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append(
            {
                "node": parts[0],
                "cpu": parts[1],
                "cpu_percent": _percent(parts[2]),
                "memory": parts[3],
                "memory_percent": _percent(parts[4]),
            }
        )
    return rows


@tool(
    "get_resource_usage",
    description=(
        "Read live CPU and memory usage with 'kubectl top' and cross-reference it against the "
        "requests and limits on each container, so CPU throttling risk and memory pressure are "
        "visible. Reports when metrics-server is missing instead of failing."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "metrics"),
)
async def get_resource_usage(args: ResourceUsageArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    selector = _safe_token(args.selector, "selector") if args.selector else None

    top_args = ["top", "pods", "--containers", "--no-headers"]
    if args.pod:
        top_args.insert(2, _safe_token(args.pod, "pod"))
    elif selector:
        top_args += ["-l", selector]
    top_cmd = _build(
        ctx,
        args=top_args,
        purpose="read live CPU and memory usage per container",
        context=context,
        namespace=namespace,
        targets=[args.pod] if args.pod else ["pods"],
        pod=args.pod,
        tool_name="get_resource_usage",
    )
    pod_args = ["get", "pods", "-o", "json"]
    if selector:
        pod_args += ["-l", selector]
    pods_cmd = _build(
        ctx,
        args=pod_args,
        purpose="read the requests and limits that live usage is compared against",
        context=context,
        namespace=namespace,
        targets=["pods"],
        tool_name="get_resource_usage",
    )
    commands = [top_cmd, pods_cmd]
    nodes_cmd = None
    if args.include_nodes:
        nodes_cmd = _build(
            ctx,
            args=["top", "nodes", "--no-headers"],
            purpose="read node level CPU and memory pressure",
            context=context,
            targets=["nodes"],
            tool_name="get_resource_usage",
        )
        commands.append(nodes_cmd)

    records = await _run_batch(ctx, commands)
    pods_record = _require_ok(records[pods_cmd.id], "listing pods")
    top_record = records[top_cmd.id]

    metrics_available = top_record.ok
    metrics_note = None
    if not metrics_available:
        code, message, _ = _explain(top_record)
        metrics_note = (
            "metrics-server does not appear to be installed in this cluster; "
            "requests and limits are reported without live usage"
            if code == "metrics_unavailable"
            else f"kubectl top failed ({code}): {message}"
        )
    usage = _parse_top_containers(top_record.stdout) if metrics_available else {}

    now = time.time()
    pod_objects = _items(_parse_json(pods_record, "listing pods"))
    rows: list[dict[str, Any]] = []
    for pod in pod_objects:
        view = _pod_view(pod, now)
        if args.pod and view["name"] != args.pod:
            continue
        for container in view["containers"]:
            used_cpu, used_mem = usage.get((view["name"], str(container["name"])), (None, None))
            cpu_limit = _parse_cpu(container["cpu_limit"])
            mem_limit = _parse_memory(container["memory_limit"])
            cpu_request = _parse_cpu(container["cpu_request"])
            mem_request = _parse_memory(container["memory_request"])
            cpu_pct_limit = _pct(used_cpu, cpu_limit)
            mem_pct_limit = _pct(used_mem, mem_limit)
            rows.append(
                {
                    "pod": view["name"],
                    "container": container["name"],
                    "cpu_used": _fmt_cpu(used_cpu),
                    "cpu_request": container["cpu_request"],
                    "cpu_limit": container["cpu_limit"],
                    "cpu_percent_of_request": _pct(used_cpu, cpu_request),
                    "cpu_percent_of_limit": cpu_pct_limit,
                    "memory_used": _fmt_memory(used_mem),
                    "memory_request": container["memory_request"],
                    "memory_limit": container["memory_limit"],
                    "memory_percent_of_request": _pct(used_mem, mem_request),
                    "memory_percent_of_limit": mem_pct_limit,
                    "cpu_throttling_risk": bool(
                        cpu_pct_limit is not None and cpu_pct_limit >= CPU_PRESSURE_PCT
                    ),
                    "memory_pressure": bool(
                        mem_pct_limit is not None and mem_pct_limit >= MEMORY_PRESSURE_PCT
                    ),
                    "no_cpu_limit": container["cpu_limit"] is None,
                    "no_memory_limit": container["memory_limit"] is None,
                    "oom_killed_recently": view["oom_killed"],
                    "restarts": view["restarts"],
                }
            )

    rows.sort(key=lambda r: (r["memory_percent_of_limit"] or 0, r["cpu_percent_of_limit"] or 0),
              reverse=True)
    throttling = [r for r in rows if r["cpu_throttling_risk"]]
    pressure = [r for r in rows if r["memory_pressure"] or r["oom_killed_recently"]]

    nodes: list[dict[str, Any]] = []
    if nodes_cmd is not None and records[nodes_cmd.id].ok:
        nodes = _parse_top_nodes(records[nodes_cmd.id].stdout)

    summary = (
        f"{len(rows)} containers in {context}/{namespace}: {len(throttling)} near their CPU "
        f"limit, {len(pressure)} under memory pressure"
    )
    if metrics_note:
        summary = f"{summary}. {metrics_note}"

    evidence = [_evidence(ctx, pods_record, summary, tool_name="get_resource_usage")]
    if metrics_available:
        evidence.append(
            _evidence(
                ctx,
                top_record,
                f"live container usage in {context}/{namespace}",
                tool_name="get_resource_usage",
            )
        )
    return ToolResult(
        tool="get_resource_usage",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "metrics_available": metrics_available,
            "metrics_note": metrics_note,
            "containers": rows[:60],
            "cpu_throttling_risk": [f"{r['pod']}/{r['container']}" for r in throttling],
            "memory_pressure": [f"{r['pod']}/{r['container']}" for r in pressure],
            "nodes": nodes,
        },
        evidence=evidence,
    )


@tool(
    "get_rollout_status",
    description=(
        "Report rollout progress for a deployment, statefulset, or daemonset: replica counts, "
        "current revision, progress conditions, and whether the rollout is stalled."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "rollout"),
)
async def get_rollout_status(args: RolloutStatusArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    kind = _safe_token(args.kind, "kind")
    name = _safe_token(args.name, "name")
    ref = f"{kind}/{name}"

    # --watch=false keeps this a bounded read instead of blocking until the
    status_cmd = _build(
        ctx,
        args=["rollout", "status", ref, "--watch=false", f"--timeout={args.timeout_s}s"],
        purpose=f"check rollout progress for {ref}",
        context=context,
        namespace=namespace,
        targets=[ref],
        tool_name="get_rollout_status",
        timeout_s=args.timeout_s + 15,
    )
    get_cmd = _build(
        ctx,
        args=["get", ref, "-o", "json"],
        purpose=f"read replica counts and conditions for {ref}",
        context=context,
        namespace=namespace,
        targets=[ref],
        tool_name="get_rollout_status",
    )
    records = await _run_batch(ctx, [status_cmd, get_cmd])
    get_record = _require_ok(records[get_cmd.id], f"reading {ref}")
    status_record = records[status_cmd.id]

    obj = _parse_json(get_record, f"reading {ref}")
    meta = obj.get("metadata") or {}
    spec = obj.get("spec") or {}
    status = obj.get("status") or {}
    conditions = [
        {
            "type": c.get("type"),
            "status": c.get("status"),
            "reason": c.get("reason"),
            "message": str(c.get("message", ""))[:300],
        }
        for c in (status.get("conditions") or [])
        if isinstance(c, dict)
    ]
    stalled = any(c.get("reason") == "ProgressDeadlineExceeded" for c in conditions)
    desired = spec.get("replicas")
    if kind.lower().startswith("daemonset"):
        desired = status.get("desiredNumberScheduled")
    ready = status.get("readyReplicas") or status.get("numberReady") or 0
    generation_lag = int(meta.get("generation") or 0) - int(status.get("observedGeneration") or 0)
    complete = bool(desired is not None and int(ready or 0) >= int(desired or 0) and not stalled)

    message = status_record.stdout.strip().splitlines()[-1:] or []
    summary = (
        f"{ref} in {context}/{namespace}: {ready}/{desired} ready, "
        f"{'complete' if complete else 'stalled' if stalled else 'in progress'}"
    )
    return ToolResult(
        tool="get_rollout_status",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "resource": ref,
            "desired": desired,
            "ready": ready,
            "updated": status.get("updatedReplicas") or status.get("updatedNumberScheduled"),
            "available": status.get("availableReplicas") or status.get("numberAvailable"),
            "unavailable": status.get("unavailableReplicas"),
            "revision": (meta.get("annotations") or {}).get("deployment.kubernetes.io/revision"),
            "observed_generation_lag": generation_lag,
            "complete": complete,
            "stalled": stalled,
            "conditions": conditions,
            "kubectl_message": message[0] if message else None,
            "images": [
                c.get("image")
                for c in (((spec.get("template") or {}).get("spec") or {}).get("containers") or [])
                if isinstance(c, dict)
            ],
        },
        evidence=[_evidence(ctx, get_record, summary, tool_name="get_rollout_status")],
    )


@tool(
    "summarise_pod_health",
    description=(
        "Health picture for one service: per-pod readiness, restart counts, pods that restarted "
        "in the last hour, and waiting or termination reasons such as CrashLoopBackOff and "
        "OOMKilled, cross-referenced with namespace warning events."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R1,
    tags=("kubernetes", "health", "diagnosis"),
)
async def summarise_pod_health(args: PodHealthArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    selector = _safe_token(args.selector, "selector") if args.selector else None
    pods, pods_record = await _fetch_pods(
        ctx,
        context=context,
        namespace=namespace,
        selector=selector,
        tool_name="summarise_pod_health",
    )
    if args.service:
        service = _safe_token(args.service, "service")
        pods = [
            p
            for p in pods
            if str(p["name"]).startswith(service)
            or service in {p["labels"].get("app"), p["labels"].get("app.kubernetes.io/name")}
        ]

    reasons: Counter[str] = Counter()
    for pod in pods:
        for waiting in pod["waiting"]:
            reasons[str(waiting["reason"])] += 1
        for terminated in pod["last_terminated"]:
            reasons[str(terminated["reason"])] += 1

    unhealthy = [p for p in pods if p["unhealthy"]]
    restarting = [p for p in pods if p["restarted_recently"]]
    oom = [p for p in pods if p["oom_killed"]]
    crashloop = [
        p for p in pods if any(w["reason"] == "CrashLoopBackOff" for w in p["waiting"])
    ]

    evidence = [
        _evidence(
            ctx,
            pods_record,
            f"pod health for {args.service or selector or 'all pods'} in {context}/{namespace}",
            tool_name="summarise_pod_health",
        )
    ]
    warning_events: list[dict[str, Any]] = []
    if args.include_events and pods:
        events_cmd = _build(
            ctx,
            args=["get", "events", "-o", "json", "--field-selector=type=Warning"],
            purpose="cross-reference warning events with unhealthy pods",
            context=context,
            namespace=namespace,
            targets=[p["name"] for p in unhealthy][:10] or ["events"],
            tool_name="summarise_pod_health",
        )
        events_record = await _run_one(ctx, events_cmd)
        if events_record.ok:
            names = {str(p["name"]) for p in pods}
            grouped = _group_events(
                _items(_parse_json(events_record, "reading events")),
                within_minutes=None,
                limit=100,
            )
            warning_events = [
                g for g in grouped if str(g["object"]).split("/", 1)[-1] in names
            ][:15]
            evidence.append(
                _evidence(
                    ctx,
                    events_record,
                    f"warning events touching these pods in {context}/{namespace}",
                    tool_name="summarise_pod_health",
                )
            )

    summary = (
        f"{len(pods)} pods, {len(unhealthy)} unhealthy, {len(restarting)} restarted in the last "
        f"hour, {len(crashloop)} in CrashLoopBackOff, {len(oom)} OOMKilled"
    )
    return ToolResult(
        tool="summarise_pod_health",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "service": args.service,
            "pod_count": len(pods),
            "healthy": len(pods) - len(unhealthy),
            "restarted_recently": [p["name"] for p in restarting],
            "crash_looping": [p["name"] for p in crashloop],
            "oom_killed": [p["name"] for p in oom],
            "reason_counts": dict(reasons.most_common()),
            "pods": [
                {
                    key: pod[key]
                    for key in (
                        "name",
                        "phase",
                        "ready",
                        "restarts",
                        "restarted_recently",
                        "last_restart_age",
                        "waiting",
                        "last_terminated",
                        "age",
                        "node",
                    )
                }
                for pod in pods[:50]
            ],
            "warning_events": warning_events,
        },
        evidence=evidence,
    )


# ---------------------------------------------------------------------------


@tool(
    "exec_readonly",
    description=(
        "Run a read-only command inside a running container. This is R2 elevated inspection "
        "and the policy engine will ask for approval before it runs."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R2,
    requires_approval=True,
    tags=("kubernetes", "exec"),
)
async def exec_readonly(args: ExecReadonlyArgs, ctx: ToolContext) -> ToolResult:
    context, namespace = await _scope(ctx, args)
    pod = _safe_token(args.pod, "pod")
    if not args.command:
        raise ToolError("exec_readonly needs a command to run", code="invalid_arguments")

    exec_args = ["exec", pod]
    if args.container:
        exec_args += ["-c", _safe_token(args.container, "container")]
    exec_args += ["--", *args.command]

    payload = shlex.join(args.command)
    command = _build(
        ctx,
        args=exec_args,
        purpose=f"inspect {pod} from inside the container: {payload}",
        expected_effect="reads state inside a running container; makes no change on its own",
        context=context,
        namespace=namespace,
        targets=[f"pod/{pod}"],
        pod=pod,
        container=args.container,
        tool_name="exec_readonly",
    )
    # No local allow list here.
    record = await _run_one(ctx, command)
    _require_ok(record, f"exec into {pod}")

    output = record.stdout
    artifact = _artifacts(ctx).put(
        record.combined_output(),
        kind="kubectl_exec",
        session_id=ctx.session_id,
        metadata={
            "context": context,
            "namespace": namespace,
            "pod": pod,
            "container": args.container,
            "command": record.display,
        },
    )
    lines = output.splitlines()
    summary = f"exec in {pod} ({context}/{namespace}) returned {len(lines)} lines"
    return ToolResult(
        tool="exec_readonly",
        summary=summary,
        data={
            "context": context,
            "namespace": namespace,
            "pod": pod,
            "container": args.container,
            "command": args.command,
            "exit_code": record.exit_code,
            "line_count": len(lines),
            "head": lines[:40],
            "artifact_ref": artifact.ref,
        },
        artifact_ref=artifact.ref,
        truncated=record.truncated,
        evidence=[_evidence(ctx, record, summary, tool_name="exec_readonly")],
    )


# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PreparedMutation:
    command: ProposedCommand
    session_id: str | None
    risk: RiskClass
    requires_approval: bool
    preview: str
    created_at: float = field(default_factory=time.time)
    executed_at: float | None = None


#: Prepared mutations awaiting execution, keyed by command id and scoped by
_PREPARED: dict[str, _PreparedMutation] = {}
_PREPARED_TTL_S = 3600.0
_MAX_PREPARED = 128

#: Scope flags are re-injected from the resolved context, so any copy the model
_SCOPE_FLAGS = {"--context": "context", "-n": "namespace", "--namespace": "namespace"}


def _split_scope_flags(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    rest: list[str] = []
    context: str | None = None
    namespace: str | None = None
    pending: str | None = None
    for arg in argv:
        if pending:
            if pending == "context":
                context = arg
            else:
                namespace = arg
            pending = None
            continue
        if arg in _SCOPE_FLAGS:
            pending = _SCOPE_FLAGS[arg]
            continue
        if "=" in arg and arg.split("=", 1)[0] in _SCOPE_FLAGS:
            flag, value = arg.split("=", 1)
            if _SCOPE_FLAGS[flag] == "context":
                context = value
            else:
                namespace = value
            continue
        rest.append(arg)
    return rest, context, namespace


_RESOURCE_REF = re.compile(r"^[a-z][a-z0-9.-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _mutation_targets(argv: list[str]) -> list[str]:
    """Resource references named in a mutation, for the pre-execution display."""
    targets = [arg for arg in argv if _RESOURCE_REF.match(arg)]
    if targets:
        return targets
    # `kubectl delete deployment api` style: verb, kind, then names.
    positional = [a for a in argv if not a.startswith("-")]
    if len(positional) >= 3:
        return [f"{positional[1]}/{name}" for name in positional[2:]]
    return positional[1:2]


def _forget_stale() -> None:
    cutoff = time.time() - _PREPARED_TTL_S
    for key, entry in list(_PREPARED.items()):
        if entry.created_at < cutoff:
            _PREPARED.pop(key, None)
    while len(_PREPARED) > _MAX_PREPARED:
        oldest = min(_PREPARED, key=lambda k: _PREPARED[k].created_at)
        _PREPARED.pop(oldest, None)


@tool(
    "prepare_mutation",
    description=(
        "Build a Kubernetes mutation without running it. Returns the exact command, resolved "
        "cluster and namespace, target objects, risk class, expected impact, and rollback "
        "guidance. Pass the returned command_id to execute_approved_mutation to run it."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R0,
    tags=("kubernetes", "mutation", "preview"),
)
async def prepare_mutation(args: PrepareMutationArgs, ctx: ToolContext) -> ToolResult:
    body = list(args.command)
    if body and body[0].rsplit("/", 1)[-1] == "kubectl":
        body = body[1:]
    body, inline_context, inline_namespace = _split_scope_flags(body)
    if not body:
        raise ToolError("prepare_mutation needs a kubectl command", code="invalid_arguments")

    context = await _resolve_context(ctx, args.context or inline_context)
    namespace = await _resolve_namespace(ctx, args.namespace or inline_namespace, context)
    targets = _mutation_targets(body)

    command = _build(
        ctx,
        args=body,
        purpose=args.purpose or f"kubectl {body[0]} on {', '.join(targets) or namespace}",
        expected_effect=args.expected_effect
        or f"changes {', '.join(targets) or 'objects'} in {context}/{namespace}",
        context=context,
        namespace=namespace,
        targets=targets,
        tool_name="prepare_mutation",
    )
    decision = _executor(ctx).policy.evaluate(command)
    assessment = decision.assessment

    _forget_stale()
    _PREPARED[command.id] = _PreparedMutation(
        command=command,
        session_id=ctx.session_id,
        risk=assessment.risk,
        requires_approval=decision.needs_approval or assessment.requires_approval,
        preview=command.render_preview(),
    )

    rollback = assessment.rollback_hint or (
        "no automatic rollback is known for this command; capture the current manifest first"
    )
    if decision.denied:
        verdict_text = "Denied by policy"
    elif decision.needs_approval:
        verdict_text = "Approval required"
    else:
        verdict_text = "Within auto-execute policy"
    summary = (
        f"prepared {assessment.risk.value} mutation on {context}/{namespace}: "
        f"{command.display}. {verdict_text}."
    )
    return ToolResult(
        ok=not decision.denied,
        tool="prepare_mutation",
        summary=summary,
        error=decision.reason if decision.denied else None,
        error_code="policy_denied" if decision.denied else None,
        data={
            "command_id": command.id,
            "command": command.display,
            "argv": command.argv,
            "context": context,
            "namespace": namespace,
            "targets": targets,
            "risk": assessment.risk.value,
            "risk_summary": assessment.summary,
            "reasons": assessment.reasons,
            "requires_approval": decision.needs_approval or assessment.requires_approval,
            "denied": decision.denied,
            "denial_reason": decision.reason if decision.denied else None,
            "production_target": assessment.production_target,
            "reversible": assessment.reversible,
            "expected_effect": command.expected_effect,
            "rollback": rollback,
            "preview": command.render_preview(),
            "next_step": (
                "call execute_approved_mutation with this command_id; the operator will be "
                "asked to approve before anything runs"
            ),
        },
        proposed_commands=[command],
    )


@tool(
    "execute_approved_mutation",
    description=(
        "Run a mutation previously built by prepare_mutation, identified by its command_id. "
        "The executor raises the approval prompt and refuses anything the policy engine denies."
    ),
    capability=Capability.KUBERNETES,
    risk=RiskClass.R3,
    mutating=True,
    requires_approval=True,
    tags=("kubernetes", "mutation"),
)
async def execute_approved_mutation(
    args: ExecuteApprovedMutationArgs, ctx: ToolContext
) -> ToolResult:
    entry, command = _lookup_prepared(args, ctx)
    if entry is not None and entry.executed_at is not None:
        raise ToolError(
            f"command {command.id} has already been executed; prepare it again to re-run it",
            code="already_executed",
        )

    # The executor evaluates policy, raises the approval, waits for a human, and
    record = await _run_one(ctx, command, timeout_s=command.timeout_s)
    if entry is not None:
        entry.executed_at = time.time()

    risk = record.risk.value
    if not record.ok:
        code, message, retryable = _explain(record)
        return ToolResult(
            ok=False,
            tool="execute_approved_mutation",
            summary=f"{command.display} did not run: {message}",
            error=message,
            error_code=code,
            data={
                "command_id": command.id,
                "command": command.display,
                "outcome": record.outcome.value,
                "exit_code": record.exit_code,
                "risk": risk,
                "approval_id": record.approval_id,
                "retryable": retryable,
                "stderr": record.stderr[:1000],
            },
            evidence=[
                _evidence(
                    ctx,
                    record,
                    f"{command.display} was not executed: {message}",
                    tool_name="execute_approved_mutation",
                )
            ],
        )

    summary = (
        f"executed {risk} mutation on {command.context.cluster_context}/"
        f"{command.context.namespace}: {command.display}"
    )
    assessment = command.assessment
    return ToolResult(
        tool="execute_approved_mutation",
        summary=summary,
        data={
            "command_id": command.id,
            "command": command.display,
            "context": command.context.cluster_context,
            "namespace": command.context.namespace,
            "targets": command.context.targets,
            "risk": risk,
            "approval_id": record.approval_id,
            "approved_by": record.approved_by,
            "exit_code": record.exit_code,
            "output": record.combined_output(2000),
            "rollback": assessment.rollback_hint if assessment else None,
            "verify_with": (
                f"kubectl --context {command.context.cluster_context} "
                f"-n {command.context.namespace} get {command.context.targets[0]}"
                if command.context.targets
                else None
            ),
        },
        artifact_ref=record.artifact_ref,
        evidence=[_evidence(ctx, record, summary, tool_name="execute_approved_mutation")],
    )


def _lookup_prepared(
    args: ExecuteApprovedMutationArgs, ctx: ToolContext
) -> tuple[_PreparedMutation | None, ProposedCommand]:
    if args.command_id:
        entry = _PREPARED.get(args.command_id)
        if entry is None:
            raise ToolError(
                f"unknown prepared command: {args.command_id}; call prepare_mutation first",
                code="unknown_command",
            )
        if entry.session_id and ctx.session_id and entry.session_id != ctx.session_id:
            raise ToolError(
                "that prepared command belongs to another session",
                code="wrong_session",
            )
        return entry, entry.command

    if args.approval_id:
        broker = ctx.approvals or _executor(ctx).approvals
        request = broker.get(args.approval_id)
        if request is None:
            raise ToolError(
                f"approval {args.approval_id} is not pending; re-run prepare_mutation and use "
                "the returned command_id",
                code="unknown_approval",
            )
        entry = _PREPARED.get(request.command.id)
        return entry, request.command

    raise ToolError(
        "execute_approved_mutation needs a command_id or an approval_id",
        code="invalid_arguments",
    )
