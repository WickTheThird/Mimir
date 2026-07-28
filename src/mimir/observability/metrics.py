"""In-process telemetry (ADR 20).

MIMIR is a single-user local service, so this is a lightweight in-memory
collector rather than a Prometheus exporter. It records the things ADR 20 lists
and exposes them through ``GET /metrics`` and ``mimir doctor``.

Counters are process-lifetime. Nothing here is persisted, because the durable
audit trail lives in the database and the artifact store; these numbers exist to
answer "is it behaving" during a session.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

_WINDOW = 200


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    counters: Counter[str] = field(default_factory=Counter)
    latencies: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=_WINDOW))
    )
    _lock: Lock = field(default_factory=Lock)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self.latencies[name].append(seconds)

    def summary(self, name: str) -> dict[str, float]:
        with self._lock:
            samples = sorted(self.latencies.get(name, ()))
        if not samples:
            return {}
        count = len(samples)
        return {
            "count": count,
            "mean": round(sum(samples) / count, 4),
            "p50": round(samples[count // 2], 4),
            "p95": round(samples[min(count - 1, int(count * 0.95))], 4),
            "max": round(samples[-1], 4),
        }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.latencies.clear()
            self.started_at = time.time()


METRICS = Metrics()


# -- typed recorders ---------------------------------------------------------


def record_model_call(alias: str, latency_s: float, *, error: bool = False) -> None:
    METRICS.increment(f"model.{alias}.calls")
    if error:
        METRICS.increment(f"model.{alias}.errors")
    else:
        METRICS.observe(f"model.{alias}", latency_s)


def record_tool_call(name: str, latency_s: float, *, ok: bool) -> None:
    METRICS.increment(f"tool.{name}.calls")
    if not ok:
        METRICS.increment(f"tool.{name}.failures")
    METRICS.observe(f"tool.{name}", latency_s)


def record_command(risk: str, outcome: str, latency_s: float) -> None:
    METRICS.increment(f"command.{risk}.{outcome}")
    METRICS.observe("command", latency_s)


def record_approval(status: str, waited_s: float) -> None:
    METRICS.increment(f"approval.{status}")
    METRICS.observe("approval_wait", waited_s)


def record_specialist(name: str, latency_s: float, *, failed: bool) -> None:
    METRICS.increment(f"specialist.{name}.runs")
    if failed:
        METRICS.increment(f"specialist.{name}.failures")
    METRICS.observe(f"specialist.{name}", latency_s)


def record_retrieval(hits: int) -> None:
    METRICS.increment("retrieval.queries")
    METRICS.increment("retrieval.hits", hits)
    if hits == 0:
        METRICS.increment("retrieval.misses")


def record_web_fetch(*, ok: bool) -> None:
    METRICS.increment("web.fetches")
    if not ok:
        METRICS.increment("web.failures")


def record_session(confidence: float, evidence_count: int, duration_s: float) -> None:
    METRICS.increment("sessions.completed")
    METRICS.increment("sessions.evidence", evidence_count)
    METRICS.observe("session", duration_s)
    bucket = "high" if confidence >= 0.7 else "medium" if confidence >= 0.4 else "low"
    METRICS.increment(f"sessions.confidence.{bucket}")


def snapshot() -> dict[str, Any]:
    """Everything ADR 20 lists that is available in process."""
    from mimir.llm.router import get_router

    with METRICS._lock:
        counters = dict(METRICS.counters)
        latency_names = list(METRICS.latencies)
        uptime = time.time() - METRICS.started_at

    latencies = {name: METRICS.summary(name) for name in latency_names}

    model_calls: list[dict[str, Any]] = []
    try:
        router = get_router()
        model_calls = [
            {
                "alias": record.alias,
                "model": record.model,
                "latency_s": round(record.latency_s, 3),
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "tool_calls": record.tool_calls,
                "purpose": record.purpose,
                "error": record.error,
            }
            for record in router.call_log[-50:]
        ]
    except Exception:  # noqa: BLE001 - metrics must never raise
        pass

    retrieval_queries = counters.get("retrieval.queries", 0)
    return {
        "uptime_s": round(uptime, 1),
        "counters": counters,
        "latencies": {k: v for k, v in latencies.items() if v},
        "recent_model_calls": model_calls,
        "derived": {
            "retrieval_hit_rate": (
                round(
                    1 - counters.get("retrieval.misses", 0) / retrieval_queries,
                    3,
                )
                if retrieval_queries
                else None
            ),
            "tool_failure_rate": _rate(counters, ".calls", ".failures"),
            "model_error_rate": _rate(counters, ".calls", ".errors"),
        },
    }


def _rate(counters: dict[str, int], total_suffix: str, failure_suffix: str) -> float | None:
    total = sum(v for k, v in counters.items() if k.endswith(total_suffix))
    failures = sum(v for k, v in counters.items() if k.endswith(failure_suffix))
    return round(failures / total, 3) if total else None
