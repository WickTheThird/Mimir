"""What MIMIR is doing, read from its own records.

The database is opened read-only. The monitor watches work that is often
writing to that same database, and a dashboard that can take a write lock on
the store it observes is a dashboard that can stall the thing it is measuring.
``mode=ro`` makes that impossible rather than unlikely.

Everything reported here is derived from rows MIMIR wrote for its own purposes.
Nothing is instrumented specially for display, so the monitor cannot show a
healthy picture that the audit trail would contradict. When those two disagree
the audit trail is the one that is wrong, and that has happened: sessions and
evidence accumulated normally while the execution table stayed empty.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.config import Settings, get_settings

TERMINAL = ("completed", "failed", "cancelled", "aborted")


@dataclass
class SessionRow:
    id: str
    status: str
    task_type: str
    interface: str
    title: str
    created_at: float
    updated_at: float
    completed_at: float | None
    confidence: float | None
    error: str | None

    @property
    def running(self) -> bool:
        return self.completed_at is None and self.status not in TERMINAL

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.created_at)

    @property
    def duration_s(self) -> float:
        end = self.completed_at or time.time()
        return max(0.0, end - self.created_at)


@dataclass
class EvalRunRow:
    id: str
    suite: str
    model_alias: str
    total: int
    passed: int
    failed: int
    created_at: float
    completed_at: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contaminated(self) -> bool:
        return bool(self.metadata.get("contaminated"))

    @property
    def contamination_reason(self) -> str:
        return str(self.metadata.get("contaminated_reason", ""))

    @property
    def external_calls(self) -> int:
        try:
            return int(self.metadata.get("external_calls") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class RoleTelemetry:
    """Measured cost of one role, from model_calls rather than from guesswork.

    Before the telemetry repair this panel could not exist: the table was empty,
    so per-role latency and token cost were unknowable and the only visible
    number was wall-clock for the whole investigation.
    """

    role: str
    model: str
    calls: int = 0
    failed: int = 0
    retries: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    @property
    def share_of_time(self) -> float:
        return 0.0  # filled in by the caller against the total


@dataclass
class TelemetryHealth:
    """Whether the audit and telemetry trails agree with observed activity."""

    sessions_checked: int = 0
    sessions_without_calls: int = 0
    model_calls_total: int = 0
    orphaned_rows: int = 0
    complete: bool = True
    instrumented_since: float | None = None
    """Timestamp of the oldest telemetry row.

    Sessions older than this predate the instrumentation, so their lack of
    telemetry is expected and must not be reported as a defect.
    """

    @property
    def summary(self) -> str:
        if not self.sessions_checked:
            return "no recent sessions"
        if self.sessions_without_calls:
            return (
                f"{self.sessions_without_calls}/{self.sessions_checked} recent "
                "sessions recorded no model calls"
            )
        return f"{self.sessions_checked}/{self.sessions_checked} recent sessions instrumented"


@dataclass
class InFlightEval:
    """Progress of a run that has not been persisted yet.

    ``eval_runs`` gets one row when the run finishes, so during the hour a model
    suite takes there is nothing in it to read. Progress is therefore counted
    from the per-case sessions the harness writes as it goes, bounded to those
    created after the evaluate process started.

    ``total`` is the number of model cases in the corpus. It is a separate count
    from the corpus size, because deterministic cases open no session and
    including them would make the run look permanently stalled at 40%.
    """

    pid: int
    started_at: float
    completed: int = 0
    total: int = 0
    last_case_at: float | None = None
    mean_case_s: float | None = None

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def fraction(self) -> float:
        return min(1.0, self.completed / self.total) if self.total else 0.0

    @property
    def eta_s(self) -> float | None:
        if not self.total or self.completed < 2 or self.mean_case_s is None:
            return None
        return max(0.0, (self.total - self.completed) * self.mean_case_s)

    @property
    def idle_s(self) -> float | None:
        if self.last_case_at is None:
            return None
        return max(0.0, time.time() - self.last_case_at)


@dataclass
class Activity:
    db_path: str = ""
    readable: bool = True
    error: str = ""
    sessions: list[SessionRow] = field(default_factory=list)
    running: list[SessionRow] = field(default_factory=list)
    latest_run: EvalRunRow | None = None
    in_flight: InFlightEval | None = None
    roles: list[RoleTelemetry] = field(default_factory=list)
    telemetry: TelemetryHealth = field(default_factory=TelemetryHealth)
    sessions_last_hour: int = 0
    evidence_total: int = 0
    executions_total: int = 0
    approvals_pending: int = 0
    model_calls_total: int = 0

    @property
    def audit_gap(self) -> str:
        """A one-line warning when the audit trail contradicts the activity.

        Not decoration. The executions table sat at zero across seventeen
        sessions because nothing populated it, and the only visible symptom was
        a number nobody was looking at.
        """
        if self.sessions and self.model_calls_total == 0:
            return "model_calls empty: per-call telemetry is not recorded"
        if self.telemetry.sessions_without_calls:
            return self.telemetry.summary
        if self.telemetry.orphaned_rows:
            return f"{self.telemetry.orphaned_rows} model_calls rows have no session"
        return ""

    def throughput_per_min(self, window_s: float = 600.0) -> float | None:
        cutoff = time.time() - window_s
        done = [s for s in self.sessions if s.completed_at and s.completed_at >= cutoff]
        if len(done) < 2:
            return None
        span = max(1.0, time.time() - min(s.completed_at or 0.0 for s in done))
        return len(done) * 60.0 / span


def _connect(settings: Settings) -> sqlite3.Connection:
    path = settings.home / "mimir.db"
    if not path.exists():
        raise FileNotFoundError(f"no database at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)


def _json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _count(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    try:
        clause = f" where {where}" if where else ""
        row = conn.execute(f"select count(*) from {table}{clause}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def collect(settings: Settings | None = None, *, limit: int = 12) -> Activity:
    """One read-only sweep of the activity tables."""
    active = settings or get_settings()
    out = Activity(db_path=str(active.home / "mimir.db"))

    try:
        conn = _connect(active)
    except (FileNotFoundError, sqlite3.Error) as exc:
        out.readable = False
        out.error = str(exc)
        return out

    try:
        rows = conn.execute(
            "select id, status, task_type, interface, title, created_at, "
            "updated_at, completed_at, final_confidence, error "
            "from sessions order by created_at desc limit ?",
            (max(limit, 40),),
        ).fetchall()
        out.sessions = [
            SessionRow(
                id=r[0] or "",
                status=r[1] or "",
                task_type=r[2] or "",
                interface=r[3] or "",
                title=r[4] or "",
                created_at=float(r[5] or 0.0),
                updated_at=float(r[6] or 0.0),
                completed_at=float(r[7]) if r[7] else None,
                confidence=float(r[8]) if r[8] is not None else None,
                error=r[9],
            )
            for r in rows
        ]
        out.running = [s for s in out.sessions if s.running]

        # A missing or older table degrades this one panel. It previously made
        # collect() report the entire store unreadable, so one absent table
        # blanked the whole dashboard.
        try:
            run = conn.execute(
                "select id, suite, model_alias, total, passed, failed, created_at, "
                "completed_at, metadata_json from eval_runs order by created_at desc limit 1"
            ).fetchone()
        except sqlite3.Error:
            run = None
        if run:
            out.latest_run = EvalRunRow(
                id=run[0] or "",
                suite=run[1] or "",
                model_alias=run[2] or "",
                total=int(run[3] or 0),
                passed=int(run[4] or 0),
                failed=int(run[5] or 0),
                created_at=float(run[6] or 0.0),
                completed_at=float(run[7]) if run[7] else None,
                metadata=_json(run[8]),
            )

        _collect_telemetry(conn, out)

        hour_ago = time.time() - 3600
        out.sessions_last_hour = _count(conn, "sessions", f"created_at >= {hour_ago}")
        out.evidence_total = _count(conn, "evidence")
        out.executions_total = _count(conn, "executions")
        out.model_calls_total = _count(conn, "model_calls")
        out.approvals_pending = _count(conn, "approvals", "status = 'pending'")
    except sqlite3.Error as exc:
        out.readable = False
        out.error = str(exc)
    finally:
        conn.close()

    out.sessions = out.sessions[:limit]
    return out


def _collect_telemetry(conn: sqlite3.Connection, out: Activity) -> None:
    """Per-role cost, measured over the recent window rather than all time.

    Bounded to the last hour so the figures describe what is happening now. An
    all-time mean would be dominated by whatever model was configured longest
    ago, which is the opposite of what a live monitor is for.
    """
    since = time.time() - 3600
    try:
        rows = conn.execute(
            "select task_class, model, count(*), sum(latency_ms), "
            "sum(coalesce(prompt_tokens,0)), sum(coalesce(completion_tokens,0)), "
            "sum(case when ok then 0 else 1 end), sum(retries) "
            "from model_calls where created_at >= ? "
            "group by task_class, model order by sum(latency_ms) desc",
            (since,),
        ).fetchall()
    except sqlite3.Error:
        return

    out.roles = [
        RoleTelemetry(
            role=r[0] or "(unattributed)",
            model=r[1] or "",
            calls=int(r[2] or 0),
            total_latency_ms=float(r[3] or 0.0),
            prompt_tokens=int(r[4] or 0),
            completion_tokens=int(r[5] or 0),
            failed=int(r[6] or 0),
            retries=int(r[7] or 0),
        )
        for r in rows
    ]

    health = TelemetryHealth()
    try:
        health.model_calls_total = _count(conn, "model_calls")
        # Only sessions from after telemetry started being recorded can be
        # judged on whether they recorded any. Counting older ones reported
        # "17/20 sessions recorded no model calls" at a moment when every
        # session since the fix had recorded them correctly: history rendered
        # as a present fault, which is the failure this monitor exists to avoid.
        first = conn.execute("select min(created_at) from model_calls").fetchone()
        floor = float(first[0]) if first and first[0] else None
        if floor is None:
            health.complete = health.model_calls_total > 0
            out.telemetry = health
            return
        health.instrumented_since = floor
        window = max(since, floor)
        recent = conn.execute(
            "select s.id, count(m.row_id) from sessions s "
            "left join model_calls m on m.session_id = s.id "
            "where s.created_at >= ? group by s.id",
            (window,),
        ).fetchall()
        health.sessions_checked = len(recent)
        health.sessions_without_calls = sum(1 for _, n in recent if not n)
        health.orphaned_rows = conn.execute(
            "select count(*) from model_calls m "
            "left join sessions s on m.session_id = s.id where s.id is null"
        ).fetchone()[0]
    except sqlite3.Error:
        return
    health.complete = not health.sessions_without_calls and not health.orphaned_rows
    out.telemetry = health


def tail_log(path: Path | None, *, lines: int = 8, max_bytes: int = 200_000) -> list[str]:
    """Last lines of a log file, read from the end.

    Bounded so that pointing the monitor at a multi-gigabyte log does not read
    the whole thing into memory once a second.
    """
    if path is None or not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()
            content = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [line for line in content.splitlines() if line.strip()][-lines:]


def in_flight_eval(
    pid: int, started_at: float, total: int, settings: Settings | None = None
) -> InFlightEval:
    """Count completed cases for a running evaluation.

    Sessions are attributed by creation time rather than by any run id, because
    the harness does not mint one until it persists. Bounding on the process
    start is what keeps a previous run's sessions out of this one's count.
    """
    active = settings or get_settings()
    out = InFlightEval(pid=pid, started_at=started_at, total=total)
    try:
        conn = _connect(active)
    except (FileNotFoundError, sqlite3.Error):
        return out
    try:
        rows = conn.execute(
            "select created_at, completed_at from sessions "
            "where interface = 'eval' and created_at >= ? order by created_at",
            (started_at,),
        ).fetchall()
    except sqlite3.Error:
        return out
    finally:
        conn.close()

    durations = [
        float(end) - float(start) for start, end in rows if start and end
    ]
    out.completed = len(rows)
    if rows:
        stamps = [float(r[1]) for r in rows if r[1]]
        out.last_case_at = max(stamps) if stamps else None
    if durations:
        out.mean_case_s = sum(durations) / len(durations)
    return out
