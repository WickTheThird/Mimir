"""Log analysis helpers (ADR 9.5, 5.6, goal G4).

These helpers are the analytical core of log and timeout diagnosis. They never
hand raw log volume back to the model (ADR R7 context explosion): logs live in
the artifact store and are addressed by ``input_ref``; every helper returns a
compact structured summary plus :class:`~mimir.models.evidence.Evidence` items
that cite line numbers inside the artifact.

Pipeline:

* :data:`ingest_logs`             - accept pasted text (ADR 5.6) -> ``input_ref``.
* :data:`filter_logs`             - narrow by pattern, level, and time window.
* :data:`group_repeated_errors`   - collapse variable parts into templates.
* :data:`extract_correlation_ids` - find trace/request ids by key and by shape.
* :data:`correlate_logs`          - merge several artifacts onto one timeline.
* :data:`detect_timeout_patterns` - explicit timeouts, round-number duration
  clusters, retry amplification, pool exhaustion, restart loops.
* :data:`compare_before_after`    - error-group frequency diff.
* :data:`summarise_log_volume`    - histogram, level mix, top talkers, bursts.

The line parser is configuration-free by design. It recognises ISO8601/RFC3339,
Go standard library, klog, syslog, common-log-format and epoch timestamps, JSON
lines, logfmt key=value pairs, zap-style tab separated fields, and bracketed
levels. Java and Python stack-trace continuation lines are attached to the entry
above them. Lines that match nothing are kept as unparsed entries rather than
dropped, because an unparseable line is often the interesting one.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, Field

from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.models.evidence import Citation, Evidence, EvidenceKind, Freshness, SourceType
from mimir.safety.injection import wrap_untrusted
from mimir.tools.artifacts import Artifact, ArtifactStore, get_artifact_store
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool

log = get_logger(__name__)

#: Hard caps so a helper can never blow up the model context (ADR R7).
MAX_PARSE_LINES = 400_000
MAX_SAMPLE_LINES = 20
MAX_LINE_CHARS = 300
MAX_GROUPS = 25
MAX_EVIDENCE = 8
MAX_TIMELINE_ROWS = 40
PREVIEW_LINES = 25


# ---------------------------------------------------------------------------
# parsed entry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LogEntry:
    """One normalised log record. ``line_no`` is 1-based into the artifact."""

    line_no: int
    raw: str
    message: str
    timestamp: float | None = None
    level: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    continuation: str = ""
    """Stack-trace or wrapped lines that belong to this entry."""

    parsed: bool = True
    """False when no timestamp, level, or structured field could be recovered."""

    @property
    def end_line(self) -> int:
        return self.line_no + (self.continuation.count("\n") + 1 if self.continuation else 0)

    def searchable(self) -> str:
        return self.raw if not self.continuation else f"{self.raw}\n{self.continuation}"


# ---------------------------------------------------------------------------
# timestamp parsing
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_CORE = (
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|z|[+-]\d{2}:?\d{2})?"
)

_TS_ISO = re.compile(rf"^\s*\[?(?P<ts>{_ISO_CORE})\]?[\s,|]*")
_TS_GO = re.compile(r"^\s*(?P<ts>\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?)\s*")
_TS_KLOG = re.compile(r"^(?P<lvl>[IWEF])(?P<ts>\d{4} \d{2}:\d{2}:\d{2}\.\d{6})\s+")
_TS_SYSLOG = re.compile(
    r"^\s*(?P<ts>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+"
    r"\d{2}:\d{2}:\d{2})\s+"
)
_TS_CLF = re.compile(
    r"^\s*\S+\s+\S+\s+\S+\s+\[(?P<ts>\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4})\]\s*"
)
_TS_EPOCH = re.compile(r"^\s*\[?(?P<ts>1\d{9}(?:\.\d{1,9})?|1\d{12})\]?\s+")
_ISO_ANYWHERE = re.compile(rf"(?P<ts>{_ISO_CORE})")

_KLOG_LEVELS = {"I": "INFO", "W": "WARN", "E": "ERROR", "F": "FATAL"}


def _iso_to_epoch(token: str) -> float | None:
    text = token.strip().replace(",", ".")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    text = text.replace(" ", "T", 1)
    # fromisoformat wants exactly 3 or 6 fractional digits.
    match = re.search(r"\.(\d+)", text)
    if match and len(match.group(1)) not in (3, 6):
        digits = (match.group(1) + "000000")[:6]
        text = text[: match.start()] + "." + digits + text[match.end() :]
    # +0000 without a colon is accepted by 3.11+, but normalise for safety.
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _partial_to_epoch(month: int, day: int, clock: str, micros: int = 0) -> float | None:
    """Syslog and klog omit the year. Assume the current year."""
    try:
        hour, minute, second = (int(p) for p in clock.split(":"))
        now = datetime.now(UTC)
        stamp = datetime(
            now.year, month, day, hour, minute, second, micros, tzinfo=UTC
        )
    except ValueError:
        return None
    return stamp.timestamp()


def _strip_timestamp(line: str) -> tuple[float | None, str, str | None]:
    """Return ``(epoch, remainder, level_hint)`` for a leading timestamp."""
    match = _TS_ISO.match(line)
    if match:
        return _iso_to_epoch(match.group("ts")), line[match.end() :], None

    match = _TS_GO.match(line)
    if match:
        token = match.group("ts").replace("/", "-", 2)
        return _iso_to_epoch(token), line[match.end() :], None

    match = _TS_KLOG.match(line)
    if match:
        raw = match.group("ts")
        micros = int(raw.split(".")[1])
        epoch = _partial_to_epoch(int(raw[:2]), int(raw[2:4]), raw[5:13], micros)
        return epoch, line[match.end() :], _KLOG_LEVELS.get(match.group("lvl"))

    match = _TS_SYSLOG.match(line)
    if match:
        parts = match.group("ts").split()
        epoch = _partial_to_epoch(_MONTHS[parts[0].lower()], int(parts[1]), parts[2])
        return epoch, line[match.end() :], None

    match = _TS_CLF.match(line)
    if match:
        try:
            stamp = datetime.strptime(match.group("ts"), "%d/%b/%Y:%H:%M:%S %z")
            return stamp.timestamp(), line[match.end() :], None
        except ValueError:
            return None, line, None

    match = _TS_EPOCH.match(line)
    if match:
        value = float(match.group("ts"))
        if value > 1e11:  # milliseconds
            value /= 1000.0
        return value, line[match.end() :], None

    return None, line, None


# ---------------------------------------------------------------------------
# level and field parsing
# ---------------------------------------------------------------------------

_LEVEL_CANON = {
    "TRACE": "TRACE", "TRC": "TRACE", "VERBOSE": "TRACE",
    "DEBUG": "DEBUG", "DBG": "DEBUG", "DEBU": "DEBUG",
    "INFO": "INFO", "INF": "INFO", "INFORMATION": "INFO", "NOTICE": "INFO",
    "WARN": "WARN", "WARNING": "WARN", "WRN": "WARN",
    "ERROR": "ERROR", "ERR": "ERROR", "SEVERE": "ERROR", "EROR": "ERROR",
    "FATAL": "FATAL", "CRITICAL": "FATAL", "CRIT": "FATAL", "PANIC": "FATAL",
    "EMERG": "FATAL", "ALERT": "FATAL", "DPANIC": "FATAL",
}
LEVEL_ORDER = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
_LEVEL_RANK = {name: index for index, name in enumerate(LEVEL_ORDER)}

_BRACKET_LEVEL = re.compile(r"^\s*[\[<(]\s*([A-Za-z]{3,11})\s*[\]>)][\s:|-]*")
_BARE_LEVEL = re.compile(r"^\s*([A-Za-z]{3,11})\s*[:\t|]\s*")
_TAB_LEVEL = re.compile(r"^\s*([A-Za-z]{3,11})\t")
#: A level followed only by whitespace, as emitted by Go's slog, zap's console
#: encoder, and most logfmt writers ("... ERROR checkout msg=..."). Matching a
#: bare word is only safe because _canon_level rejects anything that is not a
#: known level name, so an ordinary first word does not become a level.
_SPACED_LEVEL = re.compile(r"^\s*([A-Za-z]{3,11})\s+")

_KV_RE = re.compile(
    r"""(?P<k>[A-Za-z_][A-Za-z0-9_.\-]{0,48})=(?P<v>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s,;]*)"""
)

_JSON_TS_KEYS = ("ts", "time", "timestamp", "@timestamp", "eventTime", "t", "asctime")
_JSON_LEVEL_KEYS = ("level", "lvl", "severity", "levelname", "log.level", "loglevel")
_JSON_MSG_KEYS = ("msg", "message", "event", "log", "text", "@message", "short_message")


def _canon_level(token: str | None) -> str | None:
    if not token:
        return None
    return _LEVEL_CANON.get(token.strip().upper())


def _strip_level(line: str) -> tuple[str | None, str]:
    for pattern in (_BRACKET_LEVEL, _TAB_LEVEL, _BARE_LEVEL, _SPACED_LEVEL):
        match = pattern.match(line)
        if match:
            level = _canon_level(match.group(1))
            if level:
                return level, line[match.end() :]
    return None, line


def _coerce_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float) or value is None:
        return str(value)
    return json.dumps(value, default=str)[:400]


def _flatten(obj: dict[str, Any], prefix: str = "", depth: int = 2) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth > 0:
            out.update(_flatten(value, prefix=f"{name}.", depth=depth - 1))
        else:
            out[name] = _coerce_scalar(value)
        if len(out) > 64:
            break
    return out


def _json_timestamp(value: str) -> float | None:
    epoch = _iso_to_epoch(value)
    if epoch is not None:
        return epoch
    try:
        number = float(value)
    except ValueError:
        return None
    if number > 1e14:  # microseconds
        return number / 1_000_000.0
    if number > 1e11:  # milliseconds
        return number / 1000.0
    return number


def _parse_json_line(line: str, line_no: int) -> LogEntry | None:
    stripped = line.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    fields = _flatten(payload)
    timestamp = None
    for key in _JSON_TS_KEYS:
        if key in fields:
            timestamp = _json_timestamp(fields[key])
            if timestamp is not None:
                break
    level = None
    for key in _JSON_LEVEL_KEYS:
        level = _canon_level(fields.get(key))
        if level:
            break
    message = ""
    for key in _JSON_MSG_KEYS:
        if fields.get(key):
            message = fields[key]
            break
    if not message:
        skip = set(_JSON_TS_KEYS) | set(_JSON_LEVEL_KEYS)
        message = " ".join(f"{k}={v}" for k, v in fields.items() if k not in skip)[:600]
    if fields.get("error") and fields["error"] not in message:
        message = f"{message} error={fields['error']}"
    return LogEntry(
        line_no=line_no,
        raw=line,
        message=message.strip(),
        timestamp=timestamp,
        level=level,
        fields=fields,
    )


_CONTINUATION_RE = re.compile(
    r"^(?:\s+at\s|\s*Caused by:|\s*\.\.\.\s*\d+\s+more\b|\s*Suppressed:"
    r"|Traceback \(most recent call last\):|\s+File \"|\s+\w+Error\b|\s*}\s*$)"
)


def _looks_like_continuation(line: str) -> bool:
    if not line:
        return False
    if _CONTINUATION_RE.match(line):
        return True
    # Indented lines with no timestamp of their own belong to the entry above.
    return bool(line[:1] in (" ", "\t") and line.strip())


def _parse_line(line: str, line_no: int) -> LogEntry:
    entry = _parse_json_line(line, line_no)
    if entry is not None:
        return entry

    timestamp, remainder, level_hint = _strip_timestamp(line)
    level, remainder = _strip_level(remainder)
    level = level or level_hint

    fields: dict[str, str] = {}
    for match in _KV_RE.finditer(line):
        value = match.group("v")
        if value[:1] in ("\"", "'") and len(value) >= 2:
            value = value[1:-1]
        fields[match.group("k")] = value
        if len(fields) >= 40:
            break

    if timestamp is None:
        for key in _JSON_TS_KEYS:
            if key in fields:
                timestamp = _json_timestamp(fields[key])
                if timestamp is not None:
                    break
    if timestamp is None:
        found = _ISO_ANYWHERE.search(line[:200])
        if found:
            timestamp = _iso_to_epoch(found.group("ts"))
    if level is None:
        for key in _JSON_LEVEL_KEYS:
            level = _canon_level(fields.get(key))
            if level:
                break

    message = remainder.strip()
    for key in _JSON_MSG_KEYS:
        if fields.get(key):
            message = fields[key]
            break
    if not message:
        message = line.strip()

    return LogEntry(
        line_no=line_no,
        raw=line,
        message=message,
        timestamp=timestamp,
        level=level,
        fields=fields,
        parsed=bool(timestamp is not None or level is not None or fields),
    )


def parse_log_text(text: str, *, max_lines: int = MAX_PARSE_LINES) -> tuple[list[LogEntry], bool]:
    """Parse raw log text into entries. Returns ``(entries, truncated)``."""
    entries: list[LogEntry] = []
    truncated = False
    for index, line in enumerate(text.splitlines(), start=1):
        if index > max_lines:
            truncated = True
            break
        if not line.strip():
            continue
        if entries and _looks_like_continuation(line):
            previous = entries[-1]
            if previous.continuation:
                previous.continuation += "\n" + line
            else:
                previous.continuation = line
            continue
        entries.append(_parse_line(line, index))
    return entries, truncated


# ---------------------------------------------------------------------------
# message templating
# ---------------------------------------------------------------------------

_MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(rf"{_ISO_CORE}"), "<ts>"),
    (re.compile(r"\bhttps?://[^\s\"'<>]+"), "<url>"),
    (re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)*-[a-z0-9]{8,10}-[a-z0-9]{5}\b"), "<pod>"),
    (re.compile(r'"(?:[^"\\]|\\.)*"'), "<str>"),
    (re.compile(r"'(?:[^'\\]|\\.)*'"), "<str>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b"), "<addr>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<hex>"),
    (re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:ns|µs|us|ms|s|sec|secs|seconds|m|min|h)\b"),
     "<dur>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<hex>"),
    (re.compile(r"(?<![\w.])\d+(?:\.\d+)?\b"), "<num>"),
]

_WS_RE = re.compile(r"\s+")


def template_of(message: str, *, limit: int = 220) -> str:
    """Mask variable parts so near-identical messages collapse into one group.

    ``timeout after 30001ms`` and ``timeout after 29997ms`` both become
    ``timeout after <dur>``.
    """
    text = message
    for pattern, replacement in _MASKS:
        text = pattern.sub(replacement, text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:limit]


# ---------------------------------------------------------------------------
# artifact access
# ---------------------------------------------------------------------------

_PARSE_CACHE: dict[str, tuple[list[LogEntry], bool]] = {}
_PARSE_CACHE_MAX = 8


def _store(ctx: ToolContext) -> ArtifactStore:
    store = ctx.artifacts
    if store is None:
        store = get_artifact_store(ctx.settings)
    return store


def _artifact(ctx: ToolContext, ref: str) -> Artifact:
    try:
        return _store(ctx).require(ref)
    except KeyError as exc:
        raise ToolError(str(exc), code="unknown_artifact") from exc


def _entries(ctx: ToolContext, ref: str) -> tuple[Artifact, list[LogEntry], bool]:
    artifact = _artifact(ctx, ref)
    cached = _PARSE_CACHE.get(ref)
    if cached is None:
        cached = parse_log_text(artifact.read())
        if len(_PARSE_CACHE) >= _PARSE_CACHE_MAX:
            _PARSE_CACHE.pop(next(iter(_PARSE_CACHE)))
        _PARSE_CACHE[ref] = cached
    return artifact, cached[0], cached[1]


def _source_type(artifact: Artifact) -> SourceType:
    raw = str(artifact.metadata.get("source_type", ""))
    try:
        return SourceType(raw)
    except ValueError:
        return SourceType.COMMAND_OUTPUT


def _label(artifact: Artifact) -> str:
    return str(artifact.metadata.get("label") or artifact.metadata.get("command") or artifact.ref)


def _clip(text: str, limit: int = MAX_LINE_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="milliseconds")


def _evidence(
    *,
    claim: str,
    artifact: Artifact,
    lines: list[int],
    excerpt: str,
    tool_name: str,
    kind: EvidenceKind = EvidenceKind.OBSERVED,
    confidence: float = 0.8,
    structured: dict[str, Any] | None = None,
) -> Evidence:
    ordered = sorted(lines)[:12]
    source_type = _source_type(artifact)
    citations = [
        Citation(
            source_type=source_type,
            locator=f"{artifact.ref}:{line}",
            path=_label(artifact),
            start_line=line,
            end_line=line,
            retrieved_at=artifact.created_at,
        )
        for line in ordered
    ]
    return Evidence(
        claim=claim,
        kind=kind,
        source_type=source_type,
        source_id=artifact.ref,
        excerpt=_clip(excerpt, 600),
        citations=citations,
        collected_at=artifact.created_at,
        freshness=Freshness.LIVE,
        confidence=confidence,
        collected_by=tool_name,
        artifact_ref=artifact.ref,
        structured={"line_numbers": ordered, **(structured or {})},
    )


def _preview(artifact: Artifact, entries: list[LogEntry], count: int = PREVIEW_LINES) -> str:
    body = "\n".join(f"{e.line_no}: {_clip(e.raw)}" for e in entries[:count])
    return wrap_untrusted(
        body,
        source_type=_source_type(artifact),
        source_id=f"{_label(artifact)} ({artifact.ref})",
        note="log lines, prefixed with their line number in the artifact",
    )


def _span(entries: list[LogEntry]) -> tuple[float | None, float | None]:
    stamps = [e.timestamp for e in entries if e.timestamp is not None]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


# ---------------------------------------------------------------------------
# ingest_logs
# ---------------------------------------------------------------------------


class IngestLogsInput(BaseModel):
    text: str = Field(description="Raw log text, typically pasted by the operator.")
    label: str = Field(default="", description="Short human label, e.g. 'checkout-api pod logs'.")
    origin: Literal["pasted", "command_output", "file", "kubectl", "container"] = "pasted"
    service: str = Field(default="", description="Optional service name for correlation output.")


_ORIGIN_SOURCE = {
    "pasted": SourceType.USER_PROVIDED,
    "command_output": SourceType.COMMAND_OUTPUT,
    "file": SourceType.COMMAND_OUTPUT,
    "kubectl": SourceType.COMMAND_OUTPUT,
    "container": SourceType.COMMAND_OUTPUT,
}


@tool(
    name="ingest_logs",
    description=(
        "Store pasted or captured log text in the artifact store and return an input_ref. "
        "Every other log helper takes that ref, so raw log volume never enters the model "
        "context. Returns line counts, the observed time span, and level distribution."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "ingest"),
)
async def ingest_logs(args: IngestLogsInput, ctx: ToolContext) -> ToolResult:
    if not args.text.strip():
        raise ToolError("no log text supplied", code="empty_input")

    source_type = _ORIGIN_SOURCE[args.origin]
    artifact = _store(ctx).put(
        args.text,
        kind="logs",
        session_id=ctx.session_id,
        metadata={
            "label": args.label or f"{args.origin} logs",
            "origin": args.origin,
            "service": args.service,
            "source_type": source_type.value,
        },
    )
    entries, truncated = parse_log_text(artifact.read())
    _PARSE_CACHE[artifact.ref] = (entries, truncated)

    first, last = _span(entries)
    levels = Counter(e.level or "UNKNOWN" for e in entries)
    unparsed = sum(1 for e in entries if not e.parsed)
    summary = (
        f"stored {len(entries)} entries as {artifact.ref} "
        f"({artifact.size_bytes} bytes, {unparsed} unparsed)"
    )
    return ToolResult(
        tool="ingest_logs",
        summary=summary,
        artifact_ref=artifact.ref,
        truncated=truncated,
        data={
            "input_ref": artifact.ref,
            "label": args.label or f"{args.origin} logs",
            "entries": len(entries),
            "unparsed_entries": unparsed,
            "bytes": artifact.size_bytes,
            "first_timestamp": _iso(first),
            "last_timestamp": _iso(last),
            "levels": dict(levels.most_common()),
            "preview": _preview(artifact, entries, 10),
        },
    )


# ---------------------------------------------------------------------------
# filter_logs
# ---------------------------------------------------------------------------


def _parse_time(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    epoch = _iso_to_epoch(text)
    if epoch is not None:
        return epoch
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 1000.0 if number > 1e11 else number


def _compile(patterns: list[str], case_sensitive: bool) -> list[re.Pattern[str]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    out = []
    for raw in patterns:
        try:
            out.append(re.compile(raw, flags))
        except re.error as exc:
            raise ToolError(f"invalid regex {raw!r}: {exc}", code="invalid_pattern") from exc
    return out


class FilterLogsInput(BaseModel):
    input_ref: str
    patterns: list[str] = Field(
        default_factory=list, description="Regexes; an entry matching any of them is kept."
    )
    exclude_patterns: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list, description="e.g. ['ERROR','FATAL'].")
    min_level: str = Field(default="", description="Keep this level and anything above it.")
    since: str = Field(default="", description="ISO8601 or epoch seconds, inclusive.")
    until: str = Field(default="", description="ISO8601 or epoch seconds, inclusive.")
    case_sensitive: bool = False
    max_samples: int = Field(default=MAX_SAMPLE_LINES, ge=1, le=100)


@tool(
    name="filter_logs",
    description=(
        "Filter a stored log artifact by regex, level, and time window. Writes the matching "
        "lines to a new artifact and returns only counts plus a small sample, so large "
        "result sets stay out of the model context."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "filter"),
)
async def filter_logs(args: FilterLogsInput, ctx: ToolContext) -> ToolResult:
    artifact, entries, truncated = _entries(ctx, args.input_ref)
    includes = _compile(args.patterns, args.case_sensitive)
    excludes = _compile(args.exclude_patterns, args.case_sensitive)
    wanted = {lvl.strip().upper() for lvl in args.levels if lvl.strip()}
    floor = _LEVEL_RANK.get(args.min_level.strip().upper(), -1) if args.min_level else -1
    since = _parse_time(args.since)
    until = _parse_time(args.until)
    if args.since and since is None:
        raise ToolError(f"could not parse since={args.since!r}", code="invalid_time")
    if args.until and until is None:
        raise ToolError(f"could not parse until={args.until!r}", code="invalid_time")

    matched: list[LogEntry] = []
    for entry in entries:
        if since is not None and (entry.timestamp is None or entry.timestamp < since):
            continue
        if until is not None and (entry.timestamp is None or entry.timestamp > until):
            continue
        if wanted and (entry.level or "UNKNOWN") not in wanted:
            continue
        if floor >= 0 and _LEVEL_RANK.get(entry.level or "", -1) < floor:
            continue
        haystack = entry.searchable()
        if includes and not any(p.search(haystack) for p in includes):
            continue
        if excludes and any(p.search(haystack) for p in excludes):
            continue
        matched.append(entry)

    body = "\n".join(entry.searchable() for entry in matched)
    out_ref = None
    if body:
        out = _store(ctx).put(
            body,
            kind="logs",
            session_id=ctx.session_id,
            metadata={
                "label": f"filtered({_label(artifact)})",
                "source_type": _source_type(artifact).value,
                "derived_from": artifact.ref,
                "filters": {
                    "patterns": args.patterns,
                    "exclude_patterns": args.exclude_patterns,
                    "levels": sorted(wanted),
                    "min_level": args.min_level,
                    "since": args.since,
                    "until": args.until,
                },
            },
        )
        out_ref = out.ref

    first, last = _span(matched)
    sample = [
        {"line": e.line_no, "ts": _iso(e.timestamp), "level": e.level, "text": _clip(e.message)}
        for e in matched[: args.max_samples]
    ]
    evidence: list[Evidence] = []
    if matched:
        evidence.append(
            _evidence(
                claim=f"{len(matched)} of {len(entries)} log entries match the filter",
                artifact=artifact,
                lines=[e.line_no for e in matched[:12]],
                excerpt="\n".join(_clip(e.raw) for e in matched[:3]),
                tool_name="filter_logs",
                structured={"matched": len(matched), "scanned": len(entries)},
            )
        )
    return ToolResult(
        tool="filter_logs",
        summary=f"{len(matched)}/{len(entries)} entries matched; result stored as {out_ref}",
        artifact_ref=out_ref,
        truncated=truncated or len(matched) > args.max_samples,
        evidence=evidence,
        data={
            "input_ref": args.input_ref,
            "output_ref": out_ref,
            "matched": len(matched),
            "scanned": len(entries),
            "first_match_ts": _iso(first),
            "last_match_ts": _iso(last),
            "levels": dict(Counter(e.level or "UNKNOWN" for e in matched).most_common()),
            "sample": sample,
        },
    )


# ---------------------------------------------------------------------------
# group_repeated_errors
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MessageGroup:
    template: str
    count: int = 0
    first_seen: float | None = None
    last_seen: float | None = None
    first_line: int = 0
    last_line: int = 0
    example: str = ""
    lines: list[int] = field(default_factory=list)
    levels: Counter[str] = field(default_factory=Counter)

    def observe(self, entry: LogEntry) -> None:
        self.count += 1
        self.levels[entry.level or "UNKNOWN"] += 1
        if len(self.lines) < 50:
            self.lines.append(entry.line_no)
        if not self.example:
            self.example = entry.message
            self.first_line = entry.line_no
        self.last_line = entry.line_no
        if entry.timestamp is not None:
            if self.first_seen is None or entry.timestamp < self.first_seen:
                self.first_seen = entry.timestamp
            if self.last_seen is None or entry.timestamp > self.last_seen:
                self.last_seen = entry.timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "count": self.count,
            "levels": dict(self.levels.most_common()),
            "first_seen": _iso(self.first_seen),
            "last_seen": _iso(self.last_seen),
            "first_line": self.first_line,
            "last_line": self.last_line,
            "example": _clip(self.example),
            "example_lines": self.lines[:8],
        }


def build_groups(
    entries: list[LogEntry], *, min_level: str = "", include_all: bool = False
) -> list[MessageGroup]:
    floor = _LEVEL_RANK.get(min_level.upper(), -1) if min_level else -1
    groups: dict[str, MessageGroup] = {}
    for entry in entries:
        rank = _LEVEL_RANK.get(entry.level or "", -1)
        if not include_all and floor >= 0 and rank < floor:
            continue
        key = template_of(entry.message)
        group = groups.get(key)
        if group is None:
            group = groups[key] = MessageGroup(template=key)
        group.observe(entry)
    return sorted(groups.values(), key=lambda g: (-g.count, g.first_line))


class GroupErrorsInput(BaseModel):
    input_ref: str
    min_level: str = Field(
        default="WARN", description="Only group this level and above; '' groups everything."
    )
    top_n: int = Field(default=MAX_GROUPS, ge=1, le=100)


@tool(
    name="group_repeated_errors",
    description=(
        "Collapse repeated log messages into templates by masking variable parts (UUIDs, hex "
        "ids, addresses, durations, numbers, quoted strings), so 'timeout after 30001ms' and "
        "'timeout after 29997ms' become one group. Returns groups ranked by count with first "
        "and last seen, an example, and citable line numbers."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "grouping"),
)
async def group_repeated_errors(args: GroupErrorsInput, ctx: ToolContext) -> ToolResult:
    artifact, entries, truncated = _entries(ctx, args.input_ref)
    groups = build_groups(entries, min_level=args.min_level)
    if not groups:
        return ToolResult(
            tool="group_repeated_errors",
            summary=f"no entries at or above {args.min_level or 'any level'} in {args.input_ref}",
            data={"input_ref": args.input_ref, "groups": [], "scanned": len(entries)},
        )

    shown = groups[: args.top_n]
    total = sum(g.count for g in groups)
    evidence = [
        _evidence(
            claim=f"{g.count}x '{_clip(g.template, 120)}'",
            artifact=artifact,
            lines=g.lines,
            excerpt=g.example,
            tool_name="group_repeated_errors",
            structured={"template": g.template, "count": g.count},
            confidence=0.85,
        )
        for g in shown[:MAX_EVIDENCE]
    ]
    top = shown[0]
    return ToolResult(
        tool="group_repeated_errors",
        summary=(
            f"{len(groups)} distinct templates over {total} entries; "
            f"top: {top.count}x '{_clip(top.template, 100)}'"
        ),
        truncated=truncated or len(groups) > args.top_n,
        evidence=evidence,
        data={
            "input_ref": args.input_ref,
            "scanned": len(entries),
            "grouped_entries": total,
            "distinct_templates": len(groups),
            "groups": [g.as_dict() for g in shown],
        },
    )


# ---------------------------------------------------------------------------
# correlation ids
# ---------------------------------------------------------------------------

_ID_KEY_STEMS = frozenset(
    {
        "trace", "request", "correlation", "corr", "span", "parent", "session",
        "job", "txn", "transaction", "operation", "req", "invocation", "order",
        "flow", "conversation", "call", "run",
    }
)
_EXTRA_ID_KEYS = frozenset({"traceparent", "traceid", "requestid", "spanid", "tid", "rid"})

_SHAPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("traceparent", re.compile(r"\b00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}\b")),
    ("uuid", re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)),
    ("hex32", re.compile(r"(?<![0-9a-fA-F-])[0-9a-f]{32}(?![0-9a-fA-F-])")),
    ("hex16", re.compile(r"(?<![0-9a-fA-F-])[0-9a-f]{16}(?![0-9a-fA-F-])")),
]

_SERVICE_KEYS = (
    "service", "service.name", "svc", "app", "application", "component", "logger",
    "container", "pod", "source", "unit", "host", "kubernetes.container_name",
)


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def is_correlation_key(key: str) -> bool:
    norm = _normalise_key(key)
    if norm in _EXTRA_ID_KEYS:
        return True
    if norm.startswith("x") and norm[1:] in _EXTRA_ID_KEYS:
        return True
    stem = norm[1:] if norm.startswith("x") and len(norm) > 3 else norm
    return stem.endswith("id") and stem[:-2] in _ID_KEY_STEMS


def entry_service(entry: LogEntry, fallback: str = "") -> str:
    for key in _SERVICE_KEYS:
        value = entry.fields.get(key)
        if value:
            return value[:60]
    return fallback


def entry_id_for_key(entry: LogEntry, key: str) -> str | None:
    """Look a correlation id up by exact key, then by any alias of that key."""
    direct = entry.fields.get(key)
    if direct:
        return direct
    target = _normalise_key(key)
    for name, value in entry.fields.items():
        if value and _normalise_key(name) == target:
            return value
    match = re.search(rf"{re.escape(key)}\s*[=:]\s*\"?([\w.:-]+)", entry.raw, re.IGNORECASE)
    return match.group(1) if match else None


def _shape_ids(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for shape, pattern in _SHAPE_PATTERNS:
        for match in pattern.finditer(text):
            found.append((shape, match.group(0)))
            if len(found) >= 8:
                return found
    return found


class ExtractIdsInput(BaseModel):
    input_ref: str
    top_n: int = Field(default=10, ge=1, le=50)
    min_occurrences: int = Field(default=2, ge=1)


@tool(
    name="extract_correlation_ids",
    description=(
        "Find trace, request, correlation, and span ids in a log artifact, both by key name "
        "(trace_id, request_id, x-request-id, correlation_id, traceparent) and by value shape "
        "(UUID, 32-hex, W3C traceparent). Ranks candidate keys and ids by frequency and "
        "reports which ids span more than one service."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "correlation"),
)
async def extract_correlation_ids(args: ExtractIdsInput, ctx: ToolContext) -> ToolResult:
    artifact, entries, truncated = _entries(ctx, args.input_ref)
    default_service = str(artifact.metadata.get("service") or "")

    key_values: dict[str, Counter[str]] = defaultdict(Counter)
    shape_values: dict[str, Counter[str]] = defaultdict(Counter)
    id_services: dict[str, set[str]] = defaultdict(set)
    id_lines: dict[str, list[int]] = defaultdict(list)
    id_key: dict[str, str] = {}
    id_span: dict[str, list[float]] = defaultdict(list)

    for entry in entries:
        service = entry_service(entry, default_service)
        seen: list[tuple[str, str]] = []
        for name, value in entry.fields.items():
            if value and is_correlation_key(name):
                key_values[name][value] += 1
                seen.append((name, value))
        for shape, value in _shape_ids(entry.searchable()):
            shape_values[shape][value] += 1
            if not any(v == value for _, v in seen):
                seen.append((f"<{shape}>", value))
        for name, value in seen:
            id_key.setdefault(value, name)
            id_services[value].add(service or "unknown")
            if len(id_lines[value]) < 40:
                id_lines[value].append(entry.line_no)
            if entry.timestamp is not None:
                id_span[value].append(entry.timestamp)

    ranked_keys = sorted(
        key_values.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])
    )
    keys_out = [
        {
            "key": name,
            "occurrences": sum(counter.values()),
            "distinct_values": len(counter),
            "example": next(iter(counter)),
        }
        for name, counter in ranked_keys[: args.top_n]
    ]
    shapes_out = [
        {"shape": shape, "occurrences": sum(c.values()), "distinct_values": len(c)}
        for shape, c in sorted(shape_values.items(), key=lambda kv: -sum(kv[1].values()))
    ]

    candidates = [
        value
        for value, lines in id_lines.items()
        if len(lines) >= args.min_occurrences or len(id_services[value]) > 1
    ]
    candidates.sort(key=lambda v: (-len(id_services[v]), -len(id_lines[v]), v))
    top_ids = []
    for value in candidates[: args.top_n]:
        stamps = sorted(id_span[value])
        top_ids.append(
            {
                "id": value,
                "key": id_key.get(value, "<shape>"),
                "occurrences": len(id_lines[value]),
                "services": sorted(id_services[value]),
                "lines": id_lines[value][:10],
                "first_seen": _iso(stamps[0]) if stamps else None,
                "last_seen": _iso(stamps[-1]) if stamps else None,
                "elapsed_s": round(stamps[-1] - stamps[0], 3) if len(stamps) > 1 else None,
            }
        )
    multi = [row for row in top_ids if len(row["services"]) > 1]

    best_key = keys_out[0]["key"] if keys_out else (shapes_out[0]["shape"] if shapes_out else "")
    evidence = [
        _evidence(
            claim=(
                f"correlation id {row['id']} appears {row['occurrences']}x across "
                f"{len(row['services'])} service(s)"
            ),
            artifact=artifact,
            lines=list(row["lines"]),
            excerpt=f"{row['key']}={row['id']} services={row['services']}",
            tool_name="extract_correlation_ids",
            structured={"id": row["id"], "services": row["services"]},
            confidence=0.8,
        )
        for row in (multi or top_ids)[:MAX_EVIDENCE]
    ]
    return ToolResult(
        tool="extract_correlation_ids",
        summary=(
            f"{len(id_lines)} distinct ids; best key '{best_key}'; "
            f"{len(multi)} of the top ids span more than one service"
        ),
        truncated=truncated,
        evidence=evidence,
        data={
            "input_ref": args.input_ref,
            "recommended_key": best_key,
            "keys": keys_out,
            "shapes": shapes_out,
            "top_ids": top_ids,
            "multi_service_ids": [row["id"] for row in multi],
        },
    )


# ---------------------------------------------------------------------------
# timeout vocabulary, shared by correlate_logs and detect_timeout_patterns
# ---------------------------------------------------------------------------

#: Configured timeouts cluster on round values. Natural latency does not.
ROUND_TIMEOUTS_S: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0,
    20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0, 600.0,
)

_TIMEOUT_RE = re.compile(
    r"(?i)\b("
    r"context deadline exceeded|deadline exceeded|context canceled|context cancelled|"
    r"timed?[ -]?out|timeout|i/o timeout|read timeout|write timeout|"
    r"etimedout|esockettimedout|deadlineexceeded|"
    r"socket ?timeout ?exception|timeoutexception|"
    r"canceling statement due to statement timeout|"
    r"gateway ?time-?out|504|upstream request timeout|client timeout"
    r")\b"
)

_POOL_RE = re.compile(
    r"(?i)("
    r"connection pool (?:is )?(?:exhausted|full|saturated)|pool exhausted|"
    r"too many (?:connections|clients)|sorry, too many clients already|"
    r"remaining connection slots are reserved|max_connections|"
    r"timeout acquiring connection|unable to acquire connection|"
    r"connection is not available, request timed out after|hikaripool|"
    r"no available connection|pool timeout|acquire timeout|"
    r"connection limit exceeded|out of shared memory"
    r")"
)

_RETRY_RE = re.compile(
    r"(?i)\b(retry|retrying|retries|attempt\s*[=#:]?\s*\d+|backoff|"
    r"back-off|re-?dialing|redelivery)\b"
)

_CRASH_RE = re.compile(
    r"(?i)("
    r"crashloopbackoff|oomkilled|back-?off restarting failed container|"
    r"liveness probe failed|readiness probe failed|exit code 137|exit code 143|"
    r"signal: killed|sigsegv|sigkill|fatal error: |panic: |"
    r"container restart|restarting failed container"
    r")"
)

_DURATION_RE = re.compile(
    r"(?<![\w.])(?P<n>\d+(?:\.\d+)?)\s*(?P<u>ns|µs|us|ms|milliseconds?|s|secs?|seconds?|"
    r"m|mins?|minutes?|h|hours?)\b",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "ns": 1e-9, "us": 1e-6, "µs": 1e-6,
    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hour": 3600.0, "hours": 3600.0,
}

_DURATION_FIELD_RE = re.compile(
    r"(?i)(duration|elapsed|latency|took|took_?time|response_?time|rt|time_?taken|"
    r"wait|waited|spent|cost)"
)


def nearest_round_timeout(seconds: float, rel_tol: float = 0.05) -> tuple[float, float] | None:
    """Return ``(candidate, relative_distance)`` when ``seconds`` sits on a round value."""
    if seconds <= 0:
        return None
    best: tuple[float, float] | None = None
    for candidate in ROUND_TIMEOUTS_S:
        distance = abs(seconds - candidate) / candidate
        if best is None or distance < best[1]:
            best = (candidate, distance)
    if best is not None and best[1] <= rel_tol:
        return best
    return None


def _field_duration_seconds(key: str, value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    norm = _normalise_key(key)
    if norm.endswith("ns"):
        return number * 1e-9
    if norm.endswith(("us", "micros")):
        return number * 1e-6
    if norm.endswith("ms"):
        return number * 1e-3
    if norm.endswith(("s", "sec", "seconds")):
        return number
    # Bare duration fields are conventionally milliseconds once they get large.
    return number * 1e-3 if number > 1000 else number


def entry_durations(entry: LogEntry) -> list[float]:
    """Every duration mentioned by an entry, normalised to seconds."""
    out: list[float] = []
    for match in _DURATION_RE.finditer(entry.message):
        unit = match.group("u").lower()
        factor = _UNIT_SECONDS.get(unit)
        if factor is None:
            continue
        out.append(float(match.group("n")) * factor)
    for key, value in entry.fields.items():
        if not _DURATION_FIELD_RE.search(key):
            continue
        seconds = _field_duration_seconds(key, value)
        if seconds is not None and seconds not in out:
            out.append(seconds)
    return [d for d in out if 0 < d < 86400]


# ---------------------------------------------------------------------------
# correlate_logs
# ---------------------------------------------------------------------------


class CorrelateLogsInput(BaseModel):
    input_refs: list[str] = Field(min_length=2, description="Two or more log artifact refs.")
    key: str = Field(default="", description="Correlation key; empty auto-detects the best one.")
    ids: list[str] = Field(default_factory=list, description="Restrict to these id values.")
    time_tolerance_s: float = Field(default=5.0, ge=0.0, le=3600.0)
    max_ids: int = Field(default=3, ge=1, le=10)


def _auto_key(per_ref: dict[str, list[LogEntry]]) -> str:
    """Pick the correlation key with the widest coverage across the artifacts."""
    scores: Counter[str] = Counter()
    spread: dict[str, set[str]] = defaultdict(set)
    for ref, entries in per_ref.items():
        for entry in entries:
            for name, value in entry.fields.items():
                if value and is_correlation_key(name):
                    scores[name] += 1
                    spread[name].add(ref)
    if not scores:
        return ""
    return max(scores, key=lambda k: (len(spread[k]), scores[k]))


def _id_of(entry: LogEntry, key: str) -> str | None:
    if key:
        value = entry_id_for_key(entry, key)
        if value:
            return value
    shapes = _shape_ids(entry.searchable())
    return shapes[0][1] if shapes else None


@tool(
    name="correlate_logs",
    description=(
        "Merge several log artifacts onto one timeline keyed on a correlation id, within a "
        "time tolerance, so caller and callee timestamps line up. Reports each side's elapsed "
        "time, the handoff and response gaps, whether either side's elapsed time sits on a "
        "round timeout value, and a verdict on which side owns the timeout."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "correlation", "timeout"),
)
async def correlate_logs(args: CorrelateLogsInput, ctx: ToolContext) -> ToolResult:
    sources: dict[str, tuple[Artifact, list[LogEntry]]] = {}
    for ref in args.input_refs:
        artifact, entries, _ = _entries(ctx, ref)
        sources[ref] = (artifact, entries)

    key = args.key or _auto_key({ref: e for ref, (_, e) in sources.items()})

    # id -> ref -> entries
    buckets: dict[str, dict[str, list[LogEntry]]] = defaultdict(lambda: defaultdict(list))
    wanted = {i for i in args.ids if i}
    for ref, (_, entries) in sources.items():
        for entry in entries:
            value = _id_of(entry, key)
            if not value or (wanted and value not in wanted):
                continue
            buckets[value][ref].append(entry)

    cross = {i: refs for i, refs in buckets.items() if len(refs) >= 2}
    if not cross:
        return ToolResult(
            tool="correlate_logs",
            summary=(
                f"no id under key '{key or 'auto'}' appears in more than one of the "
                f"{len(sources)} artifacts"
            ),
            data={
                "key": key,
                "input_refs": args.input_refs,
                "correlated_ids": 0,
                "distinct_ids": len(buckets),
            },
        )

    ordered_ids = sorted(
        cross, key=lambda i: (-len(cross[i]), -sum(len(v) for v in cross[i].values()))
    )
    results: list[dict[str, Any]] = []
    evidence: list[Evidence] = []

    for value in ordered_ids[: args.max_ids]:
        per_source: list[dict[str, Any]] = []
        rows: list[tuple[float, str, LogEntry]] = []
        for ref, entries in cross[value].items():
            artifact, _ = sources[ref]
            label = _label(artifact)
            stamps = sorted(e.timestamp for e in entries if e.timestamp is not None)
            elapsed = (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
            near = nearest_round_timeout(elapsed) if elapsed else None
            per_source.append(
                {
                    "input_ref": ref,
                    "label": label,
                    "service": entry_service(entries[0], str(artifact.metadata.get("service", ""))),
                    "entries": len(entries),
                    "first_seen": _iso(stamps[0]) if stamps else None,
                    "last_seen": _iso(stamps[-1]) if stamps else None,
                    "elapsed_s": round(elapsed, 4) if elapsed is not None else None,
                    "nearest_round_timeout_s": near[0] if near else None,
                    "round_timeout_distance": round(near[1], 4) if near else None,
                    "explicit_timeout": any(_TIMEOUT_RE.search(e.searchable()) for e in entries),
                    "lines": [e.line_no for e in entries[:10]],
                    "_first": stamps[0] if stamps else None,
                    "_last": stamps[-1] if stamps else None,
                }
            )
            for entry in entries:
                rows.append((entry.timestamp if entry.timestamp is not None else math.inf,
                             label, entry))

        timed = [s for s in per_source if s["_first"] is not None]
        timed.sort(key=lambda s: s["_first"])
        verdict = "unclear"
        owner = "unclear"
        detail: dict[str, Any] = {}
        if len(timed) >= 2:
            caller, callee = timed[0], timed[-1]
            handoff = callee["_first"] - caller["_first"]
            response = caller["_last"] - callee["_last"]
            detail = {
                "caller": caller["label"],
                "callee": callee["label"],
                "handoff_gap_s": round(handoff, 4),
                "response_gap_s": round(response, 4),
                "observed_gap_s": round(callee["_last"] - caller["_first"], 4),
                "aligned_within_tolerance": abs(response) <= args.time_tolerance_s,
                "clock_skew_suspected": handoff < -args.time_tolerance_s,
            }
            caller_hit = caller["round_timeout_distance"]
            callee_hit = callee["round_timeout_distance"]
            if caller_hit is not None and (callee_hit is None or caller_hit < callee_hit):
                owner = "caller"
                verdict = (
                    f"{caller['label']} gave up after {caller['elapsed_s']}s, which sits on the "
                    f"round value {caller['nearest_round_timeout_s']}s; "
                    f"{callee['label']} ran for {callee['elapsed_s']}s. The deadline looks "
                    "caller-side (client or proxy configuration), not callee-side."
                )
            elif callee_hit is not None:
                owner = "callee"
                verdict = (
                    f"{callee['label']} stopped after {callee['elapsed_s']}s on the round value "
                    f"{callee['nearest_round_timeout_s']}s; the limit looks callee-side "
                    "(server or database statement timeout)."
                )
            elif caller["explicit_timeout"] and not callee["explicit_timeout"]:
                owner = "caller"
                verdict = (
                    f"only {caller['label']} logs an explicit timeout; {callee['label']} shows "
                    "no deadline of its own, so the caller abandoned the request."
                )
            else:
                verdict = "no round-number deadline on either side; latency looks organic"

        for source in per_source:
            source.pop("_first", None)
            source.pop("_last", None)

        rows.sort(key=lambda r: (r[0], r[2].line_no))
        timeline = [
            {
                "ts": _iso(entry.timestamp),
                "source": label,
                "line": entry.line_no,
                "level": entry.level,
                "text": _clip(entry.message, 180),
            }
            for _, label, entry in rows[:MAX_TIMELINE_ROWS]
        ]
        results.append(
            {
                "id": value,
                "sources": per_source,
                "timeout_owner": owner,
                "verdict": verdict,
                **detail,
                "timeline": timeline,
                "timeline_truncated": len(rows) > MAX_TIMELINE_ROWS,
            }
        )

        primary_ref = next(iter(cross[value]))
        evidence.append(
            _evidence(
                claim=f"correlated id {value}: {verdict}",
                artifact=sources[primary_ref][0],
                lines=[entry.line_no for _, _, entry in rows[:10]],
                excerpt="\n".join(
                    f"{row['ts']} {row['source']} L{row['line']} {row['text']}"
                    for row in timeline[:6]
                ),
                tool_name="correlate_logs",
                kind=EvidenceKind.INFERRED if owner != "unclear" else EvidenceKind.OBSERVED,
                confidence=0.75 if owner != "unclear" else 0.55,
                structured={"id": value, "timeout_owner": owner, **detail},
            )
        )

    owners = Counter(r["timeout_owner"] for r in results)
    return ToolResult(
        tool="correlate_logs",
        summary=(
            f"key '{key or 'shape'}': {len(cross)} ids span >=2 artifacts; "
            f"timeout owner {dict(owners)}"
        ),
        evidence=evidence[:MAX_EVIDENCE],
        truncated=len(cross) > args.max_ids,
        data={
            "key": key,
            "input_refs": args.input_refs,
            "time_tolerance_s": args.time_tolerance_s,
            "distinct_ids": len(buckets),
            "correlated_ids": len(cross),
            "results": results,
        },
    )


# ---------------------------------------------------------------------------
# detect_timeout_patterns
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {
    "connection_pool_exhaustion": 0.92,
    "restart_loop": 0.88,
    "configured_timeout_cluster": 0.84,
    "retry_amplification": 0.78,
    "explicit_timeout": 0.70,
}


class DetectTimeoutsInput(BaseModel):
    input_ref: str
    correlation_key: str = Field(default="", description="Empty auto-detects the id key.")
    min_cluster_samples: int = Field(default=3, ge=2, le=100)
    cluster_tolerance: float = Field(default=0.05, ge=0.001, le=0.5)
    top_n: int = Field(default=10, ge=1, le=40)


def _finding(
    kind: str,
    title: str,
    *,
    count: int,
    lines: list[int],
    detail: dict[str, Any],
    excerpt: str,
) -> dict[str, Any]:
    weight = _SEVERITY_WEIGHT.get(kind, 0.5)
    return {
        "kind": kind,
        "title": title,
        "count": count,
        "score": round(weight * (1.0 + math.log10(max(count, 1))), 4),
        "confidence": weight,
        "evidence_lines": sorted(lines)[:12],
        "excerpt": _clip(excerpt),
        **detail,
    }


def _cluster_durations(
    samples: list[tuple[LogEntry, float]], tolerance: float, min_samples: int
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for candidate in ROUND_TIMEOUTS_S:
        members = [
            (entry, value)
            for entry, value in samples
            if abs(value - candidate) / candidate <= tolerance
        ]
        if len(members) < min_samples:
            continue
        values = [v for _, v in members]
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        clusters.append(
            {
                "candidate_s": candidate,
                "samples": len(members),
                "mean_s": round(mean, 4),
                "stdev_s": round(stdev, 4),
                "coefficient_of_variation": round(stdev / mean, 5) if mean else 0.0,
                "min_s": round(min(values), 4),
                "max_s": round(max(values), 4),
                "lines": [e.line_no for e, _ in members[:12]],
                "example": _clip(members[0][0].message, 160),
            }
        )
    # A tight cluster of many samples is the strongest signal of a configured limit.
    clusters.sort(key=lambda c: (-c["samples"], c["coefficient_of_variation"]))
    return clusters


def _retry_chains(
    entries: list[LogEntry], key: str
) -> list[dict[str, Any]]:
    buckets: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in entries:
        value = _id_of(entry, key)
        if value:
            buckets[value].append(entry)

    chains: list[dict[str, Any]] = []
    for value, group in buckets.items():
        if len(group) < 3:
            continue
        stamps = [e.timestamp for e in group if e.timestamp is not None]
        retries = sum(1 for e in group if _RETRY_RE.search(e.searchable()))
        attempts = sorted(
            {
                int(m.group(1))
                for e in group
                for m in re.finditer(r"(?i)attempt\s*[=#:]?\s*(\d+)", e.searchable())
            }
        )
        gaps = [round(b - a, 4) for a, b in pairwise(stamps)]
        growing = len(gaps) >= 2 and all(
            b >= a * 1.4 for a, b in pairwise(gaps) if a > 0
        )
        if retries == 0 and not growing and len(attempts) < 3:
            continue
        chains.append(
            {
                "id": value,
                "occurrences": len(group),
                "retry_markers": retries,
                "attempts_seen": attempts[:10],
                "gaps_s": gaps[:10],
                "exponential_backoff": bool(growing),
                "total_span_s": round(stamps[-1] - stamps[0], 4) if len(stamps) > 1 else None,
                "lines": [e.line_no for e in group[:12]],
                "example": _clip(group[0].message, 160),
            }
        )
    chains.sort(key=lambda c: (-c["occurrences"], c["id"]))
    return chains


@tool(
    name="detect_timeout_patterns",
    description=(
        "Rank timeout-shaped evidence in a log artifact: explicit timeout and deadline "
        "messages, durations clustering on round values (which indicates a configured "
        "timeout rather than natural latency), retry amplification on a single correlation "
        "id with growing backoff, connection pool exhaustion, and container restart loops. "
        "Each finding carries the line numbers that support it."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "timeout", "diagnosis"),
)
async def detect_timeout_patterns(args: DetectTimeoutsInput, ctx: ToolContext) -> ToolResult:
    artifact, entries, truncated = _entries(ctx, args.input_ref)
    findings: list[dict[str, Any]] = []

    # (a) explicit timeout and deadline messages, collapsed into templates.
    explicit = [e for e in entries if _TIMEOUT_RE.search(e.searchable())]
    for group in build_groups(explicit, include_all=True)[:3]:
        findings.append(
            _finding(
                "explicit_timeout",
                f"{group.count}x timeout message: {_clip(group.template, 120)}",
                count=group.count,
                lines=group.lines,
                detail={
                    "template": group.template,
                    "first_seen": _iso(group.first_seen),
                    "last_seen": _iso(group.last_seen),
                },
                excerpt=group.example,
            )
        )

    # (b) durations clustering on round values.
    samples: list[tuple[LogEntry, float]] = [
        (entry, value) for entry in entries for value in entry_durations(entry)
    ]
    clusters = _cluster_durations(samples, args.cluster_tolerance, args.min_cluster_samples)
    for cluster in clusters[:3]:
        findings.append(
            _finding(
                "configured_timeout_cluster",
                (
                    f"{cluster['samples']} durations cluster on {cluster['candidate_s']}s "
                    f"(mean {cluster['mean_s']}s, cv {cluster['coefficient_of_variation']})"
                ),
                count=cluster["samples"],
                lines=cluster["lines"],
                detail={k: v for k, v in cluster.items() if k not in ("lines", "example")},
                excerpt=cluster["example"],
            )
        )

    # (c) retry amplification on a single correlation id.
    key = args.correlation_key or _auto_key({args.input_ref: entries})
    for chain in _retry_chains(entries, key)[:3]:
        findings.append(
            _finding(
                "retry_amplification",
                (
                    f"id {chain['id']} retried {chain['occurrences']}x"
                    + (" with growing backoff" if chain["exponential_backoff"] else "")
                ),
                count=chain["occurrences"],
                lines=chain["lines"],
                detail={k: v for k, v in chain.items() if k not in ("lines", "example")},
                excerpt=chain["example"],
            )
        )

    # (d) connection pool exhaustion.
    pool = [e for e in entries if _POOL_RE.search(e.searchable())]
    if pool:
        findings.append(
            _finding(
                "connection_pool_exhaustion",
                f"{len(pool)} connection pool saturation messages",
                count=len(pool),
                lines=[e.line_no for e in pool],
                detail={
                    "templates": [g.template for g in build_groups(pool, include_all=True)[:4]],
                    "first_seen": _iso(
                        min((e.timestamp for e in pool if e.timestamp), default=None)
                    ),
                },
                excerpt=pool[0].message,
            )
        )

    # (e) restart loops, because G4 also asks whether pods are restarting.
    crash = [e for e in entries if _CRASH_RE.search(e.searchable())]
    if crash:
        findings.append(
            _finding(
                "restart_loop",
                f"{len(crash)} container crash or restart messages",
                count=len(crash),
                lines=[e.line_no for e in crash],
                detail={
                    "templates": [g.template for g in build_groups(crash, include_all=True)[:4]]
                },
                excerpt=crash[0].message,
            )
        )

    findings.sort(key=lambda f: -f["score"])
    findings = findings[: args.top_n]
    evidence = [
        _evidence(
            claim=item["title"],
            artifact=artifact,
            lines=list(item["evidence_lines"]),
            excerpt=item["excerpt"],
            tool_name="detect_timeout_patterns",
            kind=EvidenceKind.INFERRED
            if item["kind"] == "configured_timeout_cluster"
            else EvidenceKind.OBSERVED,
            confidence=item["confidence"],
            structured={"kind": item["kind"], "count": item["count"]},
        )
        for item in findings[:MAX_EVIDENCE]
    ]
    headline = findings[0]["title"] if findings else "no timeout patterns detected"
    return ToolResult(
        tool="detect_timeout_patterns",
        summary=f"{len(findings)} finding(s) over {len(entries)} entries; top: {headline}",
        truncated=truncated,
        evidence=evidence,
        data={
            "input_ref": args.input_ref,
            "correlation_key": key,
            "scanned": len(entries),
            "duration_samples": len(samples),
            "findings": findings,
        },
    )


# ---------------------------------------------------------------------------
# compare_before_after
# ---------------------------------------------------------------------------


class CompareInput(BaseModel):
    before_ref: str
    after_ref: str
    min_level: str = Field(default="WARN", description="'' compares every message template.")
    top_n: int = Field(default=15, ge=1, le=60)
    min_rate_change: float = Field(
        default=1.5, ge=1.0, description="Ratio at which a rate change is reported."
    )


def _rate_per_minute(group: MessageGroup, span_s: float | None) -> float | None:
    if not span_s or span_s <= 0:
        return None
    return round(group.count / (span_s / 60.0), 4)


@tool(
    name="compare_before_after",
    description=(
        "Diff two log artifacts by error-group frequency: which message templates appeared, "
        "which disappeared, and which changed rate. Rates are normalised per minute using "
        "each artifact's observed time span, so windows of different length still compare."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "diff"),
)
async def compare_before_after(args: CompareInput, ctx: ToolContext) -> ToolResult:
    before_art, before_entries, _ = _entries(ctx, args.before_ref)
    after_art, after_entries, _ = _entries(ctx, args.after_ref)

    before_groups = {g.template: g for g in build_groups(before_entries, min_level=args.min_level)}
    after_groups = {g.template: g for g in build_groups(after_entries, min_level=args.min_level)}

    b_first, b_last = _span(before_entries)
    a_first, a_last = _span(after_entries)
    b_span = (b_last - b_first) if (b_first is not None and b_last is not None) else None
    a_span = (a_last - a_first) if (a_first is not None and a_last is not None) else None

    appeared = [
        {
            "template": t,
            "count": g.count,
            "rate_per_min": _rate_per_minute(g, a_span),
            "first_line": g.first_line,
            "example": _clip(g.example, 160),
        }
        for t, g in after_groups.items()
        if t not in before_groups
    ]
    disappeared = [
        {
            "template": t,
            "count": g.count,
            "rate_per_min": _rate_per_minute(g, b_span),
            "first_line": g.first_line,
            "example": _clip(g.example, 160),
        }
        for t, g in before_groups.items()
        if t not in after_groups
    ]
    changed: list[dict[str, Any]] = []
    for template, after_group in after_groups.items():
        before_group = before_groups.get(template)
        if before_group is None:
            continue
        b_rate = _rate_per_minute(before_group, b_span)
        a_rate = _rate_per_minute(after_group, a_span)
        if b_rate and a_rate:
            ratio = a_rate / b_rate
        else:
            ratio = after_group.count / max(before_group.count, 1)
        if ratio >= args.min_rate_change or ratio <= 1.0 / args.min_rate_change:
            changed.append(
                {
                    "template": template,
                    "before_count": before_group.count,
                    "after_count": after_group.count,
                    "before_rate_per_min": b_rate,
                    "after_rate_per_min": a_rate,
                    "ratio": round(ratio, 3),
                    "direction": "increased" if ratio > 1 else "decreased",
                    "before_lines": before_group.lines[:6],
                    "after_lines": after_group.lines[:6],
                }
            )

    appeared.sort(key=lambda r: -r["count"])
    disappeared.sort(key=lambda r: -r["count"])
    changed.sort(key=lambda r: -abs(math.log(max(r["ratio"], 1e-6))))

    evidence: list[Evidence] = []
    for row in appeared[:4]:
        evidence.append(
            _evidence(
                claim=f"new after the change: {_clip(row['template'], 120)} ({row['count']}x)",
                artifact=after_art,
                lines=[row["first_line"]],
                excerpt=row["example"],
                tool_name="compare_before_after",
                structured={"template": row["template"], "count": row["count"]},
            )
        )
    for row in changed[:4]:
        evidence.append(
            _evidence(
                claim=(
                    f"rate {row['direction']} {row['ratio']}x: {_clip(row['template'], 110)}"
                ),
                artifact=after_art,
                lines=list(row["after_lines"]),
                excerpt=row["template"],
                tool_name="compare_before_after",
                kind=EvidenceKind.INFERRED,
                confidence=0.7,
                structured={k: row[k] for k in ("before_count", "after_count", "ratio")},
            )
        )

    return ToolResult(
        tool="compare_before_after",
        summary=(
            f"{len(appeared)} new, {len(disappeared)} gone, {len(changed)} changed rate "
            f"(before {_label(before_art)} -> after {_label(after_art)})"
        ),
        evidence=evidence[:MAX_EVIDENCE],
        truncated=max(len(appeared), len(disappeared), len(changed)) > args.top_n,
        data={
            "before_ref": args.before_ref,
            "after_ref": args.after_ref,
            "before": {
                "entries": len(before_entries),
                "templates": len(before_groups),
                "span_s": round(b_span, 3) if b_span else None,
                "window": [_iso(b_first), _iso(b_last)],
            },
            "after": {
                "entries": len(after_entries),
                "templates": len(after_groups),
                "span_s": round(a_span, 3) if a_span else None,
                "window": [_iso(a_first), _iso(a_last)],
            },
            "appeared": appeared[: args.top_n],
            "disappeared": disappeared[: args.top_n],
            "rate_changed": changed[: args.top_n],
        },
    )


# ---------------------------------------------------------------------------
# summarise_log_volume
# ---------------------------------------------------------------------------

_BUCKET_LADDER = (1, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600, 21600, 86400)


class VolumeInput(BaseModel):
    input_ref: str
    bucket_seconds: int = Field(default=0, ge=0, description="0 picks a bucket from the span.")
    top_n: int = Field(default=10, ge=1, le=50)
    max_buckets: int = Field(default=40, ge=4, le=200)


@tool(
    name="summarise_log_volume",
    description=(
        "Volume profile of a log artifact: a histogram over time buckets, the level "
        "distribution, the top talkers by service or logger, and burst detection for "
        "buckets that sit well above the median rate."
    ),
    capability=Capability.LOGS,
    risk=RiskClass.R0,
    tags=("logs", "volume"),
)
async def summarise_log_volume(args: VolumeInput, ctx: ToolContext) -> ToolResult:
    artifact, entries, truncated = _entries(ctx, args.input_ref)
    if not entries:
        return ToolResult(
            tool="summarise_log_volume",
            summary=f"{args.input_ref} contains no log entries",
            data={"input_ref": args.input_ref, "entries": 0},
        )

    first, last = _span(entries)
    span = (last - first) if (first is not None and last is not None) else 0.0
    bucket = args.bucket_seconds
    if bucket <= 0:
        target = max(span / args.max_buckets, 1.0) if span else 60.0
        bucket = next((b for b in _BUCKET_LADDER if b >= target), _BUCKET_LADDER[-1])

    histogram: Counter[int] = Counter()
    errors_per_bucket: Counter[int] = Counter()
    for entry in entries:
        if entry.timestamp is None or first is None:
            continue
        index = int((entry.timestamp - first) // bucket)
        histogram[index] += 1
        if _LEVEL_RANK.get(entry.level or "", -1) >= _LEVEL_RANK["ERROR"]:
            errors_per_bucket[index] += 1

    counts = [histogram.get(i, 0) for i in range(max(histogram) + 1)] if histogram else []
    median = statistics.median(counts) if counts else 0.0
    mean = statistics.fmean(counts) if counts else 0.0
    stdev = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    threshold = max(mean + 2 * stdev, 3 * median, 5)
    bursts = [
        {
            "bucket_start": _iso((first or 0) + index * bucket),
            "count": count,
            "errors": errors_per_bucket.get(index, 0),
            "times_median": round(count / median, 2) if median else None,
        }
        for index, count in enumerate(counts)
        if count >= threshold
    ]

    talkers: Counter[str] = Counter()
    default_service = str(artifact.metadata.get("service") or "")
    for entry in entries:
        talkers[entry_service(entry, default_service) or "unknown"] += 1

    levels = Counter(e.level or "UNKNOWN" for e in entries)
    error_entries = [
        e for e in entries if _LEVEL_RANK.get(e.level or "", -1) >= _LEVEL_RANK["ERROR"]
    ]
    evidence: list[Evidence] = []
    if bursts:
        peak = max(bursts, key=lambda b: b["count"])
        evidence.append(
            _evidence(
                claim=(
                    f"log volume bursts to {peak['count']} entries in the {bucket}s bucket "
                    f"starting {peak['bucket_start']}"
                ),
                artifact=artifact,
                lines=[e.line_no for e in error_entries[:8]] or [entries[0].line_no],
                excerpt=(error_entries[0].message if error_entries else entries[0].message),
                tool_name="summarise_log_volume",
                kind=EvidenceKind.INFERRED,
                confidence=0.7,
                structured={"bucket_seconds": bucket, "peak": peak["count"]},
            )
        )

    return ToolResult(
        tool="summarise_log_volume",
        summary=(
            f"{len(entries)} entries over {round(span, 1)}s; {levels.get('ERROR', 0)} errors; "
            f"{len(bursts)} burst bucket(s) at {bucket}s resolution"
        ),
        truncated=truncated,
        evidence=evidence,
        data={
            "input_ref": args.input_ref,
            "entries": len(entries),
            "unparsed_entries": sum(1 for e in entries if not e.parsed),
            "window": [_iso(first), _iso(last)],
            "span_s": round(span, 3),
            "bucket_seconds": bucket,
            "entries_per_minute": round(len(entries) / (span / 60.0), 2) if span else None,
            "levels": dict(levels.most_common()),
            "top_talkers": [
                {"source": name, "entries": count}
                for name, count in talkers.most_common(args.top_n)
            ],
            "histogram": [
                {"bucket_start": _iso((first or 0) + i * bucket), "count": c}
                for i, c in enumerate(counts[: args.max_buckets])
            ],
            "histogram_truncated": len(counts) > args.max_buckets,
            "bursts": bursts[: args.top_n],
        },
    )
