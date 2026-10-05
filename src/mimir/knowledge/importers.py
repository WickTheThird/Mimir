"""Importers for prior assistant transcripts (ADR 11.5, NG4)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from mimir.config import Settings, get_settings
from mimir.knowledge.store import (
    Confidence,
    DocumentMetadata,
    KnowledgeStore,
    VerificationStatus,
    slugify,
)
from mimir.logging import get_logger
from mimir.redaction import redact

log = get_logger(__name__)

IMPORT_ROOT = "imports"
KNOWN_TOOLS = ("claude", "codex", "chatgpt", "other")
MAX_MESSAGE_CHARS = 6000
MAX_EXCERPTS = 6
MAX_EXCERPT_CHARS = 400

_COMMAND_RE = re.compile(
    r"(?m)^\s*(?:\$\s*)?((?:kubectl|k9s|helm|sdm|docker|psql|mysql|redis-cli|curl|git|aws|gcloud|"
    r"az|systemctl|journalctl|kannel|bearerbox|smsbox|nc|dig|openssl|jq|tail|grep)\s+[^\n`]{3,180})"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLAIM_HINT_RE = re.compile(
    r"(?i)\b(runs? on|lives? in|is deployed|is configured|defaults? to|is owned by|maps? to|"
    r"the root cause|caused by|because|the fix|resolved by|namespace|cluster|timeout|"
    r"connection pool|restart)\b"
)


@dataclass(slots=True)
class ImportedMessage:
    role: str
    text: str
    timestamp: float | None = None


@dataclass(slots=True)
class ImportedConversation:
    """One transcript, already redacted, not yet trusted."""

    source_tool: str
    conversation_id: str
    title: str
    messages: list[ImportedMessage] = field(default_factory=list)
    started_at: date | None = None
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def user_messages(self) -> list[ImportedMessage]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def assistant_messages(self) -> list[ImportedMessage]:
        return [m for m in self.messages if m.role == "assistant"]

    def transcript(self, max_chars: int = 40000) -> str:
        out: list[str] = []
        total = 0
        for message in self.messages:
            block = f"[{message.role}] {message.text}"
            total += len(block)
            if total > max_chars:
                out.append("...[transcript truncated]")
                break
            out.append(block)
        return "\n\n".join(out)


Summariser = Callable[[ImportedConversation], str]


@dataclass(slots=True)
class ImportResult:
    tool: str
    conversations_seen: int = 0
    notes_written: int = 0
    skipped: int = 0
    doc_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.tool}: {self.conversations_seen} conversations, "
            f"{self.notes_written} candidate notes written to imports/{self.tool}/, "
            f"{self.skipped} skipped, {len(self.errors)} errors"
        )


# -- parsing ---------------------------------------------------------------


def _clean(text: str) -> str:
    """Redact and normalise a message body. Every parser routes through this."""
    if not text:
        return ""
    body = redact(str(text)).strip()
    if len(body) > MAX_MESSAGE_CHARS:
        body = body[:MAX_MESSAGE_CHARS] + " ...[truncated]"
    return body


def _flatten_content(content: Any) -> str:
    """Handle the string, list-of-blocks, and dict content shapes in the wild."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"} or "text" in item:
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "tool_use":
                    payload = json.dumps(item.get("input", {}), default=str)[:600]
                    parts.append(f"[tool_use {item.get('name', '')}] {payload}")
                elif item.get("type") == "tool_result":
                    parts.append(f"[tool_result] {_flatten_content(item.get('content'))[:600]}")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if "parts" in content:
            return "\n".join(str(p) for p in content.get("parts") or [] if isinstance(p, str))
        if "text" in content:
            return str(content["text"])
    return ""


def _timestamp(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _date_from(timestamp: float | None) -> date | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC).date()
    except (OverflowError, OSError, ValueError):
        return None


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def parse_claude_session(path: Path) -> ImportedConversation | None:
    """Parse a Claude Code session JSONL file."""
    messages: list[ImportedMessage] = []
    title = ""
    session_id = path.stem
    first_ts: float | None = None
    cwd = ""

    for row in iter_jsonl(path):
        row_type = str(row.get("type") or row.get("role") or "")
        session_id = str(row.get("sessionId") or row.get("session_id") or session_id)
        cwd = str(row.get("cwd") or cwd)
        stamp = _timestamp(row.get("timestamp") or row.get("created_at"))
        if stamp and (first_ts is None or stamp < first_ts):
            first_ts = stamp

        if row_type == "summary" and row.get("summary"):
            title = title or str(row["summary"])
            continue

        payload = row.get("message") if isinstance(row.get("message"), dict) else row
        role = str(payload.get("role") or row_type or "").lower()
        if role not in {"user", "assistant", "system"}:
            continue
        text = _clean(_flatten_content(payload.get("content")))
        if not text:
            continue
        messages.append(ImportedMessage(role=role, text=text, timestamp=stamp))

    if not messages:
        return None
    if not title:
        title = _derive_title(messages)
    return ImportedConversation(
        source_tool="claude",
        conversation_id=session_id,
        title=title,
        messages=messages,
        started_at=_date_from(first_ts) or _date_from(path.stat().st_mtime),
        source_path=str(path),
        metadata={"cwd": cwd} if cwd else {},
    )


def parse_codex_session(path: Path) -> ImportedConversation | None:
    """Parse a Codex CLI rollout JSONL file."""
    messages: list[ImportedMessage] = []
    first_ts: float | None = None
    conversation_id = path.stem

    for row in iter_jsonl(path):
        stamp = _timestamp(row.get("timestamp") or row.get("created_at"))
        if stamp and (first_ts is None or stamp < first_ts):
            first_ts = stamp
        payload = row
        for key in ("payload", "item", "message"):
            nested = row.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
        conversation_id = str(
            row.get("conversation_id") or payload.get("conversation_id") or conversation_id
        )
        role = str(payload.get("role") or payload.get("type") or "").lower()
        if role not in {"user", "assistant", "system"}:
            continue
        text = _clean(_flatten_content(payload.get("content") or payload.get("text")))
        if not text:
            continue
        messages.append(ImportedMessage(role=role, text=text, timestamp=stamp))

    if not messages:
        return None
    return ImportedConversation(
        source_tool="codex",
        conversation_id=conversation_id,
        title=_derive_title(messages),
        messages=messages,
        started_at=_date_from(first_ts) or _date_from(path.stat().st_mtime),
        source_path=str(path),
    )


def parse_chatgpt_export(path: Path) -> list[ImportedConversation]:
    """Parse a ChatGPT ``conversations.json`` data export."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("chatgpt_export_unreadable", path=str(path), error=str(exc))
        return []

    conversations_raw: list[dict[str, Any]]
    if isinstance(raw, list):
        conversations_raw = [c for c in raw if isinstance(c, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("conversations"), list):
        conversations_raw = [c for c in raw["conversations"] if isinstance(c, dict)]
    elif isinstance(raw, dict):
        conversations_raw = [raw]
    else:
        return []

    out: list[ImportedConversation] = []
    for entry in conversations_raw:
        mapping = entry.get("mapping")
        nodes: list[dict[str, Any]] = []
        if isinstance(mapping, dict):
            nodes = [n for n in mapping.values() if isinstance(n, dict)]
        elif isinstance(entry.get("messages"), list):
            nodes = [{"message": m} for m in entry["messages"] if isinstance(m, dict)]

        collected: list[tuple[float, ImportedMessage]] = []
        for node in nodes:
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author")
            role = str((author or {}).get("role", "")).lower() if isinstance(author, dict) else ""
            role = role or str(message.get("role", "")).lower()
            if role not in {"user", "assistant", "system", "tool"}:
                continue
            text = _clean(_flatten_content(message.get("content")))
            if not text:
                continue
            stamp = _timestamp(message.get("create_time")) or 0.0
            collected.append(
                (stamp, ImportedMessage(role="assistant" if role == "tool" else role,
                                        text=text, timestamp=stamp or None))
            )
        if not collected:
            continue
        collected.sort(key=lambda item: item[0])
        messages = [message for _, message in collected]
        created = _timestamp(entry.get("create_time")) or collected[0][0] or None
        out.append(
            ImportedConversation(
                source_tool="chatgpt",
                conversation_id=str(entry.get("id") or entry.get("conversation_id") or path.stem),
                title=str(entry.get("title") or "").strip() or _derive_title(messages),
                messages=messages,
                started_at=_date_from(created) or _date_from(path.stat().st_mtime),
                source_path=str(path),
            )
        )
    return out


def parse_text_dump(path: Path, *, tool: str = "other") -> ImportedConversation | None:
    """Parse a generic Markdown or plain-text dump as a single-message note."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("text_dump_unreadable", path=str(path), error=str(exc))
        return None
    body = _clean(raw)
    if not body:
        return None
    title = ""
    for line in raw.splitlines():
        if line.strip().startswith("# "):
            title = line.strip()[2:].strip()
            break
    return ImportedConversation(
        source_tool=tool if tool in KNOWN_TOOLS else "other",
        conversation_id=path.stem,
        title=title or path.stem.replace("-", " ").replace("_", " ").title(),
        messages=[ImportedMessage(role="user", text=body)],
        started_at=_date_from(path.stat().st_mtime),
        source_path=str(path),
    )


def _derive_title(messages: list[ImportedMessage]) -> str:
    for message in messages:
        if message.role == "user" and message.text.strip():
            first_line = message.text.strip().splitlines()[0]
            return (first_line[:90] + ("..." if len(first_line) > 90 else "")) or "Imported note"
    return "Imported conversation"


# -- summarisation ---------------------------------------------------------


def extractive_summary(conversation: ImportedConversation) -> str:
    """Model-free candidate summary."""
    lines: list[str] = []

    requests = [m.text.strip() for m in conversation.user_messages if m.text.strip()]
    if requests:
        first = requests[0].splitlines()
        lines.append("## Original request\n")
        lines.append("\n".join(first[:6])[:800])
        lines.append("")

    commands: list[str] = []
    for message in conversation.messages:
        for match in _COMMAND_RE.finditer(message.text):
            command = match.group(1).strip()
            if command not in commands:
                commands.append(command)
    if commands:
        lines.append("## Commands referenced\n")
        lines.extend(f"- `{command}`" for command in commands[:20])
        lines.append("")

    claims: list[str] = []
    for message in conversation.assistant_messages:
        for sentence in _SENTENCE_RE.split(message.text):
            candidate = " ".join(sentence.split())
            plausible = 40 <= len(candidate) <= 300 and _CLAIM_HINT_RE.search(candidate)
            if plausible and candidate not in claims:
                claims.append(candidate)
    if claims:
        lines.append("## Candidate claims (unverified)\n")
        lines.extend(f"- {claim}" for claim in claims[:12])
        lines.append("")

    excerpts = [m for m in conversation.messages if len(m.text) > 120][:MAX_EXCERPTS]
    if excerpts:
        lines.append("## Excerpts\n")
        for message in excerpts:
            snippet = " ".join(message.text.split())[:MAX_EXCERPT_CHARS]
            lines.append(f"- **{message.role}:** {snippet}")
        lines.append("")

    if not lines:
        lines.append("_No extractable content; the transcript held no usable claims._")
    return "\n".join(lines).strip()


# -- writing ---------------------------------------------------------------


class ConversationImporter:
    """Turns exports into ``imports/<tool>/`` candidate notes."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        *,
        settings: Settings | None = None,
        summariser: Summariser | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or KnowledgeStore(settings=self.settings)
        self.summariser = summariser or extractive_summary

    # -- the one place an import may write --------------------------------

    def write_note(self, conversation: ImportedConversation, *, overwrite: bool = False) -> str:
        """Write one candidate note. Refuses any destination outside ``imports/``."""
        tool = conversation.source_tool if conversation.source_tool in KNOWN_TOOLS else "other"
        category = f"{IMPORT_ROOT}/{tool}"
        if not category.startswith(f"{IMPORT_ROOT}/"):  # pragma: no cover - defensive
            raise ValueError("imports may only be written under imports/")

        original = conversation.started_at
        prefix = original.isoformat() if original else datetime.now(tz=UTC).date().isoformat()
        doc_id = self.store.unique_doc_id(
            category, f"{prefix}-{slugify(conversation.title)}"
        )

        summary = redact(self.summariser(conversation))
        body = _render_import_body(conversation, summary)
        metadata = DocumentMetadata(
            title=conversation.title[:160],
            category=category,
            created_at=original,
            # Deliberately never set: an import has not been verified against
            last_verified=None,
            source=f"{tool} export: {Path(conversation.source_path).name}",
            confidence=Confidence.LOW,
            tags=["import", tool, "unverified"],
            verification_status=VerificationStatus.UNVERIFIED,
            imported_from=conversation.source_path,
            original_date=original,
            sources=[conversation.source_path] if conversation.source_path else [],
        )
        metadata.extra["conversation_id"] = conversation.conversation_id
        metadata.extra["imported_at"] = datetime.now(tz=UTC).date().isoformat()
        metadata.extra["message_count"] = len(conversation.messages)
        self.store.write(doc_id, metadata, body, overwrite=overwrite)
        log.info("import_note_written", doc_id=doc_id, tool=tool)
        return doc_id

    def _run(self, tool: str, conversations: Iterable[ImportedConversation]) -> ImportResult:
        result = ImportResult(tool=tool)
        for conversation in conversations:
            result.conversations_seen += 1
            if len(conversation.messages) < 2 and tool != "other":
                result.skipped += 1
                continue
            try:
                result.doc_ids.append(self.write_note(conversation))
                result.notes_written += 1
            except (OSError, ValueError) as exc:
                result.errors.append(f"{conversation.conversation_id}: {exc}")
        log.info("import_complete", summary=result.summary())
        return result

    # -- per-tool entry points --------------------------------------------

    def import_claude(self, path: Path, *, limit: int | None = None) -> ImportResult:
        files = _jsonl_files(path, limit)
        parsed = (parse_claude_session(f) for f in files)
        return self._run("claude", (c for c in parsed if c is not None))

    def import_codex(self, path: Path, *, limit: int | None = None) -> ImportResult:
        files = _jsonl_files(path, limit)
        parsed = (parse_codex_session(f) for f in files)
        return self._run("codex", (c for c in parsed if c is not None))

    def import_chatgpt(self, path: Path, *, limit: int | None = None) -> ImportResult:
        target = Path(path)
        if target.is_dir():
            candidates = sorted(target.rglob("conversations.json"))
        else:
            candidates = [target]
        conversations: list[ImportedConversation] = []
        for candidate in candidates:
            conversations.extend(parse_chatgpt_export(candidate))
        if limit:
            conversations = conversations[:limit]
        return self._run("chatgpt", conversations)

    def import_text(
        self, path: Path, *, tool: str = "other", limit: int | None = None
    ) -> ImportResult:
        target = Path(path)
        files = (
            sorted(f for f in target.rglob("*") if f.suffix.lower() in {".md", ".markdown", ".txt"})
            if target.is_dir()
            else [target]
        )
        if limit:
            files = files[:limit]
        parsed = (parse_text_dump(f, tool=tool) for f in files)
        return self._run(tool if tool in KNOWN_TOOLS else "other", (c for c in parsed if c))

    def import_path(self, path: Path, *, tool: str = "", limit: int | None = None) -> ImportResult:
        """Dispatch on the shape of ``path`` when the caller did not say."""
        target = Path(path).expanduser()
        chosen = (tool or "").strip().lower()
        if chosen == "claude":
            return self.import_claude(target, limit=limit)
        if chosen == "codex":
            return self.import_codex(target, limit=limit)
        if chosen == "chatgpt":
            return self.import_chatgpt(target, limit=limit)
        if chosen in {"text", "other", "markdown"}:
            return self.import_text(target, tool="other", limit=limit)
        if target.is_file() and target.name == "conversations.json":
            return self.import_chatgpt(target, limit=limit)
        if target.is_file() and target.suffix == ".jsonl":
            return self.import_claude(target, limit=limit)
        if target.is_dir() and any(target.rglob("conversations.json")):
            return self.import_chatgpt(target, limit=limit)
        if target.is_dir() and any(target.rglob("*.jsonl")):
            return self.import_claude(target, limit=limit)
        return self.import_text(target, tool="other", limit=limit)


def _jsonl_files(path: Path, limit: int | None) -> list[Path]:
    target = Path(path).expanduser()
    files = sorted(target.rglob("*.jsonl")) if target.is_dir() else [target]
    return files[:limit] if limit else files


def _render_import_body(conversation: ImportedConversation, summary: str) -> str:
    original = conversation.started_at.isoformat() if conversation.started_at else "unknown"
    header = [
        f"# {conversation.title}",
        "",
        f"> Imported from **{conversation.source_tool}** on "
        f"{datetime.now(tz=UTC).date().isoformat()}. Original conversation date: {original}.",
        "> This is untrusted source material (ADR 11.5, NG4). Nothing here has been checked "
        "against a live system.",
        "> Verify a claim before acting on it, then promote it with "
        "`promote_memory_note`.",
        "",
    ]
    return "\n".join(header) + "\n" + summary + "\n"


def discover_default_exports() -> dict[str, list[Path]]:
    """Best-effort lookup of the usual local export locations."""
    home = Path.home()
    found: dict[str, list[Path]] = {}
    claude = home / ".claude" / "projects"
    if claude.is_dir():
        found["claude"] = sorted(claude.rglob("*.jsonl"))[:500]
    codex = home / ".codex" / "sessions"
    if codex.is_dir():
        found["codex"] = sorted(codex.rglob("*.jsonl"))[:500]
    downloads = home / "Downloads"
    if downloads.is_dir():
        found["chatgpt"] = sorted(downloads.glob("*/conversations.json"))[:20]
    return {tool: paths for tool, paths in found.items() if paths}
