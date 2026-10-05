"""Import curated agent memory files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

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

MEMORY_ROOT = "imports/agent-memory"

MAX_BYTES = 60_000
"""A curated memory is a paragraph."""

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# An index file lists the others.
_INDEX_NAMES = frozenset({"memory.md", "index.md", "readme.md"})


@dataclass
class MemoryFile:
    path: Path
    title: str
    body: str
    kind: str = "note"
    """The tool's own classification: user, feedback, project, reference."""

    project: str = ""
    modified: date | None = None

    @property
    def source_tool(self) -> str:
        parts = {p.lower() for p in self.path.parts}
        if ".claude" in parts:
            return "claude"
        if ".codex" in parts:
            return "codex"
        if ".config" in parts and "gpt" in str(self.path).lower():
            return "chatgpt"
        return "other"


@dataclass
class MemoryImportResult:
    files_seen: int = 0
    notes_written: int = 0
    skipped: int = 0
    doc_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.notes_written} written, {self.skipped} skipped, "
            f"{len(self.errors)} failed, of {self.files_seen} seen"
        )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Read the small flat frontmatter these files use."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lstrip("- ")
        value = value.strip().strip("\"'")
        if key and value:
            fields.setdefault(key, value)
    return fields, text[match.end() :]


def _project_of(path: Path) -> str:
    """Recover the working directory from the mangled project folder name."""
    for part in path.parts:
        if not (part.startswith("-Users-") or part.startswith("-home-")):
            continue
        bits = [b for b in part.split("-") if b]
        if bits[:1] in (["Users"], ["home"]) and len(bits) > 2:
            bits = bits[2:]
        if bits[:1] == ["Documents"] and len(bits) > 1:
            bits = bits[1:]
        return "-".join(bits)
    return ""


def parse_memory_file(path: Path) -> MemoryFile | None:
    """Read one curated memory file, or ``None`` if it is not one."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not raw.strip():
        return None

    fields, body = _parse_frontmatter(raw)
    body = body.strip()
    if not body:
        return None

    title = fields.get("description") or fields.get("name") or ""
    if not title:
        heading = next(
            (line.lstrip("# ").strip() for line in body.splitlines() if line.startswith("#")),
            "",
        )
        title = heading or path.stem.replace("-", " ")

    project = _project_of(path)

    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    except OSError:
        modified = None

    return MemoryFile(
        path=path,
        # Redacted like the body.
        title=redact(title)[:160],
        body=body,
        kind=fields.get("type", "note"),
        project=project,
        modified=modified,
    )


def discover_memory_files(home: Path | None = None) -> list[Path]:
    """Where these tools keep curated memory on this machine."""
    home = home or Path.home()
    found: list[Path] = []
    roots = (
        home / ".claude" / "projects",
        home / ".codex" / "projects",
    )
    for root in roots:
        if root.is_dir():
            found.extend(sorted(root.glob("*/memory/*.md")))
    for single in (home / ".claude" / "CLAUDE.md", home / ".codex" / "AGENTS.md"):
        if single.is_file():
            found.append(single)
    return [p for p in found if p.name.lower() not in _INDEX_NAMES]


class AgentMemoryImporter:
    """Writes curated memory into ``imports/agent-memory/``, and nowhere else."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or KnowledgeStore(settings=self.settings)

    def write_note(self, memory: MemoryFile, *, overwrite: bool = False) -> str:
        category = f"{MEMORY_ROOT}/{memory.source_tool}"
        if not category.startswith(f"{MEMORY_ROOT}/"):  # pragma: no cover - defensive
            raise ValueError("agent memory may only be written under imports/agent-memory/")

        doc_id = self.store.unique_doc_id(
            category, slugify(memory.project or "") + "-" + memory.path.stem
            if memory.project
            else memory.path.stem,
        )
        metadata = DocumentMetadata(
            title=memory.title,
            category=category,
            created_at=memory.modified,
            last_verified=None,
            source=f"{memory.source_tool} memory: {memory.path.name}",
            # Higher than a transcript summary and lower than anything MIMIR
            confidence=Confidence.MEDIUM,
            tags=sorted({"import", "agent-memory", memory.source_tool, memory.kind}),
            verification_status=VerificationStatus.UNVERIFIED,
            imported_from=str(memory.path),
            original_date=memory.modified,
            sources=[str(memory.path)],
        )
        if memory.project:
            metadata.extra["project"] = memory.project
        metadata.extra["imported_at"] = datetime.now(tz=UTC).date().isoformat()

        self.store.write(doc_id, metadata, redact(memory.body), overwrite=overwrite)
        return doc_id

    def run(
        self, paths: list[Path] | None = None, *, overwrite: bool = False
    ) -> MemoryImportResult:
        result = MemoryImportResult()
        for path in paths if paths is not None else discover_memory_files():
            result.files_seen += 1
            memory = parse_memory_file(path)
            if memory is None:
                result.skipped += 1
                continue
            try:
                result.doc_ids.append(self.write_note(memory, overwrite=overwrite))
                result.notes_written += 1
            except (OSError, ValueError) as exc:
                result.errors.append(f"{path.name}: {exc}")
        log.info("agent_memory_import", summary=result.summary())
        return result


__all__ = [
    "AgentMemoryImporter",
    "MemoryFile",
    "MemoryImportResult",
    "discover_memory_files",
    "parse_memory_file",
]
