"""Markdown memory store (ADR 11.1 layers, 11.2 layout, 11.3 metadata, G5).

Every unit of curated knowledge is a Markdown file on disk with optional YAML
frontmatter. Nothing here is a database: the files are the source of truth so
they stay diffable, reviewable, and editable without MIMIR running.

Three ideas carry the module:

* :class:`MemoryLayer` maps a directory under ``knowledge/`` to one of the ADR
  11.1 layers, and each layer maps onto a :class:`SourceType` so retrieval can
  rank documents with the ADR 11.4 trust ladder already encoded in
  :data:`mimir.models.evidence.TRUST_ORDER`.
* :class:`DocumentMetadata` is the ADR 11.3 frontmatter, parsed leniently and
  written back in a stable field order.
* :func:`compute_freshness` turns ``last_verified`` and ``expires_after`` into a
  :class:`Freshness` value. ADR R2 asks for visible freshness metadata, not
  suppression, so a stale document is still returned; it is simply marked.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, field_validator

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.evidence import Citation, Freshness, SourceType

log = get_logger(__name__)

MARKDOWN_SUFFIXES = (".md", ".markdown")


class MemoryLayer(StrEnum):
    """ADR 11.1 layers, named by their ADR 11.2 directory."""

    STABLE = "stable"
    RUNBOOKS = "runbooks"
    HISTORY_INCIDENTS = "history/incidents"
    HISTORY_INVESTIGATIONS = "history/investigations"
    IMPORTS = "imports"
    SESSIONS = "sessions"
    SKILLS = "skills"
    UNKNOWN = "unknown"


#: ADR 11.4 mapping. Stable knowledge and runbooks are both curated and reviewed
#: so they share the runbook tier; anything imported or session-derived sits on
#: the "curated imported memory" rung until it is promoted.
LAYER_SOURCE_TYPE: dict[MemoryLayer, SourceType] = {
    MemoryLayer.STABLE: SourceType.RUNBOOK,
    MemoryLayer.RUNBOOKS: SourceType.RUNBOOK,
    MemoryLayer.SKILLS: SourceType.RUNBOOK,
    MemoryLayer.HISTORY_INCIDENTS: SourceType.HISTORICAL_INCIDENT,
    MemoryLayer.HISTORY_INVESTIGATIONS: SourceType.HISTORICAL_INCIDENT,
    MemoryLayer.IMPORTS: SourceType.IMPORTED_MEMORY,
    MemoryLayer.SESSIONS: SourceType.IMPORTED_MEMORY,
    MemoryLayer.UNKNOWN: SourceType.IMPORTED_MEMORY,
}

#: Tie-break inside a single trust tier. Lower wins.
LAYER_RANK: dict[MemoryLayer, int] = {
    MemoryLayer.STABLE: 0,
    MemoryLayer.RUNBOOKS: 1,
    MemoryLayer.SKILLS: 2,
    MemoryLayer.HISTORY_INCIDENTS: 3,
    MemoryLayer.HISTORY_INVESTIGATIONS: 4,
    MemoryLayer.SESSIONS: 5,
    MemoryLayer.IMPORTS: 6,
    MemoryLayer.UNKNOWN: 7,
}

#: Layers that require an explicit human approval before anything is written
#: into them (ADR 11.6 step 4, NG4).
CURATED_LAYERS: frozenset[MemoryLayer] = frozenset(
    {MemoryLayer.STABLE, MemoryLayer.RUNBOOKS, MemoryLayer.SKILLS}
)

#: ADR 11.2 directory layout, created on demand.
DIRECTORY_LAYOUT: tuple[str, ...] = (
    "stable/services",
    "stable/repositories",
    "stable/environments",
    "stable/conventions",
    "runbooks/kubernetes",
    "runbooks/sdm",
    "runbooks/kannel",
    "runbooks/tankers",
    "runbooks/databases",
    "runbooks/incidents",
    "history/incidents",
    "history/investigations",
    "imports/claude",
    "imports/codex",
    "imports/chatgpt",
    "skills",
    "sessions",
)


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def weight(self) -> float:
        return {"low": 0.35, "medium": 0.65, "high": 0.9}[self.value]


class VerificationStatus(StrEnum):
    """ADR 11.6 step 3. The default is deliberately the pessimistic one."""

    UNVERIFIED = "unverified"
    USER_CONFIRMED = "user_confirmed"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([dwmy]?)\s*$", re.IGNORECASE)
_DURATION_DAYS = {"": 1, "d": 1, "w": 7, "m": 30, "y": 365}


def parse_duration_days(value: str | int | None) -> int | None:
    """Parse ``180``, ``"180d"``, ``"6m"``, ``"1y"`` into a day count."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    match = _DURATION_RE.match(str(value))
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    return amount * _DURATION_DAYS.get(unit, 1)


def _coerce_date(value: Any) -> date | None:
    if value in (None, "", []):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC).date()
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[: len(pattern) + 6], pattern).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _coerce_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\n]", value)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, list | tuple | set):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


class DocumentMetadata(BaseModel):
    """ADR 11.3 frontmatter. Parsed leniently, validated strictly once parsed."""

    title: str = ""
    category: str = ""
    service: str | None = None
    environment: str | None = None
    created_at: date | None = None
    last_verified: date | None = None
    source: str = ""
    confidence: Confidence = Confidence.MEDIUM
    owner: str | None = None
    supersedes: list[str] = Field(default_factory=list)
    expires_after: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Extensions beyond ADR 11.3, needed by 11.5 (imports) and 11.6 (promotion).
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    contradicts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    imported_from: str | None = None
    original_date: date | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    #: Written in this order so a hand-edited file keeps a predictable shape.
    ADR_FIELDS: ClassVar[tuple[str, ...]] = (
        "title",
        "category",
        "service",
        "environment",
        "created_at",
        "last_verified",
        "source",
        "confidence",
        "owner",
        "supersedes",
        "expires_after",
        "tags",
    )
    EXTENSION_FIELDS: ClassVar[tuple[str, ...]] = (
        "verification_status",
        "contradicts",
        "sources",
        "imported_from",
        "original_date",
    )

    @field_validator("created_at", "last_verified", "original_date", mode="before")
    @classmethod
    def _dates(cls, v: Any) -> Any:
        return _coerce_date(v)

    @field_validator("supersedes", "tags", "contradicts", "sources", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> Any:
        return _coerce_list(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: Any) -> Any:
        if v in (None, ""):
            return Confidence.MEDIUM
        if isinstance(v, int | float):
            # MemoryProposal carries a float; bucket it rather than reject it.
            if v >= 0.75:
                return Confidence.HIGH
            return Confidence.LOW if v < 0.4 else Confidence.MEDIUM
        text = str(v).strip().lower()
        if text in {"low", "medium", "high"}:
            return text
        if text in {"unknown", "none"}:
            return Confidence.LOW
        return Confidence.MEDIUM

    @field_validator("verification_status", mode="before")
    @classmethod
    def _verification(cls, v: Any) -> Any:
        if v in (None, ""):
            return VerificationStatus.UNVERIFIED
        text = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        if text in {s.value for s in VerificationStatus}:
            return text
        if text in {"true", "yes", "confirmed"}:
            return VerificationStatus.VERIFIED
        return VerificationStatus.UNVERIFIED

    @field_validator("expires_after", mode="before")
    @classmethod
    def _expires(cls, v: Any) -> Any:
        return None if v in (None, "") else str(v).strip()

    @property
    def expires_after_days(self) -> int | None:
        return parse_duration_days(self.expires_after)

    @classmethod
    def from_frontmatter(cls, data: dict[str, Any] | None) -> DocumentMetadata:
        """Build metadata from raw YAML, keeping unknown keys in ``extra``."""
        raw = dict(data or {})
        known = set(cls.model_fields) - {"extra"}
        payload = {k: v for k, v in raw.items() if k in known}
        leftovers = {k: v for k, v in raw.items() if k not in known}
        # Common aliases seen in hand-written notes.
        for alias, target in (("date", "created_at"), ("verified", "last_verified")):
            if target not in payload and alias in leftovers:
                payload[target] = leftovers.pop(alias)
        meta = cls(**payload)
        meta.extra = leftovers
        return meta

    def to_frontmatter(self) -> dict[str, Any]:
        """Serialisable mapping in ADR field order, empty values dropped."""
        out: dict[str, Any] = {}
        for name in (*self.ADR_FIELDS, *self.EXTENSION_FIELDS):
            value = getattr(self, name)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, date):
                out[name] = value.isoformat()
            elif isinstance(value, StrEnum):
                out[name] = value.value
            else:
                out[name] = value
        for key, value in self.extra.items():
            out.setdefault(key, value)
        return out


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split a ``---`` fenced YAML header from the body. Missing header is fine."""
    if not text.startswith("---"):
        return None, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            header = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            try:
                data = yaml.safe_load(header) if header.strip() else {}
            except yaml.YAMLError as exc:
                log.warning("frontmatter_parse_failed", error=str(exc))
                return None, text
            if not isinstance(data, dict):
                return None, text
            return data, body.lstrip("\n")
    return None, text


_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def infer_title(body: str, path: Path) -> str:
    match = _H1_RE.search(body)
    if match:
        return match.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").strip().title()


def layer_for_relative(relative: Path) -> MemoryLayer:
    parts = relative.as_posix().split("/")
    if not parts:
        return MemoryLayer.UNKNOWN
    head = parts[0]
    if head == "history" and len(parts) > 1:
        if parts[1] == "incidents":
            return MemoryLayer.HISTORY_INCIDENTS
        if parts[1] == "investigations":
            return MemoryLayer.HISTORY_INVESTIGATIONS
        return MemoryLayer.HISTORY_INVESTIGATIONS
    for layer in (
        MemoryLayer.STABLE,
        MemoryLayer.RUNBOOKS,
        MemoryLayer.IMPORTS,
        MemoryLayer.SESSIONS,
        MemoryLayer.SKILLS,
    ):
        if head == layer.value:
            return layer
    return MemoryLayer.UNKNOWN


def slugify(text: str, *, max_len: int = 72) -> str:
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return (slug[:max_len].rstrip("-")) or "note"


def compute_freshness(
    metadata: DocumentMetadata,
    *,
    stale_after_days: int,
    today: date | None = None,
) -> Freshness:
    """ADR R2. Age is reported, never used to hide a document."""
    now = today or datetime.now(tz=UTC).date()
    verified = metadata.last_verified or metadata.created_at
    if verified is None:
        return Freshness.UNKNOWN

    explicit_window = metadata.expires_after_days
    age_days = (now - verified).days
    if explicit_window is not None and age_days > explicit_window:
        return Freshness.STALE
    if age_days > stale_after_days:
        return Freshness.STALE
    if age_days <= 1:
        return Freshness.LIVE
    return Freshness.RECENT


@dataclass(slots=True)
class MemoryDocument:
    """One Markdown file plus everything derived from its location."""

    doc_id: str
    """Root-relative path with the extension dropped, for example
    ``runbooks/kubernetes/pod-restart-investigation``."""

    path: Path
    root: Path
    metadata: DocumentMetadata
    body: str
    layer: MemoryLayer
    mtime: float
    size: int
    content_hash: str

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(self.root).as_posix()

    @property
    def source_type(self) -> SourceType:
        return LAYER_SOURCE_TYPE.get(self.layer, SourceType.IMPORTED_MEMORY)

    @property
    def layer_rank(self) -> int:
        return LAYER_RANK.get(self.layer, 9)

    @property
    def title(self) -> str:
        return self.metadata.title or infer_title(self.body, self.path)

    def freshness(self, stale_after_days: int, today: date | None = None) -> Freshness:
        return compute_freshness(
            self.metadata, stale_after_days=stale_after_days, today=today
        )

    def citation(self, heading_path: str = "", line: int | None = None) -> Citation:
        locator = self.doc_id if not heading_path else f"{self.doc_id}#{heading_path}"
        return Citation(
            source_type=self.source_type,
            locator=locator,
            path=self.relative_path,
            start_line=line,
            title=self.title,
            retrieved_at=self.mtime,
        )

    def render(self) -> str:
        return render_document(self.metadata, self.body)


def render_document(metadata: DocumentMetadata, body: str) -> str:
    """Frontmatter plus body, with field order preserved."""
    front = metadata.to_frontmatter()
    header = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=100).strip()
    return f"---\n{header}\n---\n\n{body.strip()}\n"


class KnowledgeStore:
    """CRUD over the ADR 11.2 directory tree."""

    def __init__(self, root: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = Path(root or self.settings.knowledge.root).expanduser()

    # -- layout ----------------------------------------------------------

    def ensure_layout(self) -> list[Path]:
        created: list[Path] = []
        for relative in DIRECTORY_LAYOUT:
            path = self.root / relative
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(path)
        return created

    def resolve(self, doc_id: str) -> Path:
        """Map a document id or root-relative path to an absolute path.

        Refuses anything that escapes the knowledge root: document ids reach
        this method from model output.
        """
        cleaned = str(doc_id).strip().lstrip("/")
        if not cleaned:
            raise ValueError("empty document id")
        candidate = Path(cleaned)
        if candidate.suffix.lower() not in MARKDOWN_SUFFIXES:
            candidate = candidate.with_suffix(".md")
        absolute = (self.root / candidate).resolve()
        root = self.root.resolve()
        if not absolute.is_relative_to(root):
            raise ValueError(f"document id escapes the knowledge root: {doc_id}")
        return absolute

    def doc_id_for(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.root.resolve())
        return relative.with_suffix("").as_posix()

    # -- read ------------------------------------------------------------

    def iter_paths(self, layer: MemoryLayer | None = None) -> Iterator[Path]:
        base = self.root
        if layer is not None and layer is not MemoryLayer.UNKNOWN:
            base = self.root / layer.value
        if not base.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                if Path(name).suffix.lower() in MARKDOWN_SUFFIXES:
                    yield Path(dirpath) / name

    def load(self, path: Path) -> MemoryDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        front, body = split_frontmatter(raw)
        metadata = DocumentMetadata.from_frontmatter(front)
        relative = path.resolve().relative_to(self.root.resolve())
        if not metadata.title:
            metadata.title = infer_title(body, path)
        if not metadata.category:
            metadata.category = relative.parent.as_posix() if relative.parent != Path(".") else ""
        stat = path.stat()
        return MemoryDocument(
            doc_id=relative.with_suffix("").as_posix(),
            path=path,
            root=self.root,
            metadata=metadata,
            body=body,
            layer=layer_for_relative(relative),
            mtime=stat.st_mtime,
            size=stat.st_size,
            content_hash=hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:32],
        )

    def get(self, doc_id: str) -> MemoryDocument | None:
        try:
            path = self.resolve(doc_id)
        except ValueError:
            return None
        if not path.is_file():
            return None
        return self.load(path)

    def documents(self, layer: MemoryLayer | None = None) -> list[MemoryDocument]:
        out: list[MemoryDocument] = []
        for path in self.iter_paths(layer):
            try:
                out.append(self.load(path))
            except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
                log.warning("document_load_failed", path=str(path), error=str(exc))
        return out

    def list(
        self,
        *,
        layer: MemoryLayer | None = None,
        category: str | None = None,
        service: str | None = None,
        environment: str | None = None,
        tag: str | None = None,
        limit: int = 200,
    ) -> list[MemoryDocument]:
        out: list[MemoryDocument] = []
        for doc in self.documents(layer):
            meta = doc.metadata
            if category and not meta.category.startswith(category):
                continue
            if service and (meta.service or "").lower() != service.lower():
                continue
            if environment and (meta.environment or "").lower() != environment.lower():
                continue
            if tag and tag.lower() not in {t.lower() for t in meta.tags}:
                continue
            out.append(doc)
        out.sort(key=lambda d: (d.layer_rank, d.doc_id))
        return out[:limit]

    # -- write -----------------------------------------------------------

    def write(
        self,
        doc_id: str,
        metadata: DocumentMetadata,
        body: str,
        *,
        overwrite: bool = False,
    ) -> MemoryDocument:
        path = self.resolve(doc_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"document already exists: {doc_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        relative = path.resolve().relative_to(self.root.resolve())
        if not metadata.category:
            metadata.category = relative.parent.as_posix() if relative.parent != Path(".") else ""
        if metadata.created_at is None:
            metadata.created_at = datetime.now(tz=UTC).date()
        path.write_text(render_document(metadata, body), encoding="utf-8")
        log.info("memory_document_written", doc_id=doc_id, layer=layer_for_relative(relative).value)
        return self.load(path)

    def unique_doc_id(self, directory: str, title: str, *, suffix: str = "") -> str:
        """Allocate a free ``<directory>/<slug>`` id, appending ``-2``, ``-3``..."""
        base = slugify(f"{title}{('-' + suffix) if suffix else ''}")
        candidate = f"{directory.strip('/')}/{base}"
        counter = 2
        while self.resolve(candidate).exists():
            candidate = f"{directory.strip('/')}/{base}-{counter}"
            counter += 1
        return candidate

    def delete(self, doc_id: str) -> bool:
        path = self.resolve(doc_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def touch_verification(self, doc_id: str, *, when: date | None = None) -> MemoryDocument | None:
        """Re-stamp ``last_verified``. This is how a stale document is refreshed."""
        doc = self.get(doc_id)
        if doc is None:
            return None
        doc.metadata.last_verified = when or datetime.now(tz=UTC).date()
        return self.write(doc_id, doc.metadata, doc.body, overwrite=True)


_store: KnowledgeStore | None = None


def get_knowledge_store(settings: Settings | None = None) -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore(settings=settings)
    return _store


def reset_knowledge_store() -> None:
    global _store
    _store = None
