"""Incremental chunk index over the Markdown memory (ADR 11.2, G5, R7).

The files on disk stay the source of truth; this is a derived cache that can be
deleted and rebuilt at any time. It lives under ``$MIMIR_HOME/cache`` rather
than inside ``knowledge/`` so the knowledge tree stays clean in version control.

Design points:

* Chunks, not whole documents, are indexed. A runbook is a sequence of steps and
  the useful retrieval unit is a section, not a 400-line file (ADR R7 context
  explosion).
* Splitting is heading-aware so every chunk keeps a heading path that can be
  shown to the user and turned into a :class:`~mimir.models.evidence.Citation`.
* Reindexing is incremental: a document is re-chunked only when its content hash
  changes, so a full sweep over an unchanged tree costs one ``stat`` per file.
* Keyword search uses SQLite FTS5 when the local build has it and degrades to a
  LIKE scan when it does not.
* Embeddings are optional and stored as float32 BLOBs; similarity is computed in
  numpy rather than by an extension, which keeps the dependency surface at zero.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mimir.config import Settings, get_settings
from mimir.knowledge.embeddings import Embedder, cosine_scores, from_blob, get_embedder, to_blob
from mimir.knowledge.store import KnowledgeStore, MemoryDocument, MemoryLayer
from mimir.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "1"
DEFAULT_MAX_CHUNK_CHARS = 1500
DEFAULT_MIN_CHUNK_CHARS = 60
MAX_VECTOR_CANDIDATES = 20000

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class Chunk:
    """One retrievable section of a document."""

    ordinal: int
    heading_path: str
    text: str
    start_line: int
    end_line: int

    def chunk_id(self, doc_id: str) -> str:
        return f"{doc_id}::{self.ordinal}"


def chunk_markdown(
    body: str,
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split on headings, then on paragraph boundaries when a section is long."""
    lines = body.splitlines()
    chunks: list[Chunk] = []
    stack: list[str] = []
    buffer: list[str] = []
    buffer_start = 1
    heading_path = ""
    in_fence = False

    def flush(end_line: int) -> None:
        nonlocal buffer, buffer_start
        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            buffer_start = end_line + 1
            return
        if len(text) >= min_chars or not chunks:
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    heading_path=heading_path,
                    text=text,
                    start_line=buffer_start,
                    end_line=end_line,
                )
            )
        else:
            # Too small to stand alone: fold it into the previous chunk so a one
            # line section is still retrievable.
            previous = chunks[-1]
            previous.text = f"{previous.text}\n\n{text}"
            previous.end_line = end_line
        buffer = []
        buffer_start = end_line + 1

    for number, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading is not None:
            flush(number - 1)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            heading_path = " > ".join(part for part in stack if part)
            buffer_start = number
            continue

        buffer.append(line)
        current_len = sum(len(entry) + 1 for entry in buffer)
        if current_len >= max_chars and not in_fence and not line.strip():
            flush(number)

    flush(len(lines))
    return chunks


@dataclass(slots=True)
class MemoryFilters:
    """Retrieval filters shared by the index and the retriever."""

    layers: tuple[MemoryLayer, ...] = ()
    category: str | None = None
    service: str | None = None
    environment: str | None = None
    tags: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    exclude_layers: tuple[MemoryLayer, ...] = ()

    def where(self) -> tuple[str, list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if self.layers:
            placeholders = ",".join("?" for _ in self.layers)
            clauses.append(f"d.layer IN ({placeholders})")
            params.extend(layer.value for layer in self.layers)
        if self.exclude_layers:
            placeholders = ",".join("?" for _ in self.exclude_layers)
            clauses.append(f"d.layer NOT IN ({placeholders})")
            params.extend(layer.value for layer in self.exclude_layers)
        if self.category:
            clauses.append("d.category LIKE ?")
            params.append(f"{self.category.strip('/')}%")
        if self.service:
            clauses.append("lower(d.service) = ?")
            params.append(self.service.lower())
        if self.environment:
            clauses.append("lower(d.environment) = ?")
            params.append(self.environment.lower())
        for tag in self.tags:
            clauses.append("d.tags LIKE ?")
            params.append(f"%|{tag.lower()}|%")
        if self.doc_ids:
            placeholders = ",".join("?" for _ in self.doc_ids)
            clauses.append(f"d.doc_id IN ({placeholders})")
            params.extend(self.doc_ids)
        return (" AND ".join(clauses), params)


@dataclass(slots=True)
class ChunkHit:
    """A chunk plus the denormalised document row needed for ranking."""

    chunk_id: str
    doc_id: str
    ordinal: int
    heading_path: str
    text: str
    start_line: int
    end_line: int
    layer: str
    title: str
    category: str
    service: str | None
    environment: str | None
    tags: tuple[str, ...]
    score: float = 0.0


@dataclass(slots=True)
class IndexStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    embedded: int = 0
    duration_s: float = 0.0
    embedder: str = "none"
    fts: bool = True
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.added} added, {self.updated} updated, {self.unchanged} unchanged, "
            f"{self.removed} removed, {self.chunks} chunks, {self.embedded} embedded "
            f"({self.duration_s:.2f}s, keyword={'fts5' if self.fts else 'like'}, "
            f"embedder={self.embedder})"
        )


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts_probe")
    except sqlite3.OperationalError:
        return False
    return True


def _fts_query(query: str) -> str:
    """Build a permissive FTS5 MATCH expression.

    Tokens are OR-ed rather than AND-ed: recall matters more than precision here
    because the hybrid scorer, trust ladder, and freshness pass all re-rank
    afterwards.
    """
    tokens = [t for t in _FTS_TOKEN_RE.findall(query) if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:24])


class KnowledgeIndex:
    """SQLite-backed keyword plus vector index over the memory store."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        *,
        settings: Settings | None = None,
        db_path: Path | None = None,
        embedder: Embedder | None = None,
        use_embeddings: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or KnowledgeStore(settings=self.settings)
        self.db_path = Path(
            db_path or (self.settings.cache_dir / "knowledge-index.sqlite")
        ).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self.fts_enabled = _fts5_available(self._conn)
        enabled = (
            self.settings.knowledge.embeddings_enabled if use_embeddings is None else use_embeddings
        )
        self.embedder: Embedder | None = embedder if enabled else None
        if enabled and self.embedder is None:
            self.embedder = get_embedder(self.settings)
        self._ensure_schema()

    # -- schema ----------------------------------------------------------

    def _ensure_schema(self) -> None:
        cur = self._conn
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id              TEXT PRIMARY KEY,
                path                TEXT NOT NULL,
                layer               TEXT NOT NULL,
                title               TEXT,
                category            TEXT,
                service             TEXT,
                environment         TEXT,
                tags                TEXT,
                source              TEXT,
                owner               TEXT,
                confidence          TEXT,
                verification_status TEXT,
                supersedes          TEXT,
                contradicts         TEXT,
                created_at          TEXT,
                last_verified       TEXT,
                expires_after       TEXT,
                mtime               REAL,
                size                INTEGER,
                content_hash        TEXT,
                indexed_at          REAL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                ordinal      INTEGER NOT NULL,
                heading_path TEXT,
                text         TEXT NOT NULL,
                start_line   INTEGER,
                end_line     INTEGER
            );
            CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id);
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                model    TEXT NOT NULL,
                dim      INTEGER NOT NULL,
                vector   BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        if self.fts_enabled:
            cur.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    heading_path,
                    text,
                    tokenize='porter unicode61'
                );
                """
            )
        stored_version = self._get_meta("schema_version")
        if stored_version not in (None, SCHEMA_VERSION):
            log.info("knowledge_index_schema_changed", old=stored_version, new=SCHEMA_VERSION)
            self.clear()
        self._set_meta("schema_version", SCHEMA_VERSION)
        self._conn.commit()

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def clear(self) -> None:
        self._conn.executescript(
            "DELETE FROM documents; DELETE FROM chunks; DELETE FROM embeddings;"
        )
        if self.fts_enabled:
            self._conn.execute("DELETE FROM chunks_fts")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- indexing --------------------------------------------------------

    def _delete_document(self, doc_id: str) -> None:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        ids = [(row["chunk_id"],) for row in rows]
        if ids:
            self._conn.executemany("DELETE FROM embeddings WHERE chunk_id = ?", ids)
            if self.fts_enabled:
                self._conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", ids)
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def _insert_document(self, doc: MemoryDocument) -> list[tuple[str, str]]:
        meta = doc.metadata
        tags = "|" + "|".join(sorted({t.lower() for t in meta.tags})) + "|" if meta.tags else ""
        self._conn.execute(
            """
            INSERT INTO documents (
                doc_id, path, layer, title, category, service, environment, tags, source,
                owner, confidence, verification_status, supersedes, contradicts, created_at,
                last_verified, expires_after, mtime, size, content_hash, indexed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                doc.doc_id,
                doc.relative_path,
                doc.layer.value,
                doc.title,
                meta.category,
                meta.service,
                meta.environment,
                tags,
                meta.source,
                meta.owner,
                meta.confidence.value,
                meta.verification_status.value,
                "|".join(meta.supersedes),
                "|".join(meta.contradicts),
                meta.created_at.isoformat() if meta.created_at else None,
                meta.last_verified.isoformat() if meta.last_verified else None,
                meta.expires_after,
                doc.mtime,
                doc.size,
                doc.content_hash,
                time.time(),
            ),
        )
        pending: list[tuple[str, str]] = []
        for chunk in chunk_markdown(doc.body):
            chunk_id = chunk.chunk_id(doc.doc_id)
            self._conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, ordinal, heading_path, text, "
                "start_line, end_line) VALUES (?,?,?,?,?,?,?)",
                (
                    chunk_id,
                    doc.doc_id,
                    chunk.ordinal,
                    chunk.heading_path,
                    chunk.text,
                    chunk.start_line,
                    chunk.end_line,
                ),
            )
            if self.fts_enabled:
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, title, heading_path, text) "
                    "VALUES (?,?,?,?)",
                    (chunk_id, doc.title, chunk.heading_path, chunk.text),
                )
            embed_text = f"{doc.title}\n{chunk.heading_path}\n{chunk.text}".strip()
            pending.append((chunk_id, embed_text))
        return pending

    def reindex(self, *, force: bool = False, embed: bool = True) -> IndexStats:
        """Sync the index with disk. Only changed documents are re-chunked."""
        started = time.perf_counter()
        stats = IndexStats(fts=self.fts_enabled, embedder=getattr(self.embedder, "name", "none"))

        known = {
            row["doc_id"]: row["content_hash"]
            for row in self._conn.execute("SELECT doc_id, content_hash FROM documents")
        }
        seen: set[str] = set()
        pending_embeddings: list[tuple[str, str]] = []

        for path in self.store.iter_paths():
            try:
                doc = self.store.load(path)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                stats.errors.append(f"{path}: {exc}")
                continue
            seen.add(doc.doc_id)
            previous = known.get(doc.doc_id)
            if previous == doc.content_hash and not force:
                stats.unchanged += 1
                continue
            self._delete_document(doc.doc_id)
            pending_embeddings.extend(self._insert_document(doc))
            if previous is None:
                stats.added += 1
            else:
                stats.updated += 1

        for doc_id in set(known) - seen:
            self._delete_document(doc_id)
            stats.removed += 1

        self._conn.commit()
        stats.chunks = int(
            self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        )

        if embed and self.embedder is not None:
            stats.embedded = self._sync_embeddings(pending_embeddings, force=force)

        stats.duration_s = time.perf_counter() - started
        log.info("knowledge_reindexed", summary=stats.summary())
        return stats

    def _sync_embeddings(self, pending: list[tuple[str, str]], *, force: bool) -> int:
        assert self.embedder is not None
        model_name = self.embedder.name
        if self._get_meta("embedder") != model_name or force:
            # A different model means the stored vectors are not comparable.
            self._conn.execute("DELETE FROM embeddings")
            self._set_meta("embedder", model_name)
            pending = [
                (row["chunk_id"], f"{row['title']}\n{row['heading_path']}\n{row['text']}".strip())
                for row in self._conn.execute(
                    "SELECT c.chunk_id, c.heading_path, c.text, d.title FROM chunks c "
                    "JOIN documents d ON d.doc_id = c.doc_id"
                )
            ]
        else:
            existing = {
                row["chunk_id"]
                for row in self._conn.execute("SELECT chunk_id FROM embeddings")
            }
            pending = [item for item in pending if item[0] not in existing]

        if not pending:
            self._conn.commit()
            return 0
        try:
            vectors = self.embedder.embed([text for _, text in pending])
        except Exception as exc:  # noqa: BLE001 - embedding is best-effort
            log.warning("embedding_failed", error=str(exc), model=model_name)
            self._conn.commit()
            return 0
        rows = [
            (chunk_id, model_name, len(vector), to_blob(vector))
            for (chunk_id, _), vector in zip(pending, vectors, strict=False)
        ]
        self._conn.executemany(
            "INSERT INTO embeddings (chunk_id, model, dim, vector) VALUES (?,?,?,?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET model=excluded.model, dim=excluded.dim, "
            "vector=excluded.vector",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # -- query -----------------------------------------------------------

    _HIT_COLUMNS = (
        "c.chunk_id, c.doc_id, c.ordinal, c.heading_path, c.text, c.start_line, c.end_line, "
        "d.layer, d.title, d.category, d.service, d.environment, d.tags"
    )

    @staticmethod
    def _row_to_hit(row: sqlite3.Row, score: float) -> ChunkHit:
        tags = tuple(t for t in (row["tags"] or "").split("|") if t)
        return ChunkHit(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            ordinal=row["ordinal"],
            heading_path=row["heading_path"] or "",
            text=row["text"],
            start_line=row["start_line"] or 1,
            end_line=row["end_line"] or 1,
            layer=row["layer"],
            title=row["title"] or row["doc_id"],
            category=row["category"] or "",
            service=row["service"],
            environment=row["environment"],
            tags=tags,
            score=score,
        )

    def keyword_search(
        self, query: str, *, limit: int = 50, filters: MemoryFilters | None = None
    ) -> list[ChunkHit]:
        filters = filters or MemoryFilters()
        where, params = filters.where()
        if self.fts_enabled:
            match = _fts_query(query)
            if not match:
                return []
            sql = (
                f"SELECT {self._HIT_COLUMNS}, "
                "bm25(chunks_fts, 0.0, 2.0, 1.5, 1.0) AS rank FROM chunks_fts "
                "JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
                "JOIN documents d ON d.doc_id = c.doc_id "
                "WHERE chunks_fts MATCH ?"
            )
            args: list[object] = [match]
            if where:
                sql += f" AND {where}"
                args.extend(params)
            sql += " ORDER BY rank LIMIT ?"
            args.append(limit)
            try:
                rows = self._conn.execute(sql, args).fetchall()
            except sqlite3.OperationalError as exc:  # pragma: no cover - malformed MATCH
                log.warning("fts_query_failed", error=str(exc), query=query)
                return []
            # bm25 returns negative numbers with the best match most negative.
            return [self._row_to_hit(row, -float(row["rank"])) for row in rows]
        return self._like_search(query, limit=limit, where=where, params=params)

    def _like_search(
        self, query: str, *, limit: int, where: str, params: list[object]
    ) -> list[ChunkHit]:
        """Fallback used when the local SQLite build has no FTS5."""
        tokens = [t.lower() for t in _FTS_TOKEN_RE.findall(query) if len(t) > 1][:12]
        if not tokens:
            return []
        sql = (
            f"SELECT {self._HIT_COLUMNS} FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id WHERE ("
            + " OR ".join("lower(c.text) LIKE ? OR lower(d.title) LIKE ?" for _ in tokens)
            + ")"
        )
        args: list[object] = []
        for token in tokens:
            args.extend([f"%{token}%", f"%{token}%"])
        if where:
            sql += f" AND {where}"
            args.extend(params)
        sql += " LIMIT ?"
        args.append(limit * 4)
        hits: list[ChunkHit] = []
        for row in self._conn.execute(sql, args).fetchall():
            haystack = f"{row['title']} {row['heading_path']} {row['text']}".lower()
            score = float(sum(haystack.count(token) for token in tokens))
            hits.append(self._row_to_hit(row, score))
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]

    def vector_search(
        self, query: str, *, limit: int = 50, filters: MemoryFilters | None = None
    ) -> list[ChunkHit]:
        if self.embedder is None:
            return []
        try:
            query_vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - never fail retrieval on the embedder
            log.warning("query_embedding_failed", error=str(exc))
            return []

        filters = filters or MemoryFilters()
        where, params = filters.where()
        sql = (
            f"SELECT {self._HIT_COLUMNS}, e.vector AS vector FROM embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id "
            "JOIN documents d ON d.doc_id = c.doc_id"
        )
        args: list[object] = []
        if where:
            sql += f" WHERE {where}"
            args.extend(params)
        sql += " LIMIT ?"
        args.append(MAX_VECTOR_CANDIDATES)
        rows = self._conn.execute(sql, args).fetchall()
        if not rows:
            return []
        matrix = np.vstack([from_blob(row["vector"]) for row in rows])
        if matrix.shape[1] != query_vector.shape[0]:
            log.warning(
                "embedding_dimension_mismatch",
                stored=int(matrix.shape[1]),
                query=int(query_vector.shape[0]),
            )
            return []
        scores = cosine_scores(query_vector, matrix)
        order = np.argsort(-scores)[:limit]
        return [self._row_to_hit(rows[int(i)], float(scores[int(i)])) for i in order]

    # -- inspection ------------------------------------------------------

    def document_row(self, doc_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()

    def document_rows(self, doc_ids: list[str]) -> dict[str, sqlite3.Row]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self._conn.execute(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", doc_ids
        ).fetchall()
        return {row["doc_id"]: row for row in rows}

    def superseded_by(self) -> dict[str, list[str]]:
        """Map ``target doc_id -> [documents that claim to supersede it]``.

        Cheap enough to run on every retrieval, and it catches the case that
        matters for ADR 11.5: a retrieved note has already been replaced by one
        the query did not surface.
        """
        out: dict[str, list[str]] = {}
        for row in self._conn.execute(
            "SELECT doc_id, supersedes FROM documents WHERE supersedes IS NOT NULL "
            "AND supersedes != ''"
        ):
            for target in str(row["supersedes"]).split("|"):
                key = target.strip()
                if key:
                    out.setdefault(key, []).append(row["doc_id"])
        return out

    def counts(self) -> dict[str, int]:
        return {
            "documents": int(
                self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            ),
            "chunks": int(self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]),
            "embeddings": int(
                self._conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
            ),
        }


_index: KnowledgeIndex | None = None


def get_knowledge_index(settings: Settings | None = None) -> KnowledgeIndex:
    global _index
    if _index is None:
        _index = KnowledgeIndex(settings=settings)
    return _index


def reset_knowledge_index() -> None:
    global _index
    if _index is not None:
        _index.close()
    _index = None
