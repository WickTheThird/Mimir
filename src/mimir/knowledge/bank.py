"""The working set: what MIMIR currently has in mind, and what it has let go."""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS activations (
    doc_id        TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    project       TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL DEFAULT '',
    hits          INTEGER NOT NULL DEFAULT 0,
    first_recall  REAL NOT NULL,
    last_recall   REAL NOT NULL,
    last_query    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS activations_last ON activations(last_recall);

CREATE TABLE IF NOT EXISTS work (
    doc_id     TEXT PRIMARY KEY,
    project    TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL DEFAULT '',
    happened   TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL DEFAULT '',
    stale      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS work_project ON work(project);
"""

HALF_LIFE_S = 6 * 3600.0
"""Six hours."""

ACTIVE_THRESHOLD = 0.25
"""Below this a document has left the working set. It stays in the store."""

MAX_WORKING_SET = 40
"""A cap as well as a threshold."""


@dataclass(frozen=True)
class Recalled:
    doc_id: str
    title: str
    project: str
    kind: str
    hits: int
    last_recall: float
    activation: float

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.last_recall)

    def render(self) -> str:
        return f"{self.title or self.doc_id} ({self.activation:.2f})"


def _activation(hits: int, last_recall: float, now: float | None = None) -> float:
    """Hits against an exponential half-life."""
    now = now if now is not None else time.time()
    elapsed = max(0.0, now - last_recall)
    return hits * math.exp(-elapsed * math.log(2) / HALF_LIFE_S)


class MemoryBank:
    """Activation over the knowledge store. Never writes to the store itself."""

    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path) if path else self.settings.home / "memory-bank.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- recall ----------------------------------------------------------

    def touch(self, doc_ids: Sequence[str], *, query: str = "", meta=None) -> int:
        """Mark documents as recalled now. One hit each, however many chunks."""
        meta = meta or {}
        now = time.time()
        touched = 0
        for doc_id in dict.fromkeys(d for d in doc_ids if d):
            info = meta.get(doc_id, {})
            self._db.execute(
                """
                INSERT INTO activations
                    (doc_id, title, project, kind, hits, first_recall, last_recall, last_query)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    hits = hits + 1,
                    last_recall = excluded.last_recall,
                    last_query = excluded.last_query,
                    title = CASE WHEN activations.title = '' THEN excluded.title
                                 ELSE activations.title END
                """,
                (
                    doc_id,
                    str(info.get("title", ""))[:200],
                    str(info.get("project", "")),
                    str(info.get("kind", "")),
                    now,
                    now,
                    query[:200],
                ),
            )
            touched += 1
        self._db.commit()
        return touched

    def working_set(self, limit: int = MAX_WORKING_SET) -> list[Recalled]:
        """What is currently in mind, strongest first."""
        now = time.time()
        rows = self._db.execute(
            "SELECT * FROM activations ORDER BY last_recall DESC LIMIT 500"
        ).fetchall()
        out = [
            Recalled(
                doc_id=row["doc_id"],
                title=row["title"],
                project=row["project"],
                kind=row["kind"],
                hits=row["hits"],
                last_recall=row["last_recall"],
                activation=_activation(row["hits"], row["last_recall"], now),
            )
            for row in rows
        ]
        out = [r for r in out if r.activation >= ACTIVE_THRESHOLD]
        out.sort(key=lambda r: r.activation, reverse=True)
        return out[:limit]

    def forget(self, *, now: float | None = None) -> int:
        """Drop what has decayed out of the working set."""
        now = now if now is not None else time.time()
        stale = [
            row["doc_id"]
            for row in self._db.execute("SELECT doc_id, hits, last_recall FROM activations")
            if _activation(row["hits"], row["last_recall"], now) < ACTIVE_THRESHOLD
        ]
        for doc_id in stale:
            self._db.execute("DELETE FROM activations WHERE doc_id = ?", (doc_id,))
        self._db.commit()
        if stale:
            log.debug("memory_bank_forgot", count=len(stale))
        return len(stale)

    # -- the ledger ------------------------------------------------------

    def rebuild_ledger(self) -> int:
        """Index what the stored memory says was done, by project."""
        from mimir.knowledge.store import KnowledgeStore

        store = KnowledgeStore(settings=self.settings)
        self._db.execute("DELETE FROM work")
        count = 0
        for document in store.documents():
            meta = document.metadata
            project = str(meta.extra.get("project") or "")
            kind = (
                next((t for t in meta.tags if t in
                      ("project", "feedback", "reference", "user", "incident")), "")
                or str(meta.category or "").rsplit("/", 1)[-1]
            )
            happened = str(meta.original_date or meta.created_at or "")
            self._db.execute(
                "INSERT OR REPLACE INTO work (doc_id, project, kind, title, happened, "
                "source, stale) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.doc_id,
                    project,
                    kind,
                    meta.title[:200],
                    happened,
                    str(meta.source or ""),
                    int(meta.last_verified is None),
                ),
            )
            count += 1
        self._db.commit()
        log.info("memory_ledger_rebuilt", documents=count)
        return count

    def projects(self) -> list[dict[str, object]]:
        """What has been worked on, most recent first."""
        rows = self._db.execute(
            """
            SELECT project,
                   COUNT(*)          AS notes,
                   MAX(happened)     AS latest,
                   SUM(stale)        AS unverified
            FROM work
            WHERE project <> ''
            GROUP BY project
            ORDER BY latest DESC, notes DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def about(self, project: str, limit: int = 20) -> list[dict[str, object]]:
        rows = self._db.execute(
            "SELECT * FROM work WHERE project LIKE ? ORDER BY happened DESC LIMIT ?",
            (f"%{project}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, object]:
        work = self._db.execute("SELECT COUNT(*) FROM work").fetchone()[0]
        projects = self._db.execute(
            "SELECT COUNT(DISTINCT project) FROM work WHERE project <> ''"
        ).fetchone()[0]
        tracked = self._db.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
        return {
            "documents": work,
            "projects": projects,
            "tracked": tracked,
            "working_set": len(self.working_set()),
            "half_life_hours": HALF_LIFE_S / 3600,
        }


_BANK: MemoryBank | None = None


def get_memory_bank(settings: Settings | None = None) -> MemoryBank:
    global _BANK
    if _BANK is None:
        _BANK = MemoryBank(settings=settings)
    return _BANK


__all__ = [
    "ACTIVE_THRESHOLD",
    "HALF_LIFE_S",
    "MemoryBank",
    "Recalled",
    "get_memory_bank",
]
