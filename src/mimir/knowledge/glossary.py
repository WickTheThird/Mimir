"""What the operator's words turn out to mean."""

from __future__ import annotations

import difflib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS associations (
    term       TEXT NOT NULL,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    scope      TEXT NOT NULL DEFAULT '',
    hits       INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL,
    source     TEXT NOT NULL DEFAULT 'observed',
    PRIMARY KEY (term, name, scope)
);
CREATE INDEX IF NOT EXISTS associations_term ON associations(term);
"""

_WORD = re.compile(r"[a-z][a-z0-9]{2,}", re.IGNORECASE)

# Words that appear in every operational sentence and identify nothing.
_STOP = frozenset({
    "logs", "log", "pods", "pod", "namespace", "cluster", "clusters", "dev",
    "prod", "production", "staging", "last", "show", "give", "check", "find",
    "list", "from", "that", "this", "with", "have", "has", "any", "the",
    "into", "inside", "about", "curious", "would", "like", "want", "need",
    "please", "there", "their", "which", "what", "when", "where", "just",
    "also", "some", "them", "then", "than", "over", "under", "your", "you",
    "are", "was", "were", "been", "being", "does", "did", "doing", "kubectl",
    "deployment", "deployments", "service", "services", "workload",
    "workloads", "container", "containers", "restart", "restarts", "error",
    "errors", "file", "files", "code", "repo", "repository", "test", "tests",
    "for", "and", "not", "all", "get", "see", "its", "his", "her", "our",
    "why", "how", "who", "can", "should", "could", "will", "one", "two",
    "ten", "out", "off", "run", "new", "old", "now", "yet", "but",
})

MAX_NAMES = 3
"""Above this, a term does not resolve anything."""

MIN_RATIO = 0.8
MAX_LENGTH_GAP = 2
"""How close a term has to be to a name to be treated as meaning it."""


@dataclass(frozen=True)
class Association:
    term: str
    name: str
    kind: str
    scope: str = ""
    hits: int = 1
    source: str = "observed"

    def render(self) -> str:
        where = f" in {self.scope}" if self.scope else ""
        return f"{self.term!r} has meant {self.name} ({self.kind}{where})"


def terms_of(text: str) -> list[str]:
    """Candidate terms in what the operator wrote."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _WORD.finditer(text or ""):
        word = match.group(0).lower()
        if word in _STOP or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


class Glossary:
    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path) if path else self.settings.home / "glossary.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- writing ---------------------------------------------------------

    def record(
        self,
        term: str,
        name: str,
        kind: str,
        *,
        scope: str = "",
        source: str = "observed",
    ) -> None:
        term, name = term.strip().lower(), name.strip()
        if not term or not name or term == name.lower():
            return
        now = time.time()
        self._db.execute(
            """
            INSERT INTO associations (term, name, kind, scope, first_seen, last_seen, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term, name, scope) DO UPDATE SET
                hits = hits + 1, last_seen = excluded.last_seen
            """,
            (term, name, kind, scope, now, now, source),
        )
        self._db.commit()

    def learn(self, instruction: str, observed_names: list[str], *, scope: str = "") -> int:
        """Associate the operator's words with names seen in the same turn."""
        # A composite is not a name.
        names = sorted({
            n.strip() for n in observed_names
            if n and len(n) > 3 and "/" not in n and "." not in n
        })
        if not names:
            return 0

        # Nothing is learned from a name the operator already typed.
        said = (instruction or "").lower()
        learned = 0
        for term in terms_of(instruction):
            for name in _close_to(term, names):
                if name.lower() in said:
                    continue
                self.record(term, name, "observed name", scope=scope)
                learned += 1
        return learned

    # -- reading ---------------------------------------------------------

    def names(self) -> dict[str, str]:
        """Every distinct name known, with its kind."""
        rows = self._db.execute("SELECT DISTINCT name, kind FROM associations").fetchall()
        return dict(rows)

    def lookup(self, text: str, *, limit: int = 6) -> list[Association]:
        """Exact resolutions first, then near misses against known names."""
        terms = terms_of(text)
        if not terms:
            return []

        out: list[Association] = []
        known: dict[str, str] | None = None
        for term in terms:
            rows = self._db.execute(
                "SELECT term, name, kind, scope, hits, source FROM associations "
                "WHERE term = ? ORDER BY hits DESC, last_seen DESC",
                (term,),
            ).fetchall()
            if rows:
                if len(rows) <= MAX_NAMES:
                    out.extend(Association(*row) for row in rows)
                continue

            if known is None:
                known = self.names()
            for name in _close_to(term, list(known)):
                out.append(
                    Association(term, name, known.get(name, "name"), source="near match")
                )
            if len(out) >= limit:
                break
        return out[:limit]

    def all(self, limit: int = 200) -> list[Association]:
        rows = self._db.execute(
            "SELECT term, name, kind, scope, hits, source FROM associations "
            "ORDER BY hits DESC, term LIMIT ?",
            (limit,),
        ).fetchall()
        return [Association(*row) for row in rows]

    def hint(self, text: str, *, limit: int = 2) -> str:
        """One short line, or nothing."""
        said = (text or "").lower()
        found = [
            a for a in self.lookup(text, limit=limit * 3)
            if a.name.lower() not in said
        ][:limit]
        if not found:
            return ""
        pairs = "; ".join(f"{a.term} = {a.name}" for a in found)
        return f"(earlier in this estate: {pairs}. A lead, not a fact.)"

    def prune(self) -> int:
        """Remove associations that should never have been recorded."""
        cursor = self._db.execute(
            "DELETE FROM associations WHERE name LIKE '%/%' OR name LIKE '%.%'"
        )
        self._db.commit()
        return cursor.rowcount or 0

    # -- seeding ---------------------------------------------------------

    def seed(self) -> int:
        """Populate from what is already known, so turn one benefits."""
        added = 0
        added += self._seed_repositories()
        added += self._seed_contexts()
        added += self._seed_memory_projects()
        log.info("glossary_seeded", associations=added)
        return added

    def _seed_repositories(self) -> int:
        try:
            from mimir.tools.repo import get_repository_directory

            repos = get_repository_directory(self.settings).all()
        except Exception:  # noqa: BLE001 - seeding must never block startup
            return 0
        count = 0
        for repo in repos:
            for term in _segments(repo.name):
                self.record(term, repo.name, "repository", source="seeded")
                count += 1
        return count

    def _seed_contexts(self) -> int:
        import subprocess

        try:
            out = subprocess.run(
                ["kubectl", "config", "get-contexts", "-o", "name"],
                capture_output=True, text=True, timeout=15, check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return 0
        count = 0
        for name in (line.strip() for line in out.splitlines()):
            if not name:
                continue
            for term in _segments(name):
                self.record(term, name, "cluster context", source="seeded")
                count += 1
        return count

    def _seed_memory_projects(self) -> int:
        try:
            from mimir.knowledge.store import KnowledgeStore

            documents = KnowledgeStore(settings=self.settings).documents()
        except Exception:  # noqa: BLE001
            return 0
        count = 0
        for document in documents:
            project = str(document.metadata.extra.get("project") or "")
            if not project:
                continue
            for term in _segments(project):
                self.record(term, project, "project", source="seeded")
                count += 1
        return count


def _segments(name: str) -> list[str]:
    """The parts of a name an operator might say on their own."""
    # Three characters, not four.
    parts = [p.lower() for p in re.split(r"[-_./]", name) if len(p) >= 3]
    return [p for p in parts if p not in _STOP]


def _close_to(term: str, names: list[str]) -> list[str]:
    """Names the term plausibly meant."""
    exact = [n for n in names if term in _segments(n)]
    if not exact:
        exact = [
            n
            for n in names
            for seg in _segments(n)
            if seg.startswith(term) and len(seg) - len(term) <= MAX_LENGTH_GAP
        ]
    if exact:
        return exact[:MAX_NAMES] if len(exact) <= MAX_NAMES else []
    segments = {seg: n for n in names for seg in _segments(n)}
    candidates = [
        seg for seg in segments if abs(len(seg) - len(term)) <= MAX_LENGTH_GAP
    ]
    close = difflib.get_close_matches(term, candidates, n=2, cutoff=MIN_RATIO)
    return [segments[c] for c in close]


_GLOSSARY: Glossary | None = None


def get_glossary(settings: Settings | None = None) -> Glossary:
    global _GLOSSARY
    if _GLOSSARY is None:
        _GLOSSARY = Glossary(settings=settings)
    return _GLOSSARY


__all__ = ["Association", "Glossary", "get_glossary", "terms_of"]
