"""What the organisation accumulates from every completed task."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml

from mimir.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routing (
    session_id TEXT PRIMARY KEY, task_type TEXT, interface TEXT, specialists TEXT,
    tools_cited TEXT, tool_calls INTEGER, evidence INTEGER, decisions INTEGER,
    decisions_acted INTEGER, rounds INTEGER, confidence REAL, duration_s REAL, created_at REAL
);
CREATE TABLE IF NOT EXISTS decision_outcomes (
    session_id TEXT, field TEXT, choice TEXT, probability REAL, margin REAL,
    calibrated INTEGER, acted INTEGER, backend TEXT, node TEXT, confidence REAL, created_at REAL
);
CREATE INDEX IF NOT EXISTS decision_outcomes_field ON decision_outcomes(field);
"""


class ExperienceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def record(self, session: Any) -> dict[str, Any]:
        """Everything a completed session teaches, in one call that cannot raise."""
        try:
            summary = self._record(session)
            self._db.commit()
            return summary
        except Exception as exc:  # noqa: BLE001 - experience must not fail the session
            log.warning("experience_record_failed", error=str(exc))
            return {}

    def _record(self, session: Any) -> dict[str, Any]:
        cited_tools = sorted({
            (e.source_id.split(":")[0] if ":" in e.source_id else str(e.source_type))
            for e in session.evidence
        })
        decisions = session.metadata.get("decisions") or []
        rounds = len(session.metadata.get("assess") or []) or 1
        self._db.execute(
            "INSERT OR REPLACE INTO routing VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session.session_id, str(session.task_type or ""), session.interface,
             json.dumps(sorted({r.specialist.value for r in session.reports})),
             json.dumps(cited_tools), sum(r.tool_calls for r in session.reports),
             len(session.evidence), len(decisions), sum(1 for d in decisions if d.get("acted")),
             rounds, session.final_confidence, session.duration_s, time.time()),
        )
        for d in decisions:
            self._db.execute(
                "INSERT INTO decision_outcomes VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (session.session_id, d.get("field"), d.get("choice"), d.get("probability"),
                 d.get("margin"), int(bool(d.get("calibrated"))), int(bool(d.get("acted"))),
                 d.get("backend"), d.get("node"), session.final_confidence, time.time()),
            )
        return {"specialists": len(session.reports), "decisions": len(decisions), "rounds": rounds}

    def routing_for(self, task_type: str, *, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute(
            "SELECT * FROM routing WHERE task_type=? ORDER BY created_at DESC LIMIT ?", (task_type, limit))]

    def preferred_specialists(self, task_type: str, *, min_confidence: float = 0.6) -> list[str]:
        """Specialists that appeared in confident answers for this shape, most often first."""
        counts: dict[str, int] = {}
        for row in self.routing_for(task_type, limit=200):
            if row["confidence"] >= min_confidence:
                for s in json.loads(row["specialists"] or "[]"):
                    counts[s] = counts.get(s, 0) + 1
        return [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    def close(self) -> None:
        self._db.close()


_NOUN = re.compile(r"\b[a-z][a-z0-9]+(?:-[a-z0-9]+)+\b|\b[A-Z][A-Za-z]{3,}\b|\b\d+(?:\.\d+)?\s*(?:Gi|Mi|ms|s|%)\b")


def corpus_draft(session: Any, *, min_confidence: float = 0.7) -> dict[str, Any] | None:
    """A candidate corpus case from a confident, evidenced session, or None."""
    answer = session.final_answer
    if answer is None or session.final_confidence < min_confidence or not session.evidence:
        return None
    nouns = list(dict.fromkeys(_NOUN.findall(answer.answer or "")))[:4]
    if not nouns:
        return None
    return {
        "id": f"draft-{session.session_id[-8:]}",
        "kind": str(session.task_type or "general_question").replace("_investigation", "_evidence"),
        "prompt": session.user_request,
        "description": f"Drafted from session {session.session_id}; confidence {session.final_confidence}. Verify before accepting.",
        "expect_contains": nouns,
        "tags": ["draft", "from-use"],
    }


def write_corpus_draft(home: Path, session: Any) -> Path | None:
    draft = corpus_draft(session)
    if draft is None:
        return None
    folder = Path(home) / "corpus-drafts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{draft['id']}.yaml"
    path.write_text(yaml.safe_dump({"cases": [draft]}, sort_keys=False), encoding="utf-8")
    return path


def repo_lesson(repo: str, *, files_changed: list[str], test_command: str, tools_used: list[str],
                stopped: str, notes: str = "") -> dict[str, Any]:
    """A memory note body for what a coding task learned about a repository."""
    lines = [
        f"# {repo}: lessons from a coding task",
        "",
        f"- Outcome: {stopped}",
        f"- Files touched: {', '.join(files_changed) or 'none'}",
        f"- Test command that ran: `{test_command}`" if test_command else "- Test command: not run",
        f"- Tools that found things: {', '.join(tools_used) or 'none'}",
    ]
    if notes:
        lines += ["", notes.strip()]
    return {
        "title": f"{repo}: coding task lesson",
        "body": "\n".join(lines),
        "category": f"repos/{repo}",
        "confidence": 0.5,
        "sources": files_changed,
        "verification_status": "unverified",
    }


def get_experience_store(settings: Any) -> ExperienceStore:
    return ExperienceStore(Path(settings.home) / "experience.db")


__all__ = ["ExperienceStore", "corpus_draft", "get_experience_store", "repo_lesson", "write_corpus_draft"]
