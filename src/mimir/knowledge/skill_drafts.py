"""Turn a completed task into a skill draft. Borrowed from Hermes Agent.

Hermes Agent's loop: after a task, write what worked as a reusable skill in
the agentskills.io shape, keep it, improve it on reuse. MIMIR already has
the skill format (SKILL.md with frontmatter), the registry, the runner and
the tests. What it lacked was the step that writes one from experience.

Two rules that are MIMIR's, not Hermes's:

* **Drafts live outside the live roots.** A draft is written to
  ``$MIMIR_HOME/knowledge/skills-drafts/<name>/SKILL.md`` and the registry
  never sees it until a person runs ``mimir skills promote``. The project
  does not auto-promote anything (ADR 11.6, NG4), and a skill the planner
  can select is an instruction the model will follow.
* **A draft carries a test.** The session it came from supplies the input
  and the discriminating nouns of the answer, so the skill runner can check
  the skill still does what it was drafted from.

Only confident, evidenced sessions that actually used tools qualify; a
chat that needed no investigation teaches nothing worth a file.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from mimir.logging import get_logger

log = get_logger(__name__)

DRAFTS_DIR = "skills-drafts"
_SLUG = re.compile(r"[^a-z0-9]+")
_NOUN = re.compile(r"\b[a-z][a-z0-9]+(?:-[a-z0-9]+)+\b|\b[A-Z][A-Za-z]{3,}\b")


def _slug(text: str, limit: int = 48) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:limit].rstrip("-") or "task"


def _known_tools() -> set[str]:
    """Names the registry knows. A draft may only name helpers that exist;
    the loader enforces it at read time and this enforces it at write time,
    so a draft is never born invalid."""
    try:
        from mimir.tools.base import load_all_tools

        return set(load_all_tools().names())
    except Exception:  # noqa: BLE001 - a draft is best effort
        return set()


def _tools_in_order(session: Any) -> list[str]:
    seen: list[str] = []
    for e in session.evidence:
        tool = e.source_id.split(":")[0] if ":" in e.source_id else ""
        if tool and tool not in seen and not tool.startswith(("imports/", "skills/", "stable/")):
            seen.append(tool)
    for record in getattr(session, "commands_executed", []) or []:
        head = record.argv[0] if record.argv else ""
        if head and head not in seen:
            seen.append(head)
    return seen


def draft_from_session(session: Any, *, min_confidence: float = 0.7) -> dict[str, Any] | None:
    """A SKILL.md frontmatter and body from an ops session, or None."""
    answer = session.final_answer
    if answer is None or session.final_confidence < min_confidence or not session.evidence:
        return None
    tools = _tools_in_order(session)
    known = _known_tools()
    tools = [t for t in tools if t in known] if known else tools
    if not tools:
        return None
    specialists = [r.specialist.value for r in session.reports if not r.failed]
    specialist = specialists[0] if specialists else "log_analyst"
    nouns = list(dict.fromkeys(_NOUN.findall(answer.answer or "")))[:3]
    task = str(session.task_type or "investigation").replace("_", " ")
    name = f"learned-{_slug(task)}-{_slug(session.user_request)[:24]}"
    steps = []
    for r in session.reports:
        if r.failed:
            continue
        head = (r.conclusion or "").strip().splitlines()
        steps.append(f"- {r.specialist.value}: {head[0][:200] if head else 'gathered evidence'}")
    frontmatter = {
        "name": name,
        "version": "0.1.0",
        "description": f"Learned from a confident {task} session: {session.user_request[:100]}",
        "when_to_use": f"A question like: {session.user_request[:160]}",
        "specialist": specialist,
        "max_risk": "R1",
        "author": "mimir (draft from session)",
        "updated_at": time.strftime("%Y-%m-%d"),
        "tags": ["draft", "learned", *[t for t in tools[:3]]],
        "allowed_tools": tools[:8],
        "inputs": [{"name": "question", "description": "The operator's question.", "required": True}],
        "tests": [{
            "name": "reproduces-the-session",
            "input": session.user_request[:300],
            "assertions": [f"contains: {n}" for n in nouns] or ["contains: evidence"],
        }],
        "outputs": [{"name": "answer", "description": "An evidence-backed answer with citations."}],
    }
    body = "\n".join([
        f"# {task.title()} (learned)",
        "",
        f"Drafted from session {session.session_id} at confidence {session.final_confidence}.",
        "A person has not reviewed this yet. Promote it only if the steps generalise.",
        "",
        "## What worked",
        f"1. Tools, in the order they produced cited evidence: {', '.join(tools)}.",
        *(f"{i + 2}. {s[2:]}" for i, s in enumerate(steps[:5])),
        "",
        "## What settled it",
        (answer.answer or "")[:600],
        "",
        "## Do not",
        "- Name anything that no tool listed.",
        "- State a count from a note nobody verified.",
        "- Answer absence from a search that did not run.",
    ])
    return {"name": name, "frontmatter": frontmatter, "body": body}


def draft_from_coding(repo: str, instruction: str, *, files_changed: list[str], tools_used: list[str],
                      test_command: str, diff_summary: str = "") -> dict[str, Any] | None:
    if not files_changed:
        return None
    name = f"learned-code-{_slug(repo)}-{_slug(instruction)[:24]}"
    frontmatter = {
        "name": name, "version": "0.1.0",
        "description": f"Learned from a coding task in {repo}: {instruction[:100]}",
        "when_to_use": f"A change like: {instruction[:160]} in {repo}",
        "specialist": "repository_explorer", "max_risk": "R1",
        "author": "mimir (draft from task)", "updated_at": time.strftime("%Y-%m-%d"),
        "tags": ["draft", "learned", "code", repo],
        "allowed_tools": ([t for t in tools_used if t and t in (_known_tools() or {t})][:8]
                          or ["repository_map", "search_repository", "read_file_range"]),
        "inputs": [{"name": "instruction", "description": "The change asked for.", "required": True}],
        "tests": [{"name": "touches-the-right-files", "input": instruction[:300],
                   "assertions": [f"contains: {Path(f).name}" for f in files_changed[:3]]}],
        "outputs": [{"name": "diff", "description": "A reviewed diff in a task worktree."}],
    }
    body = "\n".join([
        f"# Changing {repo} (learned)", "",
        f"Files this kind of change touched: {', '.join(files_changed)}.",
        f"Test command that ran: `{test_command}`." if test_command else "Tests: run the repository's suite.",
        "", "## Steps",
        "1. Ask repository_map for the symbols and the tests that cover them before searching.",
        "2. Make the change in the task worktree only.",
        "3. Run the affected tests, then the full suite, before reporting.",
        "", "## Do not",
        "- Delete anything the instruction did not name.",
        "- Edit a test to make it pass.",
        *( ["", "## From the diff", diff_summary[:800]] if diff_summary else []),
    ])
    return {"name": name, "frontmatter": frontmatter, "body": body}


def write_draft(home: Path, draft: dict[str, Any]) -> Path:
    folder = Path(home) / "knowledge" / DRAFTS_DIR / draft["name"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text("---\n" + yaml.safe_dump(draft["frontmatter"], sort_keys=False, allow_unicode=True)
                    + "---\n\n" + draft["body"] + "\n", encoding="utf-8")
    return path


def list_drafts(home: Path) -> list[Path]:
    folder = Path(home) / "knowledge" / DRAFTS_DIR
    return sorted(folder.glob("*/SKILL.md")) if folder.is_dir() else []


def promote(home: Path, name: str, live_root: Path) -> Path:
    """Move a draft into a live skills root. The one step that needs a person."""
    src = Path(home) / "knowledge" / DRAFTS_DIR / name
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(f"no draft named {name}")
    dst = Path(live_root) / name
    if dst.exists():
        raise FileExistsError(f"a skill named {name} already exists at {dst}")
    shutil.move(str(src), str(dst))
    return dst / "SKILL.md"


__all__ = ["DRAFTS_DIR", "draft_from_coding", "draft_from_session", "list_drafts", "promote", "write_draft"]
