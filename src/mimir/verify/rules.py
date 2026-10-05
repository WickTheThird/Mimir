"""Project rules as data, checked without a model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import yaml

from mimir.logging import get_logger

log = get_logger(__name__)

MAX_RULE_BYTES = 200_000


@dataclass(frozen=True)
class Violation:
    rule: str
    title: str
    path: str
    line: int
    excerpt: str
    why: str = ""
    severity: str = "error"

    @property
    def blocking(self) -> bool:
        return self.severity == "error"

    def render(self) -> str:
        return f"{self.path}:{self.line} {self.title} [{self.rule}]"


@dataclass(frozen=True)
class Rule:
    """One checkable project invariant."""

    id: str
    title: str
    why: str = ""
    paths: tuple[str, ...] = ()
    """Glob patterns this applies to. Empty means every file."""

    forbid: str = ""
    """A pattern that must not appear."""

    require: str = ""
    """A pattern that must appear, but only in files that match ``when``."""

    when: str = ""
    """The trigger for ``require``. Without it, require applies to every file."""

    severity: str = "error"
    source: str = ""

    def applies_to(self, path: str) -> bool:
        if not self.paths:
            return True
        candidate = PurePath(path)
        return any(
            candidate.match(pattern) or PurePath(f"x/{path}").match(pattern)
            for pattern in self.paths
        )

    def check(self, path: str, text: str) -> list[Violation]:
        if not self.applies_to(path):
            return []
        out: list[Violation] = []
        if self.forbid:
            for match in re.finditer(self.forbid, text, re.MULTILINE):
                out.append(
                    Violation(
                        rule=self.id,
                        title=self.title,
                        path=path,
                        line=text[: match.start()].count("\n") + 1,
                        excerpt=match.group(0)[:120],
                        why=self.why,
                        severity=self.severity,
                    )
                )
        if self.require:
            triggered = re.search(self.when, text, re.MULTILINE) if self.when else True
            if triggered and not re.search(self.require, text, re.MULTILINE):
                out.append(
                    Violation(
                        rule=self.id,
                        title=self.title,
                        path=path,
                        line=1,
                        excerpt=f"required pattern absent: {self.require}",
                        why=self.why,
                        severity=self.severity,
                    )
                )
        return out


def _rule_from(data: dict[str, Any], source: str) -> Rule | None:
    identifier = str(data.get("id") or "").strip()
    title = str(data.get("title") or "").strip()
    if not identifier or not title:
        return None
    if not (data.get("forbid") or data.get("require")):
        return None
    for key in ("forbid", "require", "when"):
        pattern = data.get(key)
        if pattern:
            try:
                re.compile(str(pattern))
            except re.error as exc:
                # A rule that cannot compile is skipped loudly rather than
                log.warning("rule_pattern_invalid", rule=identifier, key=key, error=str(exc))
                return None
    paths = data.get("paths") or []
    return Rule(
        id=identifier,
        title=title,
        why=str(data.get("why") or "").strip(),
        paths=tuple(str(p) for p in paths),
        forbid=str(data.get("forbid") or ""),
        require=str(data.get("require") or ""),
        when=str(data.get("when") or ""),
        severity="warning" if str(data.get("severity", "error")) != "error" else "error",
        source=source,
    )


def load_rules(roots: list[Path]) -> list[Rule]:
    """Every rule from every ``*.yaml`` under the given directories."""
    out: list[Rule] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.y*ml")):
            try:
                if path.stat().st_size > MAX_RULE_BYTES:
                    continue
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            except (OSError, yaml.YAMLError) as exc:
                log.warning("rule_file_unreadable", path=str(path), error=str(exc))
                continue
            entries = data if isinstance(data, list) else data.get("rules") or []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                rule = _rule_from(entry, source=path.name)
                if rule is not None:
                    out.append(rule)
    return out


def rule_roots(settings: Any, repo_root: Path | None = None) -> list[Path]:
    """Where rules live: the operator's home, then the repository itself."""
    roots = [Path(settings.home) / "rules"]
    if repo_root is not None:
        roots.append(Path(repo_root) / ".mimir" / "rules")
    return roots


def check_rules(rules: list[Rule], path: str, text: str) -> list[Violation]:
    out: list[Violation] = []
    for rule in rules:
        out.extend(rule.check(path, text))
    return out


__all__ = ["Rule", "Violation", "check_rules", "load_rules", "rule_roots"]
