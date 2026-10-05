"""Skill discovery, indexing, and selection (ADR 10.2, 10.5)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.specialist import SpecialistName
from mimir.skills.loader import (
    SKILL_FILENAME,
    Skill,
    SkillLoadResult,
    estimate_tokens,
    load_skill_file,
)
from mimir.tools.base import ToolRegistry

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

# : Dropped before scoring.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
        "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
        "of", "in", "on", "at", "to", "for", "from", "with", "without", "by",
        "it", "its", "my", "our", "your", "we", "i", "you", "they", "them",
        "do", "does", "did", "doing", "done", "can", "could", "should", "would",
        "what", "why", "how", "when", "where", "who", "which",
        "get", "got", "has", "have", "had", "not", "no", "any", "some", "there",
        "help", "me", "please", "need", "want", "know", "tell", "show", "about",
        "up", "out", "into", "over", "again", "just", "now",
    }
)

# : Field weights.
_WEIGHT_NAME = 3.0
_WEIGHT_TAGS = 2.5
_WEIGHT_WHEN = 2.0
_WEIGHT_DESCRIPTION = 1.0

# Calibrated against the seed corpus: correct top-ranked skills score roughly
_MIN_SCORE = 0.12


# : Suffixes stripped when expanding a token.
_SUFFIXES = ("ing", "ed", "es", "s")


def _expand(terms: set[str]) -> set[str]:
    """A token plus its crude stem, so plural and tense variants still match."""
    out = set(terms)
    for term in terms:
        for suffix in _SUFFIXES:
            if len(term) > len(suffix) + 3 and term.endswith(suffix):
                out.add(term[: -len(suffix)])
                break
    return out


def _stem(word: str) -> str:
    """Very small suffix stripper: enough to tie 'restarting' to 'restart'."""
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenise(text: str) -> set[str]:
    words = {_stem(w) for w in _WORD_RE.findall(text.lower()) if len(w) > 1}
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


@dataclass(slots=True)
class SkillSelection:
    skill: Skill
    score: float
    matched: list[str] = field(default_factory=list)
    explicit: bool = False

    @property
    def reason(self) -> str:
        if self.explicit:
            return "requested by name"
        if not self.matched:
            return "no keyword overlap"
        return f"matched {', '.join(sorted(self.matched))}"


@dataclass(slots=True)
class ShadowedSkill:
    name: str
    winner: Path
    shadowed: Path


class SkillRegistry:
    """Name index over the discovered skills. Level 1 only."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        roots: list[Path] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._roots = roots if roots is not None else self.settings.skill_roots()
        self._tool_registry = tool_registry
        self._skills: dict[str, Skill] = {}
        self._failures: list[SkillLoadResult] = []
        self._shadowed: list[ShadowedSkill] = []
        self._loaded = False

    # -- discovery --------------------------------------------------------

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def _candidate_files(self) -> list[Path]:
        seen: set[Path] = set()
        found: list[Path] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            # Depth one is the Agent Skills layout; depth two allows grouping
            for pattern in (f"*/{SKILL_FILENAME}", f"*/*/{SKILL_FILENAME}"):
                for path in sorted(root.glob(pattern)):
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    found.append(path)
        return found

    def reload(self) -> SkillRegistry:
        self._skills.clear()
        self._failures.clear()
        self._shadowed.clear()
        for path in self._candidate_files():
            result = load_skill_file(path, tool_registry=self._tool_registry)
            if result.skill is None:
                self._failures.append(result)
                log.warning("skill_invalid", path=str(path), errors=result.errors)
                continue
            existing = self._skills.get(result.skill.name)
            if existing is not None:
                self._shadowed.append(
                    ShadowedSkill(
                        name=result.skill.name,
                        winner=existing.source_path,
                        shadowed=result.skill.source_path,
                    )
                )
                continue
            self._skills[result.skill.name] = result.skill
        self._loaded = True
        log.info(
            "skills_loaded",
            count=len(self._skills),
            failures=len(self._failures),
            shadowed=len(self._shadowed),
            roots=[str(r) for r in self._roots],
        )
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    # -- lookup -----------------------------------------------------------

    def get(self, name: str) -> Skill | None:
        self._ensure_loaded()
        return self._skills.get(name.strip())

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            known = ", ".join(self.names()) or "none discovered"
            raise KeyError(f"unknown skill: {name}. Known skills: {known}")
        return skill

    def names(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._skills)

    def all(self) -> list[Skill]:
        self._ensure_loaded()
        return [self._skills[name] for name in sorted(self._skills)]

    def for_specialist(self, specialist: SpecialistName) -> list[Skill]:
        return [s for s in self.all() if s.specialist == specialist]

    @property
    def failures(self) -> list[SkillLoadResult]:
        self._ensure_loaded()
        return list(self._failures)

    @property
    def shadowed(self) -> list[ShadowedSkill]:
        self._ensure_loaded()
        return list(self._shadowed)

    # -- level 1 ----------------------------------------------------------

    def briefs(self, specialist: SpecialistName | None = None) -> list[dict[str, str]]:
        return [
            s.brief() for s in self.all() if specialist is None or s.specialist == specialist
        ]

    def catalogue(
        self,
        specialist: SpecialistName | None = None,
        *,
        max_chars: int = 4000,
        line_chars: int = 200,
    ) -> str:
        """Level 1 for the whole library, one line per skill."""
        skills = [s for s in self.all() if specialist is None or s.specialist == specialist]
        lines: list[str] = []
        used = 0
        for index, skill in enumerate(skills):
            line = skill.brief_line(line_chars)
            if used + len(line) + 1 > max_chars:
                lines.append(f"[{len(skills) - index} more skills omitted; narrow by specialist]")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def catalogue_tokens(self, specialist: SpecialistName | None = None) -> int:
        return estimate_tokens(self.catalogue(specialist))

    # -- selection --------------------------------------------------------

    def score(self, skill: Skill, terms: set[str]) -> tuple[float, list[str]]:
        """Rank one skill against a query."""
        if not terms:
            return 0.0, []
        expanded = _expand(terms)
        fields = (
            (_WEIGHT_NAME, _expand(set(tokenise(skill.name.replace("-", " "))))),
            (_WEIGHT_TAGS, _expand(set(tokenise(" ".join(skill.tags))))),
            (_WEIGHT_WHEN, _expand(set(tokenise(skill.when_to_use)))),
            (_WEIGHT_DESCRIPTION, _expand(set(tokenise(skill.description)))),
        )
        # A term scores once, at the weight of the strongest field it appears
        best: dict[str, float] = {}
        for weight, bag in fields:
            for term in expanded & bag:
                if weight > best.get(term, 0.0):
                    best[term] = weight
        if not best:
            return 0.0, []
        total = sum(best.values())
        denominator = len(terms) * _WEIGHT_NAME
        return min(1.0, total / denominator), sorted(best)

    def select(
        self,
        request: str,
        *,
        specialist: SpecialistName | None = None,
        explicit: list[str] | None = None,
        limit: int | None = None,
        min_score: float = _MIN_SCORE,
    ) -> list[SkillSelection]:
        """Rank skills for a request. Explicit names always come first."""
        self._ensure_loaded()
        cap = limit if limit is not None else self.settings.skills.max_auto_selected
        chosen: list[SkillSelection] = []
        seen: set[str] = set()

        for name in explicit or []:
            skill = self.get(name)
            if skill is None:
                log.warning("skill_explicitly_selected_but_unknown", skill=name)
                continue
            chosen.append(SkillSelection(skill=skill, score=1.0, explicit=True))
            seen.add(skill.name)

        terms = tokenise(request)
        scored: list[SkillSelection] = []
        for skill in self.all():
            if skill.name in seen:
                continue
            if specialist is not None and skill.specialist != specialist:
                continue
            value, matched = self.score(skill, terms)
            if value >= min_score:
                scored.append(SkillSelection(skill=skill, score=value, matched=matched))

        # Ties broken by name so selection is deterministic across runs.
        scored.sort(key=lambda s: (-s.score, s.skill.name))
        remaining = max(0, cap - len(chosen))
        return chosen + scored[:remaining]


_registry: SkillRegistry | None = None


def get_skill_registry(settings: Settings | None = None) -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry(settings)
    return _registry


def reset_skill_registry() -> None:
    global _registry
    _registry = None
