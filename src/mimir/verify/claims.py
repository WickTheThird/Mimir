"""Deterministic claim support checking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mimir.models.evidence import Evidence, SourceType

# Source types that can never support an assertion about the observed world.
NON_OBSERVATIONAL: frozenset[SourceType] = frozenset({SourceType.MODEL_KNOWLEDGE})

_IDENTIFIER = re.compile(
    r"""(?:
        [\w./-]+\.(?:py|ts|tsx|js|go|rs|java|rb|sql|ya?ml|toml|json|md)(?::\d+(?:-\d+)?)?
      | [A-Za-z_][\w]*\.[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*
      | [A-Za-z_][\w]{3,}_[\w]+
      | \b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b
      | https?://\S+
    )""",
    re.VERBOSE,
)

_COMMAND_HINT = re.compile(
    r"\b(kubectl|psql|docker|systemctl|journalctl|curl|git|npm|pytest|helm)\b"
)

_STOPWORDS: frozenset[str] = frozenset(
    (
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "at",
    "by", "for", "with", "from", "as", "it", "its", "it's", "their", "there", "here", "we",
    "you", "i", "not", "no", "yes", "do", "does", "did", "done", "can", "could", "should",
    "would", "may", "might", "must", "will", "shall", "when", "where", "which", "who",
    "whom", "whose", "what", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "too", "very", "just",
    "also", "into", "over", "under", "again", "further", "once", "during", "while", "about",
    "against", "between", "through", "before", "after", "above", "below", "up", "down",
    "out", "off"
    )
)


class ClaimKind(StrEnum):
    OBSERVED = "observed"
    """An assertion about the world. Must resolve to observational evidence."""

    INFERENCE = "inference"
    """Derived from observations. Should reference support but may reason past it."""

    UNVERIFIED = "unverified"
    """Already labelled by the answer as unconfirmed. Not scored as a failure."""


@dataclass
class ClaimSupport:
    claim: str
    kind: ClaimKind
    evidence_ids: list[str] = field(default_factory=list)
    supported: bool = False
    reason: str = ""
    subjects: list[str] = field(default_factory=list)
    """Identifier-like tokens the claim is about."""

    @property
    def factual(self) -> bool:
        return self.kind is ClaimKind.OBSERVED


@dataclass
class AnswerSupport:
    claims: list[ClaimSupport] = field(default_factory=list)
    dangling_citations: list[str] = field(default_factory=list)
    """Citations on the answer that resolve to no evidence item at all."""

    resolved_citations: list[str] = field(default_factory=list)

    @property
    def factual_claims(self) -> list[ClaimSupport]:
        return [c for c in self.claims if c.factual]

    @property
    def unsupported(self) -> list[ClaimSupport]:
        return [c for c in self.factual_claims if not c.supported]

    @property
    def total(self) -> int:
        return len(self.factual_claims)

    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported)

    @property
    def clean(self) -> bool:
        return not self.unsupported and not self.dangling_citations

    def summary(self) -> str:
        if not self.total:
            return "no factual claims"
        return (
            f"{self.total - self.unsupported_count}/{self.total} factual claims "
            f"supported, {len(self.dangling_citations)} dangling citation(s)"
        )


def subjects_of(claim: str) -> list[str]:
    """Identifier-like things the claim is about."""
    found = [m.group(0) for m in _IDENTIFIER.finditer(claim)]
    found.extend(m.group(0) for m in _COMMAND_HINT.finditer(claim))
    seen: list[str] = []
    for item in found:
        normalised = item.rstrip(".,;:)")
        if normalised and normalised not in seen:
            seen.append(normalised)
    return seen


def significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z_][\w-]{2,}", text.lower())
        if token not in _STOPWORDS
    }


def _evidence_haystack(item: Evidence) -> str:
    parts = [item.claim, item.excerpt, item.source_id]
    parts.extend(c.render() for c in item.citations)
    parts.extend(c.path or "" for c in item.citations)
    return " ".join(p for p in parts if p).lower()


def _subject_matches(subject: str, haystack: str) -> bool:
    """A subject matches on its own terms, or on its last path segment."""
    lowered = subject.lower()
    if lowered in haystack:
        return True
    tail = lowered.rsplit("/", 1)[-1].split(":", 1)[0]
    return len(tail) > 4 and tail in haystack


def check_claim(
    claim: str, kind: ClaimKind, evidence: list[Evidence], *, min_overlap: int = 2
) -> ClaimSupport:
    """Resolve one claim against the evidence gathered this session."""
    support = ClaimSupport(claim=claim, kind=kind, subjects=subjects_of(claim))

    usable = [
        item
        for item in evidence
        if item.supports and item.source_type not in NON_OBSERVATIONAL
    ]
    if not usable:
        support.reason = (
            "no observational evidence was gathered"
            if not evidence
            else "all evidence is contradicting or non-observational"
        )
        return support

    if support.subjects:
        for item in usable:
            haystack = _evidence_haystack(item)
            hits = [s for s in support.subjects if _subject_matches(s, haystack)]
            if hits:
                support.evidence_ids.append(item.id)
                support.supported = True
        support.reason = (
            f"subject(s) {', '.join(support.subjects[:3])} found in "
            f"{len(support.evidence_ids)} evidence item(s)"
            if support.supported
            else f"claim names {', '.join(support.subjects[:3])}, absent from all evidence"
        )
        return support

    tokens = significant_tokens(claim)
    best = 0
    for item in usable:
        overlap = len(tokens & significant_tokens(_evidence_haystack(item)))
        best = max(best, overlap)
        if overlap >= min_overlap:
            support.evidence_ids.append(item.id)
            support.supported = True
    support.reason = (
        f"{len(support.evidence_ids)} evidence item(s) share >= {min_overlap} terms"
        if support.supported
        else f"names nothing concrete and best evidence overlap was {best} term(s)"
    )
    return support


def check_answer(answer: Any, evidence: list[Evidence]) -> AnswerSupport:
    """Check every claim in a final answer, and every citation it carries."""
    out = AnswerSupport()
    if answer is None:
        return out

    for text in getattr(answer, "observed_facts", []) or []:
        out.claims.append(check_claim(text, ClaimKind.OBSERVED, evidence))
    for text in getattr(answer, "inferences", []) or []:
        out.claims.append(check_claim(text, ClaimKind.INFERENCE, evidence))
    for text in getattr(answer, "unverified", []) or []:
        out.claims.append(
            ClaimSupport(
                claim=text,
                kind=ClaimKind.UNVERIFIED,
                supported=True,
                reason="declared unverified by the answer",
            )
        )

    known = {item.id for item in evidence}
    locators = {
        c.render().lower() for item in evidence for c in item.citations
    } | {item.source_id.lower() for item in evidence if item.source_id}
    for citation in getattr(answer, "citations", []) or []:
        text = str(citation).strip()
        if not text:
            continue
        lowered = text.lower()
        if (
            text in known
            or lowered in locators
            or any(lowered in loc or loc in lowered for loc in locators if loc)
        ):
            out.resolved_citations.append(text)
        else:
            out.dangling_citations.append(text)
    return out


def demote_unsupported(answer: Any, support: AnswerSupport) -> tuple[Any, int, int]:
    """Move unsupported factual claims out of the observed set, mechanically."""
    unsupported = {c.claim for c in support.unsupported}
    if not unsupported and not support.dangling_citations:
        return answer, 0, 0

    kept = [c for c in (answer.observed_facts or []) if c not in unsupported]
    moved = [c for c in (answer.observed_facts or []) if c in unsupported]
    answer.observed_facts = kept
    answer.unverified = [
        *(answer.unverified or []),
        *[f"{claim} (no supporting evidence found)" for claim in moved],
    ]

    dropped = len(support.dangling_citations)
    if support.dangling_citations:
        answer.citations = list(support.resolved_citations)

    total = support.total or 1
    if moved:
        ratio = (total - len(moved)) / total
        answer.confidence = round(min(answer.confidence, answer.confidence * ratio), 3)
    return answer, len(moved), dropped


def unsupported_brief(support: AnswerSupport, limit: int = 6) -> str:
    """A repair instruction naming exactly what failed to resolve."""
    lines = [
        "These stated facts could not be resolved to any evidence gathered this "
        "session. For each one: either cite the specific evidence that supports "
        "it, restate it as an inference, or drop it. Do not invent a citation.",
    ]
    for claim in support.unsupported[:limit]:
        lines.append(f"  - {claim.claim[:200]}\n      ({claim.reason})")
    if support.dangling_citations:
        lines.append(
            "These citations do not correspond to any evidence item and will be "
            "removed: " + ", ".join(support.dangling_citations[:6])
        )
    return "\n".join(lines)


def attach_resolved_citations(
    answer: Any, support: AnswerSupport, evidence: list[Evidence], *, limit: int = 12
) -> int:
    """Cite the evidence that actually resolved each surviving claim."""
    by_id = {item.id: item for item in evidence}
    wanted: list[str] = []
    for claim in support.claims:
        if not claim.supported:
            continue
        for evidence_id in claim.evidence_ids:
            item = by_id.get(evidence_id)
            if item is None:
                continue
            rendered = [c.render() for c in item.citations] or (
                [item.source_id] if item.source_id else []
            )
            for text in rendered:
                if text and text not in wanted:
                    wanted.append(text)

    existing = list(answer.citations or [])
    added = [c for c in wanted if c not in existing][: max(0, limit - len(existing))]
    if added:
        answer.citations = [*existing, *added]
    return len(added)
