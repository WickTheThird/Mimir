"""Did the answer name anything that was never observed?"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# A DNS label with at least two segments.
_DNS_NAME = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){1,}\b")

# Dotted or slashed paths: file paths, image references, resource references.
_PATH = re.compile(r"\b[\w.-]+(?:/[\w.-]+){1,}\b")

_IGNORE = frozenset({"n/a", "e.g", "i.e"})
"""Exceptions the segment rule below does not catch."""

# Hyphenated English is not built from arbitrary words: one half is almost
_MODIFIERS = frozenset({
    "auto", "back", "best", "built", "case", "closed", "compile", "cross",
    "day", "de", "double", "dry", "end", "fail", "far", "first", "front",
    "full", "good", "half", "hard", "high", "human", "in", "inter", "intra",
    "last", "left", "line", "live", "local", "long", "low", "machine", "mid",
    "multi", "near", "next", "non", "off", "on", "one", "only", "open", "opt",
    "out", "over", "per", "post", "pre", "re", "read", "real", "remote",
    "right", "round", "run", "second", "self", "semi", "short", "side",
    "single", "so", "soft", "step", "sub", "super", "third", "top", "trade",
    "to", "two", "under", "up", "well", "worst", "write",
})


def _is_prose(token: str) -> bool:
    """Whether a hyphenated token is English rather than a name."""
    segments = token.split("-")
    if not 2 <= len(segments) <= 3:
        return False
    return bool(_MODIFIERS & {segments[0], segments[-1]})



@dataclass
class Grounding:
    checked: int = 0
    ungrounded: list[str] = field(default_factory=list)
    supported: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded

    @property
    def rate(self) -> float:
        return round(len(self.ungrounded) / self.checked, 3) if self.checked else 0.0

    def brief(self) -> str:
        if self.ok:
            return ""
        shown = ", ".join(self.ungrounded[:6])
        more = f" and {len(self.ungrounded) - 6} more" if len(self.ungrounded) > 6 else ""
        return (
            f"{len(self.ungrounded)} name(s) in this answer appear in nothing that was "
            f"read: {shown}{more}"
        )


def identifiers(text: str) -> list[str]:
    """Identifier-shaped tokens, in order, deduplicated."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_DNS_NAME, _PATH):
        for match in pattern.finditer(text or ""):
            token = match.group(0).strip(".,;:)").lower()
            if token in seen or len(token) < 8:
                continue
            # Version strings and log timestamps are not names.
            digits = sum(c.isdigit() for c in token)
            if digits * 2 >= len(token):
                continue
            if token in _IGNORE or ("/" not in token and _is_prose(token)):
                continue
            seen.add(token)
            found.append(token)
    return found


def check(answer: str, observed: str, *, asked: str = "") -> Grounding:
    """Which identifiers in ``answer`` appear in what was read or asked."""
    haystack = f"{observed}\n{asked}".lower()
    result = Grounding()
    for token in identifiers(answer):
        result.checked += 1
        if token in haystack:
            result.supported.append(token)
        else:
            result.ungrounded.append(token)
    return result


def demote_ungrounded(answer: Any, grounding: Grounding) -> tuple[Any, int]:
    """Move facts naming things nobody observed into the unverified set."""
    if grounding.ok:
        return answer, 0
    invented = set(grounding.ungrounded)

    def names_invented(text: str) -> bool:
        lowered = (text or "").lower()
        return any(token in lowered for token in invented)

    facts = list(getattr(answer, "observed_facts", None) or [])
    kept = [f for f in facts if not names_invented(f)]
    moved = [f for f in facts if names_invented(f)]
    answer.observed_facts = kept
    answer.unverified = [
        *(getattr(answer, "unverified", None) or []),
        *[f"{f} (names nothing that was read)" for f in moved],
    ]
    if names_invented(getattr(answer, "answer", "")):
        answer.answer = f"{answer.answer}\n\nWarning: {grounding.brief()}."
    answer.confidence = round(min(getattr(answer, "confidence", 0.5), 0.3), 3)
    return answer, len(moved) or len(invented)


__all__ = ["Grounding", "check", "demote_ungrounded", "identifiers"]
