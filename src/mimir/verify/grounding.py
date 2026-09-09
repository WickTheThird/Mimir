"""Did the answer name anything that was never observed?

The council's answers go through a claim gate: a stated fact must be supported
by evidence, and a citation must be evidence rather than ornament. The agent
loops had no equivalent, and the first real operations run showed why that
matters. Asked for a workload that does not exist, the model correctly reported
its absence and then listed the workloads that were present, and seventeen of
the names in that list were invented. They were plausible, consistent with the
namespace's naming convention, and entirely absent from every tool result.

This is the cheapest possible check on that, and it is deterministic. An
identifier in the answer either appeared in something the tools returned, or in
what the operator said, or it appeared from nowhere. No model is consulted,
because asking a model whether it made something up is asking the same faculty
that made it up.

It deliberately checks only identifiers. Prose cannot be checked this way, and
pretending otherwise would produce a gate that fires on ordinary English and
gets switched off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A DNS label with at least two segments. Three was the first cut, to keep
# hyphenated English out, and it let five invented workload names through in a
# real run because service names are routinely two words: messaging-sms,
# messaging-webhooks. Two segments plus an explicit stoplist catches those and
# costs a stoplist that has to be maintained, which is the better trade: a
# missed invention is silent, and a false positive is a visible line that says
# which word it means.
_DNS_NAME = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){1,}\b")

# Dotted or slashed paths: file paths, image references, resource references.
_PATH = re.compile(r"\b[\w.-]+(?:/[\w.-]+){1,}\b")

_IGNORE = frozenset({"n/a", "e.g", "i.e"})
"""Exceptions the segment rule below does not catch."""

# Hyphenated English is not built from arbitrary words: one half is almost
# always a modifier. Listing the modifiers is a rule, where listing the
# compounds they form is an inventory that grows forever and is always one
# behind. "read-only", "in-memory" and "third-party" are all caught by their
# first segment; "messaging-squad", "kube-system" and "nomic-embed-text" have
# no modifier in either position and are checked.
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
    """Whether a hyphenated token is English rather than a name.

    Bounded at three segments. "up-to-date" is three and is prose; a name long
    enough to have four is a name, and skipping it because it happens to start
    with a word like "read" would lose the check on exactly the long generated
    names it exists to catch.
    """
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
            if token in _IGNORE or ("/" not in token and _is_prose(token)):
                continue
            seen.add(token)
            found.append(token)
    return found


def check(answer: str, observed: str, *, asked: str = "") -> Grounding:
    """Which identifiers in ``answer`` appear in what was read or asked.

    ``asked`` is part of the ground truth on purpose. An operator naming a
    workload that turns out not to exist has not been hallucinated at by the
    model repeating the name back, and flagging it would train the reader to
    ignore this warning.
    """
    haystack = f"{observed}\n{asked}".lower()
    result = Grounding()
    for token in identifiers(answer):
        result.checked += 1
        if token in haystack:
            result.supported.append(token)
        else:
            result.ungrounded.append(token)
    return result


__all__ = ["Grounding", "check", "identifiers"]
