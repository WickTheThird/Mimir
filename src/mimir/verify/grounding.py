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

# A DNS label with at least three segments: pod and workload names, and almost
# nothing an English sentence contains. Two segments would match "read-only"
# and "fail-open", which are words this project's own output is full of.
_DNS_NAME = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b")

# Dotted or slashed paths: file paths, image references, resource references.
_PATH = re.compile(r"\b[\w.-]+(?:/[\w.-]+){1,}\b")

_IGNORE = frozenset({
    # Shapes that look like identifiers and are ordinary vocabulary.
    "up-to-date", "out-of-date", "read-only", "fail-open", "fail-closed",
    "day-to-day", "end-to-end", "n/a",
})


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
            if token in _IGNORE or token in seen or len(token) < 8:
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
