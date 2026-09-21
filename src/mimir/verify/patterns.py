"""A retry storm has a shape. One request is not a storm.

con-retry-innocent-b failed in every run of every model from 7B to 117B.
Shown a single request followed by connection pool exhaustion, with a batch
job that opened four hundred connections sitting in plain view, MIMIR
diagnoses a retry storm. Shown a real one, it also diagnoses a retry storm,
so the diagnosis carries no information.

Retries with exponential backoff leave a signature: the same identifier
several times, with the gap between attempts roughly doubling. That is a
pattern over timestamps, not a judgement, and it either is present or it is
not.

The gate is one-directional on purpose. Finding the signature does not
establish that retries caused the incident, so it never upgrades an answer.
Its absence does establish that this particular story is unsupported, so it
demotes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WORD_COUNTS = {
    "once": 1, "twice": 2, "thrice": 3,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_REPEAT = re.compile(
    r"\b(?:the same [\w ]{0,24}?(?:id|identifier|request|line|message)\b[^.]{0,40}?)"
    r"(?:(\d+)|(" + "|".join(_WORD_COUNTS) + r"))\s*times?\b",
    re.IGNORECASE,
)
_REPEAT_ONCE = re.compile(
    r"\bthe same [\w ]{0,24}?(?:id|identifier|request|line|message)\b[^.]{0,24}?\bonce\b",
    re.IGNORECASE,
)
# Greedy over the whole list. A non-greedy version stopped at the second
# value and read "gaps of 1s, 2s and 4s" as two gaps. The verdict came out
# right anyway, which is how a parsing bug survives a passing test.
_GAPS = re.compile(
    r"\bgaps? of ((?:[0-9.]+\s*(?:ms|s|m|h)\b[\s,]*(?:and\s*)?)+)", re.IGNORECASE
)
_DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)\b", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

# A cause attributed to retrying. Deliberately narrow: an answer that merely
# mentions a retry configuration is not diagnosing a storm.
_RETRY_CAUSE = re.compile(
    r"\b(retry storm|retry amplification|retrying|retries|retry loop"
    r"|repeated retries|client retries|exponential backoff)\b",
    re.IGNORECASE,
)

MIN_ATTEMPTS = 3
"""Two points make a gap. Three make a trend.

Two attempts always look like backoff because a single gap cannot fail to be
consistent with doubling, so the threshold is three or the test passes on
every pair of lines in any log.
"""


@dataclass(slots=True)
class RetryEvidence:
    occurrences: int = 0
    gaps: list[float] = field(default_factory=list)

    @property
    def doubling(self) -> bool:
        """Whether each gap is roughly twice the one before it.

        The tolerance is wide because real backoff carries jitter, and a
        scheduler under load stretches gaps further. Narrow tolerance would
        reject genuine storms, and this gate's job is to catch the answer that
        has no pattern at all, not to grade the pattern's tidiness.
        """
        if len(self.gaps) < 2:
            return False
        return all(
            previous > 0 and 1.4 <= (nxt / previous) <= 4.0
            for previous, nxt in zip(self.gaps, self.gaps[1:], strict=False)
        )

    @property
    def is_storm(self) -> bool:
        return self.occurrences >= MIN_ATTEMPTS and self.doubling


def from_timestamps(timestamps: list[float]) -> RetryEvidence:
    """The production path: actual times of repeated identical events."""
    ordered = sorted(timestamps)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    return RetryEvidence(occurrences=len(ordered), gaps=gaps)


def from_text(text: str) -> RetryEvidence:
    """The stated path: a situation described rather than retrieved."""
    evidence = RetryEvidence()
    match = _REPEAT.search(text or "")
    if match:
        digits, word = match.group(1), match.group(2)
        evidence.occurrences = int(digits) if digits else _WORD_COUNTS[word.lower()]
    elif _REPEAT_ONCE.search(text or ""):
        evidence.occurrences = 1
    gaps = _GAPS.search(text or "")
    if gaps:
        evidence.gaps = [
            float(value) * _UNIT_SECONDS[unit.lower()]
            for value, unit in _DURATION.findall(gaps.group(1))
        ]
    return evidence


def claims_retries(text: str) -> bool:
    return bool(_RETRY_CAUSE.search(text or ""))


def demote_unsupported_retry(answer: Any, evidence: RetryEvidence) -> tuple[Any, int]:
    """Withdraw a retry diagnosis that the timing does not show.

    Only ever demotes. A present signature is consistent with retries having
    caused the incident but does not establish it, so a gate that promoted on
    a match would be manufacturing confidence from a correlation.
    """
    text = " ".join(
        filter(
            None,
            [
                getattr(answer, "answer", "") or "",
                *(getattr(answer, "observed_facts", None) or []),
            ],
        )
    )
    if not claims_retries(text) or evidence.is_storm:
        return answer, 0

    reason = (
        f"only {evidence.occurrences} occurrence(s) recorded, and a retry "
        f"storm needs at least {MIN_ATTEMPTS}"
        if evidence.occurrences < MIN_ATTEMPTS
        else "the gaps between attempts do not widen the way backoff does"
    )
    facts = list(getattr(answer, "observed_facts", None) or [])
    kept = [f for f in facts if not claims_retries(f)]
    moved = [f for f in facts if claims_retries(f)]
    answer.observed_facts = kept
    answer.unverified = [
        *(getattr(answer, "unverified", None) or []),
        *[f"{f} ({reason})" for f in moved],
    ]
    if claims_retries(getattr(answer, "answer", "")):
        answer.answer = (
            f"{answer.answer}\n\nThe retry explanation is not supported by the "
            f"timing: {reason}. Look for another source of the connections."
        )
    answer.confidence = round(min(getattr(answer, "confidence", 0.5), 0.35), 3)
    return answer, len(moved) or 1


__all__ = [
    "MIN_ATTEMPTS",
    "RetryEvidence",
    "claims_retries",
    "demote_unsupported_retry",
    "from_text",
    "from_timestamps",
]
