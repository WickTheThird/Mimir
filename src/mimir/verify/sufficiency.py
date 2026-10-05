"""Was the evidence good enough to support the answer that was given?"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Retrieval(StrEnum):
    """What happened when the system went to look."""

    OBSERVED = "observed"
    """A query ran and returned data."""

    EMPTY = "empty"
    """A query ran and returned nothing. This is a finding."""

    FAILED = "failed"
    """A query could not run, or ran and produced no usable output."""

    UNKNOWN = "unknown"
    """Nothing said either way."""


# Explicit markers that the looking failed.
_FAILED = re.compile(
    r"\b("
    r"timed out|timeout(?:s)?|connection refused|connection reset"
    r"|could not (?:reach|connect|be reached|be queried|query|list)"
    r"|unable to (?:reach|connect|query|list)"
    r"|unreachable|no listing was produced|no output was produced"
    r"|permission denied|forbidden|unauthori[sz]ed"
    r"|did not respond|failed to (?:list|reach|connect|query)"
    r")\b",
    re.IGNORECASE,
)

# Markers that the looking completed.
_COMPLETED = re.compile(
    r"\b("
    r"answered successfully|returned successfully|completed successfully"
    r"|all (?:\w+ )?(?:clusters|namespaces|hosts|nodes) (?:answered|responded|returned)"
    r"|the (?:search|query|listing|scan) completed"
    r")\b",
    re.IGNORECASE,
)

_EMPTY = re.compile(
    r"\b("
    r"returned no \w+|found no \w+|no \w+ (?:were |was )?found"
    r"|returned nothing|found nothing|no results|no matches"
    r"|(?:an? )?empty (?:list|listing|result|response)"
    r"|no pods at all|listed no \w+"
    r")\b",
    re.IGNORECASE,
)

# A definite claim that something does not exist.
_ABSENCE_CLAIM = re.compile(
    r"\b("
    r"there (?:is|are) no\b|there (?:is|are)n't (?:any|a)\b"
    r"|no such \w+|does not exist|do not exist|doesn't exist|don't exist"
    r"|(?:is|are) not (?:present|running|deployed|there)"
    r"|no \w+ (?:is|are) running"
    r"|nothing (?:is |was )?(?:running|deployed|found|present)"
    r")",
    re.IGNORECASE,
)

_PRESENCE_CLAIM = re.compile(
    r"\b(there (?:is|are) (?:a|an|\d+|several|multiple)\b|(?:is|are) running\b)",
    re.IGNORECASE,
)


class Currency(StrEnum):
    """How current the thing the answer rests on actually is."""

    CURRENT = "current"
    """Observed this run, or verified inside the freshness window."""

    STALE = "stale"
    """On record, and either never verified or verified long ago."""

    UNKNOWN = "unknown"


# A note nobody ever checked is not a fact, however confidently it is written.
_NEVER_VERIFIED = re.compile(
    r"\b(never been verified|never verified|unverified|not been verified"
    r"|no(?:ne)? verification|has not been checked|never checked)\b",
    re.IGNORECASE,
)

# Age stated in the text.
_AGE_MONTHS = re.compile(r"\b(\d+)\s*(?:month|year)s?\s*(?:ago|old)\b", re.IGNORECASE)
_AGE_DAYS = re.compile(r"\b(\d+)\s*days?\s*(?:ago|old)\b", re.IGNORECASE)

_NO_LIVE_CHECK = re.compile(
    r"\b(no live check|not been (?:re)?checked live|without checking"
    r"|no current (?:check|reading|observation))\b",
    re.IGNORECASE,
)

# The imperative only.
_ALREADY_HEDGED = re.compile(
    r"\b(verify|confirm|re-?check|check (?:it|this|first)|before relying)\b",
    re.IGNORECASE,
)

_NONE_STATED = re.compile(
    r"\b(none|no pods?|there (?:is|are) no\b|no such \w+|no \w+ (?:is|are|were) (?:running|listed|found)|nothing (?:is |was )?"
    r"(?:running|listed|returned|found)|returned (?:nothing|no results))\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Sufficiency:
    """Whether the evidence supports the definiteness of the answer."""

    retrieval: Retrieval = Retrieval.UNKNOWN
    absence_claims: list[str] = field(default_factory=list)
    presence_claims: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def overreaching(self) -> bool:
        """A definite existence claim resting on a search that is known to have failed."""
        return self.retrieval is Retrieval.FAILED and bool(
            self.absence_claims or self.presence_claims
        )


def classify_retrieval(
    *,
    observations: str = "",
    risks: str = "",
    executions: int = 0,
    failed: int = 0,
    empty: int = 0,
    observed: int = 0,
) -> Retrieval:
    """Whether the looking succeeded, failed, or ran and found nothing."""
    # Structured signals first, and they settle it.
    if failed:
        return Retrieval.FAILED
    if observed or executions:
        return Retrieval.OBSERVED
    if empty:
        return Retrieval.EMPTY

    blob = f"{observations}\n{risks}"
    if _FAILED.search(blob):
        return Retrieval.FAILED
    if _EMPTY.search(blob):
        # "returned no pods" already says a query ran and came back with
        return Retrieval.EMPTY
    if _COMPLETED.search(blob) or executions:
        return Retrieval.OBSERVED
    return Retrieval.UNKNOWN


def classify_currency(
    *,
    observations: str = "",
    freshness: list[str] | None = None,
    stale_after_days: int = 30,
    executions: int = 0,
) -> Currency:
    """Whether what the answer rests on is current or merely on record."""
    if executions:
        return Currency.CURRENT

    # The operator's own statement about the record comes first.
    if _NEVER_VERIFIED.search(observations) or _AGE_MONTHS.search(observations):
        return Currency.STALE
    days = _AGE_DAYS.search(observations)
    if days:
        # An age was stated.
        return Currency.STALE if int(days.group(1)) > stale_after_days else Currency.CURRENT

    values = {str(f).lower() for f in (freshness or [])}
    if values and values <= {"stale", "unknown"}:
        return Currency.STALE
    if values & {"live", "recent"}:
        return Currency.CURRENT
    if _NO_LIVE_CHECK.search(observations):
        return Currency.STALE
    return Currency.UNKNOWN


def demote_stale(answer: Any, currency: Currency) -> tuple[Any, int]:
    """Say that a stale record must be verified before it is relied on."""
    if currency is not Currency.STALE:
        return answer, 0
    prose = getattr(answer, "answer", "") or ""
    facts = list(getattr(answer, "observed_facts", None) or [])
    if not facts and _ALREADY_HEDGED.search(prose):
        # Already says what to do.
        answer.confidence = round(min(getattr(answer, "confidence", 0.5), 0.35), 3)
        return answer, 0

    answer.observed_facts = []
    answer.unverified = [
        *(getattr(answer, "unverified", None) or []),
        *[f"{f} (from an unverified or stale record)" for f in facts],
    ]
    if not _ALREADY_HEDGED.search(prose):
        answer.answer = (
            f"{prose}\n\nThis rests on a stored note that has not been "
            f"verified recently. Verify it against the live system before "
            f"relying on the value."
        ).strip()
    answer.confidence = round(min(getattr(answer, "confidence", 0.5), 0.35), 3)
    return answer, len(facts) or 1


_UNKNOWN_STATED = re.compile(r"^\s*unknown\b|\bunknown\b", re.IGNORECASE)


def state_verdict(answer: Any, retrieval: Retrieval, *, noun: str = "items") -> tuple[Any, int]:
    """Lead with the computed verdict word under a failed or empty retrieval."""
    prose = getattr(answer, "answer", "") or ""
    if retrieval is Retrieval.FAILED and not _UNKNOWN_STATED.search(prose):
        answer.answer = f"Unknown: the search did not complete. {prose}".strip()
        return answer, 1
    if retrieval is Retrieval.EMPTY and not re.search(r"\bnone\b", prose, re.IGNORECASE):
        from mimir.verify.grounding import identifiers

        if not identifiers(prose):
            answer.answer = f"None: the listing returned no {noun}. {prose}".strip()
            return answer, 1
    return answer, 0


def demote_empty(answer: Any, retrieval: Retrieval, *, noun: str = "items") -> tuple[Any, int]:
    """Under an empty listing, say that nothing was listed."""
    if retrieval is not Retrieval.EMPTY:
        return answer, 0
    prose = getattr(answer, "answer", "") or ""
    if _NONE_STATED.search(prose):
        return answer, 0
    from mimir.verify.grounding import identifiers

    if identifiers(prose):
        # It names things.
        return answer, 0
    answer.answer = (
        f"None. The listing returned no {noun}, so there are none to name. "
        + prose
    ).strip()
    return answer, 1


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def check(
    answer_text: str,
    *,
    observations: str = "",
    risks: str = "",
    executions: int = 0,
    retrieval: Retrieval | None = None,
) -> Sufficiency:
    """Does this answer claim more than the looking can support?"""
    if retrieval is None:
        retrieval = classify_retrieval(
            observations=observations, risks=risks, executions=executions
        )
    result = Sufficiency(retrieval=retrieval)
    for sentence in _sentences(answer_text):
        if _ABSENCE_CLAIM.search(sentence):
            result.absence_claims.append(sentence)
        elif _PRESENCE_CLAIM.search(sentence):
            result.presence_claims.append(sentence)
    if result.overreaching:
        result.reason = (
            "the search did not complete, so absence cannot be established"
        )
    return result


def demote_overreach(answer: Any, sufficiency: Sufficiency) -> tuple[Any, int]:
    """Move claims that outrun the evidence into the unverified set."""
    if not sufficiency.overreaching:
        return answer, 0

    claims = [*sufficiency.absence_claims, *sufficiency.presence_claims]
    answer.unverified = [
        *(getattr(answer, "unverified", None) or []),
        *[f"{c} ({sufficiency.reason})" for c in claims],
    ]

    facts = getattr(answer, "observed_facts", None) or []
    kept = [f for f in facts if not _ABSENCE_CLAIM.search(f) and not _PRESENCE_CLAIM.search(f)]
    moved = len(facts) - len(kept)
    answer.observed_facts = kept

    prose = getattr(answer, "answer", "") or ""
    if _ABSENCE_CLAIM.search(prose) or _PRESENCE_CLAIM.search(prose):
        # Replace, do not prefix.
        answer.answer = (
            f"Unknown: {sufficiency.reason}. The evidence does not settle "
            f"this question. See the unverified list for what was claimed "
            f"before the evidence was checked."
        )

    answer.confidence = round(min(getattr(answer, "confidence", 0.5), 0.3), 3)
    return answer, len(claims) + moved


__all__ = [
    "Currency",
    "Retrieval",
    "Sufficiency",
    "check",
    "classify_currency",
    "classify_retrieval",
    "demote_empty",
    "demote_overreach",
    "demote_stale",
    "state_verdict",
]
