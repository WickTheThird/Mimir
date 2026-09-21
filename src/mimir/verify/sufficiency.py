"""Was the evidence good enough to support the answer that was given?

Four failures survived every run of every model measured on this corpus,
from 7B to 117B. Scale moved none of them, which means they are not the
model failing to be clever enough. Each one is a fact the system already
knows and does not carry into the answer.

The worst is absence. Asked whether a billing pod exists, MIMIR answers
"no" both when a complete search found nothing and when every cluster timed
out and produced no listing at all. Those are opposite situations. The first
is an answer; the second is a failure to look. On call they demand opposite
actions, and the operator cannot tell them apart from the text.

The rule is not a judgement and does not need a model: **you cannot prove
absence from a search that did not run.** This module computes whether the
looking succeeded, and refuses definite claims that outrun it.

The same argument covers the other three:

* An empty listing names nothing, so no name may appear in the answer.
* A note nobody has verified in fourteen months is not a current fact.
* A retry storm has a shape in the timestamps. One request is not a storm.

Each is arithmetic or a pattern over data already in hand.
"""

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
    """A query could not run, or ran and produced no usable output. This is
    not a finding, and the distinction from EMPTY is the whole point."""

    UNKNOWN = "unknown"
    """Nothing said either way. Treated as insufficient for a definite claim,
    because the alternative is assuming a search happened that may not have."""


# Explicit markers that the looking failed. Kept narrow on purpose: a broad
# pattern like "error" fires on an answer that merely discusses errors, which
# would demote correct answers and look like the gate working.
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

# Markers that the looking completed. Only consulted when no failure marker is
# present, because a partial search is a failed one for the purpose of proving
# absence: some clusters answering does not make the unanswered ones empty.
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

# A definite claim that something does not exist. These are the sentences that
# a failed search cannot support.
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
    """On record, and either never verified or verified long ago. Usable as a
    lead. Not usable as a statement of what is true now."""

    UNKNOWN = "unknown"


# A note nobody ever checked is not a fact, however confidently it is written.
_NEVER_VERIFIED = re.compile(
    r"\b(never been verified|never verified|unverified|not been verified"
    r"|no(?:ne)? verification|has not been checked|never checked)\b",
    re.IGNORECASE,
)

# Age stated in the text. Months and years are always past any sane window;
# days are compared against it.
_AGE_MONTHS = re.compile(r"\b(\d+)\s*(?:month|year)s?\s*(?:ago|old)\b", re.IGNORECASE)
_AGE_DAYS = re.compile(r"\b(\d+)\s*days?\s*(?:ago|old)\b", re.IGNORECASE)

_NO_LIVE_CHECK = re.compile(
    r"\b(no live check|not been (?:re)?checked live|without checking"
    r"|no current (?:check|reading|observation))\b",
    re.IGNORECASE,
)

_ALREADY_HEDGED = re.compile(
    r"\b(verify|verif(?:y|ied|ication)|confirm|re-?check|check (?:it|this|first)"
    r"|before relying)\b",
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
        """A definite existence claim resting on a search that did not run."""
        return self.retrieval in (Retrieval.FAILED, Retrieval.UNKNOWN) and bool(
            self.absence_claims or self.presence_claims
        )


def classify_retrieval(
    *, observations: str = "", risks: str = "", executions: int = 0
) -> Retrieval:
    """Whether the looking succeeded, failed, or ran and found nothing.

    ``observations`` is everything the system has to go on: tool output,
    evidence excerpts, and the statement of the situation. ``risks`` is the
    session's recorded failures, which in production carry the tool errors.

    Failure wins over completion. A search where some targets answered and
    others timed out cannot establish that the missing ones hold nothing, and
    treating a partial result as complete is precisely the error this exists
    to stop.
    """
    blob = f"{observations}\n{risks}"
    if _FAILED.search(blob):
        return Retrieval.FAILED
    if _EMPTY.search(blob):
        # "returned no pods" already says a query ran and came back with
        # nothing, which is a finding. The failure branch above has already
        # taken every case where the running itself is in doubt.
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
    """Whether what the answer rests on is current or merely on record.

    A note written fourteen months ago that nobody has ever verified is
    treated exactly like one verified three days ago, on every model measured.
    The difference is arithmetic, and the memory bank already stores the
    timestamp that makes it.

    ``freshness`` carries the structured verdicts from the evidence when there
    is evidence; the text markers cover the case where the situation is stated
    rather than retrieved.
    """
    if executions:
        return Currency.CURRENT
    values = {str(f).lower() for f in (freshness or [])}
    if values and values <= {"stale", "unknown"}:
        return Currency.STALE
    if values & {"live", "recent"}:
        return Currency.CURRENT

    if _NEVER_VERIFIED.search(observations):
        return Currency.STALE
    if _AGE_MONTHS.search(observations):
        return Currency.STALE
    days = _AGE_DAYS.search(observations)
    if days and int(days.group(1)) > stale_after_days:
        return Currency.STALE
    if days:
        # An age was stated and it is inside the window. That is a positive
        # statement of currency, not an absence of one.
        return Currency.CURRENT
    if _NO_LIVE_CHECK.search(observations):
        return Currency.STALE
    return Currency.UNKNOWN


def demote_stale(answer: Any, currency: Currency) -> tuple[Any, int]:
    """Say that a stale record must be verified before it is relied on.

    Not deletion and not silence. The remembered value is the most useful
    thing available and the operator should see it. What must change is its
    status: a lead to confirm rather than a reading to act on.

    An answer that already tells the operator to verify is left alone, because
    appending a second instruction to an answer that gave the right one reads
    as a system that does not understand its own output.
    """
    if currency is not Currency.STALE:
        return answer, 0
    prose = getattr(answer, "answer", "") or ""
    facts = list(getattr(answer, "observed_facts", None) or [])
    if not facts and _ALREADY_HEDGED.search(prose):
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


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def check(answer_text: str, *, observations: str = "", risks: str = "",
          executions: int = 0) -> Sufficiency:
    """Does this answer claim more than the looking can support?"""
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
            if retrieval is Retrieval.FAILED
            else "no completed search is on record for this question"
        )
    return result


def demote_overreach(answer: Any, sufficiency: Sufficiency) -> tuple[Any, int]:
    """Move claims that outrun the evidence into the unverified set.

    Demotion, not deletion, for the reason the claim gate already gives: the
    claim may be true, and what is false is presenting it as established. The
    operator still sees it, labelled honestly.

    The prose answer is rewritten as well, which the claim gate does not do. A
    demoted bullet under an intact answer that still reads "there is no
    billing pod" leaves the wrong conclusion in the line the operator reads.
    """
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
        answer.answer = (
            f"Unknown: {sufficiency.reason}. The evidence does not settle "
            f"this question. Previous wording, which claimed more than the "
            f"evidence supports: {prose}"
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
    "demote_overreach",
    "demote_stale",
]
