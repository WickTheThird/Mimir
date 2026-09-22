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
    """Nothing said either way.

    Deliberately NOT treated as insufficient. Forty-seven of the fifty-two
    model cases in this corpus classify here, because a prompt describing a
    situation rarely narrates whether a search ran. Demoting on unknown fired
    on almost every case: "the pod is running" is a presence claim, and most
    ops answers contain one.

    A gate that fires on nearly everything is not a gate. Unknown is the
    absence of information about the looking, not evidence that the looking
    failed, and unsupported claims are already the claim gate's job. This gate
    earns its place on the one thing nothing else catches: an explicit failure
    to look, read as a finding."""


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

# The imperative only. "cannot be verified" describes the problem; it does not
# tell the operator what to do, and a corpus case that expects the instruction
# was satisfied by the description on the first measured run.
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
        """A definite existence claim resting on a search that is known to
        have failed. See :attr:`Retrieval.UNKNOWN` for why silence does not
        count."""
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
    """Whether the looking succeeded, failed, or ran and found nothing.

    ``observations`` must be the operator's own statement of the situation
    and nothing else. ``risks`` carries recorded tool failures.

    **Do not pass retrieved documents here.** The first version was fed the
    session's evidence excerpts, which include runbooks and skill bodies, and
    those discuss timeouts and unreachable hosts as subject matter. The word
    "timeout" inside a runbook about investigating rollouts classified a
    healthy case as a failed search, and the gate then injected "Unknown"
    into an answer that was correct. A comment two screens up warns that a
    broad pattern fires on text that merely discusses failures; the pattern
    below was then handed exactly that text.

    Failure wins over completion. A search where some targets answered and
    others timed out cannot establish that the missing ones hold nothing, and
    treating a partial result as complete is precisely the error this exists
    to stop.
    """
    # Structured signals first, and they settle it. A command that exited
    # non-zero failed; one that exited zero with output observed; one that
    # exited zero with nothing came back empty. No prose can override an exit
    # code, and this is the production path: the text branches below exist
    # for a situation that is stated rather than executed.
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

    # The operator's own statement about the record comes first. On the first
    # measured run a "recent" freshness on an unrelated memory hit outranked
    # "never been verified and written 14 months ago", because structured
    # signals were consulted before the statement. Structured freshness is
    # authoritative about the evidence it is attached to, and says nothing
    # about a note the operator is describing.
    if _NEVER_VERIFIED.search(observations) or _AGE_MONTHS.search(observations):
        return Currency.STALE
    days = _AGE_DAYS.search(observations)
    if days:
        # An age was stated. Inside the window it is a positive statement of
        # currency, not an absence of one.
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
        # Already says what to do. The number still comes down: a stale
        # record does not become a confident answer by being described well.
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
    """Lead with the computed verdict word under a failed or empty retrieval.

    Found in the full sweep: with the retrieval correctly classified as
    failed, the model wrote "there is insufficient evidence to confirm
    whether billing pods exist" and no overreach fired because no definite
    claim was made. Right in substance, and an operator scanning for the one
    word that settles it did not get it. The verdict is computed; its word
    should be the first thing in the answer, whatever the prose around it.

    failed -> "Unknown: the search did not complete." empty -> "None: the
    listing returned no <noun>." Nothing is removed from the prose.
    """
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
    """Under an empty listing, say that nothing was listed.

    On the first measured run the listing was empty, the model answered that
    it could not determine which pods were running for lack of tools, and the
    answer named nothing: grounding had nothing to demote and sufficiency had
    no overreach to refuse. Both gates were right and the answer was still
    wrong, because an empty result is a finding and the answer treated it as
    an absence of information. This states the finding. Computed from the
    retrieval verdict; no model.
    """
    if retrieval is not Retrieval.EMPTY:
        return answer, 0
    prose = getattr(answer, "answer", "") or ""
    if _NONE_STATED.search(prose):
        return answer, 0
    from mimir.verify.grounding import identifiers

    if identifiers(prose):
        # It names things. Whether they were observed is grounding's call.
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
    """Does this answer claim more than the looking can support?

    ``retrieval`` may be supplied already decided, by the structured signals
    or by the decision model. The text classification is the fallback for a
    stated situation, not the authority.
    """
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
        # Replace, do not prefix. The first version appended the original
        # wording after "Previous wording:", which kept the false sentence in
        # the text verbatim. The gate fired, demoted correctly, and the answer
        # still said "there is no billing pod".
        #
        # The claim is not lost: demote_overreach has already moved it into
        # unverified, which is where a claim the evidence cannot support
        # belongs. Keeping it in the prose as well is not transparency, it is
        # the answer still saying the wrong thing.
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
