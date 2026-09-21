"""Typed decisions from a discriminative model.

A System One model does not write an answer, it scores the answers you allow.
The context is encoded once and each candidate is scored against it, so what
comes back is a probability over a closed set rather than a string somebody has
to parse. Nothing can be emitted that was not offered.

That is a different shape from everything else MIMIR asks a model, and it fits
exactly two places. MIMIR reports an uncalibrated support score and refuses to
call it a probability, because the learned-confidence attempt reached an AUC of
0.510 on 21 cases; a model calibrated by construction is that gap. And the
best-of-k selector can check that a change parses, links up and passes its
tests, but not whether it did what was asked, which is why its tie break prefers
the smallest diff and the incomplete attempts are the smallest.

It fits nowhere else, and the reason is worth writing down. Risk classification
is deterministic because safety must not depend on a probability. Triage routing
is a rule that is right every time it has been measured. Tool choice is already
100% under a constrained decoder. Replacing a rule that is always right with a
classifier that is right ninety percent of the time is a regression, however
good the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

MAX_OPTIONS = 26
"""Per the reference implementations. A larger set is refused rather than
silently truncated, because dropping the option that was correct produces a
confident answer from a smaller world."""

MAX_CONTEXT_CHARS = 8_000
"""About 2,048 tokens, which is what these models accept. MIMIR's own prompts
are four times that, so a caller has to choose what to send rather than hope."""


@dataclass(frozen=True)
class Choice:
    """One field to decide, and the answers allowed for it."""

    name: str
    options: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not 1 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(
                f"{self.name}: {len(self.options)} options, but a typed decision "
                f"allows 1 to {MAX_OPTIONS}"
            )
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"{self.name}: duplicate options")


@dataclass(frozen=True)
class Verdict:
    """What came back, and how sure it was."""

    field: str
    choice: str
    probability: float
    distribution: dict[str, float] = field(default_factory=dict)
    truncated: bool = False
    """Whether the context was cut to fit. A decision made on part of the
    evidence is not the same as a decision made on all of it, and a caller that
    cannot tell them apart will treat them the same."""

    calibrated: bool = True
    """Whether :attr:`probability` means anything.

    A discriminative model scores every option against one encoding of the
    context, so its number is a probability by construction. A backend that
    instead constrains a generative model to a closed set gets the closed set
    but not the score: what comes back is a softmax over a first token, which
    ranks poorly and whose magnitude is meaningless. Both are useful and they
    are not interchangeable.

    Callers that gate on a threshold must check this first. The alternative
    was for an uncalibrated backend to report a number anyway, and a caller
    comparing 0.0 against a 0.7 floor would discard every correct decision
    while looking like it was being careful.

    Same rule as FinalAnswer.probability, which stays None until a calibration
    model exists: a field that cannot be trusted says so structurally rather
    than by convention."""

    @property
    def margin(self) -> float:
        """Distance from the runner-up. A win by a nose is not a decision."""
        ranked = sorted(self.distribution.values(), reverse=True)
        return round(ranked[0] - ranked[1], 4) if len(ranked) > 1 else 1.0

    @property
    def confident(self) -> bool:
        """Whether a threshold gate may be applied to this verdict at all."""
        return self.calibrated

    def render(self) -> str:
        marks = "".join(
            [" (truncated)" if self.truncated else "",
             "" if self.calibrated else " (uncalibrated)"]
        )
        score = f" p={self.probability:.2f}" if self.calibrated else ""
        return f"{self.field}={self.choice}{score}{marks}"


@runtime_checkable
class Decider(Protocol):
    """What any backend has to provide."""

    @property
    def available(self) -> bool:
        """Whether this can actually answer. Absence is not a negative
        verdict, and a caller must be able to tell the difference."""

    @property
    def name(self) -> str: ...

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        """Score every field against one context, in a single pass."""


class NoDecider:
    """The default. Answers nothing, and says so.

    A missing decision model must not look like a decision. Every caller
    handles this by falling back to what it did before, never by treating an
    absent verdict as a low score.
    """

    available = False
    name = "none"

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        return {}


def clip(context: str) -> tuple[str, bool]:
    """Fit a context to the window, and report whether anything was lost.

    The tail is kept rather than the head: in every use here the question and
    the material it is about are at the end, and the preamble is what can go.
    """
    if len(context) <= MAX_CONTEXT_CHARS:
        return context, False
    return context[-MAX_CONTEXT_CHARS:], True


__all__ = [
    "MAX_CONTEXT_CHARS",
    "MAX_OPTIONS",
    "Choice",
    "Decider",
    "NoDecider",
    "Verdict",
    "clip",
]
