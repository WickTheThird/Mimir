"""Typed decisions from a discriminative model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

MAX_OPTIONS = 26
"""Per the reference implementations."""

MAX_CONTEXT_CHARS = 8_000
"""About 2,048 tokens, which is what these models accept."""


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
    """Whether the context was cut to fit."""

    calibrated: bool = True
    """Whether :attr:`probability` means anything."""

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
        """Whether this can actually answer."""

    @property
    def name(self) -> str: ...

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        """Score every field against one context, in a single pass."""


class NoDecider:
    """The default."""

    available = False
    name = "none"

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        return {}


def clip(context: str) -> tuple[str, bool]:
    """Fit a context to the window, and report whether anything was lost."""
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
