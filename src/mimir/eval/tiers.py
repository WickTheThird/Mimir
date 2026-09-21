"""Compare deployment tiers on one corpus.

The laptop and the always-on box run different models, so they get different
numbers. Treating the smaller one as close enough is the assumption this
project has watched fail repeatedly, and the only way to know what an operator
gives up by being away from the workstation is to measure it.

Two things this reports that a single pass count does not.

The deterministic cases are separated out. Thirty-one of the eighty-four are
policy checks a rule answers perfectly every run, so they enter every headline
figure as guaranteed marks and flatten the difference between tiers. The model
case rate is where a tier difference actually lives.

And pair consistency is reported beside the rate rather than folded into it. A
tier can pass most cases while answering both twins of every contrastive pair
the same way, which is the difference between reading the evidence and
recognising the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Tier:
    name: str
    model: str
    total: int = 0
    passed: int = 0
    deterministic_total: int = 0
    deterministic_passed: int = 0
    pair_consistency: float | None = None
    pairs: int = 0
    pending: int = 0
    """Cases decided but not yet built. Excluded from every figure above."""

    @property
    def model_total(self) -> int:
        return self.total - self.deterministic_total

    @property
    def model_passed(self) -> int:
        return self.passed - self.deterministic_passed

    @property
    def rate(self) -> float | None:
        return round(self.passed / self.total, 3) if self.total else None

    @property
    def model_rate(self) -> float | None:
        """The number that actually separates tiers."""
        return (
            round(self.model_passed / self.model_total, 3) if self.model_total else None
        )


def load(path: Path, name: str, deterministic_ids: set[str]) -> Tier:
    payload = json.loads(Path(path).read_text())
    results = payload.get("results") or []
    tier = Tier(
        name=name,
        model=str(payload.get("model") or payload.get("model_alias") or "unrecorded"),
    )
    for result in results:
        # A pending case is decided but unbuilt. The harness excludes it from
        # its own totals, and counting it here reported a larger corpus than
        # was run and scored the unbuilt cases as passes, because a case that
        # never executed leaves ``passed`` at its default. Two tiers compared
        # this way both got the same free marks and the table looked fine.
        if result.get("pending"):
            tier.pending += 1
            continue
        passed = bool(result.get("passed"))
        tier.total += 1
        tier.passed += int(passed)
        if result.get("case_id") in deterministic_ids:
            tier.deterministic_total += 1
            tier.deterministic_passed += int(passed)
    tier.pair_consistency = payload.get("pair_consistency")
    tier.pairs = int(payload.get("pairs_seen") or 0)
    return tier


def render(tiers: list[Tier]) -> str:
    """A table that refuses to hide the guaranteed marks."""
    lines = [
        f"{'tier':<14}{'model':<20}{'all':>10}{'model-only':>13}{'pairs':>9}",
        "-" * 66,
    ]
    for tier in tiers:
        rate = f"{tier.passed}/{tier.total}" if tier.total else "-"
        model_rate = (
            f"{tier.model_passed}/{tier.model_total}" if tier.model_total else "-"
        )
        pairs = (
            f"{tier.pair_consistency:.2f}"
            if tier.pair_consistency is not None
            else "n/a"
        )
        lines.append(
            f"{tier.name:<14}{tier.model:<20}{rate:>10}{model_rate:>13}{pairs:>9}"
        )
    lines.append("")
    lines.append(
        "all includes the deterministic policy cases, which pass every run and "
        "enter every tier's figure identically."
    )
    lines.append(
        "pairs is the fraction of contrastive pairs answered correctly on both "
        "sides; a tier keying on the shape of the question scores near zero here "
        "while its case rate looks ordinary."
    )
    pending = {t.pending for t in tiers}
    if pending != {0}:
        counts = ", ".join(f"{t.name} {t.pending}" for t in tiers)
        lines.append(
            f"excluded as pending (decided, not yet built): {counts}. These are "
            "not in any figure above."
        )
    return "\n".join(lines)


__all__ = ["Tier", "load", "render"]
