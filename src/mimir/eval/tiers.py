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
from dataclasses import dataclass, field
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
    verdicts: dict[str, bool] = field(default_factory=dict)
    """case_id -> passed, for the cases that ran. Needed to ask whether two
    tiers failed the *same* cases, which a pass count cannot answer."""
    case_pairs: dict[str, str] = field(default_factory=dict)
    """case_id -> pair name, for the contrastive cases."""

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
        case_id = str(result.get("case_id") or "")
        tier.verdicts[case_id] = passed
        if result.get("pair"):
            tier.case_pairs[case_id] = str(result["pair"])
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


def pair_table(tiers: list[Tier]) -> str:
    """Which pairs each tier answers on both sides, side by side.

    The headline rate cannot tell a tier that is weaker from a tier that is
    wrong about different things. Two tiers scoring 63 and 64 might disagree
    on thirty cases or on one. Only the per-case verdicts say which.
    """
    names = sorted({p for t in tiers for p in t.case_pairs.values()})
    if not names:
        return "no contrastive pairs in these runs."

    def both_sides(tier: Tier, pair: str) -> bool | None:
        sides = [
            tier.verdicts[c] for c, p in tier.case_pairs.items() if p == pair
        ]
        # One side missing means the pair was not actually tested as a pair.
        return all(sides) if len(sides) == 2 else None

    width = max(len(n) for n in names) + 2
    head = f"{'pair':<{width}}" + "".join(f"{t.name:<10}" for t in tiers)
    lines = [head, "-" * len(head)]
    for pair in names:
        row = f"{pair:<{width}}"
        marks = [both_sides(t, pair) for t in tiers]
        for mark in marks:
            row += f"{'both' if mark else 'split' if mark is False else 'n/a':<10}"
        if len({m for m in marks if m is not None}) > 1:
            row += "  <- tiers differ"
        lines.append(row)
    return "\n".join(lines)


def shared_failures(runs: list[Tier]) -> tuple[int, int, list[str]]:
    """Cases every run attempted, and whether they all reached the same verdict.

    Read the warning before using the agreement figure for anything.

    Agreement between two single runs of different models is not evidence
    about the models. Measured on this corpus, one run of qwen3-coder:30b
    agreed with qwen2.5:7b on 84% of model cases and another run of the *same*
    30B agreed with the same 7B on 64%, while the two 30B runs agreed with
    each other on 72%. The cross-model figure moved 20 points depending on
    which run was picked, which is more than the gap the statistic was being
    used to detect.

    So the pair (agreeing, compared) is descriptive only. The third element -
    cases that failed in every run supplied - is the part that supports an
    argument, and only in proportion to how many runs were supplied. Pass
    replicates, not one run per model, and see :func:`stable_failures`.

    Returns (agreeing, compared, case_ids that failed in every run).
    """
    if len(runs) < 2:
        return (0, 0, [])
    common = set(runs[0].verdicts)
    for run in runs[1:]:
        common &= set(run.verdicts)
    agreeing = sum(1 for c in common if len({r.verdicts[c] for r in runs}) == 1)
    failed_everywhere = sorted(
        c for c in common if not any(r.verdicts[c] for r in runs)
    )
    return (agreeing, len(common), failed_everywhere)


def stable_failures(
    runs: list[Tier], minimum_runs: int = 3
) -> tuple[list[str], str | None]:
    """Cases that failed in every run, with a caveat when there are too few.

    A case that fails once is a case that failed once. Per-case churn on this
    corpus is roughly a fifth, so at two runs a quarter of the cases that look
    permanently broken are not: three of the nine contrastive failures that a
    two-run comparison called structural passed in a third run that was
    already on disk.

    Returning the caveat alongside the list is the point. The previous version
    of this comparison returned the list alone, it read as a finding, and it
    was written into a document as one.

    Returns (case_ids failing in every run, caveat or None when satisfied).
    """
    if not runs:
        return ([], "no runs supplied")
    common = set(runs[0].verdicts)
    for run in runs[1:]:
        common &= set(run.verdicts)
    failing = sorted(c for c in common if not any(r.verdicts[c] for r in runs))
    if len(runs) >= minimum_runs:
        return (failing, None)
    return (
        failing,
        f"{len(runs)} run(s), below the {minimum_runs} this corpus needs: "
        f"per-case churn is about 20%, so roughly {len(failing) // 5} of these "
        f"{len(failing)} would pass on another run. Not evidence of a "
        f"structural failure yet.",
    )


__all__ += ["pair_table", "shared_failures", "stable_failures"]
