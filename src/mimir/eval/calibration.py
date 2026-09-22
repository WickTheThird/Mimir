"""Calibrate the decision log against outcomes, per decision type.

ADR-004 step 5. Every decision the graph makes is logged with its options,
choice and probability. Once a session has an outcome, each logged decision
is a (probability, was_it_right) pair, and a decision type with enough pairs
can be checked and corrected.

Two rules carried over from the rest of the evaluation work:

* **Grouped, held out.** A temperature fitted on the pairs it is then
  scored on reports a calibration it does not have. Folds are grouped by
  session so a session's decisions never straddle the split.
* **Nothing is called calibrated until it is measured to be.** Below the
  minimum sample the report says "insufficient" and no temperature is
  written. `Verdict.calibrated` stays False for that decision type and the
  probability floor stays off, which is the honest state.

What "right" means differs per field and is defined here, not guessed:
for ``retrieval``, ``sufficient`` and ``conflict`` the case outcome is the
label (a decision that fed a passing answer was right); for ``next`` and
``target`` the same. This is coarse and stated as such. A finer label needs
the decision-level ground truth that step 8's records will carry.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

MIN_SAMPLES = 30
"""Fewer than this and temperature scaling fits noise. Stated once."""


@dataclass(slots=True)
class Sample:
    session_id: str
    field: str
    probability: float
    correct: bool
    calibrated_at_source: bool = False


@dataclass(slots=True)
class FieldReport:
    field: str
    samples: int
    accuracy: float | None = None
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None
    temperature: float | None = None
    status: str = "insufficient"
    """insufficient | unchanged | improved. Only "improved" writes a temperature."""

    def render(self) -> str:
        if self.status == "insufficient":
            return f"{self.field:12} n={self.samples:<4} insufficient (need {MIN_SAMPLES})"
        return (f"{self.field:12} n={self.samples:<4} acc={self.accuracy:.2f} "
                f"ece {self.ece_before:.3f}->{self.ece_after:.3f} "
                f"brier {self.brier_before:.3f}->{self.brier_after:.3f} "
                f"T={self.temperature:.2f} {self.status}")


def samples_from(results: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]) -> list[Sample]:
    """Join an eval report's results to the stored sessions' decision logs."""
    out: list[Sample] = []
    for r in results:
        sid = r.get("session_id") or ""
        state = sessions.get(sid)
        if not state:
            continue
        for d in (state.get("metadata") or {}).get("decisions") or []:
            try:
                p = float(d.get("probability", 0.0))
            except (TypeError, ValueError):
                continue
            if not d.get("acted", True):
                continue
            out.append(Sample(sid, str(d.get("field")), p, bool(r.get("passed")),
                              bool(d.get("calibrated"))))
    return out


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    if not pairs:
        return 0.0
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for p, c in pairs:
        buckets[min(bins - 1, int(p * bins))].append((p, c))
    total = len(pairs)
    return sum(
        len(b) / total * abs(sum(c for _, c in b) / len(b) - sum(p for p, _ in b) / len(b))
        for b in buckets.values()
    )


def brier(pairs: list[tuple[float, bool]]) -> float:
    return sum((p - float(c)) ** 2 for p, c in pairs) / len(pairs) if pairs else 0.0


def _scale(p: float, t: float) -> float:
    """Temperature scaling on a binary probability via its logit."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) / t
    return 1 / (1 + math.exp(-z))


def fit_temperature(pairs: list[tuple[float, bool]]) -> float:
    """Grid search on negative log likelihood. Small, exact enough, no deps."""
    best_t, best_nll = 1.0, float("inf")
    for i in range(1, 61):
        t = i / 10
        nll = -sum(
            math.log(max(1e-9, _scale(p, t) if c else 1 - _scale(p, t))) for p, c in pairs
        )
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def _folds(samples: list[Sample], k: int = 5) -> list[tuple[list[Sample], list[Sample]]]:
    """Grouped by session, so one session never straddles a split."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.session_id].append(s)
    ordered = sorted(groups)
    folds: list[tuple[list[Sample], list[Sample]]] = []
    for i in range(k):
        held = {g for j, g in enumerate(ordered) if j % k == i}
        test = [s for g in held for s in groups[g]]
        train = [s for g in ordered if g not in held for s in groups[g]]
        if test and train:
            folds.append((train, test))
    return folds


def calibrate(samples: list[Sample]) -> dict[str, FieldReport]:
    by_field: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_field[s.field].append(s)
    reports: dict[str, FieldReport] = {}
    for name, rows in sorted(by_field.items()):
        rep = FieldReport(field=name, samples=len(rows))
        if len(rows) < MIN_SAMPLES:
            reports[name] = rep
            continue
        pairs = [(s.probability, s.correct) for s in rows]
        rep.accuracy = sum(c for _, c in pairs) / len(pairs)
        rep.ece_before, rep.brier_before = ece(pairs), brier(pairs)
        # held-out: fit on train folds, score on the held fold, pool the scores
        scored: list[tuple[float, bool]] = []
        for train, test in _folds(rows):
            t = fit_temperature([(s.probability, s.correct) for s in train])
            scored += [(_scale(s.probability, t), s.correct) for s in test]
        rep.ece_after, rep.brier_after = ece(scored), brier(scored)
        rep.temperature = fit_temperature(pairs)
        rep.status = "improved" if rep.ece_after < rep.ece_before - 0.01 else "unchanged"
        reports[name] = rep
    return reports


def render(reports: dict[str, FieldReport]) -> str:
    lines = [r.render() for r in reports.values()] or ["no decisions logged"]
    lines.append("")
    lines.append("held-out, grouped by session. only 'improved' earns a temperature; "
                 "'unchanged' keeps calibrated=False and the floor off.")
    return "\n".join(lines)


def load_sessions(db_path: Any, session_ids: list[str]) -> dict[str, dict[str, Any]]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out: dict[str, dict[str, Any]] = {}
    for sid in session_ids:
        row = conn.execute("select state_json from sessions where id=?", (sid,)).fetchone()
        if row:
            out[sid] = json.loads(row[0])
    conn.close()
    return out


__all__ = ["MIN_SAMPLES", "FieldReport", "Sample", "brier", "calibrate", "ece",
           "fit_temperature", "load_sessions", "render", "samples_from"]
