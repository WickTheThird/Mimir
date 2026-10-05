"""Role-to-model assignment benchmarking (ADR 18.4, ADR-002 section 5)."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

from mimir.config import Settings, get_settings
from mimir.eval.harness import EvalHarness
from mimir.logging import get_logger

log = get_logger(__name__)


def enumerate_assignments(
    roles: list[str], models: list[str], limit: int | None = None
) -> list[dict[str, str]]:
    """Every mapping of role to model, optionally capped."""
    uniform = [dict.fromkeys(roles, model) for model in models]
    mixed = [
        dict(zip(roles, combo, strict=True))
        for combo in itertools.product(models, repeat=len(roles))
        if len(set(combo)) > 1
    ]
    ordered = uniform + mixed
    return ordered[:limit] if limit else ordered


def apply_assignment(settings: Settings, assignment: dict[str, str]) -> dict[str, str]:
    """Point the routing table at an assignment. Returns the previous one."""
    routing = settings.models.routing
    previous = {role: getattr(routing, role) for role in assignment}
    for role, alias in assignment.items():
        setattr(routing, role, alias)
    return previous


async def run_matrix(
    roles: list[str],
    models: list[str],
    *,
    corpus: Path | None = None,
    limit: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Score every assignment against the corpus and rank the results."""
    active = settings or get_settings()
    cases = EvalHarness.load_corpus(corpus)
    model_cases = [c for c in cases if not c.deterministic]
    if not model_cases:
        log.warning("matrix_no_model_cases")
        return []

    unknown = [m for m in models if m not in active.models.profiles]
    if unknown:
        raise ValueError(
            f"unknown model alias(es): {', '.join(unknown)}. "
            f"Configured: {', '.join(active.models.profiles)}"
        )

    assignments = enumerate_assignments(roles, models, limit)
    harness = EvalHarness(active)
    rows: list[dict[str, Any]] = []

    for index, assignment in enumerate(assignments, start=1):
        previous = apply_assignment(active, assignment)
        label = ", ".join(f"{role}={alias}" for role, alias in assignment.items())
        log.info("matrix_combination_start", index=index, total=len(assignments),
                 assignment=label)
        started = time.perf_counter()
        try:
            # A fresh runner per combination, so no model client or router cache
            from mimir.graph.runner import InvestigationRunner
            from mimir.llm.router import ModelRouter

            runner = InvestigationRunner(settings=active, router=ModelRouter(active))
            report = await harness.run_model_cases(
                model_cases, runner=runner, label=f"matrix:{label}"
            )
            await runner.aclose()
        except Exception as exc:  # noqa: BLE001 - one bad combination must not end the sweep
            log.warning("matrix_combination_failed", assignment=label, error=str(exc))
            apply_assignment(active, previous)
            continue
        finally:
            apply_assignment(active, previous)

        harness.persist(report, suite="matrix", name=label)
        rows.append(
            {
                "assignment": assignment,
                "passed": report.passed,
                "total": report.total,
                "unsupported_claim_rate": report.unsupported_claim_rate,
                "mean_duration_s": (
                    sum(r.duration_s for r in report.results) / len(report.results)
                    if report.results
                    else 0.0
                ),
                "wall_clock_s": time.perf_counter() - started,
                "run_id": report.run_id,
            }
        )

    # Rank by correctness first, then by honesty, then by speed.
    rows.sort(
        key=lambda r: (
            -r["passed"],
            r["unsupported_claim_rate"] if r["unsupported_claim_rate"] is not None else 1.0,
            r["mean_duration_s"],
        )
    )
    return rows
