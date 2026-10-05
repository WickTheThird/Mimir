"""Shared FastAPI dependencies."""

from __future__ import annotations

from mimir.config import get_settings
from mimir.graph.runner import InvestigationRunner, get_runner

_runner: InvestigationRunner | None = None


def get_runner_dependency() -> InvestigationRunner:
    global _runner
    if _runner is None:
        _runner = get_runner(get_settings())
    return _runner


async def shutdown_runner() -> None:
    global _runner
    if _runner is not None:
        await _runner.aclose()
        _runner = None
