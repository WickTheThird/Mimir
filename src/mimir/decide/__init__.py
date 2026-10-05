"""Typed decisions: score the answers you allow, rather than generate one."""

from __future__ import annotations

import asyncio
from typing import Any

from mimir.decide.backends import KevDecider, NimbleDecider, build_decider
from mimir.decide.base import (
    MAX_CONTEXT_CHARS,
    MAX_OPTIONS,
    Choice,
    Decider,
    NoDecider,
    Verdict,
    clip,
)
from mimir.decide.local import LocalDecider


async def decide_async(
    decider: Any, context: str, fields: list[Choice], *, session_id: str = ""
) -> dict[str, Verdict]:
    """Ask any backend from async code without blocking the loop."""
    if decider is None or not getattr(decider, "available", False) or not fields:
        return {}
    native = getattr(decider, "decide_async", None)
    if native is not None:
        return await native(context, fields, session_id=session_id)
    return await asyncio.to_thread(decider.decide, context, fields)


__all__ = [
    "MAX_CONTEXT_CHARS",
    "MAX_OPTIONS",
    "Choice",
    "Decider",
    "KevDecider",
    "LocalDecider",
    "NimbleDecider",
    "NoDecider",
    "Verdict",
    "build_decider",
    "clip",
    "decide_async",
]
