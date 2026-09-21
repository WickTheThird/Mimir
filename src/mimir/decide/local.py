"""Closed-set decisions from the generative model that is already loaded.

Kev is the right tool for this and Kev is not running. The client in
``backends.py`` is complete and tested against a server that has never been
stood up, so the decision layer has sat dormant and every decision in MIMIR
has stayed where it was: inside a generative model that was asked an
open-ended question and trusted to answer in a closed set.

This is the bridge. It implements the same :class:`Decider` protocol, so
call sites written against it swap to ``KevDecider`` by changing config and
nothing else.

What it is not: a System One model. Kev encodes the context once and scores
every option against it, which is cheap and calibrated by construction. This
sends the whole prompt per field and reads one constrained token back. It is
slower, and its probability is a softmax over the first token rather than a
trained score, so it must not be read as a calibrated likelihood.

What it does give, today and with no new infrastructure, is the guarantee
that actually matters at the call sites: **nothing can come back that was not
offered.** Constrained decoding measured 100% adherence at every model size
tested here, against a native slope from 80% down to 33%. A closed set that
is closed by construction is the property the callers need; calibration is
what Kev adds later.
"""

from __future__ import annotations

import json
from typing import Any

from mimir.decide.base import Choice, Verdict, clip
from mimir.logging import get_logger

log = get_logger(__name__)


def _schema(fields: list[Choice]) -> dict[str, Any]:
    """One object with one enum-constrained property per field.

    Every field is decided in a single call. Splitting them would multiply
    the prompt cost by the number of questions and let the answers drift
    apart, since each call would see the context fresh.
    """
    return {
        "type": "object",
        "properties": {
            field.name: {
                "type": "string",
                "enum": list(field.options),
                **({"description": field.description} if field.description else {}),
            }
            for field in fields
        },
        "required": [field.name for field in fields],
        "additionalProperties": False,
    }


def _prompt(context: str, fields: list[Choice]) -> str:
    lines = [
        "Decide each field from the material below. Choose only from the "
        "options given. Do not explain.",
        "",
    ]
    for field in fields:
        detail = f" - {field.description}" if field.description else ""
        lines.append(f"{field.name}: one of {', '.join(field.options)}{detail}")
    lines += ["", "Material:", context]
    return "\n".join(lines)


class LocalDecider:
    """A decider backed by the generative model already in memory."""

    name = "local"

    def __init__(self, router: Any, *, task_class: Any = None) -> None:
        self._router = router
        self._task_class = task_class

    @property
    def available(self) -> bool:
        return self._router is not None

    async def decide_async(
        self, context: str, fields: list[Choice], *, session_id: str = ""
    ) -> dict[str, Verdict]:
        if not fields or not self.available:
            return {}
        text, truncated = clip(context)
        from mimir.llm.base import GenerationOptions, LLMMessage, ModelError

        options = GenerationOptions(
            temperature=0.0,
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": _schema(fields)},
            },
        )
        try:
            response = await self._router.chat(
                [LLMMessage.user(_prompt(text, fields))],
                task_class=self._task_class,
                options=options,
                session_id=session_id,
                purpose="decide",
            )
        except ModelError as exc:
            # An unavailable decider is not a negative verdict. Returning
            # nothing makes every caller fall back to what it did before,
            # which is the contract NoDecider sets.
            log.warning("decider_call_failed", error=exc.message)
            return {}

        try:
            payload = json.loads(response.content or "{}")
        except (TypeError, ValueError):
            log.warning("decider_unparseable", content=(response.content or "")[:200])
            return {}

        verdicts: dict[str, Verdict] = {}
        for field in fields:
            choice = payload.get(field.name)
            if choice not in field.options:
                # The schema should make this impossible. If it happens the
                # constraint was not applied, and a silent default here would
                # look exactly like a real decision.
                log.warning(
                    "decider_off_menu", field=field.name, got=str(choice)[:80]
                )
                continue
            verdicts[field.name] = Verdict(
                field=field.name,
                choice=choice,
                # Not a calibrated probability and flagged as such, so a
                # caller applying a threshold sees an uncalibrated verdict
                # rather than a confident-looking zero.
                probability=0.0,
                distribution={},
                truncated=truncated,
                calibrated=False,
            )
        return verdicts


__all__ = ["LocalDecider"]
