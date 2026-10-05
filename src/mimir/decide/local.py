"""Closed-set decisions from the generative model that is already loaded."""

from __future__ import annotations

import json
from typing import Any

from mimir.decide.base import Choice, Verdict, clip
from mimir.logging import get_logger

log = get_logger(__name__)


def _schema(fields: list[Choice]) -> dict[str, Any]:
    """One object with one enum-constrained property per field."""
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
            # An unavailable decider is not a negative verdict.
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
                # The schema should make this impossible.
                log.warning(
                    "decider_off_menu", field=field.name, got=str(choice)[:80]
                )
                continue
            verdicts[field.name] = Verdict(
                field=field.name,
                choice=choice,
                # Not a calibrated probability and flagged as such, so a
                probability=0.0,
                distribution={},
                truncated=truncated,
                calibrated=False,
            )
        return verdicts


__all__ = ["LocalDecider"]
