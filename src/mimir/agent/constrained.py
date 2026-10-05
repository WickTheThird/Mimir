"""Make a tool call the only thing the model can emit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mimir.llm.base import ToolCall
from mimir.logging import get_logger

log = get_logger(__name__)

ANSWER = "answer"
"""The branch that ends a turn. Without it the loop cannot stop."""

MAX_SAY = 400


@dataclass
class ConstrainedStep:
    """One decoded step: either a tool call or the end of the turn."""

    say: str = ""
    tool: str = ""
    arguments: dict[str, Any] | None = None

    @property
    def finished(self) -> bool:
        return not self.tool or self.tool == ANSWER

    def as_tool_call(self) -> ToolCall | None:
        if self.finished:
            return None
        return ToolCall(name=self.tool, arguments=self.arguments or {})


ANSWER_HELP = (
    "Finish. Choose this as soon as you have what was asked for, and put the "
    "answer in say. Every other branch runs a tool and continues."
)


def _branch(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    branch: dict[str, Any] = {
        "type": "object",
        "description": ANSWER_HELP if name == ANSWER else f"Call {name}.",
        "properties": {
            "say": {
                "type": "string",
                "description": (
                    "The answer for the operator."
                    if name == ANSWER
                    else "One short sentence saying what you are about to do."
                ),
            },
            "tool": {"const": name},
        },
        "required": ["tool", "say"],
    }
    if arguments is not None:
        branch["properties"]["arguments"] = arguments
        branch["required"].append("arguments")
    return branch


def build_schema(specs: list[Any], hidden: tuple[str, ...] = ()) -> dict[str, Any]:
    """A union over the tools, discriminated by name."""
    branches = []
    for spec in specs:
        schema = spec.json_schema()
        properties = schema.get("properties") or {}
        for name in hidden:
            properties.pop(name, None)
        required = schema.get("required")
        if required:
            schema["required"] = [r for r in required if r not in hidden]
        branches.append(_branch(spec.name, schema))
    branches.append(_branch(ANSWER, None))
    return {"anyOf": branches}


def parse_step(content: str) -> ConstrainedStep:
    """Read one decoded step."""
    text = (content or "").strip()
    if not text:
        return ConstrainedStep(say="")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # The constraint should make this impossible.
        log.warning("constrained_output_was_not_json", chars=len(text))
        return ConstrainedStep(say=text[:MAX_SAY])
    if not isinstance(payload, dict):
        return ConstrainedStep(say=str(payload)[:MAX_SAY])
    return ConstrainedStep(
        say=str(payload.get("say") or "")[:MAX_SAY],
        tool=str(payload.get("tool") or ""),
        arguments=payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
    )


__all__ = ["ANSWER", "ConstrainedStep", "build_schema", "parse_step"]
