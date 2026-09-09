"""Make a tool call the only thing the model can emit.

Measured, this model produces a tool call for 80% of prompts at a four tool
surface and 33% at eighteen. The rest of the time it writes the call into its
prose, complete with closing tags, and a turn with no tool calls looks exactly
like a turn that finished. A regex that detects that shape and asks the model
to try again was the first response, and it treats a decoder problem at the
wrong layer.

Constraining the decoder removes the failure instead of detecting it. The
runtime is given a JSON schema and can only emit tokens that keep the output
valid against it, so "wrote the call as prose" is not a thing that can happen.
The schema is a union over the available tools, discriminated by name, with
each branch carrying that tool's own argument schema. Choosing a tool that does
not exist and inventing an argument name are both excluded by construction
rather than validated afterwards.

There is one escape branch, ``answer``, because a loop whose only legal move is
another tool call cannot stop.

What this costs is the model's native tool-calling template, which was trained
on and may choose better. That is an empirical question, and mimir eval probe
tool_adherence answers it rather than an argument.
"""

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
    """A union over the tools, discriminated by name.

    ``hidden`` arguments are removed exactly as they are from the native
    schemas, so the two paths offer the model the same choices and a
    measurement comparing them is comparing the decoder rather than the
    surface.
    """
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
    """Read one decoded step.

    The schema guarantees the shape, so this is not defensive parsing; it is
    the two cases where a runtime can still hand back something else. An empty
    response ends the turn rather than raising, because a turn that produced
    nothing is finished whatever the reason.
    """
    text = (content or "").strip()
    if not text:
        return ConstrainedStep(say="")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # The constraint should make this impossible. If a runtime ever returns
        # unconstrained text anyway, treat it as the answer rather than losing
        # it, and say so in the log so the assumption can be checked.
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
