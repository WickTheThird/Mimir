"""Multi-step coding tasks with checkpoints.

Plan step 6c. The loop is one instruction, one worktree, N turns. A task
that needs "the model, then the migration, then the endpoint" had no
representation, so the model either did it all in one diff or lost the
thread. Here the instruction becomes an ordered list of steps (open prose,
tier 3, one structured call), each step runs through the same loop, and a
step that passed the gate is committed in the worktree as a checkpoint. A
failed step is rolled back to the last checkpoint rather than left half
done, and the task reports which steps landed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

MAX_STEPS = 6

_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": MAX_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "instruction": {"type": "string"},
                    "done_when": {"type": "string"},
                },
                "required": ["title", "instruction", "done_when"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class Step:
    title: str
    instruction: str
    done_when: str = ""
    status: str = "pending"
    """pending | done | failed | skipped"""
    checkpoint: str = ""
    detail: str = ""


@dataclass(slots=True)
class TaskPlan:
    instruction: str
    steps: list[Step] = field(default_factory=list)

    @property
    def landed(self) -> list[Step]:
        return [s for s in self.steps if s.status == "done"]

    def render(self) -> str:
        return "\n".join(f"[{s.status:7}] {s.title}" for s in self.steps)


async def plan_steps(router: Any, instruction: str, *, session_id: str = "") -> TaskPlan:
    """Split one instruction into ordered steps. One call, closed shape.

    A single-step instruction comes back as one step; the plan machinery is
    then a no-op with a checkpoint, which costs nothing.
    """
    from mimir.llm.base import GenerationOptions, LLMMessage, ModelError

    prompt = (
        "Split this coding instruction into the smallest ordered list of steps that "
        "each leave the repository working. One step if it is already one thing. "
        "For each step give a title, the instruction for that step alone, and how to "
        "tell it is done.\n\nInstruction:\n" + instruction
    )
    try:
        response = await router.chat(
            [LLMMessage.user(prompt)],
            options=GenerationOptions(
                temperature=0.0, max_tokens=800,
                response_format={"type": "json_schema", "json_schema": {"name": "plan", "schema": _SCHEMA}},
            ),
            session_id=session_id, purpose="coding:plan",
        )
        payload = json.loads(response.content or "{}")
        steps = [Step(str(s["title"]), str(s["instruction"]), str(s.get("done_when", "")))
                 for s in payload.get("steps", []) if s.get("instruction")]
    except (ModelError, ValueError, KeyError, TypeError) as exc:
        log.warning("plan_failed", error=str(exc))
        steps = []
    if not steps:
        steps = [Step("the task", instruction)]
    return TaskPlan(instruction=instruction, steps=steps[:MAX_STEPS])


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)


def checkpoint(root: Path, title: str) -> str:
    """Commit the worktree as it stands. Returns the commit id, or ''."""
    _git(root, "add", "-A")
    done = _git(root, "-c", "user.email=mimir@local", "-c", "user.name=mimir",
                "commit", "-q", "-m", f"checkpoint: {title}")
    if done.returncode != 0:
        return ""
    return _git(root, "rev-parse", "--short", "HEAD").stdout.strip()


def rollback(root: Path) -> None:
    """Back to the last checkpoint. Untracked files from the failed step go too."""
    _git(root, "reset", "-q", "--hard")
    _git(root, "clean", "-qfd")


async def run_plan(agent: Any, plan: TaskPlan, *, worktree_root: Path,
                   gate_ok: Any = None) -> TaskPlan:
    """Run each step through the agent; checkpoint what lands, roll back what fails.

    ``gate_ok(agent)`` decides whether a step's result may be kept. Default:
    the loop stopped normally and made a change. The caller can pass the
    change gate or the test result instead, which is what the corpus does.
    """
    from mimir.agent.loop import AgentEventType

    ok = gate_ok or (lambda a: a.outcome.stopped == "done" and bool(a.outcome.files_changed))
    for index, step in enumerate(plan.steps):
        context = ""
        if index:
            context = ("Steps already done and committed: "
                       + "; ".join(s.title for s in plan.steps[:index] if s.status == "done") + ". ")
        text: list[str] = []
        async for event in agent.run(f"{context}Now: {step.instruction}"
                                     + (f" Done when: {step.done_when}" if step.done_when else "")):
            if event.type is AgentEventType.TEXT:
                text.append(event.text)
        step.detail = "".join(text).strip()[:1000]
        if ok(agent):
            step.checkpoint = checkpoint(worktree_root, step.title)
            step.status = "done"
        else:
            rollback(worktree_root)
            step.status = "failed"
            # Later steps depend on this one; do not build on a rollback.
            for later in plan.steps[index + 1:]:
                later.status = "skipped"
            break
    return plan


__all__ = ["MAX_STEPS", "Step", "TaskPlan", "checkpoint", "plan_steps", "rollback", "run_plan"]
