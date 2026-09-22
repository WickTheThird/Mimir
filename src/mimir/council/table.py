"""The specialists as a table (plan step 11).

Twelve named specialists each with a prompt were twelve places for
behaviour to drift. With the decisions moved out of them (ADR-004), a
specialist is: a name, a prompt, the capabilities it may use, the risk it
may reach, the task class it is billed to, and a budget. That is a row.

The four dicts that already held these were the table in all but name.
This module is the name, and one place to read it. ``build_council`` reads
rows; the coordinator is told which rows have produced confident answers
for the question shape before, from the experience store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mimir.council.prompts import SPECIALIST_PROMPTS
from mimir.council.specialists import (
    SPECIALIST_CAPABILITIES,
    SPECIALIST_MAX_RISK,
    SPECIALIST_TASK_CLASS,
    SpecialistBudget,
)
from mimir.models.specialist import SpecialistName
from mimir.safety.risk import RiskClass
from mimir.tools.base import Capability


@dataclass(frozen=True, slots=True)
class SpecialistRow:
    name: SpecialistName
    prompt: str
    capabilities: tuple[Capability, ...]
    max_risk: RiskClass
    task_class: str
    budget: SpecialistBudget = field(default_factory=SpecialistBudget)

    @property
    def tools_only(self) -> bool:
        """A row with no capabilities reasons from what it is given."""
        return not self.capabilities

    def render(self) -> str:
        caps = ", ".join(c.value for c in self.capabilities) or "none"
        return (f"{self.name.value:24} risk<={self.max_risk.value:3} tools={caps:40} "
                f"calls<={self.budget.max_tool_calls} rounds<={self.budget.max_iterations}")


def _row(name: SpecialistName) -> SpecialistRow:
    return SpecialistRow(
        name=name,
        prompt=SPECIALIST_PROMPTS.get(name, ""),
        capabilities=tuple(SPECIALIST_CAPABILITIES.get(name, ())),
        max_risk=SPECIALIST_MAX_RISK.get(name, RiskClass.R1),
        task_class=str(SPECIALIST_TASK_CLASS.get(name, "default")),
    )


SPECIALISTS: dict[SpecialistName, SpecialistRow] = {name: _row(name) for name in SpecialistName}


def render_table() -> str:
    return "\n".join(row.render() for row in SPECIALISTS.values())


def preferred_for(task_type: str, settings: Any) -> list[str]:
    """Rows that produced confident answers for this question shape before."""
    try:
        from mimir.knowledge.experience import get_experience_store

        return get_experience_store(settings).preferred_specialists(task_type)
    except Exception:  # noqa: BLE001 - advice, never a failure
        return []


__all__ = ["SPECIALISTS", "SpecialistRow", "preferred_for", "render_table"]
