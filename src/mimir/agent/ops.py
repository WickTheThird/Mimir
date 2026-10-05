"""The operations loop: read a cluster and answer with what it said."""

from __future__ import annotations

import re
from typing import Any

from mimir.agent.loop import AgentLoop
from mimir.agent.request import ParsedRequest, parse_request

OPS_TOOLS: tuple[str, ...] = (
    # find the thing
    "find_workloads",
    "get_current_context",
    "list_workloads",
    "summarise_pod_health",
    # read it
    "get_logs",
    "get_events",
    "describe_resource",
    # what happened last time
    "search_memory",
)
"""Eight tools, and the count is a budget rather than a preference."""

MAX_SCHEMA_CHARS = 11_500
"""A budget, and the reason for it is not the one first written here."""

SYSTEM = """\
You are MIMIR reading a Kubernetes estate on behalf of an operator. Everything
here is read-only; you cannot change anything and should not offer to.

{context}

How to work:
- The operator has usually already named the namespace, the workload or the
  cluster. Use what they named. Do not re-derive it, and do not list every
  namespace in a cluster to find one you were given.
- When they describe what they want rather than naming a namespace ("any
  messaging outbound pod in a cluster with ch1"), call find_workloads once. It
  searches every namespace, and every context whose name matches, in one go.
  Do not go looking namespace by namespace.
- When they describe a cluster instead of naming it, get_current_context
  returns every context in the kubeconfig. Never answer from the current
  context when they described a different one.
- Narrow with filters rather than by reading long lists: list_workloads takes
  name_contains, summarise_pod_health takes a pod name prefix.
- A workload search that returns nothing does not mean nothing is running.
  Pods can outlive the object that made them, or be owned by a kind you did not
  ask for. Check the pods before concluding something is absent.
- If an exact name matches nothing, try the shortest distinctive part of it
  before concluding it is absent. Operators typo names and shorten them.
- When the thing they named really is not there, say so in one line and then
  list what is there. A bare "not found" makes them ask the obvious next
  question; the list answers it.
- When the operator asked to see something, produce it. An answer that explains
  why you did not fetch the logs is not an answer to a request for logs.
- Leaving the namespace out does not search every namespace. It reuses the one
  the operator named. Do not report having searched more widely than you did.

Say what you are about to do in one short sentence, then call the tool. When
you have what was asked for, show it and stop.

Report what the cluster said. If you did not retrieve something, say that
plainly rather than describing what you would have found.
"""


# : The tools that carry out each stated action.
_ACTION_TOOLS: dict[str, frozenset[str]] = {
    "logs": frozenset({"get_logs"}),
    "events": frozenset({"get_events"}),
    "describe": frozenset({"describe_resource"}),
    "status": frozenset({"summarise_pod_health", "list_workloads"}),
    "restarts": frozenset({"summarise_pod_health"}),
}

_ASKED_NAMESPACE = re.compile(
    r"(?:-n|--namespace|\bnamespace)\s+([a-z0-9][\w.-]*)", re.IGNORECASE
)
"""The namespace the operator named, if they named one."""


class OpsAgent(AgentLoop):
    """Reads a cluster. Defaults the operator's context and namespace in."""

    label = "ops"

    def __init__(self, *, environment: Any = None, tools=OPS_TOOLS, **kwargs: Any) -> None:
        self.environment = environment
        self.scope: dict[str, str] = {}
        """Context and namespace this turn has actually used."""

        self.asked: dict[str, str] = {}
        """Scope the operator stated in the instruction itself."""

        self.request = ParsedRequest()
        """Everything the instruction stated, extracted by rule."""

        self.satisfied: set[str] = set()
        """Tools that have succeeded this turn."""

        self.located: dict[str, tuple[str, str]] = {}
        """Pod name to the context and namespace it was actually found in."""

        super().__init__(tools=tools, system=SYSTEM.format(context=_describe(environment)),
                         **kwargs)

    def specs_now(self) -> list[Any]:
        """Everything, until the operator's stated action has been carried out."""
        wanted = _ACTION_TOOLS.get(self.request.action)
        if wanted and self.satisfied & wanted:
            return []
        return self.specs

    def note_success(self, name: str, result: Any) -> None:
        """Remember where each pod was found."""
        self.satisfied.add(name)
        if name != "find_workloads":
            return
        for row in (getattr(result, "data", {}) or {}).get("matches") or []:
            pod = str(row.get("pod") or "")
            if pod:
                self.located[pod] = (str(row.get("context") or ""),
                                     str(row.get("namespace") or ""))

    def hidden(self) -> tuple[str, ...]:
        """Arguments the loop supplies, kept out of the schema."""
        return ("limit",)

    def note_instruction(self, instruction: str) -> None:
        """Take the stated parameters out of the sentence, by rule."""
        # Deliberately not restated in the prompt.
        self.request = parse_request(instruction)
        self.satisfied = set()
        self.located = {}
        self.asked = {
            k: v for k, v in (
                ("namespace", self.request.namespace),
                ("context", self.request.context),
            ) if v
        }
        self.scope = {}

    def bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Fill in the scope, never override one that was stated."""
        fields = self._fields(name)
        bound = dict(arguments)

        # Operators describe a cluster as often as they name one: "a dev
        if name == "get_current_context" and "include_contexts" in fields:
            bound.setdefault("include_contexts", True)

        # The stated parameters are supplied when the call omits them, and a
        if self.request.tail and "tail" in fields:
            bound["tail"] = self.request.tail
        if self.request.since and "since" in fields and not bound.get("since"):
            bound["since"] = self.request.since
        if name == "find_workloads":
            # The environment is part of the cluster constraint, not separate
            cluster = " ".join(
                v for v in (self.request.context_contains, self.request.environment) if v
            )
            if self.request.name_contains and not bound.get("name_contains"):
                bound["name_contains"] = self.request.name_contains
            # Set, like the namespace below.
            if cluster:
                bound["context_contains"] = cluster

            # Namespace is set, not defaulted.
            bound["namespace_contains"] = self.request.namespace or None

        # A pod this turn already located carries the cluster it was found in.
        named = str(bound.get("target") or bound.get("name") or "")
        found = self.located.get(named)
        if found and found[0]:
            # Set, not filled.
            if "context" in fields:
                bound["context"] = found[0]
            if "namespace" in fields and found[1]:
                bound["namespace"] = found[1]

        for field, attribute in (("context", "cluster_context"), ("namespace", "namespace")):
            if field not in fields:
                continue
            if bound.get(field):
                self.scope[field] = str(bound[field])
                continue
            stated = getattr(self.environment, attribute, None) if self.environment else None
            fallback = stated or self.asked.get(field) or self.scope.get(field)
            if fallback:
                bound[field] = str(fallback)
        return bound


def _describe(environment: Any) -> str:
    if environment is None:
        return "No cluster context is set; the operator must name one."
    bits = []
    if getattr(environment, "cluster_context", None):
        bits.append(f"context {environment.cluster_context}")
    if getattr(environment, "namespace", None):
        bits.append(f"namespace {environment.namespace}")
    if not bits:
        return "No cluster context is set; use the one the operator names."
    return (
        f"The operator's session is set to {' and '.join(bits)}. "
        "Calls default to it unless they name another."
    )


__all__ = ["OPS_TOOLS", "OpsAgent"]
