"""The operations loop: read a cluster and answer with what it said.

This exists because of a run that failed in a specific and instructive way.
Asked for the last ten log lines from a pod in a named namespace in a cluster
whose context contained "ch1", MIMIR planned a kubernetes investigation, ran
two specialists for twelve tool calls, listed two hundred namespaces twice,
retrieved thirty five pods, filtered none of them by name, never called
get_logs, and then synthesised a confident conclusion that no matching pod
existed.

Every tool it needed was in its hand. list_workloads takes name_contains and
it passed none; get_logs takes a pod name and it was never called. The council
is not the wrong council here, it is the wrong machine: nothing in the request
needed deliberation. The namespace was named, the cluster filter was named,
the action was named, and the number of lines was named. That is an
instruction, and an instruction wants a short loop that carries it out and
shows what came back.

So this is the same loop as the coding mode with a different surface, and the
same reason for existing: the graph is for questions whose answer has to be
argued for. A directive whose target is already named is not one of those.
"""

from __future__ import annotations

from typing import Any

from mimir.agent.loop import AgentLoop

OPS_TOOLS: tuple[str, ...] = (
    # find the thing
    "get_current_context",
    "list_workloads",
    "summarise_pod_health",
    # read it
    "get_logs",
    "get_events",
    "describe_resource",
    "get_rollout_status",
    "get_resource_usage",
    # what happened last time
    "search_memory",
    "find_similar_incidents",
)
"""Ten tools. Finding, reading, and what was learned before.

list_namespaces is deliberately absent. The run this module was written for
listed two hundred namespaces twice while looking for one the operator had
already named, which cost most of the wall clock and contributed nothing. When
a namespace really is unknown, list_workloads across contexts answers the same
question against a hundredth of the output.
"""

SYSTEM = """\
You are MIMIR reading a Kubernetes estate on behalf of an operator. Everything
here is read-only; you cannot change anything and should not offer to.

{context}

How to work:
- The operator has usually already named the namespace, the workload or the
  cluster. Use what they named. Do not re-derive it, and do not list every
  namespace in a cluster to find one you were given.
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

Say what you are about to do in one short sentence, then call the tool. When
you have what was asked for, show it and stop.

Report what the cluster said. If you did not retrieve something, say that
plainly rather than describing what you would have found.
"""


class OpsAgent(AgentLoop):
    """Reads a cluster. Defaults the operator's context and namespace in."""

    label = "ops"

    def __init__(self, *, environment: Any = None, tools=OPS_TOOLS, **kwargs: Any) -> None:
        self.environment = environment
        self.scope: dict[str, str] = {}
        """Context and namespace this turn has actually used."""

        super().__init__(tools=tools, system=SYSTEM.format(context=_describe(environment)),
                         **kwargs)

    def bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Fill in the scope, never override one that was stated.

        A default rather than a binding, which is the opposite of the coding
        loop: there the worktree is not negotiable, here the operator may well
        be asking about a namespace other than the one the prompt is set to,
        and silently rewriting the argument would answer a question nobody
        asked while looking like it worked.

        The order matters. When neither the call nor the operator names a
        namespace, the value used is the last one this turn used, not the one
        the kubeconfig binds to the context. A real run searched three names in
        messaging-squad, omitted the namespace on the next three calls, and
        silently searched perfectscale instead, because that is what the
        kubeconfig binds to that context. Three empty results in a row, from a
        namespace nobody had mentioned.
        """
        fields = self._fields(name)
        bound = dict(arguments)
        for field, attribute in (("context", "cluster_context"), ("namespace", "namespace")):
            if field not in fields:
                continue
            if bound.get(field):
                self.scope[field] = str(bound[field])
                continue
            stated = getattr(self.environment, attribute, None) if self.environment else None
            fallback = stated or self.scope.get(field)
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
