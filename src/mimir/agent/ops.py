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
"""Eight tools, and the count is a budget rather than a preference.

Adding a ninth broke tool calling outright. qwen3-coder:30b stopped emitting
tool calls and wrote them into its prose instead, at temperature zero, and
bisecting showed no single culprit: each field description was fine alone and
the three together were not. It is cumulative schema volume, so the surface has
a measured ceiling and a test that holds it there.

What went to make room: get_rollout_status and get_resource_usage, which are
specialised follow-ups rather than ways to find or read something, and
find_similar_incidents, which overlaps search_memory. The council still has all
three.

list_namespaces was never here. The run this module was written for listed two
hundred namespaces twice while looking for one the operator had already named.
find_workloads answers that question in one call.
"""

MAX_SCHEMA_CHARS = 11_500
"""A budget, and the reason for it is not the one first written here.

It was set from a single observation: one prompt at temperature zero produced a
tool call at 11,050 characters of schema and not at 12,276, and that was
recorded as a cliff. Measured properly, over fifteen distinct prompts per size,
there is no cliff. There is a slope:

    3,693 chars,  4 tools   80%
    6,721 chars,  6 tools   67%
   10,313 chars,  9 tools   53%
   13,095 chars, 12 tools   47%
   16,143 chars, 14 tools   40%
   20,220 chars, 18 tools   33%

Two things follow, and the second matters more. A smaller surface really is
better, so the budget stays. And no surface is reliable: at four tools one
prompt in five still produces no tool call, so trimming the tool list cannot
fix this and never could. The fix is constrained decoding, which makes a tool
call the only thing the model can emit, or sampling more than once and letting
the deterministic gate choose. At 53% per attempt, three attempts reach 90%.

mimir eval probe tool_adherence reproduces the table."""

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

        super().__init__(tools=tools, system=SYSTEM.format(context=_describe(environment)),
                         **kwargs)

    def hidden(self) -> tuple[str, ...]:
        """Arguments the loop supplies, kept out of the schema.

        Schema volume is the constraint, so an argument the model never needs
        to choose should not cost the tokens to describe."""
        return ("limit",)

    def note_instruction(self, instruction: str) -> None:
        """Take the stated parameters out of the sentence, by rule.

        Everything found here is bound onto the calls rather than left for the
        model to remember. A request naming a namespace, a cluster fragment, a
        workload and a line count gives the model four chances to drop one, and
        a dropped parameter fails silently: the call succeeds against the wrong
        scope and the answer reads as if it were about the right one.
        """
        # Deliberately not restated in the prompt. Listing the parsed
        # parameters back to the model as "the operator stated: ..." stopped
        # qwen3-coder emitting tool calls, the same failure the glossary hint
        # caused in the same position. The parameters do not need saying: they
        # are bound onto the calls below, which is both more reliable than
        # asking and the reason the parser exists.
        self.request = parse_request(instruction)
        self.asked = {
            k: v for k, v in (
                ("namespace", self.request.namespace),
                ("context", self.request.context),
            ) if v
        }
        self.scope = {}

    def bind(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Fill in the scope, never override one that was stated.

        A default rather than a binding, which is the opposite of the coding
        loop: there the worktree is not negotiable, here the operator may well
        be asking about a namespace other than the one the prompt is set to,
        and silently rewriting the argument would answer a question nobody
        asked while looking like it worked.

        The order matters: what the call states, then what the operator's
        session is set to, then what the operator wrote in the instruction,
        then the last value this turn used, and only then the kubeconfig.

        Both fallbacks were added after watching a run go wrong without them.
        Without the last-used value, a turn searched three names in
        messaging-squad, omitted the namespace on the next three calls, and
        silently searched perfectscale, because that is what the kubeconfig
        binds to that context. And without the operator's own words ranking
        above it, one call that named "default" made every later omission mean
        default too, so a request scoped to messaging-squad finished by
        reporting on a namespace nobody asked about.
        """
        fields = self._fields(name)
        bound = dict(arguments)

        # Operators describe a cluster as often as they name one: "a dev
        # cluster with ch1 in it". Answering that needs the list of contexts,
        # and the list is one kubeconfig read behind a flag the model has to
        # remember to set. It did not, so a request scoped to ch1 ran entirely
        # against the current context and reported on the wrong cluster
        # without ever saying which one it had read.
        if name == "get_current_context" and "include_contexts" in fields:
            bound.setdefault("include_contexts", True)

        # The stated parameters are supplied when the call omits them, and a
        # count the operator gave is not negotiable: "the last 10 logs" that
        # returns a hundred lines has answered a different question.
        if self.request.tail and "tail" in fields:
            bound["tail"] = self.request.tail
        if self.request.since and "since" in fields and not bound.get("since"):
            bound["since"] = self.request.since
        if name == "find_workloads":
            # The environment is part of the cluster constraint, not separate
            # from it. Left out, "any outbound pod in dev in a cluster with
            # ch1" returned the ch1 production clusters too.
            cluster = " ".join(
                v for v in (self.request.context_contains, self.request.environment) if v
            )
            if self.request.name_contains and not bound.get("name_contains"):
                bound["name_contains"] = self.request.name_contains
            # Set, like the namespace below. The model passed "ch1" and dropped
            # the environment, so a request that said dev searched the ch1
            # production clusters as well.
            if cluster:
                bound["context_contains"] = cluster

            # Namespace is set, not defaulted. The operator either named one or
            # did not, and the parser knows which. Left to the model, "any
            # outbound pod inside dev" became a namespace filter of "dev": no
            # namespace is called dev, so a search that would have found both
            # pods returned nothing, and the emptiness looked like an answer.
            bound["namespace_contains"] = self.request.namespace or None

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
