"""Graph assembly (ADR 6.2 C2).

LangGraph is used for what the ADR asks of it: durable execution, conditional
routing, parallel subgraph delegation, checkpointing, and session continuation.

Shape:

    START
      -> resolve_context
      -> recall_memory
      -> coordinate
      -> (ask_user | select_skills)
      -> dispatch        [fan out with Send, one branch per planned step]
      -> gather
      -> (verify | safety_review)
      -> safety_review
      -> synthesise
      -> curate_memory
      -> finalise
      -> END

Approvals are not modelled as graph interrupts. They arise inside a tool call,
several frames below any node boundary, so they are brokered by
:class:`~mimir.safety.approvals.ApprovalBroker` which any interface can resolve.
The graph checkpoint still makes the run durable across a restart.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from mimir.graph.nodes import (
    NodeDeps,
    ask_user,
    coordinate,
    curate_memory,
    finalise,
    gather,
    recall_memory,
    resolve_context,
    run_specialist_step,
    safety_review,
    select_skills,
    synthesise,
    verify,
)
from mimir.graph.state import GraphState
from mimir.logging import get_logger

log = get_logger(__name__)

NodeFn = Callable[[GraphState, NodeDeps], Awaitable[dict[str, Any]]]


def _bind(fn: NodeFn, deps: NodeDeps) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def node(state: GraphState) -> dict[str, Any]:
        return await fn(state, deps)

    node.__name__ = fn.__name__
    return node


def _route_after_coordinate(state: GraphState) -> str:
    return "ask_user" if state.get("route") == "ask_user" else "select_skills"


def _dispatch(state: GraphState) -> list[Send] | str:
    """Fan out one branch per planned step (ADR 7.2, 10.5 subagent execution).

    Returning a list of Send objects is what makes specialists run in parallel
    with independent contexts, rather than one long shared conversation.
    """
    steps = state.get("pending_steps") or []
    if not steps:
        return "gather"
    session = state["session"]
    round_number = state.get("round", 0)
    return [
        Send("specialist", {"session": session, "step": step, "round": round_number})
        for step in steps
    ]


def _route_after_gather(state: GraphState) -> str:
    route = state.get("route", "safety_review")
    return route if route in ("verify", "safety_review") else "safety_review"


def build_graph(deps: NodeDeps, *, parallel: bool = True) -> StateGraph:
    """Construct the investigation graph. Compile it with a checkpointer."""
    graph: StateGraph = StateGraph(GraphState)

    graph.add_node("resolve_context", _bind(resolve_context, deps))
    graph.add_node("recall_memory", _bind(recall_memory, deps))
    graph.add_node("coordinate", _bind(coordinate, deps))
    graph.add_node("ask_user", _bind(ask_user, deps))
    graph.add_node("select_skills", _bind(select_skills, deps))
    graph.add_node("specialist", _bind_step(run_specialist_step, deps))
    graph.add_node("gather", _bind(gather, deps))
    graph.add_node("verify", _bind(verify, deps))
    graph.add_node("safety_review", _bind(safety_review, deps))
    graph.add_node("synthesise", _bind(synthesise, deps))
    graph.add_node("curate_memory", _bind(curate_memory, deps))
    graph.add_node("finalise", _bind(finalise, deps))

    graph.add_edge(START, "resolve_context")
    graph.add_edge("resolve_context", "recall_memory")
    graph.add_edge("recall_memory", "coordinate")
    graph.add_conditional_edges(
        "coordinate",
        _route_after_coordinate,
        {"ask_user": "ask_user", "select_skills": "select_skills"},
    )
    graph.add_edge("ask_user", END)

    if parallel:
        graph.add_conditional_edges("select_skills", _dispatch, ["specialist", "gather"])
    else:
        # Sequential mode exists for constrained machines and for debugging, where
        # interleaved specialist output is hard to follow.
        graph.add_conditional_edges("select_skills", _dispatch_serial, ["specialist", "gather"])

    graph.add_edge("specialist", "gather")
    graph.add_conditional_edges(
        "gather",
        _route_after_gather,
        {"verify": "verify", "safety_review": "safety_review"},
    )
    graph.add_edge("verify", "safety_review")
    graph.add_edge("safety_review", "synthesise")
    graph.add_edge("synthesise", "curate_memory")
    graph.add_edge("curate_memory", "finalise")
    graph.add_edge("finalise", END)
    return graph


def _dispatch_serial(state: GraphState) -> list[Send] | str:
    steps = state.get("pending_steps") or []
    if not steps:
        return "gather"
    session = state["session"]
    return [
        Send("specialist", {"session": session, "step": steps[0], "round": state.get("round", 0)})
    ]


def _bind_step(
    fn: Callable[[Any, NodeDeps], Awaitable[dict[str, Any]]], deps: NodeDeps
) -> Callable[[Any], Awaitable[dict[str, Any]]]:
    async def node(payload: Any) -> dict[str, Any]:
        return await fn(payload, deps)

    node.__name__ = fn.__name__
    return node
