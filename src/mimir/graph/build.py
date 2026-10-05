"""Graph assembly (ADR 6.2 C2)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from mimir.graph.nodes import (
    NodeDeps,
    ask_user,
    assess,
    coordinate,
    curate_memory,
    finalise,
    gather,
    recall_memory,
    replan,
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
    route = state.get("route")
    if route in ("done", "ask_user"):
        return route
    return "select_skills"


def _dispatch(state: GraphState) -> list[Send] | str:
    """Fan out one branch per planned step (ADR 7.2, 10.5 subagent execution)."""
    steps = state.get("pending_steps") or []
    if not steps:
        return "gather"
    session = state["session"]
    round_number = state.get("round", 0)
    return [
        Send("specialist", {"session": session, "step": step, "round": round_number})
        for step in steps
    ]


def _route_after_assess(state: GraphState) -> str:
    route = state.get("route", "safety_review")
    if route in ("replan", "replan_ask"):
        return "replan"
    return route if route in ("verify", "safety_review") else "safety_review"


def _route_after_replan(state: GraphState) -> str:
    route = state.get("route", "safety_review")
    return route if route in ("select_skills", "ask_user", "safety_review") else "safety_review"


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
    graph.add_node("assess", _bind(assess, deps))
    graph.add_node("replan", _bind(replan, deps))
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
        # "done" is the triage exit: conversational input answered directly,
        {"ask_user": "ask_user", "select_skills": "select_skills", "done": "finalise"},
    )
    graph.add_edge("ask_user", END)

    if parallel:
        graph.add_conditional_edges("select_skills", _dispatch, ["specialist", "gather"])
    else:
        # Sequential mode exists for constrained machines and for debugging, where
        graph.add_conditional_edges("select_skills", _dispatch_serial, ["specialist", "gather"])

    graph.add_edge("specialist", "gather")
    # The recurrent edge (ADR-003 phase 3, ADR-004 step 2).
    graph.add_edge("gather", "assess")
    graph.add_conditional_edges(
        "assess",
        _route_after_assess,
        {"verify": "verify", "safety_review": "safety_review", "replan": "replan"},
    )
    graph.add_conditional_edges(
        "replan",
        _route_after_replan,
        {"select_skills": "select_skills", "ask_user": "ask_user",
         "safety_review": "safety_review"},
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
