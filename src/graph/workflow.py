"""Graph wiring: the three assistant branches.

    route
      ├── ACADEMIC    -> retrieve -> generate_answer
      ├── PLAN_CREATE -> collect -> generate -> validate
      └── PLAN_MODIFY -> collect -> modify -> validate
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    ACADEMIC,
    PLAN_CREATE,
    PLAN_MODIFY,
    collect_plan_info,
    generate_answer,
    generate_plan,
    modify_plan,
    retrieve,
    route_intent,
    validate_plan,
)
from src.graph.state import AssistantState

logger = logging.getLogger(__name__)


def _after_route(state: AssistantState) -> str:
    return state.get("intent", ACADEMIC)


@lru_cache(maxsize=1)
def build_graph():
    """Compile the workflow once per process."""
    builder = StateGraph(AssistantState)

    builder.add_node("route", route_intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("collect_info", collect_plan_info)
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("modify_plan", modify_plan)
    builder.add_node("validate_plan", validate_plan)

    builder.set_entry_point("route")

    builder.add_conditional_edges(
        "route",
        _after_route,
        {ACADEMIC: "retrieve", PLAN_CREATE: "collect_info", PLAN_MODIFY: "collect_info"},
    )

    # Academic Q&A
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", END)

    # Study plan creation
    builder.add_edge("generate_plan", "validate_plan")
    builder.add_edge("validate_plan", END)

    # Study plan modification. Routed from the shared collector so new
    # constraints merge into the stored request before the plan is rebuilt.
    builder.add_edge("collect_info", "generate_plan")
    builder.add_conditional_edges(
        "collect_info",
        _after_route,
        {PLAN_CREATE: "generate_plan", PLAN_MODIFY: "modify_plan"},
    )
    builder.add_edge("modify_plan", "validate_plan")

    return builder.compile()


_GRAPH = None


def run_assistant(
    question: str,
    history: list[dict] | None = None,
    plan=None,
    plan_request: dict | None = None,
) -> AssistantState:
    """Run one turn through the workflow and return the final state.

    `plan` and `plan_request` are carried in by the caller (the Streamlit
    session), which is what makes plan modification possible across turns.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    state: AssistantState = {
        "question": question,
        "history": history or [],
        "plan": plan,
        "plan_request": plan_request or {},
    }
    return _GRAPH.invoke(state, config={"recursion_limit": 25})