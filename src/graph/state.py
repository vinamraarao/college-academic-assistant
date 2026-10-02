"""Graph state. Kept small and explicit so each node's inputs/outputs are obvious."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document

from src.graph.planner import StudyPlan


def _keep_last(_current, new) -> object:
    """Reducer that replaces the value, used where LangGraph would otherwise
    require an explicit operator."""
    return new


class AssistantState(TypedDict, total=False):
    """State carried through the assistant workflow.

    `history` is the only field that grows; it is trimmed before it reaches a
    prompt so it cannot crowd out the retrieved context.
    """

    # --- Input ---
    question: str
    history: Annotated[list[dict], _keep_last]

    # --- Routing ---
    intent: Annotated[str, _keep_last]

    # --- Retrieval ---
    retrieved_docs: Annotated[list[tuple[Document, float]], _keep_last]
    context: Annotated[str, _keep_last]
    sources: Annotated[list[dict], _keep_last]

    # --- Answer ---
    answer: Annotated[str, _keep_last]
    grounded: Annotated[bool, _keep_last]  # False when nothing relevant was found
    calculator_note: Annotated[str, _keep_last]

    # --- Study plan ---
    plan_request: Annotated[dict, _keep_last]
    plan: Annotated[StudyPlan | None, _keep_last]
    missing_fields: Annotated[list[str], _keep_last]

    # --- Errors ---
    error: Annotated[str, _keep_last]