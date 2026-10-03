"""LangGraph nodes: routing, retrieval, answering, planning and plan updates.

Each node takes the shared state and returns only the keys it changes. Nodes are
plain functions, which keeps them unit-testable without running the graph.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from src.graph.planner import (
    StudyPlan,
    StudyPlanError,
    create_plan,
    format_plan,
    parse_unavailable_days,
)
from src.graph.planner_parser import (
    describe_days,
    missing_plan_fields,
    parse_hours_per_day,
    parse_plan_request,
)
from src.graph.state import AssistantState
from src.llm.model import get_llm, invoke
from src.llm.prompts import (
    GROUNDED_PROMPT,
    NO_CONTEXT_ANSWER,
    PLAN_PROMPT,
    PLAN_UPDATE_PROMPT,
    REWRITE_PROMPT,
)
from src.rag.retriever import CollegeRetriever, get_retriever
from src.tools.calculator import calculate
from src.utils.helpers import clean_llm_text, history_to_text, recent_user_turns, trim_history

logger = logging.getLogger(__name__)

ACADEMIC = "ACADEMIC"
PLAN_CREATE = "PLAN_CREATE"
PLAN_MODIFY = "PLAN_MODIFY"

# Cheap pre-routing so obvious commands skip an LLM round-trip.
_PLAN_HINTS = re.compile(
    r"\b(?:study plan|study schedule|revision plan|revision schedule|"
    r"(?:make|create|build|give me|generate)\s+(?:me\s+)?a\s+plan|"
    r"prepare for (?:my )?exams?|plan my studies|schedule my studies)\b",
    re.I,
)
_MODIFY_HINTS = re.compile(
    r"\b(?:cannot|can't|can not|unable to|no longer|won't|will not)\s+"
    r"(?:study|revise|attend)\b"
    r"|^\s*(?:remove|drop|add|change|increase|decrease|reduce|make|shift)\b"
    r"|\b(?:move|shift|reschedule)\b",
    re.I,
)
# Phrases that mean "this is a college question", which must win over the
# planning heuristic even when the message happens to contain numbers.
_ACADEMIC_QUESTION_HINTS = re.compile(
    r"\b(?:attendance|marks?|percentage|pass(?:ing)?|credit|exam (?:paper|pattern|schedule)|"
    r"hall ticket|deadline|eligibility|regulation|semester duration|curriculum|"
    r"what (?:is|are) the (?:rules?|criteria|policies))\b",
    re.I,
)


# --- Routing ---------------------------------------------------------------
def route_intent(state: AssistantState) -> dict:
    """Decide between academic Q&A, plan creation, and plan modification.

    Routing is fully rule-based. A 1.5b model was tested as the classifier and
    proved unreliable in both directions: it sent "what is the attendance
    requirement" to the study planner, and after the prompts were tightened it
    started sending plain academic questions there too. The rules below are
    predictable, instant, and cheap, so the LLM is not consulted at all.
    """
    question = state.get("question", "")
    has_plan = state.get("plan") is not None

    if has_plan and _MODIFY_HINTS.search(question):
        return {"intent": PLAN_MODIFY}

    if _PLAN_HINTS.search(question):
        return {"intent": PLAN_MODIFY if has_plan else PLAN_CREATE}

    # A message carrying at least two scheduling constraints is a plan request.
    parsed = parse_plan_request(question)
    constraint_hits = sum(
        bool(parsed.get(key))
        for key in ("subjects", "days_until_exam", "hours_per_day")
    )
    academic_signal = bool(_ACADEMIC_QUESTION_HINTS.search(question))

    if constraint_hits >= 2 and not academic_signal:
        return {"intent": PLAN_MODIFY if has_plan else PLAN_CREATE}

    # Any remaining planning signal, such as an explicit "create a plan", with no
    # competing academic intent.
    if _PLAN_HINTS.search(question) and not academic_signal:
        return {"intent": PLAN_MODIFY if has_plan else PLAN_CREATE}

    # Default: everything else is a college knowledge-base question.
    return {"intent": ACADEMIC}


# --- Retrieval -------------------------------------------------------------
# A question carrying one of these is already answerable on its own; rewriting
# it would only add latency.
_DANGLING_REFERENCE = re.compile(
    r"\b(?:it|its|that|this|they|them|those|these|the same|also)\b",
    re.I,
)


@lru_cache(maxsize=64)
def _rewrite_query_cached(history_tuple: tuple[tuple[str, str], ...], question: str) -> str:
    """Cached version of query rewrite - takes hashable history tuple."""
    history = [{"role": r, "content": c} for r, c in history_tuple]

    stripped = question.strip()
    has_reference = bool(_DANGLING_REFERENCE.search(stripped))
    is_elliptical = len(stripped.split()) <= 4
    if not (has_reference or is_elliptical):
        return question

    try:
        messages = REWRITE_PROMPT.format_messages(
            history=history_to_text(history), question=question
        )
        rewritten = clean_llm_text(invoke(get_llm(), messages))
        return rewritten if len(rewritten) > 5 else question
    except Exception as exc:
        logger.warning("Query rewrite failed, using raw question: %s", exc)
        return question


def _rewrite_query(state: AssistantState) -> str:
    """Expand a follow-up into a standalone question when history exists.

    Rewriting costs a full LLM round-trip, so it is skipped unless the question
    actually contains a pronoun or short elliptical phrase that would otherwise
    retrieve poorly. Most follow-ups ("What documents are required for it?")
    need it; standalone questions do not.
    """
    question = state.get("question", "")
    history = trim_history(state.get("history", []))
    if not history:
        return question

    # Convert history to hashable tuple for caching
    history_tuple = tuple((m.get("role", "user"), str(m.get("content", "")).strip())
                          for m in history if m.get("content"))
    return _rewrite_query_cached(history_tuple, question)


def retrieve(state: AssistantState) -> dict:
    """Fetch relevant college chunks for the question."""
    query = _rewrite_query(state)
    try:
        retriever: CollegeRetriever = get_retriever()
        hits = retriever.retrieve(query)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return {
            "error": (
                "I could not search the college documents right now. "
                f"Details: {exc}"
            ),
            "retrieved_docs": [],
            "context": "",
            "sources": [],
        }

    return {
        "retrieved_docs": hits,
        "context": retriever.format_context(hits),
        "sources": retriever.format_sources(hits),
        "grounded": bool(hits),
    }


# --- Answering -------------------------------------------------------------
def generate_answer(state: AssistantState) -> dict:
    """Answer strictly from the retrieved context, citing real page metadata."""
    if state.get("error"):
        return {}

    hits = state.get("retrieved_docs", [])
    if not hits:
        # Refuse without calling the LLM: with no evidence there is nothing to
        # ground an answer in, and a model asked to answer anyway will improvise.
        return {"answer": NO_CONTEXT_ANSWER, "grounded": False, "sources": []}

    history = trim_history(state.get("history", []))
    try:
        messages = GROUNDED_PROMPT.format_messages(
            context=state.get("context", ""),
            history=history_to_text(history) or "(none)",
            question=state.get("question", ""),
        )
        answer = clean_llm_text(invoke(get_llm(), messages))
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        return {"error": f"The assistant could not generate an answer: {exc}"}

    return {
        "answer": _append_citations(answer, state.get("sources", [])),
        "grounded": True,
    }


def _append_citations(answer: str, sources: list[dict]) -> str:
    """Attach the retrieved files/pages to the answer.

    Citations are added from the retrieval metadata rather than trusted to the
    model, which frequently drops them at this size.
    """
    if not sources:
        return answer
    seen: list[str] = []
    for source in sources:
        page = source.get("page")
        label = (
            f"{source['source']}, page {page}"
            if page not in (None, 0, "")
            else f"{source['source']}"
        )
        if label not in seen:
            seen.append(label)
    listing = "; ".join(seen)
    return f"{answer}\n\n**Sources:** {listing}"


def calculator_node(state: AssistantState) -> dict:
    """Answer a pure arithmetic request without involving the knowledge base.

    Only fires on an unambiguous expression, so "how many hours is 3 + 3" does
    not get hijacked mid-sentence.
    """
    question = state.get("question", "")
    match = re.fullmatch(r"[^a-zA-Z]*([0-9][0-9\s+\-*/^().%]*)[^a-zA-Z]*", question.strip())
    if not match:
        return {}
    try:
        value = calculate(match.group(1))
        shown = round(value, 4)
        return {"answer": f"{match.group(1).strip()} = {shown}", "grounded": False}
    except Exception:
        return {}


# --- Study planning --------------------------------------------------------
def collect_plan_info(state: AssistantState) -> dict:
    """Merge inputs parsed from this message with anything already known."""
    parsed = parse_plan_request(state.get("question", ""))
    existing = dict(state.get("plan_request") or {})

    merged = {**parsed, **existing}
    # Explicit new constraints always win over the previously stored ones.
    for key in ("subjects", "days_until_exam", "hours_per_day", "session_minutes"):
        if parsed.get(key):
            merged[key] = parsed[key]
    if parsed.get("unavailable_days"):
        merged["unavailable_days"] = sorted(
            set(existing.get("unavailable_days", [])) | set(parsed["unavailable_days"])
        )
    return {"plan_request": merged, "missing_fields": missing_plan_fields(merged)}


def generate_plan(state: AssistantState) -> dict:
    """Build the day-by-day schedule deterministically, then summarise it."""
    if state.get("error"):
        return {}
    request = state.get("plan_request", {})

    missing = missing_plan_fields(request)
    if missing:
        return {
            "missing_fields": missing,
            "answer": (
                "I can build that study plan, but I still need: "
                + ", ".join(missing)
                + "."
            ),
        }

    try:
        plan = create_plan(
            subjects=request["subjects"],
            days_until_exam=request["days_until_exam"],
            hours_per_day=request["hours_per_day"],
            unavailable_days=request.get("unavailable_days", []),
            session_minutes=request.get("session_minutes", 60),
        )
    except StudyPlanError as exc:
        return {"answer": f"I could not build the plan: {exc}", "error": str(exc)}

    summary = ""
    try:
        messages = PLAN_PROMPT.format_messages(plan=format_plan(plan))
        summary = clean_llm_text(invoke(get_llm(), messages))
        if not _is_usable_summary(summary):
            summary = ""
    except Exception as exc:
        logger.warning("Plan summary skipped: %s", exc)

    return {
        "plan": plan,
        "missing_fields": [],
        "answer": summary or (
            f"Study plan created for {', '.join(plan.subjects)}: "
            f"{plan.total_sessions} sessions over {len(plan.schedule)} study days "
            f"({plan.total_hours} hours total)."
        ),
    }


def modify_plan(state: AssistantState) -> dict:
    """Apply a change request to the existing plan and recompute it.

    The previous plan's inputs are reused, so only the changed values need to be
    restated by the user.
    """
    plan: StudyPlan | None = state.get("plan")
    request = state.get("plan_request", {})

    if plan is None:
        return {"answer": "You do not have a study plan yet. Tell me your subjects, exam dates and available hours first."}

    message = state.get("question", "")
    new_unavailable = parse_unavailable_days(message)
    unavailable = sorted(set(plan.unavailable_days) | set(new_unavailable))

    # "I can now study 4 hours" and "I can no longer study on X" both update hours.
    hours = parse_hours_per_day(message) or plan.hours_per_day
    days = plan.days_until_exam
    subjects = plan.subjects

    try:
        updated = create_plan(
            subjects=subjects,
            days_until_exam=days,
            hours_per_day=hours,
            unavailable_days=unavailable,
            session_minutes=plan.session_minutes,
        )
    except StudyPlanError as exc:
        return {"answer": f"I could not update the plan: {exc}"}

    changed_days = describe_days(sorted(set(new_unavailable) - set(plan.unavailable_days)))
    explanation = ""
    if changed_days and changed_days != "none":
        explanation = f"I have removed {changed_days} from your available study days."
    elif hours != plan.hours_per_day:
        explanation = f"I have updated your daily study time to {hours} hours."
    else:
        explanation = "I have regenerated your plan with your updated constraints."

    try:
        messages = PLAN_UPDATE_PROMPT.format_messages(
            plan=format_plan(updated), request=message
        )
        detail = clean_llm_text(invoke(get_llm(), messages))
        if detail:
            explanation = f"{explanation}\n\n{detail}"
    except Exception as exc:
        logger.warning("Plan update summary skipped: %s", exc)

    return {
        "plan": updated,
        "plan_request": {
            "subjects": subjects,
            "days_until_exam": days,
            "hours_per_day": hours,
            "session_minutes": plan.session_minutes,
            "unavailable_days": unavailable,
        },
        "answer": explanation,
    }


def validate_plan(state: AssistantState) -> dict:
    """Final check that the plan we are about to show is actually usable."""
    plan: StudyPlan | None = state.get("plan")
    if plan is None:
        return {}
    problems = []
    if not plan.subjects:
        problems.append("no subjects")
    if plan.total_sessions == 0:
        problems.append("no study sessions were scheduled")
    if plan.hours_per_day <= 0:
        problems.append("daily study hours is zero")
    blocked = set(plan.unavailable_days)
    for day in plan.schedule:
        if any(blocked) and _weekday_index(day["weekday"]) in blocked:
            problems.append(f"a session was scheduled on unavailable {day['weekday']}")
            break
    if problems:
        logger.error("Plan validation failed: %s", problems)
        return {
            "plan": None,
            "error": "I generated an inconsistent plan and discarded it. Please restate your constraints.",
        }
    return {}


def _weekday_index(name: str) -> int:
    from src.graph.planner import WEEKDAY_NAMES

    try:
        return WEEKDAY_NAMES.index(name)
    except ValueError:
        return -1


# A small model sometimes answers a summarisation request with a question or a
# refusal. Those must not reach the user in place of the plan.
_SUMMARY_DEFECTS = re.compile(
    r"\b(?:not sure|unsure|please provide|provide more information|"
    r"clarify|what (?:is|are) your|which plan|could you (?:please )?(?:provide|clarify))\b",
    re.I,
)


def _is_usable_summary(text: str) -> bool:
    return bool(text) and len(text) > 30 and not _SUMMARY_DEFECTS.search(text)