"""End-to-end tests that require a running Ollama with the configured model.

Every test here is skipped automatically when Ollama or its model is missing,
so `pytest` still passes on a clean machine.

Assertions target grounding behaviour (was it answered from the documents? does
it cite a real file?) rather than exact wording, so these remain valid when the
sample PDFs are replaced by the real college documents.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_llm

pytestmark = requires_llm


@pytest.fixture(scope="module")
def graph():
    from src.graph.workflow import build_graph

    return build_graph()


def run(graph, question, plan=None, plan_request=None, history=None):
    return graph.invoke(
        {
            "question": question,
            "history": history or [],
            "plan": plan,
            "plan_request": plan_request or {},
        },
        config={"recursion_limit": 25},
    )


# --- 6. Normal academic question ------------------------------------------
def test_academic_question_is_grounded(graph):
    result = run(graph, "What is the attendance requirement?")
    assert result.get("grounded") is True
    assert result.get("answer")
    assert result.get("sources"), "a grounded answer must carry sources"
    for source in result["sources"]:
        assert source["source"].lower().endswith(".pdf")


def test_academic_answer_cites_real_pages(graph):
    result = run(graph, "How many credits are required for the B.Tech program?")
    answer = result.get("answer", "")
    assert "Regulations" in answer, "expected the academic regulations to be cited"


def test_routing_sends_college_question_to_academic(graph):
    result = run(graph, "What is the attendance requirement?")
    assert result.get("intent") == "ACADEMIC"


# --- 7. Follow-up question -------------------------------------------------
@pytest.mark.parametrize(
    "first,followup",
    [
        ("What is the internship eligibility requirement?",
         "What documents are required for it?"),
    ],
)
def test_followup_resolves_pronoun(graph, first, followup):
    """'it' must resolve to the previous turn's subject."""
    initial = run(graph, first)
    history = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": (initial.get("answer") or "")[:300]},
    ]
    result = run(graph, followup, history=history)
    assert result.get("grounded") is True, f"follow-up not grounded: {followup!r}"
    assert result.get("sources")


def test_followup_internepship_pulls_internship_document(graph):
    initial = run(graph, "What is the internship eligibility requirement?")
    history = [
        {"role": "user", "content": "What is the internship eligibility requirement?"},
        {"role": "assistant", "content": (initial.get("answer") or "")[:300]},
    ]
    result = run(graph, "What documents are required for it?", history=history)
    sources = {s["source"] for s in result.get("sources", [])}
    assert any("Internship" in s for s in sources), (
        f"expected the internship document, got {sources}"
    )


# --- 8. Unknown question ---------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of Japan?",
        "Who is the hostel warden?",
        "What is the library fine amount?",
    ],
)
def test_unknown_question_is_refused(graph, question):
    result = run(graph, question)
    assert result.get("grounded") is False
    assert "could not find" in result.get("answer", "").lower()


def test_refusal_does_not_invent_college_facts(graph):
    result = run(graph, "What is the hostel mess fee?")
    answer = result.get("answer", "").lower()
    # A refusal must not smuggle in a number or currency amount.
    assert "could not find" in answer
    for token in ("rupees", "₹", "rs.", "per month", "lakh"):
        assert token not in answer, f"refusal appears to invent a figure: {token!r}"


# --- 9. Study plan generation ---------------------------------------------
def test_study_plan_generated(graph):
    result = run(
        graph,
        "I have Data Structures, Database Management Systems and Operating Systems. "
        "My exams are in 20 days. I can study 3 hours per day.",
    )
    plan = result.get("plan")
    assert plan is not None, "a study plan should have been created"
    assert "Data Structures" in plan.subjects
    assert plan.total_sessions > 0


# --- 10. Study plan modification ------------------------------------------
def test_study_plan_modification_keeps_previous_state(graph):
    created = run(
        graph,
        "I have Data Structures and Operating Systems. My exams are in 20 days. "
        "I can study 3 hours per day.",
    )
    plan = created["plan"]

    modified = run(
        graph,
        "I cannot study on Sunday.",
        plan=plan,
        plan_request=created["plan_request"],
    )
    updated = modified["plan"]
    assert updated is not None
    assert updated.unavailable_days == [6]
    assert updated.subjects == plan.subjects, "subjects must survive the update"
    assert updated.total_sessions < plan.total_sessions
    assert "Sunday" not in {d["weekday"] for d in updated.schedule}


# --- Grounding integrity ---------------------------------------------------
def test_no_llm_call_when_retrieval_is_empty(graph, monkeypatch):
    """With no evidence the LLM must not be called at all."""
    from src.graph import nodes

    called = {"n": 0}
    original = nodes.invoke

    def counting_invoke(llm, prompt):
        called["n"] += 1
        return original(llm, prompt)

    monkeypatch.setattr(nodes, "invoke", counting_invoke)
    result = run(graph, "What is the capital of Japan?")
    assert result.get("grounded") is False
    assert called["n"] == 0, "LLM was invoked despite no retrieved evidence"


def test_citations_are_not_invented_by_the_model(graph):
    """Every filename mentioned in an answer must be a real indexed document."""
    from src.rag.vector_store import load_index

    known = {d.metadata["source"] for d in load_index().docstore._dict.values()}
    result = run(graph, "What is the attendance requirement?")
    answer = result.get("answer", "")

    for token in answer.replace(",", " ").split():
        if token.lower().endswith(".pdf"):
            assert token in known, f"answer cites an unknown document: {token!r}"