"""Regression tests for the four reported defects.

Each test names the behaviour that was wrong and asserts the corrected
behaviour, so a future change that reintroduces the bug fails here.

The bugs, in the order they were reported:
  1. "make a study table for 4th semester" was routed to normal RAG and came
     back as "I could not find this information".
  2. The CSE syllabus answer listed the wrong subjects under the wrong
     semesters, even though the PDF contains all of them.
  3. An internship document was cited as a source for a syllabus question.
  4. Existing behaviour must keep working (covered by the rest of the suite).
"""

from __future__ import annotations

import pytest

from src.graph.nodes import ACADEMIC, PLAN_CREATE, route_intent
from src.graph.planner_parser import parse_plan_request, parse_subjects
from src.rag.syllabus import (
    format_semester_table,
    parse_semester_subjects,
    referenced_semesters,
    resolve_single_semester,
)

# Phrasings that must all reach the study planner. "study table" and
# "timetable" are the ones that used to fall through to RAG.
STUDY_TABLE_PHRASINGS = [
    "Make a study table for 4th semester subjects.",
    "Can you make study table for 4th semester subject?",
    "Create a study plan for 4th semester.",
    "Give me a timetable for my 4th semester subjects.",
    "I have 15 days to study my semester subjects.",
    "Prepare a study schedule for my subjects.",
    "prepare a study plan for my subjects",
    "make me a study table",
]

# Questions that must keep going to the knowledge base.
ACADEMIC_PHRASINGS = [
    "What is the attendance requirement?",
    "What is the capital of Japan?",
    "How many credits are required for the B.Tech program?",
    "What are the internship requirements?",
    "What is the exam malpractice policy?",
    "What happens if I fail a subject?",
]


# --- 1. Study-planning requests reach the planner ---------------------------
@pytest.mark.parametrize("message", STUDY_TABLE_PHRASINGS)
def test_study_planning_phrasing_is_not_routed_to_rag(message):
    """A study request must never be answered by the RAG refusal."""
    assert route_intent({"question": message, "plan": None})["intent"] == PLAN_CREATE


@pytest.mark.parametrize("message", ACADEMIC_PHRASINGS)
def test_academic_questions_still_route_to_rag(message):
    for question in (message,):
        assert route_intent({"question": question, "plan": None})["intent"] == ACADEMIC


@pytest.mark.parametrize(
    "message",
    [
        "Can you make study table for 4th semester subject?",
        "Make a study table for 4th semester subjects.",
        "I have 15 days to study my semester subjects.",
        "create a study plan for my semester subjects",
        "I have 10 days to prepare for my subjects.",
        "What is the attendance requirement?",
    ],
)
def test_prompt_wording_is_never_taken_as_a_subject(message):
    """'4th semester', 'my subjects' and '?' were accepted as subject names."""
    assert parse_subjects(message) == []


def test_real_subject_lists_are_still_extracted():
    """The stricter rule must not throw away genuine subject names."""
    assert parse_plan_request(
        "I have DBMS, OS and Computer Networks. My exams are in 20 days. "
        "I can study 3 hours per day."
    )["subjects"] == ["DBMS", "OS", "Computer Networks"]


@pytest.mark.parametrize(
    "message,hours",
    [
        ("Make a study table. I have 10 days and 3 hours a day.", 3.0),
        ("Give me a timetable. 15 days, 2 hours daily.", 2.0),
        ("I can study 3 hours per day.", 3.0),
    ],
)
def test_hours_per_day_is_read_from_daily_phrasing(message, hours):
    assert parse_plan_request(message)["hours_per_day"] == hours


# --- 2. Syllabus parsing ----------------------------------------------------
@pytest.fixture(scope="module")
def syllabus_text():
    """Text of the real CSE syllabus PDF, skipped when data/ is absent."""
    pytest.importorskip("pypdf")
    from pathlib import Path

    from src.config import DATA_DIR
    from src.rag.loader import load_pdfs

    path = Path(DATA_DIR) / "04_Syllabus_CSE_NMAMIT.pdf"
    if not path.exists():
        pytest.skip(f"syllabus PDF not present at {path}")
    pages, _ = load_pdfs(DATA_DIR)
    text = "\n".join(p["page_content"] for p in pages if "Syllabus" in p["source"])
    if not text:
        pytest.skip("syllabus text not extractable")
    return text


def test_syllabus_parser_finds_every_semester(syllabus_text):
    semesters = parse_semester_subjects(syllabus_text)
    assert set(semesters) == set(range(1, 9)), (
        f"expected semesters I-VIII, got {sorted(semesters)}"
    )


def test_semester_four_subjects_match_the_document(syllabus_text):
    """The reported answer was wrong about Semester IV."""
    semesters = parse_semester_subjects(syllabus_text)
    assert "Database Management Systems" in semesters[4]
    assert "Design & Analysis of Algorithms" in semesters[4]
    assert "Internship-I" in semesters[4]


def test_semester_eight_subjects_match_the_document(syllabus_text):
    semesters = parse_semester_subjects(syllabus_text)
    assert semesters[8] == ["Internship-II", "Major Project Phase-II"]


def test_syllabus_subjects_are_never_invented(syllabus_text):
    """Every parsed name must appear verbatim in the document."""
    semesters = parse_semester_subjects(syllabus_text)
    for subjects in semesters.values():
        for subject in subjects:
            assert subject in syllabus_text


def test_unknown_semester_resolves_to_nothing(syllabus_text):
    semesters = parse_semester_subjects(syllabus_text)
    assert resolve_single_semester("plan for my 9th semester", semesters) == []


def test_single_semester_resolution_is_used_by_the_planner(syllabus_text):
    """Two semesters at once exceed a normal plan's session count."""
    semesters = parse_semester_subjects(syllabus_text)
    assert resolve_single_semester("study my 4th semester", semesters) == semesters[4]
    assert resolve_single_semester("study 3rd and 4th semester", semesters) == []


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Make a study table for 4th semester subjects. 10 days, 3 hours a day.", [4]),
        ("Give me a timetable for my 4th semester subjects.", [4]),
        ("semester 3 and 4", [3, 4]),
        ("Semester 1, 2 and 3", [1, 2, 3]),
        ("plan for my subjects", []),
    ],
)
def test_referenced_semesters(message, expected):
    assert referenced_semesters(message) == expected


def test_format_semester_table_keeps_source_wording(syllabus_text):
    semesters = parse_semester_subjects(syllabus_text)
    table = format_semester_table(semesters)
    assert "Semester 4:" in table
    assert "Database Management Systems" in table


def test_calendar_document_is_not_mistaken_for_a_syllabus():
    """Its 'I Semester'/'II Semester' are column headers, not subject rows."""
    from src.rag.loader import load_pdfs

    pages, _ = load_pdfs()
    for page in pages:
        if "Calendar" not in page["source"]:
            continue
        assert not parse_semester_subjects(page["page_content"]), (
            f"calendar page {page['page']} parsed as a subject list"
        )