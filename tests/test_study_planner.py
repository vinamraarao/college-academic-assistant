"""Study planner, calculator, workflow, and Streamlit startup tests."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.graph.planner import (
    WEEKDAY_NAMES,
    StudyPlanError,
    create_plan,
    format_plan,
    parse_unavailable_days,
)
from src.graph.planner_parser import parse_plan_request
from src.graph.nodes import ACADEMIC, PLAN_CREATE, PLAN_MODIFY, route_intent
from src.tools.calculator import (
    CalculatorError,
    calculate,
    percentage,
    study_hour_breakdown,
)


# --- Study plan generation -------------------------------------------------
def test_create_plan_schedules_all_subjects():
    plan = create_plan(["DBMS", "OS", "CN"], days_until_exam=20, hours_per_day=3.0)
    assert plan.total_sessions > 0
    assert set(plan.subject_hours) == {"DBMS", "OS", "CN"}
    assert plan.total_hours > 0


def test_plan_respects_hours_per_day():
    plan = create_plan(["DBMS"], days_until_exam=5, hours_per_day=2.0, session_minutes=60)
    for day in plan.schedule:
        assert day["hours"] <= 2.0


def test_plan_excludes_unavailable_days():
    plan = create_plan(
        ["Maths"], days_until_exam=14, hours_per_day=1.0, unavailable_days=[6]  # Sunday
    )
    weekdays = {day["weekday"] for day in plan.schedule}
    assert "Sunday" not in weekdays
    assert plan.unavailable_days == [6]


def test_excluding_a_day_reduces_sessions():
    baseline = create_plan(["Maths", "OS"], days_until_exam=14, hours_per_day=2.0)
    blocked = create_plan(
        ["Maths", "OS"], days_until_exam=14, hours_per_day=2.0, unavailable_days=[6]
    )
    assert blocked.total_sessions < baseline.total_sessions
    assert blocked.total_hours < baseline.total_hours


def test_every_session_has_a_subject_and_duration():
    plan = create_plan(["A", "B", "C"], days_until_exam=10, hours_per_day=3.0)
    for day in plan.schedule:
        assert day["sessions"]
        for session in day["sessions"]:
            assert session["subject"] in plan.subjects
            assert session["minutes"] > 0


def test_plan_days_are_consecutive_and_within_window():
    plan = create_plan(["A"], days_until_exam=7, hours_per_day=1.0)
    dates = [date.fromisoformat(d["date"]) for d in plan.schedule]
    assert dates == sorted(dates)
    assert dates[-1] - dates[0] <= timedelta(days=6)


def test_plan_covers_roughly_equal_time_per_subject():
    plan = create_plan(["A", "B", "C"], days_until_exam=15, hours_per_day=3.0)
    values = list(plan.subject_hours.values())
    assert max(values) - min(values) <= 2.0, "subject time should be balanced"


# --- Invalid plan input ----------------------------------------------------
def test_no_subjects_raises():
    with pytest.raises(StudyPlanError, match="at least one subject"):
        create_plan([], days_until_exam=10, hours_per_day=2.0)


def test_zero_days_raises():
    with pytest.raises(StudyPlanError, match="at least 1"):
        create_plan(["A"], days_until_exam=0, hours_per_day=2.0)


def test_zero_hours_raises():
    with pytest.raises(StudyPlanError, match="greater than 0"):
        create_plan(["A"], days_until_exam=10, hours_per_day=0)


def test_all_days_unavailable_raises():
    with pytest.raises(StudyPlanError, match="unavailable"):
        create_plan(["A"], days_until_exam=5, hours_per_day=1.0, unavailable_days=list(range(7)))


def test_too_few_slots_raises_clear_error():
    """Six subjects but only one study slot: the error must say so plainly."""
    with pytest.raises(StudyPlanError, match="Not enough study slots"):
        create_plan(
            ["A", "B", "C", "D", "E", "F"],
            days_until_exam=1,
            hours_per_day=1.0 / 6,  # rounds to a single short session
        )


def test_format_plan_mentions_subjects():
    plan = create_plan(["DBMS", "OS"], days_until_exam=5, hours_per_day=1.0)
    text = format_plan(plan)
    assert "DBMS" in text and "OS" in text


# --- Free-text parsing -----------------------------------------------------
def test_parse_requirement_example():
    result = parse_plan_request(
        "I have DBMS, OS and Computer Networks. My exams are in 20 days. "
        "I can study 3 hours per day."
    )
    assert result["subjects"] == ["DBMS", "OS", "Computer Networks"]
    assert result["days_until_exam"] == 20
    assert result["hours_per_day"] == 3.0


def test_parse_unavailable_sunday():
    assert parse_unavailable_days("I cannot study on Sunday.") == [6]
    assert 5 in parse_unavailable_days("Remove Saturday from my plan")


def test_parse_subjects_empty_for_modification():
    assert parse_plan_request("I cannot study on Sunday.")["subjects"] == []


# --- Routing ---------------------------------------------------------------
@pytest.mark.parametrize(
    "message,expected",
    [
        ("What is the attendance requirement?", ACADEMIC),
        ("What is the capital of Japan?", ACADEMIC),
        ("How many marks do I need to pass?", ACADEMIC),
        ("What happens if I fail a subject?", ACADEMIC),
        ("Make me a study plan", PLAN_CREATE),
        (
            "I have DBMS and OS. Exams in 20 days. I can study 3 hours per day.",
            PLAN_CREATE,
        ),
    ],
)
def test_routing(message, expected):
    assert route_intent({"question": message, "plan": None})["intent"] == expected


def test_modification_routed_when_plan_exists():
    state = {
        "question": "I cannot study on Sunday.",
        "plan": create_plan(["A"], days_until_exam=10, hours_per_day=1.0),
    }
    assert route_intent(state)["intent"] == PLAN_MODIFY


# --- Calculator tool -------------------------------------------------------
def test_calculator_arithmetic():
    assert calculate("2 + 3 * 4") == 14
    assert calculate("(10 + 5) / 3") == pytest.approx(5.0)
    assert calculate("2 ^ 10") == 1024


def test_calculator_percentage_operator():
    assert calculate("50%") == 0.5


def test_calculator_rejects_code_injection():
    for payload in ("__import__('os').system('ls')", "1; import os", "open('x')"):
        with pytest.raises(CalculatorError):
            calculate(payload)


def test_calculator_rejects_empty():
    with pytest.raises(CalculatorError):
        calculate("")


def test_calculator_division_by_zero():
    with pytest.raises(CalculatorError, match="zero"):
        calculate("5 / 0")


def test_percentage_helper():
    assert percentage(45, 60) == 75.0
    with pytest.raises(CalculatorError):
        percentage(1, 0)


def test_study_hour_breakdown():
    result = study_hour_breakdown(["A", "B"], days=10, hours_per_day=2.0)
    assert result["total_hours"] == 20.0
    assert result["hours_per_subject"] == 10.0


def test_weekday_names_are_complete():
    assert len(WEEKDAY_NAMES) == 7
    assert WEEKDAY_NAMES[6] == "Sunday"


# --- Streamlit startup -----------------------------------------------------
def test_streamlit_app_imports():
    """app.py must import cleanly under Streamlit's bare-mode warnings."""
    import importlib.util
    import warnings
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "app.py"
    assert path.exists(), "app.py is missing"

    # Streamlit warns about bare-mode execution; ignore just that.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        spec = importlib.util.spec_from_file_location("app_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_plan_snapshot_roundtrips_through_study_plan():
    """The UI stores a plan snapshot in session state and rebuilds it later.

    `StudyPlan.to_dict()` returns derived values that are not constructor
    arguments, so the UI must snapshot only the dataclass fields.
    """
    from src.graph.planner import StudyPlan

    plan = create_plan(["A", "B"], days_until_exam=10, hours_per_day=2.0,
                       unavailable_days=[6])
    snapshot = {
        "subjects": plan.subjects,
        "days_until_exam": plan.days_until_exam,
        "hours_per_day": plan.hours_per_day,
        "unavailable_days": plan.unavailable_days,
        "session_minutes": plan.session_minutes,
        "schedule": plan.schedule,
        "subject_hours": plan.subject_hours,
    }
    restored = StudyPlan(**snapshot)
    assert restored.subjects == plan.subjects
    assert restored.unavailable_days == [6]
    assert restored.total_sessions == plan.total_sessions
    assert restored.total_hours == plan.total_hours

    # The derived keys must NOT be passed back in, or construction raises.
    with pytest.raises(TypeError):
        StudyPlan(**plan.to_dict())