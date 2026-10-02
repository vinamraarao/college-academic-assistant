"""Deterministic study-plan scheduling.

The schedule itself is computed in Python, not by the LLM. A 1.5b model cannot
reliably do date arithmetic across 20 days, and a plan with an impossible
session count or a session on an unavailable day would be worse than no plan.
The LLM is used only to summarise the schedule that was actually computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

from src.config import DEFAULT_SESSION_MINUTES

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

# Accepted spellings for "I cannot study on X".
_DAY_ALIASES: dict[str, int] = {}
for _i, _name in enumerate(WEEKDAY_NAMES):
    _DAY_ALIASES[_name.lower()] = _i
    _DAY_ALIASES[_name.lower()[:3]] = _i
_DAY_ALIASES.update({"tues": 1, "thur": 3, "thurs": 3})


class StudyPlanError(ValueError):
    """Raised when the plan inputs cannot produce a usable schedule."""


@dataclass
class StudyPlan:
    """A computed plan plus the inputs needed to recompute it after a change."""

    subjects: list[str]
    days_until_exam: int
    hours_per_day: float
    unavailable_days: list[int] = field(default_factory=list)
    session_minutes: int = DEFAULT_SESSION_MINUTES
    schedule: list[dict] = field(default_factory=list)
    subject_hours: dict[str, float] = field(default_factory=dict)

    @property
    def total_sessions(self) -> int:
        return len(self.schedule)

    @property
    def total_hours(self) -> float:
        return round(sum(day["hours"] for day in self.schedule), 2)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["total_sessions"] = self.total_sessions
        data["total_hours"] = self.total_hours
        return data


def parse_unavailable_days(text: str) -> list[int]:
    """Extract weekday indices from free text such as 'I cannot study on Sunday'."""
    found: set[int] = set()
    lowered = (text or "").lower()
    for alias, index in _DAY_ALIASES.items():
        if alias in lowered:
            found.add(index)
    return sorted(found)


def create_plan(
    subjects: list[str],
    days_until_exam: int,
    hours_per_day: float,
    unavailable_days: list[int] | None = None,
    session_minutes: int = DEFAULT_SESSION_MINUTES,
    start_date: date | None = None,
) -> StudyPlan:
    """Spread study time evenly across subjects over the available days.

    Days excluded by `unavailable_days` are dropped entirely. Subjects are
    rotated round-robin so a short plan still touches every subject, and any
    remainder is distributed in whole sessions.
    """
    subjects = [s.strip() for s in subjects if s and s.strip()]
    if not subjects:
        raise StudyPlanError("No subjects were provided. Name at least one subject.")
    if days_until_exam <= 0:
        raise StudyPlanError(
            "The number of days until the exam must be at least 1. "
            "Enter how many days you have left to study."
        )
    if hours_per_day <= 0:
        raise StudyPlanError("Available study hours per day must be greater than 0.")
    if session_minutes <= 0:
        raise StudyPlanError("Session length must be greater than 0 minutes.")

    unavailable = sorted(set(unavailable_days or []))
    unavailable = [d for d in unavailable if 0 <= d <= 6]
    if len(unavailable) >= 7:
        raise StudyPlanError(
            "Every day of the week was marked unavailable, so no study time remains."
        )

    today = start_date or date.today()
    study_days: list[date] = []
    for offset in range(days_until_exam):
        day = today + timedelta(days=offset)
        if day.weekday() not in unavailable:
            study_days.append(day)

    if not study_days:
        raise StudyPlanError(
            "Every remaining day before the exam was marked unavailable."
        )

    sessions_per_day = max(1, int(round(hours_per_day * 60 / session_minutes)))
    usable_hours = sessions_per_day * session_minutes / 60
    total_sessions = len(study_days) * sessions_per_day

    if total_sessions < len(subjects):
        raise StudyPlanError(
            f"Not enough study slots: {total_sessions} available for "
            f"{len(subjects)} subjects. Increase available hours, reduce the exam "
            "countdown days, or study on more days."
        )

    # Even split of whole sessions across subjects, remainder to the earliest.
    base, extra = divmod(total_sessions, len(subjects))
    allocations = {subject: base + (1 if i < extra else 0)
                   for i, subject in enumerate(subjects)}

    schedule: list[dict] = []
    sessions_per_subject = {s: 0 for s in subjects}
    for day_index, day in enumerate(study_days):
        entries = []
        for slot in range(sessions_per_day):
            # Round-robin over subjects, skipping any already filled.
            for offset in range(len(subjects)):
                subject = subjects[(day_index * sessions_per_day + slot + offset) % len(subjects)]
                if sessions_per_subject[subject] < allocations[subject]:
                    sessions_per_subject[subject] += 1
                    entries.append({"subject": subject, "minutes": session_minutes})
                    break
        schedule.append(
            {
                "date": day.isoformat(),
                "weekday": WEEKDAY_NAMES[day.weekday()],
                "sessions": entries,
                "hours": round(len(entries) * session_minutes / 60, 2),
            }
        )

    subject_hours = {
        subject: round(count * session_minutes / 60, 2)
        for subject, count in sessions_per_subject.items()
    }

    return StudyPlan(
        subjects=subjects,
        days_until_exam=days_until_exam,
        hours_per_day=hours_per_day,
        unavailable_days=unavailable,
        session_minutes=session_minutes,
        schedule=schedule,
        subject_hours=subject_hours,
    )


def format_plan(plan: StudyPlan, max_days: int = 10) -> str:
    """Render the schedule for the prompt, trimmed so it stays compact."""
    lines = [
        f"Subjects: {', '.join(plan.subjects)}",
        f"Days until exam: {plan.days_until_exam} (unavailable weekdays excluded)",
        f"Hours per study day: {plan.hours_per_day}",
        f"Session length: {plan.session_minutes} minutes",
        f"Total: {plan.total_sessions} sessions, {plan.total_hours} hours",
        "Hours per subject: "
        + ", ".join(f"{s}={h}h" for s, h in plan.subject_hours.items()),
        "",
        "Schedule:",
    ]
    for day in plan.schedule[:max_days]:
        subjects = ", ".join(
            f"{s['subject']} ({s['minutes']}m)" for s in day["sessions"]
        )
        lines.append(f"  {day['weekday']} {day['date']}: {subjects}")
    if len(plan.schedule) > max_days:
        lines.append(f"  ... and {len(plan.schedule) - max_days} more day(s) follow the same pattern")
    return "\n".join(lines)