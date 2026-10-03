"""Parse study-plan constraints out of free text.

Regex-based and deliberately conservative: on a miss the node asks the user for
the missing value rather than inventing one.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from src.graph.planner import WEEKDAY_NAMES, parse_unavailable_days

_SUBJECT_HINT = re.compile(
    r"\b(?:subjects?|subject list|studying|studying for|preparing for|for|and)\b",
    re.I,
)
# Messages that only adjust an existing plan, e.g. "I cannot study on Sunday."
_MODIFY_ONLY = re.compile(
    r"\b(?:cannot|can't|can not|unable to|no longer|won't|will not)\s+"
    r"(?:study|revise|attend|do)\b"
    r"|^\s*(?:remove|drop|add|change|increase|decrease|reduce|make)\b"
    r"|\b(?:instead of|rather than|more|less|only)\s+\d",
    re.I,
)
_EXAM_DATE = re.compile(
    r"exam(?:s)?[^.]*?on\s+([0-9]{1,2})(?:st|nd|rd|th)?\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*([0-9]{4})?",
    re.I,
)
_DAYS_COUNT = re.compile(
    r"(\d+)\s*(?:days?|weeks?)\b(?!\s*(ago|later|hour|hr|min))", re.I
)
_HOURS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b(?!\s*exam)", re.I
)
# "and 3 hours a day" / "2 hrs daily" states the unit after the number.
_HOURS_PER_DAY = re.compile(
    r"(?:and\s+)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b(?:\s*(?:a|per)\s*)?(?:day|daily)",
    re.I,
)
_SESSION = re.compile(r"(\d+)\s*(?:minutes?|mins?)\b", re.I)
# Interrogative openings. A question asks about something; it never enumerates
# subjects, and treating one as a subject list produced nonsense schedules.
_QUESTION = re.compile(
    r"^\s*(?:what|which|who|whom|whose|when|where|why|how|is|are|was|were|"
    r"do|does|did|can|could|should|will|would|tell|explain|list)\b",
    re.I,
)

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)}

# Words that appear in the prompt but are never subject names.
_STOP = {
    "i", "my", "me", "have", "has", "am", "are", "is", "can", "study", "studies",
    "studying", "subject", "subjects", "for", "and", "the", "in", "on", "with",
    "hours", "hour", "day", "days", "per", "exam", "exams", "until", "about",
    "only", "please", "make", "create", "plan", "schedule", "prepare", "preparing",
    "cannot", "cant", "not", "available", "available", "study", "it", "to", "a",
    "an", "each", "every", "total", "study", "exam", "each", "minutes", "minute",
    # Study-planning vocabulary. Without these, "4th semester", "my subjects"
    # and "to study my semester subjects" were accepted as subject names.
    "semester", "sem", "semesters", "table", "timetable", "prepared", "give",
    "need", "want", "help", "time", "times", "more", "less", "any", "all",
    "what", "which", "how", "many", "when", "where", "who", "why", "if",
    "there", "then", "than", "from", "by", "as", "at", "or", "but", "so",
    "week", "weeks", "month", "months", "syllabus", "current", "of", "get",
    "got", "us", "we", "our", "you", "your", "they", "them", "their",
}
# Ordinals and bare numbers carry no subject name ("4th semester" -> "4th").
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)?$", re.I)
# Any punctuation at all means this was a sentence fragment, not a subject.
_PUNCT_ONLY = re.compile(r"^[^\w&]+$")


def _is_junk_subject(name: str) -> bool:
    """True when the candidate is prompt wording rather than a subject name.

    The rule is that a real subject keeps at least one word that carries
    meaning on its own; "my subjects", "4th semester" and "to study my
    semester subjects" do not.
    """
    if not name or _PUNCT_ONLY.match(name):
        return True
    return not any(
        word.lower() not in _STOP and not _ORDINAL.match(word)
        for word in re.findall(r"[\w&+.]+", name)
    )


def parse_exam_date(text: str) -> date | None:
    """Return an absolute exam date mentioned in the text, if any."""
    match = _EXAM_DATE.search(text or "")
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _MONTHS.get(month_name.lower()[:3])
    if not month:
        return None
    try:
        return date(int(year or date.today().year), month, int(day))
    except ValueError:
        return None


def parse_days(text: str) -> int | None:
    """Days until the exam, from an explicit date or a 'N days' phrase."""
    exam_date = parse_exam_date(text)
    if exam_date:
        delta = (exam_date - date.today()).days
        return max(1, delta)
    match = _DAYS_COUNT.search(text or "")
    if match:
        try:
            value = int(match.group(1))
            if 0 < value <= 365:
                return value
        except ValueError:
            return None
    return None


def parse_hours_per_day(text: str) -> float | None:
    """Hours the student can study per day."""
    # Prefer the phrasing that names the period, so "15 days, 2 hours daily"
    # reads as two hours a day rather than two hours total.
    match = _HOURS_PER_DAY.search(text or "") or _HOURS.search(text or "")
    if not match:
        return None
    value = float(match.group(1))
    return value if 0 < value <= 16 else None


def parse_session_minutes(text: str) -> int | None:
    match = _SESSION.search(text or "")
    if match:
        value = int(match.group(1))
        return value if 0 < value <= 240 else None
    return None


def parse_subjects(text: str) -> list[str]:
    """Best-effort subject extraction from a list phrased in plain English.

    Pure function: no shared state, so concurrent Streamlit sessions cannot
    bleed subjects into each other's plans.
    """
    if not text:
        return []
    # A sentence that only adjusts an existing plan is not a subject list.
    if _MODIFY_ONLY.search(text):
        return []
    # A question asks about something; it never enumerates subjects. Without
    # this, "What are the internship requirements?" yielded itself as a subject.
    if _QUESTION.search(text):
        return []

    # Prefer the segment that follows an explicit subject cue:
    # "I have DBMS, OS and Computer Networks. Exams in 20 days."
    pool = text
    cue = re.search(
        r"\b(?:subjects?|studying for|preparing for|plan for|schedule for|"
        r"revision for|revise for|i have|i'?m studying)\b[:\s]*",
        text,
        flags=re.I,
    )
    if cue:
        pool = text[cue.end():]
    pool = re.split(r"[.;\n]", pool)[0]

    subjects: list[str] = []
    for raw in re.split(r",|\band\b|&", pool, flags=re.I):
        name = raw.strip(" .\n\t-")
        # A leading verb is a cue word, not part of the subject name.
        name = re.sub(
            r"^(?:study|studying|prep|preparing|revise|revising|cover|covering|"
            r"learn|learning|practice|practicing)\s+(?:for\s+)?",
            "",
            name,
            flags=re.I,
        )
        # Remove trailing constraints ("DBMS in 20 days" -> "DBMS").
        name = re.split(r"\b(?:in|after|starting|because)\b", name, flags=re.I)[0]
        name = _HOURS.sub("", name)
        name = _DAYS_COUNT.sub("", name)
        name = re.sub(r"\bfor\b", "", name, flags=re.I)
        name = _SESSION.sub("", name)
        name = re.sub(r"\bI\s+(?:can|could|will|have|am)\b.*$", "", name, flags=re.I)
        name = re.sub(r"\bexam(?:s)?\b.*$", "", name, flags=re.I)
        name = name.strip(" .,\n\t-")
        # Allow slightly longer names for multi-word subjects like "Computer Networks"
        if not name or len(name) > 50 or len(name.split()) > 8:
            continue
        if _is_junk_subject(name):
            continue
        if name.lower() not in [s.lower() for s in subjects]:
            subjects.append(name)
    return subjects


def parse_plan_request(text: str) -> dict:
    """Pull every planning input we can from free text.

    Missing values are left out of the result so the caller can ask for them
    instead of guessing.
    """
    subjects = parse_subjects(text)
    days = parse_days(text)
    hours = parse_hours_per_day(text)
    session = parse_session_minutes(text)
    unavailable = parse_unavailable_days(text)
    exam_date = parse_exam_date(text)

    result: dict = {"unavailable_days": unavailable, "subjects": subjects}
    if days:
        result["days_until_exam"] = days
    if hours:
        result["hours_per_day"] = hours
    if session:
        result["session_minutes"] = session
    if exam_date:
        result["exam_date"] = exam_date.isoformat()
    return result


def missing_plan_fields(request: dict) -> list[str]:
    """Human-readable names of the inputs still needed to build a plan."""
    missing = []
    if not request.get("subjects"):
        missing.append("subjects you need to study")
    if not request.get("days_until_exam"):
        missing.append("number of days until your exam")
    if not request.get("hours_per_day"):
        missing.append("hours you can study per day")
    return missing


def describe_days(days: list[int]) -> str:
    return ", ".join(WEEKDAY_NAMES[d] for d in days) if days else "none"