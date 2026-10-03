"""Read the semester/subject table straight out of the syllabus PDF.

The syllabus lists every semester and its subjects in one table, and a 1.5b model
cannot be trusted to reproduce it: asked for the CSE subjects it returns only
Semester I, and when shown a different set of chunks it labels the wrong
semesters. Wrong subject names are worse than no answer, so the table is parsed
here and the model is only asked to present it.

Everything here works off the retrieved chunk text, so it obeys the same
grounding rule as the rest of the assistant: if the table is not in the context,
there is no syllabus to report.
"""

from __future__ import annotations

import re

# Roman numerals as written in the syllabus "Semester" column.
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
          "VII": 7, "VIII": 8, "IX": 9, "X": 10}

_SEMESTER_LABEL = re.compile(r"^(?:SEMESTER\s*)?([IVX]{1,5})\b(?:\s*SEMESTER)?$", re.I)
# The table header introduces the semester -> subjects pairing.
_TABLE_START = re.compile(r"^\s*semester\b", re.I | re.M)
_CREDITS = re.compile(r"^\s*credits?\b", re.I)
# Curriculum items are listed with semicolons. A row without one is prose or a
# calendar entry ("Commencement of classes"), not a subject list, and must not be
# offered to the student as if it were.
_SUBJECT_DELIMITER = re.compile(r"[;]")

# "4th semester", "4 semester", "semester 4", "4th sem", "IV semester".
_ORDINALS = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
             "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X"}
_SEMESTER_MENTION = re.compile(
    r"\b(?:semester|sem)\s*(?:number\s*)?([0-9]{1,2}|[IVX]{1,5})\b"
    r"|\b([0-9]{1,2})(?:st|nd|rd|th)?\s*(?:semester|sem)\b",
    re.I,
)
# "...and 4" / ", 2 and 3" continue the list started by an earlier mention.
_SEMESTER_LIST_TAIL = re.compile(
    r"(?:,|\band\b)\s*([0-9]{1,2}|[IVX]{1,5})\b", re.I
)
# A list of semesters does not run into a study-hours figure: in
# "4th semester subjects. I have 10 days and 3 hours a day" the trailing
# "and 3 hours" is not a third semester.
_NOT_A_LIST_TAIL = re.compile(
    r"(?:[0-9]{1,2}|[IVX]{1,5})\s*(?:hour|hr|min|minute|day|week|month|%"
    r"mark|credit|subject|course|question)s?\b",
    re.I,
)
# "3rd and 4th semester": the ordinals lead and only the last carries the noun.
_ORDINAL_SEMESTER_TAIL = re.compile(
    r"\b([0-9]{1,2})(?:st|nd|rd|th)\b(?=[\s\S]{0,20}?\b(?:semester|sem)\b)",
    re.I,
)


def _to_roman(token: str) -> int | None:
    token = token.strip().upper()
    if token.isdigit():
        return _ROMAN.get(_ORDINALS.get(token, ""))
    return _ROMAN.get(token)


def parse_semester_subjects(text: str) -> dict[int, list[str]]:
    """Extract {semester number: [subject names]} from syllabus table text.

    Returns an empty dict when the text is not the syllabus table, which is how
    callers tell "no syllabus in context" apart from "syllabus has no such
    semester".
    """
    if not text:
        return {}

    # Only look after the table header; a Roman numeral earlier in the page is
    # part of a heading, not a row.
    header = _TABLE_START.search(text)
    body = text[header.start():] if header else text

    lines = [line.strip() for line in body.splitlines()]

    semesters: dict[int, list[str]] = {}
    for i, line in enumerate(lines):
        match = _SEMESTER_LABEL.match(line)
        if not match:
            continue
        number = _to_roman(match.group(1))
        if number is None or number in semesters:
            continue
        # Consecutive semester labels ("I Semester", "II Semester") are table
        # column headers, as in the academic calendar, where the row underneath
        # is a date or an event and the label has no subjects of its own.
        # A real row is a lone label followed by the subject list.
        if i + 1 < len(lines) and _SEMESTER_LABEL.match(lines[i + 1]):
            continue
        if i + 1 >= len(lines):
            continue
        row = lines[i + 1]
        # The row below the semester holds the subjects; anything else (a credit
        # count, a heading, end of table) means this column is not what we think.
        if not row or row.isdigit() or _CREDITS.match(row):
            continue
        if not _SUBJECT_DELIMITER.search(row):
            continue
        subjects = [part.strip() for part in row.split(";") if part.strip()]
        if subjects:
            semesters[number] = subjects
    return semesters


def semesters_in_text(text: str) -> dict[int, list[str]]:
    """Alias kept for readability at call sites."""
    return parse_semester_subjects(text)


def referenced_semesters(text: str) -> list[int]:
    """Semester numbers the student named, e.g. '4th semester' -> [4]."""
    if not text:
        return []
    found: list[int] = []
    for arabic, roman in _SEMESTER_MENTION.findall(text):
        number = _to_roman(arabic) if arabic else _to_roman(roman)
        if number and number not in found:
            found.append(number)
    # "semester 3 and 4" names two semesters; the second has no word of its own.
    if found:
        for match in _SEMESTER_LIST_TAIL.finditer(text):
            tail = text[match.start():]
            if _NOT_A_LIST_TAIL.search(tail):
                continue
            number = _to_roman(match.group(1))
            if number and number not in found:
                found.append(number)

    # "3rd and 4th semester" reads the other way round: the ordinals come first
    # and only the last one carries the word "semester".
    for match in _ORDINAL_SEMESTER_TAIL.finditer(text):
        number = _to_roman(match.group(1))
        if number and number not in found:
            found.append(number)
    return found


def format_semester_table(semesters: dict[int, list[str]]) -> str:
    """Render the semester table, keeping the syllabus' own subject wording."""
    lines = []
    for number in sorted(semesters):
        lines.append(f"Semester {number}: " + "; ".join(semesters[number]))
    return "\n".join(lines)


def resolve_semester_subject_names(text: str, semesters: dict[int, list[str]]) -> list[str]:
    """Subject names for whichever semesters the student asked about.

    An empty result means the student named a semester that the syllabus does
    not cover, so the caller must ask rather than invent subjects.
    """
    wanted = referenced_semesters(text)
    if not wanted or not semesters:
        return []
    resolved: list[str] = []
    for number in wanted:
        for subject in semesters.get(number, []):
            if subject not in resolved:
                resolved.append(subject)
    return resolved


def resolve_single_semester(text: str, semesters: dict[int, list[str]]) -> list[str]:
    """Subject names for exactly one semester the student named.

    Used by the study planner. Two semesters at once exceed the number of
    sessions a normal plan has, and the planner reports that clearly, so only
    an unambiguous single-semester request is resolved here; anything else is
    left to the student to spell out.
    """
    wanted = referenced_semesters(text)
    if len(wanted) != 1 or wanted[0] not in semesters:
        return []
    return list(semesters[wanted[0]])