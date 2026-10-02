"""Calculator tool: the project's external tool, also exposed to the LLM.

Uses Python's own arithmetic rather than eval, so no user-supplied text is ever
executed as code.
"""

from __future__ import annotations

import math
import re

_OPERATORS = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "**", "%": "%"}


class CalculatorError(ValueError):
    """Raised for unsupported or malformed expressions."""


def _to_python_expression(expression: str) -> str:
    """Normalise math symbols and reject anything that is not arithmetic.

    Only digits, whitespace, the four operators, parentheses, dots, and the
    names of a small whitelist of functions may appear.
    """
    expr = (expression or "").strip()
    if not expr:
        raise CalculatorError("No expression was provided.")

    cleaned = expr.replace("^", "**")
    cleaned = re.sub(r"(?<=\d)\s*%", "/ 100", cleaned)  # "50%" -> "50 / 100"

    allowed = set("0123456789.+-*/() \t")
    allowed |= set(_FUNCTIONS)
    stripped = re.sub(r"[a-zA-Z_]+", "", cleaned)
    if not set(stripped) <= allowed:
        raise CalculatorError(
            f"Unsupported characters in '{expression}'. Use numbers and + - * / ^ ( )."
        )

    # `x` and `**` are fine; reject names that slipped past the character check.
    names = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", cleaned))
    unknown = names - set(_FUNCTIONS)
    if unknown:
        raise CalculatorError(
            f"Unknown term(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(_FUNCTIONS))}"
        )
    return cleaned


_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "pow": pow,
}

_EVAL_ENV = {**{name: getattr(math, name) for name in dir(math)
                if not name.startswith("_")}, **_FUNCTIONS, "__builtins__": {}}


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression and return the numeric result."""
    expr = _to_python_expression(expression)
    try:
        result = eval(expr, _EVAL_ENV, {})  # noqa: S307 - sanitised above
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except Exception as exc:
        raise CalculatorError(f"Could not evaluate '{expression}': {exc}") from exc
    return float(result)


def percentage(part: float, whole: float) -> float:
    """What percentage `part` is of `whole`."""
    if whole == 0:
        raise CalculatorError("Cannot compute a percentage of zero.")
    return round(part / whole * 100, 2)


def study_hour_breakdown(
    subjects: list[str], days: int, hours_per_day: float
) -> dict:
    """Split total available study time evenly across subjects."""
    if not subjects:
        raise CalculatorError("No subjects were provided.")
    if days <= 0 or hours_per_day <= 0:
        raise CalculatorError("Days and hours per day must both be greater than 0.")

    total_hours = days * hours_per_day
    per_subject = total_hours / len(subjects)
    return {
        "total_hours": round(total_hours, 2),
        "subjects": len(subjects),
        "hours_per_subject": round(per_subject, 2),
        "breakdown": {s: round(per_subject, 2) for s in subjects},
    }


def describe(tool_name: str) -> str:
    """One-line description used when the tool is offered to the LLM."""
    return {
        "calculate": "Evaluate an arithmetic expression, e.g. '3 * 4 + 10'.",
        "percentage": "Compute what percent a part is of a whole.",
        "study_hour_breakdown": "Split total study time evenly across subjects.",
    }.get(tool_name, "Unknown calculator tool.")


# --- LangChain tool wrappers ------------------------------------------------
try:
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression. Use for arithmetic, percentages and
        study-hour totals. Input example: '(120 * 0.75) + 3 * 4'."""
        return str(calculate(expression))

    @tool
    def percentage_calculator(part: float, whole: float) -> str:
        """Compute what percentage `part` is of `whole`."""
        return f"{percentage(part, whole)}%"

    TOOLS = [calculator, percentage_calculator]

except ImportError:  # pragma: no cover - langchain-core always present in practice
    TOOLS = []