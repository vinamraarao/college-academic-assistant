"""Shared helpers: message history, formatting, and app-level errors."""

from __future__ import annotations

import logging
import re
from typing import Iterable

from src.config import MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)


# --- Errors surfaced to the user ------------------------------------------
class AppError(RuntimeError):
    """Base class for errors that have a user-facing message."""


class OllamaUnavailableError(AppError):
    """Ollama is not installed or its server is not reachable."""


class ModelMissingError(AppError):
    """Ollama is running but the configured model has not been pulled."""


# --- History ---------------------------------------------------------------
def trim_history(
    history: list[dict],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict]:
    """Keep only the most recent turns.

    History feeds query rewriting and the chat prompt; letting it grow unbounded
    would eventually crowd out the retrieved college context, which is the part
    that must stay intact.
    """
    return history[-max_messages:] if max_messages > 0 else []


def history_to_text(history: Iterable[dict]) -> str:
    """Render history as a compact transcript for prompt embedding."""
    lines = []
    for message in history:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def recent_user_turns(history: Iterable[dict], n: int = 2) -> list[str]:
    """The last n user messages, oldest first — used for pronoun resolution."""
    users = [
        str(m.get("content", "")).strip()
        for m in history
        if m.get("role") == "user" and str(m.get("content", "")).strip()
    ]
    return users[-n:]


# --- Text ------------------------------------------------------------------
def clean_llm_text(text: str) -> str:
    """Strip preamble the small model tends to add ("Sure! Here is...")."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^(sure|of course|absolutely|certainly)[,!.]?\s*", "", cleaned, flags=re.I)
    return cleaned.strip()


def format_citation(source: str, page: object) -> str:
    """'Academic_Regulations.pdf, page 12' — page omitted when unknown."""
    if page in (None, 0, "", "0"):
        return source
    return f"{source}, page {page}"


def truncate(text: str, limit: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"