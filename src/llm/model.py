"""Ollama LLM access with actionable failure messages."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import LLM_TEMPERATURE, LLM_TIMEOUT, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.utils.helpers import ModelMissingError, OllamaUnavailableError

logger = logging.getLogger(__name__)


def check_ollama(base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> list[str]:
    """Return the locally installed model names, or raise a setup error.

    Separates "Ollama is not running" from "the model is not pulled", which
    need different fixes from the user.
    """
    try:
        from ollama import Client

        response = Client(host=base_url).list()
        return [m.get("model", m.get("name", "")) for m in response.get("models", [])]
    except Exception as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {base_url}.\n"
            "  1. Is Ollama installed?  https://ollama.com/download\n"
            "  2. Start it: 'ollama serve'  (on Windows, launch Ollama from the Start menu)\n"
            f"  Underlying error: {exc}"
        ) from exc


def ensure_model_available(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
    """Raise ModelMissingError unless `model` (or a matching tag) is installed."""
    installed = check_ollama(base_url)
    wanted = model.split(":")[0]
    if not any(name == model or name.startswith(wanted) for name in installed):
        available = ", ".join(installed) if installed else "(none installed)"
        raise ModelMissingError(
            f"Model '{model}' is not installed on this machine.\n"
            f"  Download it:  ollama pull {model}\n"
            f"  Installed models: {available}"
        )


@lru_cache(maxsize=2)
def get_llm(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> ChatOllama:
    """Return a cached ChatOllama client.

    Small local models are slow to load; caching keeps Streamlit reruns from
    paying that cost repeatedly.
    """
    ensure_model_available(model, base_url)
    logger.info("Connecting to Ollama model %s at %s", model, base_url)
    return ChatOllama(
        model=model,
        temperature=LLM_TEMPERATURE,
        base_url=base_url,
        num_ctx=4096,
        request_timeout=LLM_TIMEOUT,
    )


def invoke(llm: ChatOllama, prompt: str | list) -> str:
    """Send a prompt (raw text or ChatPrompt messages) and return plain text.

    Passing the full message list matters: system messages carry the rules and,
    for the planner, the generated schedule itself.
    """
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):  # some providers return content blocks
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return (content or "").strip()
    except Exception as exc:
        raise RuntimeError(
            f"The Ollama model failed to generate a response: {exc}\n"
            "Check that Ollama is still running and the model is loaded."
        ) from exc