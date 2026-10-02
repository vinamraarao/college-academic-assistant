"""Shared pytest fixtures.

Tests that need the LLM are marked `llm` and skipped automatically when Ollama
or its model is unavailable, so the suite still runs on a clean machine.
"""

from __future__ import annotations

import pytest

from src.config import OLLAMA_MODEL


def _llm_ready() -> bool:
    try:
        from src.llm.model import check_ollama

        wanted = OLLAMA_MODEL.split(":")[0]
        return any(wanted in name for name in check_ollama())
    except Exception:
        return False


LLM_AVAILABLE = _llm_ready()

requires_llm = pytest.mark.skipif(
    not LLM_AVAILABLE,
    reason=f"Ollama or model '{OLLAMA_MODEL}' unavailable - start Ollama and pull the model",
)


@pytest.fixture(scope="session")
def index_available() -> bool:
    """Whether a FAISS index has been built for data/."""
    from src.config import FAISS_DIR
    from src.rag.vector_store import INDEX_FILE

    return (FAISS_DIR / INDEX_FILE).exists()


@pytest.fixture
def sample_pages() -> list[dict]:
    """One synthetic page, so loader/splitter tests do not depend on data/."""
    return [
        {
            "page_content": (
                "Attendance Requirement. Students must maintain a minimum of 75 "
                "percent attendance in every subject for a semester. A student "
                "below this threshold is not eligible to write the examination. "
            ),
            "source": "Test_Document.pdf",
            "page": 1,
        }
    ]