"""Central configuration. Every tunable lives here and can be overridden by .env."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# HuggingFace emits one INFO line per HTTP request while downloading or
# checking a model file, which buries our own progress output.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# Silence "unauthenticated requests to the HF Hub" - expected for public models.
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Paths -----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
FAISS_DIR = PROJECT_ROOT / os.getenv("FAISS_DIR", "storage/faiss")

# --- Retrieval / chunking --------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)
TOP_K = _env_int("TOP_K", 4)
# Chunks scoring below this fraction of the best hit are dropped, so a weak
# match never gets padded out with unrelated text.
SCORE_THRESHOLD = _env_float("SCORE_THRESHOLD", 0.25)
# Absolute relevance floor. A question whose best match is weaker than this is
# treated as unanswerable from the documents. Measured on the sample corpus,
# in-scope questions scored 0.24+ while out-of-scope noise scored <=0.233, so 0.235
# sits between them. Raise it to refuse more, lower it to recall more.
MIN_SCORE = _env_float("MIN_SCORE", 0.236)

# Pages yielding less than this many characters are flagged as likely scans.
MIN_CHARS_PER_PAGE = _env_int("MIN_CHARS_PER_PAGE", 50)

# --- LLM -------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 180)

# --- Conversation ----------------------------------------------------------
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 6)

# --- Study planner ---------------------------------------------------------
DEFAULT_SESSION_MINUTES = _env_int("DEFAULT_SESSION_MINUTES", 60)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)