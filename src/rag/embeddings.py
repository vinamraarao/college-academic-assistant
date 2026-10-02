"""Sentence-Transformers embedding model wrapper."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_CACHE: dict[str, Embeddings] = {}


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> Embeddings:
    """Return a cached HuggingFaceEmbeddings instance.

    The model is loaded once per process — Streamlit reruns would otherwise
    re-download and re-initialise it on every interaction.
    """
    if model_name in _CACHE:
        return _CACHE[model_name]

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        # Fail fast if the model directory is broken rather than at query time.
        embeddings.embed_query("connectivity check")
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model '{model_name}'. "
            f"Check it is spelled correctly and downloadable. Details: {exc}"
        ) from exc

    _CACHE[model_name] = embeddings
    logger.info("Loaded embedding model %s", model_name)
    return embeddings