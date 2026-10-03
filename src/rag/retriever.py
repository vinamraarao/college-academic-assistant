"""Retrieval over the FAISS index with a configurable relevance floor."""

from __future__ import annotations

import logging
import warnings

from langchain_core.documents import Document

from src.config import MIN_SCORE, SCORE_THRESHOLD, TOP_K
from src.rag.vector_store import FAISS, load_index

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when the vector store cannot be queried."""


class CollegeRetriever:
    """Similarity search over the college knowledge base.

    Chunks scoring far below the best hit are discarded. Without this, an
    unrelated question still returns k chunks, which invites the LLM to build a
    confident-sounding answer out of noise.
    """

    def __init__(
        self,
        store: FAISS,
        top_k: int = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.store = store
        self.top_k = max(1, top_k)
        self.score_threshold = score_threshold
        # Absolute floor, applied on top of the relative one. Without it a
        # question whose only "match" is noise (e.g. best score 0.03) still
        # passes, and irrelevant text reaches the prompt where the model may
        # build a plausible-sounding answer from it.
        self.min_score = min_score

    def _similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        try:
            # Cosine distance can legitimately produce a negative relevance
            # score for an unrelated chunk, which LangChain warns about. The
            # value is still meaningful for ranking, so warn and carry on.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Relevance scores must be")
                return self.store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """Return (chunk, relevance) pairs, strongest first.

        The list may be shorter than top_k, or empty, when nothing in the
        knowledge base is relevant — that empty case is the signal used to tell
        the user the answer is not in the college documents.
        """
        if not query or not query.strip():
            return []

        k = top_k or self.top_k
        hits = self._similarity_search(query.strip(), k)
        if not hits:
            return []

        best = max(score for _, score in hits)
        # Both gates must pass: the absolute floor rejects noise-only matches,
        # the relative floor drops weak neighbours of a good hit.
        if best < self.min_score:
            logger.debug(
                "No chunk cleared the absolute floor (best=%.3f < %.3f)",
                best, self.min_score,
            )
            return []

        floor = max(self.min_score, best * self.score_threshold)
        kept = [(doc, score) for doc, score in hits if score >= floor]
        logger.debug(
            "Retrieved %d/%d chunks (best=%.3f floor=%.3f)", len(kept), len(hits), best, floor
        )
        return kept

    @staticmethod
    def format_context(
        hits: list[tuple[Document, float]],
        max_chars: int = 6000,
    ) -> str:
        """Render retrieved chunks into a numbered block for the prompt.

        Each chunk is tagged with its real source file and page so the model
        can cite it and the answer's citations are checkable.
        """
        if not hits:
            return ""
        blocks: list[str] = []
        used = 0
        for i, (doc, _score) in enumerate(hits, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            location = f"{source}" + (f", page {page}" if page else "")
            text = doc.page_content.strip()
            block = f"[Chunk {i}] (Source: {location})\n{text}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def format_sources(hits: list[tuple[Document, float]]) -> list[dict]:
        """Citation records for display in the UI, merged per file+page."""
        seen: set[tuple[str, object]] = set()
        sources: list[dict] = []
        for doc, score in hits:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": source, "page": page, "score": round(score, 4)})
        return sources


_STORE: FAISS | None = None


def get_retriever(
    top_k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    min_score: float = MIN_SCORE,
) -> CollegeRetriever:
    """Build a retriever over the on-disk index, loading it once per process."""
    global _STORE
    if _STORE is None:
        _STORE = load_index()
    return CollegeRetriever(
        _STORE, top_k=top_k, score_threshold=score_threshold, min_score=min_score
    )