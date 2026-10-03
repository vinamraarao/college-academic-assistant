"""Retrieval over the FAISS index with a configurable relevance floor."""

from __future__ import annotations

import logging
import re
import warnings

from langchain_core.documents import Document

from src.config import MIN_SCORE, SCORE_THRESHOLD, TOP_K
from src.rag.vector_store import FAISS, load_index

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when the vector store cannot be queried."""


_URL = re.compile(r"https?://\S+|www\.\S+")
# Function words carry no document identity, so they are dropped before a
# question is compared against the document names.
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can did do does for from had has
    have how i if in into is it its me my no not of on or our she should so than
    that the their them then there these they this those to us was we were what
    when where which who whom whose why will with would you your tell explain
    about current please need want any all more less much many name list give""".split()
)
# Stripped plural/gerund endings so "syllabus" and "syllabi"-style variants and
# "requirements"/"requirement" match the same document.
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "es", "s")


def _normalise(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    return {
        term
        for word in re.findall(r"[a-z0-9]+", _URL.sub(" ", text or "").lower())
        if len(word) > 2 and word not in _STOPWORDS
        for term in (_normalise(word),)
    }


def _document_terms(source: str) -> set[str]:
    """Meaningful words from a file name: 04_Syllabus_CSE_NMAMIT.pdf -> syllabus, cse."""
    stem = re.sub(r"^\d+_", "", source or "").replace(".pdf", "")
    return {
        _normalise(word)
        for word in re.findall(r"[a-z0-9]+", stem.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def _relevant_documents(
    hits: list[tuple[Document, float]], query: str
) -> list[tuple[Document, float]]:
    """Keep documents whose name the question refers to.

    Retrieval scores alone cannot do this: every document shares boilerplate,
    so a syllabus question scored the internship document at 95% of the best
    hit and it was cited as a source. Matching the question against the
    document names separates them, because "syllabus" and "internship" are
    named in exactly one file each.

    A document whose name shares nothing with the question is still kept when no
    other retrieved document matches either, since the title is a hint and not
    a requirement.
    """
    query_terms = _terms(query)
    if not query_terms:
        return hits

    on_topic: set[str] = set()
    for doc, _score in hits:
        source = doc.metadata.get("source", "")
        if query_terms & _document_terms(source):
            on_topic.add(source)

    if not on_topic:
        return hits
    return [hit for hit in hits if hit[0].metadata.get("source", "") in on_topic]


def _page_text(store: FAISS, source: str, page: object) -> str:
    """Reassemble every chunk that came from one page, in reading order.

    Chunks of a page are found by their metadata rather than by index
    position, since the docstore holds chunks from every document.
    """
    parts: list[tuple[int, str]] = []
    for chunk in store.docstore._dict.values():
        meta = getattr(chunk, "metadata", None) or {}
        if meta.get("source") != source or meta.get("page") != page:
            continue
        content = getattr(chunk, "page_content", "") or ""
        if content:
            parts.append((meta.get("chunk_index", 0), content))
    if not parts:
        return ""
    return "\n".join(text for _order, text in sorted(parts, key=lambda p: p[0]))


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
    def complete_hits(
        hits: list[tuple[Document, float]],
        store: FAISS,
        max_chars: int = 6000,
    ) -> list[tuple[Document, float]]:
        """Expand each hit to the whole of its own page.

        Chunks are cut at ~800 characters, so a page is split across several of
        them and retrieval sees only the ones that happen to resemble the query.
        On the syllabus that meant the model was shown 4 of the 6 chunks and
        numbered the semesters wrongly, because the pieces it did see stopped
        mid-table. Restoring the page keeps a list intact.

        Order and scores are preserved; each page is added at most once and the
        total stays inside `max_chars`.
        """
        if not hits:
            return []
        by_page: dict[tuple[str, object], str] = {}
        for doc, _score in hits:
            by_page.setdefault(
                (doc.metadata.get("source"), doc.metadata.get("page")), ""
            )

        used = sum(len(doc.page_content) for doc, _ in hits)
        for key in list(by_page):
            if used >= max_chars:
                break
            text = _page_text(store, *key)
            if text and len(text) > len(by_page[key]):
                used += len(text) - len(by_page[key])
                by_page[key] = text

        expanded: list[tuple[Document, float]] = []
        seen: set[tuple[str, object]] = set()
        for doc, score in hits:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            text = by_page.get(key) or doc.page_content
            expanded.append(
                (
                    Document(page_content=text, metadata=dict(doc.metadata)),
                    score,
                )
            )
        return expanded

    @staticmethod
    def format_sources(
        hits: list[tuple[Document, float]],
        query: str = "",
    ) -> list[dict]:
        """Citation records for display in the UI, merged per file+page.

        When a `query` is given, documents that the question clearly does not
        ask about are left out. Every citation then names a document that is
        about the subject of the question, rather than one that merely scored
        well on shared boilerplate.
        """
        hits = _relevant_documents(hits, query) if query else hits
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