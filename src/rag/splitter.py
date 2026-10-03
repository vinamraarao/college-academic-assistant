"""Turn per-page text into overlapping chunks, keeping page metadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


class SplitterConfigError(ValueError):
    """Raised when chunk size and overlap are inconsistent."""


def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Recursive splitter that prefers paragraph, then line, then sentence breaks."""
    if chunk_size <= 0:
        raise SplitterConfigError(f"CHUNK_SIZE must be positive, got {chunk_size}.")
    if chunk_overlap < 0:
        raise SplitterConfigError(f"CHUNK_OVERLAP must be >= 0, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise SplitterConfigError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE "
            f"({chunk_size}); otherwise chunks would loop indefinitely."
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Separators are tried in order, so a chunk rarely breaks mid-sentence.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split page records into Documents that retain source file and page number."""
    splitter = build_splitter(chunk_size, chunk_overlap)
    documents: list[Document] = []

    for page in pages:
        text = page.get("page_content", "").strip()
        if not text:
            continue
        for position, chunk in enumerate(splitter.split_text(text)):
            cleaned = chunk.strip()
            if len(cleaned) < 20:
                continue  # fragments too small to carry meaning hurt retrieval
            documents.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "source": page.get("source", "unknown"),
                        "page": page.get("page", 0),
                        # Reading order, so a page can be reassembled at
                        # retrieval time from the individual chunks.
                        "chunk_index": position,
                    },
                )
            )

    if not documents:
        raise SplitterConfigError(
            "Splitting produced no usable chunks. Check that the PDFs contain "
            "selectable text (not scanned images)."
        )
    return documents