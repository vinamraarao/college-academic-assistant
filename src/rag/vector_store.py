"""FAISS vector store: build, persist, and load the college knowledge index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL, FAISS_DIR
from src.rag.embeddings import get_embeddings
from src.rag.loader import load_pdfs
from src.rag.splitter import split_pages

logger = logging.getLogger(__name__)

INDEX_FILE = "college_index"

# FAISS defaults to L2 distance, which yields unbounded relevance scores and
# makes a fixed cutoff meaningless. Cosine gives a true -1..1 similarity, so the
# relevance floor in the retriever behaves as intended.
DISTANCE_STRATEGY = DistanceStrategy.COSINE


class IndexNotFoundError(RuntimeError):
    """Raised when no FAISS index has been built yet."""


def build_index(
    data_dir: Path | str = DATA_DIR,
    save_dir: Path | str = FAISS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL,
    verbose: bool = True,
) -> FAISS:
    """Run the full ingestion pipeline and persist the FAISS index."""
    if verbose:
        print(f"[1/4] Loading PDFs from {data_dir} ...")
    pages, report = load_pdfs(data_dir)
    if verbose:
        print("      " + report.summary().replace("\n", "\n      "))

    if verbose:
        print(f"[2/4] Splitting into chunks (size={chunk_size}, overlap={chunk_overlap}) ...")
    documents: list[Document] = split_pages(pages, chunk_size, chunk_overlap)

    if verbose:
        print(f"[3/4] Embedding {len(documents)} chunks with {embedding_model} ...")
    embeddings = get_embeddings(embedding_model)
    store = FAISS.from_documents(
        documents, embeddings, distance_strategy=DISTANCE_STRATEGY
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(save_path / INDEX_FILE))
    if verbose:
        print(f"[4/4] Saved index ({len(documents)} vectors) to {save_path / INDEX_FILE}")
    return store


def load_index(
    save_dir: Path | str = FAISS_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    embeddings=None,
) -> FAISS:
    """Load a previously built FAISS index, with a clear error if it is missing."""
    path = Path(save_dir) / INDEX_FILE
    if not path.exists():
        raise IndexNotFoundError(
            f"No FAISS index at {path}. Build it first:\n"
            "    python -m src.rag.vector_store --build"
        )
    return FAISS.load_local(
        str(path),
        embeddings or get_embeddings(embedding_model),
        allow_dangerous_deserialization=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect the FAISS index.")
    parser.add_argument("--build", action="store_true", help="(Re)build the index from data/")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--save-dir", default=str(FAISS_DIR))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.build:
        store = build_index(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Indexed {store.index.ntotal} chunks.")
        return 0

    store = load_index(args.save_dir)
    print(f"Index ready: {store.index.ntotal} chunks in {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())