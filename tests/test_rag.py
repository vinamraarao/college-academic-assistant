"""RAG pipeline tests: loading, splitting, embeddings, FAISS, retrieval."""

from __future__ import annotations

import pytest
from pypdf import PdfWriter

from src.config import DATA_DIR, FAISS_DIR
from src.rag.embeddings import get_embeddings
from src.rag.loader import DataDirectoryError, discover_pdfs, load_pdfs
from src.rag.retriever import CollegeRetriever
from src.rag.splitter import SplitterConfigError, build_splitter, split_pages
from src.rag.vector_store import INDEX_FILE, IndexNotFoundError, build_index, load_index


def _write_text_pdf(path, text: str) -> None:
    """Write a minimal one-page PDF with extractable text.

    pypdf cannot create text content, so a hand-built minimal PDF is used. It is
    small enough to keep inline and avoids pulling reportlab into the tests.
    """
    from tests.pdf_fixtures import build_pdf_bytes

    path.write_bytes(build_pdf_bytes(text))


# --- 1. PDF loading --------------------------------------------------------
def test_discover_pdfs_finds_data_files(index_available):
    pdfs = discover_pdfs(DATA_DIR)
    assert len(pdfs) > 0, "No PDFs found in data/"
    assert all(p.suffix == ".pdf" for p in pdfs)


def test_discover_pdfs_missing_directory_raises(tmp_path):
    with pytest.raises(DataDirectoryError, match="not found"):
        discover_pdfs(tmp_path / "does_not_exist")


def test_discover_pdfs_empty_directory_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataDirectoryError, match="No PDF files found"):
        discover_pdfs(tmp_path / "empty")


def test_load_pdfs_extracts_text_with_page_metadata(index_available):
    pages, report = load_pdfs(DATA_DIR)
    assert len(pages) > 0
    assert report.pages_total > 0
    for page in pages:
        assert page["page_content"].strip()
        assert page["source"].endswith(".pdf")
        assert isinstance(page["page"], int) and page["page"] >= 1


def test_load_pdfs_reports_empty_directory(tmp_path):
    (tmp_path / "none").mkdir()
    with pytest.raises(DataDirectoryError, match="No PDF files found"):
        load_pdfs(tmp_path / "none")


def test_corrupted_pdf_is_reported_not_crashing(tmp_path):
    """A broken file must be reported, not raise a raw traceback."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is definitely not a pdf")
    with pytest.raises(DataDirectoryError):
        load_pdfs(tmp_path)


def test_unreadable_pdf_reports_failure_reason(tmp_path):
    """A valid PDF plus a corrupt sibling: loading succeeds, corruption is recorded."""
    _write_text_pdf(tmp_path / "valid.pdf", "Attendance requires seventy five percent presence.")

    (tmp_path / "corrupt.pdf").write_bytes(b"%PDF-1.4 broken")
    pages, report = load_pdfs(tmp_path, min_chars_per_page=1)
    assert "corrupt.pdf" in report.files_failed
    assert "corrupt.pdf" in report.summary()
    assert pages, "the valid document should still load"


# --- 2. Text splitting -----------------------------------------------------
def test_split_pages_creates_chunks_with_metadata(sample_pages):
    documents = split_pages(sample_pages, chunk_size=200, chunk_overlap=20)
    assert len(documents) > 0
    for doc in documents:
        assert doc.page_content.strip()
        assert doc.metadata["source"] == "Test_Document.pdf"
        assert doc.metadata["page"] == 1


def test_split_pages_respects_chunk_size(sample_pages):
    documents = split_pages(sample_pages, chunk_size=150, chunk_overlap=0)
    # RecursiveCharacterTextSplitter may exceed the limit by a little on a final
    # chunk; the assertion guards against wholesale ignoring of the setting.
    assert max(len(d.page_content) for d in documents) <= 200


def test_smaller_chunks_produce_more_documents(sample_pages):
    big = split_pages(sample_pages, chunk_size=800, chunk_overlap=50)
    small = split_pages(sample_pages, chunk_size=100, chunk_overlap=10)
    assert len(small) >= len(big)


def test_splitter_rejects_overlap_larger_than_chunk():
    with pytest.raises(SplitterConfigError, match="smaller than CHUNK_SIZE"):
        build_splitter(chunk_size=100, chunk_overlap=150)


def test_splitter_rejects_invalid_values():
    with pytest.raises(SplitterConfigError, match="positive"):
        build_splitter(chunk_size=0)
    with pytest.raises(SplitterConfigError, match=">= 0"):
        build_splitter(chunk_size=100, chunk_overlap=-1)


# --- 3. Embeddings ---------------------------------------------------------
def test_embeddings_produce_normalised_vectors():
    embeddings = get_embeddings()
    vector = embeddings.embed_query("attendance requirement")
    assert isinstance(vector, list)
    assert len(vector) == 384, "all-MiniLM-L6-v2 should give 384 dimensions"
    magnitude = sum(value ** 2 for value in vector) ** 0.5
    assert abs(magnitude - 1.0) < 1e-3, "embeddings should be unit-normalised"


def test_similar_text_scores_higher_than_unrelated():
    embeddings = get_embeddings()
    query = embeddings.embed_query("minimum attendance percentage")
    same = embeddings.embed_query("students need 75 percent attendance")
    different = embeddings.embed_query("the best recipe for chocolate cake")

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cosine(query, same) > cosine(query, different)


def test_embeddings_are_cached():
    assert get_embeddings() is get_embeddings()


# --- 4. FAISS creation -----------------------------------------------------
def test_build_and_load_index(index_available, tmp_path):
    store = build_index(DATA_DIR, tmp_path, verbose=False)
    assert store.index.ntotal > 0
    assert (tmp_path / INDEX_FILE).exists()
    reloaded = load_index(tmp_path, embeddings=get_embeddings())
    assert reloaded.index.ntotal == store.index.ntotal


def test_load_index_missing_raises_with_instructions(tmp_path):
    with pytest.raises(IndexNotFoundError, match="--build"):
        load_index(tmp_path / "empty_dir", embeddings=get_embeddings())


# --- 5. Retrieval ----------------------------------------------------------
@pytest.fixture(scope="module")
def retriever(request):
    from src.config import FAISS_DIR as default_dir

    if not (default_dir / INDEX_FILE).exists():
        pytest.skip("No FAISS index built. Run: python -m src.rag.vector_store --build")
    return CollegeRetriever(load_index(default_dir, embeddings=get_embeddings()))


def test_relevant_question_returns_matching_chunk(retriever):
    hits = retriever.retrieve("What is the minimum attendance requirement?")
    assert len(hits) > 0, "should retrieve at least one chunk"
    document, score = hits[0]
    assert "attendance" in document.page_content.lower()
    assert 0 <= score <= 1
    assert document.metadata["source"].endswith(".pdf")


def test_irrelevant_question_returns_no_chunks(retriever):
    hits = retriever.retrieve("What is the capital city of Japan?")
    assert hits == [], "unrelated questions must return nothing"


def test_retrieval_respects_top_k(retriever):
    assert len(retriever.retrieve("attendance marks examination", top_k=2)) <= 2


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_format_context_includes_source_and_page(retriever):
    hits = retriever.retrieve("attendance requirement")
    context = retriever.format_context(hits)
    assert "Chunk 1" in context
    assert "Source:" in context
    assert ".pdf" in context


def test_format_sources_dedupes(retriever):
    hits = retriever.retrieve("attendance")
    sources = retriever.format_sources(hits)
    keys = [(s["source"], s["page"]) for s in sources]
    assert len(keys) == len(set(keys))


def test_format_context_handles_empty(retriever):
    assert retriever.format_context([]) == ""