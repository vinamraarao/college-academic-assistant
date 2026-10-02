"""Knowledge-base behaviour tests against whatever PDFs are in data/.

These tests deliberately assert *retrieval and grounding behaviour* rather than
exact answer wording, so they stay valid when the college PDFs are replaced.
They are skipped automatically when no index has been built.
"""

from __future__ import annotations

import pytest

from src.config import DATA_DIR, FAISS_DIR
from src.rag.loader import discover_pdfs, load_pdfs
from src.rag.retriever import get_retriever
from src.rag.splitter import split_pages
from src.rag.vector_store import INDEX_FILE, load_index

pytestmark = pytest.mark.skipif(
    not (FAISS_DIR / INDEX_FILE).exists(),
    reason="No FAISS index built. Run: python -m src.rag.vector_store --build",
)


@pytest.fixture(scope="module")
def retriever():
    return get_retriever(top_k=3)


@pytest.fixture(scope="module")
def kb_sources() -> set[str]:
    """Filenames currently in the knowledge base."""
    return {store_doc.metadata["source"] for store_doc in
            load_index().docstore._dict.values()}


# --- Knowledge-base composition ------------------------------------------
def test_all_pdfs_are_indexed():
    """Every PDF in data/ must be represented in the index."""
    on_disk = {p.name for p in discover_pdfs(DATA_DIR)}
    indexed = {d.metadata["source"] for d in load_index().docstore._dict.values()}
    assert on_disk == indexed, (
        f"Index is out of date. On disk: {sorted(on_disk)}; "
        f"in index: {sorted(indexed)}. Rebuild with --build"
    )


def test_every_pdf_extracts_text():
    pages, report = load_pdfs(DATA_DIR)
    assert len(pages) > 0
    assert not report.files_failed, f"Unreadable PDFs: {report.files_failed}"


def test_index_chunks_carry_page_metadata():
    for doc in load_index().docstore._dict.values():
        assert doc.metadata.get("source"), "chunk is missing a source filename"
        assert isinstance(doc.metadata.get("page"), int)
        assert doc.metadata["page"] >= 1


# --- In-scope retrieval ----------------------------------------------------
# Each case names the document that should win. Content itself is not asserted,
# so these survive the real college PDFs replacing the samples.
IN_SCOPE = [
    ("What is the attendance requirement?", "attendance"),
    ("How do I register for an examination?", "registration"),
    ("What happens if I fail a course?", "fail"),
    ("What documents are needed for an internship?", "internship"),
    ("How is an internship evaluated?", "internship"),
    ("What is the exam malpractice policy?", "examination"),
    ("When does the semester start?", "semester"),
]


@pytest.mark.parametrize("question,topic", IN_SCOPE)
def test_in_scope_question_retrieves_evidence(retriever, question, topic):
    """An answerable question must return at least one chunk above the floor."""
    hits = retriever.retrieve(question)
    assert hits, f"no evidence retrieved for answerable question: {question!r}"
    assert hits[0][1] >= retriever.min_score


def test_in_scope_questions_hit_expected_documents(retriever):
    """Each question should surface the document that actually contains it."""
    expected = [
        ("What is the attendance requirement?", "01_Academic_Regulations_NMAMIT.pdf"),
        ("What documents are required for an internship?", "03_Internship_Guidelines_NMAMIT.pdf"),
        ("How do I register for an examination?", "02_Examination_Guidelines_NMAMIT.pdf"),
        ("When does the semester start?", "06_Academic_Calendar_2026-27_NMAMIT.pdf"),
    ]
    for question, want in expected:
        hits = retriever.retrieve(question)
        assert hits, f"no evidence for {question!r}"
        got = hits[0][0].metadata["source"]
        assert got == want, f"{question!r} retrieved {got}, expected {want}"


def _available() -> set[str]:
    return {d.metadata["source"] for d in load_index().docstore._dict.values()}


# --- Out-of-scope refusal --------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of Japan?",
        "Who won the FIFA World Cup in 1998?",
        "What is the hostel mess fee?",
        "Who is the hostel warden?",
        "What is the library fine amount?",
        "What is the placement percentage last year?",
    ],
)
def test_out_of_scope_question_returns_no_evidence(retriever, question):
    """Unanswerable questions must clear the absolute relevance floor."""
    assert retriever.retrieve(question) == [], (
        f"expected no evidence for {question!r}, "
        f"got {[round(s, 3) for _, s in retriever.retrieve(question)]}"
    )


def test_absolute_floor_blocks_weak_matches(retriever):
    """A very high floor must reject everything, proving the gate works."""
    strict = type(retriever)(retriever.store, top_k=3, min_score=0.99)
    assert strict.retrieve("What is the attendance requirement?") == []


# --- Citations -------------------------------------------------------------
def test_context_tags_each_chunk_with_source_and_page(retriever):
    hits = retriever.retrieve("What is the attendance requirement?")
    assert hits
    context = retriever.format_context(hits)
    assert "[Chunk 1]" in context
    assert "Source:" in context
    for document, _ in hits:
        assert document.metadata["source"] in context


def test_sources_are_unique_per_file_and_page(retriever):
    hits = retriever.retrieve("attendance examination semester internship")
    sources = retriever.format_sources(hits)
    keys = [(s["source"], s["page"]) for s in sources]
    assert len(keys) == len(set(keys)), "duplicate citations emitted"


def test_every_citation_names_a_real_file(retriever, kb_sources):
    hits = retriever.retrieve("What is the attendance requirement?")
    for source in retriever.format_sources(hits):
        assert source["source"] in kb_sources
        assert source["source"].lower().endswith(".pdf")


# --- Query rewriting cost --------------------------------------------------
def test_standalone_question_skips_llm_rewrite():
    """A self-contained question must not pay for a rewrite round-trip."""
    from src.graph.nodes import _rewrite_query

    state = {
        "question": "What is the attendance requirement?",
        "history": [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}],
    }
    assert _rewrite_query(state) == state["question"]


def test_pronoun_question_is_rewritten():
    """A question containing a dangling reference must still be rewritten."""
    from src.graph import nodes

    called = {"n": 0}

    def fake_invoke(llm, prompt):
        called["n"] += 1
        return "What documents are required for an internship?"

    original = nodes.invoke
    nodes.invoke = fake_invoke
    try:
        state = {
            "question": "What documents are required for it?",
            "history": [{"role": "user", "content": "What is internship eligibility?"},
                        {"role": "assistant", "content": "Check eligibility."}],
        }
        assert nodes._rewrite_query(state) == "What documents are required for an internship?"
        assert called["n"] == 1
    finally:
        nodes.invoke = original

# --- Chunking over real content ------------------------------------------
def test_splitting_real_pdfs_produces_documents():
    pages, _ = load_pdfs(DATA_DIR)
    documents = split_pages(pages)
    assert documents
    assert all(doc.page_content.strip() for doc in documents)
