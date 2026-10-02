"""Demonstrate the difference between a bare LLM and the RAG pipeline.

Sends the same questions through:
    A) Ollama alone        - general knowledge, no documents
    B) Ollama + FAISS      - the PDFs in data/ retrieved as context

Questions are drawn from the documents currently in data/, so the comparison
reflects the real knowledge base rather than invented examples. Replace the PDFs
in data/ and re-run to compare against the real college documents.

Run:  python -m scripts.compare_llm_vs_rag
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.model import get_llm, invoke  # noqa: E402
from src.llm.prompts import DIRECT_LLM_PROMPT, GROUNDED_PROMPT, NO_CONTEXT_ANSWER  # noqa: E402
from src.rag.retriever import get_retriever  # noqa: E402

QUESTIONS = [
    "What is the minimum attendance requirement?",
    "What documents are required for an internship?",
    "How is an internship evaluated?",
    "When do the internal assessments take place?",
    "What is the capital of Japan?",
    "What is the hostel mess fee?",
]


def main() -> int:
    try:
        llm = get_llm()
    except Exception as exc:
        print(exc)
        return 1

    try:
        retriever = get_retriever()
    except Exception as exc:
        print(f"\nCould not open the knowledge base: {exc}")
        return 1

    for question in QUESTIONS:
        print("=" * 78)
        print(f"QUESTION: {question}")

        print("\n--- A) DIRECT LLM (no documents) " + "-" * 40)
        try:
            messages = DIRECT_LLM_PROMPT.format_messages(question=question)
            print(invoke(llm, messages)[:700])
        except Exception as exc:
            print(f"Error: {exc}")

        print("\n--- B) RAG (retrieved from college PDFs) " + "-" * 31)
        try:
            hits = retriever.retrieve(question)
            if not hits:
                print(NO_CONTEXT_ANSWER)
                print("[No relevant chunk found - the LLM was not called.]")
            else:
                context = retriever.format_context(hits)
                sources = ", ".join(
                    f"{s['source']} p{s.get('page')}" for s in retriever.format_sources(hits)
                )
                messages = GROUNDED_PROMPT.format_messages(
                    context=context, history="(none)", question=question
                )
                print(invoke(llm, messages)[:700])
                print(f"[Sources: {sources}]")
        except Exception as exc:
            print(f"Error: {exc}")
        print()

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())