"""End-to-end demonstration of every acceptance criterion, in one run.

Exercises the PDF files currently in data/ (sample/dummy documents until the real
college PDFs are supplied).

Run:  python -m scripts.demo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.workflow import run_assistant  # noqa: E402
from src.rag.loader import discover_pdfs, load_pdfs  # noqa: E402
from src.rag.retriever import get_retriever  # noqa: E402
from src.rag.vector_store import load_index  # noqa: E402
from src.tools.calculator import calculate  # noqa: E402

PASS, FAIL = "[PASS]", "[FAIL]"
results: list[bool] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append(condition)
    print(f"{PASS if condition else FAIL} {label}")
    if detail:
        print(f"       {detail}")


def main() -> int:
    print("=" * 78)
    print("AI COLLEGE ACADEMIC ASSISTANT - ACCEPTANCE DEMO")
    print("Knowledge base: the PDF files currently in data/")
    print("=" * 78)

    # --- 1. PDF loading and index contents ---
    print("\n--- KNOWLEDGE BASE ---")
    pdfs = discover_pdfs()
    pages, report = load_pdfs()
    check("PDFs loaded", len(pages) > 0,
          f"{len(pages)} page(s) from {len(pdfs)} file(s)")

    store = load_index()
    indexed = {d.metadata["source"] for d in store.docstore._dict.values()}
    on_disk = {p.name for p in pdfs}
    check("Index matches data/ folder", indexed == on_disk,
          f"{len(indexed)} document(s) indexed")
    check("No extraction failures", not report.files_failed,
          str(report.files_failed) if report.files_failed else "none")
    for name in sorted(indexed):
        print(f"       - {name}")

    # --- 2. RAG retrieval ---
    print("\n--- RAG RETRIEVAL ---")
    retriever = get_retriever(top_k=3)
    hits = retriever.retrieve("What is the attendance requirement?")
    check("Relevant chunks retrieved", len(hits) > 0,
          f"top score {hits[0][1]:.3f} from {hits[0][0].metadata['source']}" if hits else "none")

    # --- 3. Real college question + citation ---
    print("\n--- GROUNDED ANSWER + CITATION ---")
    result = run_assistant("What is the attendance requirement?")
    check("College question answered", bool(result.get("grounded")),
          (result.get("answer") or "")[:80].replace("\n", " "))
    sources = result.get("sources") or []
    check("Source filename displayed", bool(sources),
          ", ".join(s["source"] for s in sources[:3]))
    check("Page number displayed",
          all(s.get("page") for s in sources) if sources else False,
          f"page(s): {sorted({s.get('page') for s in sources})}" if sources else "")
    check("Citations come from retrieval metadata",
          all(s["source"] in indexed for s in sources))

    # --- 4. Follow-up question ---
    print("\n--- FOLLOW-UP QUESTION ---")
    initial = run_assistant("What is the internship eligibility requirement?")
    history = [
        {"role": "user", "content": "What is the internship eligibility requirement?"},
        {"role": "assistant", "content": (initial.get("answer") or "")[:300]},
    ]
    followup = run_assistant("What documents are required for it?", history=history)
    check("Follow-up resolved 'it'", bool(followup.get("grounded")),
          (followup.get("answer") or "")[:90].replace("\n", " "))
    follow_sources = {s["source"] for s in followup.get("sources", [])}
    check("Follow-up used the right document",
          any("Internship" in s for s in follow_sources), ", ".join(sorted(follow_sources)))

    # --- 5. Unknown question ---
    print("\n--- UNKNOWN QUESTION (must refuse) ---")
    unknown = run_assistant("What is the capital of Japan?")
    answer = (unknown.get("answer") or "").lower()
    check("Out-of-scope question refused", "could not find" in answer)
    check("Not fabricated", "tokyo" not in answer)
    fee = run_assistant("What is the hostel mess fee?")
    fee_answer = (fee.get("answer") or "").lower()
    check("Absent college fact refused",
          fee.get("grounded") is False and "could not find" in fee_answer,
          "no invented figure")

    # --- 6. Study plan generation ---
    print("\n--- STUDY PLANNER ---")
    created = run_assistant(
        "I have Data Structures, Database Management Systems, Operating Systems "
        "and Computer Organization. My exams are in 20 days. "
        "I can study 3 hours per day."
    )
    plan = created.get("plan")
    check("Plan generated", plan is not None,
          f"{plan.total_sessions} sessions, {plan.total_hours}h" if plan else "")
    check("All subjects scheduled",
          bool(plan) and set(plan.subject_hours) == set(plan.subjects),
          str(plan.subject_hours) if plan else "")

    # --- 7. Study plan modification ---
    print("\n--- STUDY PLAN MODIFICATION ---")
    modified = run_assistant("I cannot study on Sunday.",
                             plan=plan, plan_request=created.get("plan_request"))
    updated = modified.get("plan")
    check("Plan modified", bool(updated) and updated.unavailable_days == [6])
    check("Previous state preserved",
          bool(updated) and updated.subjects == plan.subjects)
    check("Unavailable day excluded",
          bool(updated) and "Sunday" not in {d["weekday"] for d in updated.schedule},
          f"{len(plan.schedule)} -> {len(updated.schedule)} study days")

    # --- 8. Calculator tool ---
    print("\n--- CALCULATOR TOOL ---")
    check("Arithmetic", calculate("2 + 3 * 4") == 14, "2 + 3 * 4 = 14")
    check("Percentage", calculate("50%") == 0.5, "50% = 0.5")
    try:
        calculate("__import__('os')")
        check("Code injection blocked", False, "expression executed!")
    except Exception:
        check("Code injection blocked", True)

    print("\n" + "=" * 78)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())