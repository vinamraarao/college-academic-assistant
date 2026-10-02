# AI College Academic Assistant

A retrieval-augmented assistant that answers questions **only** from your college's
own PDF documents, cites the file and page it used, refuses when the answer is not
there, and builds personalised study plans.

Everything runs locally. No paid services, no cloud LLM, no external database.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Technology stack](#4-technology-stack)
5. [Folder structure](#5-folder-structure)
6. [Python installation](#6-python-installation)
7. [Virtual environment setup](#7-virtual-environment-setup)
8. [Package installation](#8-package-installation)
9. [Ollama installation](#9-ollama-installation)
10. [Ollama model setup](#10-ollama-model-setup)
11. [PDF placement](#11-pdf-placement)
12. [Building the FAISS index](#12-building-the-faiss-index)
13. [Running the Streamlit application](#13-running-the-streamlit-application)
14. [Running tests](#14-running-tests)
15. [Running the Google Colab notebook](#15-running-the-google-colab-notebook)
16. [Git/GitHub instructions](#16-gitgithub-instructions)
17. [Example questions](#17-example-questions)
18. [Known limitations](#18-known-limitations)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. Project overview

A student asks a college question in plain language. The system searches the college
PDFs, finds the relevant passages, and asks a local LLM to answer using only those
passages. The answer comes back with the source file and page.

The same assistant also builds study plans: the student names their subjects, exam
date and available hours, and gets a day-by-day schedule that skips the days they
said they cannot study.

**The central design rule:** the assistant never invents college information. If the
documents do not contain the answer, it says so. See
[docs/LLM_VS_RAG.md](docs/LLM_VS_RAG.md) for the measured difference this makes.

> ### About the documents currently in `data/`
>
> `data/` currently holds **six PDFs** from NMAMIT (NMAMIT Institute of Technology):
> `01_Academic_Regulations_NMAMIT.pdf`, `02_Examination_Guidelines_NMAMIT.pdf`,
> `03_Internship_Guidelines_NMAMIT.pdf`, `04_Syllabus_CSE_NMAMIT.pdf`,
> `05_Student_FAQs_NMAMIT.pdf`, `06_Academic_Calendar_2026-27_NMAMIT.pdf`.
>
> These are the actual college documents for NMAMIT's B.Tech. CSE programme, covering
> academic regulations, examination guidelines, internship requirements, the full CSE
> syllabus with semester-wise subjects and credits, the student FAQ, and the academic
> calendar. All content is verified from the official NMAMIT/Nitte sources linked in the
> PDFs themselves.
>
> **Never hand-edit these PDFs.** Rebuild the FAISS index after any change:
>
> ```bash
> python -m src.rag.vector_store --build
> 
> # clear the sample documents
> rm data/*.pdf                     # PowerShell: Remove-Item data\*.pdf
>
> # add the real college PDFs
> cp ~/Downloads/Academic_Regulations.pdf data/
> cp ~/Downloads/Examination_Guidelines.pdf data/
>
> # rebuild so the index matches
> python -m src.rag.vector_store --build
> ```
>
> Earlier placeholder documents were moved to `archive/placeholder_pdfs/` rather than
> deleted, so they stay out of the index but are recoverable.

---

## 2. Features

| Feature | Behaviour |
|---|---|
| **Grounded Q&A** | Answers only from retrieved PDF passages |
| **Citations** | Every grounded answer shows `filename, page N` from real metadata |
| **Refusal** | "I could not find this information in the available college documents." |
| **Unknown-question handling** | Out-of-scope questions (e.g. "capital of Japan") are refused, not answered |
| **Follow-up questions** | "What happens if I don't meet it?" resolves "it" from history |
| **Study planner** | Subjects + exam date + daily hours → day-by-day schedule |
| **Plan modification** | "I cannot study on Sunday" updates the existing plan, keeping subjects |
| **Unavailable days** | Blocked days are removed from the schedule |
| **Calculator tool** | Arithmetic, percentages and study-hour splits |
| **Multi-PDF** | Any number of PDFs in `data/`, including subfolders |
| **Poor-extraction detection** | Scanned/blank pages are reported, not silently ignored |
| **Error messages** | Setup problems explained in plain language, not stack traces |

---

## 3. Architecture

### RAG pipeline

```
PDF files in data/
   │
   ▼ discover_pdfs()      find every *.pdf
   ▼ load_pdfs()          pypdf text extraction, one record per page,
   │                       source + page metadata kept
   ▼ split_pages()        RecursiveCharacterTextSplitter (overlapping chunks)
   ▼ get_embeddings()     all-MiniLM-L6-v2, normalised
   ▼ FAISS.from_documents()  cosine distance
   ▼ save_local()         storage/faiss/college_index
   │
   ▼ CollegeRetriever.retrieve()   similarity search + relevance floor
   ▼ format_context()    numbered blocks tagged with file and page
   ▼ GROUNDED_PROMPT      context + question → Ollama
   ▼ answer + citations
```

### LangGraph workflow

```
                        ┌──────────────► collect_info ──► generate_plan ──┐
                        │  PLAN_CREATE                                  │
   user message ──► route ─┼─► collect_info ──► modify_plan ──────────────┼─► validate ──► END
                        │  PLAN_MODIFY                                  │
                        │                                               │
                        └─ ACADEMIC ──► retrieve ──► generate_answer ───┘
```

- **route** — rule-based. Tested with an LLM classifier and it misrouted in both
  directions, so the decision is made with deterministic patterns instead
  (see `src/graph/nodes.py`).
- **retrieve** — FAISS search with a relevance floor.
- **generate_answer** — grounded answer, or refusal when retrieval is empty.
  **The LLM is not called at all when nothing relevant is found.**
- **collect_info → generate_plan / modify_plan** — schedule is computed
  deterministically in Python; the LLM only writes the summary.
- **validate_plan** — rejects a plan that schedules time on an unavailable day.

### Where Ollama runs

| Component | Colab | Local |
|---|:---:|:---:|
| PDF loading, extraction, chunking | ✅ | ✅ |
| Embeddings + FAISS retrieval | ✅ | ✅ |
| Ollama LLM generation | ❌ | ✅ |
| LangGraph + Streamlit | ❌ | ✅ |

Ollama is a local application and **does not run in Colab**. The notebook therefore
demonstrates retrieval only, which is the part worth experimenting with. Generation is
demonstrated locally.

---

## 4. Technology stack

| Layer | Library | Version used |
|---|---|---|
| Language | Python | 3.12 (3.11+ works) |
| PDF parsing | pypdf | 4.x+ |
| Chunking | langchain-text-splitters | 1.1.x |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 6.1.x |
| Vector store | faiss-cpu | 1.x |
| LLM serving | Ollama (`qwen2.5:1.5b`) | 0.35 |
| LLM integration | langchain-ollama | 1.1.x |
| Orchestration | langgraph | 1.2.x |
| UI | streamlit | 1.64.x |
| Config | python-dotenv | 1.x |
| Tests | pytest | 8.x |

**Not used, deliberately:** n8n, MongoDB, React, any paid or cloud service.

> **Dependency note:** `langchain-community` prints a deprecation warning — it is being
> sunset upstream, but FAISS support still lives there. If it is removed in future, the
> swap is confined to `src/rag/vector_store.py` and `src/rag/retriever.py`.

---

## 5. Folder structure

```
college-ai-assistant/
├── app.py                     Streamlit UI
├── requirements.txt
├── .env.example               copy to .env
├── .gitignore
│
├── data/                      ← put your college PDFs here
│   └── *_DUMMY.pdf            6 sample docs; replace with the real ones
│
├── notebooks/
│   └── rag_setup_and_testing.ipynb   Colab: retrieval experimentation
│
├── src/
│   ├── config.py              all settings, .env driven
│   ├── rag/
│   │   ├── loader.py          PDF discovery, extraction, quality report
│   │   ├── splitter.py        chunking with validation
│   │   ├── embeddings.py      Sentence-Transformers model
│   │   ├── vector_store.py    FAISS build / load
│   │   └── retriever.py       search + relevance floor
│   ├── llm/
│   │   ├── model.py           Ollama client + setup errors
│   │   └── prompts.py         all prompt templates
│   ├── graph/
│   │   ├── state.py           graph state
│   │   ├── nodes.py           routing, retrieval, answering, planning
│   │   ├── workflow.py        graph wiring
│   │   ├── planner.py         deterministic scheduling
│   │   └── planner_parser.py  free-text constraint extraction
│   ├── tools/
│   │   └── calculator.py      calculator tool (arithmetic, %, hours)
│   └── utils/
│       └── helpers.py         history trimming, shared errors
│
├── scripts/
│   ├── demo.py                 one-command acceptance demo
│   ├── compare_llm_vs_rag.py   direct-LLM vs RAG comparison
│   ├── make_placeholder_pdfs.py
│   ├── verify_notebook.py
│   └── fix_notebook_source.py
│
├── archive/
│   └── placeholder_pdfs/       earlier sample docs, kept out of the index
│
├── docs/
│   └── LLM_VS_RAG.md          measured comparison
│
├── storage/faiss/             generated index (gitignored)
└── tests/
    ├── conftest.py
    ├── pdf_fixtures.py
    ├── test_rag.py             24 tests - pipeline
    ├── test_knowledge_base.py  24 tests - knowledge-base behaviour, citations
    ├── test_study_planner.py   32 tests - planner, routing, calculator
    ├── test_end_to_end.py      14 tests - LLM flows (need Ollama)
    └── test_cases.md           manual demo checklist
```

---

## 6. Python installation

Check your version:

```bash
python --version
```

Requires **3.11 or newer**. Verified on 3.12.10.

---

## 7. Virtual environment setup

```bash
python -m venv .venv
```

**Windows**

```bat
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Your prompt should show `(.venv)`.

---

## 8. Package installation

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch and sentence-transformers, so expect a few minutes and ~2 GB
of disk on first run.

Then create your config:

```bash
cp .env.example .env        # Windows PowerShell: copy .env.example .env
```

---

## 9. Ollama installation

1. Download from **https://ollama.com/download** (Windows, macOS or Linux).
2. Install and launch it.
   - **Windows:** start "Ollama" from the Start menu. It runs in the background.
   - **macOS/Linux:** run `ollama serve` in a terminal.

Verify:

```bash
ollama --version
curl http://localhost:11434/api/tags
```

An empty `{"models":[]}` means the server is running and no models are installed yet.

---

## 10. Ollama model setup

Download the model (~1 GB):

```bash
ollama pull qwen2.5:1.5b
```

Confirm:

```bash
ollama list
```

Test it directly before involving Python:

```bash
ollama run qwen2.5:1.5b "Reply with exactly: model ready"
```

Expect `model ready`.

To use a different model, set it in `.env` — nothing is hardcoded:

```env
OLLAMA_MODEL=qwen2.5:3b
```

Then re-pull it: `ollama pull qwen2.5:3b`. If you have the RAM, `qwen2.5:3b` or `7b`
give noticeably better answers; `1.5b` runs comfortably on a student laptop.

---

## 11. PDF placement

Put your college PDFs in `data/` — any number of files, in subfolders if you prefer:

```bash
cp ~/Downloads/Academic_Regulations.pdf data/
cp ~/Downloads/Examination_Guidelines.pdf data/
cp ~/Downloads/Academic_Calendar.pdf data/2026/
```

`data/2026/Academic_Calendar.pdf` is discovered automatically. Every PDF found
becomes part of the knowledge base.

**Currently in `data/`:**

| File | Covers |
|---|---|
| `Academic_Calendar_DUMMY.pdf` | Semester dates, internal assessment dates |
| `Academic_Regulations_DUMMY.pdf` | Attendance, credits, grading, backlog |
| `Examination_Guidelines_DUMMY.pdf` | Eligibility, registration, passing, malpractice |
| `Internship_Guidelines_DUMMY.pdf` | Eligibility, duration, approval, documents, evaluation |
| `Student_FAQ_DUMMY.pdf` | Common questions |
| `Syllabus_DUMMY.pdf` | B.Tech CSE Semester 3 subjects and units |

All six are sample documents. Replace them with the real college PDFs and rebuild.

Check what the system sees:

```bash
python -c "from src.rag.loader import discover_pdfs; [print(p.name) for p in discover_pdfs()]"
```

> **Scanned PDFs:** if pages are images rather than text, extraction returns nothing.
> The loader reports these explicitly ("no extractable text (likely scanned)") rather
> than returning bad data. OCR them first:
>
> ```bash
> ocrmypdf data/scanned.pdf data/scanned_ocr.pdf
> ```
>
> Not currently installed on this machine. `ocrmypdf` bundles Tesseract; on Windows it
> can also be installed via `winget install UB-Mannheim.TesseractOCR`. This was needed
> for phone-scanner PDFs that yield zero extractable characters.

---

## 12. Building the FAISS index

```bash
python -m src.rag.vector_store --build
```

Output:

```
[1/4] Loading PDFs from ...\data ...
      Loaded 6 PDF(s), 6 page(s) of text.
[2/4] Splitting into chunks (size=800, overlap=120) ...
[3/4] Embedding 8 chunks with sentence-transformers/all-MiniLM-L6-v2 ...
[4/4] Saved index (8 vectors) to ...\storage\faiss\college_index
Indexed 8 chunks.
```

The first build downloads the embedding model (~90 MB).

Check it loads:

```bash
python -m src.rag.vector_store
```

**Rebuild whenever the PDFs change.** `test_all_pdfs_are_indexed` fails if the index
and `data/` disagree, which catches a forgotten rebuild.

Override settings inline if needed:

```bash
python -m src.rag.vector_store --build --chunk-size 1000 --chunk-overlap 150
```

### Retrieval tuning

Retrieval applies **two gates** before anything reaches the model:

| Setting | Default | Effect |
|---|---|---|
| `TOP_K` | 4 | Maximum chunks retrieved per question |
| `SCORE_THRESHOLD` | 0.25 | Relative floor — drops weak neighbours of a good hit |
| `MIN_SCORE` | 0.236 | Absolute floor — if the *best* match is weaker, the question is unanswerable |

`MIN_SCORE` is what stops a noise-only match from reaching the prompt. On the current
corpus, in-scope questions score 0.24+ and out-of-scope noise scores <=0.233,
sample corpus, in-scope questions score **0.27–0.55** and out-of-scope noise **<0.03**,
so 0.236 separates them cleanly.

> **Re-measure after swapping in the real PDFs.** That gap is specific to these
> documents. Use the Colab notebook (section 9) which reports a score per question, or:
>
> ```bash
> python -c "
> from src.rag.retriever import get_retriever
> r = get_retriever()
> for q in ['What is the attendance requirement?', 'What is the capital of Japan?']:
>     h = r.retrieve(q)
>     print(round(h[0][1],3) if h else 0.0, q)
> "
> ```
>
> Raise `MIN_SCORE` to refuse more aggressively, lower it to recall more.

---

## 13. Running the Streamlit application

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. First load takes a few seconds while the embedding
model loads.

The sidebar shows live status:

- 🟢 **Knowledge base: N chunks indexed** — index built and loadable
- 🔴 **Knowledge base not built** — run the `--build` command
- 🟢 **Ollama: qwen2.5:1.5b ready** — model present
- 🔴 **Ollama not reachable** — start Ollama, then reload

**To demonstrate:**
1. Ask *"What is the attendance requirement?"* → grounded answer with `Source: file, page N`
2. Ask *"What happens if I don't meet it?"* → resolves the reference from history
3. Ask *"What is the capital of Japan?"* → refuses
4. Ask *"I have DBMS, OS and Computer Networks. My exams are in 20 days. I can study 3 hours per day."* → study plan table appears
5. Ask *"I cannot study on Sunday."* → plan rebuilds without Sunday

---

## 14. Running tests

```bash
python -m pytest tests/ -v
```

Current status: **94 passed**.

```bash
python -m pytest tests/test_rag.py -v             # pipeline: load, split, embed, FAISS, retrieve
python -m pytest tests/test_knowledge_base.py -v  # knowledge-base behaviour and citations
python -m pytest tests/test_study_planner.py -v   # planner, routing, calculator
python -m pytest tests/test_end_to_end.py -v      # LLM flows (needs Ollama)
```

Tests that need Ollama **skip automatically** when it is unavailable, so the suite
passes on a clean machine. Manual demo scenarios are in
[tests/test_cases.md](tests/test_cases.md).

The comparison script doubles as a demo:

```bash
python -m scripts.compare_llm_vs_rag
```

### One-command acceptance demo

Runs every acceptance criterion and prints a pass/fail table — useful as a
final check before a demo or submission:

```bash
python -m scripts.demo
```

```
--- RAG PIPELINE ---
[PASS] PDFs loaded with page metadata
[PASS] FAISS index built
[PASS] Relevant chunks retrieved
--- GROUNDED ANSWERS ---
[PASS] College question answered
[PASS] Source/page shown
--- UNKNOWN INFORMATION ---
[PASS] Out-of-scope question refused
[PASS] No fabricated answer
--- FOLLOW-UP QUESTIONS ---
[PASS] Follow-up resolved from history
--- STUDY PLANNER ---
[PASS] Plan generated
[PASS] Plan modified
[PASS] Previous state preserved
[PASS] No Sunday session
--- CALCULATOR TOOL ---
[PASS] Arithmetic
[PASS] Code injection blocked

RESULT: 16/16 checks passed
```

Requires a built index and a running Ollama with the model installed.

---

## 15. Running the Google Colab notebook

1. Open Google Colab → **File → Upload notebook**
2. Upload `notebooks/rag_setup_and_testing.ipynb`
3. Upload your PDFs into a `data/` folder (or let cell 6 generate placeholders)
4. **Runtime → Run all**

It covers: install → load → extract → split → embed → build FAISS → search → inspect
retrieved chunks → test questions → score chart → chunk-size tuning → download index.

**The notebook does not use Ollama** — Ollama cannot run in Colab. It validates the
retrieval half; generation is demonstrated locally.

---

## 16. Git/GitHub instructions

```bash
git init
git add .
git commit -m "AI College Academic Assistant - RAG, LangGraph study planner, Streamlit UI"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `storage/faiss/`, `__pycache__/`, `.venv/` and
caches, so no secrets or large binaries are committed.

**FAISS indexes are not in the repository.** They are regenerated in one command on
each machine:

```bash
python -m src.rag.vector_store --build
```

Add this to the README when someone clones:

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:1.5b
python -m src.rag.vector_store --build
streamlit run app.py
```

---

## 17. Example questions

**Academic Q&A (grounded)**

- What is the attendance requirement?
- How do I register for an examination?
- What happens if I fail a course?
- What documents are required for an internship?
- How is an internship evaluated?
- When do the internal assessments take place?
- What are the credits for Database Management Systems?

**Follow-ups (need a previous question)**

- What documents are required for it?
- What units does it have?
- What happens if I don't meet it?

**Expected refusal** (none of these are in the documents)

- What is the capital of Japan?
- What is the hostel mess fee?
- Who is the hostel warden?
- What is the library fine amount?
- What is the placement percentage last year?

**Study planning** (subjects taken from the sample syllabus)

- I have Data Structures, Database Management Systems, Operating Systems and Computer
  Organization. My exams are in 20 days. I can study 3 hours per day.

**Plan modification (after a plan exists)**

- I cannot study on Sunday.
- I can now study 4 hours per day.

---

## 18. Known limitations

1. **`MIN_SCORE` is tuned for the current corpus.** The 0.236 floor separates cleanly for
   the NMAMIT PDFs (in-scope 0.24+, noise <=0.233). Re-measure after any change to
   `data/` — see the tuning notes in section 12.
   college PDFs are added. Every answer changes accordingly. Nothing is hardcoded.
2. **Scanned PDFs are not OCR'd.** The system detects them and reports them by name
   rather than returning bad data, but you must OCR them first (see section 11). OCR
   tooling is not currently installed on this machine.
4. **Cosine similarity is not verification.** Retrieval matches text similarity; a chunk
   can match vocabulary without answering the question. The two relevance gates reduce
   but do not eliminate this.
5. **Small-model limitations.** `qwen2.5:1.5b` occasionally produces a weak plan summary
   or drops nuance. Guards reject clearly defective summaries; `qwen2.5:3b` or `7b`
   improve quality if your machine allows.
6. **Answers can be slow on first use.** The embedding model loads once per process.
   Query rewriting is only invoked for follow-ups containing a pronoun or a very short
   elliptical phrase, so standalone questions skip the extra LLM round-trip.
7. **English only.** No multilingual handling.
8. **Single-session conversation.** History lives in the Streamlit session and is not
   persisted between browser sessions.
9. **Plan parsing is regex-based.** It handles common phrasings, not arbitrary sentences.
   Unparsed input produces a request for the missing details rather than a wrong guess.
10. **No access control.** The app is local-only and assumes a trusted user.
11. **No re-ranking stage.** FAISS scores are used directly; a cross-encoder reranker
    would improve precision but adds a dependency.
12. **Extraction artifacts.** Some PDFs contain en-dashes or bullets that pypdf renders
    oddly (e.g. `Academic Calendar � DUMMY DATA`). Content and retrieval are unaffected,
    but exact-string matching against the raw PDF may differ.

---

## 19. Troubleshooting

**"No FAISS index at ..."**
```bash
python -m src.rag.vector_store --build
```

**"Knowledge-base folder not found"**
The `data/` directory is missing. Create it and add PDFs.

**"No PDF files found in ..."**
`data/` is empty. Add your college PDFs and rebuild.

**"No usable text could be extracted" / "likely scanned"**
The PDFs contain images, not text. Run OCR:
```bash
ocrmypdf input.pdf data/output_ocr.pdf
```

**"Cannot reach Ollama"**
Ollama is not running. Start it (`ollama serve`, or launch from the Start menu on
Windows), then reload the Streamlit page.

**"Model 'qwen2.5:1.5b' is not installed"**
```bash
ollama pull qwen2.5:1.5b
```

**Answers say the information was not found**
Expected when the documents do not cover the question. Check the sidebar chunk count —
if it is 0, the index is empty. To confirm retrieval is working:
```bash
python -c "from src.rag.retriever import get_retriever; print(len(get_retriever().retrieve('attendance')))"
```

**Retrieval seems weak**
Try different chunk settings in `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
`SCORE_THRESHOLD`) and rebuild. Experiment in the Colab notebook first — cell 10 compares
chunk sizes for you.

**Streamlit shows an error but the terminal is silent**
Run `streamlit run app.py --logger.level debug`.

**Tests skip the end-to-end file**
Expected: Ollama is not running or the model is missing. Start Ollama and re-run.

**FAISS AVX2 warning**
`Could not load library with AVX2 support` is harmless — it falls back to standard FAISS
automatically.