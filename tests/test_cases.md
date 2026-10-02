# Test Cases

Automated coverage lives in `test_rag.py`, `test_study_planner.py` and
`test_end_to_end.py` (63 tests, all passing). This file records the manual
scenarios to walk through during a demo, including cases that are awkward to
assert automatically.

Run everything with:

```bash
python -m pytest tests/ -v
```

---

## A. RAG pipeline

| # | Scenario | Expected result | Covered by |
|---|----------|-----------------|------------|
| 1 | PDFs present in `data/` | All files discovered and loaded | `test_discover_pdfs_finds_data_files` |
| 2 | `data/` folder missing | `DataDirectoryError` with a clear message | `test_discover_pdfs_missing_directory_raises` |
| 3 | `data/` folder empty | `DataDirectoryError` naming the folder | `test_discover_pdfs_empty_directory_raises` |
| 4 | Corrupt PDF alongside a good one | Loads the good file, reports the bad one by name | `test_unreadable_pdf_reports_failure_reason` |
| 5 | Only corrupt PDFs | Raises with the reason, never a raw traceback | `test_corrupted_pdf_is_reported_not_crashing` |
| 6 | Text extraction | Every page yields text + `source` + `page` | `test_load_pdfs_extracts_text_with_page_metadata` |
| 7 | Chunking | Chunks respect size and keep metadata | `test_split_pages_creates_chunks_with_metadata` |
| 8 | Overlap >= chunk size | `SplitterConfigError` | `test_splitter_rejects_overlap_larger_than_chunk` |
| 9 | Embeddings | 384-dim, unit-normalised, cached | `test_embeddings_produce_normalised_vectors` |
| 10 | Embedding semantics | Related text scores above unrelated | `test_similar_text_scores_higher_than_unrelated` |
| 11 | Index build + reload | Same vector count after reload | `test_build_and_load_index` |
| 12 | Index missing | `IndexNotFoundError` with the build command | `test_load_index_missing_raises_with_instructions` |
| 13 | Relevant question | Returns a matching chunk with a real page number | `test_relevant_question_returns_matching_chunk` |
| 14 | Irrelevant question | Returns **zero** chunks | `test_irrelevant_question_returns_no_chunks` |

## B. Conversation behaviour

| # | Scenario | Expected result | Covered by |
|---|----------|-----------------|------------|
| 15 | "What is the attendance requirement?" | Grounded answer, `grounded=True`, sources present | `test_academic_question_is_grounded` |
| 16 | "What happens if I don't meet it?" (after #15) | Resolves "it" to attendance; still grounded | `test_followup_question_resolves_reference` |
| 17 | "What is the capital of Japan?" | Refuses; does **not** say Tokyo | `test_unknown_question_is_refused` |
| 18 | College topic absent from the PDFs | Refuses or answers only from what exists | `test_unknown_college_question_is_refused` |

## C. Study planner

| # | Scenario | Expected result | Covered by |
|---|----------|-----------------|------------|
| 19 | 3 subjects, 20 days, 3 h/day | Plan created, all subjects scheduled | `test_study_plan_generated` |
| 20 | 3 h/day available | No day exceeds that | `test_plan_respects_hours_per_day` |
| 21 | "I cannot study on Sunday." | No Sunday in the schedule | `test_plan_excludes_unavailable_days` |
| 22 | Blocking a day | Fewer sessions and fewer hours | `test_excluding_a_day_reduces_sessions` |
| 23 | Subject balance | Times within 2 h of each other | `test_plan_covers_roughly_equal_time_per_subject` |
| 24 | Modification keeps prior state | Subjects preserved, Sunday removed | `test_study_plan_modification_keeps_previous_state` |
| 25 | No subjects | Error naming the missing input | `test_no_subjects_raises` |
| 26 | 0 days / 0 hours | Validation errors, no crash | `test_zero_days_raises`, `test_zero_hours_raises` |
| 27 | Every day blocked | Error explaining no time remains | `test_all_days_unavailable_raises` |
| 28 | 6 subjects, 1 slot | "Not enough study slots" error | `test_too_few_slots_raises_clear_error` |
| 29 | "Make me a study plan" alone | Asks for the missing details | `test_routing` |
| 30 | Free-text parsing | Subjects/days/hours extracted from a sentence | `test_parse_requirement_example` |

## D. Routing

| # | Message | Expected branch |
|---|---------|-----------------|
| 31 | "What is the attendance requirement?" | ACADEMIC |
| 32 | "What is the capital of Japan?" | ACADEMIC |
| 33 | "How many marks do I need to pass?" | ACADEMIC |
| 34 | "Make me a study plan" | PLAN_CREATE |
| 35 | Full constraint sentence | PLAN_CREATE |
| 36 | "I cannot study on Sunday." with an active plan | PLAN_MODIFY |

Covered by `test_routing` and `test_modification_routed_when_plan_exists`.

## E. Calculator tool

| # | Scenario | Expected result | Covered by |
|---|----------|-----------------|------------|
| 37 | "2 + 3 * 4" | 14 (correct precedence) | `test_calculator_arithmetic` |
| 38 | "(10 + 5) / 3" | 5.0 | `test_calculator_arithmetic` |
| 39 | "2 ^ 10" | 1024 | `test_calculator_arithmetic` |
| 40 | "50%" | 0.5 | `test_calculator_percentage_operator` |
| 41 | `__import__('os').system('ls')` | Rejected — no code execution | `test_calculator_rejects_code_injection` |
| 42 | "5 / 0" | "Division by zero" error | `test_calculator_division_by_zero` |
| 43 | Empty string | Clear error | `test_calculator_rejects_empty` |

## F. Streamlit

| # | Scenario | Expected result | Covered by |
|---|----------|-----------------|------------|
| 44 | `app.py` imports cleanly | No syntax or import errors | `test_streamlit_app_imports` |
| 45 | App starts | Serves HTTP 200 on the chosen port | manual (see below) |
| 46 | Grounded answer shown | "Answered from your college documents" + page citation | manual |

### Manual startup check

```bash
python -m streamlit run app.py --server.port 8511
```

Then confirm:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8511
```

Expected: `200`, with the sidebar showing a green "Knowledge base" and a green
"Ollama" status.

---

## G. Error handling (manual)

Stop Ollama, then ask a question. Expected: a clear message that Ollama cannot
be reached, not a stack trace. The sidebar should show "Ollama not reachable".

Delete `storage/faiss/`, then reload the app. Expected: "Knowledge base not
built" plus the exact `--build` command to fix it.

Rename `data/` temporarily. Expected: a `DataDirectoryError` explaining the
folder is missing, not a `FileNotFoundError`.

## H. LLM vs RAG comparison

Run `scripts/compare_llm_vs_rag.py`, which sends the same questions twice — once
to the bare LLM, once through retrieval. See `docs/LLM_VS_RAG.md`.