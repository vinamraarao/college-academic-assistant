# Direct LLM vs RAG — why retrieval is needed

Regenerate this comparison at any time:

```bash
python -m scripts.compare_llm_vs_rag
```

**Basis:** every response below is real output produced by that script against the
PDF files currently in `data/`. As of this writing those are six **dummy/sample**
documents (`*_DUMMY.pdf`) that contain invented example rules, not real college
information. The behaviour shown is what happens with *any* set of documents:
replace the PDFs and re-run the script to compare against the real college
documents.

## The two paths

**A. Direct LLM**

```
User question ──▶ Ollama ──▶ Answer
```

The model answers from the weights it was trained on. It has never seen the PDFs
in `data/`.

**B. RAG (this project)**

```
User question ──▶ FAISS retrieval ──▶ Relevant chunks + page metadata
                                        │
                                        ▼
                                      Prompt ──▶ Ollama ──▶ Answer + citations
```

The same model, but the documents are supplied as context at question time and
any answer is cited back to a file and page.

## Observed behaviour

### Question: "What is the minimum attendance requirement?"

**Direct LLM:**

> The minimum attendance requirement varies by institution and academic program.
> Generally, most colleges and universities require a certain percentage of
> classes to be attended... For example, many institutions require at least **75%
> of classes to be attended** to graduate. However, some programs may have higher
> attendance requirements...

It mentions 75%, but as a *guess about colleges in general* — "many institutions
require" — not as this college's rule, and it never commits to a number for this
institution. It also hedges toward 90–100% as possibilities.

**RAG:**

> The minimum attendance requirement for students to be eligible for
> semester-end examinations is at least 75%.
> (Source: `Academic_Regulations_DUMMY.pdf`, page 1)

One number, stated as policy, traceable to a page. The document says 75%; the RAG
answer is 75%.

### Question: "What documents are required for an internship?"

**Direct LLM:**

> 1. **Resume/CV** ... 2. **Cover Letter** ... 3. **Application Form** ...
> 4. **References** ... 5. **Portfolio**

A plausible generic job-application list. **None of it is in the internship
document**, which actually specifies: offer letter, joining confirmation,
completion certificate, work report, student report.

**RAG:**

> Typical documentation for an internship may include an offer letter, joining
> confirmation, completion certificate, work report, and student report.
>
> Sources: `Internship_Guidelines_DUMMY.pdf` p1

This is the clearest case in the comparison. The direct answer sounds authoritative
and is entirely wrong for this institution.

### Question: "How is an internship evaluated?"

**Direct LLM:** a generic list — performance, learning outcomes, professional
development. No mention of the document's actual criteria.

**RAG:**

> Internships may be evaluated using the report, presentation, employer feedback,
> and department criteria.
> (Source: `Internship_Guidelines_DUMMY.pdf`, page 1)

### Question: "When do the internal assessments take place?"

**Direct LLM:**

> Internal assessments typically take place during the examination period,
> usually at the end of the academic year... students complete their internal
> assessments in the last few months of the school year.

**Wrong for this college.** The calendar document gives specific dates.

**RAG:**

> Internal assessments take place on **15 September and 20 October**.
>
> Sources: `Academic_Calendar_DUMMY.pdf` p1

Specific, dated, cited. This is the failure mode that matters most: a student
planning around "the end of the academic year" would be planning around a
statement the college never made.

### Question: "What is the capital of Japan?"

**Direct LLM:** "The capital of Japan is Tokyo."

**RAG:**

> I could not find this information in the available college documents.

No LLM call was made — retrieval returned nothing.

### Question: "What is the hostel mess fee?"

**Direct LLM:** an explanation of what a hostel mess fee *is*, but no amount.

**RAG:** refuses — the documents contain no fee information.

Note the honest difference: the direct model happened not to invent a figure here,
but on the internship documents it confidently invented a five-item list. The
failure is not predictable from the question alone; grounding removes the whole
class of risk.

## Summary

| | Direct LLM | RAG |
|---|---|---|
| Source of college facts | Model weights, never the PDFs | The actual PDF text |
| Citation to a page | Not possible | Yes, from real metadata |
| Specific vs generic | Generic advice, often wrong here | Specific, document-backed |
| Out-of-scope questions | Answers from general knowledge | Declines, and says why |
| Invented detail | Common (resume/cover letter/portfolio) | Prevented by grounding |
| Refuses when nothing matches | No | Yes |
| Knowledge updates | Needs retraining | Re-run `--build` on new PDFs |
| Response time | Faster (no retrieval) | Slower (retrieval + longer prompt) |

## Why RAG matters for this project

Every fact about attendance, marks, dates, eligibility or policy comes from a
document the user can open and check. Two properties follow:

1. **Verifiability.** Each answer names a file and page, taken from retrieval
   metadata — never from the model's own output. A test asserts that any `.pdf`
   filename appearing in an answer is a real indexed document.
2. **Refusal.** When retrieval returns nothing above the relevance floor, the
   assistant stops without calling the LLM at all. A test asserts the LLM is not
   invoked in that case.

The cost is latency and index maintenance. For college regulations — where a wrong
percentage or date has real consequences — that trade is worth it.

## Honest limitations

- Retrieval is **cosine similarity**, not verification. A chunk can match on
  vocabulary without answering the question. Two gates reduce this: a relative
  floor (drop weak neighbours of a good hit) and an absolute floor `MIN_SCORE`.
- On the current dummy corpus, in-scope questions score 0.27–0.55 and
  out-of-scope noise scores below 0.03, so `MIN_SCORE=0.15` separates them. **This
  gap is specific to these documents** — re-measure after swapping in the real
  PDFs, using the Colab notebook (section 9) which reports a score per question.
- Small embedding models can miss paraphrases. `CHUNK_SIZE`, `CHUNK_OVERLAP`,
  `TOP_K` and `MIN_SCORE` in `.env` are the tuning knobs.
- No accuracy percentages are claimed. Everything above is observed output and
  will change once the real college PDFs replace the samples.