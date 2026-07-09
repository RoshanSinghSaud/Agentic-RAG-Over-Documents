# Layer 2 (CRAG) — Verification Traces

Real runs of `python main.py ask "..."` against the corpus, captured to show that the
CRAG router (`decide_after_grading` in `src/nodes.py`) takes the correct branch in each
situation. This is the evidence behind the `v0.2-crag` tag.

**Corpus:** 8 foundational LLM/RAG papers — Attention Is All You Need, RAG, DPR, ReAct,
Lost-in-the-Middle, HyDE, Corrective-RAG, Self-RAG (`data/`).

**The router has three branches:**

- `sufficient` → `generate` — enough chunks graded relevant; answer straight from the corpus.
- `weak, loops remaining` → `transform_query` → `retrieve` — rewrite the question and re-retrieve (the corrective cycle, capped by `MAX_RETRIEVAL_LOOPS`).
- `weak, loops exhausted` → `web_search` → `generate` — corpus can't support the question; fall back to Tavily.

---

## 1. Sufficient path — strong retrieval, answered from the corpus

**Branch fired:** `sufficient → generate`

```text
$ python main.py ask "What is multi-head self-attention?"
--- GRADE: 4/5 chunks relevant (sufficient -> generate) ---
=== ANSWER ===
Multi-head self-attention is an attention mechanism that allows the model to jointly attend
to information from different representation subspaces at different positions. It consists of
several attention layers running in parallel, where each attention head computes attention
independently and the results are concatenated and projected to form the final output values.
This approach helps to mitigate the limitations of single-head attention, such as averaging
inhibiting the model's ability to capture diverse information [2][4].
=== SOURCES ===
[1] data/attention-is-all-you-need.pdf, p.1
[2] data/attention-is-all-you-need.pdf, p.4
[3] data/attention-is-all-you-need.pdf, p.2
[4] data/attention-is-all-you-need.pdf, p.3
```

**Reads as:** 4 of 5 retrieved chunks graded relevant → over the `MIN_RELEVANT_DOCS`
threshold → routed straight to `generate`. Answer is grounded in the source paper with
page-level citations. No query rewrite, no web fallback. Also doubles as a **retriever
health check**: the hybrid dense + BM25 + RRF retriever pulls the right chunks for an
in-corpus question.

---

## 2. Corpus can't answer → re-retrieval loop exhausts → web fallback

**Branch fired:** `weak → transform_query (x2) → weak → web_search → generate`

```text
$ python main.py ask "What problem does Reciprocal Rank Fusion solve?"
--- GRADE: 0/5 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 1): What specific information retrieval challenges does Reciprocal Rank
    Fusion address in the context of combining multiple ranked lists of search results? ---
--- GRADE: 0/5 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 2): What specific challenges in information retrieval does Reciprocal Rank
    Fusion aim to address when integrating multiple ranked lists of search results? ---
--- GRADE: 0/5 chunks relevant (weak -> corrective path) ---
--- WEB SEARCH (Tavily): ... ---
(question was rewritten to: '...')
=== ANSWER ===
Reciprocal Rank Fusion (RRF) aims to address the challenge of integrating multiple ranked
lists of search results that come from different retrieval methods, which often produce
scores on fundamentally different scales... RRF simplifies this by using only the position
of each result in the ranked lists rather than comparing raw scores [1][3].
=== SOURCES ===
[1] https://www.paradedb.com/learn/search-concepts/reciprocal-rank-fusion  [web]
[2] https://www.ai21.com/glossary/tech/what-is-reciprocal-rank-fusion-rrf  [web]
[3] https://spice.ai/learn/reciprocal-rank-fusion  [web]
```

**Reads as:** RRF is **not** in the corpus — it is implemented in `src/retrieval.py`, but no
RRF paper is ingested — so grading 0/5 relevant is *correct*, not a bug. The re-retrieval
loop rewrites the query twice (`MAX_RETRIEVAL_LOOPS = 2`), still finds nothing relevant, then
correctly falls back to a Tavily web search. Web-sourced context is tagged `[web]`. This is
the corrective loop's full escalation path firing end to end.

---

## 3. Sufficient path on a vaguely-phrased question — retriever robustness

**Branch fired:** `sufficient → generate`

```text
$ python main.py ask "does it matter where I put the important sentence in a long prompt?"
--- GRADE: 3/5 chunks relevant (sufficient -> generate) ---
=== ANSWER ===
Yes, it does matter where you place the important sentence in a long prompt. Performance is
significantly higher when relevant information is placed at the start of the input context,
and highest when it occurs at the very end; it rapidly degrades when models must reason over
information in the middle of their input context [2][3].
=== SOURCES ===
[1] data/Self-RAG.pdf, p.22
[2] data/Lost in the Middle- How Language Models Use Long Contexts.pdf, p.3
[3] data/Lost in the Middle- How Language Models Use Long Contexts.pdf, p.4
```

**Reads as:** even a colloquial, non-technical phrasing retrieved the right Lost-in-the-Middle
chunks (3/5 relevant) and answered from the corpus. This shows the hybrid retriever is strong
enough that the **early re-retrieval-recovery branch** (`weak → transform → sufficient`,
without reaching the web) rarely fires on natural questions — there is usually nothing to
recover from. To observe that branch deliberately, raise `MIN_RELEVANT_DOCS` in `src/config.py`
(e.g. to 5) so a good first pass is forced onto the corrective path, then reset it to 2.

---

## How to reproduce

```bash
python main.py ingest        # once, if .chroma is not built
python main.py ask "What is multi-head self-attention?"                 # -> branch 1
python main.py ask "What problem does Reciprocal Rank Fusion solve?"    # -> branch 2 (web)
```

## What this proves

The graph inspects its own retrieval and routes on the result instead of doing one blind
pass: it answers from the corpus when retrieval is strong, and detects when the corpus cannot
support a question and escalates (rewrite → re-retrieve → web fallback) rather than
hallucinating. Retriever health is confirmed independently (branch 1/3), so the web fallback
in branch 2 is an honest "not in corpus" decision, not a symptom of broken retrieval.
