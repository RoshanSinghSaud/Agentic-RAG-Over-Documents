# Layer 3 (Self-RAG) — Verification Traces

Real runs of `python main.py ask "..."` against the corpus, captured to show that the
Self-RAG grader (`grade_answer` in `src/nodes.py`) and its router (`decide_after_answer`)
take the correct branch in each situation. This is the evidence behind the `v0.3-self-rag` tag.

**Corpus:** 8 foundational LLM/RAG papers — Attention Is All You Need, RAG, DPR, ReAct,
Lost-in-the-Middle, HyDE, Corrective-RAG, Self-RAG (`data/`).

**What Layer 3 adds:** every generated answer is graded before it can leave the graph —
first for **grounding** (is every claim supported by the retrieved chunks, or did the model
hallucinate?), then for **usefulness** (does it actually address the question?). The router
has four outcomes:

- `grounded + useful` → **END** — answer accepted.
- `honest abstention` → **END** — the exact "I don't have enough information…" string is
  recognized (`ABSTENTION` constant) and accepted immediately; a correct "I don't know"
  is a valid terminal answer, not a failure to loop on.
- `NOT grounded` → `generate` — regenerate from the same context (hallucination is a
  generation problem), capped by `generate_count` vs `MAX_GENERATION_LOOPS`.
- `grounded but off-topic` → `transform_query` → `retrieve` — the model faithfully
  summarized the wrong material, so the fix is better retrieval, sharing the `loop_count`
  budget with CRAG.

---

## 1. Accept path — grounded, on-topic, answered from the corpus

**Branch fired:** `grounded=yes, useful=yes → accept (END)`

```text
$ python main.py ask "What is Corrective RAG and how does it decide when to trigger a web search?"
--- GRADE: 4/5 chunks relevant (sufficient -> generate) ---
--- GRADE ANSWER: grounded=yes, addresses-question=yes -> accept ---

=== ANSWER ===

Corrective RAG (CRAG) is proposed to improve the robustness of generation by self-correcting
the results of retrievers and enhancing the utilization of documents for augmenting generation.
It employs a lightweight retrieval evaluator that estimates and triggers three knowledge
retrieval actions: Correct, Incorrect, and Ambiguous. A web search is triggered in the cases
of Incorrect and Ambiguous actions, as these situations indicate that the retrieved documents
may not be reliable or the accuracy of the retrieval is hard to distinguish. This corrective
action helps overcome challenges where no reliable knowledge can be referred to, thereby
broadening the spectrum of retrieved information and improving the overall quality of the
generation process [1][2][4].

=== SELF-CHECK: passed (grounded + addresses the question) ===

=== SOURCES (context handed to the generator) ===

[1] data/Corrective-RAG.pdf, p.1
[2] data/Corrective-RAG.pdf, p.9
[3] data/Corrective-RAG.pdf, p.1
[4] data/Corrective-RAG.pdf, p.4
```

**Reads as:** CRAG passed the documents (4/5 relevant), the hallucination grader confirmed
every claim is supported by the CRAG paper chunks, the answer grader confirmed it addresses
the question → accepted on the first pass. The `SELF-CHECK` line is Layer 3's new visible
output: the answer was *verified*, not just generated.

---

## 2. Abstention path — full corrective cascade, then an honest "I don't know"

**Branches fired:** CRAG `weak → transform ×2 → web_search`, then Self-RAG
`honest abstention → accept (END)`

```text
$ python main.py ask "What is the population of Kathmandu?"
--- GRADE: 0/5 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 1): What is the current population estimate of Kathmandu, the capital city of Nepal? ---
--- GRADE: 0/5 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 2): What is the most recent population estimate for Kathmandu, the capital city of Nepal, as of 2023? ---
--- GRADE: 0/5 chunks relevant (weak -> corrective path) ---
--- WEB SEARCH (Tavily): What is the most recent population estimate for Kathmandu, the capital city of Nepal, as of 2023? ---
--- GRADE ANSWER: honest abstention -> accept ---

(question was rewritten to: 'What is the most recent population estimate for Kathmandu, the capital city of Nepal, as of 2023?')

=== ANSWER ===

I don't have enough information in the provided documents to answer that.

=== SELF-CHECK: passed (grounded + addresses the question) ===

=== SOURCES (context handed to the generator) ===

[1] https://data.who.int/countries/524 [web]
[2] https://www.facebook.com/UNioneducation1/posts/... [web]
[3] https://en.wikipedia.org/wiki/Kathmandu [web]
```

**Reads as:** the corpus (RAG papers) contains nothing about Kathmandu, so CRAG's full
corrective cascade fired: 0/5 relevant → query rewrite ×2 (exhausting `MAX_RETRIEVAL_LOOPS=2`)
→ Tavily fallback. The Tavily snippets that came back didn't contain a clean population figure
either, so the generator — bound by its answer-only-from-context prompt — abstained. Layer 3's
abstention short-circuit then recognized the exact `ABSTENTION` string and accepted it
**without spending any grader calls or correction loops**: an honest "I don't know" is correct
behavior, not a failure. Critically, the run *terminated* — both loop budgets held.

*Design note:* this trace also shows why the abstention short-circuit exists. Without it, the
answer-usefulness grader would flag "I don't know" as not resolving the question and loop
pointlessly until the budget ran out, wasting several LLM calls to arrive at the same answer.

---

## 3. Rewrite-rescue path — CRAG repairs retrieval, generator still abstains honestly

**Branches fired:** CRAG `weak → transform ×2 → sufficient`, then Self-RAG
`honest abstention → accept (END)`. Run with `TOP_K = 2` (deliberately squeezed context).

```text
$ python main.py ask "How does the self-reflection mechanism in Self-RAG differ from the retrieval evaluator in CRAG, and which papers introduced each?"
--- GRADE: 1/2 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 1): What are the differences between the self-reflection mechanism utilized in Self-Recursive Augmented Generation (Self-RAG) ... ---
--- GRADE: 1/2 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 2): What are the distinctions between the self-reflection mechanism implemented in Self-Recursive Augmented Generation (Self-RAG) ... ---
--- GRADE: 2/2 chunks relevant (sufficient -> generate) ---
--- GRADE ANSWER: honest abstention -> accept ---

=== ANSWER ===

I don't have enough information in the provided documents to answer that.

=== SELF-CHECK: passed (grounded + addresses the question) ===

=== SOURCES (context handed to the generator) ===

[1] data/Self-RAG.pdf, p.0
[2] data/Corrective-RAG.pdf, p.9
```

**Reads as:** the query rewrites *repaired retrieval* (1/2 relevant → 2/2, one chunk from
each paper — exactly the right two documents) without ever needing the web fallback. But two
chunks were too little context for a multi-hop comparison question, so the generator — bound
by its answer-only-from-context prompt — abstained rather than padding the gaps from
pretraining. The abstention short-circuit accepted it. **No hallucination occurred to
correct** — the strict generate prompt prevented the failure upstream of the grader.

**Two design observations from this trace:**

1. *The transform prompt invented acronym expansions.* It rewrote Self-RAG as
   "Self-**Recursive** Augmented Generation" and CRAG as "**Contextual** Retrieval-Augmented
   Generation" — both wrong (actual: Self-Reflective / Corrective). Retrieval survived
   because the surrounding vocabulary was strong, but a wrong expansion could easily steer
   BM25 toward the wrong terms. Fix: instruct the rewriter to keep original technical terms
   verbatim and only expand acronyms it is certain about.
2. *Layered defenses mean the inner grader rarely fires.* The hallucination grader is the
   last line of defense behind (a) hybrid retrieval, (b) CRAG document grading, and (c) a
   strict grounded-generation prompt. When the first three work, honest abstention beats
   hallucination — which is the desired system behavior. The grader earns its keep on the
   runs where the generator slips anyway.

---

## 4. Correction path — hallucination → regenerate → budget cap (fault-injection test)

**Branches fired:** Self-RAG `NOT grounded → regenerate` and
`generation budget exhausted → best effort (END)`.

**Setup:** with the layered defenses working (hybrid retrieval + CRAG grading + strict
grounded-generation prompt), natural hallucinations are rare — so this branch was exercised
by deliberate fault injection: the generate prompt's "do not use outside knowledge"
constraint was temporarily replaced with "fill any gaps from your own knowledge", with
`TOP_K = 2`. (Both reverted after the test.)

```text
$ python main.py ask "How does the self-reflection mechanism in Self-RAG differ from the retrieval evaluator in CRAG, and which papers introduced each?"
--- GRADE: 1/2 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 1): ... Self-Retrieval-Augmented Generation (Self-RAG) ... Contextual Retrieval-Augmented Generation (CRAG) ... ---
--- GRADE: 1/2 chunks relevant (weak -> corrective path) ---
--- TRANSFORM (loop 2): ... ---
--- GRADE: 1/2 chunks relevant (weak -> corrective path) ---
--- WEB SEARCH (Tavily): ... ---
--- GRADE ANSWER: NOT grounded (hallucination), retry 1 ---
--- GRADE ANSWER: NOT grounded (hallucination), retry 2 ---
--- SELF-RAG: generation budget exhausted -> accept best effort ---

=== ANSWER ===
[fluent multi-paragraph comparison of Self-RAG reflection tokens vs CRAG's evaluator,
including claims and attributions not fully supported by the retrieved context]

=== SELF-CHECK: best effort (correction budget exhausted) after 2 failed grounding check(s) ===

=== SOURCES (context handed to the generator) ===
[1] data/Self-RAG.pdf, p.0
[2] https://www.kore.ai/blog/self-reflective-retrieval-augmented-generation-self-rag [web]
[3] https://medium.com/@sahin.samia/self-rag-... [web]
[4] https://www.linkedin.com/pulse/self-rag-... [web]
```

**Reads as:** the permissive prompt made the generator pad thin context from pretraining,
and the hallucination grader caught it — twice. Sequence: generation #1 flagged NOT grounded
(retry 1) → regenerate → generation #2 also flagged (retry 2) → `generate_count` hit
`MAX_GENERATION_LOOPS=2` → the router stopped looping and returned the best-effort answer,
**honestly labeled** as unverified by the `SELF-CHECK: best effort` line instead of being
passed off as a checked answer. This is the trace that proves (a) the grader detects
ungrounded generation, (b) the regenerate edge works, and (c) the budget cap terminates the
loop.

**Observations:**

1. *The grader was right both times* — the answer attributes specifics to CRAG that the two
   retrieved chunks + web snippets don't establish, and its CRAG description ("contextual
   understanding of retrieved data") appears to be contaminated by the transform step's wrong
   acronym expansion ("Contextual RAG") rather than the paper itself.
2. *Regenerating with the same permissive prompt can't fix a prompt-level fault* — the loop
   correctly did its job and then correctly gave up. In production the regenerate path exists
   for stochastic slips, where a second sample often lands grounded.
3. *Run-to-run variance:* the same question in run #3 reached 2/2 relevant after rewrites; this
   run stayed 1/2 and fell through to web search — LLM graders are probabilistic, which is why
   the loop budgets exist.

**Post-test fixes committed:** original generate prompt and `TOP_K = 5` restored; the
transform prompt now keeps technical terms and acronyms verbatim (it had invented wrong
expansions — "Self-*Recursive*", "*Contextual* RAG" — in every corrective run); `main.py`'s
SELF-CHECK line relabeled from "regeneration(s)" to "failed grounding check(s)" (the counter
counts grader failures — the final failure terminates rather than regenerating, so the old
label overcounted by one).
