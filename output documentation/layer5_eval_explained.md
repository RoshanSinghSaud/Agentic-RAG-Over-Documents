# Layer 5, part 1 — Evaluation: what we built, what happened, and what it changed

This is the plain-language record of the eval work (July 23–24, 2026). The
other layer docs in this folder are verification traces; this one is longer,
because the eval produced a chain of findings that deserve a full narrative.

---

## 1. Why an eval exists at all

Through Layer 4, "does the system work?" was answered by typing questions at
the CLI and eyeballing the output. That found real bugs (the World Cup grader
bug, the "What is RAG?" over-abstention) — but it can't answer the question
that actually matters when you *change* something: **did this change make the
system better or worse, and by how much?** Manual poking doesn't scale, isn't
repeatable, and has no memory.

The eval is that manual poking written down once, so it becomes repeatable and
countable. Same 20 questions every time; change one thing; re-run; compare
numbers.

## 2. The two artifacts

**`eval/golden_set.json` — the exam paper.** 20 hand-written questions, each
with the answer we'd accept (`ground_truth`), the papers the answer should come
from (`expected_sources`), and a `category`. The five categories are not
topics — they are *behaviors of the graph* we want to force:

| category | what it forces | what "correct" means |
|---|---|---|
| `lookup` (7) | plain retrieve → generate | answers, cites the right paper |
| `multi_hop` (4) | fusion across ≥2 papers | answers using chunks from both papers |
| `no_answer` (3) | abstention path | says "I don't know" — not a made-up answer |
| `needs_web` (3) | weak-retrieval → approval gate | asks permission for web search, then abstains cleanly (eval auto-denies) |
| `known_failure` (3) | bugs witnessed in Layer 4 testing | regression tracking — do old bugs come back? |

Abstention entries store the placeholder `<ABSTENTION>`; the harness swaps in
the real constant from `src/nodes.py` at load time, so the exam paper can never
drift out of sync with the prompt wording.

**`eval/run_eval.py` — the examiner.** A script that runs the exam and marks
it. No new architecture: it drives the *same graph* as `main.py`, just with
in-memory checkpoints (a fresh thread per question, `checkpoint.db` untouched)
and with the human-approval gate answered automatically ("n" — deny by
default, matching the gate's own fail-closed rule).

## 3. What one eval run actually does

For each of the 20 questions, in order:

1. **Ask** — `graph.invoke(...)` with the question, exactly like a CLI turn.
   If the graph pauses at the approval gate, the script records *that it
   asked* (`gate_fired`) and auto-answers "n".
2. **Record the raw outcome** — the answer text, the chunks the generator saw,
   their source paths, whether the answer was the abstention sentence, how many
   rewrite/regenerate loops were spent.
3. **Mark behavior (pure Python, no LLM)** — one rule per category, from the
   table above. Produces the `behavior_pass` column. Also counts how many
   `expected_sources` actually showed up in the retrieved chunks
   (`citation_hits`).
4. **Mark quality (RAGAS — an LLM as examiner)** — four scores, 0 to 1:
   - **answer_correctness**: does the answer say what `ground_truth` says?
   - **faithfulness**: is every claim in the answer supported by the retrieved
     chunks? (Low = the model made things up.)
   - **context_precision**: of the chunks retrieved, how many were actually
     useful?
   - **context_recall**: of the information needed for the reference answer,
     how much did the retrieved chunks contain?
   Skips are deliberate: an honest "I don't know" is not scored for
   faithfulness (that would punish correct behavior), and context metrics only
   run where the corpus is supposed to contain the answer.
5. **Write it down** — `eval/results/results_<label>.csv` (one row per
   question, every column above) and `summary_<label>.md` (the per-category
   table). The `--label` flag names the run, which is what makes before/after
   comparison possible: run once as `v0.4`, change something, run again as
   `v0.4-fixed`, and diff the two files.

One engineering note that mattered: RAGAS ≤0.4.3 crashes on import next to the
langchain 1.x line (it imports a module that no longer exists). The harness
injects a tiny inert stand-in module before importing RAGAS
(`_shim_legacy_vertexai()` — see the comment there). RAGAS's modern metrics
then talk to OpenAI directly, so nothing else touches the removed code.

## 4. "After comparison, what was done?" — the actual sequence of events

This is the part that turns numbers into engineering. The scores never fix
anything themselves — **their job is to point at where to look.** Here is the
chain, step by step, as it really happened:

1. **First full run (`v0.4`): 15/20 behavior pass.** Fine overall — but one
   cell stood out: multi-hop questions scored **1/4**, with answer
   correctness ~0.27 while faithfulness was ~0.93. That combination is
   specific: *the model wasn't lying (faithful), it just didn't have the right
   material in front of it.* Suspicion: retrieval.
2. **Looked at the raw rows** (`results_v0.4.csv`), not the summary. The
   failing multi-hop rows showed `citation_hits = 1/2` — chunks from only one
   of the two required papers — and the trace lines showed the grader
   receiving only ~2 chunks when config says `TOP_K = 6`.
3. **Tested the suspicion directly**: called `hybrid_retrieve()` by hand.
   It returned 2 documents. Then counted the index: **2,872 chunks — exactly
   4 × 718.** The corpus had been ingested four times;
   `Chroma.from_documents` *appends* instead of replacing, and nothing in
   `ingest()` prevented re-runs from stacking duplicates.
4. **Why duplicates starve retrieval**: dense search returns 8 candidates, but
   they're mostly 4 copies of the same 2 unique chunks; BM25 likewise. RRF
   de-duplicates while fusing — and after collapsing the copies, only ~2
   unique chunks survive to the generator. The system had been answering
   with a third of its intended context, silently, for weeks.
5. **The fix** (one line, in `src/ingestion.py`): make `ingest()` wipe the old
   index before writing, so re-running it can never duplicate. Delete
   `.chroma`, re-ingest once → 718 chunks.
6. **Re-ran the same exam** as `v0.4-fixed` and compared the two summaries:

   | run | behavior | ctx recall | ctx precision | faithfulness | ans correctness |
   |---|---|---|---|---|---|
   | `v0.4` (dup index) | 15/20 | 0.765 | 0.697 | 0.941 | 0.657 |
   | `v0.4-fixed` (clean) | 15/20 | **0.856** | **0.759** | **0.972** | 0.633 |

7. **Read the comparison honestly.** The gains landed exactly where a
   retrieval bug should show: context recall up overall, multi-hop recall
   0.556 → 0.722, and q05's recall going 0.0 → 1.0. The small down-moves in
   LLM-judged correctness (< 0.1) are grader noise, not signal — with 3–7
   questions per cell, only large moves mean anything.
8. **The number that did NOT move was the second finding.** Behavior pass
   stayed 15/20 with the *same five failures* — proving the duplicate bug and
   the behavioral failures were **two separate problems**. With retrieval now
   healthy, the survivors are cleanly isolated as prompt-level bugs:
   - **Comparative questions** (q02, q06, q15): the document grader judges
     chunks against the whole comparison — and since no single chunk discusses
     *both* papers, it can reject everything. On "How does CRAG differ from
     HyDE?" it rejected 18/18 chunks across three retrieval attempts. A
     CRAG-only chunk *is* relevant to a CRAG-vs-HyDE question; the grader
     prompt disagrees. Fix lives in `GRADE_PROMPT`.
   - **Definitional over-abstention** (q12, q05-family): "How does HyDE
     retrieve documents…" answered "I don't know" while holding 6/6 relevant
     chunks — the strongest reproduction yet of the Layer 4 "What is RAG?"
     bug, now unblamable on thin context. Fix lives in `GENERATE_PROMPT`'s
     abstention rule.
   - Also persisting: q03's *attribution* failure — asked what the **RAG
     paper** says about GPT-4, the system answered with GPT-4 numbers from the
     **Self-RAG paper** (faithfulness 1.0, correctness 0.06 — grounded in the
     wrong source). A grader can't catch this; the generator must respect
     which paper a claim came from.

Both prompt fixes are queued as the `v0.5` row of the README table — each will
be a one-prompt change followed by a re-run, so each gets its own measured
delta.

## 5. How it helped — the three sentences that summarize this

1. The eval **converted opinion into measurement**: "the system seems okay"
   became "15/20 correct behaviors, weakest on multi-hop (1/4), and here are
   the five failing questions by name."
2. The measurement **found a silent data bug on day one** — a 4×-duplicated
   index throttling every answer's context — that three layers of
   self-correcting graders had no way to notice, because every individual step
   was locally "correct."
3. The re-run after the fix **separated the fixed problem from the remaining
   ones**: retrieval metrics recovered, behavior failures persisted, and what
   remains is now two named, reproducible, prompt-level bugs with regression
   tests already in the golden set.

## 6. Current status / open items

- Harness, golden set (20 q), and two measured runs: **done** (this document).
- Queued next: the two prompt fixes (one at a time, re-measuring after each);
  growing the golden set toward ~50; Layer 4 leftovers (Ctrl+C-at-gate test,
  gate input variants, CLAUDE.md log entries).
- Descoped deliberately: re-running the harness against old git tags
  (v0.1–v0.3) — those graphs don't share today's state shape, so that table
  would measure compatibility adapters, not layers. The README table tracks
  the shipped system forward instead.
