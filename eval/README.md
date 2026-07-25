# Evaluation harness

The headline of this project's README is a **per-layer numbers table** (baseline
RAG vs +CRAG vs +Self-RAG vs +Memory/HITL), produced by running the _same_ golden
Q&A set against each layer. That's only possible because it's one codebase with
one harness.

## Running it

```bash
python -m eval.run_eval --label v0.4              # full run: graph + RAGAS
python -m eval.run_eval --label v0.4 --skip-ragas # behavior checks only (fast)
python -m eval.run_eval --only q05 --skip-ragas   # debug one question
python -m eval.run_eval --label v0.4-web --approve-web  # auto-APPROVE the gate
```

Outputs land in `eval/results/`: `results_<label>.csv` (one row per question)
and `summary_<label>.md` (per-category table for the README).

## Golden set — `eval/golden_set.json`

A JSON list (20 entries) of objects:

```json
{
  "id": "q01",
  "category": "lookup",
  "question": "...",
  "ground_truth": "...",
  "expected_sources": ["attention-is-all-you-need"],
  "notes": "why this question is in the set / what to check when it fails"
}
```

Field rules:

- **category** — one of five, each mapped to a graph behavior it forces:
  - `lookup`: one chunk answers it; tests the plain retrieve→generate path.
  - `multi_hop`: needs chunks from ≥2 papers; tests RRF fusion / reranking.
  - `no_answer`: unanswerable from corpus AND web; success = clean abstention.
  - `needs_web`: outside the corpus but web-answerable; success = grader flags
    weak docs → question reaches the human approval gate (then abstains under
    the harness's auto-decline policy).
  - `known_failure`: a bug personally witnessed in testing; regression tracker.
    One primary label per question; overlaps go in `notes`.
- **ground_truth** — phrased in the papers' own vocabulary (not blog-speak),
  since RAGAS answer-correctness compares against it. The literal string
  `<ABSTENTION>` is a placeholder: the harness substitutes `nodes.ABSTENTION`
  at load time so the golden set can never drift out of sync with the code.
- **expected_sources** — filename stems from `data/`, matched case-insensitively
  as substrings of each Document's `source` metadata path. Not used by RAGAS;
  drives the separate citation-accuracy check.
- **notes** — for humans only; no code reads it.

## HITL gate policy

Automated eval can't wait at `human_approval_gate`. The harness runs with the
gate **auto-declined** (deny-by-default, same as garbage input), so `needs_web`
questions are scored on _reaching the gate and abstaining gracefully_. Passing
`--approve-web` flips to auto-approve and scores the web-fallback path instead.

## Scoring

Two independent layers of scoring per question:

1. **Deterministic behavior checks** (no LLM): did the graph do the right
   _thing_ — abstain, escalate to the gate, retrieve the expected sources?
   These are the `behavior_pass` / `citation_hits` columns.
2. **RAGAS** metrics (LLM-judged): `answer_correctness` (all rows),
   `faithfulness` (only substantive answers with contexts — scoring an honest
   "I don't know" would punish correct behavior), `context_precision` /
   `context_recall` (only rows with expected corpus sources).

### RAGAS compatibility note

ragas ≤0.4.3 crashes on import beside langchain-community 0.4.x (the
langchain 1.x line): it imports `ChatVertexAI` from a legacy module that no
longer exists. `run_eval.py` injects an inert stand-in module before importing
ragas (`_shim_legacy_vertexai()`); at runtime ragas 0.4's modern metrics
(`ragas.metrics.collections`) call OpenAI through `instructor` directly, so
nothing else touches the removed code. Remove the shim when a fixed ragas
ships. Metric LLM: `config.LLM_MODEL`, `max_tokens=8192` (faithfulness
statement-decomposition overruns smaller limits).

## Status

- [x] Golden set v1 — 20 questions, all 8 corpus papers covered
      (7 lookup / 4 multi_hop / 3 no_answer / 3 needs_web / 3 known_failure)
- [x] `eval/run_eval.py` — runner + behavior checks + RAGAS + CSV/summary
- [x] Baseline recorded: `results_v0.4.csv` / `summary_v0.4.md` (15/20 behavior
      pass; multi_hop weakest at 1/4 — see root README findings)
- [ ] Fix duplicate-ingestion bug (index held every chunk 4×), re-ingest, re-run
- [ ] Grow set toward ~50; add per-layer baseline table to root README
