# Evaluation harness

The headline of this project's README is a **per-layer numbers table** (baseline
RAG vs +CRAG vs +Self-RAG), produced by running the *same* golden Q&A set against
each layer. That's only possible because it's one codebase with one harness.

## Plan
1. Hand-write a golden Q&A set (~50 pairs) in `eval/golden.jsonl`, one object per
   line: `{"question": "...", "answer": "...", "type": "lookup|multihop|no-answer|ambiguous"}`.
   Cover all four types — the `no-answer` ones exercise the CRAG web-search fallback.
2. Score with **RAGAS** (locked choice): answer correctness, faithfulness/grounding,
   context precision/recall, plus a citation-accuracy check.
3. Run at every layer; append results to the README table.

## Not built yet
The scorer (`eval/run_eval.py`) is added once the Layer 1 baseline produces
answers and the golden set exists. Install with `pip install ragas datasets`.
