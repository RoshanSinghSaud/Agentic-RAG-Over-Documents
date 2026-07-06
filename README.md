# Agentic RAG over Documents

A self-correcting **agentic RAG** system over documents, built on **LangGraph**.
Hybrid retrieval (dense + BM25 + RRF) → answer with verified citations, evolving
layer by layer into a graph that grades its own retrieval (**CRAG**) and its own
answers (**Self-RAG**), with memory, a human-approval gate before web search, and
a FastAPI + Docker deployment.

> Full design lives in `CLAUDE.md` (the project brief). This README tracks build
> status and how to run it.

## Build status

| Layer                                                                   | Tag                | Status                    |
| ----------------------------------------------------------------------- | ------------------ | ------------------------- |
| 1 — Baseline RAG (hybrid retrieve → generate w/ citations)              | `v0.1-baseline`    | **done**                  |
| 2 — CRAG (grade docs, conditional routing, transform_query, web_search) | `v0.2-crag`        | **built — runnable**      |
| 3 — Self-RAG (grade answer, regenerate/re-retrieve)                     | `v0.3-self-rag`    | not started               |
| 4 — Memory + HITL (checkpointer + one interrupt gate)                   | `v0.4-memory-hitl` | not started               |
| 5 — Productionize (FastAPI, Docker, eval, LangSmith)                    | `v1.0`             | not started               |

_(Per-layer eval comparison table goes here once the harness runs — that's the headline.)_

## Architecture (Layer 2 — CRAG)

```
question ──> retrieve (dense + BM25 + RRF) ──> grade_documents
                  ^                                  │
                  │            ┌─────────────────────┼─────────────────────┐
                  │            │ sufficient          │ weak, loops left    │ weak, loops exhausted
                  │            v                     v                     v
                  │         generate          transform_query          web_search (Tavily)
                  │            │                     │                     │
                  │           END                    │                  generate ──> END
                  └──────────────────────────────────┘
```

The graph grades every retrieved chunk (structured-output LLM grader), keeps the
relevant ones, and — if too few survive — rewrites the query and re-retrieves
(capped at `MAX_RETRIEVAL_LOOPS` cycles), then falls back to a Tavily web search
as a last resort. Web-sourced chunks are tagged `[web]` in the output.

## Quickstart

```bash
# 1. Create a virtualenv and install deps
python -m venv agentic-rag && source agentic-rag/bin/activate

pip install -r requirements.txt

# 2. Add your key
cp .env.example .env        # set OPENAI_API_KEY and TAVILY_API_KEY (Layer 2)

# 3. Add a corpus  (see data/README.md for the suggested papers)
#    drop PDFs / markdown into ./data

# 4. Build the index, then ask
python main.py ingest
python main.py ask "What problem does Reciprocal Rank Fusion solve?"
```

## Layout

```
src/
  config.py      # all tunables (models, paths, chunking, retrieval k's)
  state.py       # the shared GraphState TypedDict
  ingestion.py   # load + chunk + persist Chroma
  retrieval.py   # dense + BM25 + RRF fusion
  nodes.py       # retrieve, generate, grade_documents, transform_query, web_search + router
  graph.py       # the LangGraph StateGraph wiring
main.py          # CLI: ingest / ask
eval/            # golden Q&A set + RAGAS harness (planned)
data/            # corpus (git-ignored)
```

## Stack

LangGraph · LangChain · OpenAI (GPT + `text-embedding-3-small`) · ChromaDB ·
BM25 (`rank_bm25`) · Tavily · RAGAS (eval) · FastAPI + Docker (Layer 5).
