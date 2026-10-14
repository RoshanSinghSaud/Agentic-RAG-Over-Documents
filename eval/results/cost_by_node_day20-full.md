# Cost per query by node — `day20-full`

20 queries · project `agentic-rag` · pulled 2026-10-09 09:46

| node | LLM calls / q | tokens in / q | tokens out / q | $ / query | % of cost | p50 latency / run | time / q | % of time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `retrieve` | 0.0 | 0 | 0 | $0.000000 | 0% | 0.30s | 0.94s | 17% |
| `grade_documents` | 10.8 | 4,542 | 65 | $0.000720 | 69% | 0.84s | 1.68s | 30% |
| `transform_query` | 0.8 | 84 | 13 | $0.000021 | 2% | 0.83s | 0.67s | 12% |
| `human_approval_gate` | 0.0 | 0 | 0 | $0.000000 | 0% | 0.00s | 0.00s | 0% |
| `generate` | 1.0 | 904 | 62 | $0.000168 | 16% | 1.11s | 1.37s | 25% |
| `grade_answer` | 1.1 | 912 | 7 | $0.000136 | 13% | 1.51s | 0.90s | 16% |
| `finalize` | 0.0 | 0 | 0 | $0.000000 | 0% | 0.00s | 0.00s | 0% |

- **Cost per query:** mean $0.001046 · median $0.001011 · max $0.001446 (`eval-q06`) · ≈ $1.05 per 1,000 queries
- **Tokens per query:** 6,442 in / 147 out
- **End-to-end latency:** median 5.7s per query
- **Most expensive node:** `grade_documents` — 69% of cost, 10.8 LLM calls per query
- **Slowest node:** `grade_documents` — 30% of node time
- **By path:** direct 12 q at $0.000864 · corrective (rewrote) 8 q at $0.001318 (1.5×)

Not in the cost column: the query embedding inside `retrieve` (OpenAIEmbeddings isn't traced as an LLM run; ~20 tokens at $0.02/1M, effectively $0) and Tavily credits in `web_search`.
