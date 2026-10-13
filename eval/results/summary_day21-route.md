## Eval summary — `day21-route` (20 questions)

| category | n | behavior pass | faithfulness | answer correctness | context precision | context recall |
|---|---|---|---|---|---|---|
| lookup | 7 | 7/7 | — | — | — | — |
| multi_hop | 4 | 1/4 | — | — | — | — |
| no_answer | 3 | 2/3 | — | — | — | — |
| needs_web | 3 | 2/3 | — | — | — | — |
| known_failure | 3 | 2/3 | — | — | — | — |
| **all** | 20 | **14/20** | — | — | — | — |

**Cost:** $0.001025 per question (mean) · 14.0 LLM calls · 6,783 tokens · MODEL_ROUTING=True GRADE_BATCH=False MAX_GRAPH_STEPS=20

Failures:
- **q02** (multi_hop): How does CRAG's corrective action differ from Self-RAG's approach to handling poor retrieval?
- **q03** (no_answer): What does the RAG paper say about GPT-4's performance?
- **q06** (multi_hop): How does CRAG differ from HyDE?
- **q07** (needs_web): What is the capital of Nepal?
- **q15** (multi_hop): What retriever does the RAG architecture build on, and how does that retriever represent passages?
- **q20** (known_failure): Explain that more simply.
