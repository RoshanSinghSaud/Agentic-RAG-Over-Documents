## Eval summary — `day21-off` (20 questions)

| category | n | behavior pass | faithfulness | answer correctness | context precision | context recall |
|---|---|---|---|---|---|---|
| lookup | 7 | 6/7 | 1.000 | 0.641 | 0.890 | 0.893 |
| multi_hop | 4 | 1/4 | 0.933 | 0.369 | 0.549 | 0.722 |
| no_answer | 3 | 2/3 | 1.000 | 0.685 | — | — |
| needs_web | 3 | 3/3 | — | 1.000 | — | — |
| known_failure | 3 | 3/3 | 1.000 | 0.851 | 0.333 | 1.000 |
| **all** | 20 | **15/20** | 0.982 | 0.678 | 0.746 | 0.856 |

**Cost:** $0.001055 per question (mean) · 13.7 LLM calls · 6,592 tokens · MODEL_ROUTING=False GRADE_BATCH=False MAX_GRAPH_STEPS=20

Failures:
- **q02** (multi_hop): How does CRAG's corrective action differ from Self-RAG's approach to handling poor retrieval?
- **q03** (no_answer): What does the RAG paper say about GPT-4's performance?
- **q06** (multi_hop): How does CRAG differ from HyDE?
- **q12** (lookup): How does HyDE retrieve relevant documents without any relevance labels?
- **q15** (multi_hop): What retriever does the RAG architecture build on, and how does that retriever represent passages?
