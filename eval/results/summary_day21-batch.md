## Eval summary — `day21-batch` (20 questions)

| category | n | behavior pass | faithfulness | answer correctness | context precision | context recall |
|---|---|---|---|---|---|---|
| lookup | 7 | 6/7 | 0.976 | 0.630 | 0.903 | 0.893 |
| multi_hop | 4 | 1/4 | 0.800 | 0.369 | 0.677 | 0.500 |
| no_answer | 3 | 2/3 | 1.000 | 0.602 | — | — |
| needs_web | 3 | 2/3 | — | 1.000 | — | — |
| known_failure | 3 | 3/3 | 0.889 | 0.857 | 0.333 | 0.500 |
| **all** | 20 | **14/20** | 0.922 | 0.663 | 0.780 | 0.729 |

**Cost:** $0.000778 per question (mean) · 4.1 LLM calls · 4,686 tokens · MODEL_ROUTING=False GRADE_BATCH=True MAX_GRAPH_STEPS=20

Failures:
- **q02** (multi_hop): How does CRAG's corrective action differ from Self-RAG's approach to handling poor retrieval?
- **q03** (no_answer): What does the RAG paper say about GPT-4's performance?
- **q06** (multi_hop): How does CRAG differ from HyDE?
- **q07** (needs_web): What is the capital of Nepal?
- **q12** (lookup): How does HyDE retrieve relevant documents without any relevance labels?
- **q15** (multi_hop): What retriever does the RAG architecture build on, and how does that retriever represent passages?
