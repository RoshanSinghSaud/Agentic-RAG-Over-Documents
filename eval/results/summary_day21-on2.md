## Eval summary — `day21-on2` (20 questions)

| category | n | behavior pass | faithfulness | answer correctness | context precision | context recall |
|---|---|---|---|---|---|---|
| lookup | 7 | 7/7 | 0.978 | 0.820 | 0.897 | 0.905 |
| multi_hop | 4 | 1/4 | 0.833 | 0.302 | 0.677 | 0.500 |
| no_answer | 3 | 2/3 | 1.000 | 0.701 | — | — |
| needs_web | 3 | 2/3 | — | 1.000 | — | — |
| known_failure | 3 | 3/3 | 0.667 | 0.897 | 0.583 | 1.000 |
| **all** | 20 | **15/20** | 0.918 | 0.737 | 0.798 | 0.778 |

**Cost:** $0.000724 per question (mean) · 4.2 LLM calls · 4,728 tokens · MODEL_ROUTING=True GRADE_BATCH=True MAX_GRAPH_STEPS=20

Failures:
- **q02** (multi_hop): How does CRAG's corrective action differ from Self-RAG's approach to handling poor retrieval?
- **q03** (no_answer): What does the RAG paper say about GPT-4's performance?
- **q06** (multi_hop): How does CRAG differ from HyDE?
- **q07** (needs_web): What is the capital of Nepal?
- **q15** (multi_hop): What retriever does the RAG architecture build on, and how does that retriever represent passages?
