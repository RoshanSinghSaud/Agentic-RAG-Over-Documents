## Eval summary — `v0.4` (20 questions)

| category | n | behavior pass | faithfulness | answer correctness | context precision | context recall |
|---|---|---|---|---|---|---|
| lookup | 7 | 6/7 | 0.958 | 0.723 | 0.786 | 0.964 |
| multi_hop | 4 | 1/4 | 0.933 | 0.274 | 0.528 | 0.556 |
| no_answer | 3 | 2/3 | 1.000 | 0.685 | — | — |
| needs_web | 3 | 3/3 | — | 0.917 | — | — |
| known_failure | 3 | 3/3 | 0.800 | 0.725 | 0.583 | 0.000 |
| **all** | 20 | **15/20** | 0.941 | 0.657 | 0.697 | 0.765 |

Failures:
- **q02** (multi_hop): How does CRAG's corrective action differ from Self-RAG's approach to handling poor retrieval?
- **q03** (no_answer): What does the RAG paper say about GPT-4's performance?
- **q06** (multi_hop): How does CRAG differ from HyDE?
- **q12** (lookup): How does HyDE retrieve relevant documents without any relevance labels?
- **q15** (multi_hop): What retriever does the RAG architecture build on, and how does that retriever represent passages?
