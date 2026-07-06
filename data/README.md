# Corpus — foundational LLM / RAG papers

Drop the source documents for the system here (PDF, `.md`, or `.txt`). They are
**git-ignored** on purpose (keeps the repo light, avoids redistributing papers).

The chosen corpus is the foundational papers behind this very system — so it can
answer questions about the techniques it's built from, and naturally yields
lookup, multi-hop, and "not in the corpus" questions for the eval set.

Suggested starter set (download the PDFs from arXiv):

| Paper | arXiv | Why it's in here |
|---|---|---|
| Attention Is All You Need | 1706.03762 | The Transformer — the base of everything |
| Retrieval-Augmented Generation (RAG) | 2005.11401 | The original RAG formulation |
| Dense Passage Retrieval (DPR) | 2004.04906 | The dense half of hybrid retrieval |
| ReAct | 2210.03629 | Reason+act loops — the agentic mindset |
| Lost in the Middle | 2307.03172 | Why context ordering / reranking matters |
| HyDE | 2212.10496 | Hypothetical-document query expansion |
| Corrective RAG (CRAG) | 2401.15884 | Layer 2 of this project |
| Self-RAG | 2310.11511 | Layer 3 of this project |

Download e.g. from `https://arxiv.org/pdf/<id>` (verify each link in a browser).
After adding files, run `python main.py ingest`.
