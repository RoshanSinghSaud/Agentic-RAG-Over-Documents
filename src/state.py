"""The single graph state every node reads and writes.

Layer 1 only uses `question`, `documents`, and `generation`. The remaining
fields are declared now (with total=False so they're optional) to keep the
state forward-compatible with later layers — no refactor needed when you add
CRAG, Self-RAG, and memory.
"""
from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str            # current (possibly rewritten) question
    documents: List[Document]  # retrieved / filtered chunks
    generation: str          # the answer
    web_search: bool         # Layer 2 (CRAG): set by the document grader
    loop_count: int          # Layer 2/3: caps the transform_query -> retrieve cycle

    # --- Layer 3 (Self-RAG): answer self-correction ---
    generate_count: int      # caps the regenerate loop on hallucinated answers
    answer_grounded: bool    # grade_answer verdict: is the answer supported by the docs?
    answer_useful: bool      # grade_answer verdict: does the answer address the question?

    messages: list           # Layer 4: multi-turn memory
