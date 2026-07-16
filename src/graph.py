"""Layer 3: the Self-RAG graph — answers that grade and correct themselves.

Layer 2 (CRAG) made the graph inspect its own *retrieval*. Layer 3 closes the
second loop: after generating, the graph inspects its own *answer* and routes
on the verdict instead of blindly returning it.

    START -> retrieve -> grade_documents
                             |
        +--------------------+----------------------+
        | sufficient         | weak, loops left     | weak, loops exhausted
        v                    v                      v
     generate         transform_query           web_search
        |                    |                      |
        v               (back to retrieve)          v
   grade_answer  <----------------------------- generate
        |
        +-- grounded & useful ------------------> END
        +-- hallucinated (budget left) ---------> generate      (regenerate)
        +-- off-topic    (budget left) ---------> transform_query -> retrieve
        +-- any budget exhausted ---------------> END (best-effort answer)

New LangGraph concept vs Layer 2: *conditional routing on the model's own
output*. The Layer 2 router branched on retrieval quality (input side); the
Self-RAG router branches on generation quality (output side). Same primitive —
`add_conditional_edges` — pointed at a different stage of the pipeline, which
is what turns RAG into a closed feedback loop.

Both corrective cycles stay capped:
  * regenerate loop  — `generate_count` vs config.MAX_GENERATION_LOOPS
  * re-retrieve loop — `loop_count`     vs config.MAX_RETRIEVAL_LOOPS (shared
    with CRAG, so total query rewrites are budgeted in one place)
"""
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    # --- nodes ---
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)   # CRAG: grade each chunk
    g.add_node("transform_query", nodes.transform_query)   # CRAG: rewrite question
    g.add_node("web_search", nodes.web_search)             # CRAG: web fallback
    g.add_node("generate", nodes.generate)
    g.add_node("grade_answer", nodes.grade_answer)         # Self-RAG: grade the answer

    # --- edges ---
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")

    # The CRAG router: decide_after_grading returns the next node's name.
    g.add_conditional_edges(
        "grade_documents",
        nodes.decide_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    g.add_edge("transform_query", "retrieve")  # the re-retrieval cycle
    g.add_edge("web_search", "generate")       # fallback context -> answer

    # Layer 3: every answer is graded before it can leave the graph.
    g.add_edge("generate", "grade_answer")

    # The Self-RAG router: accept, regenerate, or rewrite-and-re-retrieve.
    g.add_conditional_edges(
        "grade_answer",
        nodes.decide_after_answer,
        {
            "end": END,
            "generate": "generate",              # hallucination -> regenerate
            "transform_query": "transform_query",  # off-topic -> better retrieval
        },
    )

    return g.compile()


# Compiled graph importable as `from src.graph import graph`
graph = build_graph()
