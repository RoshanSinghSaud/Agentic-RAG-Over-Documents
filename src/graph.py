"""The agentic-RAG graph: retrieve -> generate with two self-correction loops
(CRAG on retrieval, Self-RAG on the answer), multi-turn memory via a
checkpointer, and one human approval gate before the web fallback.

    START -> retrieve -> grade_documents
                             |
       sufficient ---------> generate
       weak, loops left ---> transform_query -> retrieve      (rewrite cycle)
       weak, loops done ---> human_approval_gate
                                 approved -> web_search -> generate
                                 denied   -> generate    (best effort, no web)

    generate -> grade_answer
                    grounded & useful ----------> finalize -> END
                    hallucinated (budget left) -> generate    (regenerate)
                    off-topic    (budget left) -> transform_query -> retrieve
                    any budget exhausted -------> finalize -> END (best effort)

Loop budgets: query rewrites are capped by config.MAX_RETRIEVAL_LOOPS
(`loop_count`, shared by the CRAG and Self-RAG paths so all rewrites draw on
one budget); regenerations by config.MAX_GENERATION_LOOPS (`generate_count`).
"""
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)

    # --- nodes ---
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)   # CRAG: grade each chunk
    g.add_node("transform_query", nodes.transform_query)   # CRAG: rewrite question
    g.add_node("web_search", nodes.web_search)             # CRAG: web fallback
    g.add_node("generate", nodes.generate)
    g.add_node("grade_answer", nodes.grade_answer)         # Self-RAG: grade the answer
    g.add_node("finalize", nodes.finalize)                 # Layer 4: answer -> history
    g.add_node("human_approval_gate", nodes.human_approval_gate)  # Layer 4: HITL

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
            "web_search": "human_approval_gate",  # Layer 4: gate before web_search
        },
    )

    # Layer 4: the one HITL interrupt, gating the highest-stakes action.
    # decide_after_grading stays unchanged — it only ever inspects
    # document-grading verdicts. What happens after approval is a separate
    # concern, routed by its own router.
    g.add_conditional_edges(
        "human_approval_gate",
        nodes.decide_after_approval,
        {
            "web_search": "web_search",
            "generate": "generate",  # denied -> best-effort with existing docs
        },
    )

    g.add_edge("transform_query", "retrieve")  # the re-retrieval cycle
    g.add_edge("web_search", "generate")       # fallback context -> answer

    # Layer 3: every answer is graded before it can leave the graph.
    g.add_edge("generate", "grade_answer")

    # The Self-RAG router: accept, regenerate, or rewrite-and-re-retrieve.
    # Every terminal verdict routes through `finalize`, which appends the
    # accepted answer to the thread's message history before END — inside the
    # graph, so the history write shares the turn's checkpoint.
    g.add_conditional_edges(
        "grade_answer",
        nodes.decide_after_answer,
        {
            "end": "finalize",
            "generate": "generate",                # hallucination -> regenerate
            "transform_query": "transform_query",  # off-topic -> better retrieval
        },
    )
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


# Checkpointer-less compile for import-time uses (e.g. view_graph.py rendering
# graph.png). main.py builds its own instance with a SqliteSaver — interrupt()
# only works through that one.
graph = build_graph()
