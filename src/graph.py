"""Layer 2: the CRAG graph — retrieval that grades and corrects itself.

Layer 1 was a straight line (retrieve -> generate). Layer 2 makes the graph
agentic: after retrieving, it *grades* its own chunks and routes on the result.

    START -> retrieve -> grade_documents
                             |
        +--------------------+----------------------+
        | sufficient         | weak, loops left     | weak, loops exhausted
        v                    v                      v
     generate         transform_query           web_search
        |                    |                      |
       END              (back to retrieve)       generate -> END

Two new LangGraph concepts vs Layer 1:
  * conditional edges — `add_conditional_edges` routes on the return value of
    `nodes.decide_after_grading` instead of always following one edge.
  * cycles — `transform_query -> retrieve` loops back to an earlier node.
    The loop is capped by `loop_count` in state (see config.MAX_RETRIEVAL_LOOPS),
    so a corpus that simply can't answer the question degrades to a web search
    instead of spinning forever.
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
    g.add_edge("generate", END)                # Layer 3 replaces this with grade_answer

    return g.compile()


# Compiled graph importable as `from src.graph import graph`
graph = build_graph()
