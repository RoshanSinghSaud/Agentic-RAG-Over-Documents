"""Layer 1: the baseline RAG graph — a linear retrieve -> generate pipeline.

This is deliberately the simplest possible StateGraph. Layers 2+ turn this into
an agentic graph by adding grading nodes, conditional edges, and cycles.
"""
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", nodes.retrieve)
    g.add_node("generate", nodes.generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()


# Compiled graph importable as `from src.graph import graph`
graph = build_graph()
