"""LangGraph nodes for Layer 1 (baseline RAG).

Each node is a plain function: it receives the graph state and returns a dict of
the keys it wants to update. LangGraph merges that dict back into the state.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config, retrieval
from .state import GraphState

_llm = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    return _llm


GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise question-answering assistant. Answer ONLY from the "
            "numbered context below. After each claim, cite the supporting source(s) "
            "inline like [1] or [1][3]. If the context does not contain the answer, "
            'reply exactly: "I don\'t have enough information in the provided '
            'documents to answer that." Do not use outside knowledge.\n\n'
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _format_context(documents) -> str:
    """Number each chunk so the model can cite [1], [2], ... and we can map back."""
    blocks = []
    for i, d in enumerate(documents, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        loc = src + (f", p.{page}" if page is not None else "")
        blocks.append(f"[{i}] ({loc})\n{d.page_content}")
    return "\n\n".join(blocks)


def retrieve(state: GraphState) -> GraphState:
    """Hybrid-retrieve chunks for the current question."""
    docs = retrieval.hybrid_retrieve(state["question"])
    return {"documents": docs}


def generate(state: GraphState) -> GraphState:
    """Generate an answer grounded in the retrieved chunks, with inline citations."""
    context = _format_context(state["documents"])
    chain = GENERATE_PROMPT | _get_llm()
    answer = chain.invoke(
        {"context": context, "question": state["question"]}
    ).content
    return {"generation": answer}
