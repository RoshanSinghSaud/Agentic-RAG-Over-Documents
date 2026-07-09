"""LangGraph nodes for Layers 1-2.

Each node is a plain function: it receives the graph state and returns a dict of
the keys it wants to update. LangGraph merges that dict back into the state.

Layer 1 (baseline): retrieve -> generate.
Layer 2 (CRAG):     grade_documents, transform_query, web_search, plus the
                    routing function `decide_after_grading`. The graph becomes
                    agentic here: it inspects its own retrieval and corrects it.
"""
from typing import List, Literal
import importlib

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import config, retrieval
from .state import GraphState

_llm = None
_tavily = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    return _llm


def _get_tavily():
    
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient  # tavily-python
        _tavily = TavilyClient()  # reads TAVILY_API_KEY from the environment
    return _tavily


# =========================================================================
# Layer 1 — retrieve & generate
# =========================================================================

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
    """Hybrid-retrieve chunks for the current (possibly rewritten) question."""
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


# =========================================================================
# Layer 2 — CRAG: grade_documents, transform_query, web_search, routing
# =========================================================================

class GradeDocument(BaseModel):
    """Structured verdict for one retrieved chunk (forces a clean yes/no —
    no parsing of free-form LLM text)."""

    binary_score: Literal["yes", "no"] = Field(
        description="Is the document relevant to the question? 'yes' or 'no'."
    )


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a retrieved document chunk is "
            "relevant to a user question. It does not need to fully answer the "
            "question — grade 'yes' if it contains keywords, concepts, or partial "
            "information related to the question; 'no' only if it is clearly "
            "off-topic. The goal is to filter out erroneous retrievals, not to be "
            "maximally strict.",
        ),
        (
            "human",
            "Retrieved chunk:\n\n{document}\n\nUser question: {question}",
        ),
    ]
)

TRANSFORM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite questions to improve document retrieval. Given the "
            "original question, produce ONE improved version: make implicit "
            "intent explicit, expand acronyms, and use vocabulary likely to "
            "appear in technical papers on the topic. Return only the rewritten "
            "question, nothing else.",
        ),
        ("human", "Original question: {question}"),
    ]
)


def grade_documents(state: GraphState) -> GraphState:
    """CRAG step 1: LLM-grade every retrieved chunk for relevance.

    Keeps only the chunks graded 'yes'. If fewer than
    ``config.MIN_RELEVANT_DOCS`` survive, retrieval is considered weak and the
    ``web_search`` flag is raised — the router then decides whether to rewrite
    the query and re-retrieve, or fall back to the web.

    Grading is batched (one LLM call per chunk, dispatched concurrently via
    ``.batch``) and uses structured output, so a verdict is always a clean
    'yes'/'no' rather than free text we'd have to parse.
    """
    question = state["question"]
    docs = state["documents"]

    grader = GRADE_PROMPT | _get_llm().with_structured_output(GradeDocument)
    verdicts = grader.batch(
        [{"question": question, "document": d.page_content} for d in docs]
    )

    relevant = [d for d, v in zip(docs, verdicts) if v.binary_score == "yes"]
    weak = len(relevant) < config.MIN_RELEVANT_DOCS

    print(
        f"--- GRADE: {len(relevant)}/{len(docs)} chunks relevant "
        f"{'(weak -> corrective path)' if weak else '(sufficient -> generate)'} ---"
    )
    return {"documents": relevant, "web_search": weak}


def transform_query(state: GraphState) -> GraphState:
    """CRAG step 2a: rewrite the question for better retrieval.

    Also increments ``loop_count`` — this is the counter that caps the
    re-retrieval cycle so the graph can't spin forever.
    """
    chain = TRANSFORM_PROMPT | _get_llm()
    better = chain.invoke({"question": state["question"]}).content.strip()
    loops = state.get("loop_count", 0) + 1
    print(f"--- TRANSFORM (loop {loops}): {better} ---")
    return {"question": better, "loop_count": loops}


def web_search(state: GraphState) -> GraphState:
    """CRAG step 2b: last-resort fallback — search the web via Tavily.

   # NOTE (Layer 4): the human-approval `interrupt` gate will be inserted
    immediately before this node. At Layer 2 it runs ungated.
    """
    question = state["question"]
    print(f"--- WEB SEARCH (Tavily): {question} ---")
    response = _get_tavily().search(query=question, max_results=config.WEB_SEARCH_K)

    web_docs: List[Document] = [
        Document(
            page_content=r["content"],
            metadata={"source": r["url"], "title": r.get("title", ""), "web": True},
        )
        for r in response.get("results", [])
    ]
    return {"documents": state["documents"] + web_docs, "web_search": False}


def decide_after_grading(state: GraphState) -> str:
    """Conditional edge after ``grade_documents`` — the CRAG router.

    Returns the name of the next node:

    - ``"generate"``        — enough relevant chunks survived grading.
    - ``"transform_query"`` — retrieval was weak and we still have loop budget:
                              rewrite the query and re-retrieve (the cycle).
    - ``"web_search"``      — retrieval was weak and the loop cap
                              (``config.MAX_RETRIEVAL_LOOPS``) is exhausted:
                              fall back to the web, then generate.
    """
    if not state.get("web_search", False):
        return "generate"
    if state.get("loop_count", 0) < config.MAX_RETRIEVAL_LOOPS:
        return "transform_query"
    return "web_search"
