"""LangGraph nodes. Each node is a plain function: it receives the graph state
and returns a dict of the keys it wants to update; the ``decide_*`` functions
are the conditional-edge routers.

Layer map: retrieve/generate (L1) - document grading, query rewrite, web
fallback (L2, CRAG) - answer grading (L3, Self-RAG) - finalize and the human
approval gate (L4, memory + HITL).
"""
import logging
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import config, retrieval, routing
from .state import GraphState

from langgraph.types import interrupt

logger = logging.getLogger(__name__)

_llms: dict[str, ChatOpenAI] = {}
_tavily = None


def _get_llm(model: str | None = None) -> ChatOpenAI:
    """One cached client per model name; default config.LLM_MODEL.

    gpt-5-family models are reasoning models: they reject `temperature`, and
    their hidden reasoning tokens are billed as output, so they run with
    reasoning_effort="minimal".
    """
    model = model or config.LLM_MODEL
    if model not in _llms:
        if model.startswith("gpt-5"):
            _llms[model] = ChatOpenAI(model=model, reasoning_effort="minimal")
        else:
            _llms[model] = ChatOpenAI(model=model, temperature=0)
    return _llms[model]


def _get_tavily():
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient  # tavily-python
        _tavily = TavilyClient()  # reads TAVILY_API_KEY from the environment
    return _tavily


# The exact string the generator emits when the corpus can't answer the question.
# Defined once so both the generate prompt and the Self-RAG grader agree on it:
# an honest "I don't know" is a *valid* terminal answer, not a failure to loop on.
ABSTENTION = (
    "I don't have enough information in the provided documents to answer that."
)


# =========================================================================
# Layer 1 — retrieve & generate
# =========================================================================

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise question-answering assistant with two distinct "
            "sources: (1) a numbered context of chunks retrieved from a document "
            "corpus, and (2) the conversation history with this user. First decide "
            "what kind of question this is:\n"
            "- About the DOCUMENT CORPUS / subject matter -> answer ONLY from the "
            "numbered context below, citing source(s) inline like [1] or [1][3] for "
            "every claim. Ignore the conversation history for this kind of question.\n"
            "- About the CONVERSATION ITSELF (e.g. what was asked or said earlier) "
            "-> answer from the conversation history instead, with no citation "
            "needed. The numbered context is very likely irrelevant noise for this "
            "kind of question (a failed document search) — ignore it even if it is "
            "present.\n"
            "If the relevant source (context for corpus questions, history for "
            "conversation questions) does not contain the answer, reply exactly: "
            '"' + ABSTENTION + '" '
            "Do not use outside knowledge beyond the context and the conversation "
            "history.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
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


def _format_history(messages) -> str:
    """Render prior turns as plain text for prompts that can't take BaseMessage
    objects directly (e.g. the grader prompts, which use a string human template
    rather than a MessagesPlaceholder)."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def retrieve(state: GraphState) -> GraphState:
    """Hybrid-retrieve chunks for the current (possibly rewritten) question."""
    docs = retrieval.hybrid_retrieve(state["question"])
    return {"documents": docs}


def _turn_question(state: GraphState) -> str:
    """This turn's question as the user asked it: the last human message.
    state["question"] may have been rewritten by transform_query."""
    for m in reversed(state.get("messages", [])):
        if m.type == "human":
            return m.content
    return state["question"]


def generate(state: GraphState) -> GraphState:
    """Generate an answer grounded in the retrieved chunks, with inline citations."""
    context = _format_context(state["documents"])
    # Include the full message list, even this turn's own question: {question}
    # below is state["question"], which CRAG's transform_query may have rewritten
    # for retrieval — the raw, originally-phrased question (the one that actually
    # signals "this is about our conversation") only survives in the last message.
    history = state.get("messages", [])

    # Day 21 model routing: a simple question is answered by the small model.
    # Classified on the user's own wording, not on a rewrite.
    route = routing.classify(_turn_question(state)) if config.MODEL_ROUTING else "complex"
    model = config.SMALL_LLM_MODEL if route == "simple" else config.LLM_MODEL
    print(f"--- GENERATE: {route} -> {model} ---")

    chain = GENERATE_PROMPT | _get_llm(model)
    answer = chain.invoke(
        {"context": context, "question": state["question"], "history": history}
    ).text  # .text, not .content: str for every model, incl. block-style output
    # gpt-5-nano sometimes types the abstention sentence with a curly
    # apostrophe (’). Every abstention check matches the exact string, so a
    # curly one slips past grade_answer's short-circuit and the eval. Normalise
    # once, here, at the source.
    answer = answer.replace("\u2019", "'")
    return {"generation": answer, "route": route}


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
            "intent explicit and use vocabulary likely to appear in technical "
            "papers on the topic. Keep every technical term, paper name, and "
            "acronym from the original EXACTLY as written — never expand or "
            "reword an acronym unless you are completely certain of its "
            "expansion. Return only the rewritten question, nothing else.",
        ),
        ("human", "Original question: {question}"),
    ]
)


class ChunkGrade(BaseModel):
    """One verdict inside a batched grading call."""

    chunk: int = Field(description="The chunk's number, as shown in brackets.")
    binary_score: Literal["yes", "no"] = Field(
        description="Is this chunk relevant to the question? 'yes' or 'no'."
    )


class GradeDocumentsBatch(BaseModel):
    """One relevance verdict per retrieved chunk."""

    grades: List[ChunkGrade] = Field(
        description="Exactly one entry per chunk, in the order given."
    )


# Same grading instructions as GRADE_PROMPT, sent once for the whole round
# instead of once per chunk (Day 21: the per-chunk fan-out was 69% of cost).
GRADE_BATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            GRADE_PROMPT.messages[0].prompt.template
            + " You will be given several numbered chunks: grade each one "
            "independently, on its own merits, and return exactly one verdict "
            "per chunk.",
        ),
        ("human", "Retrieved chunks:\n\n{chunks}\n\nUser question: {question}"),
    ]
)


def _grade_one_call(question: str, docs) -> List[str]:
    """Grade every chunk in a single LLM call; returns 'yes'/'no' per chunk.

    The model labels each verdict with the chunk's number, and verdicts are
    matched back by that number, not by position. A chunk the model skipped is
    kept (fail open): the grader exists to drop clear noise, and the answer
    graders downstream still check grounding.
    """
    numbered = "\n\n".join(f"[{i}]\n{d.page_content}" for i, d in enumerate(docs, 1))
    grader = GRADE_BATCH_PROMPT | _get_llm().with_structured_output(GradeDocumentsBatch)
    result = grader.invoke({"question": question, "chunks": numbered})
    by_chunk = {g.chunk: g.binary_score for g in result.grades}
    missing = [i for i in range(1, len(docs) + 1) if i not in by_chunk]
    if missing:
        logger.warning("batched grader returned no verdict for chunk(s) %s; keeping them", missing)
    return [by_chunk.get(i, "yes") for i in range(1, len(docs) + 1)]


def grade_documents(state: GraphState) -> GraphState:
    """CRAG step 1: LLM-grade every retrieved chunk for relevance.

    Keeps only the chunks graded 'yes'. If fewer than
    ``config.MIN_RELEVANT_DOCS`` survive, the ``web_search`` flag is raised and
    the router decides between a query rewrite and the web fallback.

    Grading is batched (one call per chunk via ``.batch``) with structured
    output, so a verdict is always a clean 'yes'/'no'.
    """
    question = state["question"]
    docs = state["documents"]

    if config.GRADE_BATCH and len(docs) > 1:
        scores = _grade_one_call(question, docs)
    else:
        grader = GRADE_PROMPT | _get_llm().with_structured_output(GradeDocument)
        verdicts = grader.batch(
            [{"question": question, "document": d.page_content} for d in docs]
        )
        scores = [v.binary_score for v in verdicts]

    relevant = [d for d, s in zip(docs, scores) if s == "yes"]
    weak = len(relevant) < config.MIN_RELEVANT_DOCS

    print(
        f"--- GRADE: {len(relevant)}/{len(docs)} chunks relevant "
        f"{'(weak -> corrective path)' if weak else '(sufficient -> generate)'} ---"
    )
    return {"documents": relevant, "web_search": weak}


def transform_query(state: GraphState) -> GraphState:
    """CRAG step 2a: rewrite the question for better retrieval.

    Also increments ``loop_count``. Both the CRAG path (weak retrieval) and the
    Self-RAG path (off-topic answer) route through here, so this one counter
    budgets all query rewrites.
    """
    chain = TRANSFORM_PROMPT | _get_llm()
    better = chain.invoke({"question": state["question"]}).content.strip()
    loops = state.get("loop_count", 0) + 1
    print(f"--- TRANSFORM (loop {loops}): {better} ---")
    return {"question": better, "loop_count": loops}


def web_search(state: GraphState) -> GraphState:
    """CRAG step 2b: last-resort fallback — search the web via Tavily.

    Gated by ``human_approval_gate`` (Layer 4): this node only runs after an
    explicit human approval, since it's the one action that reaches outside
    the document corpus.

    Degrades instead of crashing: if Tavily fails (missing/invalid key,
    network, quota), log at ERROR and continue to generate with the corpus
    documents only; the generate prompt's abstention rule keeps that answer
    honest. Before this, any Tavily failure surfaced as a 500 from /resume.
    """
    question = state["question"]
    print(f"--- WEB SEARCH (Tavily): {question} ---")
    try:
        response = _get_tavily().search(query=question, max_results=config.WEB_SEARCH_K)
    except Exception:
        logger.exception("web_search failed; continuing without web results")
        return {"web_search": False}

    web_docs: List[Document] = [
        Document(
            page_content=r["content"],
            metadata={"source": r["url"], "title": r.get("title", ""), "web": True},
        )
        for r in response.get("results", [])
    ]
    return {"documents": state["documents"] + web_docs, "web_search": False}


# =========================================================================
# Day 21 — global step budget
# =========================================================================
# MAX_RETRIEVAL_LOOPS and MAX_GENERATION_LOOPS cap each loop on its own; this
# is the budget above both. LangGraph fills state["remaining_steps"] from the
# recursion limit, which build_graph sets to config.MAX_GRAPH_STEPS. A router
# only takes a branch if every node it commits to, all the way to END, still
# fits. Otherwise it takes the cheapest way out, so the run ends with a
# best-effort answer instead of a GraphRecursionError (an HTTP 500).

# Nodes each branch still needs before END.
STEPS_REGENERATE = 3      # generate -> grade_answer -> finalize
STEPS_WEB_PATH = 5        # gate -> web_search -> generate -> grade_answer -> finalize
STEPS_REWRITE_ROUND = 6   # transform_query -> retrieve -> grade_documents, then finish


def _fits(state: GraphState, steps: int) -> bool:
    """Can `steps` more nodes still run? A router sees the same
    remaining_steps as the node it follows; the next node gets one fewer, and
    one with 0 left can't run. Missing outside a compiled graph (router unit
    tests), which means no limit."""
    return state.get("remaining_steps", 10**6) - 1 >= steps


def decide_after_grading(state: GraphState) -> str:
    """Conditional edge after ``grade_documents`` — the CRAG router.

    - ``"generate"``        — enough relevant chunks survived grading.
    - ``"transform_query"`` — weak retrieval, rewrite budget left: re-retrieve.
    - ``"web_search"``      — weak retrieval, budget exhausted: web fallback
                              (the graph maps this verdict to the approval gate).
    """
    if not state.get("web_search", False):
        return "generate"
    if state.get("loop_count", 0) < config.MAX_RETRIEVAL_LOOPS:
        if _fits(state, STEPS_REWRITE_ROUND):
            return "transform_query"
        print("--- STEP BUDGET: no room for another rewrite ---")
    if _fits(state, STEPS_WEB_PATH):
        return "web_search"
    print("--- STEP BUDGET: spent -> answering from current documents ---")
    return "generate"


# =========================================================================
# Layer 3 — Self-RAG: grade_answer + routing
# =========================================================================
#
# Two independent verdicts on the generated answer, each a structured-output
# call: grounding (Self-RAG "ISSUP" — supported by docs/history?) and
# usefulness (Self-RAG "ISUSE" — addresses the question?).
# `decide_after_answer` routes on the pair: accept, regenerate (capped by
# MAX_GENERATION_LOOPS), or rewrite-and-re-retrieve (capped by the shared
# MAX_RETRIEVAL_LOOPS budget).


class GradeHallucination(BaseModel):
    """Is the generated answer grounded in the retrieved facts?"""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if every substantive claim in the answer is supported by the "
            "provided facts; 'no' if the answer contains information not present "
            "in the facts (a hallucination)."
        )
    )


class GradeAnswer(BaseModel):
    """Does the generated answer actually resolve the user's question?"""

    binary_score: Literal["yes", "no"] = Field(
        description="Does the answer address and resolve the question? 'yes' or 'no'."
    )


HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an LLM answer is grounded in / "
            "supported by a set of retrieved facts, OR by the conversation history "
            "below (an answer about what was discussed earlier is grounded if it "
            "accurately reflects that history). Ignore inline citation markers like "
            "[1]. Grade 'yes' if every substantive claim is supported by the facts "
            "or the history; grade 'no' if the answer asserts anything not present "
            "in either. Judge only grounding, not whether the answer is a good "
            "reply.",
        ),
        (
            "human",
            "Set of facts:\n\n{documents}\n\nConversation history:\n\n{history}\n\n"
            "LLM answer:\n\n{generation}",
        ),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an LLM answer addresses and "
            "resolves a user question. Grade 'yes' if it does; grade 'no' if it is "
            "off-topic, evasive, or answers a different question. Judge only "
            "relevance to the question, not whether it is grounded in sources.",
        ),
        (
            "human",
            "User question:\n\n{question}\n\nLLM answer:\n\n{generation}",
        ),
    ]
)


def grade_answer(state: GraphState) -> GraphState:
    """Self-RAG: grade the generated answer for grounding, then usefulness.

    The usefulness grader only runs if the answer is grounded — there's no
    point asking "is this on-topic?" about a hallucinated answer.

    An honest abstention short-circuits to accepted: looping on a correct
    "I don't know" would only waste calls.

    When the answer is NOT grounded we bump ``generate_count`` here (rather than
    in ``generate``), so the counter measures *hallucination retries* specifically
    and stays independent of the query-rewrite budget (``loop_count``).
    """
    question = state["question"]
    generation = state["generation"]

    # Honest "I don't know" — grounded and appropriate; accept and stop.
    if ABSTENTION.lower() in generation.lower():
        print("--- GRADE ANSWER: honest abstention -> accept ---")
        return {"answer_grounded": True, "answer_useful": True}

    facts = _format_context(state["documents"])
    history_text = _format_history(state.get("messages", []))

    hallu_grader = HALLUCINATION_PROMPT | _get_llm().with_structured_output(
        GradeHallucination
    )
    grounded = (
        hallu_grader.invoke(
            {"documents": facts, "history": history_text, "generation": generation}
        ).binary_score
        == "yes"
    )

    if not grounded:
        retries = state.get("generate_count", 0) + 1
        print(f"--- GRADE ANSWER: NOT grounded (hallucination), retry {retries} ---")
        return {"answer_grounded": False, "answer_useful": False,
                "generate_count": retries}

    ans_grader = ANSWER_PROMPT | _get_llm().with_structured_output(GradeAnswer)
    useful = (
        ans_grader.invoke({"question": question, "generation": generation}).binary_score
        == "yes"
    )
    print(
        f"--- GRADE ANSWER: grounded=yes, addresses-question="
        f"{'yes -> accept' if useful else 'no -> re-retrieve'} ---"
    )
    return {"answer_grounded": True, "answer_useful": useful}


def decide_after_answer(state: GraphState) -> str:
    """Conditional edge after ``grade_answer`` — the Self-RAG router.

    - ``"end"``             — grounded AND useful (accept), OR a corrective
                              budget is exhausted (best effort beats spinning).
    - ``"generate"``        — hallucinated, generation budget left: regenerate.
    - ``"transform_query"`` — grounded but off-topic, rewrite budget left:
                              re-retrieve better context.
    """
    grounded = state.get("answer_grounded", True)
    useful = state.get("answer_useful", True)

    if grounded and useful:
        return "end"

    if not grounded:
        if state.get("generate_count", 0) < config.MAX_GENERATION_LOOPS:
            if _fits(state, STEPS_REGENERATE):
                return "generate"
            print("--- STEP BUDGET: spent -> accept best effort ---")
            return "end"
        print("--- SELF-RAG: generation budget exhausted -> accept best effort ---")
        return "end"

    # grounded but not useful -> better retrieval, if we still have budget
    if state.get("loop_count", 0) < config.MAX_RETRIEVAL_LOOPS:
        if _fits(state, STEPS_REWRITE_ROUND):
            return "transform_query"
        print("--- STEP BUDGET: spent -> accept best effort ---")
        return "end"
    print("--- SELF-RAG: rewrite budget exhausted -> accept best effort ---")
    return "end"


# =========================================================================
# Layer 4 — memory: finalize
# =========================================================================


def finalize(state: GraphState) -> GraphState:
    """Terminal node: append the accepted answer to the conversation history.

    This lives *inside* the graph (routed to by ``decide_after_answer``'s
    ``"end"`` verdict) rather than as a ``graph.update_state`` call in main.py,
    for two reasons:

    1. Atomicity — the AI turn lands in the same checkpoint as the rest of the
       turn, so a crash between "answer produced" and "history updated" can't
       leave the thread's memory missing its own reply.
    2. ``update_state`` with no ``as_node`` attributes the write to the last
       node that ran and then re-evaluates that node's conditional edges to
       plan the next step — which called ``decide_after_answer`` a second time
       per turn (the doubled "budget exhausted" print).
    """
    return {"messages": [AIMessage(content=state["generation"])]}


# =========================================================================
# Layer 4 — HITL: the one approval gate, placed before web_search
# =========================================================================


def human_approval_gate(state: GraphState) -> GraphState:
    """Pause before the web-search fallback — the highest-stakes action, since
    it's the only point where the graph reaches outside the document corpus.

    Surfaces the pending query and blocks via ``interrupt()`` until the caller
    resumes the run with ``Command(resume=...)``. Requires a checkpointer
    (main.py's ``build_graph(checkpointer=saver)``) — without one, `interrupt`
    has nowhere to save the paused state and will raise.

    On resume the node re-runs from the top, so nothing above the
    ``interrupt()`` call may have side effects. Normalization of the human's
    reply happens here and only here — one comparison site, and anything
    unrecognized fails closed (denied).
    """
    decision = interrupt(
        {
            "reason": "Retrieved documents were insufficient; the graph wants "
                      "to fall back to a web search.",
            "question": state["question"],
            "instruction": "Approve this web search? (y/n)",
        }
    )
    approved = str(decision).strip().lower() in ("y", "yes", "true")
    print(f"--- HITL: web search {'approved' if approved else 'denied'} ---")
    return {"web_search_approved": approved}


def decide_after_approval(state: GraphState) -> str:
    """Conditional edge after ``human_approval_gate``.

    - ``"web_search"`` — approved: proceed to the Tavily fallback.
    - ``"generate"``   — denied: answer from whatever (possibly weak) documents
                         survived grading rather than dead-ending the turn; the
                         generate prompt's abstention rule keeps this honest.
    """
    if state.get("web_search_approved", False):
        return "web_search"
    print("--- HITL: denied -> generating best-effort answer without web search ---")
    return "generate"
