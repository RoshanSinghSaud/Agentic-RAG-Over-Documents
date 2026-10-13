"""Classify a question as simple or complex, for model routing (Day 21).

Rule-based on purpose: it costs nothing, adds no latency, and is easy to
explain and test. It errs toward "complex", because the two mistakes don't
cost the same: a hard question sent to the small model costs answer quality,
an easy one sent to the strong model costs a fraction of a cent.

Only `generate` is routed. The graders stay on the strong model: replaying 60
real grading calls, gpt-4.1-nano rejected 16 of the 30 chunks gpt-4o-mini
kept and gpt-5-nano 7 of 30. A grader that rejects good chunks sends the
question round the rewrite loop, which costs more than the cheaper model saves.

Measured on the golden set, routing generate alone saved 3%, and twice the
small model answered with no documents from its own knowledge (q07, q20),
which breaks the "answer only from the corpus" contract. Off by default.
"""
from __future__ import annotations

import re
from typing import Literal

# Comparisons and relationships need the model to hold two things at once.
_COMPARE = re.compile(
    r"\b(compare[sd]?|comparison|contrast|differ|differs|difference|differences|"
    r"versus|vs\.?|trade-?offs?|both|relationship|relate[sd]?)\b",
    re.IGNORECASE,
)
# Two questions in one: "What X does Y, and why ...?"
_MULTI_PART = re.compile(
    r",\s*(and|or|but)\s+(how|why|what|which|when|where|who)\b", re.IGNORECASE
)
MAX_SIMPLE_WORDS = 20

Route = Literal["simple", "complex"]


def classify(question: str) -> Route:
    q = question.strip()
    if (
        _COMPARE.search(q)
        or _MULTI_PART.search(q)
        or q.count("?") > 1
        or len(q.split()) > MAX_SIMPLE_WORDS
    ):
        return "complex"
    return "simple"
