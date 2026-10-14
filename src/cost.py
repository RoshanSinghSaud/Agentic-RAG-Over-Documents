"""Per-request token and cost accounting (Layer 8 Day 21).

A CostTracker is passed as a callback on each graph.invoke. Callbacks are
inherited by every child run, so LangChain calls the tracker's on_llm_end for
every chat-model call in every node; no node has to know it exists.
grade_documents runs its calls on threads (.batch), so the counters are
updated under the parent class's lock.

Prices are kept here, not fetched, so a request can be costed offline and the
number in the log is reproducible. Update them when OpenAI's prices change.

cost_usd is at LIST price: an upper bound that ignores OpenAI's prompt-cache
discount (a repeated prompt prefix of 1,024+ tokens is billed at half price
for these models). cached_input_tokens is logged next to it so the billed
cost can be derived. List price is also what keeps eval runs comparable:
re-running the golden set repeats identical prompts and hits the cache
(53-77% of input tokens on Day 21's reruns vs 0-6% on the first run), which
unique user questions don't. LangSmith shows the cache-discounted number.
"""
from __future__ import annotations

import json
import logging
import sys

from langchain_core.callbacks import UsageMetadataCallbackHandler

# USD per 1M tokens: (input, output). Standard tier, checked 2026-10-12.
# Keys are prefixes: OpenAI reports dated names such as gpt-4o-mini-2024-07-18.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-nano": (0.05, 0.40),  # reasoning tokens are billed as output
}


def price_for(model: str) -> tuple[float, float] | None:
    """Longest matching prefix wins, so a future "gpt-4o-mini-x" can't be
    priced as something shorter by accident. None = unknown model."""
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            return PRICES[name]
    return None


class CostTracker(UsageMetadataCallbackHandler):
    """Token usage per model (inherited) plus a count of LLM calls."""

    def __init__(self) -> None:
        super().__init__()
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        super().on_llm_end(response, **kwargs)
        with self._lock:
            self.llm_calls += 1

    def summary(self) -> dict:
        with self._lock:
            usage = {m: dict(u) for m, u in self.usage_metadata.items()}
            calls = self.llm_calls
        tokens_in = sum(u.get("input_tokens", 0) for u in usage.values())
        tokens_out = sum(u.get("output_tokens", 0) for u in usage.values())
        cached = sum((u.get("input_token_details") or {}).get("cache_read", 0) or 0
                     for u in usage.values())
        cost, unpriced = 0.0, []
        for model, u in usage.items():
            price = price_for(model)
            if price is None:
                unpriced.append(model)  # tokens still counted; cost undercounts
                continue
            cost += (u.get("input_tokens", 0) * price[0]
                     + u.get("output_tokens", 0) * price[1]) / 1e6
        out = {
            "llm_calls": calls,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cached_input_tokens": cached,
            "cost_usd": round(cost, 8),
            "models": {m: [u.get("input_tokens", 0), u.get("output_tokens", 0)]
                       for m, u in usage.items()},
        }
        if unpriced:
            out["unpriced_models"] = unpriced
        return out


# One JSON object per line on stdout, nothing else on the line. On EC2 the
# awslogs driver ships stdout to CloudWatch, and Logs Insights discovers the
# fields of a JSON log event by itself (fields cost_usd, route, ...), which it
# can't do for JSON behind the root logger's "INFO: name -" prefix.
# Never put question text in here: it's user content, and the rag-errors
# metric filter counts any event containing the word ERROR.
_cost_log = logging.getLogger("agentic_rag.cost")


def log_request_cost(**fields) -> None:
    if not _cost_log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _cost_log.addHandler(handler)
        _cost_log.setLevel(logging.INFO)
        _cost_log.propagate = False
    _cost_log.info(json.dumps({"event": "request_cost", **fields}, default=str))
