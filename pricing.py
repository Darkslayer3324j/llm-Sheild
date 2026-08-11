"""
pricing.py — token counting and cost estimation.

Two honesty notes up front, because a cost proxy that lies about cost is
worse than no cost proxy at all:

1. Prices below are approximate, hand-maintained defaults (USD per 1M
   tokens) and WILL drift out of date. They're a starting point you should
   verify against each provider's current pricing page and edit here — this
   is a plain dict, not a magic source of truth. `pricing_overrides.json`
   (optional, same directory as the DB) is loaded on top if present, so you
   can update prices without touching code.

2. Only OpenAI-family models get an exact token count, via `tiktoken`.
   Anthropic and Gemini don't expose their tokenizer for local use, so
   those are estimated at ~4 characters/token, a widely-used rule of thumb
   that's usually within ~10-15% of the real count. Good enough for a
   circuit breaker (which cares about "getting close to the limit", not
   penny-perfect accounting) — not good enough for an invoice.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import tiktoken

logger = logging.getLogger("llm_shield.pricing")

# USD per 1,000,000 tokens. (input, output)
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # --- OpenAI ---
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    # --- Anthropic ---
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    # --- Gemini ---
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Fallback rate for any model not found above (kept deliberately mid-pack,
# not zero — an unknown model silently costing "$0" would defeat the point
# of a circuit breaker).
FALLBACK_PRICING: tuple[float, float] = (2.00, 6.00)

_CHARS_PER_TOKEN_ESTIMATE = 4


def _load_overrides(db_path: str) -> dict[str, tuple[float, float]]:
    override_path = Path(db_path).parent / "pricing_overrides.json"
    if not override_path.exists():
        return {}
    try:
        raw = json.loads(override_path.read_text())
        return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    except (ValueError, OSError, KeyError, IndexError) as exc:
        logger.warning("Failed to load pricing_overrides.json: %s", exc)
        return {}


def get_pricing(model: str, db_path: str = "llm_shield.db") -> tuple[float, float]:
    """Return (input_price_per_1m, output_price_per_1m) for a model name,
    matching by longest known prefix so e.g. 'gpt-4o-2024-08-06' resolves
    to the 'gpt-4o' entry."""
    table = {**DEFAULT_PRICING, **_load_overrides(db_path)}

    if model in table:
        return table[model]

    best_match: str | None = None
    for known_model in table:
        if model.startswith(known_model):
            if best_match is None or len(known_model) > len(best_match):
                best_match = known_model
    if best_match:
        return table[best_match]

    logger.info("No pricing entry for model '%s', using fallback rate.", model)
    return FALLBACK_PRICING


def _is_openai_family(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-", "text-embedding"))


_tiktoken_unavailable = False  # sticky flag: once a download fails, stop retrying every call


def count_tokens(text: str, model: str) -> int:
    """Count tokens in `text` for `model`. Exact for OpenAI-family models via
    tiktoken; a documented ~4-chars/token estimate for everything else.

    tiktoken lazily downloads its BPE merge file from a Microsoft-hosted
    blob on first use per encoding. In a firewalled or fully offline
    deployment (common for a "local, zero-trust" tool like this one) that
    download fails — we catch broad Exception here (not just the specific
    HTTP error type, since the failure mode varies by environment) and fall
    back to the char-based estimate rather than 500ing every request.
    """
    global _tiktoken_unavailable

    if not text:
        return 0

    if _is_openai_family(model) and not _tiktoken_unavailable:
        try:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.warning(
                "tiktoken unavailable (%s) — falling back to char-based token "
                "estimate for this and future requests this session.",
                exc,
            )
            _tiktoken_unavailable = True

    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, db_path: str = "llm_shield.db"
) -> float:
    input_price, output_price = get_pricing(model, db_path)
    return (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price
