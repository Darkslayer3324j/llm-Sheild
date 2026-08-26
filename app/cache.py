"""
cache.py — exact-match response caching.

Hashes (provider, model, sanitized message content, temperature, top_p)
into a cache key. Purely deterministic requests (temperature=0, or repeated
identical calls during development/testing/eval loops) hit the cache instead
of paying the upstream provider again — this is usually the single biggest
cost lever a proxy like this can pull, more impactful than PII scanning is
to the "cost" half of the product.

Cache correctness note: this is an EXACT match cache — it does not do
semantic/embedding similarity matching, so a rephrased-but-equivalent prompt
is a cache miss. Exact-match is the safe default: semantic caching can
return a plausible-but-wrong answer for a subtly different question, which
is a worse failure mode than "just call the API again."
"""
from __future__ import annotations

import hashlib
import json


def compute_cache_key(provider: str, sanitized_body: dict) -> str:
    """Build a stable hash over the request's semantically-relevant fields.
    Deliberately excludes fields like `user` that don't affect the output."""
    relevant = {
        "provider": provider,
        "model": sanitized_body.get("model"),
        "messages": sanitized_body.get("messages"),
        "temperature": sanitized_body.get("temperature"),
        "top_p": sanitized_body.get("top_p"),
        "max_tokens": sanitized_body.get("max_tokens"),
    }
    canonical = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
