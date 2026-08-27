"""
auth.py — virtual API key validation and per-key rate limiting.

llm-shield's own auth is intentionally separate from upstream provider keys:
virtual keys let you hand out access to your local proxy (e.g. to teammates,
CI, or different apps) with independent daily budgets and rate limits,
without ever sharing your real OpenAI/Anthropic/Gemini key.

Rate limiting is an in-memory sliding window — correct for a single local
process, not intended to be shared across multiple llm-shield instances.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException

from app.db import ApiKeyRecord, Database

_WINDOW_SECONDS = 60


class RateLimiter:
    """Sliding-window requests-per-minute limiter, keyed by virtual API key."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit_rpm: int) -> None:
        now = time.time()
        window = self._hits[key]

        while window and now - window[0] > _WINDOW_SECONDS:
            window.popleft()

        if len(window) >= limit_rpm:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - window[0])))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit_rpm} req/min for this key). "
                f"Retry after ~{retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)


async def authenticate(
    authorization: str | None,
    db: Database,
    enable_auth: bool,
    fallback_daily_budget: float,
    fallback_rate_limit: int,
) -> ApiKeyRecord:
    """Resolve the caller's virtual key record.

    If enable_auth is False, requests are treated as an implicit "anonymous"
    key that still gets spend-tracked and rate-limited under settings'
    defaults — so the circuit breaker is never fully bypassable, even with
    auth off.
    """
    if not enable_auth:
        return ApiKeyRecord(
            key="anonymous",
            name="anonymous (auth disabled)",
            daily_budget_usd=fallback_daily_budget,
            rate_limit_rpm=fallback_rate_limit,
            is_active=True,
            is_admin=True,  # no meaningful admin boundary when auth is off
        )

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use: Authorization: Bearer <llm-shield-virtual-key>",
        )

    key = authorization.split(" ", 1)[1].strip()
    record = await db.get_api_key(key)

    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not record.is_active:
        raise HTTPException(status_code=401, detail="This API key has been revoked.")

    return record


def require_admin(record: ApiKeyRecord) -> None:
    """Raise 403 unless the authenticated key is an admin key. Used to gate
    key-management endpoints (/api/keys) and the dashboard's spend data —
    without this, anyone who can reach the port sees everyone's usage."""
    if not record.is_admin:
        raise HTTPException(status_code=403, detail="This action requires an admin API key.")
