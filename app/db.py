"""
db.py — SQLite persistence via aiosqlite for:
  - virtual API keys (llm-shield's own auth, separate from upstream provider keys)
  - the usage ledger that powers the spend circuit breaker + dashboard
  - the exact-match response cache

A single shared aiosqlite connection is reused across the app (SQLite handles
concurrent readers fine; writes are short and serialized by SQLite itself,
which is plenty for a local single-process proxy).
"""
from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    daily_budget_usd REAL NOT NULL,
    rate_limit_rpm INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key TEXT NOT NULL,
    day TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    redaction_count INTEGER NOT NULL DEFAULT 0,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_key_day ON usage_log(api_key, day);

CREATE TABLE IF NOT EXISTS response_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""


@dataclass
class ApiKeyRecord:
    key: str
    name: str
    daily_budget_usd: float
    rate_limit_rpm: int
    is_active: bool
    is_admin: bool = False


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        await self._migrate()

    async def _migrate(self) -> None:
        """Best-effort upgrade path for DBs created before is_admin existed.
        SQLite has no 'ADD COLUMN IF NOT EXISTS', so we probe and swallow
        the duplicate-column error rather than tracking a version number —
        overkill for a single additive column, but this is the pattern to
        extend if the schema grows more migrations later."""
        try:
            await self.conn.execute("ALTER TABLE api_keys ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
            await self.conn.commit()
        except aiosqlite.OperationalError:
            pass  # column already exists

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected — call connect() first.")
        return self._conn

    # -- API keys -----------------------------------------------------

    async def create_api_key(
        self, name: str, daily_budget_usd: float, rate_limit_rpm: int, is_admin: bool = False
    ) -> ApiKeyRecord:
        key = f"llmshield-{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            "INSERT INTO api_keys (key, name, daily_budget_usd, rate_limit_rpm, is_active, is_admin, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (key, name, daily_budget_usd, rate_limit_rpm, int(is_admin), now),
        )
        await self.conn.commit()
        return ApiKeyRecord(key, name, daily_budget_usd, rate_limit_rpm, True, is_admin)

    async def get_api_key(self, key: str) -> ApiKeyRecord | None:
        cursor = await self.conn.execute(
            "SELECT key, name, daily_budget_usd, rate_limit_rpm, is_active, is_admin FROM api_keys WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ApiKeyRecord(
            row["key"], row["name"], row["daily_budget_usd"], row["rate_limit_rpm"],
            bool(row["is_active"]), bool(row["is_admin"]),
        )

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        cursor = await self.conn.execute(
            "SELECT key, name, daily_budget_usd, rate_limit_rpm, is_active, is_admin FROM api_keys ORDER BY created_at"
        )
        rows = await cursor.fetchall()
        return [
            ApiKeyRecord(
                r["key"], r["name"], r["daily_budget_usd"], r["rate_limit_rpm"],
                bool(r["is_active"]), bool(r["is_admin"]),
            )
            for r in rows
        ]

    async def revoke_api_key(self, key: str) -> bool:
        cursor = await self.conn.execute("UPDATE api_keys SET is_active = 0 WHERE key = ?", (key,))
        await self.conn.commit()
        return cursor.rowcount > 0

    async def delete_api_key(self, key: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM api_keys WHERE key = ?", (key,))
        await self.conn.commit()
        return cursor.rowcount > 0

    async def has_any_keys(self) -> bool:
        cursor = await self.conn.execute("SELECT 1 FROM api_keys LIMIT 1")
        return (await cursor.fetchone()) is not None

    # -- Usage / spend --------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def get_daily_spend(self, api_key: str) -> float:
        cursor = await self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM usage_log WHERE api_key = ? AND day = ?",
            (api_key, self._today()),
        )
        row = await cursor.fetchone()
        return float(row["total"])

    async def record_usage(
        self,
        api_key: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        redaction_count: int,
        cache_hit: bool,
        latency_ms: int,
    ) -> None:
        await self.conn.execute(
            "INSERT INTO usage_log (api_key, day, timestamp, provider, model, prompt_tokens, "
            "completion_tokens, cost_usd, redaction_count, cache_hit, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                api_key,
                self._today(),
                datetime.now(timezone.utc).isoformat(),
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                redaction_count,
                int(cache_hit),
                latency_ms,
            ),
        )
        await self.conn.commit()

    async def get_dashboard_stats(self) -> dict:
        today = self._today()

        cursor = await self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS spend, "
            "COALESCE(SUM(redaction_count),0) AS redactions, "
            "COALESCE(SUM(cache_hit),0) AS cache_hits "
            "FROM usage_log WHERE day = ?",
            (today,),
        )
        today_row = await cursor.fetchone()

        cursor = await self.conn.execute(
            "SELECT model, COUNT(*) AS requests, COALESCE(SUM(cost_usd),0) AS spend "
            "FROM usage_log WHERE day = ? GROUP BY model ORDER BY spend DESC LIMIT 10",
            (today,),
        )
        by_model = [dict(r) for r in await cursor.fetchall()]

        cursor = await self.conn.execute(
            "SELECT day, COALESCE(SUM(cost_usd),0) AS spend, COUNT(*) AS requests "
            "FROM usage_log GROUP BY day ORDER BY day DESC LIMIT 7"
        )
        last_7_days = [dict(r) for r in await cursor.fetchall()][::-1]

        cursor = await self.conn.execute(
            "SELECT timestamp, provider, model, prompt_tokens, completion_tokens, "
            "cost_usd, redaction_count, cache_hit, latency_ms FROM usage_log "
            "ORDER BY id DESC LIMIT 25"
        )
        recent = [dict(r) for r in await cursor.fetchall()]

        n_requests = today_row["n"] or 0
        cache_hits = today_row["cache_hits"] or 0

        return {
            "today_spend_usd": round(today_row["spend"], 6),
            "today_requests": n_requests,
            "today_redactions": today_row["redactions"],
            "cache_hit_rate": round(cache_hits / n_requests, 4) if n_requests else 0.0,
            "spend_by_model": by_model,
            "spend_last_7_days": last_7_days,
            "recent_requests": recent,
        }

    # -- Response cache ---------------------------------------------------

    async def get_cached_response(self, cache_key: str) -> str | None:
        now = time.time()
        cursor = await self.conn.execute(
            "SELECT response_json FROM response_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, now),
        )
        row = await cursor.fetchone()
        return row["response_json"] if row else None

    async def set_cached_response(self, cache_key: str, response_json: str, ttl_seconds: int) -> None:
        now = time.time()
        await self.conn.execute(
            "INSERT OR REPLACE INTO response_cache (cache_key, response_json, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (cache_key, response_json, now, now + ttl_seconds),
        )
        await self.conn.commit()

    async def purge_expired_cache(self) -> int:
        cursor = await self.conn.execute(
            "DELETE FROM response_cache WHERE expires_at <= ?", (time.time(),)
        )
        await self.conn.commit()
        return cursor.rowcount


def generate_master_key() -> str:
    return f"llmshield-master-{secrets.token_hex(24)}"
