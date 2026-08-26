import asyncio

from app.db import Database, generate_master_key


def run(coro):
    return asyncio.run(coro)


async def _db(tmp_path) -> Database:
    db = Database(str(tmp_path / "llm_shield.db"))
    await db.connect()
    return db


def test_generate_master_key_has_expected_prefix():
    key = generate_master_key()
    assert key.startswith("llmshield-master-")
    assert generate_master_key() != generate_master_key()  # not deterministic


def test_create_and_get_api_key(tmp_path):
    async def go():
        db = await _db(tmp_path)
        created = await db.create_api_key("my-app", 5.0, 60, is_admin=False)
        fetched = await db.get_api_key(created.key)
        await db.close()
        return created, fetched

    created, fetched = run(go())
    assert fetched is not None
    assert fetched.key == created.key
    assert fetched.name == "my-app"
    assert fetched.daily_budget_usd == 5.0
    assert fetched.rate_limit_rpm == 60
    assert fetched.is_active is True
    assert fetched.is_admin is False


def test_get_api_key_returns_none_for_unknown_key(tmp_path):
    async def go():
        db = await _db(tmp_path)
        record = await db.get_api_key("does-not-exist")
        await db.close()
        return record

    assert run(go()) is None


def test_list_api_keys_returns_all_created_keys(tmp_path):
    async def go():
        db = await _db(tmp_path)
        await db.create_api_key("a", 1.0, 10)
        await db.create_api_key("b", 2.0, 20)
        records = await db.list_api_keys()
        await db.close()
        return records

    records = run(go())
    assert {r.name for r in records} == {"a", "b"}


def test_revoke_api_key_deactivates_it(tmp_path):
    async def go():
        db = await _db(tmp_path)
        created = await db.create_api_key("a", 1.0, 10)
        ok = await db.revoke_api_key(created.key)
        after = await db.get_api_key(created.key)
        await db.close()
        return ok, after

    ok, after = run(go())
    assert ok is True
    assert after.is_active is False


def test_revoke_unknown_key_returns_false(tmp_path):
    async def go():
        db = await _db(tmp_path)
        ok = await db.revoke_api_key("nope")
        await db.close()
        return ok

    assert run(go()) is False


def test_has_any_keys(tmp_path):
    async def go():
        db = await _db(tmp_path)
        before = await db.has_any_keys()
        await db.create_api_key("a", 1.0, 10)
        after = await db.has_any_keys()
        await db.close()
        return before, after

    before, after = run(go())
    assert before is False
    assert after is True


def test_record_usage_and_daily_spend(tmp_path):
    async def go():
        db = await _db(tmp_path)
        await db.record_usage(
            api_key="k1", provider="openai", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50, cost_usd=0.01,
            redaction_count=2, cache_hit=False, latency_ms=120,
        )
        await db.record_usage(
            api_key="k1", provider="openai", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50, cost_usd=0.02,
            redaction_count=0, cache_hit=True, latency_ms=5,
        )
        spend = await db.get_daily_spend("k1")
        other_spend = await db.get_daily_spend("unused-key")
        await db.close()
        return spend, other_spend

    spend, other_spend = run(go())
    assert round(spend, 4) == 0.03
    assert other_spend == 0.0


def test_dashboard_stats_aggregate_correctly(tmp_path):
    async def go():
        db = await _db(tmp_path)
        await db.record_usage(
            api_key="k1", provider="openai", model="gpt-4o-mini",
            prompt_tokens=10, completion_tokens=5, cost_usd=0.05,
            redaction_count=1, cache_hit=False, latency_ms=100,
        )
        await db.record_usage(
            api_key="k1", provider="anthropic", model="claude-3-5-haiku",
            prompt_tokens=20, completion_tokens=10, cost_usd=0.10,
            redaction_count=0, cache_hit=True, latency_ms=50,
        )
        stats = await db.get_dashboard_stats()
        await db.close()
        return stats

    stats = run(go())
    assert stats["today_requests"] == 2
    assert round(stats["today_spend_usd"], 2) == 0.15
    assert stats["today_redactions"] == 1
    assert stats["cache_hit_rate"] == 0.5
    assert len(stats["spend_by_model"]) == 2
    assert len(stats["recent_requests"]) == 2


def test_response_cache_roundtrip_and_expiry(tmp_path):
    async def go():
        db = await _db(tmp_path)
        await db.set_cached_response("key1", '{"a": 1}', ttl_seconds=3600)
        hit = await db.get_cached_response("key1")

        await db.set_cached_response("key2", '{"b": 2}', ttl_seconds=-1)  # already expired
        miss = await db.get_cached_response("key2")

        purged = await db.purge_expired_cache()
        await db.close()
        return hit, miss, purged

    hit, miss, purged = run(go())
    assert hit == '{"a": 1}'
    assert miss is None
    assert purged == 1


def test_migration_is_idempotent_when_is_admin_already_exists(tmp_path):
    async def go():
        db_path = str(tmp_path / "llm_shield.db")
        db = Database(db_path)
        await db.connect()
        await db.close()

        # Reconnecting to an already-migrated DB must not raise even though
        # the ALTER TABLE will hit an already-existing column.
        db2 = Database(db_path)
        await db2.connect()
        works = await db2.has_any_keys()
        await db2.close()
        return works

    assert run(go()) is False
