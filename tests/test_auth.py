import asyncio

import pytest
from fastapi import HTTPException

from app.auth import RateLimiter, authenticate, require_admin
from app.db import ApiKeyRecord, Database


def run(coro):
    return asyncio.run(coro)


async def _make_db(tmp_path):
    db = Database(str(tmp_path / "llm_shield.db"))
    await db.connect()
    return db


# -- RateLimiter --------------------------------------------------------------

def test_rate_limiter_allows_up_to_the_limit():
    limiter = RateLimiter()
    for _ in range(5):
        limiter.check("key-a", limit_rpm=5)  # should not raise


def test_rate_limiter_blocks_over_the_limit():
    limiter = RateLimiter()
    for _ in range(5):
        limiter.check("key-b", limit_rpm=5)
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("key-b", limit_rpm=5)
    assert exc_info.value.status_code == 429


def test_rate_limiter_tracks_keys_independently():
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("key-c", limit_rpm=3)
    limiter.check("key-d", limit_rpm=3)  # different key, fresh window


# -- authenticate ---------------------------------------------------------------

def test_authenticate_disabled_returns_admin_anonymous_identity(tmp_path):
    db = run(_make_db(tmp_path))
    record = run(authenticate(None, db, enable_auth=False, fallback_daily_budget=5.0, fallback_rate_limit=60))
    assert record.key == "anonymous"
    assert record.is_admin is True
    run(db.close())


def test_authenticate_missing_header_raises_401(tmp_path):
    db = run(_make_db(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        run(authenticate(None, db, enable_auth=True, fallback_daily_budget=5.0, fallback_rate_limit=60))
    assert exc_info.value.status_code == 401
    run(db.close())


def test_authenticate_malformed_header_raises_401(tmp_path):
    db = run(_make_db(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        run(authenticate("not-a-bearer-token", db, True, 5.0, 60))
    assert exc_info.value.status_code == 401
    run(db.close())


def test_authenticate_unknown_key_raises_401(tmp_path):
    db = run(_make_db(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        run(authenticate("Bearer nonexistent-key", db, True, 5.0, 60))
    assert exc_info.value.status_code == 401
    run(db.close())


def test_authenticate_revoked_key_raises_401(tmp_path):
    db = run(_make_db(tmp_path))
    record = run(db.create_api_key("test", 5.0, 60))
    run(db.revoke_api_key(record.key))
    with pytest.raises(HTTPException) as exc_info:
        run(authenticate(f"Bearer {record.key}", db, True, 5.0, 60))
    assert exc_info.value.status_code == 401
    run(db.close())


def test_authenticate_valid_key_returns_its_record(tmp_path):
    db = run(_make_db(tmp_path))
    created = run(db.create_api_key("test", 7.5, 30))
    fetched = run(authenticate(f"Bearer {created.key}", db, True, 5.0, 60))
    assert fetched.key == created.key
    assert fetched.daily_budget_usd == 7.5
    run(db.close())


# -- require_admin --------------------------------------------------------------

def test_require_admin_passes_for_admin_key():
    record = ApiKeyRecord("k", "n", 5.0, 60, True, is_admin=True)
    require_admin(record)  # should not raise


def test_require_admin_raises_403_for_non_admin():
    record = ApiKeyRecord("k", "n", 5.0, 60, True, is_admin=False)
    with pytest.raises(HTTPException) as exc_info:
        require_admin(record)
    assert exc_info.value.status_code == 403
