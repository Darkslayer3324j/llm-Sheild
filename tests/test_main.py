import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.pricing import estimate_cost_usd
from app.providers import ProviderError, ProviderResponse


class _FakeProvider:
    name = "openai"

    def __init__(self, content: str) -> None:
        self._content = content

    async def send(self, client, body):
        return ProviderResponse(
            body={
                "id": "chatcmpl-1",
                "choices": [{"message": {"role": "assistant", "content": self._content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
            status_code=200,
            prompt_tokens=5,
            completion_tokens=5,
        )

    async def send_stream(self, client, body, usage):
        usage.prompt_tokens = 5
        usage.completion_tokens = 5
        chunk = {
            "id": "chatcmpl-1",
            "model": body.get("model"),
            "choices": [{"index": 0, "delta": {"content": self._content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        usage.finished = True


class _FailingProvider:
    """Simulates a primary provider that always errors, to exercise the
    fallback path."""

    name = "openai"

    async def send(self, client, body):
        raise ProviderError(502, "primary provider unreachable")

    async def send_stream(self, client, body, usage):
        raise ProviderError(502, "primary provider unreachable")
        yield  # pragma: no cover - unreachable, but keeps this an async generator


def _build_provider_with_failing_primary(fallback_content: str):
    """A fake build_provider(name, settings): fails for 'openai' (the
    primary in these tests), succeeds with fallback_content for anything
    else (the fallback provider)."""

    def _build(name, settings):
        if name == "openai":
            return _FailingProvider()
        return _FakeProvider(fallback_content)

    return _build


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module.settings, "db_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(main_module.settings, "enable_auth", False)
    monkeypatch.setattr(main_module.settings, "enable_cache", True)
    monkeypatch.setattr(main_module.settings, "openai_api_key", "sk-test")
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dashboard_is_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "llm-shield" in r.text


def test_chat_completions_redacts_pii_and_reports_headers(client, monkeypatch):
    monkeypatch.setattr(main_module, "build_provider", lambda name, settings: _FakeProvider("hi there"))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "email me at joe@example.com"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["X-LLMShield-Redactions"] == "1"
    assert r.headers["X-LLMShield-Categories"] == "EMAIL"
    assert r.headers["X-LLMShield-Provider"] == "openai"
    assert r.headers["X-LLMShield-Cache"] == "MISS"
    assert "X-Request-ID" in r.headers


def test_chat_completions_second_identical_call_is_a_cache_hit(client, monkeypatch):
    monkeypatch.setattr(main_module, "build_provider", lambda name, settings: _FakeProvider("plain reply"))
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello there"}]}

    r1 = client.post("/v1/chat/completions", json=body)
    r2 = client.post("/v1/chat/completions", json=body)

    assert r1.headers["X-LLMShield-Cache"] == "MISS"
    assert r2.headers["X-LLMShield-Cache"] == "HIT"
    assert r2.json()["choices"][0]["message"]["content"] == "plain reply"


def test_chat_completions_unmask_header_restores_original_value(client, monkeypatch):
    # Simulate an upstream that reflects the sanitized placeholder back verbatim.
    monkeypatch.setattr(main_module, "build_provider", lambda name, settings: _FakeProvider("sure, [EMAIL_1] noted"))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "contact joe@example.com"}]},
        headers={"X-LLMShield-Unmask": "true"},
    )
    assert r.status_code == 200
    assert "joe@example.com" in r.json()["choices"][0]["message"]["content"]


def test_chat_completions_without_unmask_header_stays_redacted(client, monkeypatch):
    monkeypatch.setattr(main_module, "build_provider", lambda name, settings: _FakeProvider("sure, [EMAIL_1] noted"))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "contact joe@example.com"}]},
    )
    assert "[EMAIL_1]" in r.json()["choices"][0]["message"]["content"]


def test_chat_completions_oversized_request_is_rejected(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "max_request_chars", 10)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "this is way more than ten characters"}]},
    )
    assert r.status_code == 413


def test_admin_endpoints_require_admin_when_auth_enabled(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "enable_auth", True)
    r = client.get("/api/stats")
    assert r.status_code == 401


def test_buffered_fallback_is_priced_and_logged_under_the_fallback_model(client, monkeypatch):
    # gpt-4o and claude-3-5-haiku have very different per-token pricing —
    # if a fallback response ever gets priced/logged under the originally
    # requested model instead of the model that actually answered, this
    # catches it.
    monkeypatch.setattr(main_module, "build_provider", _build_provider_with_failing_primary("fallback reply"))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-LLMShield-Fallback": "anthropic/claude-3-5-haiku"},
    )
    assert r.status_code == 200
    assert r.headers["X-LLMShield-Fallback-Used"] == "true"
    assert r.headers["X-LLMShield-Provider"] == "anthropic"

    expected_cost = estimate_cost_usd("claude-3-5-haiku", 5, 5)
    assert float(r.headers["X-LLMShield-Cost-USD"]) == pytest.approx(expected_cost)
    assert expected_cost != estimate_cost_usd("gpt-4o", 5, 5)  # sanity: prices really do differ

    stats = client.get("/api/stats").json()
    assert stats["spend_by_model"][0]["model"] == "claude-3-5-haiku"


def test_streaming_fallback_is_priced_and_logged_under_the_fallback_model(client, monkeypatch):
    monkeypatch.setattr(main_module, "build_provider", _build_provider_with_failing_primary("fallback reply"))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers={"X-LLMShield-Fallback": "anthropic/claude-3-5-haiku"},
    )
    assert r.status_code == 200
    assert "fallback reply" in r.text

    stats = client.get("/api/stats").json()
    assert stats["spend_by_model"][0]["model"] == "claude-3-5-haiku"
    expected_cost = estimate_cost_usd("claude-3-5-haiku", 5, 5)
    assert stats["spend_by_model"][0]["spend"] == pytest.approx(expected_cost)


def test_v1_models_lists_configured_providers(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "gpt-4o-mini" in ids  # openai_api_key is configured in the fixture
