import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.providers import ProviderResponse


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


def test_v1_models_lists_configured_providers(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "gpt-4o-mini" in ids  # openai_api_key is configured in the fixture
