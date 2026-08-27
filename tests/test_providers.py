import httpx
import pytest

from app.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderError,
    StreamUsage,
    build_provider,
    resolve_provider_name,
)
from app.config import Settings


def _client_with_transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(coro):
    import asyncio

    return asyncio.run(coro)


# -- resolve_provider_name ---------------------------------------------------

def test_resolve_explicit_provider_wins():
    routes = {"gpt-": "openai"}
    assert resolve_provider_name("claude-3-5-sonnet", "anthropic", routes, "openai") == "anthropic"


def test_resolve_by_prefix():
    routes = {"gpt-": "openai", "claude-": "anthropic"}
    assert resolve_provider_name("claude-3-5-haiku", None, routes, "openai") == "anthropic"


def test_resolve_falls_back_to_default():
    routes = {"gpt-": "openai"}
    assert resolve_provider_name("some-unknown-model", None, routes, "openai") == "openai"


# -- build_provider -----------------------------------------------------------

def test_build_provider_requires_configured_key():
    settings = Settings(_env_file=None, openai_api_key=None)
    with pytest.raises(ProviderError) as exc_info:
        build_provider("openai", settings)
    assert exc_info.value.status_code == 500


def test_build_provider_unknown_name_raises():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    with pytest.raises(ProviderError) as exc_info:
        build_provider("not-a-real-provider", settings)
    assert exc_info.value.status_code == 400


def test_build_provider_generic_requires_base_url():
    settings = Settings(_env_file=None, generic_base_url=None)
    with pytest.raises(ProviderError):
        build_provider("generic", settings)


# -- OpenAICompatibleProvider ---------------------------------------------------

def test_openai_compatible_send_returns_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    provider = OpenAICompatibleProvider("openai", "https://api.openai.com/v1", "sk-test")

    async def go():
        async with _client_with_transport(handler) as client:
            return await provider.send(client, {"model": "gpt-4o-mini", "messages": []})

    response = run(go())
    assert response.status_code == 200
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 3


def test_openai_compatible_send_network_error_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    provider = OpenAICompatibleProvider("openai", "https://api.openai.com/v1", "sk-test")

    async def go():
        async with _client_with_transport(handler) as client:
            return await provider.send(client, {"model": "gpt-4o-mini", "messages": []})

    with pytest.raises(ProviderError) as exc_info:
        run(go())
    assert exc_info.value.status_code == 502


def test_openai_compatible_send_stream_yields_frames_and_tracks_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"id":"1","choices":[{"delta":{"content":"Hi"}}]}\n\n'
            b'data: {"id":"1","choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    provider = OpenAICompatibleProvider("openai", "https://api.openai.com/v1", "sk-test")
    usage = StreamUsage()

    async def go():
        chunks = []
        async with _client_with_transport(handler) as client:
            async for chunk in provider.send_stream(client, {"model": "gpt-4o-mini", "messages": []}, usage):
                chunks.append(chunk)
        return chunks

    chunks = run(go())
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert usage.full_text == "Hi"
    assert usage.prompt_tokens == 4
    assert usage.completion_tokens == 2
    assert usage.finished is True


# -- AnthropicProvider ----------------------------------------------------------

def test_anthropic_request_translation_hoists_system_prompt():
    provider = AnthropicProvider("https://api.anthropic.com/v1", "key", "2023-06-01")
    openai_body = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
    }
    anthropic_body = provider._to_anthropic_request(openai_body)
    assert anthropic_body["system"] == "be nice"
    assert anthropic_body["messages"] == [{"role": "user", "content": "hi"}]
    assert anthropic_body["max_tokens"] == 1024  # default when caller omits it


def test_anthropic_request_translation_folds_unknown_roles_into_user():
    provider = AnthropicProvider("https://api.anthropic.com/v1", "key", "2023-06-01")
    openai_body = {"model": "claude-3-5-sonnet", "messages": [{"role": "tool", "content": "result"}]}
    anthropic_body = provider._to_anthropic_request(openai_body)
    assert anthropic_body["messages"] == [{"role": "user", "content": "result"}]


def test_anthropic_response_translation_extracts_text_and_usage():
    provider = AnthropicProvider("https://api.anthropic.com/v1", "key", "2023-06-01")
    raw = {
        "id": "msg_1",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": "hello there"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    openai_shaped, prompt_tokens, completion_tokens = provider._to_openai_response(raw, "claude-3-5-sonnet")
    assert openai_shaped["choices"][0]["message"]["content"] == "hello there"
    assert openai_shaped["choices"][0]["finish_reason"] == "stop"
    assert prompt_tokens == 10
    assert completion_tokens == 4


def test_anthropic_send_translates_error_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    provider = AnthropicProvider("https://api.anthropic.com/v1", "key", "2023-06-01")

    async def go():
        async with _client_with_transport(handler) as client:
            return await provider.send(client, {"model": "claude-3-5-sonnet", "messages": []})

    response = run(go())
    assert response.status_code == 400
    assert response.body["error"]["type"] == "anthropic_error"
    assert response.body["error"]["message"] == "bad request"


def test_anthropic_send_stream_translates_sse_events():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'event: message_start\n'
            b'data: {"message":{"usage":{"input_tokens":7}}}\n\n'
            b'event: content_block_delta\n'
            b'data: {"delta":{"type":"text_delta","text":"hi"}}\n\n'
            b'event: message_delta\n'
            b'data: {"usage":{"output_tokens":2},"delta":{"stop_reason":"end_turn"}}\n\n'
            b'event: message_stop\n'
            b'data: {}\n\n'
        )
        return httpx.Response(200, content=body)

    provider = AnthropicProvider("https://api.anthropic.com/v1", "key", "2023-06-01")
    usage = StreamUsage()

    async def go():
        chunks = []
        async with _client_with_transport(handler) as client:
            async for chunk in provider.send_stream(client, {"model": "claude-3-5-sonnet", "messages": []}, usage):
                chunks.append(chunk)
        return chunks

    chunks = run(go())
    assert usage.full_text == "hi"
    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 2
    assert usage.finished is True
    assert chunks[-1] == b"data: [DONE]\n\n"


def test_anthropic_send_stream_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b"rate limited")

    provider = AnthropicProvider("https://api.anthropic.com/v1", "key", "2023-06-01")
    usage = StreamUsage()

    async def go():
        async with _client_with_transport(handler) as client:
            async for _ in provider.send_stream(client, {"model": "claude-3-5-sonnet", "messages": []}, usage):
                pass

    with pytest.raises(ProviderError) as exc_info:
        run(go())
    assert exc_info.value.status_code == 429
