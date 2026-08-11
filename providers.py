"""
providers.py — provider adapter layer.

llm-shield always speaks OpenAI's chat-completions shape to its own clients.
Internally, each adapter is responsible for translating that shape to
whatever the upstream provider actually expects, and translating the reply
back. This is what lets one client integration ("point base_url at
llm-shield") reach OpenAI, Anthropic, Gemini, or any self-hosted /
OpenAI-compatible model without the caller changing anything but the
`model` field.

Two adapter families:

- `OpenAICompatibleProvider`: used for OpenAI itself, Gemini (via Google's
  OpenAI-compatibility endpoint), and "generic" (Ollama, OpenRouter, Groq,
  Together, DeepSeek, vLLM, LM Studio, ...). These all already speak the
  same schema, so it's a thin passthrough — just different base_url/key.

- `AnthropicProvider`: Claude's Messages API has a materially different
  shape (system prompt is a top-level field, not a message; max_tokens is
  required; response content is a block list, not a `.choices[0].message`).
  This adapter does the request/response translation so callers never see
  the difference.
"""
from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx


class ProviderError(Exception):
    """Raised when a provider can't be reached or returns something we
    can't translate. Carries the HTTP status to forward to the client."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass
class ProviderResponse:
    body: dict
    status_code: int
    prompt_tokens: int
    completion_tokens: int


@dataclass
class StreamUsage:
    """Mutable accumulator passed into a streaming call and filled in as the
    stream progresses, so the caller can log accurate usage once the async
    generator driving the HTTP response has finished — SSE has no separate
    'return value' channel, so this side-channel object is the mechanism."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    full_text: str = field(default="")
    finished: bool = False


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def send(self, client: httpx.AsyncClient, openai_body: dict) -> ProviderResponse:
        """Send an OpenAI-shaped request body upstream and return an
        OpenAI-shaped response, regardless of what the upstream actually
        speaks natively."""
        raise NotImplementedError

    @abstractmethod
    def send_stream(
        self, client: httpx.AsyncClient, openai_body: dict, usage: StreamUsage
    ) -> AsyncIterator[bytes]:
        """Stream an OpenAI-shaped SSE response, regardless of what the
        upstream natively emits. Each yielded chunk is a ready-to-forward
        `data: {...}\\n\\n` frame (or `data: [DONE]\\n\\n`). `usage` is
        mutated in place as tokens/text arrive."""
        raise NotImplementedError


class OpenAICompatibleProvider(BaseProvider):
    """Works for OpenAI, Gemini (OpenAI-compat endpoint), and any other
    provider implementing the standard /chat/completions schema."""

    def __init__(self, name: str, base_url: str, api_key: str, extra_headers: dict | None = None) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

    async def send(self, client: httpx.AsyncClient, openai_body: dict) -> ProviderResponse:
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=openai_body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self.extra_headers,
                },
            )
        except httpx.RequestError as exc:
            raise ProviderError(502, f"{self.name} provider unreachable: {exc}") from exc

        try:
            body = resp.json()
        except ValueError:
            raise ProviderError(502, f"{self.name} returned a non-JSON response.")

        usage = body.get("usage", {})
        return ProviderResponse(
            body=body,
            status_code=resp.status_code,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def send_stream(
        self, client: httpx.AsyncClient, openai_body: dict, usage: StreamUsage
    ) -> AsyncIterator[bytes]:
        body = dict(openai_body)
        body["stream"] = True
        # Ask for a final usage-bearing chunk where the provider supports it
        # (OpenAI, and most OpenAI-compatible providers, honor this). If a
        # provider ignores it, `usage` just stays at its estimated defaults
        # and main.py falls back to counting the accumulated text.
        body.setdefault("stream_options", {"include_usage": True})

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
        ) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                raise ProviderError(resp.status_code, raw.decode(errors="ignore"))

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    usage.finished = True
                    yield b"data: [DONE]\n\n"
                    return

                try:
                    parsed = json.loads(payload)
                except ValueError:
                    continue  # skip malformed frame rather than killing the stream

                choices = parsed.get("choices") or []
                if choices:
                    delta_content = choices[0].get("delta", {}).get("content")
                    if delta_content:
                        usage.full_text += delta_content

                if "usage" in parsed and parsed["usage"]:
                    usage.prompt_tokens = parsed["usage"].get("prompt_tokens", usage.prompt_tokens)
                    usage.completion_tokens = parsed["usage"].get("completion_tokens", usage.completion_tokens)

                yield f"data: {payload}\n\n".encode()

            usage.finished = True


_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicProvider(BaseProvider):
    """Translates OpenAI chat-completions <-> Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, base_url: str, api_key: str, api_version: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version

    def _to_anthropic_request(self, openai_body: dict) -> dict:
        system_parts: list[str] = []
        messages: list[dict] = []

        for msg in openai_body.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                continue
            if role not in ("user", "assistant"):
                # Anthropic only accepts user/assistant in the messages list;
                # fold anything else (e.g. "tool") into a user turn rather
                # than dropping it silently.
                role = "user"
            messages.append({"role": role, "content": content})

        anthropic_body: dict = {
            "model": openai_body["model"],
            "messages": messages,
            # Anthropic requires max_tokens; OpenAI clients often omit it.
            "max_tokens": openai_body.get("max_tokens") or 1024,
        }
        if system_parts:
            anthropic_body["system"] = "\n\n".join(system_parts)
        if "temperature" in openai_body and openai_body["temperature"] is not None:
            anthropic_body["temperature"] = openai_body["temperature"]
        if "top_p" in openai_body and openai_body["top_p"] is not None:
            anthropic_body["top_p"] = openai_body["top_p"]

        return anthropic_body

    def _to_openai_response(self, anthropic_body: dict, model: str) -> tuple[dict, int, int]:
        text_parts = [
            block.get("text", "")
            for block in anthropic_body.get("content", [])
            if block.get("type") == "text"
        ]
        content = "".join(text_parts)

        usage = anthropic_body.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        finish_reason = _STOP_REASON_MAP.get(anthropic_body.get("stop_reason"), "stop")

        openai_shaped = {
            "id": anthropic_body.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": anthropic_body.get("model", model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return openai_shaped, prompt_tokens, completion_tokens

    async def send(self, client: httpx.AsyncClient, openai_body: dict) -> ProviderResponse:
        anthropic_request = self._to_anthropic_request(openai_body)

        try:
            resp = await client.post(
                f"{self.base_url}/messages",
                json=anthropic_request,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.api_version,
                    "Content-Type": "application/json",
                },
            )
        except httpx.RequestError as exc:
            raise ProviderError(502, f"anthropic provider unreachable: {exc}") from exc

        try:
            raw_body = resp.json()
        except ValueError:
            raise ProviderError(502, "anthropic returned a non-JSON response.")

        if resp.status_code >= 400:
            # Anthropic error shape: {"type": "error", "error": {"type": ..., "message": ...}}
            message = raw_body.get("error", {}).get("message", str(raw_body))
            return ProviderResponse(
                body={"error": {"message": message, "type": "anthropic_error"}},
                status_code=resp.status_code,
                prompt_tokens=0,
                completion_tokens=0,
            )

        openai_shaped, prompt_tokens, completion_tokens = self._to_openai_response(
            raw_body, openai_body["model"]
        )
        return ProviderResponse(
            body=openai_shaped,
            status_code=resp.status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _openai_chunk(stream_id: str, model: str, delta: dict, finish_reason: str | None) -> bytes:
        chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    async def send_stream(
        self, client: httpx.AsyncClient, openai_body: dict, usage: StreamUsage
    ) -> AsyncIterator[bytes]:
        anthropic_request = self._to_anthropic_request(openai_body)
        anthropic_request["stream"] = True
        model = openai_body["model"]
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"

        async with client.stream(
            "POST",
            f"{self.base_url}/messages",
            json=anthropic_request,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "Content-Type": "application/json",
            },
        ) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                raise ProviderError(resp.status_code, raw.decode(errors="ignore"))

            event_type: str | None = None
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                try:
                    data = json.loads(line[len("data:"):].strip())
                except ValueError:
                    continue

                if event_type == "message_start":
                    input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                    usage.prompt_tokens = input_tokens
                    yield self._openai_chunk(stream_id, model, {"role": "assistant", "content": ""}, None)

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        usage.full_text += text
                        yield self._openai_chunk(stream_id, model, {"content": text}, None)

                elif event_type == "message_delta":
                    output_tokens = data.get("usage", {}).get("output_tokens")
                    if output_tokens is not None:
                        usage.completion_tokens = output_tokens
                    stop_reason = data.get("delta", {}).get("stop_reason")
                    if stop_reason:
                        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")
                        yield self._openai_chunk(stream_id, model, {}, finish_reason)

                elif event_type == "message_stop":
                    usage.finished = True
                    yield b"data: [DONE]\n\n"
                    return

            usage.finished = True
            yield b"data: [DONE]\n\n"


def resolve_provider_name(model: str, explicit: str | None, provider_routes: dict[str, str], default: str) -> str:
    """Decide which provider a request should go to."""
    if explicit:
        return explicit
    for prefix, provider in provider_routes.items():
        if model.startswith(prefix):
            return provider
    return default


def build_provider(name: str, settings) -> BaseProvider:
    """Instantiate the right adapter for a resolved provider name."""
    if name == "openai":
        if not settings.openai_api_key:
            raise ProviderError(500, "OPENAI_API_KEY is not configured.")
        return OpenAICompatibleProvider("openai", settings.openai_base_url, settings.openai_api_key)

    if name == "anthropic":
        if not settings.anthropic_api_key:
            raise ProviderError(500, "ANTHROPIC_API_KEY is not configured.")
        return AnthropicProvider(settings.anthropic_base_url, settings.anthropic_api_key, settings.anthropic_version)

    if name == "gemini":
        if not settings.gemini_api_key:
            raise ProviderError(500, "GEMINI_API_KEY is not configured.")
        return OpenAICompatibleProvider("gemini", settings.gemini_base_url, settings.gemini_api_key)

    if name in ("generic", settings.generic_provider_label):
        if not settings.generic_base_url:
            raise ProviderError(
                500,
                "No GENERIC_BASE_URL configured (used for Ollama/OpenRouter/Groq/etc). "
                "Set GENERIC_BASE_URL and GENERIC_API_KEY in .env.",
            )
        return OpenAICompatibleProvider(
            settings.generic_provider_label,
            settings.generic_base_url,
            settings.generic_api_key or "not-required",
        )

    raise ProviderError(400, f"Unknown provider '{name}'. Configure it or fix provider_routes/model name.")
