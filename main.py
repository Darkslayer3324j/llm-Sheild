"""
main.py — llm-shield's OpenAI-compatible local proxy.

Any client SDK that supports a custom base_url (OpenAI python SDK, LangChain,
raw httpx/requests) can point at http://localhost:8000/v1 and transparently
get: PII redaction, multi-provider routing (OpenAI/Anthropic/Gemini/any
OpenAI-compatible endpoint), streaming, a spend circuit breaker, per-key
rate limiting, response caching, and provider fallback on failure —
regardless of which upstream model actually serves the request.

Request lifecycle for POST /v1/chat/completions:
  1. Authenticate the caller's virtual llm-shield key (or "anonymous" if
     auth is disabled), enforce its per-minute rate limit and a request
     size guard.
  2. Sanitize every string message body (PII -> reversible placeholders).
  3. Resolve which upstream provider this model routes to.
  4. (non-streaming only) Check the response cache for an exact-match hit.
  5. Pre-flight budget check using an estimated cost.
  6. Forward to the provider — streaming or buffered — with one automatic
     fallback retry if a X-LLMShield-Fallback header was supplied and the
     primary provider fails.
  7. Record actual usage/cost, cache the response (non-streaming), and
     optionally unmask placeholders back to real values.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.auth import ApiKeyRecord, RateLimiter, authenticate, require_admin
from app.cache import compute_cache_key
from app.config import get_settings
from app.db import Database, generate_master_key
from app.models import ChatCompletionRequest, CreateKeyRequest, RedactionSummary
from app.pricing import DEFAULT_PRICING, count_tokens, estimate_cost_usd
from app.providers import ProviderError, StreamUsage, build_provider, resolve_provider_name
from app.sanitizer import SanitizationResult, SanitizerConfig, SanitizerEngine
from app.streaming import StreamUnmasker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_shield")

settings = get_settings()

sanitizer_config = SanitizerConfig(
    mask_api_keys=settings.mask_api_keys,
    mask_emails=settings.mask_emails,
    mask_ipv4=settings.mask_ip_addresses,
    mask_ipv6=settings.mask_ip_addresses,
    mask_credit_cards=settings.mask_credit_cards,
    mask_ssn=settings.mask_ssn,
    mask_phone_numbers=settings.mask_phone_numbers,
    mask_full_names=settings.mask_full_names,
)
sanitizer = SanitizerEngine(sanitizer_config)
rate_limiter = RateLimiter()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    db = Database(settings.db_path)
    await db.connect()
    app.state.db = db

    if settings.enable_auth and not await db.has_any_keys():
        master_key = generate_master_key()
        await db.conn.execute(
            "INSERT INTO api_keys (key, name, daily_budget_usd, rate_limit_rpm, is_active, is_admin, created_at) "
            "VALUES (?, ?, ?, ?, 1, 1, datetime('now'))",
            (master_key, "master (auto-provisioned)", settings.master_key_daily_budget_usd, settings.master_key_rate_limit_rpm),
        )
        await db.conn.commit()
        logger.info("=" * 72)
        logger.info("No API keys found — created an admin master key for you:")
        logger.info("  %s", master_key)
        logger.info("Use it as: Authorization: Bearer %s", master_key)
        logger.info("This key can manage other keys and view the dashboard.")
        logger.info("Save this now — it will not be printed again. (Rotate any time")
        logger.info("with the CLI: python -m app.cli keys revoke / keys create)")
        logger.info("=" * 72)

    logger.info(
        "llm-shield started | default_provider=%s | auth=%s | cache=%s",
        settings.default_provider, settings.enable_auth, settings.enable_cache,
    )
    yield

    await app.state.http_client.aclose()
    await db.close()


app = FastAPI(
    title="llm-shield",
    description="Local zero-trust LLM proxy: PII redaction, multi-provider routing, "
    "streaming, spend circuit breaker, caching, provider fallback.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Every request gets a correlation ID, echoed back and included in logs
    — makes it possible to trace one call across client logs, llm-shield
    logs, and (if you forward it) upstream provider logs."""
    request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:16]}")
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "default_provider": settings.default_provider}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    dashboard_file = STATIC_DIR / "dashboard.html"
    return HTMLResponse(dashboard_file.read_text())


async def _require_admin(request: Request, authorization: str | None) -> ApiKeyRecord:
    db: Database = request.app.state.db
    record = await authenticate(
        authorization, db, settings.enable_auth, settings.default_daily_budget_usd, settings.default_rate_limit_rpm
    )
    require_admin(record)
    return record


@app.get("/api/stats")
async def api_stats(request: Request, authorization: str | None = Header(default=None)) -> dict:
    """Aggregate spend/usage across ALL keys — admin-only, since a
    non-admin key holder shouldn't see other keys' usage."""
    await _require_admin(request, authorization)
    db: Database = request.app.state.db
    return await db.get_dashboard_stats()


@app.get("/api/keys")
async def api_list_keys(request: Request, authorization: str | None = Header(default=None)) -> list[dict]:
    await _require_admin(request, authorization)
    db: Database = request.app.state.db
    records = await db.list_api_keys()
    return [
        {
            "key": r.key, "name": r.name, "daily_budget_usd": r.daily_budget_usd,
            "rate_limit_rpm": r.rate_limit_rpm, "is_active": r.is_active, "is_admin": r.is_admin,
        }
        for r in records
    ]


@app.post("/api/keys")
async def api_create_key(
    request: Request, payload: CreateKeyRequest, authorization: str | None = Header(default=None)
) -> dict:
    await _require_admin(request, authorization)
    db: Database = request.app.state.db
    record = await db.create_api_key(payload.name, payload.daily_budget_usd, payload.rate_limit_rpm, payload.is_admin)
    return {
        "key": record.key, "name": record.name, "daily_budget_usd": record.daily_budget_usd,
        "rate_limit_rpm": record.rate_limit_rpm, "is_admin": record.is_admin,
    }


@app.delete("/api/keys/{key}")
async def api_revoke_key(key: str, request: Request, authorization: str | None = Header(default=None)) -> dict:
    await _require_admin(request, authorization)
    db: Database = request.app.state.db
    ok = await db.revoke_api_key(key)
    if not ok:
        raise HTTPException(status_code=404, detail="No such key.")
    return {"revoked": key}


@app.get("/v1/models")
async def list_models() -> dict:
    """Best-effort model list synthesized from the pricing table + configured
    providers. Several client SDKs (LangChain, some OpenAI-SDK forks) ping
    this on init to sanity-check connectivity — without it they can fail
    to initialize even though /v1/chat/completions would have worked fine."""
    configured = set()
    if settings.openai_api_key:
        configured.add("openai")
    if settings.anthropic_api_key:
        configured.add("anthropic")
    if settings.gemini_api_key:
        configured.add("gemini")
    if settings.generic_base_url:
        configured.add(settings.generic_provider_label)

    data = []
    for model_name in DEFAULT_PRICING:
        provider_name = resolve_provider_name(model_name, None, settings.provider_routes, settings.default_provider)
        if provider_name in configured:
            data.append({"id": model_name, "object": "model", "owned_by": provider_name})

    return {"object": "list", "data": data}


# --------------------------------------------------------------------------
# Sanitization helpers
# --------------------------------------------------------------------------

def _sanitize_request(payload: ChatCompletionRequest) -> tuple[dict, dict[str, SanitizationResult]]:
    body = payload.model_dump(exclude_none=True)
    results: dict[str, SanitizationResult] = {}

    for i, message in enumerate(body.get("messages", [])):
        content = message.get("content")
        if isinstance(content, str):
            result = sanitizer.sanitize(content)
            message["content"] = result.sanitized_text
            results[str(i)] = result
        # multimodal content (list of blocks) passes through untouched —
        # scanning image URLs/base64 blobs for PII is out of scope for v1.

    return body, results


def _merge_results(results: dict[str, SanitizationResult]) -> tuple[dict[str, str], RedactionSummary]:
    combined_mapping: dict[str, str] = {}
    combined_counts: dict[str, int] = {}
    for result in results.values():
        combined_mapping.update(result.mapping)
        for category, n in result.counts.items():
            combined_counts[category] = combined_counts.get(category, 0) + n

    summary = RedactionSummary(counts=combined_counts, total_redactions=sum(combined_counts.values()))
    return combined_mapping, summary


def _unmask_response_body(response_json: dict, mapping: dict[str, str]) -> dict:
    result = SanitizationResult(sanitized_text="", mapping=mapping)
    for choice in response_json.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = result.unmask(content)
    return response_json


def _estimate_prompt_text(body: dict) -> str:
    parts = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _wants_unmask(header_value: str | None) -> bool:
    return settings.allow_response_unmasking and header_value is not None and header_value.lower() == "true"


def _parse_fallback(header_value: str | None) -> tuple[str, str] | None:
    """Parse 'provider/model' from X-LLMShield-Fallback, e.g. 'anthropic/claude-3-5-haiku'."""
    if not header_value or "/" not in header_value:
        return None
    provider_name, model = header_value.split("/", 1)
    return provider_name.strip(), model.strip()


# --------------------------------------------------------------------------
# Main proxy route
# --------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
    x_llmshield_unmask: str | None = Header(default=None),
    x_llmshield_provider: str | None = Header(default=None),
    x_llmshield_fallback: str | None = Header(default=None),
):
    db: Database = request.app.state.db
    client: httpx.AsyncClient = request.app.state.http_client
    request_id = getattr(request.state, "request_id", "unknown")

    # 1. Auth + rate limit --------------------------------------------------
    key_record = await authenticate(
        authorization, db, settings.enable_auth, settings.default_daily_budget_usd, settings.default_rate_limit_rpm
    )
    rate_limiter.check(key_record.key, key_record.rate_limit_rpm)

    # 2. Sanitize -------------------------------------------------------------
    sanitized_body, per_message_results = _sanitize_request(payload)
    combined_mapping, redaction_summary = _merge_results(per_message_results)

    prompt_text = _estimate_prompt_text(sanitized_body)
    if len(prompt_text) > settings.max_request_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Request content ({len(prompt_text)} chars) exceeds the configured limit "
            f"of {settings.max_request_chars} chars (MAX_REQUEST_CHARS).",
        )

    # 3. Resolve provider -------------------------------------------------------
    explicit_provider = x_llmshield_provider or sanitized_body.pop("provider", None)
    provider_name = resolve_provider_name(
        payload.model, explicit_provider, settings.provider_routes, settings.default_provider
    )
    try:
        provider = build_provider(provider_name, settings)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    fallback = _parse_fallback(x_llmshield_fallback)

    # 4. Cache check (non-streaming only — SSE frames aren't cacheable as one blob) --
    cache_key = None
    if settings.enable_cache and not payload.stream:
        cache_key = compute_cache_key(provider_name, sanitized_body)
        cached = await db.get_cached_response(cache_key)
        if cached is not None:
            response_json = json.loads(cached)
            if combined_mapping and _wants_unmask(x_llmshield_unmask):
                response_json = _unmask_response_body(response_json, combined_mapping)
            response = JSONResponse(status_code=200, content=response_json)
            response.headers["X-LLMShield-Cache"] = "HIT"
            response.headers["X-LLMShield-Provider"] = provider_name
            response.headers["X-LLMShield-Redactions"] = str(redaction_summary.total_redactions)
            return response

    # 5. Pre-flight budget check ------------------------------------------------
    estimated_prompt_tokens = count_tokens(prompt_text, payload.model)
    estimated_completion_tokens = sanitized_body.get("max_tokens") or 1024
    estimated_cost = estimate_cost_usd(payload.model, estimated_prompt_tokens, estimated_completion_tokens, settings.db_path)

    daily_spend_so_far = await db.get_daily_spend(key_record.key)
    if daily_spend_so_far >= key_record.daily_budget_usd:
        raise HTTPException(
            status_code=402,
            detail=f"Daily budget of ${key_record.daily_budget_usd:.2f} already reached "
            f"(${daily_spend_so_far:.4f} spent today) for this key.",
        )
    if daily_spend_so_far + estimated_cost > key_record.daily_budget_usd:
        raise HTTPException(
            status_code=402,
            detail=f"This request's estimated cost (${estimated_cost:.4f}) would exceed the "
            f"remaining daily budget (${key_record.daily_budget_usd - daily_spend_so_far:.4f} left "
            f"of ${key_record.daily_budget_usd:.2f}).",
        )

    logger.info(
        "request_id=%s key=%s provider=%s model=%s stream=%s redactions=%d",
        request_id, key_record.name, provider_name, payload.model, payload.stream, redaction_summary.total_redactions,
    )

    # 6. Forward upstream -----------------------------------------------------
    if payload.stream:
        return await _handle_streaming(
            client, db, provider, provider_name, sanitized_body, payload.model,
            key_record, redaction_summary, fallback, request_id,
            combined_mapping, _wants_unmask(x_llmshield_unmask),
        )

    return await _handle_buffered(
        client, db, provider, provider_name, sanitized_body, payload.model,
        key_record, redaction_summary, combined_mapping, x_llmshield_unmask,
        cache_key, daily_spend_so_far, fallback, request_id,
    )


async def _send_with_fallback(client, provider, provider_name, sanitized_body, fallback, request_id):
    """Try the primary provider; on failure, if a fallback 'provider/model' was
    given, swap both and retry exactly once. Returns (response, provider_name_used)."""
    try:
        return await provider.send(client, sanitized_body), provider_name
    except ProviderError as primary_exc:
        if not fallback:
            raise
        fb_provider_name, fb_model = fallback
        logger.warning(
            "request_id=%s primary provider '%s' failed (%s) — falling back to %s/%s",
            request_id, provider_name, primary_exc.detail, fb_provider_name, fb_model,
        )
        fb_body = dict(sanitized_body)
        fb_body["model"] = fb_model
        fb_provider = build_provider(fb_provider_name, settings)
        return await fb_provider.send(client, fb_body), fb_provider_name


async def _handle_buffered(
    client, db, provider, provider_name, sanitized_body, requested_model,
    key_record, redaction_summary, combined_mapping, unmask_header,
    cache_key, daily_spend_so_far, fallback, request_id,
):
    start = time.monotonic()
    try:
        provider_response, used_provider = await _send_with_fallback(
            client, provider, provider_name, sanitized_body, fallback, request_id
        )
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    latency_ms = int((time.monotonic() - start) * 1000)

    if provider_response.status_code >= 400:
        return JSONResponse(status_code=provider_response.status_code, content=provider_response.body)

    actual_cost = estimate_cost_usd(
        requested_model, provider_response.prompt_tokens, provider_response.completion_tokens, settings.db_path
    )
    await db.record_usage(
        api_key=key_record.key, provider=used_provider, model=requested_model,
        prompt_tokens=provider_response.prompt_tokens, completion_tokens=provider_response.completion_tokens,
        cost_usd=actual_cost, redaction_count=redaction_summary.total_redactions,
        cache_hit=False, latency_ms=latency_ms,
    )

    if cache_key and used_provider == provider_name:  # don't cache under the original key if a fallback answered
        await db.set_cached_response(cache_key, json.dumps(provider_response.body), settings.cache_ttl_seconds)

    response_json = provider_response.body
    if combined_mapping and _wants_unmask(unmask_header):
        response_json = _unmask_response_body(response_json, combined_mapping)

    response = JSONResponse(status_code=provider_response.status_code, content=response_json)
    response.headers["X-LLMShield-Redactions"] = str(redaction_summary.total_redactions)
    response.headers["X-LLMShield-Categories"] = ",".join(redaction_summary.counts.keys())
    response.headers["X-LLMShield-Cost-USD"] = f"{actual_cost:.6f}"
    response.headers["X-LLMShield-Daily-Spend-USD"] = f"{daily_spend_so_far + actual_cost:.6f}"
    response.headers["X-LLMShield-Cache"] = "MISS"
    response.headers["X-LLMShield-Provider"] = used_provider
    if used_provider != provider_name:
        response.headers["X-LLMShield-Fallback-Used"] = "true"
    return response


async def _unmask_sse_stream(chunks: AsyncIterator[bytes], mapping: dict[str, str]) -> AsyncIterator[bytes]:
    """Wrap a raw OpenAI-shaped SSE byte stream, unmasking placeholder text
    in each delta.content as it arrives. Uses StreamUnmasker to correctly
    handle a placeholder split across two chunk boundaries (e.g. "...[EMAIL"
    then "_1]...") rather than naively string-replacing per chunk, which
    would silently miss split placeholders."""
    unmasker = StreamUnmasker(mapping)
    last_id, last_model = "", ""

    async for chunk in chunks:
        text = chunk.decode("utf-8", errors="ignore")
        if not text.startswith("data:"):
            yield chunk
            continue

        payload = text[len("data:"):].strip()
        if payload == "[DONE]":
            leftover = unmasker.flush()
            if leftover:
                flush_chunk = {
                    "id": last_id, "object": "chat.completion.chunk", "model": last_model,
                    "choices": [{"index": 0, "delta": {"content": leftover}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(flush_chunk)}\n\n".encode()
            yield chunk
            continue

        try:
            parsed = json.loads(payload)
        except ValueError:
            yield chunk
            continue

        last_id = parsed.get("id", last_id)
        last_model = parsed.get("model", last_model)

        choices = parsed.get("choices") or []
        if choices and "content" in choices[0].get("delta", {}):
            original = choices[0]["delta"]["content"]
            choices[0]["delta"]["content"] = unmasker.feed(original)
            yield f"data: {json.dumps(parsed)}\n\n".encode()
        else:
            yield chunk


async def _handle_streaming(
    client, db, provider, provider_name, sanitized_body, requested_model,
    key_record, redaction_summary, fallback, request_id,
    combined_mapping, wants_unmask,
):
    async def event_generator() -> AsyncIterator[bytes]:
        usage = StreamUsage()
        start = time.monotonic()
        active_provider_name = provider_name
        active_provider = provider

        async def _run(prov, body, usage_sink):
            source = prov.send_stream(client, body, usage_sink)
            if wants_unmask and combined_mapping:
                source = _unmask_sse_stream(source, combined_mapping)
            async for chunk in source:
                yield chunk

        try:
            async for chunk in _run(active_provider, sanitized_body, usage):
                yield chunk
        except ProviderError as primary_exc:
            if not fallback:
                # Best effort: surface the error as one final SSE error frame
                # rather than just dying — most SSE clients tolerate an
                # extra frame better than a silently truncated stream.
                yield f"data: {json.dumps({'error': {'message': primary_exc.detail}})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            fb_provider_name, fb_model = fallback
            logger.warning(
                "request_id=%s streaming: primary provider '%s' failed (%s) — falling back to %s/%s",
                request_id, provider_name, primary_exc.detail, fb_provider_name, fb_model,
            )
            fb_body = dict(sanitized_body)
            fb_body["model"] = fb_model
            active_provider = build_provider(fb_provider_name, settings)
            active_provider_name = fb_provider_name
            usage = StreamUsage()  # restart accounting against the fallback's own output
            try:
                async for chunk in _run(active_provider, fb_body, usage):
                    yield chunk
            except ProviderError as fallback_exc:
                yield f"data: {json.dumps({'error': {'message': fallback_exc.detail}})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

        latency_ms = int((time.monotonic() - start) * 1000)
        prompt_tokens = usage.prompt_tokens or count_tokens(_estimate_prompt_text(sanitized_body), requested_model)
        completion_tokens = usage.completion_tokens or count_tokens(usage.full_text, requested_model)
        cost = estimate_cost_usd(requested_model, prompt_tokens, completion_tokens, settings.db_path)

        await db.record_usage(
            api_key=key_record.key, provider=active_provider_name, model=requested_model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost,
            redaction_count=redaction_summary.total_redactions, cache_hit=False, latency_ms=latency_ms,
        )
        logger.info(
            "request_id=%s stream complete provider=%s cost=$%.6f prompt_tokens=%d completion_tokens=%d",
            request_id, active_provider_name, cost, prompt_tokens, completion_tokens,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-LLMShield-Redactions": str(redaction_summary.total_redactions),
            "X-LLMShield-Categories": ",".join(redaction_summary.counts.keys()),
            "X-LLMShield-Provider": provider_name,
            "Cache-Control": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
