# llm-shield

[![tests](https://github.com/darkslayer3324j/llm-shield/actions/workflows/tests.yml/badge.svg)](https://github.com/darkslayer3324j/llm-shield/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A local, zero-trust LLM proxy. Point your OpenAI SDK, LangChain, or any HTTP
client at `http://localhost:8000/v1` instead of calling providers directly,
and every request gets PII-scrubbed, cost-tracked, rate-limited, and cached
— streaming included — regardless of whether it's actually headed to
OpenAI, Anthropic, Gemini, or a local Ollama model.

**Free, open source, and self-hosted forever.** See [Licensing & roadmap](#licensing--roadmap) for what that means going forward.

## Why this exists

Most "LLM gateway" tools on the market do *one* of: PII redaction, cost
tracking, or multi-provider routing. llm-shield does all three locally,
with no data leaving your machine except the (sanitized) request itself.

## Features

- **PII redaction** — emails, API keys, credit cards (Luhn-validated), SSNs,
  phone numbers, IPs, and optional full-name heuristics, with reversible
  placeholders so you can unmask a response on demand.
- **Works with any model** — native adapters for OpenAI and Anthropic
  (translating Claude's Messages API to/from the OpenAI shape, streaming
  included), plus OpenAI-compatible passthrough for Gemini, Ollama,
  OpenRouter, Groq, Together, DeepSeek, vLLM, LM Studio, or anything else
  that speaks the `/chat/completions` schema. Routing is by model-name
  prefix, an explicit `provider` field, or an `X-LLMShield-Provider` header.
- **Streaming** — `stream: true` is fully supported, including on Anthropic
  (its native event stream is translated live into OpenAI-style SSE
  chunks). Usage/cost is still tracked accurately after the stream ends,
  and `X-LLMShield-Unmask: true` works on streams too — a placeholder split
  across two SSE chunks (e.g. `"...[EMAIL"` then `"_1]..."`) is still
  correctly reassembled and unmasked, not silently missed.
- **Provider fallback** — send `X-LLMShield-Fallback: anthropic/claude-3-5-haiku`
  and if the primary provider errors out, llm-shield retries once against
  the fallback automatically (works for both buffered and streaming calls).
- **Spend circuit breaker** — real token counting (exact via `tiktoken` for
  OpenAI models, estimated for others, gracefully degrading if `tiktoken`
  can't reach the network) against a per-key daily USD budget, tracked in
  SQLite. Over budget → `402`, before the call ever goes out.
- **Virtual API keys with admin roles** — issue separate keys (own budget +
  rate limit) to different apps/teammates/CI jobs without sharing your real
  provider keys. A master **admin** key is auto-provisioned on first boot.
  Only admin keys can manage other keys or view the dashboard — a regular
  key holder can't see anyone else's spend.
- **Rate limiting** — per-key requests-per-minute cap → `429`.
- **Response caching** — exact-match cache on (provider, model, messages,
  temperature). Repeated/deterministic calls skip the upstream entirely.
- **Request size guard** — reject abnormally large payloads before they can
  cause surprise bills (`413`, configurable).
- **Request-ID tracing** — every response carries an `X-Request-ID`,
  logged end-to-end, for correlating client-side issues with server logs.
- **Live dashboard** — `/dashboard`: spend today, requests, redactions,
  cache hit rate, 7-day spend chart, request log, and full key management
  (create/revoke) — gated behind an admin key, entered in-memory only
  (never written to browser storage).
- **`/v1/models`** — synthesized model list for client SDKs that ping it on
  init.
- **CLI** — create/list/revoke virtual keys, print spend stats.
- **Docker-ready** — `Dockerfile` + `docker-compose.yml` included.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in the provider keys you actually use
python -m app.main      # or: uvicorn app.main:app --reload
```

On first boot, if auth is enabled (default) and no keys exist yet,
llm-shield prints an **admin** master key to the console — save it, it's
shown once:

```
No API keys found — created an admin master key for you:
  llmshield-master-...
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer llmshield-master-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Email me at joe@example.com"}]
  }'
```

Swap `"model"` for `"claude-sonnet-4-..."` or `"gemini-1.5-flash"` and the
exact same request routes to a different provider automatically. Set
`"stream": true` for a live SSE response — works the same way for every
provider.

Open `http://localhost:8000/dashboard`, paste your admin key when prompted,
and watch spend, redactions, and keys live.

### Docker

```bash
cp .env.example .env
docker compose up --build
```

## Managing virtual keys

**CLI:**
```bash
python -m app.cli keys create --name "my-app" --daily-budget 5.0 --rate-limit 60
python -m app.cli keys create --name "ops-admin" --admin
python -m app.cli keys list
python -m app.cli keys revoke <key>
python -m app.cli stats
```

**REST (admin key required):**
```bash
curl -X POST http://localhost:8000/api/keys -H "Authorization: Bearer <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "daily_budget_usd": 5.0, "rate_limit_rpm": 60}'

curl http://localhost:8000/api/keys -H "Authorization: Bearer <admin-key>"
curl -X DELETE http://localhost:8000/api/keys/<key> -H "Authorization: Bearer <admin-key>"
```

**Or just use the dashboard** — the "Virtual keys" panel does all of this.

## Response headers

| Header | Meaning |
|---|---|
| `X-LLMShield-Redactions` | count of PII items redacted from the request |
| `X-LLMShield-Categories` | which categories were redacted (e.g. `EMAIL,API_KEY`) |
| `X-LLMShield-Cost-USD` | actual cost of this request (buffered only) |
| `X-LLMShield-Daily-Spend-USD` | running total for this key today (buffered only) |
| `X-LLMShield-Cache` | `HIT` or `MISS` (buffered only) |
| `X-LLMShield-Provider` | which upstream actually served the request |
| `X-LLMShield-Fallback-Used` | present + `true` if the fallback provider answered |
| `X-Request-ID` | correlation ID for this request, echo it in bug reports |

Send `X-LLMShield-Unmask: true` on any request — streaming or not — to get
redacted values swapped back into the response content (never persisted
anywhere as plaintext).

## Tests

```bash
pytest tests/ -v
```

60 tests covering the sanitizer, pricing/token counting, provider
request/response translation (including streaming SSE translation for both
OpenAI-compatible and Anthropic), the boundary-safe streaming unmasker, the
SQLite spend ledger + cache + key store (including the schema migration
path), and auth/admin gating.

## Configuration reference

See `.env.example` for the full list. Notable ones:

- `PROVIDER_ROUTES` — model-prefix → provider mapping (edit `app/config.py`
  if you need routes beyond the `gpt-`/`claude-`/`gemini-` defaults).
- `ENABLE_AUTH` — set `false` for a single-user local setup with no key
  management; requests are still spend-tracked and rate-limited under an
  implicit "anonymous" identity, so the circuit breaker is never bypassable.
- `ENABLE_CACHE` / `CACHE_TTL_SECONDS` — exact-match response cache.
- `MAX_REQUEST_CHARS` — reject oversized requests before they reach a
  provider.

### Editing model pricing

`app/pricing.py` has a hand-maintained `DEFAULT_PRICING` table (USD per 1M
tokens) — **verify these against each provider's current pricing page**;
they will drift. To override without touching code, drop a
`pricing_overrides.json` next to your SQLite DB file:

```json
{ "gpt-4o-mini": [0.15, 0.60] }
```

## Known limitations (by design, for now)

- Token counts for Anthropic/Gemini are a ~4-chars/token estimate (neither
  exposes a local tokenizer), fine for a circuit breaker, not invoice-grade.
- The response cache is exact-match only, not semantic — a rephrased
  question is a cache miss by design (a wrong-but-plausible cached answer
  is a worse failure mode than an extra API call). Streaming requests
  aren't cached at all.
- Multimodal message content (image blocks) passes through unsanitized.
- Rate limiting and the response cache are in-process/local SQLite —
  correct for one instance, not for a distributed multi-process deployment.
- Provider fallback retries once, on the first failure; it doesn't chain
  through more than two providers.

## Licensing & roadmap

llm-shield is **open core**:

- **Everything in this repo** — the proxy, all provider adapters, the
  sanitizer, the spend ledger, virtual keys, caching, streaming, the
  dashboard, the CLI — is Apache-2.0 licensed (see `LICENSE`). Self-host it,
  fork it, run it commercially, modify it, no strings attached.
- **Planned, not yet built:** a hosted version (point your app at a URL
  instead of running your own instance) and team/enterprise features on top
  (SSO, cross-instance audit log export, org-level key management). Those
  would be a paid addition *on top of* this codebase, not a fork of it or a
  crippled version of what's here — the self-hosted path stays fully
  featured and free.

If you're building on this, opening issues, or want to contribute, see
`CONTRIBUTING.md`.
