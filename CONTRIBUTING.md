# Contributing to llm-shield

Thanks for considering a contribution — issues, PRs, and design discussion
are all welcome.

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/llm-shield.git
cd llm-shield
pip install -r requirements.txt
pip install pytest
cp .env.example .env   # fill in whichever provider keys you want to test against
pytest tests/ -v
```

## Before opening a PR

- Add or update tests for any behavior change — `pytest tests/ -v` should
  pass. New provider adapters, sanitizer categories, or config options all
  need coverage; see `tests/` for the existing patterns to follow.
- Run `python -m py_compile app/*.py` as a quick sanity check.
- Keep PRs scoped to one change where possible — easier to review, easier
  to revert if something's wrong.
- If you're touching the sanitizer regexes, include both a "should redact"
  and a "should NOT redact" test case — false positives are as much a bug
  as missed detections.

## Where things live

| Area | File |
|---|---|
| PII detection/redaction | `app/sanitizer.py` |
| Provider adapters (OpenAI/Anthropic/Gemini/generic) | `app/providers.py` |
| Streaming SSE translation + boundary-safe unmasking | `app/streaming.py` |
| Spend ledger, virtual keys, cache (SQLite) | `app/db.py` |
| Request lifecycle / routing | `app/main.py` |
| Token counting + pricing table | `app/pricing.py` |
| CLI | `app/cli.py` |
| Dashboard | `static/dashboard.html` |

## Reporting a security issue

If you find something that could let PII leak past the sanitizer, bypass
the spend circuit breaker, or expose the dashboard/admin API without auth,
please open an issue marked `security` (or, if you'd rather not disclose
publicly first, reach out directly) rather than a public PR with exploit
details — happy to work through a fix together before it's public.

## Ideas that would be genuinely useful

Not a promise these will be merged as-is, but these are gaps I know about:

- Semantic (embedding-based) caching as an *opt-in* mode alongside the
  existing exact-match cache
- Per-key model allowlists
- A `/metrics` endpoint for Prometheus scraping
- Additional native provider adapters (Azure OpenAI, Bedrock, Vertex AI)
