# Security

llm-shield sits between your application and an LLM provider, so a failure here
means sensitive data reaching a third party. Reports are welcome and taken
seriously.

## Reporting a vulnerability

Please report privately rather than opening a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/Darkslayer3324j/llm-Sheild/security/advisories/new),
or email nafayhassan3324j@gmail.com.

Useful things to include: the input that reproduces it, what was redacted
versus what should have been, and the version or commit. If you are reporting a
redaction miss, a syntactically-shaped **fake** secret is enough — please do not
send real credentials.

Expect an acknowledgement within a few days.

## What redaction can and cannot do

The sanitizer is pattern-based. That is a deliberate trade — it is fast enough
to run on every proxied request, deterministic, and requires no model — but it
sets a hard ceiling on what it can catch:

- **It matches known shapes.** A credential in a format the engine has no
  pattern for passes through in the clear. New providers invent new formats,
  so this list is never finished. Missing formats are exactly the kind of bug
  worth reporting.
- **It cannot read intent.** Free-form sensitive text — a medical detail, a
  home address in prose, an internal codename — has no regex. Names are
  heuristic and off by default because the heuristic false-positives on
  ordinary proper nouns.
- **Validation reduces false positives, not misses.** Luhn checks on cards and
  structural checks on SSNs stop valid-looking-but-wrong runs from being
  redacted. They do not help find things the patterns never matched.

Treat llm-shield as defence in depth. It meaningfully reduces accidental
leakage; it is not a guarantee that nothing sensitive reaches the provider, and
it should not be the only control protecting regulated data.

## Threat model

**In scope**

- Redaction misses: a credential or PII format that should be caught and is not
- Unmasking flaws: placeholder collisions, or mappings that leak original values
- Auth bypass on the proxy's own virtual API keys, or privilege escalation
  between keys
- Budget or rate-limit bypass
- Secrets written to logs, the cache, or the database in the clear
- SSRF or request smuggling through the provider-routing layer

**Out of scope**

- The upstream provider's handling of data you sent it after sanitisation
- Anything requiring an attacker who already controls the host, since the
  proxy is designed to run locally and trusts its own machine
- False positives that over-redact. Annoying, worth an issue, not a
  vulnerability.

## Operational notes

- llm-shield is meant to run on `localhost`. Exposing it to a network makes its
  virtual API keys an internet-facing authentication surface, which is not what
  they were designed for. Put a real gateway in front if you must.
- The database holds usage records and cached responses. Cached entries are
  stored **after** sanitisation, but treat the file as sensitive.
- Provider keys are read from the environment. Do not commit them, and do not
  bake them into an image.
