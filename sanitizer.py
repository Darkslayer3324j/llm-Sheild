"""
sanitizer.py — PII & sensitive-data redaction engine for llm-shield.

Design goals:
- Deterministic, stateless per call: SanitizerEngine.sanitize(text) never
  mutates shared state, so it's safe under FastAPI's concurrent async
  request handling (no shared counters across requests).
- Reversible: every redaction produces a mapping placeholder -> original
  value, so the caller can reconstruct the original text (e.g. to unmask
  an upstream response) without llm-shield ever persisting the raw PII.
- Ordered, most-specific-first matching to minimize false positives/overlap
  (e.g. an API key shouldn't get partially eaten by the generic phone regex).
- Validated where possible (Luhn-checked credit cards) rather than "any
  13-19 digit run", which would false-positive on order numbers, tracking
  numbers, etc.

This module has zero framework dependencies (no FastAPI/pydantic) so it can
be unit tested and reused (e.g. in the CLI, in gd-blackbox's log parser, etc.)
in isolation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class PIICategory(str, Enum):
    API_KEY = "API_KEY"
    EMAIL = "EMAIL"
    IPV6 = "IPV6"
    IPV4 = "IPV4"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    PHONE = "PHONE"
    FULL_NAME = "FULL_NAME"


@dataclass
class SanitizationResult:
    """Output of a sanitize() call."""

    sanitized_text: str
    # placeholder (e.g. "[EMAIL_1]") -> original raw value
    mapping: dict[str, str] = field(default_factory=dict)
    # category -> number of redactions of that category
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        return sum(self.counts.values())

    def unmask(self, text: str) -> str:
        """Reverse-map placeholders in `text` back to their original values.

        Longest-placeholder-first replacement avoids partial collisions
        (e.g. [EMAIL_1] vs [EMAIL_10]).
        """
        result = text
        for placeholder in sorted(self.mapping, key=len, reverse=True):
            result = result.replace(placeholder, self.mapping[placeholder])
        return result


@dataclass
class SanitizerConfig:
    mask_api_keys: bool = True
    mask_emails: bool = True
    mask_ipv4: bool = True
    mask_ipv6: bool = True
    mask_credit_cards: bool = True
    mask_ssn: bool = True
    mask_phone_numbers: bool = True
    mask_full_names: bool = False  # heuristic, opt-in only — see docstring below


# --- Regex library --------------------------------------------------------
# Ordered roughly most-specific -> least-specific. Compiled once at import
# time for performance (this runs on every proxied request).

_API_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),          # OpenAI project key
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                  # OpenAI legacy key
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),                # Google API key
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key ID
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                   # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{36}"),                   # GitHub OAuth token
    re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"),          # Slack token
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}=*", re.IGNORECASE),  # generic bearer token
]

_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

_IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# Simplified but practical IPv6 matcher (full + compressed forms).
_IPV6_PATTERN = re.compile(
    r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"          # full form
    r"|\b(?:[A-Fa-f0-9]{1,4}:)+:(?:[A-Fa-f0-9]{1,4}:)*[A-Fa-f0-9]{1,4}\b"  # compressed
)

# Candidate digit runs (with common separators) that MIGHT be a credit card —
# validated with Luhn before being treated as one.
_CREDIT_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_PHONE_PATTERN = re.compile(
    r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)

# Best-effort "Firstname Lastname" / "Title Firstname Lastname" heuristic.
# Deliberately conservative (requires two consecutive capitalized words) but
# WILL false-positive on proper nouns like game/product/place names — hence
# off by default. Documented, not silently applied.
_FULL_NAME_PATTERN = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Mx\.|Prof\.)?\s?"
    r"[A-Z][a-z]{1,20}\s[A-Z][a-z]{1,20}\b"
)


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum — used to confirm a digit run is plausibly a
    real card number before we redact it as CREDIT_CARD."""
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class SanitizerEngine:
    """Stateless, reusable PII redaction engine.

    Usage:
        engine = SanitizerEngine(SanitizerConfig())
        result = engine.sanitize("contact me at joe@x.com")
        result.sanitized_text  # "contact me at [EMAIL_1]"
        result.unmask(result.sanitized_text)  # "contact me at joe@x.com"
    """

    def __init__(self, config: SanitizerConfig | None = None) -> None:
        self.config = config or SanitizerConfig()

    def sanitize(self, text: str | None) -> SanitizationResult:
        if not text or not isinstance(text, str):
            return SanitizationResult(sanitized_text=text or "")

        mapping: dict[str, str] = {}
        counts: dict[str, int] = {}
        working = text

        if self.config.mask_api_keys:
            working = self._apply(working, PIICategory.API_KEY, _API_KEY_PATTERNS, mapping, counts)

        if self.config.mask_emails:
            working = self._apply(working, PIICategory.EMAIL, [_EMAIL_PATTERN], mapping, counts)

        if self.config.mask_ipv6:
            working = self._apply(working, PIICategory.IPV6, [_IPV6_PATTERN], mapping, counts)

        if self.config.mask_ipv4:
            working = self._apply(working, PIICategory.IPV4, [_IPV4_PATTERN], mapping, counts)

        if self.config.mask_credit_cards:
            working = self._apply_credit_cards(working, mapping, counts)

        if self.config.mask_ssn:
            working = self._apply(working, PIICategory.SSN, [_SSN_PATTERN], mapping, counts)

        if self.config.mask_phone_numbers:
            working = self._apply(working, PIICategory.PHONE, [_PHONE_PATTERN], mapping, counts)

        if self.config.mask_full_names:
            working = self._apply(working, PIICategory.FULL_NAME, [_FULL_NAME_PATTERN], mapping, counts)

        return SanitizationResult(sanitized_text=working, mapping=mapping, counts=counts)

    # -- internals ----------------------------------------------------

    def _apply(
        self,
        text: str,
        category: PIICategory,
        patterns: list[re.Pattern[str]],
        mapping: dict[str, str],
        counts: dict[str, int],
    ) -> str:
        counter = counts.get(category.value, 0)

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            placeholder = f"[{category.value}_{counter}]"
            mapping[placeholder] = match.group(0)
            return placeholder

        for pattern in patterns:
            text = pattern.sub(_replace, text)

        if counter:
            counts[category.value] = counter
        return text

    def _apply_credit_cards(
        self, text: str, mapping: dict[str, str], counts: dict[str, int]
    ) -> str:
        """Separate path because it needs Luhn validation before redacting —
        a non-validated match is left untouched (likely not a real card)."""
        counter = counts.get(PIICategory.CREDIT_CARD.value, 0)

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            raw = match.group(0)
            digits = re.sub(r"[ -]", "", raw)
            if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
                return raw  # not a valid card number, leave as-is
            counter += 1
            placeholder = f"[{PIICategory.CREDIT_CARD.value}_{counter}]"
            mapping[placeholder] = raw
            return placeholder

        text = _CREDIT_CARD_CANDIDATE.sub(_replace, text)
        if counter:
            counts[PIICategory.CREDIT_CARD.value] = counter
        return text
