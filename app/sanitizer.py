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
    # Also treat a bare nine-digit run as an SSN. Off by default: it cannot be
    # told apart from an order or invoice number, so it will false-positive.
    mask_ssn_without_separators: bool = False
    mask_phone_numbers: bool = True
    mask_full_names: bool = False  # heuristic, opt-in only — see docstring below


# --- Regex library --------------------------------------------------------
# Ordered roughly most-specific -> least-specific. Compiled once at import
# time for performance (this runs on every proxied request).

_API_KEY_PATTERNS: list[re.Pattern[str]] = [
    # `sk-` keys, including vendor-prefixed forms. The prefix segments matter:
    # an earlier `sk-[A-Za-z0-9]{20,}` required alphanumerics immediately after
    # `sk-`, so anything with a hyphenated vendor tag (`sk-ant-api03-…`,
    # `sk-svcacct-…`) never matched and passed through in the clear.
    re.compile(r"sk-(?:[A-Za-z0-9]{1,12}-)*[A-Za-z0-9_-]{20,}"),
    # Stripe and other `_`-delimited secret keys — the `sk-` patterns above
    # cannot match these because the delimiter is an underscore.
    re.compile(r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),                # Google API key
    re.compile(r"ya29\.[A-Za-z0-9_-]{20,}"),             # Google OAuth access token
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key ID
    re.compile(r"ASIA[0-9A-Z]{16}"),                      # AWS temporary access key ID
    # GitHub tokens: classic PAT (ghp_), OAuth (gho_), user-to-server (ghu_),
    # server-to-server (ghs_), refresh (ghr_).
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),          # GitHub fine-grained PAT
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

# SSNs are commonly written with a dash, a space, or nothing at all. Only the
# dashed form used to be matched, so "123 45 6789" and "123456789" passed
# through untouched.
_SSN_PATTERN = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")

# Separator-less SSNs are opt-in: a bare nine-digit run is indistinguishable
# from an order number, invoice ID or similar, so matching it trades false
# positives for recall. Structural validation below removes the ranges the SSA
# never issues, which cuts the obvious false positives but not all of them.
_SSN_COMPACT_PATTERN = re.compile(r"\b\d{9}\b")

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


def _ssn_valid(digits: str) -> bool:
    """Reject nine-digit runs the SSA never issues as an SSN.

    Area (first three) is never 000, 666, or 900-999; group (middle two) is
    never 00; serial (last four) is never 0000. A cheap structural check that
    removes obviously-wrong matches before anything is redacted.
    """
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in ("000", "666") or area[0] == "9":
        return False
    return group != "00" and serial != "0000"


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
            working = self._apply_ssn(working, mapping, counts)

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

    def _apply_ssn(
        self, text: str, mapping: dict[str, str], counts: dict[str, int]
    ) -> str:
        """Separate path because matches are structurally validated before
        redaction, and because the separator-less form is opt-in."""
        counter = counts.get(PIICategory.SSN.value, 0)

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            raw = match.group(0)
            if not _ssn_valid(re.sub(r"[- ]", "", raw)):
                return raw  # not a valid SSN, leave as-is
            counter += 1
            placeholder = f"[{PIICategory.SSN.value}_{counter}]"
            mapping[placeholder] = raw
            return placeholder

        text = _SSN_PATTERN.sub(_replace, text)
        if self.config.mask_ssn_without_separators:
            text = _SSN_COMPACT_PATTERN.sub(_replace, text)

        if counter:
            counts[PIICategory.SSN.value] = counter
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
