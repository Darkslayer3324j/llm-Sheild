"""
Centralized configuration for llm-shield, loaded from environment variables / .env.

Uses pydantic-settings so every value is validated and type-safe at startup —
a malformed spend limit or missing upstream key fails fast instead of blowing
up mid-request.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Provider credentials -------------------------------------------------
    # Each provider is independently optional — configure only the ones you use.
    # A model request is routed to a provider by prefix match (see
    # provider_routes below) or an explicit "provider" override on the request.

    openai_api_key: str | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    anthropic_api_key: str | None = Field(default=None)
    anthropic_base_url: str = Field(default="https://api.anthropic.com/v1")
    anthropic_version: str = Field(default="2023-06-01")

    gemini_api_key: str | None = Field(default=None)
    # Google's OpenAI-compatibility layer — lets us treat Gemini as just
    # another OpenAI-shaped endpoint instead of writing a custom adapter.
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai"
    )

    # "generic" = anything that speaks the OpenAI chat-completions schema:
    # Ollama (local), OpenRouter, Groq, Together, Mistral, DeepSeek, vLLM, etc.
    # This is what makes llm-shield work with "any model" pragmatically —
    # most providers now ship an OpenAI-compatible endpoint.
    generic_api_key: str | None = Field(default=None)
    generic_base_url: str | None = Field(default=None)
    generic_provider_label: str = Field(default="generic")

    default_provider: str = Field(
        default="openai", description="Provider used when a model doesn't match any prefix rule."
    )

    # model name prefix -> provider name. First match wins, checked in
    # insertion order.
    provider_routes: dict[str, str] = Field(
        default_factory=lambda: {
            "gpt-": "openai",
            "o1-": "openai",
            "o3-": "openai",
            "o4-": "openai",
            "chatgpt-": "openai",
            "claude-": "anthropic",
            "gemini-": "gemini",
        }
    )

    # --- Server -----------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- Sanitizer toggles --------------------------------------------------
    mask_emails: bool = True
    mask_api_keys: bool = True
    mask_credit_cards: bool = True
    mask_ssn: bool = True
    mask_phone_numbers: bool = True
    mask_ip_addresses: bool = True
    mask_full_names: bool = False  # opt-in: heuristic, prone to false positives

    # --- Response unmasking --------------------------------------------------
    allow_response_unmasking: bool = Field(default=True)

    # --- Auth / virtual keys --------------------------------------------------
    enable_auth: bool = Field(
        default=True,
        description="If true, callers must send Authorization: Bearer <virtual-key>. "
        "A master key is auto-provisioned on first boot if none exist.",
    )
    master_key_daily_budget_usd: float = Field(default=25.0)
    master_key_rate_limit_rpm: int = Field(default=120)

    # --- Cost circuit breaker --------------------------------------------------
    default_daily_budget_usd: float = Field(
        default=5.0, description="Default per-key daily budget for newly created keys."
    )
    default_rate_limit_rpm: int = Field(
        default=60, description="Default requests-per-minute cap for newly created keys."
    )

    # --- Caching --------------------------------------------------------------
    enable_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)

    # --- Guardrails --------------------------------------------------------
    max_request_chars: int = Field(
        default=400_000,
        description="Reject requests whose combined message text exceeds this many "
        "characters (413) — protects against accidental huge-payload bill shock.",
    )

    # --- Storage ----------------------------------------------------------
    db_path: str = Field(default="llm_shield.db")

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing env on every request."""
    return Settings()
