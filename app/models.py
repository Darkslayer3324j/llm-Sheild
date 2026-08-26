"""
models.py — pydantic request/response models for llm-shield's OpenAI-compatible API.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict] | None = None


class ChatCompletionRequest(BaseModel):
    """Mirrors OpenAI's /v1/chat/completions request body.

    `extra="allow"` lets fields llm-shield doesn't know about (tools,
    functions, an explicit `provider` override, ...) pass through to the
    upstream provider untouched instead of being silently dropped.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class CreateKeyRequest(BaseModel):
    name: str
    daily_budget_usd: float = Field(default=5.0, gt=0)
    rate_limit_rpm: int = Field(default=60, gt=0)
    is_admin: bool = False


class RedactionSummary(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)
    total_redactions: int = 0
