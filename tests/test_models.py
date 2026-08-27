from app.models import ChatCompletionRequest, CreateKeyRequest, RedactionSummary


def test_chat_completion_request_parses_minimal_body():
    req = ChatCompletionRequest(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert req.model == "gpt-4o-mini"
    assert req.stream is False
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hi"


def test_chat_completion_request_preserves_unknown_fields_on_dump():
    req = ChatCompletionRequest(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        provider="anthropic",
    )
    dumped = req.model_dump(exclude_none=True)
    assert dumped["provider"] == "anthropic"


def test_chat_completion_request_defaults_are_none():
    req = ChatCompletionRequest(model="gpt-4o-mini", messages=[])
    assert req.temperature is None
    assert req.top_p is None
    assert req.max_tokens is None


def test_create_key_request_defaults():
    req = CreateKeyRequest(name="my-app")
    assert req.daily_budget_usd == 5.0
    assert req.rate_limit_rpm == 60
    assert req.is_admin is False


def test_redaction_summary_defaults_to_empty():
    summary = RedactionSummary()
    assert summary.counts == {}
    assert summary.total_redactions == 0


def test_redaction_summary_with_counts():
    summary = RedactionSummary(counts={"EMAIL": 2, "SSN": 1}, total_redactions=3)
    assert summary.total_redactions == 3
