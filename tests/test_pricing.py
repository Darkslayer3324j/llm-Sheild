import json

import app.pricing as pricing


def test_get_pricing_exact_match():
    assert pricing.get_pricing("gpt-4o-mini") == pricing.DEFAULT_PRICING["gpt-4o-mini"]


def test_get_pricing_prefix_match_picks_longest():
    # "gpt-4o-2024-08-06" should resolve to "gpt-4o", not some shorter prefix.
    assert pricing.get_pricing("gpt-4o-2024-08-06") == pricing.DEFAULT_PRICING["gpt-4o"]


def test_get_pricing_unknown_model_uses_fallback():
    assert pricing.get_pricing("some-made-up-model-xyz") == pricing.FALLBACK_PRICING


def test_get_pricing_overrides_take_precedence(tmp_path):
    db_path = tmp_path / "llm_shield.db"
    overrides = tmp_path / "pricing_overrides.json"
    overrides.write_text(json.dumps({"gpt-4o-mini": [0.01, 0.02]}))

    assert pricing.get_pricing("gpt-4o-mini", str(db_path)) == (0.01, 0.02)


def test_get_pricing_ignores_malformed_overrides_file(tmp_path):
    db_path = tmp_path / "llm_shield.db"
    overrides = tmp_path / "pricing_overrides.json"
    overrides.write_text("{not valid json")

    # Falls back to the default table instead of raising.
    assert pricing.get_pricing("gpt-4o-mini", str(db_path)) == pricing.DEFAULT_PRICING["gpt-4o-mini"]


def test_count_tokens_openai_model_returns_positive_count():
    # Exercises the tiktoken path (or its char-based fallback, if tiktoken's
    # encoding file can't be fetched in this environment) without asserting
    # an exact count either way.
    text = "The quick brown fox jumps over the lazy dog."
    n = pricing.count_tokens(text, "gpt-4o-mini")
    assert isinstance(n, int) and n > 0


def test_count_tokens_non_openai_model_uses_char_estimate():
    text = "a" * 40
    n = pricing.count_tokens(text, "claude-3-5-sonnet")
    assert n == 10  # 40 chars / 4 chars-per-token


def test_count_tokens_empty_text_is_zero():
    assert pricing.count_tokens("", "gpt-4o-mini") == 0
    assert pricing.count_tokens("", "claude-3-5-sonnet") == 0


def test_count_tokens_minimum_one_for_nonempty_text():
    assert pricing.count_tokens("hi", "gemini-1.5-flash") >= 1


def test_estimate_cost_usd_matches_manual_calculation():
    input_price, output_price = pricing.DEFAULT_PRICING["gpt-4o-mini"]
    cost = pricing.estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == input_price + output_price


def test_estimate_cost_usd_zero_tokens_is_zero_cost():
    assert pricing.estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0
