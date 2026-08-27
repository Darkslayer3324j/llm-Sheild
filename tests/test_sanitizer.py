import pytest

from app.sanitizer import PIICategory, SanitizerConfig, SanitizerEngine, _luhn_valid


def test_email_is_redacted_and_unmaskable():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("contact me at joe@example.com please")
    assert "joe@example.com" not in result.sanitized_text
    assert "[EMAIL_1]" in result.sanitized_text
    assert result.counts[PIICategory.EMAIL.value] == 1
    assert result.unmask(result.sanitized_text) == "contact me at joe@example.com please"


def test_multiple_emails_get_distinct_placeholders():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("a@x.com and b@y.com")
    assert result.mapping["[EMAIL_1]"] == "a@x.com"
    assert result.mapping["[EMAIL_2]"] == "b@y.com"
    assert result.counts[PIICategory.EMAIL.value] == 2


def test_openai_api_key_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    key = "sk-" + "a" * 40
    result = engine.sanitize(f"my key is {key}")
    assert key not in result.sanitized_text
    assert result.counts[PIICategory.API_KEY.value] == 1


def test_github_pat_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    token = "ghp_" + "a" * 36
    result = engine.sanitize(f"token: {token}")
    assert token not in result.sanitized_text
    assert result.counts[PIICategory.API_KEY.value] == 1


def test_ipv4_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("connect to 192.168.1.10 now")
    assert "192.168.1.10" not in result.sanitized_text
    assert result.counts[PIICategory.IPV4.value] == 1


def test_ipv6_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("host is 2001:0db8:85a3:0000:0000:8a2e:0370:7334")
    assert result.counts[PIICategory.IPV6.value] == 1


def test_valid_credit_card_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    # 4111111111111111 is a well-known Luhn-valid test Visa number.
    result = engine.sanitize("card 4111111111111111 on file")
    assert "4111111111111111" not in result.sanitized_text
    assert result.counts[PIICategory.CREDIT_CARD.value] == 1


def test_luhn_invalid_digit_run_is_left_alone():
    engine = SanitizerEngine(SanitizerConfig())
    # Same length as a card number but fails the Luhn checksum.
    bogus = "1234567890123456"
    assert not _luhn_valid(bogus)
    result = engine.sanitize(f"order number {bogus}")
    assert bogus in result.sanitized_text
    assert PIICategory.CREDIT_CARD.value not in result.counts


def test_ssn_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("ssn 123-45-6789 on file")
    assert "123-45-6789" not in result.sanitized_text
    assert result.counts[PIICategory.SSN.value] == 1


def test_phone_number_is_redacted():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("call me at 415-555-0134")
    assert "415-555-0134" not in result.sanitized_text
    assert result.counts[PIICategory.PHONE.value] == 1


def test_full_names_off_by_default():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("Jane Smith sent this")
    assert "Jane Smith" in result.sanitized_text
    assert PIICategory.FULL_NAME.value not in result.counts


def test_full_names_redacted_when_enabled():
    engine = SanitizerEngine(SanitizerConfig(mask_full_names=True))
    result = engine.sanitize("Jane Smith sent this")
    assert "Jane Smith" not in result.sanitized_text
    assert result.counts[PIICategory.FULL_NAME.value] == 1


def test_disabled_category_is_not_touched():
    config = SanitizerConfig(mask_emails=False)
    engine = SanitizerEngine(config)
    result = engine.sanitize("mail joe@example.com")
    assert "joe@example.com" in result.sanitized_text
    assert result.counts == {}


def test_non_string_input_returns_empty_sanitized_text():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize(None)
    assert result.sanitized_text == ""
    assert result.mapping == {}


def test_unmask_handles_placeholder_number_collisions():
    engine = SanitizerEngine(SanitizerConfig())
    text = " ".join(f"user{i}@example.com" for i in range(11))
    result = engine.sanitize(text)
    # [EMAIL_1] must not incorrectly match inside [EMAIL_10] during unmask.
    assert result.unmask(result.sanitized_text) == text


def test_clean_text_is_unchanged():
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("just a normal sentence with no secrets")
    assert result.sanitized_text == "just a normal sentence with no secrets"
    assert result.total_redactions == 0


# --- Regression: secret formats that used to pass through in the clear ------
# Every value below is syntactically shaped but fake. Each of these was NOT
# redacted before the secret-detection fix; `sk-[A-Za-z0-9]{20,}` required
# alphanumerics straight after `sk-`, so any hyphen- or underscore-delimited
# vendor prefix escaped, and several token formats had no pattern at all.

@pytest.mark.parametrize(
    "label,secret",
    [
        ("anthropic", "sk-ant-api03-" + "a" * 40),
        ("openai_service_account", "sk-svcacct-" + "a" * 24),
        ("stripe_live", "sk_live_" + "a" * 24),
        ("stripe_restricted", "rk_live_" + "a" * 24),
        ("github_fine_grained", "github_pat_" + "a" * 22 + "_" + "b" * 59),
        ("github_server_to_server", "ghs_" + "a" * 36),
        ("github_user_to_server", "ghu_" + "a" * 36),
        ("google_oauth", "ya29." + "a" * 30),
        ("aws_temporary", "ASIA" + "B" * 16),
    ],
)
def test_previously_leaked_secret_formats_are_redacted(label, secret):
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize(f"here is the key {secret} ok")
    assert secret not in result.sanitized_text, f"{label} leaked"
    assert result.counts[PIICategory.API_KEY.value] == 1


def test_anthropic_key_roundtrips_through_unmask():
    """The proxy ships an Anthropic adapter, so this is the key type its own
    users are most likely to paste. It must redact and restore exactly."""
    engine = SanitizerEngine(SanitizerConfig())
    key = "sk-ant-api03-" + "a" * 40
    original = f"use {key} for the call"
    result = engine.sanitize(original)
    assert key not in result.sanitized_text
    assert result.unmask(result.sanitized_text) == original


# --- SSN separators and structural validation -------------------------------

@pytest.mark.parametrize("raw", ["123-45-6789", "123 45 6789"])
def test_ssn_is_redacted_with_either_separator(raw):
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize(f"ssn {raw} on file")
    assert raw not in result.sanitized_text
    assert result.counts[PIICategory.SSN.value] == 1


@pytest.mark.parametrize(
    "raw",
    ["000-45-6789", "666-45-6789", "900-45-6789", "123-00-6789", "123-45-0000"],
)
def test_structurally_invalid_ssns_are_left_alone(raw):
    """Areas 000/666/900-999, group 00 and serial 0000 are never issued, so
    redacting them would be a false positive."""
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize(f"ref {raw} here")
    assert raw in result.sanitized_text
    assert PIICategory.SSN.value not in result.counts


def test_compact_ssn_is_ignored_by_default():
    """A bare nine-digit run is indistinguishable from an order number, so it
    must not be redacted unless explicitly opted in."""
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("order 123456789 shipped")
    assert "123456789" in result.sanitized_text


def test_compact_ssn_is_redacted_when_enabled():
    engine = SanitizerEngine(SanitizerConfig(mask_ssn_without_separators=True))
    result = engine.sanitize("ssn 123456789 on file")
    assert "123456789" not in result.sanitized_text
    assert result.counts[PIICategory.SSN.value] == 1


def test_compact_ssn_still_validates_when_enabled():
    engine = SanitizerEngine(SanitizerConfig(mask_ssn_without_separators=True))
    result = engine.sanitize("ref 000000000 here")
    assert "000000000" in result.sanitized_text


def test_phone_numbers_are_not_swallowed_by_the_ssn_pattern():
    """The SSN middle group is two digits; a phone's is three. Guards against
    the widened separator class over-matching."""
    engine = SanitizerEngine(SanitizerConfig())
    result = engine.sanitize("call 555-123-4567 now")
    assert result.counts.get(PIICategory.SSN.value) is None
    assert result.counts[PIICategory.PHONE.value] == 1
