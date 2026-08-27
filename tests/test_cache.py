from app.cache import compute_cache_key


def _body(**overrides):
    base = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 100,
        "user": "should-be-ignored",
    }
    base.update(overrides)
    return base


def test_same_inputs_produce_same_key():
    assert compute_cache_key("openai", _body()) == compute_cache_key("openai", _body())


def test_different_provider_changes_key():
    assert compute_cache_key("openai", _body()) != compute_cache_key("anthropic", _body())


def test_different_model_changes_key():
    assert compute_cache_key("openai", _body()) != compute_cache_key("openai", _body(model="gpt-4o"))


def test_different_messages_changes_key():
    other = _body(messages=[{"role": "user", "content": "bye"}])
    assert compute_cache_key("openai", _body()) != compute_cache_key("openai", other)


def test_irrelevant_field_does_not_change_key():
    a = compute_cache_key("openai", _body(user="alice"))
    b = compute_cache_key("openai", _body(user="bob"))
    assert a == b


def test_key_is_a_hex_sha256_digest():
    key = compute_cache_key("openai", _body())
    assert len(key) == 64
    int(key, 16)  # raises if not valid hex
