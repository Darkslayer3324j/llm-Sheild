from app.streaming import StreamUnmasker


def test_feed_emits_text_with_no_brackets_immediately():
    unmasker = StreamUnmasker({})
    assert unmasker.feed("hello world") == "hello world"


def test_feed_replaces_placeholder_within_a_single_chunk():
    unmasker = StreamUnmasker({"[EMAIL_1]": "joe@example.com"})
    out = unmasker.feed("email me at [EMAIL_1] thanks")
    assert out == "email me at joe@example.com thanks"


def test_placeholder_split_across_two_chunks_is_reassembled():
    unmasker = StreamUnmasker({"[EMAIL_1]": "joe@example.com"})
    first = unmasker.feed("contact [EMAIL")
    second = unmasker.feed("_1] please")
    assert first == "contact "
    assert second == "joe@example.com please"


def test_placeholder_split_across_many_small_chunks():
    unmasker = StreamUnmasker({"[EMAIL_1]": "joe@example.com"})
    pieces = ["hi ", "[", "EMA", "IL_", "1", "]", " bye"]
    out = "".join(unmasker.feed(p) for p in pieces)
    out += unmasker.flush()
    assert out == "hi joe@example.com bye"


def test_flush_releases_trailing_literal_bracket():
    unmasker = StreamUnmasker({"[EMAIL_1]": "joe@example.com"})
    out = unmasker.feed("array index[")
    assert out == "array index"
    assert unmasker.flush() == "["


def test_unmatched_placeholder_number_is_left_as_is():
    unmasker = StreamUnmasker({"[EMAIL_1]": "joe@example.com"})
    out = unmasker.feed("[EMAIL_2] unknown") + unmasker.flush()
    assert out == "[EMAIL_2] unknown"


def test_empty_piece_is_a_no_op():
    unmasker = StreamUnmasker({"[EMAIL_1]": "joe@example.com"})
    assert unmasker.feed("") == ""
    assert unmasker.feed("[EMAIL_1]") == "joe@example.com"
