from kielikaveri.bot.text import split_message


def test_empty_text_returns_no_chunks():
    assert split_message("") == []


def test_short_text_returns_single_chunk():
    assert split_message("hello", limit=4096) == ["hello"]


def test_text_at_exact_limit_is_not_split():
    text = "a" * 10
    assert split_message(text, limit=10) == [text]


def test_splits_on_last_newline_before_limit():
    text = "a" * 5 + "\n" + "b" * 5
    chunks = split_message(text, limit=8)
    assert chunks == ["aaaaa", "bbbbb"]


def test_hard_splits_when_no_newline_available():
    text = "a" * 25
    chunks = split_message(text, limit=10)
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]


def test_reassembling_chunks_preserves_content_minus_split_newlines():
    text = "line one\n" * 5
    chunks = split_message(text, limit=20)
    assert len(chunks) > 1
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")
