"""Unit tests for text preprocessing utilities."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.preprocess import clean_text, chunk_text, truncate


def test_clean_text_basic():
    assert clean_text("  hello world  ") == "hello world"


def test_clean_text_html():
    assert "<b>" not in clean_text("<b>bold</b> text")


def test_clean_text_empty():
    assert clean_text("") == ""


def test_chunk_text_short():
    text = "short text"
    chunks = chunk_text(text, max_tokens=300)
    assert chunks == [text]


def test_chunk_text_long():
    words = ["word"] * 400
    text = " ".join(words)
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_truncate():
    text = "a" * 600
    result = truncate(text, max_chars=500)
    assert result.endswith("...")
    assert len(result) <= 503


def test_truncate_short():
    text = "hello"
    assert truncate(text, max_chars=500) == "hello"


def test_chunk_text_rejects_invalid_max_tokens():
    try:
        chunk_text("text", max_tokens=0, overlap=0)
    except ValueError as exc:
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("text", max_tokens=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_clean_text_crlf():
    """CRLF line endings must not leave a trailing space before the newline."""
    result = clean_text("line1\r\nline2")
    assert result == "line1\nline2", repr(result)
    assert " \n" not in result


def test_clean_text_bare_cr():
    """Bare CR (old Mac line endings) must be converted to LF."""
    result = clean_text("line1\rline2")
    assert result == "line1\nline2", repr(result)
    assert "\r" not in result


def test_clean_text_tabs_collapsed():
    """Tabs should be collapsed to a single space."""
    assert clean_text("hello\t\tworld") == "hello world"
