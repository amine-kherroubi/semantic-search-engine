"""Unit tests for text preprocessing utilities."""

import sys, os

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
