"""
Text preprocessing helpers.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalise and clean raw text:
    - Normalize unicode to NFC
    - Strip HTML tags
    - Collapse whitespace
    - Remove control characters
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Strip HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove control chars (except newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> list[str]:
    """
    Split long text into overlapping word-level chunks.
    This avoids truncation in the embedding model.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: list[str] = []
    step = max_tokens - overlap
    for start in range(0, len(words), step):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

    return chunks


def truncate(text: str, max_chars: int = 500) -> str:
    """Return first max_chars characters followed by '...' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
