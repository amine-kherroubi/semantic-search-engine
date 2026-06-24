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
    - Normalise line endings (CRLF and bare CR -> LF)
    - Strip HTML tags
    - Collapse whitespace (tabs, multiple spaces) to a single space
    - Collapse 3+ consecutive newlines to two
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Normalise line endings before whitespace collapse; without this, CRLF
    # sequences leave a trailing space on every line because \r matches [^\S\n]
    # and gets replaced with a space while \n is preserved.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse non-newline whitespace (spaces, tabs, etc.) to a single space
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
