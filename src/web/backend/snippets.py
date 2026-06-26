"""
Snippet extraction with naive query-term highlighting.

Not part of the original project (only full `content` was ever printed
to the console by the CLI). Added here because the brief asks for a
"snippet/highlight" per result, which needs an excerpt centred on a
matched query term rather than always the first N characters.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def make_snippet(content: str, query: str, window: int = 160) -> str:
    """
    Return a short excerpt of `content` centred on the first matched query
    term, with **double asterisks** around every matched term in the
    excerpt (the frontend renders these as <mark> highlights).

    Falls back to the first `window` characters if no query term is
    found verbatim in the content (e.g. a semantic match with no
    lexical overlap).
    """
    if not content:
        return ""

    terms = [t.lower() for t in _WORD_RE.findall(query) if len(t) > 1]
    lower_content = content.lower()

    match_pos = -1
    for term in terms:
        pos = lower_content.find(term)
        if pos != -1:
            match_pos = pos
            break

    if match_pos == -1:
        excerpt = content[: window * 2].rstrip()
        if len(content) > window * 2:
            excerpt += "..."
        return excerpt

    start = max(0, match_pos - window // 2)
    end = min(len(content), match_pos + window // 2)
    excerpt = content[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    excerpt = f"{prefix}{excerpt}{suffix}"

    if terms:
        pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in terms) + ")", re.IGNORECASE
        )
        excerpt = pattern.sub(r"**\1**", excerpt)

    return excerpt
