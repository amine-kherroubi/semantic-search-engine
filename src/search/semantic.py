"""
Semantic search using pgvector cosine similarity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from sqlalchemy import text

from src.db.connection import get_session
from src.embeddings.encoder import encode_query

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "10"))
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@dataclass
class SearchResult:
    doc_id: int
    title: str | None
    content: str
    source: str | None
    score: float  # cosine similarity  (higher = more similar)
    rank: int


def semantic_search(
    query: str,
    top_k: int = TOP_K,
    model_name: str | None = None,
) -> List[SearchResult]:
    """
    Encode *query* and retrieve the top-k most similar documents from pgvector.

    Returns a list of SearchResult sorted by descending similarity.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not query.strip():
        raise ValueError("query must be a non-empty string")

    model = model_name or _MODEL_NAME
    if not model.strip():
        raise ValueError("model_name must be a non-empty model name")
    query_vec = encode_query(query, model_name=model)

    # pgvector uses <=> for cosine distance (0 = identical, 2 = opposite)
    # similarity = 1 - distance
    sql = text("""
        SELECT
            d.id,
            d.title,
            d.content,
            d.source,
            1 - (e.embedding <=> CAST(:vec AS vector)) AS similarity
        FROM embeddings e
        JOIN documents  d ON d.id = e.doc_id
        WHERE e.model_name = :model
        ORDER BY e.embedding <=> CAST(:vec AS vector)
        LIMIT :k;
        """)

    vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vec.tolist()) + "]"

    with get_session() as session:
        rows = session.execute(
            sql, {"vec": vec_str, "model": model, "k": top_k}
        ).fetchall()

    return [
        SearchResult(
            doc_id=row.id,
            title=row.title,
            content=row.content,
            source=row.source,
            score=float(row.similarity),
            rank=idx + 1,
        )
        for idx, row in enumerate(rows)
    ]
