"""
Classical TF-IDF keyword search for side-by-side comparison.
Loads all documents into memory and uses scikit-learn TfidfVectorizer.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.db.connection import get_session
from src.search.semantic import SearchResult  # reuse same dataclass
from sqlalchemy import text


class TFIDFSearchEngine:
    """
    In-memory TF-IDF engine built from the documents table.
    Call .build() once, then .search() any number of times.
    """

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: np.ndarray | None = None  # sparse (n_docs, vocab)
        self._doc_ids: list[int] = []
        self._titles: list[str | None] = []
        self._contents: list[str] = []
        self._sources: list[str | None] = []

    def build(self) -> "TFIDFSearchEngine":
        """Fetch all documents from DB and fit the TF-IDF matrix."""
        sql = text("SELECT id, title, content, source FROM documents ORDER BY id;")
        with get_session() as session:
            rows = session.execute(sql).fetchall()

        if not rows:
            raise RuntimeError("No documents found. Run the ingestion pipeline first.")

        self._doc_ids = [r.id for r in rows]
        self._titles = [r.title for r in rows]
        self._contents = [r.content for r in rows]
        self._sources = [r.source for r in rows]

        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.95,
            min_df=2,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(self._contents)
        print(
            f"[TF-IDF] Built index: {len(rows)} docs, vocab={self._matrix.shape[1]:,}"
        )
        return self

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Return top-k results scored by TF-IDF cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if self._vectorizer is None:
            raise RuntimeError("Call .build() before .search()")

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                doc_id=self._doc_ids[i],
                title=self._titles[i],
                content=self._contents[i],
                source=self._sources[i],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]
