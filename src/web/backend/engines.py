"""
Lazy-loaded, process-wide singleton for the in-memory TF-IDF engine.

TF-IDF requires a full scan of the `documents` table to build its index
(TfidfVectorizer.fit_transform), which takes real time on a large
corpus. Rebuilding on every search request would make the API
unusably slow, so the engine is built once, lazily, on first use, and
cached for the lifetime of the server process.

This is intentionally simple (a module-level global + a lock) rather
than a full cache framework -- "keep the implementation simple and
modular" per the brief. If the underlying documents table changes
(e.g. re-ingestion), restart the server to pick up the new corpus, or
call POST /api/admin/rebuild-indices.
"""

from __future__ import annotations

import threading
import time

from src.search import TFIDFSearchEngine

_lock = threading.Lock()

_tfidf_engine: TFIDFSearchEngine | None = None

_build_times: dict[str, float] = {}


def get_tfidf_engine() -> TFIDFSearchEngine:
    """Return the process-wide TF-IDF engine, building it on first call."""
    global _tfidf_engine
    if _tfidf_engine is None:
        with _lock:
            if _tfidf_engine is None:  # re-check inside the lock
                start = time.perf_counter()
                _tfidf_engine = TFIDFSearchEngine().build()
                _build_times["tfidf"] = time.perf_counter() - start
    return _tfidf_engine


def indices_ready() -> dict[str, bool]:
    """Report which in-memory indices have been built so far (for /health)."""
    return {
        "tfidf": _tfidf_engine is not None,
        "semantic": True,  # pgvector index lives in Postgres, always "ready"
    }


def rebuild_all() -> None:
    """Force a rebuild of every in-memory index. Use after re-ingestion."""
    global _tfidf_engine
    with _lock:
        _tfidf_engine = None
    get_tfidf_engine()
