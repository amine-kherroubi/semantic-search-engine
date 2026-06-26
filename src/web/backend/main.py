#!/usr/bin/env python3
"""
FastAPI backend for the web UI (src/web).

Did not exist in the original project -- everything was a CLI script
(scripts/search.py, scripts/evaluate.py). This module wraps the existing
src/search and src/db code in a small HTTP API so the browser frontend
in src/web/frontend can query it.

Endpoints:
    GET  /api/health                 -- liveness + index-readiness check
    GET  /api/articles               -- paginated article listing
    GET  /api/articles/{doc_id}      -- single article, full content
    POST /api/search                 -- run a query against one retrieval
                                         approach (tfidf/semantic)

Run with:
    uvicorn src.web.backend.main:app --reload --port 8000
(from the project root, so the `src.*` imports resolve)
"""

from __future__ import annotations

import os
import sys
import time

# Allow `from src... import ...` to work when this file is run directly
# (e.g. `python src/web/backend/main.py`) and not just via `uvicorn
# src.web.backend.main:app` from the project root. Mirrors the same
# sys.path shim used by scripts/search.py and scripts/ingest.py.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from src.db.connection import get_session
from src.search import TFIDFSearchEngine, semantic_search
from src.utils import truncate
from src.web.backend.engines import get_tfidf_engine, indices_ready, rebuild_all
from src.web.backend.schemas import (
    ArticleDetail,
    ArticleListItem,
    ArticleListResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchStats,
)
from src.web.backend.snippets import make_snippet

app = FastAPI(
    title="Semantic Search Engine API",
    description="Lightweight API over the AG News pgvector search engine.",
    version="1.0.0",
)

# Permissive CORS: this is a local developer tool, not a public deployment.
# Tighten allow_origins if exposing this beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Articles listing
# ---------------------------------------------------------------------------


@app.get("/api/articles", response_model=ArticleListResponse)
def list_articles(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    category: str | None = Query(
        default=None, description="Filter by AG News category label"
    ),
    q: str | None = Query(
        default=None, description="Optional substring filter on title/content"
    ),
) -> ArticleListResponse:
    """
    Server-side paginated article listing.

    Note on fields: the underlying `documents` table (see sql/schema.sql)
    has no dedicated "date" column -- AG News itself ships no per-article
    publish date. `created_at` (the row's ingestion timestamp) is
    returned in its place; it reflects when the row was loaded into this
    database, not when the article was originally published. "category"
    is not a real column either; it's read out of the `metadata` JSONB
    field (`metadata->>'label'`), which is where scripts/ingest.py stores
    the AG News World/Sports/Business/Sci-Tech label.
    """
    offset = (page - 1) * page_size

    where_clauses = []
    params: dict = {"limit": page_size, "offset": offset}

    if category:
        where_clauses.append("d.metadata->>'label' = :category")
        params["category"] = category

    if q:
        where_clauses.append("(d.title ILIKE :q OR d.content ILIKE :q)")
        params["q"] = f"%{q}%"

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_sql = text(f"SELECT count(*) FROM documents d {where_sql};")
    list_sql = text(f"""
        SELECT d.id, d.title, d.source, d.metadata, d.created_at, d.content
        FROM documents d
        {where_sql}
        ORDER BY d.id
        LIMIT :limit OFFSET :offset;
        """)

    with get_session() as session:
        total = session.execute(count_sql, params).scalar_one()
        rows = session.execute(list_sql, params).fetchall()

    items = [
        ArticleListItem(
            id=row.id,
            title=row.title,
            source=row.source,
            category=(row.metadata or {}).get("label"),
            created_at=row.created_at.isoformat() if row.created_at else None,
            metadata=row.metadata or {},
            content_preview=truncate(row.content, max_chars=200),
        )
        for row in rows
    ]

    total_pages = max(1, (total + page_size - 1) // page_size)

    return ArticleListResponse(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
    )


@app.get("/api/articles/{doc_id}", response_model=ArticleDetail)
def get_article(doc_id: int) -> ArticleDetail:
    """Full content + metadata for a single article (used by the expandable preview)."""
    sql = text(
        "SELECT id, title, source, metadata, created_at, content FROM documents WHERE id = :doc_id;"
    )
    with get_session() as session:
        row = session.execute(sql, {"doc_id": doc_id}).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"No article with id {doc_id}")

    return ArticleDetail(
        id=row.id,
        title=row.title,
        source=row.source,
        category=(row.metadata or {}).get("label"),
        created_at=row.created_at.isoformat() if row.created_at else None,
        metadata=row.metadata or {},
        content=row.content,
    )


@app.get("/api/categories", response_model=list[str])
def list_categories() -> list[str]:
    """Distinct category labels, for populating a filter dropdown."""
    sql = text(
        "SELECT DISTINCT metadata->>'label' AS label FROM documents WHERE metadata->>'label' IS NOT NULL ORDER BY 1;"
    )
    with get_session() as session:
        rows = session.execute(sql).fetchall()
    return [r.label for r in rows]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Run `req.query` through the selected retrieval approach and return
    ranked results plus query statistics.

    The TF-IDF index is built lazily on first use (see engines.py) and
    reused after that; semantic search hits pgvector directly on every
    call since the ANN index already lives in Postgres.
    """
    start = time.perf_counter()
    indexing_method = ""
    notes: str | None = None

    try:
        if req.approach == "semantic":
            results = semantic_search(req.query, top_k=req.top_k)
            indexing_method = "pgvector IVFFlat (cosine similarity)"

        elif req.approach == "tfidf":
            engine: TFIDFSearchEngine = get_tfidf_engine()
            results = engine.search(req.query, top_k=req.top_k)
            indexing_method = "In-memory TF-IDF (scikit-learn, bigrams)"

        else:  # pragma: no cover - guarded by Pydantic Literal already
            raise HTTPException(
                status_code=400, detail=f"Unknown approach: {req.approach}"
            )

    except RuntimeError as exc:
        # Raised by TFIDFSearchEngine.build() when the documents table is
        # empty -- surface as a clean 503 rather than a generic 500,
        # since it's a "run ingestion first" state, not a bug.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000

    # Fetch categories for the result set in one query rather than N+1.
    doc_ids = [r.doc_id for r in results]
    categories: dict[int, str | None] = {}
    if doc_ids:
        cat_sql = text(
            "SELECT id, metadata->>'label' AS label FROM documents WHERE id = ANY(:ids);"
        )
        with get_session() as session:
            rows = session.execute(cat_sql, {"ids": doc_ids}).fetchall()
        categories = {r.id: r.label for r in rows}

    result_items = [
        SearchResultItem(
            doc_id=r.doc_id,
            rank=r.rank,
            title=r.title,
            score=r.score,
            snippet=make_snippet(r.content, req.query),
            source=r.source,
            category=categories.get(r.doc_id),
            approach=req.approach,
        )
        for r in results
    ]

    stats = SearchStats(
        query=req.query,
        approach=req.approach,
        latency_ms=round(latency_ms, 2),
        num_results=len(result_items),
        indexing_method=indexing_method,
        notes=notes,
    )

    return SearchResponse(results=result_items, stats=stats)


# ---------------------------------------------------------------------------
# Health / admin
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe + in-memory index readiness, shown in the UI's status strip."""
    try:
        with get_session() as session:
            doc_count = session.execute(
                text("SELECT count(*) FROM documents;")
            ).scalar_one()
        status = "ok"
    except Exception:
        doc_count = 0
        status = "db_unreachable"

    return HealthResponse(
        status=status, document_count=doc_count, indices_ready=indices_ready()
    )


@app.post("/api/admin/rebuild-indices")
def admin_rebuild_indices() -> dict[str, str]:
    """Force a rebuild of the in-memory TF-IDF index (e.g. after re-ingestion)."""
    rebuild_all()
    return {"status": "rebuilt"}


# ---------------------------------------------------------------------------
# Static frontend (served from the same process for simplicity)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "frontend"
)
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
