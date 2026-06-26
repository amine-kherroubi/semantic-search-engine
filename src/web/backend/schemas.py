"""
Pydantic schemas for the web API request/response bodies.

Kept separate from main.py so the route handlers stay focused on logic
and the wire format is defined in one obvious place.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalApproach = Literal["tfidf", "semantic"]


# ---- Articles listing -------------------------------------------------


class ArticleListItem(BaseModel):
    id: int
    title: str | None
    source: str | None
    category: str | None  # derived from metadata["label"]; AG News categories
    created_at: str | None  # ISO timestamp; see note in routes.py re: "date"
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_preview: str  # short preview shown on hover/expand


class ArticleListResponse(BaseModel):
    items: list[ArticleListItem]
    page: int
    page_size: int
    total: int
    total_pages: int


class ArticleDetail(BaseModel):
    id: int
    title: str | None
    source: str | None
    category: str | None
    created_at: str | None
    metadata: dict[str, Any]
    content: str


# ---- Search -------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    approach: RetrievalApproach = "semantic"
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResultItem(BaseModel):
    doc_id: int
    rank: int
    title: str | None
    score: float
    snippet: str  # query-highlighted excerpt of the content
    source: str | None
    category: str | None
    approach: RetrievalApproach


class SearchStats(BaseModel):
    query: str
    approach: RetrievalApproach
    latency_ms: float
    num_results: int
    indexing_method: str
    notes: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    stats: SearchStats


class HealthResponse(BaseModel):
    status: str
    document_count: int
    indices_ready: dict[str, bool]
