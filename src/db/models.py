"""
SQLAlchemy ORM models and CRUD helpers for documents and embeddings.
"""
from __future__ import annotations

import json
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size


class Document(Base):
    __tablename__ = "documents"

    id         = Column(Integer, primary_key=True)
    title      = Column(Text)
    content    = Column(Text, nullable=False)
    source     = Column(Text)
    meta       = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    embeddings = relationship("Embedding", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        snippet = (self.content or "")[:60].replace("\n", " ")
        return f"<Document id={self.id} title={self.title!r} content={snippet!r}...>"


class Embedding(Base):
    __tablename__ = "embeddings"

    id         = Column(Integer, primary_key=True)
    doc_id     = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(Text, nullable=False)
    embedding  = Column(Vector(EMBEDDING_DIM), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document   = relationship("Document", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<Embedding id={self.id} doc_id={self.doc_id} model={self.model_name!r}>"


# CRUD helpers

def insert_documents(session, rows: list[dict[str, Any]]) -> list[int]:
    """
    Bulk-insert documents.

    rows: list of dicts with keys: title, content, source, metadata
    Returns list of inserted IDs (same order).
    """
    objs = [
        Document(
            title=r.get("title"),
            content=r["content"],
            source=r.get("source"),
            meta=r.get("metadata", {}),
        )
        for r in rows
    ]
    session.add_all(objs)
    session.flush()  # populate .id without committing
    return [o.id for o in objs]


def insert_embeddings(session, records: list[dict[str, Any]]) -> None:
    """
    Bulk-insert embeddings.

    records: list of dicts with keys: doc_id, model_name, embedding (list[float])
    """
    objs = [
        Embedding(
            doc_id=r["doc_id"],
            model_name=r["model_name"],
            embedding=r["embedding"],
        )
        for r in records
    ]
    session.add_all(objs)
    session.flush()


def get_document(session, doc_id: int) -> Document | None:
    return session.get(Document, doc_id)
