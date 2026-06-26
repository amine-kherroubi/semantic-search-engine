-- Schema for the semantic search engine.
-- Safe to re-run: every statement uses IF NOT EXISTS / CREATE OR REPLACE.
-- Applied by scripts/setup_db.py.

create extension if not exists vector;

-- One row per searchable text chunk (a long article may be split into
-- several chunks by scripts/ingest.py, each becoming its own row).
create table if not exists documents (
    id          serial primary key,
    title       text,
    content     text not null,
    source      text,
    metadata    jsonb default '{}',
    created_at  timestamptz default now()
);

-- One row per (document, embedding model). Kept separate from
-- `documents` so a document can have embeddings from multiple models
-- and re-embedding doesn't require touching document content.
create table if not exists embeddings (
    id          serial primary key,
    doc_id      integer not null references documents(id) on delete cascade,
    model_name  text not null,            -- which model produced this vector; embeddings from different models aren't comparable
    embedding   vector(384) not null,     -- dimension must match the embedding model in use (384 = all-MiniLM-L6-v2)
    created_at  timestamptz default now()
);

-- Approximate nearest-neighbor index for fast cosine similarity search.
-- `lists = 100` is a reasonable default for tens of thousands of rows;
-- rebuild (REINDEX) after the corpus is fully loaded for best recall.
create index if not exists embeddings_ivfflat_idx
    on embeddings
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Convenience join of documents to their embeddings. Inner join, so
-- documents without an embedding yet won't appear, and documents with
-- multiple models' embeddings appear once per model.
create or replace view documents_with_embeddings as
select
    d.id,
    d.title,
    d.content,
    d.source,
    d.metadata,
    e.embedding,
    e.model_name
from documents d
join embeddings e on e.doc_id = d.id;
