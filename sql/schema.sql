create extension if not exists vector;

create table if not exists documents (
    id          serial primary key,
    title       text,
    content     text not null,
    source      text,
    metadata    jsonb default '{}',
    created_at  timestamptz default now()
);

create table if not exists embeddings (
    id          serial primary key,
    doc_id      integer not null references documents(id) on delete cascade,
    model_name  text not null,
    embedding   vector(384) not null,
    created_at  timestamptz default now()
);

create index if not exists embeddings_ivfflat_idx
    on embeddings
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

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
