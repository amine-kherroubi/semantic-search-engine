#!/usr/bin/env python3
"""
Phase 2 - Ingestion pipeline.

Downloads the ag_news dataset from Hugging Face (~120k news articles),
pre-processes the text, generates embeddings in batches, and stores
everything in PostgreSQL + pgvector.

End-to-end flow:
    1. Download/load the ag_news dataset (Hugging Face `datasets`).
    2. Clean each article's text and split it into chunks small enough
       for the embedding model's context window.
    3. Insert document rows into Postgres, capturing their generated IDs.
    4. Encode every chunk's text into a vector embedding (batched, for
       efficiency on CPU/GPU).
    5. Insert the embeddings into the `embeddings` table, linked to their
       parent document by ID.
    6. Run VACUUM ANALYZE so the query planner has fresh statistics for
       the newly-populated table (important for index selection).

Usage:
    python scripts/ingest.py [--limit N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow `from src... import ...` to work when this script is run directly
# (i.e. not installed as a package). Inserts the project root -- the
# parent of this file's directory -- at the front of sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Any, cast
from datasets import load_dataset
from tqdm import tqdm

from src.db import (
    get_session,
    get_autocommit_connection,
    insert_documents,
    insert_embeddings,
)
from src.embeddings import encode_texts
from src.utils import clean_text, chunk_text

# Default embedding model. Can be overridden via the EMBEDDING_MODEL
# environment variable, or per-invocation via the --model CLI flag.
_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for the ingestion script.

    Returns:
        Parsed argparse.Namespace exposing:
            limit (int): max number of source articles to ingest.
            batch_size (int): embedding batch size.
            model (str): sentence-transformers model name to use.
    """
    p = argparse.ArgumentParser(
        description="Ingest documents into the semantic search DB."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Max number of documents to ingest (default: 5000)",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size (default: 64)"
    )
    p.add_argument(
        "--model",
        type=str,
        default=_MODEL,
        help=f"Sentence-transformer model (default: {_MODEL})",
    )
    return p.parse_args()


def load_ag_news(limit: int) -> list[dict]:
    """Download the AG News dataset and flatten it into ingestible rows.

    Each source article may be split into multiple "chunks" (see
    `chunk_text`) if it exceeds the embedding model's token window, so
    the number of rows returned can be greater than `limit` -- this
    function returns one row per *chunk*, not one row per article.

    Args:
        limit: Maximum number of source articles to pull from the
            dataset. (Chunks produced per article are not counted
            against this limit.)

    Returns:
        A list of dicts, one per chunk, each with keys:
            "title"    -- first sentence of the article (truncated to
                          120 chars), used as a human-readable label.
            "content"  -- the cleaned, chunked article text that will
                          actually be embedded and searched over.
            "source"   -- always "ag_news" for rows from this loader.
            "metadata" -- dict with the article's category label
                          (World / Sports / Business / Sci/Tech).

    Raises:
        ValueError: If `limit` is not a positive integer.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    print("[Ingest] Downloading ag_news from Hugging Face ...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    # AG News encodes each article's category as an integer label; map it
    # back to a human-readable string so it's useful in result
    # metadata / UI display later on.
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    rows: list[dict] = []
    for item_obj in ds.select(range(min(limit, len(ds)))):
        # `datasets` yields loosely-typed dict-like objects at runtime;
        # cast to a plain dict so static type checkers / editors know
        # what keys/types to expect below.
        item = cast(dict[str, Any], item_obj)
        cleaned = clean_text(item["text"])
        title = item["text"].split(".")[0][:120]  # first sentence as title
        meta = {"label": label_map.get(item["label"], "Unknown")}
        # chunk_text splits documents that exceed the model's token window (256 tokens).
        # AG News articles are short in practice, so chunking rarely fires, but it
        # prevents silent truncation for any articles that are unusually long.
        for chunk in chunk_text(cleaned, max_tokens=256, overlap=32):
            rows.append(
                {
                    "title": title,
                    "content": chunk,
                    "source": "ag_news",
                    "metadata": meta,
                }
            )
    print(
        f"[Ingest] Loaded {len(rows):,} document chunks from {min(limit, len(ds)):,} articles."
    )
    return rows


def run(limit: int, batch_size: int, model: str) -> None:
    """Execute the full ingestion pipeline end-to-end.

    Steps:
        1. Load + chunk the dataset (`load_ag_news`).
        2. Insert document rows into Postgres, capturing DB-generated IDs.
        3. Encode every chunk's text into an embedding vector.
        4. Insert embeddings in fixed-size batches (separate DB
           transactions per batch) to avoid one massive, memory-heavy
           transaction covering the entire dataset.
        5. Run VACUUM ANALYZE on the embeddings table so pgvector's
           IVFFlat index gets picked up by the query planner.

    Args:
        limit: Max number of source articles to ingest.
        batch_size: Number of texts encoded per embedding-model forward
            pass (larger = faster on GPU but uses more memory; tune to
            your hardware).
        model: Name of the sentence-transformers model to use for
            generating embeddings. Must match the model used at query
            time, since embeddings from different models aren't
            comparable.

    Raises:
        ValueError: If `batch_size` is not a positive integer, or if
            `model` is blank/whitespace-only.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not model.strip():
        raise ValueError("model must be a non-empty model name")

    docs = load_ag_news(limit)

    print("[Ingest] Inserting documents ...")
    with get_session() as session:
        doc_ids = insert_documents(session, docs)

    print(f"[Ingest] Inserted {len(doc_ids):,} documents. Generating embeddings ...")

    # Encode all chunk texts in a single call; encode_texts is expected
    # to internally batch by `batch_size` for memory/throughput efficiency
    # (rather than us manually slicing `texts` here).
    texts = [d["content"] for d in docs]
    embeddings = encode_texts(texts, model_name=model, batch_size=batch_size)

    print("[Ingest] Storing embeddings ...")
    records = [
        {"doc_id": doc_id, "model_name": model, "embedding": emb.tolist()}
        for doc_id, emb in zip(doc_ids, embeddings)
    ]

    # Insert in chunks to avoid huge transactions. A single transaction
    # covering all ~100k+ rows would hold locks and buffer the entire
    # insert in memory; chunking keeps each transaction small and lets
    # progress be visible via tqdm as it goes.
    chunk = 500
    for i in tqdm(range(0, len(records), chunk), desc="Storing"):
        with get_session() as session:
            insert_embeddings(session, records[i : i + chunk])

    print(f"[Ingest] OK Done. {len(doc_ids):,} document chunks with embeddings stored.")

    # Prompt the PostgreSQL query planner to use the IVFFlat index.
    # Without VACUUM ANALYZE the planner may choose a sequential scan because
    # the table statistics are stale immediately after a bulk insert.
    # VACUUM cannot run inside a transaction, so we use get_autocommit_connection()
    # which returns a psycopg2 connection with autocommit already enabled.
    print("[Ingest] Running VACUUM ANALYZE to update planner statistics ...")
    vac_conn = get_autocommit_connection()
    try:
        with vac_conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE embeddings;")
    finally:
        # Always release the raw connection, even if VACUUM raised --
        # this connection was opened just for this one statement.
        vac_conn.close()
    print("[Ingest] VACUUM ANALYZE complete.")


if __name__ == "__main__":
    args = parse_args()
    run(limit=args.limit, batch_size=args.batch_size, model=args.model)
