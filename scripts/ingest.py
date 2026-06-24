#!/usr/bin/env python3
"""
Phase 2 - Ingestion pipeline.

Downloads the ag_news dataset from Hugging Face (~120k news articles),
pre-processes the text, generates embeddings in batches, and stores
everything in PostgreSQL + pgvector.

Usage:
    python scripts/ingest.py [--limit N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Any, cast
from datasets import load_dataset
from tqdm import tqdm

from src.db import get_session, get_raw_connection, insert_documents, insert_embeddings
from src.embeddings import encode_texts
from src.utils import clean_text, chunk_text

_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def parse_args() -> argparse.Namespace:
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
    """Download and flatten the AG News dataset."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    print("[Ingest] Downloading ag_news from Hugging Face ...")
    ds = load_dataset("ag_news", split="train", trust_remote_code=True)
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    rows: list[dict] = []
    for item_obj in ds.select(range(min(limit, len(ds)))):
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
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not model.strip():
        raise ValueError("model must be a non-empty model name")

    docs = load_ag_news(limit)

    print("[Ingest] Inserting documents ...")
    with get_session() as session:
        doc_ids = insert_documents(session, docs)

    print(f"[Ingest] Inserted {len(doc_ids):,} documents. Generating embeddings ...")

    texts = [d["content"] for d in docs]
    embeddings = encode_texts(texts, model_name=model, batch_size=batch_size)

    print("[Ingest] Storing embeddings ...")
    records = [
        {"doc_id": doc_id, "model_name": model, "embedding": emb.tolist()}
        for doc_id, emb in zip(doc_ids, embeddings)
    ]

    # Insert in chunks to avoid huge transactions
    chunk = 500
    for i in tqdm(range(0, len(records), chunk), desc="Storing"):
        with get_session() as session:
            insert_embeddings(session, records[i : i + chunk])

    print(f"[Ingest] OK Done. {len(doc_ids):,} document chunks with embeddings stored.")

    # Prompt the PostgreSQL query planner to use the IVFFlat index.
    # Without VACUUM ANALYZE the planner may choose a sequential scan because
    # the table statistics are stale immediately after a bulk insert.
    print("[Ingest] Running VACUUM ANALYZE to update planner statistics ...")
    with get_raw_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE embeddings;")
    print("[Ingest] VACUUM ANALYZE complete.")


if __name__ == "__main__":
    args = parse_args()
    run(limit=args.limit, batch_size=args.batch_size, model=args.model)
