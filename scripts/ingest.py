#!/usr/bin/env python3
"""
Phase 2 — Ingestion pipeline.

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

from src.db import get_session, insert_documents, insert_embeddings
from src.embeddings import encode_texts
from src.utils import clean_text

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
    print("[Ingest] Downloading ag_news from Hugging Face …")
    ds = load_dataset("ag_news", split="train", trust_remote_code=True)
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    rows = []
    for item_obj in ds.select(range(min(limit, len(ds)))):
        item = cast(dict[str, Any], item_obj)
        rows.append(
            {
                "title": item["text"].split(".")[0][:120],  # first sentence as title
                "content": clean_text(item["text"]),
                "source": "ag_news",
                "metadata": {"label": label_map.get(item["label"], "Unknown")},
            }
        )
    print(f"[Ingest] Loaded {len(rows):,} documents.")
    return rows


def run(limit: int, batch_size: int, model: str) -> None:
    docs = load_ag_news(limit)

    print("[Ingest] Inserting documents …")
    with get_session() as session:
        doc_ids = insert_documents(session, docs)

    print(f"[Ingest] Inserted {len(doc_ids):,} documents. Generating embeddings …")

    texts = [d["content"] for d in docs]
    embeddings = encode_texts(texts, model_name=model, batch_size=batch_size)

    print("[Ingest] Storing embeddings …")
    records = [
        {"doc_id": doc_id, "model_name": model, "embedding": emb.tolist()}
        for doc_id, emb in zip(doc_ids, embeddings)
    ]

    # Insert in chunks to avoid huge transactions
    chunk = 500
    for i in tqdm(range(0, len(records), chunk), desc="Storing"):
        with get_session() as session:
            insert_embeddings(session, records[i : i + chunk])

    print(f"[Ingest] ✓ Done. {len(doc_ids):,} documents with embeddings stored.")


if __name__ == "__main__":
    args = parse_args()
    run(limit=args.limit, batch_size=args.batch_size, model=args.model)
