"""
Text embedding generation using Sentence-Transformers.
Handles batching and progress tracking.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# Module-level singleton so the model loads only once per process
_model: SentenceTransformer | None = None
_loaded_model_name: str | None = None


def get_model(model_name: str | None = None) -> SentenceTransformer:
    """Return a cached SentenceTransformer, reloading only if the name changes."""
    global _model, _loaded_model_name
    name = model_name or _MODEL_NAME
    if _model is None or _loaded_model_name != name:
        print(f"[Encoder] Loading model: {name}")
        _model = SentenceTransformer(name)
        _loaded_model_name = name
    return _model


def encode_texts(
    texts: List[str],
    model_name: str | None = None,
    batch_size: int = _BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into embeddings.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32
    """
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors -> cosine sim = dot product
    )
    return embeddings.astype(np.float32)


def encode_query(query: str, model_name: str | None = None) -> np.ndarray:
    """Encode a single query string. Returns shape (dim,)."""
    return encode_texts([query], model_name=model_name, show_progress=False)[0]
