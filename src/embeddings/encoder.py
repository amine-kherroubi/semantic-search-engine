"""
Text embedding generation using Sentence-Transformers.
Handles batching and progress tracking.

Environment variables (set in .env):
    EMBEDDING_MODEL      - Sentence-Transformers model name (default: all-MiniLM-L6-v2)
    BATCH_SIZE           - Number of texts per embedding batch (default: 64)
    HF_TOKEN             - Optional Hugging Face API token. Enables higher download
                           rate limits when pulling models from the Hub. Generate one
                           at https://huggingface.co/settings/tokens (read-only scope
                           is sufficient). Leave unset to use unauthenticated access.
    CUDA_VISIBLE_DEVICES - Set to "" in .env to force CPU inference and suppress the
                           CUDA driver version warning emitted by PyTorch when the
                           installed driver is too old to be used.
"""

from __future__ import annotations

import os

# Suppress PyTorch's CUDA driver version warning.
# The warning fires when PyTorch detects a GPU but the system NVIDIA driver is
# too old to be used (driver < 525 for CUDA 12). Setting CUDA_VISIBLE_DEVICES=""
# tells PyTorch not to probe for GPUs, so inference falls back to CPU silently.
# Remove this block (or set CUDA_VISIBLE_DEVICES="" only in .env) once the driver
# is updated to a compatible version.
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
    texts: list[str],
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
