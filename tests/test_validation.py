"""Fast validation tests for user-facing pipeline parameters."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.evaluate import run_evaluation
from scripts.ingest import load_ag_news, run
from src.search.classical import TFIDFSearchEngine
from src.search.semantic import semantic_search


def test_ingest_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="limit"):
        load_ag_news(0)


def test_ingest_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        run(limit=1, batch_size=0, model="all-MiniLM-L6-v2")


def test_ingest_rejects_blank_model_name():
    with pytest.raises(ValueError, match="model"):
        run(limit=1, batch_size=1, model="   ")


def test_semantic_search_rejects_invalid_arguments_before_embedding():
    with pytest.raises(ValueError, match="top_k"):
        semantic_search("space mission", top_k=0)
    with pytest.raises(ValueError, match="query"):
        semantic_search("   ", top_k=1)
    with pytest.raises(ValueError, match="model_name"):
        semantic_search("space mission", top_k=1, model_name="   ")


def test_tfidf_search_rejects_invalid_arguments_before_build_check():
    engine = TFIDFSearchEngine()
    with pytest.raises(ValueError, match="top_k"):
        engine.search("space mission", top_k=0)
    with pytest.raises(ValueError, match="query"):
        engine.search("   ", top_k=1)


def test_evaluation_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        run_evaluation(top_k=0)
