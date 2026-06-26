#!/usr/bin/env python3
"""
Phase 4 - Validation & Critical Analysis.

This script evaluates the quality and performance of the semantic search
engine by comparing it against a classical TF-IDF baseline. It runs a
fixed battery of test queries against both retrieval methods, captures
timing and relevance metrics for each, and produces:

  1. A console table summarizing the results (via `tabulate`).
  2. A CSV file with the same data (data/evaluation_results.csv).
  3. A three-panel chart (data/evaluation_results.png) comparing:
       - query latency (semantic vs. TF-IDF)
       - result overlap between the two methods (top-k overlap)
       - average relevance score per method

The goal isn't to "prove" one method is objectively better, but to give
a quick, repeatable sanity check that the semantic search pipeline
(embedding model + pgvector) behaves sensibly relative to a simple,
well-understood baseline -- and to catch regressions over time.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow `from src... import ...` to work when this script is run directly
# (i.e. not installed as a package). Inserts the project root -- the
# parent of this file's directory -- at the front of sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: lets this run headless
# (e.g. on a server/CI box with no display) and just write PNGs to disk
# instead of trying to pop up a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.search import semantic_search, TFIDFSearchEngine
from src.utils import console

# Directory where evaluation artifacts (CSV + PNG) are written.
# Resolved relative to this file so the script works regardless of the
# current working directory it's invoked from.
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# A fixed, hand-picked set of queries spanning several different news
# topics/domains (finance, science, politics, sports, etc.). Using a
# fixed list (rather than random sampling) keeps evaluation results
# reproducible across runs, so changes to the pipeline can be compared
# apples-to-apples over time.
TEST_QUERIES = [
    "stock market crash financial crisis",
    "NASA space mission launch satellite",
    "election president campaign voters",
    "artificial intelligence machine learning neural network",
    "olympic games athletes performance",
    "climate change global warming environment",
    "pharmaceutical drug approval FDA medical",
    "football team championship trophy",
    "technology company merger acquisition",
    "terrorism security threat attack",
]


def timed_search(fn, *args, **kwargs):
    """Run `fn(*args, **kwargs)` and measure its wall-clock execution time.

    A small generic timing helper so the same timing logic doesn't need
    to be duplicated for both the semantic search call and the TF-IDF
    search call in `run_evaluation`.

    Args:
        fn: Any callable to time (e.g. `semantic_search` or `engine.search`).
        *args: Positional arguments forwarded directly to `fn`.
        **kwargs: Keyword arguments forwarded directly to `fn`.

    Returns:
        A tuple of `(result_of_fn, elapsed_seconds)`, where
        `elapsed_seconds` is measured using `time.perf_counter()` (a
        high-resolution, monotonic clock well suited to timing short
        operations -- unlike `time.time()`, it isn't affected by system
        clock adjustments).
    """
    start = time.perf_counter()
    results = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return results, elapsed


def run_evaluation(top_k: int = 10) -> None:
    """Run the full evaluation suite and write a report (CSV + chart).

    For each query in `TEST_QUERIES`, this:
      1. Runs semantic search (pgvector / embeddings) and times it.
      2. Runs TF-IDF search and times it.
      3. Computes the percentage overlap between the two result sets
         (how many of the same documents appear in both top-k lists).
      4. Computes the average relevance score returned by each method.

    Results are collected into a pandas DataFrame, printed as a table,
    saved to CSV, and visualized as a 3-panel matplotlib figure.

    Args:
        top_k: Number of results to retrieve per query for each search
            method. Must be a positive integer; it's also used as the
            denominator when computing the overlap percentage.

    Raises:
        ValueError: If `top_k` is not a positive integer.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    console.print("[bold]Building TF-IDF index ...[/bold]")
    # Build the TF-IDF engine once, up front (fitting the vectorizer over
    # the whole corpus), rather than rebuilding it per query -- that
    # fitting step is the expensive part and is query-independent.
    tfidf = TFIDFSearchEngine().build()

    rows = []
    for query in TEST_QUERIES:
        sem_results, sem_time = timed_search(semantic_search, query, top_k=top_k)
        cls_results, cls_time = timed_search(tfidf.search, query, top_k=top_k)

        # Compare result sets by document ID to measure how much the two
        # retrieval strategies agree on "what's relevant" for this query.
        sem_ids = {r.doc_id for r in sem_results}
        cls_ids = {r.doc_id for r in cls_results}
        overlap_pct = 100 * len(sem_ids & cls_ids) / top_k if top_k > 0 else 0

        # Average relevance/similarity score reported by each engine.
        # NOTE: these scores are NOT directly comparable across methods --
        # cosine similarity from dense embeddings and TF-IDF cosine score
        # live on different scales/distributions. They're tracked
        # separately mainly to spot anomalies (e.g. a method returning
        # near-zero scores for everything, which would suggest a bug)
        # rather than to declare a numeric "winner".
        sem_avg_score = np.mean([r.score for r in sem_results]) if sem_results else 0.0
        cls_avg_score = np.mean([r.score for r in cls_results]) if cls_results else 0.0

        rows.append(
            {
                "Query": query[:50],  # truncate long queries for display
                "Sem Time (ms)": round(sem_time * 1000, 1),
                "TF-IDF Time (ms)": round(cls_time * 1000, 1),
                "Overlap %": round(overlap_pct, 1),
                "Sem Avg Score": round(float(sem_avg_score), 4),
                "TFIDF Avg Score": round(float(cls_avg_score), 4),
            }
        )

    df = pd.DataFrame(rows)
    console.print("\n[bold cyan]Evaluation Results[/bold cyan]")
    print(tabulate(df, headers="keys", tablefmt="plain", showindex=False))  # type: ignore

    # --- Persist raw results as CSV for later analysis / record-keeping ---
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"\n[green]CSV saved:[/green] {csv_path}")

    # --- Build a 3-panel summary chart ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Semantic vs TF-IDF Search - Evaluation", fontsize=14, fontweight="bold"
    )

    # Panel 1: Query latency, drawn as side-by-side bars per query so it's
    # easy to see whether semantic search is consistently slower/faster
    # than the TF-IDF baseline across different topics.
    ax = axes[0]
    x = np.arange(len(TEST_QUERIES))
    w = 0.35  # bar width; offsetting by +/- w/2 places the two bars side by side
    ax.bar(x - w / 2, df["Sem Time (ms)"], w, label="Semantic", color="#4C8BF5")
    ax.bar(x + w / 2, df["TF-IDF Time (ms)"], w, label="TF-IDF", color="#F5A623")
    ax.set_title("Query Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [q[:20] for q in df["Query"]], rotation=45, ha="right", fontsize=7
    )
    ax.legend()
    ax.set_ylabel("ms")

    # Panel 2: Result overlap (%) -- how similar are the two methods' top-k
    # results, per query. Low overlap doesn't necessarily mean either
    # method is "wrong"; it can simply mean lexical similarity (TF-IDF)
    # and semantic similarity (embeddings) diverge for that query.
    ax = axes[1]
    ax.bar(x, df["Overlap %"], color="#34A853")
    ax.set_title(f"Result Overlap (top-{top_k})")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [q[:20] for q in df["Query"]], rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    # Panel 3: Average relevance score per method, per query. Plotted as
    # line charts (rather than bars) to make trends across the query set
    # easier to follow at a glance.
    ax = axes[2]
    ax.plot(df["Sem Avg Score"], "o-", label="Semantic", color="#4C8BF5")
    ax.plot(df["TFIDF Avg Score"], "s--", label="TF-IDF", color="#F5A623")
    ax.set_title("Avg Relevance Score")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [q[:20] for q in df["Query"]], rotation=45, ha="right", fontsize=7
    )
    ax.legend()
    ax.set_ylabel("score")

    plt.tight_layout()
    img_path = OUTPUT_DIR / "evaluation_results.png"
    plt.savefig(img_path, dpi=150)
    console.print(f"[green]Chart saved:[/green] {img_path}")
    plt.close()  # free the figure's memory -- matters in long-running processes


if __name__ == "__main__":
    run_evaluation()
