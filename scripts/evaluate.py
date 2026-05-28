#!/usr/bin/env python3
"""
Phase 4 - Validation & Critical Analysis.

Runs a fixed set of test queries, measures response times, computes
overlap between semantic and TF-IDF results, and produces a
summary report with charts saved to data/evaluation_results.png
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.search import semantic_search, TFIDFSearchEngine
from src.utils import console

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    start = time.perf_counter()
    results = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return results, elapsed


def run_evaluation(top_k: int = 10) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    console.print("[bold]Building TF-IDF index ...[/bold]")
    tfidf = TFIDFSearchEngine().build()

    rows = []
    for query in TEST_QUERIES:
        sem_results, sem_time = timed_search(semantic_search, query, top_k=top_k)
        cls_results, cls_time = timed_search(tfidf.search, query, top_k=top_k)

        sem_ids = {r.doc_id for r in sem_results}
        cls_ids = {r.doc_id for r in cls_results}
        overlap_pct = 100 * len(sem_ids & cls_ids) / top_k if top_k > 0 else 0

        sem_avg_score = np.mean([r.score for r in sem_results]) if sem_results else 0.0
        cls_avg_score = np.mean([r.score for r in cls_results]) if cls_results else 0.0

        rows.append(
            {
                "Query": query[:50],
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

    # Save CSV
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"\n[green]CSV saved:[/green] {csv_path}")

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Semantic vs TF-IDF Search - Evaluation", fontsize=14, fontweight="bold"
    )

    # 1. Latency comparison
    ax = axes[0]
    x = np.arange(len(TEST_QUERIES))
    w = 0.35
    ax.bar(x - w / 2, df["Sem Time (ms)"], w, label="Semantic", color="#4C8BF5")
    ax.bar(x + w / 2, df["TF-IDF Time (ms)"], w, label="TF-IDF", color="#F5A623")
    ax.set_title("Query Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [q[:20] for q in df["Query"]], rotation=45, ha="right", fontsize=7
    )
    ax.legend()
    ax.set_ylabel("ms")

    # 2. Overlap
    ax = axes[1]
    ax.bar(x, df["Overlap %"], color="#34A853")
    ax.set_title(f"Result Overlap (top-{top_k})")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [q[:20] for q in df["Query"]], rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    # 3. Average score
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
    plt.close()


if __name__ == "__main__":
    run_evaluation()
