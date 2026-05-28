#!/usr/bin/env python3
"""
Phase 3 - Interactive search CLI.

Accepts a query via argument or interactive prompt and shows
semantic results alongside TF-IDF results for comparison.

Usage:
    python scripts/search.py --query "machine learning optimization"
    python scripts/search.py          # interactive mode
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.search import semantic_search, TFIDFSearchEngine
from src.utils import compare_results, console


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search the semantic index.")
    p.add_argument("--query", type=str, default=None, help="Query string")
    p.add_argument(
        "--top-k", type=int, default=10, help="Number of results (default: 10)"
    )
    p.add_argument("--no-tfidf", action="store_true", help="Skip TF-IDF comparison")
    return p.parse_args()


def run(query: str, top_k: int, with_tfidf: bool) -> None:
    sem_results = semantic_search(query, top_k=top_k)

    if with_tfidf:
        engine = TFIDFSearchEngine().build()
        cls_results = engine.search(query, top_k=top_k)
        compare_results(sem_results, cls_results, query)
    else:
        from src.utils import print_results

        print_results(sem_results, query, method="Semantic (pgvector)")


if __name__ == "__main__":
    args = parse_args()
    top_k = args.top_k
    with_tfidf = not args.no_tfidf

    if args.query:
        run(args.query, top_k, with_tfidf)
    else:
        console.print(
            "[bold green]Semantic Search Engine[/bold green]  (Ctrl-C to quit)\n"
        )
        while True:
            try:
                query = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not query:
                continue
            run(query, top_k, with_tfidf)
            print()
