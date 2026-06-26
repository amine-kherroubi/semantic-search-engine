#!/usr/bin/env python3
"""
Phase 3 - Interactive search CLI.

Accepts a query via argument or interactive prompt and shows
semantic results alongside TF-IDF results for comparison.

Two modes of operation:
    1. One-shot:    pass --query "..." and the script runs once and exits.
    2. Interactive: omit --query and the script drops into a REPL-style
                    loop, reading queries from stdin until Ctrl-C/EOF.

By default, both a semantic (pgvector/embeddings) search and a classical
TF-IDF search are run side-by-side so the two can be visually compared.
Pass --no-tfidf to skip the TF-IDF comparison and only show semantic
results -- useful when you just want fast lookups and don't care about
the comparison view.

Usage:
    python scripts/search.py --query "machine learning optimization"
    python scripts/search.py          # interactive mode
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow `from src... import ...` to work when this script is run directly
# (i.e. not installed as a package). Inserts the project root -- the
# parent of this file's directory -- at the front of sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.search import semantic_search, TFIDFSearchEngine
from src.utils import compare_results, console


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for the search CLI.

    Returns:
        Parsed argparse.Namespace exposing:
            query (str | None): the search query, or None for interactive mode.
            top_k (int): number of results to retrieve.
            no_tfidf (bool): if True, skip the TF-IDF comparison search.
    """
    p = argparse.ArgumentParser(description="Search the semantic index.")
    p.add_argument("--query", type=str, default=None, help="Query string")
    p.add_argument(
        "--top-k", type=int, default=10, help="Number of results (default: 10)"
    )
    p.add_argument("--no-tfidf", action="store_true", help="Skip TF-IDF comparison")
    return p.parse_args()


def run(query: str, top_k: int, with_tfidf: bool) -> None:
    """Run a single search and print the results to the console.

    Always runs semantic search. Optionally also builds a fresh TF-IDF
    index and runs the same query against it, printing both result sets
    side by side for comparison.

    Note: rebuilding the TF-IDF index (`TFIDFSearchEngine().build()`) on
    every call is simple but means each comparison query pays the cost
    of re-fitting the vectorizer over the whole corpus. That's an
    acceptable trade-off for an interactive exploration tool, but this
    function would need to cache/reuse the engine instance for any
    higher-throughput use case.

    Args:
        query: The search query text.
        top_k: Number of results to retrieve.
        with_tfidf: If True, also run and display TF-IDF results for
            comparison. If False, show only semantic search results.
    """
    sem_results = semantic_search(query, top_k=top_k)

    if with_tfidf:
        engine = TFIDFSearchEngine().build()
        cls_results = engine.search(query, top_k=top_k)
        compare_results(sem_results, cls_results, query)
    else:
        # Local import: only needed in this branch, so it's deferred
        # here rather than imported unconditionally at module level.
        from src.utils import print_results

        print_results(sem_results, query, method="Semantic (pgvector)")


if __name__ == "__main__":
    args = parse_args()
    top_k = args.top_k
    with_tfidf = not args.no_tfidf

    if args.query:
        # One-shot mode: run exactly once with the provided query and exit.
        run(args.query, top_k, with_tfidf)
    else:
        # Interactive mode: keep prompting for queries until the user
        # exits with Ctrl-C (KeyboardInterrupt) or EOF (Ctrl-D / closed stdin).
        console.print(
            "[bold green]Semantic Search Engine[/bold green]  (Ctrl-C to quit)\n"
        )
        while True:
            try:
                query = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not query:
                # Skip empty input (e.g. the user just pressed Enter)
                # instead of running a pointless empty-string search.
                continue
            run(query, top_k, with_tfidf)
            print()
