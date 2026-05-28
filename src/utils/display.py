"""
Terminal display helpers using rich.
"""

from __future__ import annotations

from typing import List

from rich.console import Console

from src.search.semantic import SearchResult

console = Console()


def print_results(
    results: List[SearchResult], query: str, method: str = "Semantic"
) -> None:
    """Print a ranked list of SearchResult objects without box drawing."""
    console.print(
        f"[bold cyan]{method} Search[/bold cyan] - "
        f"Query: [yellow]{query}[/yellow]"
    )

    if not results:
        console.print("[red]No results found.[/red]")
        return

    for r in results:
        snippet = (r.content or "")[:180].replace("\n", " ")
        if len(r.content or "") > 180:
            snippet = snippet.rstrip() + "..."
        title = r.title or f"doc-{r.doc_id}"
        console.print(f"{r.rank}. Score: {r.score:.4f}")
        console.print(f"   Title: {title}")
        console.print(f"   Snippet: [dim]{snippet}[/dim]")


def compare_results(
    semantic_results: List[SearchResult],
    classical_results: List[SearchResult],
    query: str,
) -> None:
    """Print semantic and TF-IDF results side by side."""
    console.print(f"Query: {query}")
    print_results(semantic_results, query, method="Semantic (pgvector)")
    print_results(classical_results, query, method="Classical (TF-IDF)")

    sem_ids: set[int] = {r.doc_id for r in semantic_results}
    cls_ids: set[int] = {r.doc_id for r in classical_results}
    overlap: set[int] = sem_ids & cls_ids
    console.print(
        f"\nOverlap: {len(overlap)} / {len(sem_ids)} documents "
        f"appear in both result sets."
    )
