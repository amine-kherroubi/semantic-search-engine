"""
Terminal display helpers using rich.
"""
from __future__ import annotations

from typing import List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.search.semantic import SearchResult

console = Console()


def print_results(results: List[SearchResult], query: str, method: str = "Semantic") -> None:
    """Pretty-print a ranked list of SearchResult objects."""
    console.print(
        Panel(f"[bold cyan]{method} Search[/bold cyan]  ·  Query: [yellow]{query}[/yellow]",
              expand=False)
    )

    if not results:
        console.print("[red]No results found.[/red]")
        return

    table = Table(box=box.ROUNDED, show_lines=True, expand=True)
    table.add_column("#",       style="bold", width=3, justify="right")
    table.add_column("Score",   style="green", width=7, justify="right")
    table.add_column("Title",   style="bold white", min_width=20)
    table.add_column("Snippet", style="dim", min_width=40)

    for r in results:
        snippet = (r.content or "")[:180].replace("\n", " ") + "…"
        table.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            r.title or f"doc-{r.doc_id}",
            snippet,
        )

    console.print(table)


def compare_results(
    semantic_results: List[SearchResult],
    classical_results: List[SearchResult],
    query: str,
) -> None:
    """Print semantic and TF-IDF results side by side."""
    console.rule(f"[bold]Query: {query}[/bold]")
    print_results(semantic_results, query, method="Semantic (pgvector)")
    print_results(classical_results, query, method="Classical (TF-IDF)")

    # Overlap analysis
    sem_ids = {r.doc_id for r in semantic_results}
    cls_ids = {r.doc_id for r in classical_results}
    overlap = sem_ids & cls_ids
    console.print(
        f"\n[bold]Overlap:[/bold] {len(overlap)} / {len(sem_ids)} documents "
        f"appear in both result sets."
    )
