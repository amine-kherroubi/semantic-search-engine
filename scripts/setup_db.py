#!/usr/bin/env python3
"""
Initialise the database: create tables and indexes from sql/schema.sql.
Safe to re-run (uses CREATE IF NOT EXISTS).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.db.connection import get_raw_connection, test_connection
from src.utils import console


def main() -> None:
    console.print("[bold]Checking database connection …[/bold]")
    if not test_connection():
        console.print("[red]Cannot reach the database. Check your .env settings.[/red]")
        sys.exit(1)
    console.print("[green]✓ Connected[/green]")

    schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"
    sql = schema_path.read_text()

    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    console.print("[green]✓ Schema applied.[/green]")
    console.print("Run  [bold]python scripts/ingest.py[/bold]  to load documents.")


if __name__ == "__main__":
    main()
