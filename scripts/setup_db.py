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
    console.print("[bold]Checking database connection...[/bold]")
    if not test_connection():
        console.print("[red]Cannot reach the database. Check your .env settings.[/red]")
        sys.exit(1)
    console.print("[green]OK Connected[/green]")

    schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"
    sql = schema_path.read_text()

    # Split by semicolon to execute statements individually.
    # We ignore empty statements caused by trailing semicolons.
    # Extension creation is handled explicitly below so privilege failures
    # can be reported without immediately re-running the same statement.
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    schema_statements = [
        statement
        for statement in statements
        if not statement.lower().startswith("create extension")
    ]

    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Handle extension separately
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
            except Exception as e:
                console.print(
                    "[yellow]Note: Could not create extension "
                    "(it may already exist or you lack superuser privileges): "
                    f"{e}[/yellow]"
                )
                conn.rollback()

            # 2. Handle schema
            for statement in schema_statements:
                try:
                    cur.execute(statement)
                except Exception as e:
                    console.print(
                        "[red]Error executing statement: "
                        f"{statement[:50]}... - {e}[/red]"
                    )
                    raise

    console.print("[green]OK Schema applied.[/green]")
    console.print("Run [bold]python scripts/ingest.py[/bold] to load documents.")


if __name__ == "__main__":
    main()
