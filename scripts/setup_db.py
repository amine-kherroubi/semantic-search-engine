#!/usr/bin/env python3
"""
Initialise the database: create tables and indexes from sql/schema.sql.
Safe to re-run (uses CREATE IF NOT EXISTS).

What this script does, step by step:
    1. Verify the database is reachable (fail fast with a clear message
       if not, rather than letting later statements throw obscure errors).
    2. Read sql/schema.sql and split it into individual statements.
    3. Create the `vector` extension (required by pgvector) in its own
       try/except, since this step alone requires elevated privileges
       and is the most common point of failure in new environments.
    4. Execute the remaining schema statements (tables, indexes, etc.).

Run this once when setting up a new environment, and re-run safely any
time the schema needs to be (re-)applied -- the SQL uses `IF NOT EXISTS`
guards so it won't error out on objects that already exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `from src... import ...` to work when this script is run directly
# (i.e. not installed as a package). Inserts the project root -- the
# parent of this file's directory -- at the front of sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.db.connection import get_raw_connection, test_connection
from src.utils import console


def main() -> None:
    """Connect to the database and apply the schema in sql/schema.sql.

    Behavior:
        - Exits the process with status 1 if the database can't be
          reached at all (no point attempting schema statements).
        - A failure to create the `vector` extension is treated as
          non-fatal: it's logged as a warning and execution continues,
          since the extension may already exist, or may need to be
          installed out-of-band by a DBA on managed Postgres providers.
        - A failure on any *other* schema statement is treated as
          fatal: it's logged and then re-raised, since it likely
          indicates a real bug in schema.sql rather than an
          environment/privilege quirk.
    """
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
    #
    # Caveat: this is a naive split -- it assumes no statement in
    # schema.sql contains a literal semicolon inside a string, comment,
    # or function body (e.g. a PL/pgSQL `$$ ... $$` block). That holds
    # for this project's current schema (plain CREATE TABLE / CREATE
    # INDEX statements), but would need a smarter SQL-aware splitter if
    # the schema ever grows to include stored procedures or triggers.
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    schema_statements = [
        statement
        for statement in statements
        if not statement.lower().startswith("create extension")
    ]

    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Handle the pgvector extension separately from the rest of
            #    the schema. Creating an extension typically requires
            #    superuser (or at least elevated) privileges, and many
            #    managed Postgres providers either pre-install it or
            #    handle it differently -- so a failure here is expected/
            #    benign in some environments and shouldn't abort setup.
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
            except Exception as e:
                console.print(
                    "[yellow]Note: Could not create extension "
                    "(it may already exist or you lack superuser privileges): "
                    f"{e}[/yellow]"
                )
                # Roll back so the failed statement doesn't poison the
                # transaction/connection state for the statements that follow.
                conn.rollback()

            # 2. Apply the remaining schema (tables, indexes, etc.). Unlike
            #    the extension step above, a failure here IS treated as
            #    fatal -- logged, then re-raised -- since it likely points
            #    to a genuine problem in the schema rather than an
            #    environment/privilege quirk.
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
