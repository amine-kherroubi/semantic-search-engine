"""
Database connection utilities.
Uses SQLAlchemy for ORM access and psycopg2 for raw queries.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


def _db_config() -> dict[str, str | int]:
    """Return database settings from the environment."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "semantic_search"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }


def _sqlalchemy_url() -> URL:
    """Return a SQLAlchemy URL with credentials escaped correctly."""
    config = _db_config()
    return URL.create(
        drivername="postgresql+psycopg2",
        username=str(config["username"]),
        password=str(config["password"]),
        host=str(config["host"]),
        port=int(config["port"]),
        database=str(config["database"]),
    )


# SQLAlchemy engine reused across the app.
engine = create_engine(_sqlalchemy_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_raw_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a raw psycopg2 connection for SQL scripts and bulk operations."""
    config = _db_config()
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["database"],
        user=config["username"],
        password=config["password"],
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """Return True if the database is reachable."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        return True
    except Exception as exc:
        print(f"[DB] Connection failed: {exc}")
        return False


def get_autocommit_connection() -> psycopg2.extensions.connection:
    """
    Return a raw psycopg2 connection with autocommit enabled.

    Use this for statements that must run outside a transaction block, such as
    VACUUM, CREATE DATABASE, or CREATE INDEX CONCURRENTLY.  The caller is
    responsible for closing the connection (use a try/finally block).

    Unlike get_raw_connection(), this is not a context manager: autocommit
    connections have no meaningful commit/rollback semantics.
    """
    config = _db_config()
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["database"],
        user=config["username"],
        password=config["password"],
    )
    conn.autocommit = True
    return conn
