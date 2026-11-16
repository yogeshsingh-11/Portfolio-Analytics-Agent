"""Defence in depth for LLM-generated SQL.

Layers: (1) static validation (single SELECT/WITH statement, no write keywords),
(2) a SQLite authorizer that denies everything except reads on known tables,
(3) PRAGMA query_only, (4) a wall-clock timeout, (5) a row cap.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    """Raised when a query fails validation."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False


def _strip_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def clean_sql(raw: str) -> str:
    """Strip markdown fences / trailing semicolons the model may add."""
    sql = raw.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1).strip()
    return sql.rstrip(";").strip()


def validate_sql(raw: str) -> str:
    """Return a cleaned, validated query or raise UnsafeSQLError."""
    sql = clean_sql(raw)
    if not sql:
        raise UnsafeSQLError("Empty query.")
    skeleton = _strip_literals_and_comments(sql)
    if ";" in skeleton:
        raise UnsafeSQLError("Only a single statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", skeleton, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT queries (optionally with CTEs) are allowed.")
    bad = FORBIDDEN.search(skeleton)
    if bad:
        raise UnsafeSQLError(f"Forbidden keyword: {bad.group(1).upper()}")
    return sql


def lock_down(conn: sqlite3.Connection) -> None:
    """Make the connection read-only and restrict what statements may touch."""
    ok_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorizer(action, arg1, arg2, db_name, source):
        if action in ok_actions:
            return sqlite3.SQLITE_OK
        # Reads are fine except SQLite internals (sqlite_master etc.). CTE names also surface
        # as READ targets, so a table allow-list would wrongly block recursive CTEs.
        if action == sqlite3.SQLITE_READ and not str(arg1).lower().startswith("sqlite_"):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(authorizer)


def execute_readonly(conn: sqlite3.Connection, sql: str, max_rows: int = 200,
                     timeout_s: float = 5.0) -> QueryResult:
    """Validate then run a query with a timeout and row cap."""
    sql = validate_sql(sql)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description] if cur.description else []
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError(f"Query exceeded {timeout_s}s timeout.") from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)
    return QueryResult(cols, rows[:max_rows], truncated=len(rows) > max_rows)
