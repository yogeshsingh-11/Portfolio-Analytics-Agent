"""Builds the in-memory SQLite database from database_schema.sql and the CSV files."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCHEMA_FILE = ROOT / "database_schema.sql"

# Parents before children so foreign keys are satisfied.
LOAD_ORDER = [
    "sectors", "securities", "benchmarks", "portfolios", "holdings",
    "transactions", "historical_prices", "portfolio_performance", "risk_metrics",
]


def build_database(path: str = ":memory:") -> sqlite3.Connection:
    """Create the schema and load every CSV. Returns an open connection."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_FILE.read_text())
    for table in LOAD_ORDER:
        csv_path = DATA_DIR / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing data file: {csv_path}")
        pd.read_csv(csv_path).to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    return conn


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [r[0] for r in rows]
