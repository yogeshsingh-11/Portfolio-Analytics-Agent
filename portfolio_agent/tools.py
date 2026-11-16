"""Agent tools: the text-to-SQL tool and the sector exposure tool."""
from __future__ import annotations

import logging
import sqlite3

from .exposure import calculate_sector_exposure
from .sql_safety import UnsafeSQLError, execute_readonly

log = logging.getLogger(__name__)

EXPOSURE_TOOL = "exposure_calculator"
SQL_TOOL = "sql_query"


def build_schema_context(conn: sqlite3.Connection) -> str:
    """Schema DDL plus live categorical values and business rules.

    Giving the model the real enum values and entity names (a light form of
    metadata retrieval) prevents the classic 'high' vs 'High' / wrong-name misses.
    """
    ddl = "\n".join(
        r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    )

    def distinct(sql: str) -> str:
        return ", ".join(repr(r[0]) for r in conn.execute(sql) if r[0] is not None)

    return f"""DATABASE: SQLite. Tables:
{ddl}

CATEGORICAL VALUES (use exact spelling/case):
- portfolios.status: {distinct("SELECT DISTINCT status FROM portfolios")}
- portfolios.target_risk_level: {distinct("SELECT DISTINCT target_risk_level FROM portfolios")}
- portfolios.strategy_type: {distinct("SELECT DISTINCT strategy_type FROM portfolios")}
- securities.asset_type: {distinct("SELECT DISTINCT asset_type FROM securities")}
- sectors.sector_name: {distinct("SELECT sector_name FROM sectors")}
- portfolios.portfolio_name: {distinct("SELECT portfolio_name FROM portfolios")}

BUSINESS RULES:
- Market value of a holding = holdings.quantity * securities.current_price.
- Bonds (asset_type='Bond') have NULL sector_id; sector questions concern Stocks.
- holdings.current_weight is a fraction (0.18 = 18%).
- Percentages should be returned as numbers on a 0-100 scale, rounded to 2 decimals.
"""


class SQLTool:
    """Natural language -> validated, read-only SQL -> rows."""

    def __init__(self, conn: sqlite3.Connection, llm, max_attempts: int = 3, max_rows: int = 200):
        self.conn, self.llm = conn, llm
        self.max_attempts, self.max_rows = max_attempts, max_rows
        self.schema_context = build_schema_context(conn)

    def run(self, question: str) -> dict:
        error, prev_sql = None, None
        for attempt in range(1, self.max_attempts + 1):
            try:
                sql = self.llm.generate_sql(question, self.schema_context, prev_sql, error)
                res = execute_readonly(self.conn, sql, max_rows=self.max_rows)
                log.info("SQL ok (attempt %d): %s", attempt, sql)
                return {"ok": True, "sql": sql, "columns": res.columns,
                        "rows": [list(r) for r in res.rows], "row_count": len(res.rows),
                        "truncated": res.truncated, "attempts": attempt}
            except (UnsafeSQLError, sqlite3.Error, TimeoutError) as exc:
                prev_sql, error = locals().get("sql", prev_sql), str(exc)
                log.warning("SQL attempt %d failed: %s", attempt, error)
        return {"ok": False, "error": f"Could not produce a valid query: {error}",
                "sql": prev_sql, "attempts": self.max_attempts}


class ExposureTool:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def run(self, portfolio_name: str) -> dict:
        return calculate_sector_exposure(self.conn, portfolio_name)


TOOL_SPECS = [
    {
        "name": SQL_TOOL,
        "description": (
            "Answer a question by querying the portfolio database (counts, lists, filters, "
            "aggregations, top-N, joins across portfolios/holdings/securities/sectors/"
            "transactions/performance/risk). Pass the user's question verbatim, without rephrasing or adding column names."
        ),
        "parameters": {"type": "OBJECT", "properties": {
            "question": {"type": "STRING", "description": "The question to answer with SQL."}},
            "required": ["question"]},
    },
    {
        "name": EXPOSURE_TOOL,
        "description": (
            "Calculate sector exposure percentages for ONE portfolio from its equity holdings "
            "(bonds ignored). Use for 'sector exposure / breakdown / allocation of <portfolio>'."
        ),
        "parameters": {"type": "OBJECT", "properties": {
            "portfolio_name": {"type": "STRING",
                               "description": "Portfolio name (or id), e.g. 'Tech Innovation Fund'."}},
            "required": ["portfolio_name"]},
    },
]
