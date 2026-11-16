import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import compare_results, reference_exposure  # noqa: E402
from portfolio_agent import PortfolioAgent  # noqa: E402
from portfolio_agent.db import build_database  # noqa: E402
from portfolio_agent.exposure import calculate_sector_exposure, resolve_portfolio  # noqa: E402
from portfolio_agent.llm import MockBackend  # noqa: E402
from portfolio_agent.sql_safety import UnsafeSQLError, execute_readonly, lock_down, validate_sql  # noqa: E402


@pytest.fixture()
def conn():
    return build_database()


# ---------------------------------------------------------------- SQL safety
@pytest.mark.parametrize("bad", [
    "DROP TABLE portfolios", "DELETE FROM holdings", "SELECT 1; DROP TABLE x",
    "UPDATE portfolios SET status='x'", "PRAGMA table_info(portfolios)",
    "INSERT INTO sectors VALUES (99,'x','y','z')", "ATTACH DATABASE 'x.db' AS x", "",
])
def test_validator_rejects_unsafe(bad):
    with pytest.raises(UnsafeSQLError):
        validate_sql(bad)


def test_validator_accepts_fenced_cte_and_literals():
    assert validate_sql("```sql\nWITH a AS (SELECT 1) SELECT * FROM a;\n```").startswith("WITH")
    # forbidden words inside string literals are fine
    validate_sql("SELECT * FROM portfolios WHERE portfolio_name = 'Drop; Delete Fund'")


def test_authorizer_blocks_writes_even_if_validator_bypassed(conn):
    lock_down(conn)
    for stmt in ("DELETE FROM portfolios", "DROP TABLE portfolios", "CREATE TABLE t(x)"):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(stmt)
    assert conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0] == 13


def test_sqlite_master_unreadable_after_lock(conn):
    lock_down(conn)
    with pytest.raises(sqlite3.DatabaseError):
        execute_readonly(conn, "SELECT sql FROM sqlite_master")


def test_row_cap_and_timeout(conn):
    lock_down(conn)
    res = execute_readonly(conn, "SELECT * FROM historical_prices", max_rows=10)
    assert len(res.rows) == 10 and res.truncated
    with pytest.raises(TimeoutError):
        execute_readonly(
            conn, "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r",
            timeout_s=0.2)


# ---------------------------------------------------------------- exposure
def test_exposure_excludes_bonds_and_sums_to_100(conn):
    res = calculate_sector_exposure(conn, "Balanced Portfolio")  # mixed stock/bond
    assert res["ok"] and res["equity_weight_pct"] == pytest.approx(36.0)
    assert sum(e["exposure_pct"] for e in res["exposures"]) == pytest.approx(100, abs=0.05)
    assert all("UST" not in h and "CORP" not in h for e in res["exposures"] for h in e["holdings"])


def test_exposure_matches_independent_reference(conn):
    for name in ("Tech Innovation Fund", "International Equity Fund", "Total Stock Market Index Fund"):
        got = {e["sector"]: e["exposure_pct"] for e in calculate_sector_exposure(conn, name)["exposures"]}
        assert got == pytest.approx(reference_exposure(name), abs=0.011)


def test_bond_only_portfolio_is_handled(conn):
    res = calculate_sector_exposure(conn, "Fixed Income Plus")
    assert res["ok"] and res["exposures"] == [] and "no equities" in res["note"]


def test_name_resolution(conn):
    assert resolve_portfolio(conn, "international equity")[0]["portfolio_name"] == "International Equity Fund"
    assert resolve_portfolio(conn, "tech innovation fund")[0]["portfolio_id"] == 3
    assert resolve_portfolio(conn, "3")[0]["portfolio_name"] == "Tech Innovation Fund"
    p, cands = resolve_portfolio(conn, "international")  # ambiguous
    assert p is None and len(cands) >= 2
    assert resolve_portfolio(conn, "zzzz")[0] is None
    assert calculate_sector_exposure(conn, "zzzz")["ok"] is False


# ---------------------------------------------------------------- evaluator logic must be able to FAIL
def test_compare_detects_wrong_values_and_counts():
    assert compare_results([(1,), (2,)], [(2,), (1,)], ordered=False)[0]
    assert not compare_results([(1,), (2,)], [(2,), (1,)], ordered=True)[0]
    assert not compare_results([(13,)], [(12,)], ordered=False)[0]
    assert not compare_results([(1,), (2,)], [(1,)], ordered=False)[0]
    assert compare_results([("a", 1)], [("a", 1, "extra")], ordered=False)[0]
    assert not compare_results([("a", 1)], [("b", 1, "extra")], ordered=False)[0]


# ---------------------------------------------------------------- agent behaviour
def test_agent_end_to_end_and_error_paths():
    agent = PortfolioAgent(backend=MockBackend())
    assert agent.answer_question("How many portfolios do we have in total?").traces[0].result["rows"] == [[13]]
    assert agent.answer_question("").error == "empty question"
    r = agent.answer_question("Show me the weather")  # mock has no template -> graceful failure
    assert r.traces[0].result["ok"] is False and "Sorry" in r.answer


def test_gemini_wiring_builds_offline():
    """Tool declarations must be valid SDK objects (no network, no key needed)."""
    from google.genai import types
    from portfolio_agent.tools import TOOL_SPECS
    decls = [types.FunctionDeclaration(**s) for s in TOOL_SPECS]
    assert [d.name for d in decls] == ["sql_query", "exposure_calculator"]


# ---------------------------------------------------------------- SQL self-correction + Gemini loop (stubbed)
class _FlakySQLBackend(MockBackend):
    """First SQL attempt is invalid (wrong column) and the retry must receive the error."""
    def __init__(self):
        self.seen_errors = []

    def generate_sql(self, question, schema_context, prev_sql=None, error=None):
        self.seen_errors.append(error)
        return "SELECT COUNT(*) FROM portfolios" if error else "SELECT nonexistent_col FROM portfolios"


def test_sql_tool_retries_with_error_feedback():
    backend = _FlakySQLBackend()
    agent = PortfolioAgent(backend=backend)
    res = agent.answer_question("How many portfolios do we have?").traces[0].result
    assert res["ok"] and res["attempts"] == 2 and res["rows"] == [[13]]
    assert backend.seen_errors[0] is None and "nonexistent_col" in backend.seen_errors[1]


def test_unsafe_generated_sql_is_refused_not_executed():
    class Evil(MockBackend):
        def generate_sql(self, *a, **k):
            return "DROP TABLE portfolios"
    agent = PortfolioAgent(backend=Evil())
    res = agent.answer_question("how many portfolios?").traces[0].result
    assert res["ok"] is False
    assert agent.conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0] == 13


def test_gemini_session_function_calling_loop():
    """Drive GeminiSession with a stubbed model to check message plumbing without network."""
    from google.genai import types
    from portfolio_agent.llm import GeminiBackend

    be = GeminiBackend.__new__(GeminiBackend)  # skip __init__ (needs API key)
    be._types, be.model = types, "stub"
    fc = types.FunctionCall(name="exposure_calculator", args={"portfolio_name": "Tech Innovation Fund"})
    replies = iter([
        types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(function_call=fc)]))]),
        types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text="Tech is 100%.")]))]),
    ])
    be._generate = lambda contents, config: next(replies)

    agent = PortfolioAgent(backend=be)
    r = agent.answer_question("sector exposures for the Tech Innovation Fund?")
    assert r.tools_used == ["exposure_calculator"] and r.answer == "Tech is 100%." and r.error is None
