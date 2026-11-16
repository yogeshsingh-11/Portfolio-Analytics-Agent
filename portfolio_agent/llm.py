"""LLM backends. `GeminiBackend` is the real one; `MockBackend` is an offline stand-in.

Both expose the same interface so the agent, tools and evaluator are backend-agnostic:
  start(question)                      -> session with .next_step() / .submit_results()
  generate_sql(question, schema, prev_sql, error) -> str
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from .tools import EXPOSURE_TOOL, SQL_TOOL, TOOL_SPECS

log = logging.getLogger(__name__)

AGENT_SYSTEM = f"""You are a portfolio analytics assistant. Answer the user's question using tools.
- Use `{SQL_TOOL}` for questions about data in the database (counts, lists, totals, top-N, comparisons).
- Use `{EXPOSURE_TOOL}` when asked for sector exposure / sector breakdown of a specific portfolio.
- Some questions need both; call tools one after another.
- Base the answer ONLY on tool output. Never invent numbers. If a tool errors, say so plainly.
- Keep the final answer concise; show tables as short markdown tables or bullet lists."""

SQL_SYSTEM = """You translate questions into ONE read-only SQLite query.
Return ONLY the SQL, no explanation, no markdown fences. Use only the tables and columns below,
exact category spellings, and explicit JOINs. Prefer readable column aliases."""


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Step:
    calls: list[ToolCall] = field(default_factory=list)
    final: str | None = None


# --------------------------------------------------------------------------- Gemini
class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai  # imported lazily so the mock backend needs no SDK
        from google.genai import types
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example).")
        self._types = types
        # 60s per-request timeout (ms) so a stalled call fails loudly instead of hanging forever
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    def thinking(self):
        """Optional reasoning-effort cap (GEMINI_THINKING_LEVEL=minimal|low|medium|high).
        Routing and simple SQL rarely need deep reasoning, and thinking dominates latency."""
        level = os.getenv("GEMINI_THINKING_LEVEL", "").strip().upper()
        return self._types.ThinkingConfig(thinking_level=level) if level else None

    def _generate(self, contents, config, model=None):
        t0 = time.perf_counter()
        try:
            return self._generate_with_retry(contents, config, model or self.model)
        finally:
            log.info("Gemini call took %.1fs", time.perf_counter() - t0)

    def _generate_with_retry(self, contents, config, model):
        for attempt in range(4):  # retry transient rate-limit / overload errors
            try:
                return self.client.models.generate_content(
                    model=model, contents=contents, config=config)
            except Exception as exc:  # SDK raises APIError subclasses; keep this broad but narrow by message
                msg = str(exc)
                log.warning("Gemini call failed (attempt %d/4): %s", attempt + 1, msg[:300])
                if attempt < 3 and any(c in msg for c in ("429", "500", "503", "UNAVAILABLE")):
                    time.sleep(2 ** attempt)
                    continue
                if "NOT_FOUND" in msg:
                    raise RuntimeError(
                        f"Model '{model}' is unavailable for your key. Set GEMINI_MODEL in .env "
                        f"to a currently supported Gemini model. Original error: {msg}") from exc
                raise

    def start(self, question: str) -> "GeminiSession":
        return GeminiSession(self, question)

    def generate_sql(self, question, schema_context, prev_sql=None, error=None) -> str:
        t = self._types
        prompt = f"{schema_context}\n\nQUESTION: {question}"
        if error:
            prompt += f"\n\nYour previous query failed.\nPREVIOUS SQL: {prev_sql}\nERROR: {error}\nFix it."
        cfg = t.GenerateContentConfig(system_instruction=SQL_SYSTEM, temperature=0.0,
                                      thinking_config=self.thinking())
        # Optional: use a different (stronger) model just for SQL writing; routing stays on GEMINI_MODEL.
        sql_model = os.getenv("GEMINI_SQL_MODEL") or None
        return self._generate(prompt, cfg, model=sql_model).text or ""


class GeminiSession:
    def __init__(self, backend: GeminiBackend, question: str):
        t = backend._types
        self.b = backend
        self.contents = [t.Content(role="user", parts=[t.Part(text=question)])]
        decls = [t.FunctionDeclaration(**spec) for spec in TOOL_SPECS]
        self.config = t.GenerateContentConfig(
            system_instruction=AGENT_SYSTEM, temperature=0.0, thinking_config=backend.thinking(),
            tools=[t.Tool(function_declarations=decls)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        self._pending: list[ToolCall] = []

    def next_step(self) -> Step:
        resp = self.b._generate(self.contents, self.config)
        if resp.candidates and resp.candidates[0].content:
            self.contents.append(resp.candidates[0].content)
        calls = [ToolCall(c.name, dict(c.args or {})) for c in (resp.function_calls or [])]
        self._pending = calls
        return Step(calls=calls) if calls else Step(final=resp.text or "")

    def submit_results(self, results: list[dict]) -> None:
        t = self.b._types
        parts = [t.Part.from_function_response(name=c.name, response={"result": r})
                 for c, r in zip(self._pending, results)]
        self.contents.append(t.Content(role="user", parts=parts))


# --------------------------------------------------------------------------- Mock
_MOCK_SQL = [
    (r"how many portfolios", "SELECT COUNT(*) FROM portfolios"),
    (r"names of all active portfolios|active portfolios",
     "SELECT portfolio_name FROM portfolios WHERE status = 'Active'"),
    (r"securities .*technology sector",
     "SELECT s.symbol, s.company_name FROM securities s JOIN sectors x ON x.sector_id = s.sector_id "
     "WHERE x.sector_name = 'Technology'"),
    (r"total assets under management.*high",
     "SELECT SUM(total_aum) FROM portfolios WHERE target_risk_level = 'High'"),
    (r"top 5 holdings by cost basis in the growth equity fund",
     "SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis FROM holdings h "
     "JOIN securities s ON s.security_id = h.security_id JOIN portfolios p ON p.portfolio_id = h.portfolio_id "
     "WHERE p.portfolio_name = 'Growth Equity Fund' ORDER BY h.cost_basis DESC LIMIT 5"),
    (r"average current price .*each sector",
     "SELECT x.sector_name, AVG(s.current_price) AS avg_price FROM securities s "
     "JOIN sectors x ON x.sector_id = s.sector_id GROUP BY x.sector_name ORDER BY avg_price DESC"),
    (r"total value of technology sector holdings",
     "WITH v AS (SELECT h.portfolio_id, SUM(h.quantity*s.current_price) AS total, "
     "SUM(CASE WHEN x.sector_name='Technology' THEN h.quantity*s.current_price ELSE 0 END) AS tech "
     "FROM holdings h JOIN securities s ON s.security_id=h.security_id "
     "LEFT JOIN sectors x ON x.sector_id=s.sector_id GROUP BY h.portfolio_id) "
     "SELECT p.portfolio_name, v.tech AS tech_value, v.total AS total_value, "
     "ROUND(v.tech/v.total*100, 2) AS tech_percentage FROM v JOIN portfolios p USING(portfolio_id) "
     "WHERE v.tech > 0 ORDER BY tech_percentage DESC"),
    (r"more than 5 different sectors",
     "WITH sc AS (SELECT h.portfolio_id, COUNT(DISTINCT s.sector_id) AS sector_count FROM holdings h "
     "JOIN securities s ON s.security_id=h.security_id WHERE s.asset_type='Stock' "
     "GROUP BY h.portfolio_id HAVING COUNT(DISTINCT s.sector_id) > 5), "
     "hc AS (SELECT portfolio_id, COUNT(*) AS total_holdings FROM holdings GROUP BY portfolio_id) "
     "SELECT p.portfolio_name, sc.sector_count, hc.total_holdings, "
     "ROUND(CAST(sc.sector_count AS REAL)/hc.total_holdings, 3) AS diversification_ratio "
     "FROM sc JOIN hc USING(portfolio_id) JOIN portfolios p USING(portfolio_id) "
     "ORDER BY diversification_ratio DESC"),
]


class MockBackend:
    """Deterministic offline backend for smoke-testing the pipeline without an API key.

    It is a keyword router plus a small template table; it does NOT generalise.
    Use it to verify plumbing (safety, tools, evaluator), not to judge text-to-SQL quality.
    """
    name = "mock"

    def start(self, question: str) -> "MockSession":
        return MockSession(question)

    def generate_sql(self, question, schema_context, prev_sql=None, error=None) -> str:
        q = question.lower()
        for pattern, sql in _MOCK_SQL:
            if re.search(pattern, q):
                return sql
        raise ValueError("MockBackend has no template for this question; use --backend gemini.")


class MockSession:
    def __init__(self, question: str):
        self.q, self.calls, self.done = question, [], False
        self._last: list[dict] = []

    def next_step(self) -> Step:
        if self.done:
            return Step(final=_summarise(self.calls, self._last))
        self.done = True
        if re.search(r"exposure|breakdown|allocation", self.q, re.I):
            m = re.search(r"(?:for|of)\s+(?:the\s+)?(.+?)\??$", self.q.strip(), re.I)
            name = m.group(1) if m else self.q
            self.calls = [ToolCall(EXPOSURE_TOOL, {"portfolio_name": name})]
        else:
            self.calls = [ToolCall(SQL_TOOL, {"question": self.q})]
        return Step(calls=self.calls)

    def submit_results(self, results: list[dict]) -> None:
        self._last = results


def _summarise(calls: list[ToolCall], results: list[dict]) -> str:
    if not results:
        return "No tool was run."
    r = results[0]
    if not r.get("ok"):
        return f"Sorry, that failed: {r.get('error')}"
    if calls[0].name == EXPOSURE_TOOL:
        lines = [f"Sector exposure for {r['portfolio_name']} (equities only):"]
        lines += [f"- {e['sector']}: {e['exposure_pct']}%" for e in r["exposures"]]
        return "\n".join(lines)
    if r["row_count"] == 1 and len(r["columns"]) == 1:
        return f"{r['columns'][0]}: {r['rows'][0][0]}"
    return f"{r['row_count']} rows. Columns: {', '.join(r['columns'])}.\n" + json.dumps(r["rows"][:10])


def get_backend(kind: str = "auto"):
    """kind: 'gemini' | 'mock' | 'auto' (gemini if a key is present, else mock)."""
    if kind == "mock":
        return MockBackend()
    if kind == "gemini":
        return GeminiBackend()
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiBackend()
    log.warning("No GEMINI_API_KEY found; falling back to the offline MockBackend.")
    return MockBackend()
