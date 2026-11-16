"""The portfolio agent: a small tool-calling loop over the SQL and exposure tools."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .db import ROOT, build_database
from .llm import get_backend
from .sql_safety import lock_down
from .tools import EXPOSURE_TOOL, SQL_TOOL, ExposureTool, SQLTool

log = logging.getLogger(__name__)
MAX_ROWS_TO_LLM = 50


@dataclass
class ToolTrace:
    name: str
    args: dict
    result: dict


@dataclass
class AgentResponse:
    question: str
    answer: str
    traces: list[ToolTrace] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [t.name for t in self.traces]


def _shrink(result: dict) -> dict:
    """Cap what is sent back to the model so large results don't blow the context."""
    if "rows" in result and len(result["rows"]) > MAX_ROWS_TO_LLM:
        result = {**result, "rows": result["rows"][:MAX_ROWS_TO_LLM], "rows_shown_to_model": MAX_ROWS_TO_LLM}
    return result


class PortfolioAgent:
    def __init__(self, backend=None, conn=None, max_steps: int = 4):
        load_dotenv(ROOT / ".env")
        self.backend = backend or get_backend("auto")
        self.conn = conn or build_database()
        # Schema introspection reads sqlite_master, so build the tools first...
        self.sql_tool = SQLTool(self.conn, self.backend)
        self.exposure_tool = ExposureTool(self.conn)
        lock_down(self.conn)  # ...then make the connection read-only for everything after.
        self.max_steps = max_steps

    def _dispatch(self, name: str, args: dict) -> dict:
        try:
            if name == SQL_TOOL:
                return self.sql_tool.run(str(args.get("question", "")))
            if name == EXPOSURE_TOOL:
                return self.exposure_tool.run(str(args.get("portfolio_name", "")))
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        except Exception as exc:  # a tool failure must never crash the agent loop
            log.exception("Tool %s crashed", name)
            return {"ok": False, "error": f"Tool error: {exc}"}

    def answer_question(self, question: str) -> AgentResponse:
        start = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return AgentResponse(question, "Please ask a question.", error="empty question")
        if len(question) > 2000:
            return AgentResponse(question, "Question too long (max 2000 characters).", error="too long")

        resp = AgentResponse(question, "")
        try:
            session = self.backend.start(question)
            for _ in range(self.max_steps):
                step = session.next_step()
                if step.final is not None:
                    resp.answer = step.final
                    break
                results = []
                for call in step.calls:
                    log.info("Tool call: %s %s", call.name, json.dumps(call.args))
                    result = self._dispatch(call.name, call.args)
                    resp.traces.append(ToolTrace(call.name, call.args, result))
                    results.append(_shrink(result))
                session.submit_results(results)
            else:
                resp.error = "max steps reached"
                resp.answer = "I couldn't finish answering within the step limit."
        except Exception as exc:
            log.exception("Agent failure")
            resp.error = str(exc)
            resp.answer = f"Sorry, something went wrong: {exc}"
        resp.latency_s = round(time.perf_counter() - start, 2)
        return resp
