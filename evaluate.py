#!/usr/bin/env python3
"""Evaluate the agent against ground_truth_dataset.json.

text2sql questions   -> routing check + *execution accuracy*: run the agent's SQL and the
                        reference SQL and compare result sets (not SQL text).
exposure questions   -> routing check + parameter resolution + numbers vs an independent
                        pandas computation straight from the CSVs.

  python evaluate.py --backend gemini
  python evaluate.py --backend mock --json results.json
Exit code is 1 if any question fails.
"""
import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

from portfolio_agent import PortfolioAgent
from portfolio_agent.db import DATA_DIR, ROOT, build_database
from portfolio_agent.llm import get_backend
from portfolio_agent.tools import EXPOSURE_TOOL, SQL_TOOL

GT_PATH = ROOT / "ground_truth_dataset.json"
TOL = 0.011  # numeric tolerance (values are rounded to 2 dp)


# ----------------------------------------------------------------------- SQL comparison
def _norm(v):
    return round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _norm_rows(rows):
    return [tuple(_norm(v) for v in r) for r in rows]


def _key(row):
    return tuple((0, v) if isinstance(v, (int, float)) else (1, str(v)) for v in row)


def compare_results(expected_rows, actual_rows, ordered: bool) -> tuple[bool, str]:
    exp, act = _norm_rows(expected_rows), _norm_rows(actual_rows)
    if not ordered:
        exp, act = sorted(exp, key=_key), sorted(act, key=_key)
    if exp == act:
        return True, "exact match"
    if len(exp) != len(act):
        return False, f"row count {len(act)} != expected {len(exp)}"
    if not exp:
        return True, "both empty"
    # Tolerate extra columns: find, for every expected column, an actual column holding the same values.
    n_exp, n_act = len(exp[0]), len(act[0]) if act else 0
    if n_act > n_exp:
        mapping, used = [], set()
        for i in range(n_exp):
            col = [r[i] for r in exp]
            match = next((j for j in range(n_act) if j not in used and [r[j] for r in act] == col), None)
            if match is None:
                return False, "values differ"
            mapping.append(match); used.add(match)
        return True, f"match with {n_act - n_exp} extra column(s)"
    return False, "values differ"


# ----------------------------------------------------------------- exposure reference calc
def reference_exposure(portfolio_name: str) -> dict:
    """Independent implementation (pandas on raw CSVs) used to check the tool's numbers."""
    p = pd.read_csv(DATA_DIR / "portfolios.csv")
    h = pd.read_csv(DATA_DIR / "holdings.csv")
    s = pd.read_csv(DATA_DIR / "securities.csv")
    sec = pd.read_csv(DATA_DIR / "sectors.csv")
    pid = int(p.loc[p.portfolio_name == portfolio_name, "portfolio_id"].iloc[0])
    df = (h[h.portfolio_id == pid].merge(s, on="security_id")
          .query("asset_type == 'Stock'").merge(sec, on="sector_id", how="left"))
    g = df.groupby("sector_name").current_weight.sum()
    return (g / g.sum() * 100).round(2).to_dict()


# ------------------------------------------------------------------------------ runners
def eval_sql(q, resp, ref_conn):
    gt = q["ground_truth"]
    if SQL_TOOL not in resp.tools_used:
        return False, False, f"wrong routing: {resp.tools_used or 'no tool'}"
    trace = next(t for t in resp.traces if t.name == SQL_TOOL)
    if not trace.result.get("ok"):
        return True, False, f"SQL failed: {trace.result.get('error')}"
    cur = ref_conn.execute(gt["sql_query"].rstrip(";"))
    expected = cur.fetchall()
    ordered = bool(re.search(r"\border\s+by\b", gt["sql_query"], re.I))
    ok, why = compare_results(expected, [tuple(r) for r in trace.result["rows"]], ordered)
    return True, ok, why


def eval_exposure(q, resp):
    gt = q["ground_truth"]
    if gt["tool_name"] not in resp.tools_used:
        return False, False, f"wrong routing: {resp.tools_used or 'no tool'}"
    trace = next(t for t in resp.traces if t.name == EXPOSURE_TOOL)
    res = trace.result
    if not res.get("ok"):
        return True, False, f"tool error: {res.get('error')}"
    want_name = gt["parameters"]["portfolio_name"]
    if res["portfolio_name"] != want_name:
        return True, False, f"resolved '{res['portfolio_name']}', expected '{want_name}'"
    got = {e["sector"]: e["exposure_pct"] for e in res["exposures"]}
    want = reference_exposure(want_name)
    if got.keys() != want.keys():
        return True, False, f"sector set differs: {sorted(got)} vs {sorted(want)}"
    bad = {k: (got[k], want[k]) for k in want if abs(got[k] - want[k]) > TOL}
    return True, not bad, "numbers match reference" if not bad else f"mismatch {bad}"


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "gemini", "mock"], default="auto")
    ap.add_argument("--json", help="Write detailed results to this file.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)

    questions = json.loads(GT_PATH.read_text())["questions"]
    backend = get_backend(args.backend)
    agent = PortfolioAgent(backend=backend)
    ref_conn = build_database()  # separate, unrestricted connection for reference queries
    print(f"Backend: {backend.name} | {len(questions)} questions\n")

    results = []
    for q in questions:
        resp = agent.answer_question(q["question"])
        if q["type"] == "text2sql":
            routed, correct, why = eval_sql(q, resp, ref_conn)
        else:
            routed, correct, why = eval_exposure(q, resp)
        results.append({"id": q["id"], "type": q["type"], "difficulty": q["difficulty"],
                        "routed": routed, "correct": correct, "detail": why,
                        "latency_s": resp.latency_s, "tools": resp.tools_used,
                        "sql": next((t.result.get("sql") for t in resp.traces if t.result.get("sql")), None),
                        "answer": resp.answer})
        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] Q{q['id']:<2} {q['type']:<20} {q['difficulty']:<6} {resp.latency_s:>5}s  {why}")
        if not correct and args.verbose:
            print(f"        sql: {results[-1]['sql']}")

    n = len(results)
    passed = sum(r["correct"] for r in results)
    print(f"\nOverall: {passed}/{n} correct | routing {sum(r['routed'] for r in results)}/{n}")
    by = defaultdict(lambda: [0, 0])
    for r in results:
        for k in (r["type"], r["difficulty"]):
            by[k][0] += r["correct"]; by[k][1] += 1
    print("Breakdown: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by.items())))
    if args.json:
        (ROOT / args.json).write_text(json.dumps({"backend": backend.name, "passed": passed,
                                                    "total": n, "results": results}, indent=2, default=str))
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
