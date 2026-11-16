#!/usr/bin/env python3
"""Command-line portfolio agent.

  python cli.py                         # interactive
  python cli.py -q "How many portfolios do we have?"
  python cli.py --backend mock -v       # offline mode, show tool calls and SQL
"""
import argparse
import json
import logging

from dotenv import load_dotenv

from portfolio_agent import PortfolioAgent
from portfolio_agent.db import ROOT
from portfolio_agent.llm import get_backend


def show(resp, verbose: bool) -> None:
    if verbose:
        for t in resp.traces:
            print(f"  [tool] {t.name}({json.dumps(t.args)})")
            if t.result.get("sql"):
                print(f"  [sql]  {t.result['sql']}")
            if not t.result.get("ok"):
                print(f"  [err]  {t.result.get('error')}")
    print(f"\n{resp.answer}\n" + (f"({resp.latency_s}s)" if verbose else ""))


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--question", help="Ask one question and exit.")
    ap.add_argument("--backend", choices=["auto", "gemini", "mock"], default="auto")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show tool calls, SQL and latency.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    agent = PortfolioAgent(backend=get_backend(args.backend))
    print(f"Backend: {agent.backend.name}")
    if args.question:
        show(agent.answer_question(args.question), args.verbose)
        return
    print("Ask about the portfolios (blank line or 'exit' to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"", "exit", "quit"}:
            break
        show(agent.answer_question(q), args.verbose)


if __name__ == "__main__":
    main()
