"""Sector exposure calculator (equities only) plus fuzzy portfolio-name resolution."""
from __future__ import annotations

import difflib
import re
import sqlite3
from collections import defaultdict

_STOP = {"fund", "portfolio", "the", "a", "of", "for", "my", "our"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9&]+", text.lower()) if t not in _STOP}


def resolve_portfolio(conn: sqlite3.Connection, name_or_id) -> tuple[dict | None, list[str]]:
    """Map a user/LLM-supplied reference to a portfolio row.

    Returns (portfolio, candidates). portfolio is None when nothing matches or the
    reference is ambiguous; candidates then lists the possible names.
    """
    rows = conn.execute("SELECT portfolio_id, portfolio_name FROM portfolios").fetchall()
    names = {r[1]: r[0] for r in rows}
    ref = str(name_or_id).strip()

    if ref.isdigit() and int(ref) in names.values():
        pid = int(ref)
        return {"portfolio_id": pid, "portfolio_name": next(n for n, i in names.items() if i == pid)}, []
    for n, pid in names.items():
        if n.lower() == ref.lower():
            return {"portfolio_id": pid, "portfolio_name": n}, []

    q = _tokens(ref)
    if q:
        full = [n for n in names if q <= _tokens(n)]
        if len(full) == 1:
            return {"portfolio_id": names[full[0]], "portfolio_name": full[0]}, []
        if len(full) > 1:
            return None, sorted(full)

    close = difflib.get_close_matches(ref, list(names), n=3, cutoff=0.6)
    if len(close) == 1:
        return {"portfolio_id": names[close[0]], "portfolio_name": close[0]}, []
    return None, close


def calculate_sector_exposure(conn: sqlite3.Connection, portfolio: str | int) -> dict:
    """Sector exposure of a portfolio's equity holdings.

    Uses holdings.current_weight joined to securities.sector_id. Bonds are ignored.
    Two views are returned per sector:
      * exposure_pct: share of the *equity sleeve* (sums to 100)
      * portfolio_weight_pct: share of the *whole portfolio* (sums to equity_weight_pct)
    """
    resolved, candidates = resolve_portfolio(conn, portfolio)
    if resolved is None:
        hint = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
        return {"ok": False, "error": f"Could not uniquely identify portfolio '{portfolio}'.{hint}"}

    rows = conn.execute(
        """SELECT COALESCE(sec.sector_name, 'Unclassified'), s.symbol, h.current_weight
           FROM holdings h
           JOIN securities s ON s.security_id = h.security_id
           LEFT JOIN sectors sec ON sec.sector_id = s.sector_id
           WHERE h.portfolio_id = ? AND s.asset_type = 'Stock'""",
        (resolved["portfolio_id"],),
    ).fetchall()

    if not rows:
        return {"ok": True, **resolved, "equity_weight_pct": 0.0, "exposures": [],
                "note": "This portfolio holds no equities (bonds are excluded)."}

    weights: dict[str, float] = defaultdict(float)
    symbols: dict[str, list[str]] = defaultdict(list)
    for sector, symbol, w in rows:
        weights[sector] += w or 0.0
        symbols[sector].append(symbol)
    equity_total = sum(weights.values())
    if equity_total <= 0:
        return {"ok": False, "error": "Equity holdings have zero total weight."}

    exposures = [
        {"sector": s, "exposure_pct": round(w / equity_total * 100, 2),
         "portfolio_weight_pct": round(w * 100, 2), "holdings": sorted(symbols[s])}
        for s, w in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {"ok": True, **resolved, "equity_weight_pct": round(equity_total * 100, 2),
            "exposures": exposures,
            "note": "exposure_pct is normalised to the equity sleeve; bonds are excluded."}
