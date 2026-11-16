# Portfolio Analytics Agent

A command-line AI agent that answers portfolio questions in natural language. A Gemini model
decides between two tools: a **text-to-SQL tool** for questions about the data, and a
**sector exposure calculator** for equity sector breakdowns. An evaluator scores the agent
against `ground_truth_dataset.json`.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env              # then set GEMINI_API_KEY (https://aistudio.google.com/apikey)

python cli.py                     # interactive
python cli.py -v -q "What is the total AUM for portfolios with high target risk level?"
python evaluate.py --backend gemini -v --json results.json
```

`-v` prints each tool call, the generated SQL and latency. `evaluate.py` exits with code 1 if any
question fails.

Optional `.env` settings:

| Variable | Purpose |
|---|---|
| `GEMINI_MODEL` | Model for routing and answers (default `gemini-3.1-flash-lite`). Must support function calling. |
| `GEMINI_SQL_MODEL` | Optional separate model used only to write SQL. |
| `GEMINI_THINKING_LEVEL` | `minimal`/`low`/`medium`/`high`. A low level cuts latency a lot. |

Without an API key the agent falls back to an offline `MockBackend` (a keyword router plus a few
SQL templates). It exists only to test the plumbing and is **not** a text-to-SQL system.

## How it works

```
question
   │
   ▼
PortfolioAgent: tool-calling loop (Gemini function calling, max 4 steps)
   ├─► sql_query(question) ─► Gemini writes SQL from schema + live enum values + business rules
   │        └─► validate ─► read-only execution (authorizer, timeout, row cap)
   │        └─► on error, the error is fed back to the model; up to 3 attempts
   └─► exposure_calculator(portfolio_name) ─► resolve name ─► join holdings/securities/sectors ─► sector %
```

The final answer is written from tool output only. Tool crashes are caught so the agent loop never
dies; the user gets a plain error message instead.

### Sector exposure

Uses `holdings.current_weight` joined to `securities.sector_id`, **equities only** (bonds are
excluded). For each sector it returns:

- `exposure_pct`: share of the **equity sleeve** (sums to 100%). This is the headline figure.
- `portfolio_weight_pct`: share of the **whole portfolio**.
- the holdings in that sector.

The two differ for mixed portfolios (e.g. Balanced Portfolio is 36% equities). Normalising also
handles portfolios whose weights don't sum to 1 (Total Stock Market sums to 0.98). Bond-only
portfolios return an empty result with an explanatory note.

Portfolio references are resolved forgivingly: id, exact name (case-insensitive), a unique
token match (`international equity` → International Equity Fund), then close spelling. An ambiguous
reference such as `international` is rejected with the candidates listed rather than guessed.

### SQL safety

LLM-written SQL is untrusted, so there are four independent layers:

1. **Static validation:** one `SELECT`/`WITH` statement only; write/DDL keywords rejected
   (string literals and comments are stripped first, so `'Drop Fund'` is fine).
2. **SQLite authorizer:** denies everything except reads, so even a validator bypass cannot write.
   `sqlite_master` is unreadable once the connection is locked.
3. **`PRAGMA query_only`** on the connection.
4. **Resource limits:** 5-second timeout and a 200-row cap. Only the first 50 rows go back to the
   model.

### Prompt context for text-to-SQL

Besides the DDL, the prompt includes the live distinct values for status, risk level, strategy,
asset type, sector names and portfolio names, plus business rules (market value =
`quantity × current_price`; weights are fractions; bonds have no sector). This prevents common
misses such as `'high'` vs `'High'`.

## Evaluation

`evaluate.py` measures **execution accuracy**, not SQL string similarity:

- **text2sql:** checks that the SQL tool was chosen, runs the agent's SQL and the reference SQL
  and compares result sets (order-insensitive unless the reference uses `ORDER BY`; floats
  rounded to 2 dp; extra columns tolerated and reported).
- **exposure:** checks routing, the resolved portfolio, and the numbers against an independent
  pandas computation from the raw CSVs.

It prints per-question PASS/FAIL, overall and routing accuracy, a breakdown by type and
difficulty, and optionally a JSON report with every generated SQL and answer.

### Results and known limitations

- Last live run on `gemini-3.1-flash-lite`: **6/10** (easy 3/3, exposure 2/2; failures were in
  the medium/hard SQL questions). Larger models are slower or heavily queued on free-tier keys,
  so `GEMINI_SQL_MODEL` lets you use a stronger model for SQL alone.
- Hard questions (multi-step CTEs, "diversification metrics") are open to interpretation. A
  correct-looking query can differ from the reference in definition rather than in quality.
- Not built (optional in the brief): web UI, REST API, caching, hybrid-question test cases. The
  agent loop already supports chaining both tools in one answer.

## Project layout

```
cli.py                      interactive / one-shot agent
evaluate.py                 evaluator against ground_truth_dataset.json
portfolio_agent/
  agent.py                  agent loop, error containment, traces
  tools.py                  SQL tool (with retry), exposure tool, tool specs, schema context
  exposure.py               exposure maths, fuzzy portfolio resolution
  sql_safety.py             validation, authorizer, timeout, row cap
  llm.py                    Gemini backend (+ offline mock) behind one interface
  db.py                     builds the in-memory SQLite DB from schema + CSVs
tests/test_agent.py         22 offline tests (safety, exposure, evaluator logic, agent loop)
data/, database_schema.sql  provided data
```

Run the tests with `python -m pytest -q tests` (no API key needed).
