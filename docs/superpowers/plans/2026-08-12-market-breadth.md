# Market Breadth Implementation Plan

> **For agentic workers:** Execute this plan inline with test-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add S&P 500 constituent breadth, 11-sector breadth, rule-based Market Health and divergence to the existing static report without affecting drawdown safety or rules.

**Architecture:** A low-frequency updater validates and atomically replaces a local constituent reference CSV. The daily report reads only that local CSV, batch-downloads close prices for constituents and sector ETFs, derives a JSON-compatible `market_breadth` object, and sends a compact text version to DeepSeek. Rendering remains defensive: reports without the new object continue to render.

**Tech Stack:** Python 3.11, csv/urllib standard library, yfinance batch download, PyYAML, Jinja2, pytest, GitHub Actions.

## Global Constraints

- Do not change ATH, drawdown thresholds, state transitions, confirmation flow, pool amounts, or GitHub Pages deployment.
- Daily generation must never retrieve State Street holdings; it reads `data/reference/sp500_constituents.csv` only.
- A breadth failure is warning/degraded data only and must never change `drawdown_market_valid` or prevent either drawdown index from updating.
- Constituent returns and sector returns must use the S&P 500 report market date and that date's preceding valid close; no stale substitution is permitted.
- Market Health is rule-derived; it is an observation indicator, not a trading rule or an AI conclusion.
- Maintain RSS candidate-only, URL-mapping, and no-unfounded-causality rules for DeepSeek.
- Do not commit or push during this task.

---

### Task 1: Constituent reference reader and safe updater

**Files:**
- Create: `config/market_breadth.yaml`, `src/constituents.py`, `src/update_sp500_constituents.py`, `tests/test_constituents.py`, `data/reference/sp500_constituents.csv`, `.github/workflows/update-sp500-constituents.yml`

**Interfaces:**
- `normalize_yahoo_ticker(source_ticker: str) -> str`
- `load_constituents(path: Path) -> list[dict]`
- `validate_constituents(rows: list[dict], minimum_count: int = 480) -> list[dict]`
- `update_reference(reference_path: Path, fetcher: Callable) -> bool`

- [ ] Write failing tests for dot-to-dash ticker normalization, complete CSV reading, duplicate Yahoo ticker rejection, invalid constituent filtering, and a failed update preserving the prior file bytes.
- [ ] Run `python -m pytest -v tests/test_constituents.py`; confirm failures are absent-interface failures.
- [ ] Implement only reader, validator, State Street parser boundary, temporary-file write, and atomic `replace` after all validation passes.
- [ ] Run `python -m pytest -v tests/test_constituents.py` to green.
- [ ] Add weekly Monday 00:30 UTC/manual workflow that updates reference, runs constituent tests, and commits only a changed CSV without deploying Pages.

### Task 2: S&P 500 daily stock breadth

**Files:**
- Create: `src/market_breadth.py`, `tests/test_market_breadth.py`

**Interfaces:**
- `calculate_stock_breadth(constituents, close_rows_by_ticker, target_market_date, config) -> dict`
- `fetch_batched_close_rows(tickers, start_date, batch_size, downloader) -> dict[str, list[dict]]`

- [ ] Write failing tests for six advances/three declines/one unchanged, valid-count denominators, exact target-date alignment, stale ticker exclusion, and `ok`/`partial`/`invalid` coverage boundaries.
- [ ] Run the focused test module and confirm expected missing-interface failures.
- [ ] Implement batch shaped-data parsing plus target-date/previous-close return calculation with no stale-date fallback.
- [ ] Re-run the focused module to green.

### Task 3: Sector returns and sector breadth

**Files:**
- Modify: `src/market_breadth.py`, `tests/test_market_breadth.py`

**Interfaces:**
- `calculate_sector_breadth(sector_config, close_rows_by_ticker, target_market_date) -> dict`
- `sector_bar_strength(daily_return, full_scale=0.03) -> float | None`

- [ ] Write failing tests for 11 ETF daily returns, up/down/flat counts, descending sort, missing-date invalid entries, and ±3% full / ±1.5% half / capped visual strength.
- [ ] Run the focused test module and confirm failures concern unavailable sector behavior.
- [ ] Implement sector result and bar-strength derivation without changing market fetch functions.
- [ ] Re-run focused tests to green.

### Task 4: Market Health and divergence

**Files:**
- Create: `src/market_health.py`, `tests/test_market_health.py`

**Interfaces:**
- `calculate_market_health(stocks, sectors, sp500_daily_return, config) -> dict`
- `build_market_breadth_text(market_breadth: dict) -> str`

- [ ] Write failing tests for all stated health score boundaries, invalid inputs, narrow-rally boundaries, positive-breadth boundaries, fixed program summaries, and concise leading/lagging context text.
- [ ] Run `python -m pytest -v tests/test_market_health.py`; confirm expected missing-interface failures.
- [ ] Implement score, level, divergence precedence, summary and compact text generator.
- [ ] Re-run focused tests to green.

### Task 5: Daily orchestration and deterministic fixture

**Files:**
- Modify: `src/main.py`, `tests/test_main.py`

**Interfaces:**
- `build_market_breadth(reference_path, breadth_config, target_market_date, downloader=None) -> tuple[dict, list[str]]`
- Offline fixture returns ten stocks (6/3/1) and 11 sectors (7/4), yielding score approximately 0.6145 / `mixed`.

- [ ] Write failing end-to-end fixture tests for `market_breadth`, stats, health, target market date, and a simulated breadth failure leaving drawdown validity unchanged.
- [ ] Run focused tests and confirm the new report field is absent before implementation.
- [ ] Integrate breadth after Context and before news processing; emit compact breadth/sector/health logs; append breadth warnings without changing critical status calculation.
- [ ] Re-run focused tests to green.

### Task 6: Market-driven News context

**Files:**
- Modify: `src/main.py`, `src/news_prompt.py`, `tests/test_news.py`

**Interfaces:** `select_news(..., market_context={..., "market_breadth": dict, "market_breadth_text": str})`

- [ ] Write failing tests proving the payload carries breadth and that a missing/invalid breadth object does not block selection.
- [ ] Run focused news tests and confirm expected payload failure.
- [ ] Add breadth/sector-rotation priority guidance to the prompt while retaining candidate-only and no-causality constraints.
- [ ] Re-run focused news tests to green.

### Task 7: Market Breadth renderer and responsive styling

**Files:**
- Modify: `templates/report.html`, `static/style.css`, `tests/test_renderer.py`

- [ ] Write failing renderer tests for health badge and tooltip, stock advance ratio/bar, ordered sector rows/diverging bars, summary strip, mobile one-column composition, and no external UI dependency.
- [ ] Run focused renderer tests and confirm missing markup failures.
- [ ] Render the breadth block within the existing US market section, directly after Market Context, using the existing tooltip script generalized for both help-button types.
- [ ] Implement compact 38%/62% desktop layout and single-column mobile layout with CSS-only diverging bar placement from precomputed values.
- [ ] Re-run focused renderer tests to green.

### Task 8: Legacy report compatibility

**Files:**
- Modify: `tests/test_renderer.py`, `src/renderer.py` only if needed

- [ ] Write a failing legacy-report rendering test where no `market_breadth` exists.
- [ ] Run the test and confirm failure is caused by unguarded breadth template access.
- [ ] Add only defensive template guards required for legacy reports.
- [ ] Re-run legacy and breadth renderer tests to green.

### Task 9: Regression and runtime verification

**Files:** Generated output under `work/market-breadth-test/`; generated Pages files under `site/`.

- [ ] Run `python -m pytest -v`.
- [ ] Run `python -m src.main --offline-fixture --base-dir work/market-breadth-test --report-date 2026-08-12`.
- [ ] Run `python -m src.smoke --base-dir work/market-breadth-test` and inspect JSON/HTML for breadth, tooltip, and old report compatibility.
- [ ] If network is available, run a real generation in an isolated `work/` directory and report reference date, coverage, counts, sectors, health, divergence, warnings, and duration.
- [ ] Inspect `git diff --check` and `git status --short`; leave all changes uncommitted and unpushed.
