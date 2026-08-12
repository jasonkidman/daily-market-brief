# Daily Market Brief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a daily static investment report with safe market calculations, persistent drawdown cycles, RSS/DeepSeek news processing, seven-day history, and GitHub Pages workflows.

**Architecture:** Focused Python modules exchange JSON-compatible dictionaries. Configuration is YAML, durable state and reports are JSON, Jinja2 renders only `site/`, and GitHub Actions serializes all state-changing work.

**Tech Stack:** Python 3.11, yfinance, feedparser, OpenAI Python SDK, Jinja2, PyYAML, pytest, GitHub Actions, GitHub Pages.

## Global Constraints

- Use historical closing prices, never intraday highs, for ATH and drawdown.
- Never mutate drawdown state when market data validation fails.
- Keep S&P 500 and Nasdaq-100 cycles independent and archive every completed cycle.
- DeepSeek selects only supplied RSS candidates, returns no URLs, uses `deepseek-v4-flash`, and explicitly disables thinking.
- Keep only seven Asia/Shanghai calendar reports and corresponding HTML pages.
- Publish `site/` only; do not add out-of-scope services or features.

---

### Task 1: Market calculations and validation

**Files:** `tests/test_market.py`, `src/market.py`, `config/market.yaml`

**Interfaces:** Produce `calculate_market_snapshot(rows, now)` and `validate_close_rows(rows, now)` dictionaries for reporting and drawdown.

- [ ] Write tests for daily return, YTD, closing ATH, NaN, non-positive close, future date, and insufficient history.
- [ ] Run `pytest tests/test_market.py -v` and verify failures are caused by missing implementation.
- [ ] Implement only the calculation, validation, and yfinance adapter needed by those tests.
- [ ] Re-run the focused tests until green.

### Task 2: Drawdown state machine and confirmation

**Files:** `tests/test_drawdown.py`, `tests/test_confirm.py`, `src/drawdown.py`, `src/confirm_drawdown.py`, `config/drawdown_rules.yaml`

**Interfaces:** Produce `update_index_state(...)`, `update_drawdown_state(...)`, `summarize_index_state(...)`, and `confirm_tier(...)`.

- [ ] Write threshold, sticky pending, executed idempotency, cross-tier, new ATH, archive, missed-run ATH, and confirmation tests.
- [ ] Run focused tests and verify expected red failures.
- [ ] Implement deterministic state transitions with ISO timestamps and cycle IDs.
- [ ] Re-run focused tests until green.

### Task 3: RSS, dedupe, and DeepSeek boundary

**Files:** `tests/test_news.py`, `src/rss_news.py`, `src/news_dedupe.py`, `src/deepseek_client.py`, `config/news_sources.yaml`

**Interfaces:** Produce candidate dictionaries, `dedupe_candidates(...)`, `validate_selection(...)`, and `select_news(...)` with validated RSS metadata mapping.

- [ ] Write tests for URL/title/similarity dedupe, source isolation, invalid IDs/categories/counts, and three-attempt degradation.
- [ ] Run focused tests and verify expected red failures.
- [ ] Implement RSS-only acquisition, local dedupe, independent prompt, strict JSON validation, and injected retry delays.
- [ ] Re-run focused tests until green.

### Task 4: Report retention and static rendering

**Files:** `tests/test_history.py`, `tests/test_renderer.py`, `src/report.py`, `src/renderer.py`, `templates/report.html`, `static/style.css`

**Interfaces:** Produce `write_report(...)`, `retain_latest_reports(...)`, and `render_site(...)`.

- [ ] Write tests proving eighth-day deletion and current/history HTML generation.
- [ ] Run focused tests and verify expected red failures.
- [ ] Implement JSON persistence, cleanup, responsive template, status copy, cards, history navigation, and footer.
- [ ] Re-run focused tests until green.

### Task 5: End-to-end coordinator and automation

**Files:** `tests/test_main.py`, `src/main.py`, `.github/workflows/daily-report.yml`, `.github/workflows/confirm-drawdown.yml`, `requirements.txt`, `.gitignore`, `README.md`

**Interfaces:** `python -m src.main [--offline-fixture]` generates report/state/site; `python -m src.confirm_drawdown INDEX TIER` confirms and rerenders without DeepSeek.

- [ ] Write an end-to-end offline generation test and verify it fails before the coordinator exists.
- [ ] Implement coordinator, fixture mode, workflows, dependencies, ignore rules, and deployment instructions.
- [ ] Run the focused test, then `pytest -v`.
- [ ] Run `python -m src.main --offline-fixture`, inspect generated JSON and HTML, and run an HTML smoke validation.
- [ ] Review the acceptance checklist and report any unverified external deployment status separately.
