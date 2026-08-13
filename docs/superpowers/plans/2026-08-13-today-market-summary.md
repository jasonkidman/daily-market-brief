# Today Market Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact, evidence-bounded "今日市场一句话" executive summary after the report header.

**Architecture:** Build the summary only after market, breadth, drawdown, and final selected-news data are ready. A new module will construct a bounded DeepSeek payload and validate its strict three-field JSON, while a deterministic fallback guarantees an action-safe summary if AI is unavailable. The renderer displays the persisted report field only when present, preserving legacy reports.

**Tech Stack:** Python 3.9, existing DeepSeek `call_deepseek`, Jinja2, pytest.

## Global Constraints

- Do not modify RSS sources, event-level news deduplication, news ranking, Market Breadth/Context/Drawdown rules, investment strategy, or GitHub Actions.
- Reuse `src.deepseek_client.call_deepseek`; do not create a second client.
- The model may use only program-provided market data and final news, must not search, predict, invent facts, or assert unsupported causality.
- Program derives `portfolio_action`; the model only verbalizes it. For `hold`, reject obvious conflicting investment instructions and retry three times.
- Return strict JSON `{market, drivers, action}`; enforce an approximately 220-Chinese-character limit across the rendered text.
- On three AI failures or missing credentials, persist a deterministic fallback with `degraded: true`.
- Render only reports that contain `market_summary`; old reports remain valid and omit the module.
- Do not commit or push.

---

### Task 1: Summary data contract and action derivation

**Files:**
- Create: `src/market_summary.py`
- Create: `src/market_summary_prompt.py`
- Create: `tests/test_market_summary.py`

**Interfaces:**
- Produces: `derive_portfolio_action(drawdown: dict) -> str`
- Produces: `generate_market_summary(market_data, market_context, market_breadth, news, drawdown_action, api_key, call_model=call_deepseek, sleep_fn=time.sleep) -> dict`

- [ ] **Step 1: Write failing tests** for `hold`, pending and executed action derivation; a valid strict model response; bounded prompt data; and a `hold` safety conflict.
- [ ] **Step 2: Run the focused test** and verify it fails because `src.market_summary` does not exist.
- [ ] **Step 3: Implement the minimal prompt and module** using existing shared DeepSeek transport, a strict JSON validator, deterministic portfolio-action copy, and the conflict guard.
- [ ] **Step 4: Re-run the focused test** and verify it passes.

### Task 2: Fallback and evidence boundaries

**Files:**
- Modify: `src/market_summary.py`
- Modify: `tests/test_market_summary.py`

**Interfaces:**
- Produces: fallback output with `market`, `drivers`, `action`, and `degraded` fields.

- [ ] **Step 1: Write failing tests** for three model failures, empty final news, unavailable breadth, no unsupported Market Health text, and maximum length.
- [ ] **Step 2: Run the focused test** and verify the fallback behavior is absent.
- [ ] **Step 3: Implement deterministic fallback and bounded payload conversion** using S&P/Nasdaq daily return, valid breadth only, and program-owned action text.
- [ ] **Step 4: Re-run the focused test** and verify it passes.

### Task 3: Daily-report integration

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: final `news`, summarized drawdown state, market snapshots, context snapshots, and breadth output.
- Produces: `report["market_summary"]` after the news pipeline and before persistence.

- [ ] **Step 1: Write a failing offline-pipeline assertion** for persisted summary and program-derived `portfolio_action`.
- [ ] **Step 2: Run the focused test** and verify the new JSON key is absent.
- [ ] **Step 3: Integrate summary generation after final-news production**; use a deterministic offline call model and missing-key fallback for local normal runs.
- [ ] **Step 4: Re-run the focused test** and verify it passes without changing drawdown or news logic.

### Task 4: Legacy-safe executive-summary rendering

**Files:**
- Modify: `templates/report.html`
- Modify: `static/style.css`
- Modify: `tests/test_renderer.py`
- Modify: `src/smoke.py`

**Interfaces:**
- Consumes: optional `report.market_summary`.
- Produces: compact `TODAY IN ONE LINE / 今日市场一句话` section after header with muted provenance.

- [ ] **Step 1: Write failing renderer tests** for summary placement, copy, source hint, responsive CSS classes, and omission on legacy report JSON.
- [ ] **Step 2: Run the focused renderer test** and verify it fails because no summary markup exists.
- [ ] **Step 3: Add compact Jinja markup and responsive CSS**: thin left accent, pale surface, 16–17px text, no large card or data dashboard treatment.
- [ ] **Step 4: Update smoke requirements and re-run the focused test** to verify rendering and legacy compatibility.

### Task 5: Full local verification

**Files:**
- Modify: all above only as needed to resolve verified failures.

- [ ] **Step 1: Run** `python -m pytest -v`.
- [ ] **Step 2: Run** `python -m src.main --offline-fixture --base-dir work/today-market-summary --report-date 2026-08-12`.
- [ ] **Step 3: Run** `python -m src.smoke --base-dir work/today-market-summary` and inspect the generated JSON/HTML.
- [ ] **Step 4: Start** `python -m http.server 8765 --directory work/today-market-summary/site` and report the local URL; do not commit or push.
