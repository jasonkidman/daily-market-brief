# Dashboard Visual Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved compact investment-dashboard visual specification while retaining all market, breadth, drawdown, news, and history business rules; enable high-effort DeepSeek thinking only for the three approved AI tasks.

**Architecture:** Preserve the report JSON schema and existing Jinja data bindings. Reshape only template semantics and CSS layout into compact dashboard components. Extend the single DeepSeek transport with opt-in thought settings, then pass those settings only from market-summary generation, event clustering, and Top News selection; consume only final `message.content`.

**Tech Stack:** Python 3.9, Jinja2, CSS, existing OpenAI-compatible DeepSeek client, pytest.

## Global Constraints

- Do not alter yfinance, Market Context/Breadth/Health/Divergence calculations, Drawdown state/rules/pools, RSS, event dedupe, Top 8 selection rules, history retention, or GitHub Actions.
- Use the approved module order: Header, summary, indices, context, breadth, drawdown, news, footer.
- No new DeepSeek client or model; retain `deepseek-v4-flash` and do not persist or print reasoning content.
- Only Market Summary, event clustering, and Top News selection pass `thinking_enabled=True` and `reasoning_effort="high"`.
- Old reports must continue rendering. No commit or push.

---

### Task 1: Opt-in DeepSeek thinking transport

**Files:**
- Modify: `src/deepseek_client.py`
- Modify: `src/news_events.py`
- Modify: `src/market_summary.py`
- Modify: `tests/test_news.py`
- Modify: `tests/test_news_events.py`
- Modify: `tests/test_market_summary.py`

- [ ] Write failing tests that assert transport receives only final content, passes `thinking.type=enabled` and `reasoning_effort=high` for the three approved tasks, and does not include reasoning in result data.
- [ ] Run focused tests and verify they fail because the transport lacks opt-in arguments.
- [ ] Add optional keyword-only transport arguments and make three callers opt in; retain injected test-model compatibility.
- [ ] Re-run focused tests.

### Task 2: Summary, index, and context dashboard structure

**Files:**
- Modify: `templates/report.html`
- Modify: `static/style.css`
- Modify: `tests/test_renderer.py`

- [ ] Write failing renderer assertions for a summary icon/no English kicker, inline index metrics with directional point arrow, and four consistent inline SVG/CSS context icons.
- [ ] Run focused renderer tests and verify failure.
- [ ] Implement only approved markup/classes and compact visual CSS; preserve existing tooltip IDs, aria attributes, and interaction JavaScript.
- [ ] Re-run focused renderer tests.

### Task 3: Breadth and drawdown dashboard density

**Files:**
- Modify: `templates/report.html`
- Modify: `static/style.css`
- Modify: `tests/test_renderer.py`

- [ ] Write failing renderer tests for 32/68 breadth layout and compact three-column drawdown cards with program-owned values.
- [ ] Run focused renderer tests and verify failure.
- [ ] Reorganize only semantic presentation wrappers/classes; preserve breadth bars/tooltips and drawdown data/status content.
- [ ] Re-run focused renderer tests.

### Task 4: Top 3 plus compact Top 4–8 news layout

**Files:**
- Modify: `templates/report.html`
- Modify: `static/style.css`
- Modify: `tests/test_renderer.py`

- [ ] Write failing renderer assertions for three primary cards, one compact 04–08 list, source/category/title/link preservation, and narrow-screen collapse.
- [ ] Run focused renderer test and verify failure.
- [ ] Implement the approved four-column desktop structure and responsive two/single-column fallbacks without changing ranking/data selection.
- [ ] Re-run focused renderer test.

### Task 5: Full validation

**Files:**
- Modify: only as required by verified failures.

- [ ] Run `python -m pytest -v`.
- [ ] Generate an offline report and run `python -m src.smoke`.
- [ ] Start local HTTP server and inspect generated HTML for required dashboard structure; report local URL and git status.
