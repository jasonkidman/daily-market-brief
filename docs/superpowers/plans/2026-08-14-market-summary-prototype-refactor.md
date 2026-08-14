# Market Summary Prototype Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild only the “今日市场一句话” module to match the uploaded HTML prototype while preserving the rest of the page and all existing data behavior.

**Architecture:** Keep the existing optional `report.market_summary` contract and Jinja render boundary. Split its three fields into semantic presentation blocks inside the current section and style those blocks with existing design tokens; test the DOM, responsive CSS, legacy omission, and unchanged global page rules.

**Tech Stack:** Python 3.11, Jinja2, HTML5, CSS, pytest.

## Global Constraints

- Only change `templates/report.html`, `static/style.css`, and directly related assertions in `tests/test_renderer.py`.
- Preserve all other modules, global colors, `.page` width, and Python data-generation logic.
- Map `market` to the main summary, `drivers` to “今日关注”, and `action` to the strategy bar.
- Keep the module optional for reports without `market_summary`.
- Do not hand-edit generated `site/` files; use the existing render pipeline.

---

### Task 1: Semantic summary module and prototype-matched styling

**Files:**
- Modify: `tests/test_renderer.py`
- Modify: `templates/report.html`
- Modify: `static/style.css`

**Interfaces:**
- Consumes: optional `report.market_summary.{market,drivers,action}`.
- Produces: `.summary-inner`, `.summary-content`, `.market-summary`, `.summary-focus`, `.summary-focus-label`, `.summary-focus-text`, `.summary-strategy`, and `.summary-src` markup while retaining `.summary` and `.summary-icon`.

- [ ] **Step 1: Write failing renderer assertions**

Update `test_renders_today_in_one_line_after_header_and_omits_it_for_legacy_reports` and `test_today_summary_is_the_readable_primary_visual_focus` to require:

```python
assert 'class="summary-inner"' in index_html
assert 'class="market-summary"' in index_html
assert 'class="summary-focus-label">今日关注</div>' in index_html
assert 'class="summary-focus-text">市场同时关注利率预期与人工智能相关事件。</div>' in index_html
assert 'class="summary-strategy">策略：未触发额外回撤加仓，维持正常定投，备用金保持不动。</div>' in index_html
assert ".summary-inner{display:grid;grid-template-columns:46pxminmax(0,1fr)" in compact_css
assert ".summary-content{max-width:1180px}" in compact_css
assert ".summary-focus{margin-top:12px;display:flex" in compact_css
assert ".summary-strategy{margin-top:14px;background:" in compact_css
assert ".page{width:min(1440px,calc(100%-34px))" in compact_css
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_renderer.py::test_renders_today_in_one_line_after_header_and_omits_it_for_legacy_reports tests/test_renderer.py::test_today_summary_is_the_readable_primary_visual_focus -v
```

Expected: FAIL because the new semantic classes and three-level layout do not exist.

- [ ] **Step 3: Implement the minimal Jinja structure**

Replace only the contents of the existing conditional summary section with:

```html
<section class="summary" aria-labelledby="today-summary-title">
  <div class="summary-inner">
    <div class="summary-icon" aria-hidden="true">☼</div>
    <div>
      <h2 id="today-summary-title">今日市场一句话</h2>
      <div class="summary-content">
        <p class="market-summary"><strong>{{ report.market_summary.market }}</strong></p>
        <div class="summary-focus">
          <div class="summary-focus-label">今日关注</div>
          <div class="summary-focus-text">{{ report.market_summary.drivers }}</div>
        </div>
        <div class="summary-strategy">策略：{{ report.market_summary.action }}</div>
      </div>
      <div class="summary-src">基于市场数据 · Market Breadth · Top News</div>
    </div>
  </div>
</section>
```

- [ ] **Step 4: Implement the minimal module-scoped CSS**

Replace the current summary rules with prototype-derived measurements while using current variables:

```css
.summary{margin-top:18px;padding:20px 22px 16px;border:1px solid var(--line);border-radius:7px;background:var(--surface)}
.summary-inner{display:grid;grid-template-columns:46px minmax(0,1fr);gap:16px;align-items:start}
.summary-icon{width:44px;height:44px;margin-top:2px;border-radius:50%;display:grid;place-items:center;background:var(--icon-bg);color:var(--icon-fg);font-size:22px}
.summary h2{margin:2px 0 10px;font-size:19px;line-height:1.3;font-weight:700}
.summary-content{max-width:1180px}
.market-summary{margin:0;color:#243d50;font-size:16px;line-height:1.72;font-weight:450}
.market-summary strong{color:#173247;font-weight:700}
.summary-focus{margin-top:12px;display:flex;gap:12px;align-items:flex-start}
.summary-focus-label{flex:0 0 auto;margin-top:2px;padding:3px 7px;border-radius:4px;background:var(--icon-bg);color:#6a8aa0;font-size:13px;font-weight:700}
.summary-focus-text{color:#526979;font-size:15px;line-height:1.65}
.summary-strategy{margin-top:14px;padding:10px 12px;border-left:3px solid var(--blue);border-radius:4px;background:#f7fafc;color:#203f55;font-size:15px;line-height:1.6;font-weight:700}
.summary-src{margin-top:11px;text-align:right;color:#8a9aa6;font-size:12px}
```

Update only summary declarations in the existing breakpoints:

```css
@media(max-width:900px){.summary{padding:18px 16px 14px}.summary-inner{grid-template-columns:40px minmax(0,1fr);gap:12px}.summary-icon{width:38px;height:38px}.summary-content{max-width:none}.market-summary{font-size:15px}}
@media(max-width:600px){.summary{padding:18px 14px 14px}.summary-focus{gap:8px}.summary-focus-text,.summary-strategy{font-size:14px}}
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: both tests PASS.

- [ ] **Step 6: Run regression verification**

Run:

```bash
python -m pytest -v
python -m src.smoke
```

Expected: all tests PASS and smoke validation exits 0.

- [ ] **Step 7: Regenerate the current site and inspect the diff**

Run:

```bash
python -c 'from pathlib import Path; from src.renderer import render_site; root=Path("."); render_site(root/"data/reports", root/"templates/report.html", root/"static/style.css", root/"site")'
git diff --check
git diff -- templates/report.html static/style.css tests/test_renderer.py site/index.html site/style.css site/history
```

Expected: generated HTML/CSS reflects only the summary module change; no data JSON or unrelated template module changes.
