# News Investment Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the heat-oriented news selection and mixed-card layout with a validated investment-priority Top 8 and a responsive 4/2/1-column card grid based on the supplied HTML prototype.

**Architecture:** Preserve deterministic URL/title dedupe and Stage A event clustering. Enrich only Stage B so DeepSeek scores, filters, ranks, and writes bounded analysis for representative events; Python validates every field and maps trusted source metadata by `candidate_id`. The Jinja template renders persisted fields directly and provides display-only fallbacks for legacy reports.

**Tech Stack:** Python 3.11, DeepSeek OpenAI-compatible client, Jinja2, HTML5, CSS, pytest.

## Global Constraints

- Do not modify Today Market Summary, market indices, Market Context, Market Breadth, drawdown logic, global page width, global color tokens, or global font rules.
- Keep RSS-provided `source`, `url`, `published_at`, `original_title`, `event_summary`, and `topic_group` program-owned.
- Never include URLs in the Stage B model payload and never accept a model-provided URL or source.
- Allow fewer than eight articles when fewer than eight candidates meet the 50-point minimum.
- Enforce title <= 70 chars, summary <= 180 chars, impact <= 220 chars, focus <= 80 chars, reason <= 120 chars, and 1-4 tags of <= 16 chars each.
- Render desktop 4 columns, medium 2 columns, and phone 1 column without truncating text.
- Preserve legacy report rendering without mutating old JSON.

---

### Task 1: Strict investment-analysis contract

**Files:**
- Modify: `tests/test_news.py`
- Modify: `src/deepseek_client.py:13-86`

**Interfaces:**
- Consumes: Stage B JSON plus program-owned representative-event candidates.
- Produces: `validate_selection(payload: Any, candidates: list[dict]) -> list[dict]` with persisted `investment_impact`, `focus`, `tags`, and `investment_relevance_score`.

- [ ] **Step 1: Add a reusable valid enriched selection fixture**

Add to `tests/test_news.py`:

```python
def enriched_selection(cid="1", rank=1, score=92, category="美联储 / 利率"):
    return {
        "rank": rank,
        "candidate_id": cid,
        "category": category,
        "title_zh": "美联储维持政策利率不变",
        "summary_zh": "委员会维持政策利率不变，并继续关注通胀和就业数据。",
        "investment_impact": "通胀回落 → 降息空间增加 → 长端利率压力缓解 → 成长股估值获得支撑。",
        "focus": "FOMC · 官员讲话 · 10Y 美债",
        "tags": ["Fed", "10Y 美债", "成长股估值"],
        "investment_relevance_score": score,
        "selection_reason": "美国利率路径直接影响股票折现率。",
    }
```

- [ ] **Step 2: Write failing validation tests**

Add tests that assert:

```python
def test_validates_enriched_investment_fields_and_keeps_source_metadata_program_owned():
    pool = [candidate("1", "Fed holds rates", "https://trusted.example/fed", source="Reuters")]
    item = {**enriched_selection(), "source": "Fake", "url": "https://fake.example"}
    news = validate_selection({"news": [item]}, pool)
    assert news[0]["source"] == "Reuters"
    assert news[0]["url"] == "https://trusted.example/fed"
    assert news[0]["investment_relevance_score"] == 92
    assert news[0]["tags"] == ["Fed", "10Y 美债", "成长股估值"]

@pytest.mark.parametrize("patch", [
    {"investment_relevance_score": 49},
    {"investment_relevance_score": 92.5},
    {"investment_impact": "利好科技股"},
    {"tags": []},
    {"tags": ["x"] * 5},
    {"title_zh": "标" * 71},
    {"summary_zh": "摘" * 181},
    {"investment_impact": "10Y 美债 → " + "影" * 221},
    {"focus": "关" * 81},
    {"selection_reason": "理" * 121},
])
def test_rejects_invalid_investment_contract(patch):
    pool = [candidate("1", "Fed holds rates", "https://x/1")]
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**enriched_selection(), **patch}]}, pool)
```

Also test score ordering and the topic cap:

```python
def test_requires_non_increasing_scores():
    pool = [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(1, 3)]
    payload = {"news": [enriched_selection("1", 1, 70), enriched_selection("2", 2, 90)]}
    with pytest.raises(NewsSelectionError):
        validate_selection(payload, pool)

def test_rejects_third_same_topic_without_high_score_exception_reason():
    pool = [{**candidate(str(i), f"Title {i}", f"https://x/{i}"), "topic_group": "AI_CHIPS"}
            for i in range(1, 4)]
    payload = {"news": [enriched_selection(str(i), i, 90 - i) for i in range(1, 4)]}
    with pytest.raises(NewsSelectionError):
        validate_selection(payload, pool)
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_news.py -k 'investment_contract or enriched or non_increasing or third_same_topic' -v
```

Expected: FAIL because the enriched fields, categories, thresholds, and theme-cap validation are absent.

- [ ] **Step 4: Implement field and collection validators**

In `src/deepseek_client.py`, replace `ALLOWED_CATEGORIES` with the nine approved categories and add constants for field limits. Add helpers equivalent to:

```python
def _required_text(item, key, limit):
    value = str(item.get(key, "")).strip()
    if not value or len(value) > limit:
        raise NewsSelectionError(f"{key} 不合法。")
    return value

def _has_impact_path(value):
    if value == "短期资产价格影响有限，暂以观察为主。":
        return True
    variables = ("SPY", "Nasdaq", "纳指", "科技股", "美债", "收益率", "美元", "信用", "AI", "半导体", "估值", "通胀", "油价")
    connectors = ("→", "若", "如果", "导致", "使得", "进而", "从而", "有利于", "压制")
    return any(term in value for term in variables) and any(term in value for term in connectors)
```

Validate integer score 50-100, 1-4 string tags, continuous ranks, non-increasing scores, and topic concentration. For a third item in one program-owned `topic_group`, require score >= 85 and `selection_reason` containing `主题上限例外`; otherwise raise `NewsSelectionError`. Build the persisted item only from validated model analysis plus program-owned source metadata.

- [ ] **Step 5: Update existing selection tests to use the enriched fixture**

Every successful Stage B payload in `tests/test_news.py`, `tests/test_main.py`, and `tests/test_market_summary.py` must include the new required fields. Keep invalid-category and retry tests focused on their original failure reason.

- [ ] **Step 6: Run news and dependent tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_news.py tests/test_main.py tests/test_market_summary.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the contract**

```bash
git add src/deepseek_client.py tests/test_news.py tests/test_main.py tests/test_market_summary.py
git commit -m "feat: validate investment-priority news analysis"
```

---

### Task 2: Investment-priority DeepSeek selection prompt

**Files:**
- Modify: `src/news_prompt.py`
- Modify: `tests/test_news.py`

**Interfaces:**
- Consumes: existing Stage B `events`, seven-day history, and optional market context.
- Produces: strict enriched JSON accepted by Task 1 without changing `select_news` transport or retry behavior.

- [ ] **Step 1: Write failing prompt-contract tests**

Add one test asserting the prompt contains all of these exact concepts:

```python
for phrase in (
    "长期持有 SPY 与 Nasdaq-100",
    "美国资产定价的重要程度",
    "investment_relevance_score",
    "宏观/政策重要性 0-30",
    "低于50分不得入选",
    "普通产品更新",
    "普通公司融资",
    "短期资产价格影响有限，暂以观察为主。",
    "不要返回URL",
):
    assert phrase in SYSTEM_PROMPT
```

Keep the existing assertions for event-level dedupe, market-structure relevance, cautious causality, and the two-item topic principle.

- [ ] **Step 2: Run the prompt tests and verify RED**

Run:

```bash
python -m pytest tests/test_news.py -k 'prompt' -v
```

Expected: FAIL because the current prompt is heat/explanation oriented and returns only title/summary.

- [ ] **Step 3: Replace the Stage B prompt**

Write a single bounded system prompt that:

- Defines the audience as long-term SPY and Nasdaq-100 investors.
- Ranks by future US asset-pricing relevance rather than popularity.
- Lists the six approved priority tiers and default filters.
- Applies the 30/20/20/15/10/5 score dimensions.
- Allows 50-69 only to supplement stronger news and allows fewer than eight.
- Requires full transmission logic or the exact observation-only sentence.
- Preserves candidate-only facts, cautious causal language, recent-event handling, Stage A event uniqueness, and topic concentration.
- Returns exactly the Task 1 schema and never returns URLs.

- [ ] **Step 4: Run prompt and selection tests and verify GREEN**

```bash
python -m pytest tests/test_news.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the prompt**

```bash
git add src/news_prompt.py tests/test_news.py
git commit -m "feat: prioritize news by US asset pricing impact"
```

---

### Task 3: Complete offline fixture and persisted report coverage

**Files:**
- Modify: `src/main.py:198-233`
- Modify: `tests/test_main.py`
- Modify: `tests/test_renderer.py` fixture helpers that produce modern news items

**Interfaces:**
- Consumes: the strict `validate_selection` contract from Task 1.
- Produces: deterministic offline reports whose selected news contain all display fields without representing real advice.

- [ ] **Step 1: Write a failing offline report assertion**

Extend `test_offline_fixture_runs_complete_pipeline`:

```python
for item in report["news"]:
    assert item["investment_impact"]
    assert item["focus"]
    assert item["tags"]
    assert 50 <= item["investment_relevance_score"] <= 100
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m pytest tests/test_main.py::test_offline_fixture_runs_complete_pipeline -v
```

Expected: FAIL because `_offline_news()` lacks the new fields.

- [ ] **Step 3: Enrich the three offline selections**

Give the Fed, Nvidia, and oil examples approved categories, descending scores, bounded analysis, focus, tags, and a transmission path. Keep all summaries and impact copy explicitly described as offline fixture content and not real investment information.

- [ ] **Step 4: Run main and renderer fixture tests**

```bash
python -m pytest tests/test_main.py tests/test_renderer.py -v
```

Expected: PASS after any renderer test fixtures that represent new reports include complete fields; tests that explicitly exercise legacy rendering remain incomplete on purpose.

- [ ] **Step 5: Commit fixture support**

```bash
git add src/main.py tests/test_main.py tests/test_renderer.py
git commit -m "test: enrich offline news analysis fixture"
```

---

### Task 4: Responsive eight-card news module

**Files:**
- Modify: `tests/test_renderer.py`
- Modify: `templates/report.html:144-150`
- Modify: `static/style.css:84-102`

**Interfaces:**
- Consumes: up to eight persisted news items in the Task 1 schema or legacy items missing the new fields.
- Produces: one complete `.news-card` per item in a `.news-layout` responsive grid.

- [ ] **Step 1: Write failing modern and legacy renderer tests**

Create an eight-item enriched fixture and assert:

```python
assert html.count('class="card news-card"') == 8
assert 'class="news-section-meta">优先级：宏观政策 · 利率就业 · 金融 · AI · 地缘政治</div>' in html
assert "Top 8 · 投资影响优先" in html
assert html.count('class="news-impact"') == 8
assert html.count('class="news-focus"') == 8
assert html.count('class="news-tags"') == 8
assert html.count("查看原文 →") == 8
assert 'class="news-list"' not in html and 'class="nrow"' not in html
assert ".news-layout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))" in compact_css
assert "@media(max-width:1180px)" in compact_css
assert "@media(max-width:700px)" in compact_css
```

Create a separate legacy item with no new fields and assert the two exact compatibility messages from the spec, no fabricated score, and no empty tag wrapper.

- [ ] **Step 2: Run the renderer tests and verify RED**

```bash
python -m pytest tests/test_renderer.py -k 'news' -v
```

Expected: FAIL because the current layout renders three cards plus one compact list.

- [ ] **Step 3: Replace only the news-section markup**

Use the existing section boundary and degradation/empty states. Add the new title/subtitle and right-side note, then iterate over all `report.news` items:

```html
<article class="card news-card">
  <div class="news-meta"><span class="rank">{{ '%02d'|format(item.rank) }}</span><span class="news-category">{{ item.category }}</span><span class="news-source">{{ item.source }}</span></div>
  <h3>{{ item.title_zh }}</h3>
  <p class="news-summary">{{ item.summary_zh }}</p>
  <div class="news-impact"><strong>投资影响：</strong>{{ item.investment_impact or '历史报告未生成结构化投资影响，建议结合原文观察。' }}</div>
  {% if item.tags %}<div class="news-tags">{% for tag in item.tags %}<span>{{ tag }}</span>{% endfor %}</div>{% endif %}
  <div class="news-footer"><span class="news-focus">关注：{{ item.focus or '查看原文后续进展' }}</span><a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">查看原文 →</a></div>
</article>
```

Use `.get()` where Jinja compatibility requires it so missing legacy keys never become errors.

- [ ] **Step 4: Replace only news-scoped CSS**

Match the prototype using existing variables: four equal columns and 14px gaps; 15px card padding; 5px radius; flex-column cards; current blue rank pill; 17px/1.45 title; 12px/1.75 summary; soft impact block with blue left border; compact tags; light footer link. Add `@media(max-width:1180px)` two columns and `@media(max-width:700px)` one column. Do not change `.page`, global `body`, other cards, or other breakpoint declarations.

- [ ] **Step 5: Run renderer tests and verify GREEN**

```bash
python -m pytest tests/test_renderer.py -v
```

Expected: PASS with old mixed-layout assertions replaced by the new contract.

- [ ] **Step 6: Commit the frontend**

```bash
git add templates/report.html static/style.css tests/test_renderer.py
git commit -m "feat: render investment-priority news grid"
```

---

### Task 5: Full pipeline and visual verification

**Files:**
- Modify: implementation files only if a verified defect requires a scoped fix.
- Update: `design-qa.md` with current news-module evidence.
- Generate: `site/index.html`, `site/history/*.html`, `site/style.css` through `src.renderer` only.

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: verified current site and a `design-qa.md` whose final result is `passed` or `blocked`.

- [ ] **Step 1: Run the full automated suite**

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run a deterministic offline pipeline in temporary output**

```bash
python -m src.main --offline-fixture --base-dir /tmp/daily-market-brief-news --report-date 2026-08-12
python -m src.smoke --base-dir /tmp/daily-market-brief-news
```

Inspect the JSON to confirm program-owned URL/source fields and all enriched analysis fields.

- [ ] **Step 3: Regenerate the current tracked site without fetching new data**

```bash
python -c 'from pathlib import Path; from src.renderer import render_site; root=Path("."); render_site(root/"data/reports", root/"templates/report.html", root/"static/style.css", root/"site")'
python -m src.smoke
git diff --check
```

- [ ] **Step 4: Compare the supplied prototype and implementation**

Serve the source HTML and current `site/` locally. Capture both at the same desktop viewport, place them in one side-by-side comparison image, and inspect fonts, spacing, colors, copy hierarchy, and card density. Capture implementation at desktop, medium, and phone widths; verify `scrollWidth == clientWidth` at each width.

- [ ] **Step 5: Record design QA**

Update `design-qa.md` with source and implementation screenshot paths, viewports, state, side-by-side evidence, required fidelity surfaces, findings, fixes, and exact `final result: passed` only when no P0/P1/P2 findings remain.

- [ ] **Step 6: Re-run completion verification after visual fixes**

```bash
python -m pytest -v
python -m src.smoke
git diff --check
git status --short
```

Expected: full suite and smoke pass; changes are limited to news logic, news prompt, news fixtures/tests, news module source/generated files, and design QA evidence.
