# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

默认使用简体中文与我交流。

- 所有分析、解释、总结和执行结果使用中文
- 代码、命令、文件名、变量名、API 名称和必要技术术语保留英文
- 报错信息可以保留英文原文，但需要补充中文解释
- Git commit message 默认使用中文
- 修改代码前，先用中文简要说明准备做什么
- 修改完成后，用中文总结改动内容和验证结果

## What this is

A daily-generated static site (Chinese-language) reporting on a personal investment portfolio: three US indices (S&P 500, Nasdaq-100, Dow), six Market Context indicators (Russell 2000, VIX, DXY, US 10Y, gold, WTI crude), up to 8 curated global news items, and independent drawdown-driven top-up ("加仓") status for S&P 500 and Nasdaq-100. It manages a single ¥200,000 drawdown reserve — no monthly DCA logic, no trading advice, no auto-trading.

## Commands

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -v                    # run all tests
python -m pytest tests/test_market.py -v          # single file
python -m pytest tests/test_market.py::test_name -v  # single test
```

Real full generation (needs network; news degrades gracefully without a Secret):
```bash
export DEEPSEEK_API_KEY="..."
python -m src.main
python -m src.smoke
```

Deterministic offline run (no network, no LLM calls) — useful for testing the whole pipeline:
```bash
python -m src.main --offline-fixture --base-dir work/smoke --report-date 2026-08-12
python -m src.smoke --base-dir work/smoke
```
Offline-generated pages are explicitly flagged as test fixtures and must never be published as real reports.

Replay Layer 2 scoring against a persisted Stage B snapshot (for reproducing/debugging a production news-scoring run):
```bash
python -m src.main --stage-b-snapshot data/news_snapshots/YYYY-MM-DD.json
```

Confirming an actual top-up buy after real execution:
```bash
python -m src.confirm_drawdown <sp500|nasdaq100> <tier_1|tier_2|tier_3|tier_4>
```
Only a `pending` tier can be confirmed; `not_triggered` errors, `executed` is a no-op. This only rerenders existing reports — it does not fetch market data, RSS, or call the LLM.

## Architecture

Entry point is `src/main.py:generate_daily_report`, a single-pass orchestration function (not a class/pipeline framework) called once per day by `.github/workflows/daily-report.yml`. Read this function top-to-bottom to understand the whole system; it wires together the modules below in order:

1. **Market data** (`src/market.py`, `src/market_breadth.py`, `src/market_health.py`, `src/market_sentiment.py`, `src/market_signals.py`) — fetches yfinance daily `Close` data, computes daily return / YTD / drawdown-from-ATH for the 3 core indices, computes S&P 500 constituent-level market breadth (advancers/decliners, sector breadth) from `src/constituents.py`-maintained reference data, and derives market signals/sentiment/health scores. Core-index validity vs. context-index validity are tracked *independently* (`assess_market_validity`) — a context source failing does not block drawdown logic, and vice versa.

   `market_health.py` and `market_sentiment.py` are deliberately separate, not overlapping duplicates: **Health** answers "how broad is today's move" (S&P 500 constituent + sector-ETF advance/decline ratios only); **Sentiment** answers "how greedy/fearful is the market right now" (a weighted composite of VIX level 35% + Health's own score 30% + 20-day momentum 20% + small-cap-vs-large-cap relative strength 15%). Sentiment's inputs overlap with Health's (it consumes Health's score as one input) but the two scores measure different things and are both shown on the page; only Health feeds the Layer 2 LLM summary, Sentiment is display-only.

2. **Drawdown state machine** (`src/drawdown.py`) — `update_drawdown_state` reads/writes `state/drawdown_state.json` (current open cycles per index) and archives closed cycles into `state/drawdown_history.json`. Only runs when market validity for that index holds; a failed validation never creates or mutates a drawdown signal (fail-closed).

3. **News pipeline** — a multi-stage funnel, pure-code except two bounded LLM calls:
   - `src/rss_news.py` fetches candidates from `config/news_sources.yaml` (RSS only, no scraper fallback), each tagged P0/P1/P2 priority.
   - `src/news_dedupe.py` deduplicates locally.
   - `src/news_prefilter.py` (`filter_off_topic_candidates`) drops candidates matching a narrow, high-precision off-topic rule from `config/news_prefilter.yaml` (e.g. content-moderation/child-safety stories, unrelated crypto-theft wire stories) -- zero LLM cost, logged, and deliberately conservative: anything not an exact match still flows through to Scoring rather than being silently dropped.
   - `src/news_events.py` clusters candidates into events by local title-similarity (`cluster_candidates_local`) — this collapses multi-outlet coverage of the same story into one representative before scoring, but only catches near-identical titles (see the TopDedup note below for what it misses).
   - `src/news_candidates.py` builds the final Stage B candidate pool passed to the LLM.
   - `src/news_scoring.py` (`score_candidates`) sends candidate text + `config/news_focus.yaml` focus rules to 灵眸 (an OpenAI-SDK-compatible endpoint, model `gpt-5.6-terra`, JSON output, `reasoning_effort` instead of `temperature`). The model returns only candidate IDs + score/category/translation — it never invents URLs; those are mapped back from the RSS-fetched originals.
   - `src/news_top_dedup.py` (`dedupe_top_candidates`) is a second, bounded LLM call over only the top ~25 scored candidates: pure text-similarity clustering cannot tell that two very differently worded headlines describe the same real-world event/announcement (confirmed on real 2026-09-09 data), so this stage asks the model directly, scoped narrowly enough that it never reopens the old full-pool multi-stage cost problem. It never removes a candidate from Scoring's output or "更多新闻" -- it only decides who additionally gets a "今日重要新闻" slot.
   - Selection of the top N (`NEWS_TOP_N = 10` in `src/main.py`) is done by code (`_select_top_news`: sort by score desc, stable order for ties, after TopDedup's near-duplicates are excluded), not by the model.
   - `src/news_snapshot.py` persists the exact Stage B input (`data/news_snapshots/YYYY-MM-DD.json`) before scoring, fail-fast — this is what `--stage-b-snapshot` replays against (pre-TopDedup, since that snapshot is Scoring's input, not the final selection).
   - If no API key or the LLM call fails entirely, `rule_based_top_news` provides a pure-code fallback ranking rather than an empty news section (TopDedup does not run on this path).

4. **Market summary** (`src/market_summary.py`) — the second and last LLM call, generates the narrative summary from already-computed market/breadth/news data.

5. **Report assembly & rendering** — `src/report.py` writes `data/reports/YYYY-MM-DD.json` and retains only the latest 7 days; `src/renderer.py` (Jinja2, `templates/report.html` + `static/style.css`) renders `site/`, the sole Pages publish directory.

Page-level status (`ok`/`partial`/`critical`) is derived from `validity_summary` + accumulated `warnings`, separately from whether the *drawdown* signal itself is trustworthy — read `_classify_rss_warnings` and `assess_market_validity` together to see how a single failing source is judged material or not.

## Config files

- `config/market.yaml` — index/context names and yfinance tickers.
- `config/drawdown_rules.yaml` — total reserve, 70/30 pool split, per-tier thresholds/ratios.
- `config/news_sources.yaml` — RSS URLs with P0/P1/P2 priority.
- `config/news_focus.yaml` — focus text sent to the scoring LLM.
- `config/news_prefilter.yaml` — narrow, zero-LLM-cost off-topic exclusion rules applied before clustering/scoring (see `src/news_prefilter.py`).
- `config/market_breadth.yaml` — S&P 500 constituent reference file location and sector-breadth minimums.

## CI

`.github/workflows/daily-report.yml` runs at UTC `0 22 * * 0-5` — 06:00 Asia/Shanghai, Monday through Saturday (the UTC day-of-week is one behind the Shanghai one because 22:00 UTC + 8h crosses midnight; Shanghai Sunday is skipped because no new US close has happened since Saturday's report). `.github/workflows/confirm-drawdown.yml` is `.github/workflows/confirm-drawdown.yml` is manually triggered after a real top-up buy. Both share the `investment-report-state` concurrency group with `cancel-in-progress: false` to avoid concurrent state mutation. The daily workflow only commits `data/reports/`, `state/`, `site/` and only deploys Pages when `core_market_valid` is true for that day's report.
