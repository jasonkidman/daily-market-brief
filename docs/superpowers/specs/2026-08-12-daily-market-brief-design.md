# Daily Market Brief Design

## Scope

Build a Python and Jinja2 static site that produces one Asia/Shanghai-dated report per calendar day, keeps seven report JSON files and seven rendered pages, and is deployable to GitHub Pages. The site has market, drawdown, news, health, and history sections only. It has no backend, database, login, charts, trading integration, or real-time data.

## Architecture

- `src/market.py` fetches yfinance close history and computes validated close, daily return, YTD return, ATH, and ATH date.
- `src/drawdown.py` owns independent index cycles, tier transitions, missed-run ATH detection, archive creation, and UI summaries.
- `src/confirm_drawdown.py` performs the only `pending -> executed` transition and records execution metadata.
- `src/rss_news.py`, `src/news_dedupe.py`, and `src/deepseek_client.py` fetch RSS candidates, perform deterministic local deduplication, and validate model-selected candidates. RSS or AI failures degrade only the news section.
- `src/report.py` assembles report JSON and enforces seven-calendar-day retention. `src/renderer.py` renders current and history HTML from the retained JSON files. `src/main.py` coordinates the daily flow.
- YAML owns market, source, and drawdown configuration. JSON in `state/` and `data/reports/` is durable Git state. `site/` alone is the Pages artifact.

## Data Safety

Market validation runs before any drawdown transition. Invalid or stale/future/incomplete close data sets `market_data_valid=false`, displays the required critical message, and passes no update data to the drawdown state machine. Historical close rows since the prior state update are inspected so an intervening ATH starts and archives a new cycle even if the latest close is below that ATH.

## News Safety

RSS feeds are the only news acquisition mechanism. Local deduplication precedes DeepSeek. The model returns candidate IDs without URLs; the application maps validated IDs to RSS source records. Invalid model output is retried up to three total attempts with injectable delays. Exhaustion produces a visible warning while market and drawdown generation continues.

## Rendering

The Jinja2 template uses a restrained off-white, navy, and steel-blue system, a 1200px desktop container, compact cards, and a single-column mobile breakpoint. History navigation links only to retained report pages. Positive/negative colors are limited to small numerical accents and status labels.

## Verification

Unit tests cover calculation, validation, tier thresholds, sticky states, confirmation, new-ATH archival, missed runs, news dedupe/model validation/retries, RSS source isolation, and seven-day retention. The full local generation uses an offline fixture mode to deterministically exercise the complete JSON-to-HTML pipeline without requiring network access or a secret. HTML smoke checks verify headings, report/market dates, status content, history links, and absence of secret material.
