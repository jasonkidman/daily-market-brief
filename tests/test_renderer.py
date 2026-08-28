import json
import re

from src.renderer import render_site


def report(date):
    return {
        "report_date": date, "generated_at": f"{date} 10:00 CST", "market_date": "2026-08-11",
        "status": "ok", "status_label": "🟢 数据更新正常", "warnings": [],
        "market_data_valid": True,
        "market": {key: {"name": name, "close": 100, "daily_return": 0.01, "ytd_return": 0.10}
                   for key, name in [("sp500", "S&P 500"), ("nasdaq100", "Nasdaq-100"), ("dow", "Dow Jones")]},
        "drawdown": {
            "sp500": {
                "name": "S&P 500", "status": "normal", "current_drawdown": 0.06,
                "ath": 8000, "ath_date": "2026-07-31", "next_threshold": 0.10,
                "distance_to_next": 0.04, "pool": 140000, "executed_amount": 0,
                "pending_amount": 0, "remaining_amount": 140000,
                "pending_tiers": [], "executed_tiers": [],
            },
            "nasdaq100": {
                "name": "Nasdaq-100", "status": "pending", "current_drawdown": 0.17,
                "ath": 30000, "ath_date": "2026-07-31", "next_threshold": 0.20,
                "distance_to_next": 0.03, "pool": 60000, "executed_amount": 0,
                "pending_amount": 12000, "remaining_amount": 48000,
                "pending_tiers": [{"id": "tier_1", "label": "第一档", "amount": 12000}],
                "executed_tiers": [],
            },
        }, "news": [], "news_degraded": False,
    }


def modern_news_item(rank, title_prefix="新闻", summary="摘要"):
    return {
        "rank": rank,
        "category": "美联储 / 利率",
        "source": "Fixture",
        "title_zh": f"{title_prefix} {rank}",
        "summary_zh": summary,
        "url": f"https://example.com/{rank}",
        "published_at": "2026-08-12T10:00:00+00:00",
        "focus": "离线夹具关注项",
        "tags": ["离线夹具", "渲染"],
        "investment_relevance_score": max(50, 92 - rank + 1),
        "selection_reason": "仅用于验证现代新闻字段渲染，不代表真实投资建议。",
    }


def test_renders_dynamic_important_news_cards_from_persisted_fields(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["news"] = [
        modern_news_item(rank, "投资优先新闻", f"第 {rank} 条离线摘要")
        for rank in range(1, 9)
    ]
    payload["news"][0]["source"] = "Persisted Source"
    payload["news"][0]["url"] = "https://persisted.example/news-1"
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert html.count('class="card news-card"') == 8
    assert 'class="news-section-meta">优先级：宏观政策 · 利率就业 · 金融 · AI · 地缘政治</div>' in html
    assert "今日重要新闻" in html
    assert 'class="news-impact"' not in html
    assert html.count('class="news-focus"') == 8
    assert html.count('class="news-tags"') == 8
    assert html.count("查看原文 →") == 8
    assert 'class="news-list"' not in html and 'class="nrow"' not in html
    assert "Persisted Source" in html
    assert 'href="https://persisted.example/news-1"' in html
    assert ".news-layout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))" in compact_css
    assert ".news-meta.rank{min-width:31px;height:24px;padding:09px;border-radius:12px;background:var(--blue);color:var(--surface)" in compact_css
    assert ".news-impact" not in compact_css
    assert ".news-summary{margin:0;color:var(--muted);font-size:12px;line-height:1.75}" in compact_css
    assert ".news-footera{color:var(--blue);font-size:11px" in compact_css
    news_css = compact_css[compact_css.index(".news-title-note"):compact_css.index("footer{")]
    for raw_color in ("#6ea8cf", "#f7f9fb", "#40586d", "#627486", "#2e719c", "#677b8d", "#7f8d96"):
        assert raw_color not in news_css
    assert "@media(max-width:1180px){.news-layout{grid-template-columns:repeat(2,minmax(0,1fr))}}" in compact_css
    assert 'class="section-title news-section-title"' in html
    assert "@media(max-width:700px){.news-section-title{align-items:flex-start;flex-direction:column}.news-layout{grid-template-columns:1fr}.news-section-meta{white-space:normal;text-align:left}.news-card{min-height:0}}" in compact_css


def test_legacy_news_uses_exact_fallback_copy_without_tags_or_score(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["news"] = [{
        "rank": 1,
        "category": "金融市场",
        "source": "Legacy Source",
        "title_zh": "旧版新闻标题",
        "summary_zh": "旧版报告保留的摘要。",
        "url": "https://legacy.example/report",
        "published_at": "2026-08-12T10:00:00+00:00",
    }]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "投资影响" not in html
    assert "关注：查看原文后续进展" in html
    assert 'class="news-tags"' not in html
    assert "investment_relevance_score" not in html
    assert "投资关联度" not in html
    assert "Legacy Source" in html
    assert 'href="https://legacy.example/report"' in html


def test_renders_index_and_history_pages(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for date in ("2026-08-11", "2026-08-12"):
        (reports / f"{date}.json").write_text(json.dumps(report(date)), encoding="utf-8")
    template = __import__("pathlib").Path(__file__).parents[1] / "templates" / "report.html"
    style = __import__("pathlib").Path(__file__).parents[1] / "static" / "style.css"
    site = tmp_path / "site"
    render_site(reports, template, style, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "Daily Market Brief" in html
    assert "报告日期：2026-08-12" in html
    assert "行情数据日期：2026-08-11 美股收盘" in html
    assert (site / "history" / "2026-08-11.html").exists()


def test_renders_today_in_one_line_after_header_and_omits_it_for_legacy_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500小幅上涨，纳指100相对更强，市场内部仍有分化。",
        "drivers": "市场同时关注利率预期与人工智能相关事件。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
        "degraded": False,
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    legacy = report("2026-08-11")
    (reports / "2026-08-11.json").write_text(json.dumps(legacy), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    index_html = (site / "index.html").read_text(encoding="utf-8")
    legacy_html = (site / "history" / "2026-08-11.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    assert "TODAY IN ONE LINE" not in index_html and "今日市场一句话" in index_html
    assert 'class="summary-icon"' in index_html
    assert "基于市场数据 · Market Breadth · Top News" in index_html
    assert 'class="summary-inner"' in index_html
    assert 'class="market-summary"' in index_html
    assert 'class="summary-focus-label">今日关注</div>' in index_html
    assert 'class="summary-focus-text">市场同时关注利率预期与人工智能相关事件。</div>' in index_html
    assert 'class="card strategy-card"' in index_html
    assert 'id="today-strategy-title" class="strategy-title">今日策略</h2>' in index_html
    assert 'class="strategy-action strategy-pending">加仓</div>' in index_html
    assert 'class="strategy-sub">Nasdaq-100 已触发第一档，等待确认</div>' in index_html
    assert index_html.index("今日市场一句话") > index_html.index("</header>")
    assert index_html.index("今日市场一句话") < index_html.index("美国主要指数")
    assert "标普500小幅上涨" in index_html
    assert "今日市场一句话" not in legacy_html
    assert ".summary" in css
    assert ".summary-grid{margin-top:18px" in compact_css
    assert ".summary-inner{display:grid;grid-template-columns:46pxminmax(0,1fr)" in compact_css
    assert ".summary-content{max-width:1180px}" in compact_css
    assert ".summary-focus{margin-top:12px;display:flex" in compact_css
    assert ".strategy-action{font-size:36px" in compact_css
    assert ".page{width:min(1440px,calc(100%-34px))" in compact_css
    assert "@media(max-width:900px)" in compact_css


def test_today_summary_is_the_readable_primary_visual_focus(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500小幅上涨，纳指100相对更强。",
        "drivers": "市场同时关注利率预期。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert "标普500小幅上涨，纳指100相对更强。" in html
    assert 'class="summary-inner"' in html
    assert 'class="market-summary"' in html
    assert 'class="summary-focus-label">今日关注</div>' in html
    assert 'class="summary-focus-text">市场同时关注利率预期。</div>' in html
    assert 'class="card strategy-card"' in html
    assert 'id="today-strategy-title" class="strategy-title">今日策略</h2>' in html
    assert ".summary-grid{margin-top:18px" in compact_css
    assert ".summary-inner{display:grid;grid-template-columns:46pxminmax(0,1fr)" in compact_css
    assert ".summaryh2{margin:2px010px" in compact_css
    assert "font-size:19px;line-height:1.3;font-weight:700" in compact_css
    assert ".market-summary{margin:0;color:#243d50;font-size:16px;line-height:1.72" in compact_css
    assert ".summary-icon{width:44px;height:44px" in compact_css
    assert "font-size:22px" in compact_css
    assert ".summary-focus-label{flex:00auto" in compact_css
    assert ".strategy-title{margin:006px" in compact_css
    assert ".summary-src{margin-top:11px;text-align:right" in compact_css
    assert "@media(max-width:600px)" in compact_css


def test_dashboard_visual_structure_maps_existing_data_without_changing_it(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500小幅上涨，纳指100相对更强。",
        "drivers": "市场同时关注利率预期。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
    }
    for key, ticker, daily, ytd in (
        ("sp500", "^GSPC", 0.003, 0.132), ("nasdaq100", "^NDX", -0.002, 0.101), ("dow", "^DJI", 0, -0.01),
    ):
        payload["market"][key].update({"ticker": ticker, "valid": True, "daily_return": daily, "ytd_return": ytd})
    payload["market_context"] = {
        "russell2000": {"name": "Russell 2000", "valid": True, "close": 3045.48, "daily_return": 0.006},
        "vix": {"name": "VIX", "valid": True, "close": 14.55, "daily_return": -0.048},
        "dxy": {"name": "美元指数", "valid": True, "close": 100.01, "daily_return": 0.002},
        "us10y": {"name": "10Y 美债", "valid": True, "close": 4.68, "yield_change_bp": 8},
    }
    payload["market_breadth"] = {
        "stocks": {"advance_ratio": .519, "advancers": 261, "decliners": 230, "unchanged": 12, "unchanged_ratio": .024, "decline_ratio": .457, "valid_count": 503, "total_constituents": 503, "status": "ok"},
        "sectors": {"advancers": 6, "decliners": 5, "items": []},
        "health": {"valid": True, "level": "mixed", "label": "市场分化", "summary": "上涨与下跌分布较为均衡。", "divergence": None},
    }
    payload["news"] = [modern_news_item(rank, "新闻标题", f"新闻摘要 {rank}") for rank in range(1, 9)]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    assert 'class="summary-icon"' in html and "TODAY IN ONE LINE" not in html
    assert html.count('class="market-value">100.00</div>') == 3
    assert html.count('class="market-performance"') == 3
    assert 'class="market-daily stat"><strong class="up"><span class="arrow">▲</span>' in html
    assert 'class="market-daily stat"><strong class="down"><span class="arrow">▼</span>' in html
    assert 'class="market-daily stat"><strong class="flat"><span class="arrow">→</span>' in html
    assert html.count('class="icon-wrap"') == 4
    assert html.count('class="context-info"') == 4
    assert html.count('class="draw-layout"') == 2
    assert 'class="news-layout"' in html
    assert html.count('class="card news-card"') == 8
    assert 'class="nrow"' not in html
    assert '.page{width:min(1440px,calc(100%-34px))' in compact_css
    assert '.market-performance{display:flex;justify-content:space-between;align-items:baseline' in compact_css
    assert '.breadth-grid{display:grid;grid-template-columns:32%68%' in compact_css
    assert '.news-layout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}' in compact_css
    assert '.draw-layout{display:grid;grid-template-columns:100px1fr105px' in compact_css
    assert '@media(max-width:900px)' in compact_css


def test_drawdown_cards_render_three_action_zones_and_visual_progress(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    assert html.count('class="card draw-card draw-') == 2
    assert html.count('class="draw-layout"') == 2
    assert html.count('class="draw-index"') == 2
    assert html.count('class="draw-mid"') == 2
    assert html.count('class="cash"') == 2
    assert 'class="draw-status status-normal"' in html
    assert 'class="draw-status status-pending"' in html
    assert "未触发" in html
    assert "待执行" in html
    assert html.count("历史高点 ATH") == 2
    assert html.index("历史高点 ATH") < html.index("下一档触发值") < html.index('class="dtrack"') < html.index("可用金额")
    assert '<b style="left:60.0%"></b>' in html
    assert '<b style="left:85.0%"></b>' in html
    assert "第一档" in html
    assert "完成实际买入后" in html
    assert "font-variant-numeric:tabular-nums" in compact_css
    assert ".draw-layout{display:grid;grid-template-columns:100px1fr105px" in compact_css
    assert ".draw-mid{display:grid;grid-template-columns:repeat(3,1fr)" in compact_css
    assert ".dd{margin-top:5px;font-size:25px" in compact_css
    assert ".cashstrong{font-size:14px}" in compact_css
    assert "@media(max-width:900px)" in compact_css
    assert ".draw-grid" in css and "grid-template-columns:1fr" in compact_css
    assert '--bg:#f5f7f9' in compact_css.lower()
    assert '--surface:#fff' in compact_css.lower()
    assert '.market-value{' in compact_css and 'font-variant-numeric:tabular-nums' in compact_css
    assert 'font-family:Inter,Arial,"PingFangSC","MicrosoftYaHei",sans-serif' in compact_css
    assert 'h1{margin:0;font:50031px/1.05Georgia' in compact_css


def test_strategy_card_shows_hold_state_and_drawdown_rules_drawer(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500上涨0.72%。", "drivers": "市场关注利率预期。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
    }
    payload["drawdown"]["sp500"].update({
        "status": "normal", "pending_amount": 0, "pending_tiers": [], "executed_amount": 0,
        "tiers": {
            "tier_1": {"threshold": 0.10, "allocation": 0.20, "amount": 28000, "status": "not_triggered"},
            "tier_2": {"threshold": 0.15, "allocation": 0.30, "amount": 42000, "status": "not_triggered"},
            "tier_3": {"threshold": 0.20, "allocation": 0.30, "amount": 42000, "status": "not_triggered"},
            "tier_4": {"threshold": 0.25, "allocation": 0.20, "amount": 28000, "status": "not_triggered"},
        },
    })
    payload["drawdown"]["nasdaq100"].update({
        "status": "normal", "pending_amount": 0, "pending_tiers": [], "executed_amount": 0,
        "next_threshold": 0.15, "distance_to_next": 0.02,
        "tiers": {
            "tier_1": {"threshold": 0.15, "allocation": 0.20, "amount": 12000, "status": "not_triggered"},
            "tier_2": {"threshold": 0.20, "allocation": 0.30, "amount": 18000, "status": "not_triggered"},
            "tier_3": {"threshold": 0.30, "allocation": 0.30, "amount": 18000, "status": "not_triggered"},
            "tier_4": {"threshold": 0.40, "allocation": 0.20, "amount": 12000, "status": "not_triggered"},
        },
    })
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert 'class="strategy-action strategy-normal">正常定投</div>' in html
    assert 'class="strategy-sub">未触发回撤加仓</div>' in html
    assert 'class="strategy-row"><span>备用金状态</span><strong>保持不动</strong></div>' in html
    assert 'class="risk-normal">正常</strong>' in html
    assert 'id="drawdown-rules-open"' in html
    assert 'id="drawdown-rules-drawer"' in html
    assert "回撤加仓规则" in html
    assert "回撤 10%~15%" in html and "使用备用金 20%" in html
    assert "回撤 25%+" in html
    assert "回撤 15%~20%" in html and "回撤 40%+" in html
    assert "距离第一档还有" in html
    assert ".drawer{position:fixed;top:0;right:0" in compact_css
    assert ".drawer-backdrop{position:fixed;inset:0" in compact_css


def test_strategy_card_shows_pending_tier_allocation_and_amount(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500下跌。", "drivers": "市场关注利率预期。",
        "action": "已触发回撤加仓条件，等待人工确认。",
    }
    payload["drawdown"]["sp500"]["pending_tiers"] = []
    payload["drawdown"]["nasdaq100"]["pending_tiers"] = [
        {"id": "tier_1", "label": "第一档", "amount": 12000, "allocation": 0.20, "threshold": 0.15},
    ]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="strategy-action strategy-pending">加仓 20%</div>' in html
    assert 'class="strategy-sub">Nasdaq-100 已触发第一档，等待确认</div>' in html
    assert 'class="strategy-row"><span>备用金状态</span><strong>待执行 ¥12,000</strong></div>' in html
    assert 'class="risk-pending">已触发</strong>' in html
    assert "当前已进入第一档" in html
    assert "建议使用备用金比例" in html and "20%" in html
    assert "建议投入金额" in html and "¥12,000" in html


def test_strategy_card_pauses_when_market_data_invalid(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "行情数据暂不可用。", "drivers": "市场关注利率预期。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
    }
    payload["market_data_valid"] = False
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="strategy-action strategy-critical">暂停额外操作</div>' in html
    assert 'class="strategy-sub">行情数据校验失败，今日暂停回撤判断</div>' in html
    assert 'class="risk-critical">数据异常</strong>' in html


def test_drawdown_card_replaces_progress_with_last_tier_message(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["drawdown"]["sp500"].update({
        "status": "executed", "current_drawdown": 0.28, "next_threshold": None,
        "distance_to_next": None, "executed_amount": 140000, "remaining_amount": 0,
        "executed_tiers": [{"id": "tier_4", "label": "第四档", "amount": 28000}],
    })
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="draw-status status-executed"' in html
    assert 'class="dtrack complete" aria-label="已到最后一档"' in html


def test_desktop_density_uses_compact_spacing_without_shrinking_primary_numbers():
    root = __import__("pathlib").Path(__file__).parents[1]
    css = (root / "static" / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())

    assert ".section{padding:18px0" in compact_css
    assert ".section-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}" in compact_css
    assert ".market-card{padding:13px14px15px;display:flex;flex-direction:column;min-height:140px}" in compact_css
    assert ".market-value{margin-top:10px;font-size:31px;font-weight:760" in compact_css
    assert ".draw-card{padding:13px14px}" in compact_css
    assert ".news-card{min-height:290px;padding:15px15px13px" in compact_css
    assert ".news-summary{margin:0;color:var(--muted);font-size:12px;line-height:1.75}" in compact_css
    assert "@media(max-width:900px)" in compact_css


def _date_options(html):
    select = re.search(r'<select class="report-select".*?</select>', html, re.S).group(0)
    return re.findall(r'<option value="([^"]+)"( selected)?>([^<]+)</option>', select)


def test_single_report_header_select_only_contains_today(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert _date_options(html) == [("index.html", " selected", "今日 · 2026-08-12")]
    assert 'aria-label="选择日报日期"' in html
    assert "window.location.href = select.value" in html
    assert 'id="archive-loading"' in html


def test_report_select_limits_to_latest_seven_existing_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for day in range(1, 9):
        date = f"2026-08-{day:02d}"
        (reports / f"{date}.json").write_text(json.dumps(report(date)), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    options = _date_options((site / "index.html").read_text(encoding="utf-8"))
    assert len(options) == 7
    assert options[0] == ("index.html", " selected", "今日 · 2026-08-08")
    assert options[-1] == ("history/2026-08-02.html", "", "2026-08-02")
    assert all("2026-08-01" not in option for option in options)


def test_report_select_preserves_index_and_history_relative_paths(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for date in ("2026-08-10", "2026-08-11", "2026-08-12"):
        (reports / f"{date}.json").write_text(json.dumps(report(date)), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    index_html = (site / "index.html").read_text(encoding="utf-8")
    history_html = (site / "history" / "2026-08-11.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    header_html = index_html[index_html.index('<header>'):index_html.index("</header>")]

    assert "PERSONAL INVESTMENT DISCIPLINE" not in index_html
    assert 'class="header-meta"' in header_html
    assert 'class="header-tools"' in header_html
    assert 'class="report-select"' in header_html
    assert 'class="status status-ok"' in header_html
    assert _date_options(index_html) == [
        ("index.html", " selected", "今日 · 2026-08-12"),
        ("history/2026-08-11.html", "", "2026-08-11"),
        ("history/2026-08-10.html", "", "2026-08-10"),
    ]
    assert _date_options(history_html) == [
        ("../index.html", "", "今日 · 2026-08-12"),
        ("2026-08-11.html", " selected", "2026-08-11"),
        ("2026-08-10.html", "", "2026-08-10"),
    ]
    assert "header{display:grid;grid-template-columns:1frauto" in compact_css
    assert ".header-tools{display:flex;flex-direction:column;align-items:flex-end" in compact_css
    assert "select{height:31px;min-width:174px" in compact_css
    assert "h1{margin:0;font:50031px/1.05Georgia" in compact_css
    assert ".status{color:var(--green);font-size:10px" in compact_css
    assert "@media(max-width:600px)" in compact_css
    assert "header{grid-template-columns:1fr;gap:10px}" in compact_css
    assert ".header-tools{align-items:stretch}" in compact_css
    assert ".report-select{width:100%}" in compact_css


def test_market_context_renders_four_accessible_tooltips_and_vanilla_js(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_context"] = {
        "russell2000": {"name": "Russell 2000", "valid": True, "close": 3030, "daily_return": 0.01},
        "vix": {"name": "VIX", "valid": True, "close": 16.5, "daily_return": 0.10},
        "dxy": {"name": "美元指数", "valid": True, "close": 100.5, "daily_return": 0.005},
        "us10y": {"name": "10Y 美债", "valid": True, "close": 4.28, "yield_change_bp": 8},
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"
    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)
    html = (site / "index.html").read_text(encoding="utf-8")

    assert "市场环境" in html
    assert html.count('context-help-button"') == 4
    assert html.count('aria-expanded="false"') == 4
    for key, label in (("russell2000", "Russell 2000"), ("vix", "VIX"),
                       ("dxy", "美元指数"), ("us10y", "10Y 美债")):
        assert f'aria-controls="context-help-{key}"' in html
        assert f'aria-label="了解 {label} 的作用"' in html
        assert f'id="context-help-{key}"' in html
    assert "衡量美国小盘股整体表现" in html
    assert "未来约30天隐含波动率" in html
    assert "衡量美元相对一篮子主要货币" in html
    assert "全球资产定价的重要利率基准" in html
    assert "addEventListener" in html and "Escape" in html
    assert "React" not in html and "jquery" not in html.lower()


def test_old_report_without_market_context_still_renders(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"
    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "市场环境" not in html


def test_market_breadth_renders_health_stock_participation_sector_bars_and_tooltip(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    sector_values = [
        ("technology", "科技", "XLK", 0.012),
        ("financials", "金融", "XLF", -0.004),
        ("consumer_discretionary", "非必需消费", "XLY", 0.025),
        ("communication_services", "通信服务", "XLC", -0.020),
        ("industrials", "工业", "XLI", 0.003),
        ("health_care", "医疗保健", "XLV", -0.001),
        ("consumer_staples", "必需消费", "XLP", 0.0),
        ("energy", "能源", "XLE", 0.040),
        ("utilities", "公用事业", "XLU", -0.011),
        ("real_estate", "房地产", "XLRE", 0.008),
        ("materials", "材料", "XLB", -0.035),
    ]
    sector_items = [
        {"key": key, "name": name, "ticker": ticker, "valid": True,
         "daily_return": daily_return,
         "direction": "up" if daily_return > 0 else "down" if daily_return < 0 else "flat",
         "bar_strength": min(abs(daily_return) / 0.03, 1.0)}
        for key, name, ticker, daily_return in sector_values
    ]
    payload["market_breadth"] = {
        "market_date": "2026-08-11",
        "stocks": {"status": "ok", "total_constituents": 503, "valid_count": 497, "invalid_count": 6,
                   "advancers": 322, "decliners": 169, "unchanged": 6, "advance_ratio": 0.648,
                   "decline_ratio": 0.340, "unchanged_ratio": 0.012, "coverage_ratio": 0.988},
        "sectors": {"valid_count": 11, "advancers": 5, "decliners": 5, "unchanged": 1,
                    "advance_ratio": 5 / 11, "items": sector_items},
        "health": {"valid": True, "score": 0.68, "level": "healthy", "label": "市场健康",
                   "divergence": None, "summary": "多数股票与板块共同上涨，市场参与度良好。"},
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    assert "市场宽度" in html
    assert "市场健康" in html and "64.8%" in html
    assert 'class="card breadth-left"' in html
    assert 'class="breadth-copy"' in html
    assert 'class="badge health-level-healthy">市场健康</span>' in html
    assert "上涨" in html and "322" in html
    assert "下跌" in html and "169" in html
    assert "平盘" in html and "6" in html
    assert html.count('class="mini"') == 1
    assert 'class="g" style="width:64.8%"' in html
    assert 'class="n" style="width:1.2%"' in html
    assert 'class="r" style="width:34.0%"' in html
    assert html.count('class="sector-row"') == 11
    assert html.index("能源") < html.index("非必需消费") < html.index("科技") < html.index("材料")
    assert "能源 · +4.0%" in html and "当日排名：1 / 11" in html
    assert "材料 · -3.5%" in html and "当日排名：11 / 11" in html
    assert 'class="sector-cols"' in html
    assert html.count('class="sector-list"') == 2
    assert 'class="fill up" style="width:100.0%"' in html
    assert 'class="fill down" style="width:100.0%"' in html
    assert 'aria-controls="breadth-health-help"' in html
    assert "不作为独立买卖信号" in html
    assert "多数股票与板块共同上涨，市场参与度良好。" in html
    assert html.index("多数股票与板块共同上涨，市场参与度良好。") > html.index('class="card breadth-left"')
    assert ".breadth-grid{display:grid;grid-template-columns:32%68%;gap:12px;align-items:stretch}" in compact_css
    assert ".breadth-left{padding:13px14px;display:grid;grid-template-columns:1fr1.15fr" in compact_css
    assert ".sector-cols{display:grid;grid-template-columns:1fr1fr" in compact_css
    assert "grid-template-columns:78px1fr42px" in compact_css
    assert ".track{height:5px" in compact_css
    assert ".sector-name{font-size:9px" in compact_css
    assert ".sret{font-size:9px" in compact_css
    assert ".sector-row:hover.sector-tooltip" in compact_css
    assert "@media(prefers-reduced-motion:reduce)" in compact_css
    assert "@media(max-width:900px)" in compact_css
    assert ".market-grid,.context-grid,.breadth-grid,.draw-grid{grid-template-columns:1fr}" in compact_css
    assert ".sector-cols{grid-template-columns:1fr" in compact_css
    assert ".bbar{grid-column:1/2;display:flex;height:6px" in compact_css


def test_old_report_without_market_breadth_still_renders(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    assert "市场宽度" not in (site / "index.html").read_text(encoding="utf-8")


def test_information_hierarchy_maps_existing_signal_levels_to_context_classes(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_context"] = {
        "russell2000": {"name": "Russell 2000", "valid": True, "close": 3030, "daily_return": 0.003},
        "vix": {"name": "VIX", "valid": True, "close": 18, "daily_return": -0.21},
        "dxy": {"name": "美元指数", "valid": True, "close": 100.5, "daily_return": 0.001},
        "us10y": {"name": "10Y 美债", "valid": True, "close": 4.28, "yield_change_bp": -7},
    }
    payload["market_signals"] = {
        "signals": [
            {"key": "small_cap_relative", "level": "significant"},
            {"key": "vix_daily_return", "level": "strong"},
            {"key": "us10y_bp_change", "level": "significant"},
        ]
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert html.count('class="card context-card context-significant"') == 2
    assert html.count('class="card context-card context-strong"') == 1
    assert html.count('class="card context-card context-normal"') == 1
    assert 'class="context-change down">▼ -21.0%</span>' in html
    assert 'class="context-change down">▼ -7bp</span>' in html
    assert 'class="context-change up">▲ +0.1%</span>' in html


def test_information_hierarchy_styles_metrics_drawdown_states_and_news_ranks(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["drawdown"]["sp500"]["status"] = "near"
    payload["news"] = [
        modern_news_item(rank, summary="用于验证新闻视觉层级的摘要。")
        for rank in range(1, 6)
    ]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    assert 'class="card draw-card draw-near"' in html
    assert 'class="card draw-card draw-pending"' in html
    assert html.count('class="card news-card"') == 5
    assert 'class="nrow"' not in html
    assert ".statspan{color:var(--muted)}" in compact_css
    assert ".statstrong{font-variant-numeric:tabular-nums}" in compact_css
    assert ".context-change{display:block;margin-top:3px;padding:0;font-size:11.5px" in compact_css
    assert ".draw-status{font-size:8px" in compact_css
    for selector in (".status-normal", ".status-near", ".status-pending", ".status-executed"):
        assert selector in css
    assert ".draw-pending" in css
    assert ".news-card" in css and ".news-impact" not in css
    assert ".news-cardh3{margin:0010px;color:var(--ink);font-size:17px;line-height:1.45" in compact_css
    assert ".news-summary{margin:0;color:var(--muted);font-size:12px;line-height:1.75}" in compact_css
    assert ".news-footera{color:var(--blue);font-size:11px" in compact_css


def test_final_ui_diff_fix_uses_compact_warning_and_adaptive_news_grid(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["warnings"] = ["当前报告由离线测试数据生成，不代表真实市场行情或新闻。"]
    payload["news"] = [modern_news_item(rank) for rank in range(1, 4)]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    assert 'class="alert warning"' in html
    assert "🟠 部分数据源异常 · 当前报告由离线测试数据生成" in html
    assert html.count('class="card news-card"') == 3
    assert 'class="card news-list"' not in html
    assert 'class="news-layout"' in html
    assert ".alert{display:flex;gap:8px;align-items:center;margin-top:13px;padding:8px11px" in compact_css
    assert ".news-layout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}" in compact_css
    assert ".draw-layout{display:grid;grid-template-columns:100px1fr105px" in compact_css


def test_news_layout_keeps_one_two_and_eight_complete_cards(tmp_path):
    root = __import__("pathlib").Path(__file__).parents[1]
    for count in (1, 2, 8):
        reports = tmp_path / f"reports-{count}"
        reports.mkdir()
        payload = report("2026-08-12")
        payload["news"] = [modern_news_item(rank) for rank in range(1, count + 1)]
        (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
        site = tmp_path / f"site-{count}"
        render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)
        html = (site / "index.html").read_text(encoding="utf-8")
        assert 'class="news-layout"' in html
        assert html.count('class="card news-card"') == count
        assert 'class="nrow"' not in html


def test_reference_style_maps_breadth_drawdown_and_news_without_changing_data(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_breadth"] = {
        "stocks": {
            "advance_ratio": .648, "advancers": 322, "decliners": 169,
            "unchanged": 6, "unchanged_ratio": .012, "decline_ratio": .34,
            "valid_count": 497, "total_constituents": 503, "status": "ok",
        },
        "sectors": {"advancers": 8, "decliners": 3, "items": []},
        "health": {
            "valid": True, "level": "healthy", "label": "市场健康",
            "summary": "多数股票与板块共同上涨，市场参与度良好。", "divergence": None,
        },
    }
    payload["news"] = [modern_news_item(rank) for rank in range(1, 9)]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert 'class="card breadth-left"' in html
    assert 'class="breadth-copy"' in html
    assert html.index('class="ratio-label"') < html.index('class="breadth-copy"')
    assert ".breadth-left{padding:13px14px;display:grid;grid-template-columns:1fr1.15fr" in compact_css

    assert html.count('class="draw-mid"') == 2
    assert html.count('class="draw-layout"') == 2
    assert ".draw-layout{display:grid;grid-template-columns:100px1fr105px" in compact_css

    assert '<div class="news-meta"><span class="rank">01</span>' in html
    assert ".news-layout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}" in compact_css
    assert ".news-card{min-height:290px;padding:15px15px13px" in compact_css


def test_v5_visual_contract_is_the_rendered_dom_and_css_baseline(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500上涨。", "drivers": "市场关注利率。",
        "action": "备用金保持待命。",
    }
    payload["market_context"] = {
        "russell2000": {"name": "Russell 2000", "valid": True, "close": 3045.48, "daily_return": .006},
        "vix": {"name": "VIX", "valid": True, "close": 14.55, "daily_return": -.048},
        "dxy": {"name": "美元指数", "valid": True, "close": 100.01, "daily_return": .002},
        "us10y": {"name": "10Y 美债", "valid": True, "close": 4.68, "yield_change_bp": 8},
    }
    payload["market_breadth"] = {
        "stocks": {
            "advance_ratio": .519, "advancers": 261, "decliners": 241,
            "unchanged": 1, "unchanged_ratio": .002, "decline_ratio": .479,
            "valid_count": 503, "total_constituents": 503, "status": "ok",
        },
        "sectors": {"advancers": 1, "decliners": 1, "valid_count": 2, "items": [
            {"name": "科技", "valid": True, "daily_return": .01, "bar_strength": 1 / 3},
            {"name": "能源", "valid": True, "daily_return": -.01, "bar_strength": 1 / 3},
        ]},
        "health": {"valid": True, "level": "mixed", "label": "市场分化", "summary": "上涨股票略占优势，但板块扩散不足。", "divergence": None},
    }
    payload["news"] = [modern_news_item(rank) for rank in range(1, 9)]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"
    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    for class_name in (
        "page", "summary", "section-title", "section-left", "marker", "note",
        "market-performance", "icon-wrap", "context-info", "help", "breadth-grid",
        "breadth-left", "ratio-label", "ratio", "breadth-copy", "mini", "bbar",
        "formula", "sector-card", "sector-head", "track", "fill", "sret",
        "draw-grid", "draw-card", "draw-layout", "draw-index", "dd", "small",
        "draw-mid", "distance", "dtrack", "cash", "news-layout", "news-card",
        "news-meta", "rank", "news-category", "news-source", "news-summary",
        "news-tags", "news-footer", "news-focus",
    ):
        assert re.search(rf'class="[^"]*\b{re.escape(class_name)}\b', html)
    assert 'class="market-breadth-grid"' not in html
    assert 'class="drawdown-dashboard"' not in html
    assert 'class="news-dashboard' not in html
    for token in (
        "--bg:#f5f7f9", "--surface:#fff", "--ink:#14293a", "--muted:#6f7f8b",
        "--line:#dfe5e9", "--soft:#e9eef1", "--navy:#234f6d", "--blue:#4f8fbd",
        "--green:#24845b", "--red:#d34742", "--amber:#c78320",
        "--icon-bg:#eef3f6", "--icon-fg:#66869e",
    ):
        assert token in compact_css
    assert ".page{width:min(1440px,calc(100%-34px))" in compact_css
    assert ".section{padding:18px0" in compact_css
    assert ".breadth-grid{display:grid;grid-template-columns:32%68%;gap:12px;align-items:stretch}" in compact_css
    assert ".draw-layout{display:grid;grid-template-columns:100px1fr105px;gap:14px;align-items:center}" in compact_css
    assert ".news-layout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}" in compact_css
    assert "@media(max-width:900px)" in compact_css


def test_drawdown_index_titles_stay_aligned_without_wrapping_on_desktop(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert ".draw-layout{display:grid;grid-template-columns:100px1fr105px" in compact_css
    assert ".draw-index{width:100px;min-width:100px}" in compact_css
    assert "font-size:12px;white-space:nowrap" in compact_css
    assert "@media(max-width:900px)" in compact_css
    assert ".draw-index{width:auto;min-width:0}" in compact_css


def test_us_index_daily_change_has_primary_visual_weight(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert html.count('class="market-daily stat"') == 3
    assert html.count(">昨日</span>") == 3
    assert html.count('class="market-ytd stat"') == 3
    assert ".market-dailystrong{font-size:29px;font-weight:800" in compact_css
    assert ".market-dailyspan{font-size:11px" in compact_css
    assert ".market-ytdstrong{font-size:17px" in compact_css
    assert ".market-ytdspan{font-size:10px" in compact_css
    assert ".market-value{margin-top:10px;font-size:31px;font-weight:760" in compact_css


def test_us_index_cards_keep_price_and_performance_on_separate_rows(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market"]["sp500"]["daily_return"] = -0.01
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    assert html.count('class="market-value">') == 3
    assert html.count('class="market-performance"') == 3
    assert 'class="market-value down"' not in html
    assert 'class="market-daily stat"' in html
    assert 'class="market-daily stat"><strong class="down"><span class="arrow">▼</span>' in html
    assert ".market-card{padding:13px14px15px;display:flex;flex-direction:column;min-height:140px}" in compact_css
    assert ".market-value{margin-top:10px;font-size:31px;font-weight:760" in compact_css
    assert ".market-performance{display:flex;justify-content:space-between;align-items:baseline" in compact_css
    assert ".market-dailystrong{font-size:29px;font-weight:800" in compact_css
    assert ".market-ytdstrong{font-size:17px" in compact_css
    assert ".market-daily.arrow{margin-right:2px;font-size:21px;vertical-align:baseline;color:inherit}" in compact_css
