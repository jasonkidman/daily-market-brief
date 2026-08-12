import json

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


def test_drawdown_cards_render_three_action_layers_and_visual_progress(tmp_path):
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
    assert html.count('class="drawdown-card"') == 2
    assert html.count('class="drawdown-status-layer"') == 2
    assert html.count('class="drawdown-progress-layer"') == 2
    assert html.count('class="capital-layer"') == 2
    assert 'class="status-indicator status-normal"' in html
    assert 'class="status-indicator status-pending"' in html
    assert "未触发" in html
    assert "待执行" in html
    assert 'class="progress-marker" style="left: 60.0%"' in html
    assert 'class="progress-marker" style="left: 85.0%"' in html
    assert "第一档" in html
    assert "完成实际买入后" in html
    assert "font-variant-numeric:tabular-nums" in compact_css
    assert ".capital-layer{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))" in compact_css
    assert "@media(max-width:760px)" in compact_css
    assert ".drawdown-grid{grid-template-columns:1fr}" in compact_css
    assert '--paper:#f4f6f8' in compact_css.lower()
    assert '--surface:#fcfdfe' in compact_css.lower()
    assert '.close{' in compact_css
    assert 'font-family:Inter,Arial,Helvetica,sans-serif' in compact_css
    assert '.close' in css and 'font-variant-numeric: tabular-nums' in css
    assert '.masthead h1' in css and 'font-size: 36px' in css
    assert '@media(max-width:760px)' in compact_css and 'font-size:27px' in compact_css


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
    assert 'class="status-indicator status-executed"' in html
    assert "已到最后一档" in html


def test_desktop_density_uses_compact_spacing_without_shrinking_primary_numbers():
    root = __import__("pathlib").Path(__file__).parents[1]
    css = (root / "static" / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())

    assert ".section{padding:30px0" in compact_css
    assert ".section-kicker{margin-bottom:5px" in compact_css
    assert ".section-heading" in css and "margin-bottom: 15px" in css
    assert ".market-card{padding:18px" in compact_css
    assert ".close{margin:14px011px" in compact_css
    assert "font-size:2.2rem" in compact_css
    assert ".drawdown-status-layer" in css and "padding: 18px 20px 16px" in css
    assert ".drawdown-progress{margin-top:12px" in compact_css
    assert ".capital-layer" in css and "padding: 13px 20px" in css
    assert ".news-item" in css and "padding: 17px 20px" in css
    assert ".news-itemp" in compact_css and "line-height:1.55" in compact_css
    assert ".section-heading{display:flex;justify-content:space-between" in compact_css
    assert "@media(max-width:760px)" in compact_css
    assert ".section{padding:32px0" in compact_css


def test_compact_header_contains_history_navigation_status_and_preserves_links(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for date in ("2026-08-11", "2026-08-12"):
        (reports / f"{date}.json").write_text(json.dumps(report(date)), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).parents[1]
    site = tmp_path / "site"

    render_site(reports, root / "templates" / "report.html", root / "static" / "style.css", site)

    index_html = (site / "index.html").read_text(encoding="utf-8")
    history_html = (site / "history" / "2026-08-11.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    compact_css = "".join(css.split())
    header_html = index_html[index_html.index('<header class="masthead">'):index_html.index("</header>")]

    assert "PERSONAL INVESTMENT DISCIPLINE" not in index_html
    assert 'class="masthead-copy"' in header_html
    assert 'class="masthead-tools"' in header_html
    assert 'class="history-nav"' in header_html
    assert 'class="health health-ok"' in header_html
    assert 'href="index.html"' in header_html
    assert 'href="history/2026-08-11.html"' in header_html
    assert 'href="../index.html"' in history_html
    assert 'href="2026-08-11.html"' in history_html
    assert ".masthead{display:grid;grid-template-columns:minmax(0,1fr)auto" in compact_css
    assert ".masthead-tools{display:flex;flex-direction:column;align-items:flex-end" in compact_css
    assert ".history-nava" in compact_css and "height:32px" in compact_css
    assert ".mastheadh1" in compact_css and "font-size:36px" in compact_css and "line-height:1.12" in compact_css
    assert ".health" in compact_css and "font-size:.82rem" in compact_css and "padding:6px10px" in compact_css
    assert "@media(max-width:760px)" in compact_css
    assert ".masthead{grid-template-columns:1fr" in compact_css
    assert ".masthead-tools{align-items:stretch" in compact_css
    assert ".history-nav{max-width:100%;overflow-x:auto" in compact_css
