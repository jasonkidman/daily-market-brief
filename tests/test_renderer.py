import json

from src.renderer import render_site


def report(date):
    return {
        "report_date": date, "generated_at": f"{date} 10:00 CST", "market_date": "2026-08-11",
        "status": "ok", "status_label": "🟢 数据更新正常", "warnings": [],
        "market_data_valid": True,
        "market": {key: {"name": name, "close": 100, "daily_return": 0.01, "ytd_return": 0.10}
                   for key, name in [("sp500", "S&P 500"), ("nasdaq100", "Nasdaq-100"), ("dow", "Dow Jones")]},
        "drawdown": {}, "news": [], "news_degraded": False,
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
