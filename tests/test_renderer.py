import json
import re
from pathlib import Path

from src.renderer import render_site

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "templates" / "report.html"
STYLE = ROOT / "static" / "style.css"


def report(date):
    return {
        "report_date": date, "generated_at": f"{date} 10:00 CST", "market_date": "2026-08-11",
        "status": "ok", "status_label": "🟢 数据更新正常", "warnings": [],
        "market_data_valid": True,
        "market": {
            key: {"name": name, "ticker": ticker, "valid": True, "close": 100,
                  "daily_return": 0.01, "ytd_return": 0.10}
            for key, name, ticker in [
                ("sp500", "S&P 500", "^GSPC"), ("nasdaq100", "Nasdaq-100", "^NDX"), ("dow", "Dow Jones", "^DJI"),
            ]
        },
        "drawdown": {
            "sp500": {
                "name": "S&P 500", "status": "normal", "current_drawdown": 0.06,
                "ath": 8000, "ath_date": "2026-07-31", "next_threshold": 0.10,
                "distance_to_next": 0.04, "pool": 140000, "executed_amount": 0,
                "pending_amount": 0, "remaining_amount": 140000,
                "pending_tiers": [], "executed_tiers": [],
                "tiers": {
                    "tier_1": {"threshold": 0.10, "allocation": 0.20, "amount": 28000, "status": "not_triggered"},
                    "tier_2": {"threshold": 0.15, "allocation": 0.30, "amount": 42000, "status": "not_triggered"},
                },
            },
            "nasdaq100": {
                "name": "Nasdaq-100", "status": "pending", "current_drawdown": 0.17,
                "ath": 30000, "ath_date": "2026-07-31", "next_threshold": 0.20,
                "distance_to_next": 0.03, "pool": 60000, "executed_amount": 0,
                "pending_amount": 12000, "remaining_amount": 48000,
                "pending_tiers": [{"id": "tier_1", "label": "第一档", "amount": 12000, "allocation": 0.20, "threshold": 0.15}],
                "executed_tiers": [],
                "tiers": {
                    "tier_1": {"threshold": 0.15, "allocation": 0.20, "amount": 12000, "status": "pending"},
                    "tier_2": {"threshold": 0.20, "allocation": 0.30, "amount": 18000, "status": "not_triggered"},
                },
            },
        },
        "reserve": {"total": 200000, "remaining": 188000, "used": 12000, "ratio": 0.94},
        "discipline": {"monthly_dca": 10000, "holding_years_min": 20},
        "news": [], "news_degraded": False,
    }


def modern_news_item(rank, title_prefix="新闻", summary="摘要"):
    return {
        "rank": rank,
        "category": "美联储 / 利率",
        "source": "Fixture",
        "title_zh": f"{title_prefix} {rank}",
        "summary_zh": summary,
        "url": f"https://example.com/{rank}",
        "published_at": "2026-08-12T09:00:00+00:00",
        "focus": "离线夹具关注项",
        "tags": ["离线夹具", "渲染"],
        "investment_relevance_score": max(50, 92 - rank + 1),
        "selection_reason": "仅用于验证现代新闻字段渲染，不代表真实投资建议。",
    }


def render(reports_dir, site_dir):
    render_site(reports_dir, TEMPLATE, STYLE, site_dir)


def test_renders_one_news_item_per_report_entry(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["news"] = [modern_news_item(rank, "投资优先新闻", f"第 {rank} 条离线摘要") for rank in range(1, 9)]
    payload["news"][0]["source"] = "Persisted Source"
    payload["news"][0]["url"] = "https://persisted.example/news-1"
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert html.count('class="news-item"') == 8
    assert "今日重要新闻" in html
    assert "（按重要性排序）" in html
    assert html.count('class="chips"') == 8
    assert 'href="https://persisted.example/news-1"' in html


def test_news_item_shows_rank_category_title_desc_and_chips_without_fabricated_impact(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["generated_at"] = "2026-08-12 12:00 CST"
    payload["news"] = [{
        "rank": 1,
        "category": "利率政策",
        "source": "Legacy Source",
        "title_zh": "旧版新闻标题",
        "summary_zh": "旧版报告保留的摘要。",
        "url": "https://legacy.example/report",
        "published_at": "2026-08-12T01:00:00+00:00",
    }]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert '<div class="rank">1</div>' in html
    assert '<div class="category">利率政策</div>' in html
    assert "旧版新闻标题" in html
    assert "旧版报告保留的摘要。" in html
    assert '<div class="time">3小时前</div>' in html
    assert 'class="chips"' not in html
    # No fabricated 利好/利空 verdict is invented when there is no real sentiment signal.
    assert "impact-up" not in html and "impact-down" not in html
    assert '<div class="impact-label">市场影响：</div>' in html
    assert 'href="https://legacy.example/report"' in html


def test_news_empty_and_degraded_states_render_without_crashing(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["news"] = []
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"
    render(reports, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "暂无足够重要的有效新闻" in html

    payload["news_degraded"] = True
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    render(reports, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "新闻 AI 处理暂时失败" in html


def test_renders_index_and_history_pages(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for date in ("2026-08-11", "2026-08-12"):
        (reports / f"{date}.json").write_text(json.dumps(report(date)), encoding="utf-8")
    site = tmp_path / "site"
    render(reports, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "Daily Market Brief" in html
    assert "报告日期：2026-08-12" in html
    assert (site / "history" / "2026-08-11.html").exists()


def test_hero_shows_only_sp500_and_nasdaq_with_summary_and_strategy(tmp_path):
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
    site = tmp_path / "site"

    render(reports, site)

    index_html = (site / "index.html").read_text(encoding="utf-8")
    legacy_html = (site / "history" / "2026-08-11.html").read_text(encoding="utf-8")
    assert index_html.count('class="index-card"') == 2
    assert "Dow Jones" not in index_html.split('class="summary-text"')[0]
    assert '今日市场一句话' in index_html
    assert "标普500小幅上涨" in index_html
    assert 'class="card strategy"' in index_html
    assert index_html.index("今日市场一句话") > index_html.index("</header>")
    assert index_html.index("今日市场一句话") < index_html.index("今日重要新闻")
    assert "今日市场一句话" not in legacy_html


def test_index_card_shows_value_signed_change_and_ytd(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {"market": "示例。", "drivers": "示例。", "action": "示例。"}
    payload["market"]["sp500"].update({"daily_return": 0.0072, "ytd_return": 0.129, "close": 7730.99})
    payload["market"]["nasdaq100"].update({"daily_return": -0.014, "ytd_return": 0.174, "close": 29641.56})
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="index-value">7,730.99</div>' in html
    assert 'class="index-value">29,641.56</div>' in html
    assert 'class="day-change up sp500-day-change">▲ +0.7%</span>' in html
    assert 'class="day-change down sp500-day-change">▼ -1.4%</span>' in html
    assert 'class="ytd up">+12.9% <span class="small">YTD</span></div>' in html


def test_index_card_omits_sparkline_gracefully_when_history_unavailable(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {"market": "示例。", "drivers": "示例。", "action": "示例。"}
    # No "sparkline" key present, mirroring an older report schema.
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="spark-empty"' in html
    assert '<svg viewBox="0 0 420 52"' not in html


def test_index_card_renders_real_sparkline_path_when_present(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {"market": "示例。", "drivers": "示例。", "action": "示例。"}
    payload["market"]["sp500"]["sparkline"] = {"line": "M0,10 L420,4", "area": "M0,10 L420,4 L420,52 L0,52 Z"}
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'id="spark-sp500"' in html
    assert 'd="M0,10 L420,4"' in html
    assert 'd="M0,10 L420,4 L420,52 L0,52 Z"' in html


def test_strategy_card_shows_hold_state_and_drawdown_rules_drawer(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500上涨0.72%。", "drivers": "市场关注利率预期。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
    }
    payload["drawdown"]["sp500"].update({"status": "normal", "pending_amount": 0, "pending_tiers": [], "executed_amount": 0})
    payload["drawdown"]["nasdaq100"].update({
        "status": "normal", "pending_amount": 0, "pending_tiers": [], "executed_amount": 0,
        "next_threshold": 0.15, "distance_to_next": 0.02,
    })
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert '<strong class="strategy-normal">正常定投</strong>' in html
    assert '<span>未触发回撤加仓</span>' in html
    assert '<span class="label">备用金状态</span><span class="value">保持不动</span>' in html
    assert '<span class="label">当前风险状态</span><span class="value good">正常</span>' in html
    assert 'id="drawdown-rules-open"' in html
    assert 'id="drawdown-rules-drawer"' in html
    assert "回撤加仓规则" in html
    assert "回撤 10%~15%" in html and "使用备用金 20%" in html
    assert "距离第一档还有" in html


def test_strategy_card_shows_pending_tier_allocation_and_amount(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {
        "market": "标普500下跌。", "drivers": "市场关注利率预期。",
        "action": "已触发回撤加仓条件，等待人工确认。",
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert '<strong class="strategy-pending">加仓 20%</strong>' in html
    assert '<span>Nasdaq-100 已触发第一档，等待确认</span>' in html
    assert '<span class="label">备用金状态</span><span class="value">待执行 ¥12,000</span>' in html
    assert '<span class="label">当前风险状态</span><span class="value">已触发</span>' in html
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
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert '<strong class="strategy-critical">暂停额外操作</strong>' in html
    assert '<span>行情数据校验失败，今日暂停回撤判断</span>' in html
    assert '<span class="label">当前风险状态</span><span class="value critical">数据异常</span>' in html


def test_single_report_header_select_only_contains_today(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    options = _date_options(html)
    assert options == [("index.html", " selected", "今日 · 2026-08-12")]
    assert 'aria-label="选择日报日期"' in html
    assert 'class="datebox"' in html
    assert "window.location.href = select.value" in html
    assert 'id="archive-loading"' in html


def _date_options(html):
    select = re.search(r'<select class="report-select".*?</select>', html, re.S).group(0)
    return re.findall(r'<option value="([^"]+)"( selected)?>([^<]+)</option>', select)


def test_report_select_limits_to_latest_seven_existing_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for day in range(1, 9):
        date = f"2026-08-{day:02d}"
        (reports / f"{date}.json").write_text(json.dumps(report(date)), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

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
    site = tmp_path / "site"

    render(reports, site)

    index_html = (site / "index.html").read_text(encoding="utf-8")
    history_html = (site / "history" / "2026-08-11.html").read_text(encoding="utf-8")
    assert 'class="status status-ok"' in index_html
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


def test_status_label_does_not_double_up_bullet_glyph(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["status"] = "partial"
    payload["status_label"] = "🟠 部分数据源异常"
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert '<div class="status status-partial">● 部分数据源异常</div>' in html
    assert "🟠" not in html


def test_market_context_metric_grid_covers_five_tiles_in_order(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_context"] = {
        "vix": {"name": "VIX 恐慌指数", "valid": True, "close": 14.51, "daily_return": -0.046},
        "us10y": {"name": "10Y 美国收益率", "valid": True, "close": 4.67, "yield_change_bp": 0.8},
        "dxy": {"name": "美元指数 (DXY)", "valid": True, "close": 99.15, "daily_return": -0.0002},
        "gold": {"name": "黄金 (COMEX)", "valid": True, "close": 1931.45, "daily_return": 0.003},
        "wti": {"name": "原油 (WTI)", "valid": True, "close": 78.32, "daily_return": -0.012},
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "市场环境" in html
    assert html.count('class="card metric"') == 5
    order = ["VIX 恐慌指数", "10Y 美国收益率", "美元指数 (DXY)", "黄金 (COMEX)", "原油 (WTI)"]
    positions = [html.index(name) for name in order]
    assert positions == sorted(positions)
    assert '<div class="metric-value">14.51</div>' in html
    assert '<div class="metric-change down">▼ -4.6%</div>' in html
    assert '<div class="metric-value">4.67%</div>' in html
    assert '<div class="metric-change up">▲ +1bp</div>' in html


def test_market_context_metric_shows_unavailable_state_without_crashing(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_context"] = {
        "vix": {"name": "VIX 恐慌指数", "valid": False},
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"
    render(reports, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "数据暂不可用" in html


def test_old_report_without_market_context_still_renders(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    site = tmp_path / "site"
    render(reports, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "市场环境" not in html


def test_market_breadth_renders_advance_ratio_sectors_and_honest_sentiment_placeholder(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    sector_values = [
        ("technology", "科技", 0.012), ("financials", "金融", -0.004),
        ("consumer_discretionary", "非必需消费", 0.025), ("energy", "能源", 0.040),
        ("materials", "材料", -0.035),
    ]
    sector_items = [
        {"key": key, "name": name, "valid": True, "daily_return": daily,
         "bar_strength": min(abs(daily) / 0.03, 1.0)}
        for key, name, daily in sector_values
    ]
    payload["market_breadth"] = {
        "stocks": {"advance_ratio": 0.702, "advancers": 5128, "decliners": 2171, "unchanged": 12,
                   "unchanged_ratio": 0.001, "decline_ratio": 0.297, "valid_count": 503,
                   "total_constituents": 505, "status": "ok"},
        "sectors": {"advancers": 3, "decliners": 2, "items": sector_items},
        "health": {"valid": True, "level": "healthy", "label": "市场健康",
                   "summary": "多数股票与板块共同上涨。", "divergence": None},
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "市场广度" in html
    assert '<div class="big-green">70.2%</div>' in html
    assert "5128" in html and "2171" in html
    assert html.count('class="sector-row"') == 5
    # Sectors sorted descending by daily_return: 能源(+4.0%) first, 材料(-3.5%) last.
    assert html.index("能源") < html.index("非必需消费") < html.index("科技") < html.index("材料")
    assert 'class="sector-fill red"' in html
    # No market_sentiment payload provided: honest placeholder only, no fabricated score.
    assert '市场情绪' in html
    assert '<strong>—</strong><span>数据不足，暂无法评分</span>' in html
    assert "62" not in html.split("市场情绪")[1].split("</section>")[0] if "市场情绪" in html else True


def test_market_sentiment_score_renders_real_composite_and_label(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_breadth"] = {
        "stocks": {"advance_ratio": 0.6, "advancers": 300, "decliners": 200, "unchanged": 5,
                   "unchanged_ratio": 0.01, "decline_ratio": 0.4, "valid_count": 500,
                   "total_constituents": 505, "status": "ok"},
        "sectors": {"advancers": 6, "decliners": 5, "items": []},
        "health": {"valid": True, "level": "healthy", "label": "市场健康",
                   "summary": "多数股票与板块共同上涨。", "divergence": None},
    }
    payload["market_sentiment"] = {
        "vix_score": 70.0, "breadth_score": 60.0, "momentum_score": 55.0,
        "risk_appetite_score": 65.0, "market_sentiment_score": 63.3,
        "market_sentiment_label": "偏乐观",
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="gauge-text"><strong>63</strong><span>偏乐观</span>' in html
    assert 'gauge-text unavailable' not in html
    assert "#26a467" in html.split("市场情绪")[1].split("</section>")[0]


def test_market_breadth_unavailable_shows_graceful_empty_state(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_breadth"] = {
        "stocks": {"advance_ratio": None}, "sectors": {"items": []}, "health": {"valid": False},
    }
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"
    render(reports, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "市场广度数据暂不可用" in html


def test_old_report_without_market_breadth_still_renders(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    assert "市场广度" not in (site / "index.html").read_text(encoding="utf-8")


def test_risk_management_section_shows_drawdown_reserve_and_discipline(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "长期投资与风险管理" in html
    assert html.count('class="risk-value up">-') == 2
    assert "-6.0%" in html and "-17.0%" in html
    assert "回撤 10%~15%" in html and "使用备用金 20%" in html
    assert "查看完整规则 →" in html
    assert 'data-pct="94%"' in html
    assert "--reserve-deg:338.4deg" in html
    assert "¥188,000" in html and "¥200,000" in html
    assert "¥10,000" in html
    assert "20年以上" in html


def test_risk_management_section_pauses_when_market_data_invalid(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_data_valid"] = False
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "行情数据校验失败" in html
    assert "现有档位状态未被修改" in html


def test_reserve_and_discipline_fall_back_gracefully_when_absent(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    del payload["reserve"]
    del payload["discipline"]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "备用金使用情况" in html
    assert "投资纪律提醒" in html


def test_footer_renders_two_spans_with_generated_at(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-12.json").write_text(json.dumps(report("2026-08-12")), encoding="utf-8")
    site = tmp_path / "site"

    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    footer_html = html[html.index("<footer>"):html.index("</footer>")]
    assert footer_html.count("<span>") == 2
    assert "2026-08-12 10:00 CST" in footer_html


def test_v5_visual_contract_is_the_rendered_dom_and_css_baseline(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = report("2026-08-12")
    payload["market_summary"] = {"market": "标普500上涨。", "drivers": "市场关注利率。", "action": "备用金保持待命。"}
    payload["market_context"] = {
        "vix": {"name": "VIX", "valid": True, "close": 14.55, "daily_return": -0.048},
        "us10y": {"name": "10Y 美债", "valid": True, "close": 4.68, "yield_change_bp": 8},
        "dxy": {"name": "美元指数", "valid": True, "close": 100.01, "daily_return": 0.002},
        "gold": {"name": "黄金", "valid": True, "close": 1931.45, "daily_return": 0.003},
        "wti": {"name": "原油", "valid": True, "close": 78.32, "daily_return": -0.012},
    }
    payload["market_breadth"] = {
        "stocks": {"advance_ratio": 0.519, "advancers": 261, "decliners": 241, "unchanged": 1,
                   "unchanged_ratio": 0.002, "decline_ratio": 0.479, "valid_count": 503,
                   "total_constituents": 503, "status": "ok"},
        "sectors": {"advancers": 1, "decliners": 1, "items": [
            {"name": "科技", "valid": True, "daily_return": 0.01, "bar_strength": 1 / 3},
            {"name": "能源", "valid": True, "daily_return": -0.01, "bar_strength": 1 / 3},
        ]},
        "health": {"valid": True, "level": "mixed", "label": "市场分化", "summary": "上涨股票略占优势。", "divergence": None},
    }
    payload["news"] = [modern_news_item(rank) for rank in range(1, 9)]
    (reports / "2026-08-12.json").write_text(json.dumps(payload), encoding="utf-8")
    site = tmp_path / "site"
    render(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    compact_css = "".join((site / "style.css").read_text(encoding="utf-8").split())
    for class_name in (
        "page", "topbar", "title", "meta", "datebox", "status", "hero", "market-summary",
        "indices", "index-card", "index-value", "index-bottom", "day-change", "ytd", "spark",
        "summary-text", "strategy", "strategy-main", "strategy-tag", "strategy-row",
        "strategy-link", "section-head", "section-title", "bar", "news-item", "rank",
        "category", "time", "news-title", "news-desc", "impact", "metric-grid", "metric",
        "metric-value", "metric-change", "breadth", "big-green", "sector-row", "sector-fill",
        "gauge", "risk-grid", "risk-card", "risk-value", "reserve", "donut", "checks",
        "check-row", "check-ok",
    ):
        assert re.search(rf'class="[^"]*\b{re.escape(class_name)}\b', html), class_name
    for token in (
        "--bg:#f5f7fa", "--card:#fff", "--line:#dbe3ea", "--text:#14314a", "--muted:#75899b",
        "--blue:#2f6fb3", "--green:#15975a", "--red:#d94a44",
    ):
        assert token in compact_css
    assert ".title{margin:0;font-family:Georgia" in compact_css and "font-size:42px" in compact_css
    assert ".index-value{margin-top:8px;font-size:34px" in compact_css
    assert ".sp500-day-change{font-size:29px}" in compact_css
    assert ".day-change{font-weight:800;font-size:22px}" in compact_css
    assert ".strategy-mainstrong{" in compact_css and "font-size:34px" in compact_css
    assert ".metric-value{margin-top:4px;font-size:24px" in compact_css
    assert ".section-title{display:flex;align-items:center;gap:10px;font-size:20px" in compact_css
    assert "@media(max-width:1100px)" in compact_css
    assert "@media(max-width:760px)" in compact_css
