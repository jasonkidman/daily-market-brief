import json

from src.smoke import validate_generated


def test_smoke_accepts_current_important_news_heading(tmp_path):
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True)
    (reports / "2026-08-25.json").write_text(
        json.dumps(
            {
                "report_date": "2026-08-25",
                "generated_at": "2026-08-25 10:00 CST",
                "market_date": "2026-08-24",
                "status": "ok",
                "market": {},
                "drawdown": {},
                "news": [],
                "market_summary": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        " ".join(
            (
                "Daily Market Brief",
                "今日市场一句话",
                "美国主要指数",
                "回撤加仓预警",
                "今日重要新闻",
                "数据仅用于信息与个人投资纪律辅助，不构成投资建议。",
            )
        ),
        encoding="utf-8",
    )

    validate_generated(tmp_path)
