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
                "今日重要新闻",
                "市场环境",
                "市场广度",
                "长期投资与风险管理",
                "本报告仅供参考，不构成投资建议。",
                # A real URL slug legitimately containing "sk-" as part of an
                # ordinary word must not be flagged as a leaked secret.
                'href="https://example.com/anthropic-supply-chain-risk-label/"',
            )
        ),
        encoding="utf-8",
    )

    validate_generated(tmp_path)


def _write_report(reports_dir, report_date="2026-08-25"):
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{report_date}.json").write_text(
        json.dumps(
            {
                "report_date": report_date,
                "generated_at": f"{report_date} 10:00 CST",
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


_VALID_HTML = " ".join((
    "Daily Market Brief",
    "今日市场一句话",
    "今日重要新闻",
    "市场环境",
    "市场广度",
    "长期投资与风险管理",
    "本报告仅供参考，不构成投资建议。",
))


def test_smoke_rejects_an_actual_leaked_api_key_shaped_token(tmp_path):
    _write_report(tmp_path / "data" / "reports")
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        _VALID_HTML + " sk-abcDEF0123456789ghijKLMNOP", encoding="utf-8",
    )

    try:
        validate_generated(tmp_path)
        raised = False
    except SystemExit as exc:
        raised = True
        assert "secret" in str(exc)
    assert raised


def test_smoke_does_not_flag_ordinary_url_slugs_containing_sk_dash(tmp_path):
    _write_report(tmp_path / "data" / "reports")
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        _VALID_HTML + ' <a href="https://example.com/pentagons-supply-chain-risk-label/">link</a>',
        encoding="utf-8",
    )

    validate_generated(tmp_path)
