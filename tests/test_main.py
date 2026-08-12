import json

from src.main import generate_daily_report


def test_offline_fixture_runs_complete_pipeline(tmp_path):
    result = generate_daily_report(base_dir=tmp_path, offline_fixture=True, report_date="2026-08-12")
    report_path = tmp_path / "data" / "reports" / "2026-08-12.json"
    index_path = tmp_path / "site" / "index.html"
    assert report_path.exists()
    assert index_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_date"] == "2026-08-12"
    assert report["market_date"] == "2026-08-11"
    assert set(report["market"]) == {"sp500", "nasdaq100", "dow"}
    assert "Daily Market Brief" in index_path.read_text(encoding="utf-8")
    assert result == report_path
