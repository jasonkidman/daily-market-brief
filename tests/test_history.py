import json

from src.report import retain_latest_reports, write_report


def test_eighth_report_removes_oldest_json(tmp_path):
    for day in range(1, 9):
        write_report({"report_date": f"2026-08-{day:02d}"}, tmp_path)
    removed = retain_latest_reports(tmp_path, keep=7)
    assert [p.name for p in removed] == ["2026-08-01.json"]
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"2026-08-{day:02d}.json" for day in range(2, 9)]


def test_report_json_is_utf8_and_readable(tmp_path):
    path = write_report({"report_date": "2026-08-12", "status": "正常"}, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "正常"
