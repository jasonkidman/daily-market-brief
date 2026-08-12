"""Daily report persistence and seven-calendar-day retention."""

from __future__ import annotations

import json
from pathlib import Path


def write_report(report: dict, reports_dir: Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{report['report_date']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def retain_latest_reports(reports_dir: Path, keep: int = 7) -> list[Path]:
    paths = sorted(Path(reports_dir).glob("????-??-??.json"))
    removed = paths[:-keep] if keep > 0 else paths
    for path in removed:
        path.unlink()
    return removed


def load_reports(reports_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(Path(reports_dir).glob("????-??-??.json"), reverse=True):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports
