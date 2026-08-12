"""Render retained report JSON into the GitHub Pages artifact."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _percent(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _money(value):
    return f"¥{int(value):,}" if value is not None else "—"


def _number(value):
    return f"{float(value):,.2f}" if value is not None else "—"


def render_site(reports_dir: Path, template_path: Path, style_path: Path, site_dir: Path) -> None:
    reports_dir, template_path, style_path, site_dir = map(Path, (reports_dir, template_path, style_path, site_dir))
    site_dir.mkdir(parents=True, exist_ok=True)
    history_dir = site_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    for stale in history_dir.glob("????-??-??.html"):
        stale.unlink()
    shutil.copyfile(style_path, site_dir / "style.css")
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(percent=_percent, money=_money, number=_number)
    template = env.get_template(template_path.name)
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(reports_dir.glob("????-??-??.json"), reverse=True)]
    if not reports:
        raise ValueError("没有可渲染的日报 JSON。")
    nav = [{"date": item["report_date"], "label": "今日" if i == 0 else item["report_date"][5:]}
           for i, item in enumerate(reports)]
    latest = reports[0]
    (site_dir / "index.html").write_text(
        template.render(report=latest, nav=nav, is_index=True, asset_prefix="", current_date=latest["report_date"]),
        encoding="utf-8",
    )
    for report in reports:
        (history_dir / f"{report['report_date']}.html").write_text(
            template.render(report=report, nav=nav, is_index=False, asset_prefix="../", current_date=report["report_date"]),
            encoding="utf-8",
        )
