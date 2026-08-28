"""Render retained report JSON into the GitHub Pages artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _percent(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _money(value):
    return f"¥{int(value):,}" if value is not None else "—"


def _number(value):
    return f"{float(value):,.2f}" if value is not None else "—"


def _relative_time(published_at, generated_at):
    """Render a Chinese relative-time string ("3小时前") from real timestamps.

    Falls back to None (caller should show an absolute date instead) when either
    timestamp is missing or unparsable, rather than fabricating an elapsed time.
    """
    if not published_at or not generated_at:
        return None
    try:
        published = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        generated_text = str(generated_at).replace(" CST", "").strip()
        generated = datetime.strptime(generated_text, "%Y-%m-%d %H:%M").replace(tzinfo=SHANGHAI)
    except (ValueError, TypeError):
        return None
    seconds = (generated - published).total_seconds()
    if seconds < 0:
        return None
    minutes = seconds / 60
    if minutes < 60:
        return f"{max(int(minutes), 1)}分钟前"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}小时前"
    days = hours / 24
    return f"{int(days)}天前"


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
    env.filters.update(percent=_percent, money=_money, number=_number, relative_time=_relative_time)
    template = env.get_template(template_path.name)
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(reports_dir.glob("????-??-??.json"), reverse=True)]
    if not reports:
        raise ValueError("没有可渲染的日报 JSON。")
    nav = [{"date": item["report_date"]} for item in reports[:7]]
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
