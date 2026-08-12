"""Daily report orchestration CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .deepseek_client import select_news, validate_selection
from .drawdown import summarize_index_state, update_drawdown_state
from .market import calculate_market_snapshot, fetch_market
from .news_dedupe import dedupe_candidates
from .report import load_reports, retain_latest_reports, write_report
from .renderer import render_site
from .rss_news import fetch_candidates


ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _offline_market(now: datetime, market_config: dict):
    raw = {
        "sp500": [("2025-12-31", 6800), ("2026-08-07", 7140), ("2026-08-10", 7200), ("2026-08-11", 7164)],
        "nasdaq100": [("2025-12-31", 25000), ("2026-08-07", 27000), ("2026-08-10", 27200), ("2026-08-11", 26880)],
        "dow": [("2025-12-31", 48000), ("2026-08-07", 50500), ("2026-08-10", 50700), ("2026-08-11", 50900)],
    }
    snapshots, histories = {}, {}
    for key, pairs in raw.items():
        history = [{"date": day, "close": close} for day, close in pairs]
        snapshot = calculate_market_snapshot(history, now)
        snapshot.update({"name": market_config[key]["name"], "ticker": market_config[key]["ticker"], "valid": True})
        snapshots[key], histories[key] = snapshot, history
    return snapshots, histories, ["当前报告由离线测试夹具生成，不代表真实市场行情或新闻。"]


def _recent_news(reports: list[dict]) -> list[dict]:
    selected = []
    for report in reports[:7]:
        for item in report.get("news", []):
            selected.append({"title": item.get("original_title", item.get("title_zh", "")), "url": item.get("url", "")})
    return selected


def _offline_news():
    candidate = {
        "candidate_id": "offline-fixture-1", "source": "Offline Fixture", "title": "Offline generation smoke test",
        "summary": "Deterministic candidate for local pipeline verification only.",
        "published_at": "2026-08-12T00:00:00+00:00", "url": "https://example.com/offline-fixture",
        "priority": "P0",
    }
    payload = {"news": [{"rank": 1, "candidate_id": candidate["candidate_id"], "category": "市场 / 宏观",
                          "title_zh": "离线生成流程验证", "summary_zh": "此条目仅用于验证本地日报从候选数据到静态页面的完整流程，不代表真实新闻或投资信息。"}]}
    return validate_selection(payload, [candidate])


def generate_daily_report(base_dir: Path = ROOT, offline_fixture: bool = False,
                          report_date: str = None) -> Path:
    base_dir = Path(base_dir)
    config_root = ROOT / "config"
    market_config = _load_yaml(config_root / "market.yaml")
    drawdown_rules = _load_yaml(config_root / "drawdown_rules.yaml")
    news_sources = _load_yaml(config_root / "news_sources.yaml")["sources"]
    if report_date:
        now = datetime.fromisoformat(f"{report_date}T10:00:00").replace(tzinfo=SHANGHAI)
    else:
        now = datetime.now(SHANGHAI)
        report_date = now.date().isoformat()

    if offline_fixture:
        snapshots, histories, market_warnings = _offline_market(now, market_config)
    else:
        snapshots, histories, market_warnings = fetch_market(market_config, now)
    market_data_valid = all(snapshots.get(key, {}).get("valid") for key in market_config)

    state_path = base_dir / "state" / "drawdown_state.json"
    history_path = base_dir / "state" / "drawdown_history.json"
    state = _read_json(state_path, {"version": 1, "indices": {}, "executions": []})
    history = _read_json(history_path, {"cycles": []})
    validity = {key: market_data_valid for key in ("sp500", "nasdaq100")}
    updated_state, archived = update_drawdown_state(state, histories, validity, drawdown_rules, snapshots, now)
    if market_data_valid:
        _write_json(state_path, updated_state)
        if archived:
            history.setdefault("cycles", []).extend(archived)
        _write_json(history_path, history)

    retained = load_reports(base_dir / "data" / "reports") if (base_dir / "data" / "reports").exists() else []
    warnings = list(market_warnings)
    news_degraded = False
    if offline_fixture:
        news = _offline_news()
    else:
        candidates, rss_warnings = fetch_candidates(news_sources, now)
        warnings.extend(rss_warnings)
        candidates = dedupe_candidates(candidates, _recent_news(retained))
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            news, ai_warning = [], "⚠️ 新闻 AI 处理暂时失败；RSS 数据已获取，等待下一次更新。原因：未配置 AI 凭据。"
        else:
            news, ai_warning = select_news(candidates, api_key, _recent_news(retained))
        if ai_warning:
            warnings.append(ai_warning)
            news_degraded = True

    if not market_data_valid:
        status, status_label = "critical", "🔴 行情数据校验失败"
    elif warnings:
        status, status_label = "partial", "🟠 部分数据源异常"
    else:
        status, status_label = "ok", "🟢 数据更新正常"
    valid_market_dates = [item["market_date"] for item in snapshots.values() if item.get("valid")]
    report = {
        "report_date": report_date,
        "generated_at": now.strftime("%Y-%m-%d %H:%M CST"),
        "market_date": max(valid_market_dates) if valid_market_dates else None,
        "status": status,
        "status_label": status_label,
        "market_data_valid": market_data_valid,
        "market": snapshots,
        "drawdown": {key: summarize_index_state(value) for key, value in updated_state.get("indices", {}).items()},
        "news": news,
        "news_degraded": news_degraded,
        "warnings": warnings,
    }
    reports_dir = base_dir / "data" / "reports"
    report_path = write_report(report, reports_dir)
    retain_latest_reports(reports_dir, 7)
    render_site(reports_dir, ROOT / "templates" / "report.html", ROOT / "static" / "style.css", base_dir / "site")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--report-date")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    path = generate_daily_report(args.base_dir, args.offline_fixture, args.report_date)
    print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
