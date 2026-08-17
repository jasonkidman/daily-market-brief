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
from .market import calculate_context_snapshot, calculate_market_snapshot, fetch_market, fetch_market_context
from .market_breadth import build_market_breadth, build_offline_market_breadth, unavailable_market_breadth
from .market_health import build_market_breadth_text
from .market_signals import build_market_context_for_ai, calculate_market_signals
from .market_summary import derive_portfolio_action, generate_market_summary
from .news_dedupe import dedupe_candidates
from .news_events import (
    build_event_representatives,
    cluster_news_events,
    event_selection_candidates,
    validate_event_clusters,
)
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


def _offline_market(now: datetime, core_config: dict, context_config: dict):
    core_raw = {
        "sp500": [("2025-12-31", 6800), ("2026-08-07", 7140), ("2026-08-10", 7200), ("2026-08-11", 7164)],
        "nasdaq100": [("2025-12-31", 25000), ("2026-08-07", 27000), ("2026-08-10", 27200), ("2026-08-11", 26880)],
        "dow": [("2025-12-31", 48000), ("2026-08-07", 50500), ("2026-08-10", 50700), ("2026-08-11", 50900)],
    }
    context_raw = {
        "russell2000": [("2026-08-10", 3000), ("2026-08-11", 3030)],
        "vix": [("2026-08-10", 15), ("2026-08-11", 16.5)],
        "dxy": [("2026-08-10", 100), ("2026-08-11", 100.5)],
        "us10y": [("2026-08-10", 4.20), ("2026-08-11", 4.28)],
    }
    snapshots, histories = {}, {}
    for key, pairs in core_raw.items():
        history = [{"date": day, "close": close} for day, close in pairs]
        snapshot = calculate_market_snapshot(history, now)
        snapshot.update({"name": core_config[key]["name"], "ticker": core_config[key]["ticker"], "valid": True})
        snapshots[key], histories[key] = snapshot, history
    context_snapshots = {}
    for key, pairs in context_raw.items():
        rows = [{"date": day, "close": close} for day, close in pairs]
        snapshot = calculate_context_snapshot(rows, now, is_yield=(key == "us10y"))
        snapshot.update({
            "name": context_config[key]["name"], "ticker": context_config[key]["ticker"], "valid": True,
        })
        context_snapshots[key] = snapshot
    return snapshots, histories, context_snapshots, [
        "当前报告由离线测试夹具生成，不代表真实市场行情或新闻。"
    ]


def assess_market_validity(core: dict, context: dict) -> dict:
    drawdown_by_index = {
        "sp500": bool(core.get("sp500", {}).get("valid")),
        "nasdaq100": bool(core.get("nasdaq100", {}).get("valid")),
    }
    return {
        "core_market_valid": all(core.get(key, {}).get("valid") for key in ("sp500", "nasdaq100", "dow")),
        "context_market_valid": all(
            context.get(key, {}).get("valid") for key in ("russell2000", "vix", "dxy", "us10y")
        ),
        "context_any_valid": any(item.get("valid") for item in context.values()),
        "drawdown_by_index": drawdown_by_index,
        "drawdown_market_valid": all(drawdown_by_index.values()),
    }


def _log_market_status(core: dict, context: dict, market_signals: dict) -> None:
    print("[MARKET CORE]")
    for key in ("sp500", "nasdaq100", "dow"):
        print(f"{core.get(key, {}).get('name', key)} {'OK' if core.get(key, {}).get('valid') else 'FAILED'}")
    print("[MARKET CONTEXT]")
    for key in ("russell2000", "vix", "dxy", "us10y"):
        print(f"{context.get(key, {}).get('name', key)} {'OK' if context.get(key, {}).get('valid') else 'FAILED'}")
    print("[MARKET SIGNALS]")
    small = market_signals.get("small_cap_relative")
    vix = market_signals.get("vix_daily_return")
    us10y = market_signals.get("us10y_bp_change")
    print(f"Small-cap relative: {small * 100:+.2f}pct" if small is not None else "Small-cap relative: unavailable")
    print(f"VIX: {vix:+.1%}" if vix is not None else "VIX: unavailable")
    print(f"10Y: {us10y:+.0f}bp" if us10y is not None else "10Y: unavailable")


def _log_market_breadth(market_breadth: dict) -> None:
    stocks = market_breadth["stocks"]
    sectors = market_breadth["sectors"]
    health = market_breadth["health"]
    print("[MARKET BREADTH]")
    print(f"Constituents: {stocks['total_constituents']}")
    print(f"Valid: {stocks['valid_count']}")
    print(f"Advancers: {stocks['advancers']}")
    print(f"Decliners: {stocks['decliners']}")
    print(f"Unchanged: {stocks['unchanged']}")
    print(f"Coverage: {stocks['coverage_ratio']:.1%}")
    ratio = stocks.get("advance_ratio")
    print(f"Advance ratio: {ratio:.1%}" if ratio is not None else "Advance ratio: unavailable")
    print("[SECTOR BREADTH]")
    print(f"Valid: {sectors['valid_count']}/{len(sectors['items'])}")
    print(f"Advancers: {sectors['advancers']}")
    print(f"Decliners: {sectors['decliners']}")
    valid_items = [item for item in sectors["items"] if item.get("valid")]
    leading = sorted(valid_items, key=lambda item: item["daily_return"], reverse=True)[:3]
    lagging = sorted(valid_items, key=lambda item: item["daily_return"])[:3]
    if leading:
        print("Leading:")
        for item in leading:
            print(f"{item['name']} {item['daily_return']:+.2%}")
    if lagging:
        print("Lagging:")
        for item in lagging:
            print(f"{item['name']} {item['daily_return']:+.2%}")
    print("[MARKET HEALTH]")
    print(f"Score: {health['score'] * 100:.1f}" if health["score"] is not None else "Score: unavailable")
    print(f"Level: {health['level']}")
    print(f"Divergence: {health['divergence'] or 'none'}")


def _recent_news(reports: list[dict]) -> list[dict]:
    selected = []
    for report in reports[:7]:
        for item in report.get("news", []):
            selected.append({"title": item.get("original_title", item.get("title_zh", "")), "url": item.get("url", "")})
    return selected


def _recent_news_events(reports: list[dict]) -> list[dict]:
    """Load seven days of event summaries while preserving legacy report compatibility."""
    events = []
    for report in reports[:7]:
        for item in report.get("news", []):
            original_title = item.get("original_title", item.get("title_zh", ""))
            events.append({
                "report_date": report.get("report_date"),
                "event_summary": item.get("event_summary") or original_title,
                "topic_group": item.get("topic_group"),
                "original_title": original_title,
            })
    return events


def _log_news_pipeline(rss_count: int, deduped_count: int, event_representatives: list[dict],
                       recent_events: list[dict], selected: list[dict]) -> None:
    print("[NEWS PIPELINE]")
    print(f"RSS candidates: {rss_count}")
    print(f"After deterministic dedupe: {deduped_count}")
    print("[EVENT CLUSTERING]")
    print(f"Events: {len(event_representatives)}")
    print(f"Duplicates collapsed: {max(deduped_count - len(event_representatives), 0)}")
    print("Largest clusters:")
    for event in sorted(event_representatives, key=lambda item: len(item["candidate_ids"]), reverse=True)[:5]:
        print(f"{event['event_summary']}: {len(event['candidate_ids'])} articles")
    print("[EVENT HISTORY]")
    print(f"Recent events checked: {len(recent_events)}")
    print("[TOPIC DISTRIBUTION]")
    distribution = {}
    for item in selected:
        topic = item.get("topic_group") or "OTHER_SYSTEMIC"
        distribution[topic] = distribution.get(topic, 0) + 1
    for topic, count in sorted(distribution.items()):
        print(f"{topic}: {count}")
    print("[NEWS SELECTED]")
    print(f"{len(selected)} events")
    for item in selected:
        print(f"{item['rank']}. {item['source']} | {item['original_title']} | {item.get('topic_group')}")


def _offline_news():
    candidates = [
        {"candidate_id": "offline-fed-reuters", "source": "Reuters", "priority": "P0",
         "title": "Fed holds rates", "summary": "Federal Reserve held interest rates unchanged after its meeting.",
         "published_at": "2026-08-12T01:00:00+00:00", "url": "https://example.com/offline-fed-reuters"},
        {"candidate_id": "offline-fed-bbc", "source": "BBC News", "priority": "P0",
         "title": "Federal Reserve leaves rates unchanged", "summary": "Fed leaves benchmark rates unchanged.",
         "published_at": "2026-08-12T00:30:00+00:00", "url": "https://example.com/offline-fed-bbc"},
        {"candidate_id": "offline-nvidia-techcrunch", "source": "TechCrunch", "priority": "P0",
         "title": "Nvidia launches AI chip", "summary": "Nvidia introduced a new AI chip for data centers.",
         "published_at": "2026-08-12T02:00:00+00:00", "url": "https://example.com/offline-nvidia-techcrunch"},
        {"candidate_id": "offline-nvidia-ars", "source": "Ars Technica", "priority": "P1",
         "title": "Nvidia unveils new AI processor", "summary": "Nvidia revealed its latest AI processor.",
         "published_at": "2026-08-12T02:10:00+00:00", "url": "https://example.com/offline-nvidia-ars"},
        {"candidate_id": "offline-oil-bbc", "source": "BBC News", "priority": "P0",
         "title": "Oil rises after Middle East escalation", "summary": "Oil prices rose following a Middle East escalation.",
         "published_at": "2026-08-12T03:00:00+00:00", "url": "https://example.com/offline-oil-bbc"},
    ]
    events = validate_event_clusters({"events": [
        {"event_id": "event_001", "candidate_ids": ["offline-fed-reuters", "offline-fed-bbc"],
         "event_summary": "Federal Reserve held rates unchanged after its meeting.", "topic_group": "US_MARKET_MACRO"},
        {"event_id": "event_002", "candidate_ids": ["offline-nvidia-techcrunch", "offline-nvidia-ars"],
         "event_summary": "Nvidia introduced a new AI processor for data centers.", "topic_group": "AI_CHIPS"},
        {"event_id": "event_003", "candidate_ids": ["offline-oil-bbc"],
         "event_summary": "Oil prices rose after a Middle East escalation.", "topic_group": "GEOPOLITICS"},
    ]}, candidates)
    selection_candidates = event_selection_candidates(build_event_representatives(events, candidates))
    payload = {"news": [
        {"rank": 1, "candidate_id": "offline-fed-reuters", "category": "美联储 / 利率",
         "title_zh": "美联储维持利率", "summary_zh": "此条目仅用于验证事件级新闻去重后的离线日报生成流程，不代表真实新闻或投资信息。",
         "investment_impact": "离线夹具（不代表真实投资信息）：美债收益率回落 → 成长股估值压力缓解。",
         "focus": "FOMC · 美债收益率", "tags": ["Fed", "美债", "估值"],
         "investment_relevance_score": 92, "selection_reason": "演示用宏观代表事件，利率路径影响折现率。"},
        {"rank": 2, "candidate_id": "offline-nvidia-techcrunch", "category": "AI / 资本开支",
         "title_zh": "英伟达推出新 AI 芯片", "summary_zh": "此条目仅用于验证同一公司同一次事件被合并为单一代表文章，不代表真实新闻或投资信息。",
         "investment_impact": "离线夹具（不代表真实投资信息）：AI 资本开支增加 → 半导体需求预期改善。",
         "focus": "AI 资本开支 · 数据中心", "tags": ["AI", "半导体", "资本开支"],
         "investment_relevance_score": 85, "selection_reason": "演示用科技代表事件，验证事件级合并结果。"},
        {"rank": 3, "candidate_id": "offline-oil-bbc", "category": "地缘政治",
         "title_zh": "中东局势升级推动油价上涨", "summary_zh": "此条目仅用于验证独立地缘事件保留为单独新闻事件，不代表真实新闻或投资信息。",
         "investment_impact": "离线夹具（不代表真实投资信息）：油价上涨 → 通胀预期走高 → 美债收益率承压。",
         "focus": "中东局势 · 油价", "tags": ["地缘政治", "油价", "通胀"],
         "investment_relevance_score": 78, "selection_reason": "演示用独立地缘事件，验证主题分散保留。"},
    ]}
    return validate_selection(payload, selection_candidates)


def generate_daily_report(base_dir: Path = ROOT, offline_fixture: bool = False,
                          report_date: str = None) -> Path:
    base_dir = Path(base_dir)
    config_root = ROOT / "config"
    market_config = _load_yaml(config_root / "market.yaml")
    breadth_config = _load_yaml(config_root / "market_breadth.yaml")
    core_config = market_config["core"]
    context_config = market_config["context"]
    drawdown_rules = _load_yaml(config_root / "drawdown_rules.yaml")
    news_sources = _load_yaml(config_root / "news_sources.yaml")["sources"]
    if report_date:
        now = datetime.fromisoformat(f"{report_date}T10:00:00").replace(tzinfo=SHANGHAI)
    else:
        now = datetime.now(SHANGHAI)
        report_date = now.date().isoformat()

    if offline_fixture:
        snapshots, histories, context_snapshots, market_warnings = _offline_market(
            now, core_config, context_config
        )
    else:
        snapshots, histories, market_warnings = fetch_market(core_config, now)
        context_snapshots, context_warnings = fetch_market_context(context_config, now)
        market_warnings.extend(context_warnings)
    validity_summary = assess_market_validity(snapshots, context_snapshots)
    market_signals = calculate_market_signals(snapshots, context_snapshots)
    _log_market_status(snapshots, context_snapshots, market_signals)
    target_market_date = snapshots.get("sp500", {}).get("market_date")
    try:
        if not target_market_date:
            raise ValueError("S&P 500 缺少有效交易日")
        if offline_fixture:
            market_breadth = build_offline_market_breadth(
                breadth_config, target_market_date, snapshots["sp500"].get("daily_return")
            )
        else:
            market_breadth = build_market_breadth(
                ROOT / breadth_config["constituents"]["reference_file"], breadth_config, target_market_date,
                snapshots["sp500"].get("daily_return"),
            )
        breadth_warnings = []
        if market_breadth["stocks"]["status"] == "partial":
            breadth_warnings.append("部分成分股行情缺失")
        elif market_breadth["stocks"]["status"] == "invalid":
            breadth_warnings.append("市场宽度数据暂不可用")
        if market_breadth["sectors"]["valid_count"] < breadth_config["sector_minimum_valid"]:
            breadth_warnings.append("板块宽度数据不足")
    except Exception as exc:
        market_breadth = unavailable_market_breadth(target_market_date)
        breadth_warnings = [f"市场宽度数据暂不可用：{exc}"]
    _log_market_breadth(market_breadth)

    state_path = base_dir / "state" / "drawdown_state.json"
    history_path = base_dir / "state" / "drawdown_history.json"
    state = _read_json(state_path, {"version": 1, "indices": {}, "executions": []})
    history = _read_json(history_path, {"cycles": []})
    validity = validity_summary["drawdown_by_index"]
    updated_state, archived = update_drawdown_state(state, histories, validity, drawdown_rules, snapshots, now)
    if any(validity.values()):
        _write_json(state_path, updated_state)
        if archived:
            history.setdefault("cycles", []).extend(archived)
        _write_json(history_path, history)

    retained = load_reports(base_dir / "data" / "reports") if (base_dir / "data" / "reports").exists() else []
    warnings = [*market_warnings, *breadth_warnings]
    news_degraded = False
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if offline_fixture:
        news = _offline_news()
    else:
        candidates, rss_warnings = fetch_candidates(news_sources, now)
        warnings.extend(rss_warnings)
        rss_candidate_count = len(candidates)
        candidates = dedupe_candidates(candidates)
        recent_events = _recent_news_events(retained)
        if not api_key:
            news, ai_warning = [], "⚠️ 新闻 AI 处理暂时失败；RSS 数据已获取，等待下一次更新。原因：未配置 AI 凭据。"
            event_representatives = []
        else:
            events, clustering_warning = cluster_news_events(candidates, api_key)
            if clustering_warning:
                warnings.append(clustering_warning)
            event_representatives = build_event_representatives(events, candidates)
            selection_candidates = event_selection_candidates(event_representatives)
            ai_market_context = None
            if validity_summary["context_any_valid"] or market_breadth["health"].get("valid"):
                ai_market_context = {
                    "core_market": snapshots,
                    "market_context": context_snapshots,
                    "market_signals": market_signals,
                    "market_context_text": build_market_context_for_ai(
                        snapshots, context_snapshots, market_signals
                    ),
                    "market_breadth": market_breadth,
                    "market_breadth_text": build_market_breadth_text(
                        market_breadth["stocks"], market_breadth["sectors"], market_breadth["health"]
                    ),
                }
                print("[NEWS]\nMarket-driven context generated")
            news, ai_warning = select_news(
                selection_candidates, api_key, recent_events, market_context=ai_market_context
            )
        _log_news_pipeline(rss_candidate_count, len(candidates), event_representatives, recent_events, news)
        if ai_warning:
            warnings.append(ai_warning)
            news_degraded = True

    drawdown_summary = {
        key: summarize_index_state(value) for key, value in updated_state.get("indices", {}).items()
    }
    portfolio_action = derive_portfolio_action(drawdown_summary)
    market_summary = generate_market_summary(
        snapshots,
        {"items": context_snapshots, "signals": market_signals},
        market_breadth,
        news,
        portfolio_action,
        api_key,
    )

    if not validity_summary["drawdown_market_valid"]:
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
        "market_data_valid": validity_summary["drawdown_market_valid"],
        "core_market_valid": validity_summary["core_market_valid"],
        "context_market_valid": validity_summary["context_market_valid"],
        "drawdown_market_valid": validity_summary["drawdown_market_valid"],
        "market": snapshots,
        "market_context": context_snapshots,
        "market_signals": market_signals,
        "market_breadth": market_breadth,
        "drawdown": drawdown_summary,
        "portfolio_action": portfolio_action,
        "market_summary": market_summary,
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
