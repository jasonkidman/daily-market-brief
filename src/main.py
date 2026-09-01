"""Daily report orchestration CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .deepseek_client import DeepSeekUsageTracker, select_news, select_news_two_pass, validate_selection
from .drawdown import compute_suggested_topup, reserve_used_total, summarize_index_state, update_drawdown_state
from .market import (
    build_sparkline,
    calculate_context_snapshot,
    calculate_market_snapshot,
    fetch_market,
    fetch_market_context,
)
from .market_breadth import build_market_breadth, build_offline_market_breadth, unavailable_market_breadth
from .market_health import build_market_breadth_text
from .market_sentiment import calculate_market_sentiment
from .market_signals import build_market_context_for_ai, calculate_market_signals
from .market_summary import derive_portfolio_action, generate_market_summary
from .news_dedupe import dedupe_candidates
from .news_events import (
    build_event_representatives,
    cluster_news_events,
    event_selection_candidates,
    stage_a_input_counts,
    validate_event_clusters,
)
from .news_candidate_translation import translate_candidates
from .news_candidates import build_news_candidates
from .news_snapshot import load_stage_b_snapshot, write_stage_b_snapshot
from .report import load_reports, retain_latest_reports, write_report
from .renderer import render_site
from .rss_news import fetch_candidates, filter_final_candidates


ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")

# A single supplementary (non-P0) RSS source failing should not, by itself,
# flip the page-top status banner as long as the remaining sources still
# produced a healthy candidate pool. A core (P0) source failing, or the
# overall pool falling below this floor, is treated as materially degraded
# and does surface to users. See `_classify_rss_warnings`.
RSS_CORE_SOURCE_PRIORITY = "P0"
RSS_MATERIAL_CANDIDATE_FLOOR = 10


def _classify_rss_warnings(rss_warnings: list[str], news_sources: list[dict],
                           rss_candidate_count: int) -> tuple[list[dict], list[str]]:
    """Split RSS acquisition warnings into always-kept diagnostics and the
    subset material enough to surface as a user-facing page warning.

    Returns (diagnostics, material_warnings). `diagnostics` covers every RSS
    warning (kept for troubleshooting regardless of materiality); the
    `warning` text is reused verbatim from `fetch_candidates`, which formats
    it as f"{source_name} RSS 获取失败：{exc}".
    """
    priority_by_name = {source["name"]: source.get("priority", "P2") for source in news_sources}
    diagnostics, material = [], []
    for warning in rss_warnings:
        source_name = next(
            (name for name in priority_by_name if warning.startswith(f"{name} RSS 获取失败")), None
        )
        priority = priority_by_name.get(source_name, "P2")
        is_material = priority == RSS_CORE_SOURCE_PRIORITY or rss_candidate_count < RSS_MATERIAL_CANDIDATE_FLOOR
        diagnostics.append({
            "source": source_name, "priority": priority, "warning": warning, "material": is_material,
        })
        if is_material:
            material.append(warning)
    return diagnostics, material


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replay_stage_b_snapshot(snapshot_path: Path, api_key: str) -> list[dict]:
    """Replay Stage B (two-pass) from a persisted production input snapshot."""
    snapshot = load_stage_b_snapshot(snapshot_path)
    candidates = snapshot["stage_b"]["candidates"]
    selected, warning = select_news_two_pass(
        candidates,
        api_key,
        recent_selected=snapshot["stage_b"].get("recent_7_days_events", []),
        market_context=snapshot["stage_b"].get("market_context"),
    )
    if warning:
        print(warning)
    return selected


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
        "gold": [("2026-08-10", 1920), ("2026-08-11", 1931)],
        "wti": [("2026-08-10", 79), ("2026-08-11", 78.3)],
    }
    snapshots, histories = {}, {}
    for key, pairs in core_raw.items():
        history = [{"date": day, "close": close} for day, close in pairs]
        snapshot = calculate_market_snapshot(history, now)
        snapshot.update({
            "name": core_config[key]["name"], "ticker": core_config[key]["ticker"], "valid": True,
            "sparkline": build_sparkline(history),
        })
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


def _recent_news_events(reports: list[dict], current_report_date: str) -> list[dict]:
    """Load seven days of event summaries strictly before the current report date.

    Excludes reports dated the same as `current_report_date` so a same-day re-run
    (e.g. a manual re-trigger after an earlier run already published today's news)
    does not treat today's own selections as "already covered" duplicates.
    """
    prior_reports = [report for report in reports if report.get("report_date") != current_report_date]
    events = []
    for report in prior_reports[:7]:
        for item in report.get("news", []):
            original_title = item.get("original_title", item.get("title_zh", ""))
            events.append({
                "report_date": report.get("report_date"),
                "event_summary": item.get("event_summary") or original_title,
                "topic_group": item.get("topic_group"),
                "original_title": original_title,
            })
    return events


def _log_news_pipeline(rss_raw_count: int, within_24h_count: int, deduplicated_count: int,
                       stage_a_pre_cap_count: int, stage_a_actual_input_count: int,
                       event_representatives: list[dict], stage_b_candidates: list[dict],
                       stage_b_raw_count: int, stage_b_validated_count: int,
                       recent_events: list[dict], selected: list[dict],
                       stage_b_observability: dict | None = None) -> None:
    print("[NEWS PIPELINE]")
    print(f"rss_raw_count: {rss_raw_count}")
    print(f"within_24h_count: {within_24h_count}")
    print(f"deduplicated_count: {deduplicated_count}")
    print(f"stage_a_pre_cap_count: {stage_a_pre_cap_count}")
    print(f"stage_a_actual_input_count: {stage_a_actual_input_count}")
    print(f"duplicates_removed: {within_24h_count - deduplicated_count}")
    print(f"stage_a_cap_dropped: {stage_a_pre_cap_count - stage_a_actual_input_count}")
    print("[EVENT CLUSTERING]")
    print(f"stage_a_output_event_count: {len(event_representatives)}")
    print(f"clustering_collapsed: {max(stage_a_actual_input_count - len(event_representatives), 0)}")
    print("Largest clusters:")
    for event in sorted(event_representatives, key=lambda item: len(item["candidate_ids"]), reverse=True)[:5]:
        print(f"{event['event_summary']}: {len(event['candidate_ids'])} articles")
    print("[STAGE A EVENTS]")
    for event in event_representatives:
        print(f"event_id={event.get('event_id')} | category={event.get('event_category', 'other')} | title={event.get('event_summary', '')}")
    print("[STAGE B INPUT EVENTS]")
    print(f"stage_b_input_count: {len(stage_b_candidates)}")
    for item in stage_b_candidates:
        print(f"candidate_id={item.get('candidate_id')} | category={item.get('event_category', 'other')} | title={item.get('title', '')}")
    print(f"stage_b_raw_count: {stage_b_raw_count}")
    print(f"stage_b_validated_count: {stage_b_validated_count}")
    if stage_b_observability:
        for key in (
            "stage_b_selected_count", "stage_b_reserve_count", "stage_b_selected_valid_count",
            "stage_b_backfilled_count", "stage_b_final_count", "stage_b_target_count",
            "stage_b_sample_a_count", "stage_b_sample_b_count", "stage_b_intersection_count",
            "stage_b_borderline_count", "stage_b_review_keep_count",
        ):
            if key in stage_b_observability:
                print(f"{key}: {stage_b_observability.get(key, 0)}")
        if "two_pass_degraded" in stage_b_observability:
            print(f"two_pass_degraded: {stage_b_observability['two_pass_degraded']}")
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
    print(f"Final saved news count: {len(selected)}")
    for item in selected:
        print(f"{item['rank']}. {item.get('title_zh', '')} | {item['source']} | {item['original_title']} | {item.get('topic_group')}")


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
         "focus": "FOMC · 美债收益率", "tags": ["Fed", "美债", "估值"],
         "investment_relevance_score": 92, "selection_reason": "演示用宏观代表事件，利率路径影响折现率。"},
        {"rank": 2, "candidate_id": "offline-nvidia-techcrunch", "category": "AI / 资本开支",
         "title_zh": "英伟达推出新 AI 芯片", "summary_zh": "此条目仅用于验证同一公司同一次事件被合并为单一代表文章，不代表真实新闻或投资信息。",
         "focus": "AI 资本开支 · 数据中心", "tags": ["AI", "半导体", "资本开支"],
         "investment_relevance_score": 85, "selection_reason": "演示用科技代表事件，验证事件级合并结果。"},
        {"rank": 3, "candidate_id": "offline-oil-bbc", "category": "地缘政治",
         "title_zh": "中东局势升级推动油价上涨", "summary_zh": "此条目仅用于验证独立地缘事件保留为单独新闻事件，不代表真实新闻或投资信息。",
         "focus": "中东局势 · 油价", "tags": ["地缘政治", "油价", "通胀"],
         "investment_relevance_score": 78, "selection_reason": "演示用独立地缘事件，验证主题分散保留。"},
    ]}
    news = validate_selection(payload, selection_candidates)
    return news, build_news_candidates(selection_candidates, news)


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
    usage_tracker = DeepSeekUsageTracker()
    stage_b_observability = {"raw_count": 0, "validated_count": 0}
    news_source_diagnostics = []
    event_clustering_diagnostics = {"fallback_used": False, "reason": None}
    news_candidates = []
    if offline_fixture:
        news, news_candidates = _offline_news()
    else:
        candidates, rss_warnings = fetch_candidates(news_sources, now)
        rss_candidate_count = len(candidates)
        news_source_diagnostics, material_rss_warnings = _classify_rss_warnings(
            rss_warnings, news_sources, rss_candidate_count
        )
        warnings.extend(material_rss_warnings)
        candidates = filter_final_candidates(candidates, now)
        window_candidate_count = len(candidates)
        candidates = dedupe_candidates(candidates)
        stage_a_pre_cap_count, stage_a_actual_input_count = stage_a_input_counts(candidates)
        recent_events = _recent_news_events(retained, report_date)
        if not api_key:
            news, ai_warning = [], "⚠️ 新闻 AI 处理暂时失败；RSS 数据已获取，等待下一次更新。原因：未配置 AI 凭据。"
            event_representatives = []
            selection_candidates = []
            stage_a_actual_input_count = 0
        else:
            events, clustering_warning = cluster_news_events(candidates, api_key, usage_tracker=usage_tracker)
            # The fallback (one event per candidate) always preserves basic
            # (exact-URL-level) dedup -- `dedupe_candidates` already ran above
            # unconditionally -- and can never leave the pipeline empty for a
            # non-empty candidate pool, so this never rises to a page-top
            # banner; it's still fully captured for troubleshooting. Its one
            # real cost is a quality nuance, not a broken/incomplete result:
            # articles from different outlets about the same real event may
            # surface as separate Stage B events instead of being merged into
            # one, when it would normally have been merged.
            event_clustering_diagnostics = {
                "fallback_used": clustering_warning is not None,
                "reason": clustering_warning,
            }
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
            snapshot = {
                "schema_version": 1,
                "report_date": report_date,
                "run_at": datetime.now(SHANGHAI).isoformat(),
                "candidate_counts": {
                    "rss_raw": rss_candidate_count,
                    "within_24h": window_candidate_count,
                    "deduplicated": len(candidates),
                    "stage_a_pre_cap": stage_a_pre_cap_count,
                    "stage_a_actual_input": stage_a_actual_input_count,
                    "stage_a_events": len(events),
                    "stage_b_input": len(selection_candidates),
                },
                "stage_a_events": events,
                "stage_b": {
                    "candidates": selection_candidates,
                    "recent_7_days_events": recent_events,
                    "market_context": ai_market_context,
                },
            }
            # Snapshot persistence is fail-fast: a report without a replayable Stage B input is incomplete.
            write_stage_b_snapshot(base_dir, snapshot)
            news, ai_warning = select_news_two_pass(
                selection_candidates, api_key, recent_events, market_context=ai_market_context,
                usage_tracker=usage_tracker, observability=stage_b_observability,
            )
            selected_candidate_ids = {item["candidate_id"] for item in news}
            untranslated_candidates = [
                candidate for candidate in selection_candidates
                if candidate["candidate_id"] not in selected_candidate_ids
            ]
            candidate_translations = translate_candidates(
                untranslated_candidates, api_key, usage_tracker=usage_tracker,
            )
            news_candidates = build_news_candidates(selection_candidates, news, candidate_translations)
        _log_news_pipeline(
            rss_candidate_count, window_candidate_count, len(candidates), stage_a_pre_cap_count,
            stage_a_actual_input_count, event_representatives, selection_candidates,
            stage_b_observability["raw_count"], stage_b_observability["validated_count"],
            recent_events, news, stage_b_observability,
        )
        if ai_warning:
            warnings.append(ai_warning)
            news_degraded = True

    total_reserve = drawdown_rules.get("total_reserve", 0)
    executions = updated_state.get("executions", [])
    reserve_used = reserve_used_total(executions)
    remaining_reserve = max(total_reserve - reserve_used, 0)
    drawdown_summary = {
        key: summarize_index_state(value) for key, value in updated_state.get("indices", {}).items()
    }
    for key, summary in drawdown_summary.items():
        summary["already_invested"] = sum(
            int(item.get("amount", 0)) for item in executions if item.get("index") == key
        )
        summary["suggested_amount"] = compute_suggested_topup(summary, executions, key, remaining_reserve)
    portfolio_action = derive_portfolio_action(drawdown_summary)
    market_summary = generate_market_summary(
        snapshots,
        {"items": context_snapshots, "signals": market_signals},
        market_breadth,
        news,
        portfolio_action,
        api_key,
        usage_tracker=usage_tracker,
    )
    usage_tracker.log_summary()

    if not validity_summary["drawdown_market_valid"]:
        status, status_label = "critical", "🔴 行情数据校验失败"
    elif warnings:
        status, status_label = "partial", "🟠 部分数据源异常"
    else:
        status, status_label = "ok", "🟢 数据更新正常"
    valid_market_dates = [item["market_date"] for item in snapshots.values() if item.get("valid")]
    discipline_config = drawdown_rules.get("discipline", {})
    market_sentiment = calculate_market_sentiment(
        context_snapshots, snapshots, histories, market_breadth.get("health", {})
    )
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
        "market_sentiment": market_sentiment,
        "drawdown": drawdown_summary,
        "reserve": {
            "total": total_reserve,
            "remaining": remaining_reserve,
            "used": reserve_used,
            "ratio": (remaining_reserve / total_reserve) if total_reserve else None,
        },
        "discipline": {
            "monthly_dca": discipline_config.get("monthly_dca"),
            "holding_years_min": discipline_config.get("holding_years_min"),
        },
        "portfolio_action": portfolio_action,
        "market_summary": market_summary,
        "news": news,
        "news_candidates": news_candidates,
        "news_degraded": news_degraded,
        "news_source_diagnostics": news_source_diagnostics,
        "event_clustering_diagnostics": event_clustering_diagnostics,
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
    parser.add_argument("--stage-b-snapshot", type=Path)
    args = parser.parse_args()
    if args.stage_b_snapshot:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            parser.error("--stage-b-snapshot requires DEEPSEEK_API_KEY")
        selected = replay_stage_b_snapshot(args.stage_b_snapshot, api_key)
        print(f"Replayed Stage B: {len(selected)} selected news items")
        return 0
    path = generate_daily_report(args.base_dir, args.offline_fixture, args.report_date)
    print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
