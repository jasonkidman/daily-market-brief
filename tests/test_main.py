import json

import pytest

import src.main as main
from src.main import (
    _classify_rss_warnings, _log_news_pipeline, _recent_news_events, assess_market_validity, generate_daily_report,
)
from src.market_summary import generate_market_summary as _real_generate_market_summary
from src.news_scoring import score_candidates as _real_score_candidates
from src.news_top_dedup import dedupe_top_candidates as _real_dedupe_top_candidates
from src.smoke import validate_generated


def test_news_pipeline_logs_distinct_stage_counts(capsys):
    _log_news_pipeline(89, 85, 79, 72, [], [], [])

    output = capsys.readouterr().out
    assert "rss_raw_count: 89" in output
    assert "within_24h_count: 85" in output
    assert "deduplicated_count: 79" in output
    assert "duplicates_removed: 6" in output
    assert "prefiltered_count: 72" in output
    assert "off_topic_removed: 7" in output
    assert "clustered_event_count: 0" in output
    assert "clustering_collapsed: 72" in output
    assert "scoring_input_count: 0" in output


def test_recent_news_events_supports_legacy_and_v2_reports():
    events = _recent_news_events([
        {"report_date": "2026-08-12", "news": [{
            "original_title": "Legacy Fed headline", "url": "https://legacy.example",
        }]},
        {"report_date": "2026-08-11", "news": [{
            "original_title": "New headline", "event_summary": "Fed held rates", "topic_group": "US_MARKET_MACRO",
        }]},
    ], "2026-08-13")

    assert events == [
        {"report_date": "2026-08-12", "event_summary": "Legacy Fed headline", "topic_group": None,
         "original_title": "Legacy Fed headline"},
        {"report_date": "2026-08-11", "event_summary": "Fed held rates", "topic_group": "US_MARKET_MACRO",
         "original_title": "New headline"},
    ]


def test_recent_news_events_excludes_same_day_reports():
    events = _recent_news_events([
        {"report_date": "2026-08-27", "news": [{
            "original_title": "Already published earlier today", "event_summary": "Nvidia earnings",
        }]},
        {"report_date": "2026-08-26", "news": [{
            "original_title": "Yesterday headline", "event_summary": "Fed held rates",
        }]},
    ], "2026-08-27")

    assert [event["report_date"] for event in events] == ["2026-08-26"]


def test_recent_news_events_takes_seven_days_strictly_before_current_date():
    reports = [{"report_date": f"2026-08-{day:02d}", "news": [{"event_summary": f"Event {day}"}]}
               for day in range(27, 19, -1)]

    events = _recent_news_events(reports, "2026-08-27")

    assert [event["report_date"] for event in events] == [
        "2026-08-26", "2026-08-25", "2026-08-24", "2026-08-23", "2026-08-22", "2026-08-21", "2026-08-20",
    ]


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
    assert set(report["market_context"]) == {"russell2000", "vix", "dxy", "us10y", "gold", "wti"}
    assert report["market_context"]["russell2000"]["daily_return"] == pytest.approx(0.01)
    assert report["market_context"]["vix"]["daily_return"] == pytest.approx(0.10)
    assert report["market_context"]["dxy"]["daily_return"] == pytest.approx(0.005)
    assert report["market_context"]["us10y"]["yield_change_bp"] == pytest.approx(8)
    assert report["market"]["sp500"]["sparkline"]["line"].startswith("M")
    assert report["market_context"]["gold"]["daily_return"] == pytest.approx(1931 / 1920 - 1)
    assert report["market_context"]["wti"]["daily_return"] == pytest.approx(78.3 / 79 - 1)
    assert report["reserve"]["total"] == 200000
    assert report["reserve"]["remaining"] == 200000
    assert report["discipline"] == {"monthly_dca": 10000, "holding_years_min": 20}
    sentiment = report["market_sentiment"]
    assert set(sentiment) == {
        "vix_score", "breadth_score", "momentum_score", "risk_appetite_score",
        "market_sentiment_score", "market_sentiment_label",
    }
    assert all(sentiment[key] is not None for key in sentiment)
    assert 0.0 <= sentiment["market_sentiment_score"] <= 100.0
    assert report["market_signals"]["vix_daily_return"] == pytest.approx(0.10)
    assert report["market_signals"]["us10y_bp_change"] == pytest.approx(8)
    assert report["market_breadth"]["stocks"]["advance_ratio"] == pytest.approx(0.60)
    assert report["market_breadth"]["sectors"]["advance_ratio"] == pytest.approx(7 / 11)
    assert report["market_breadth"]["health"]["score"] == pytest.approx(0.61454545)
    assert report["market_breadth"]["health"]["level"] == "mixed"
    assert [item["candidate_id"] for item in report["news"]] == [
        "offline-fed-reuters", "offline-nvidia-techcrunch", "offline-oil-bbc",
    ]
    assert [item["topic_group"] for item in report["news"]] == [
        "US_MARKET_MACRO", "MEGA_CAP_TECH", "ENERGY_COMMODITIES",
    ]
    assert [item["rank"] for item in report["news"]] == [1, 2, 3]
    assert all(item["event_summary"] for item in report["news"])
    for item in report["news"]:
        assert item["category"]
        assert 50 <= item["score"] <= 100
        assert "不代表真实新闻或投资信息" in item["summary_zh"]
    assert report["portfolio_action"] == "hold"
    assert report["market_summary"]["degraded"] is True
    assert "标普500当日下跌0.5%" in report["market_summary"]["summary"]
    assert "市场同时关注" in report["market_summary"]["summary"]
    assert report["market_summary"]["action"] == "未触发额外回撤加仓，维持正常定投，备用金保持不动。"
    assert "Daily Market Brief" in index_path.read_text(encoding="utf-8")
    assert result == report_path
    assert [item["candidate_id"] for item in report["news_candidates"]] == [
        "offline-fed-reuters", "offline-nvidia-techcrunch", "offline-oil-bbc",
    ]
    assert all(item["selected"] for item in report["news_candidates"])
    # Regression: src/smoke.py's REQUIRED_HTML must stay in sync with the real
    # rendered template -- CI's smoke-check step runs validate_generated against
    # the actual site/index.html, not a hand-written fixture, so this is the
    # only place that would have caught the 2026-09-07 "今日市场一句话" ->
    # "今日结论" heading rename silently breaking CI's smoke check.
    validate_generated(tmp_path)


def test_reserve_reflects_persisted_executions_and_is_idempotent_across_runs(tmp_path):
    """End-to-end regression for the 2026-08-31 reserve restructure: a historical
    $7,500 Nasdaq-100 deployment, backfilled as an executions-ledger entry (predating
    the tier system), must show up as reserve.used/remaining without being hand-set
    in multiple places, and must not get double counted when the daily report is
    regenerated on top of the same persisted state (no new executions entry added)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    seed_state = {
        "version": 1,
        "indices": {},
        "executions": [{
            "executed_at": "2026-06-01T00:00:00+08:00", "amount": 7500,
            "index": "nasdaq100", "tier": None, "cycle_id": None,
        }],
    }
    state_path = state_dir / "drawdown_state.json"
    state_path.write_text(json.dumps(seed_state), encoding="utf-8")

    report_path = tmp_path / "data" / "reports" / "2026-08-12.json"
    generate_daily_report(base_dir=tmp_path, offline_fixture=True, report_date="2026-08-12")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reserve"]["total"] == 200000
    assert report["reserve"]["used"] == 7500
    assert report["reserve"]["remaining"] == 192500
    assert report["drawdown"]["nasdaq100"]["already_invested"] == 7500
    assert report["drawdown"]["nasdaq100"]["suggested_amount"] == 0

    # Regenerate on top of the same persisted state (simulating the next scheduled
    # run) without adding a second executions entry.
    generate_daily_report(base_dir=tmp_path, offline_fixture=True, report_date="2026-08-12")
    report_again = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_again["reserve"]["used"] == 7500
    assert report_again["reserve"]["remaining"] == 192500

    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted_state["executions"] == seed_state["executions"]


def test_drawdown_validity_ignores_dow_and_context_failures():
    core = {key: {"valid": True} for key in ("sp500", "nasdaq100", "dow")}
    context = {key: {"valid": True} for key in ("russell2000", "vix", "dxy", "us10y")}

    core["dow"]["valid"] = False
    for key in context:
        broken_context = {name: dict(value) for name, value in context.items()}
        broken_context[key]["valid"] = False
        validity = assess_market_validity(core, broken_context)
        assert validity["drawdown_by_index"] == {"sp500": True, "nasdaq100": True}
        assert validity["drawdown_market_valid"] is True


def test_drawdown_validity_is_independent_for_sp500_and_nasdaq100():
    context = {key: {"valid": True} for key in ("russell2000", "vix", "dxy", "us10y")}
    core = {"sp500": {"valid": False}, "nasdaq100": {"valid": True}, "dow": {"valid": True}}
    assert assess_market_validity(core, context)["drawdown_by_index"] == {
        "sp500": False, "nasdaq100": True,
    }
    core["sp500"]["valid"], core["nasdaq100"]["valid"] = True, False
    assert assess_market_validity(core, context)["drawdown_by_index"] == {
        "sp500": True, "nasdaq100": False,
    }


def test_breadth_failure_does_not_block_drawdown_state_updates(tmp_path, monkeypatch):
    def fail_breadth(*args, **kwargs):
        raise RuntimeError("breadth source unavailable")

    monkeypatch.setattr(main, "build_offline_market_breadth", fail_breadth)
    generate_daily_report(base_dir=tmp_path, offline_fixture=True, report_date="2026-08-12")

    report = json.loads((tmp_path / "data" / "reports" / "2026-08-12.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "state" / "drawdown_state.json").read_text(encoding="utf-8"))
    assert report["market_breadth"]["health"]["valid"] is False
    assert set(state["indices"]) == {"sp500", "nasdaq100"}


_NEWS_SOURCES = [
    {"name": "BBC News", "priority": "P0"},
    {"name": "SEC Press Releases", "priority": "P1"},
    {"name": "The Verge", "priority": "P2"},
]


def test_single_supplementary_source_failure_with_healthy_pool_is_not_material():
    diagnostics, material = _classify_rss_warnings(
        ["SEC Press Releases RSS 获取失败：EOF occurred in violation of protocol"],
        _NEWS_SOURCES,
        rss_candidate_count=50,
    )
    assert material == []
    assert diagnostics == [{
        "source": "SEC Press Releases", "priority": "P1",
        "warning": "SEC Press Releases RSS 获取失败：EOF occurred in violation of protocol",
        "material": False,
    }]


def test_core_p0_source_failure_is_always_material_even_with_healthy_pool():
    diagnostics, material = _classify_rss_warnings(
        ["BBC News RSS 获取失败：feed unavailable"], _NEWS_SOURCES, rss_candidate_count=50,
    )
    assert material == ["BBC News RSS 获取失败：feed unavailable"]
    assert diagnostics[0]["material"] is True


def test_supplementary_source_failure_is_material_when_overall_pool_is_thin():
    diagnostics, material = _classify_rss_warnings(
        ["SEC Press Releases RSS 获取失败：feed unavailable"], _NEWS_SOURCES, rss_candidate_count=3,
    )
    assert material == ["SEC Press Releases RSS 获取失败：feed unavailable"]
    assert diagnostics[0]["material"] is True


def test_mixed_warnings_classified_independently():
    diagnostics, material = _classify_rss_warnings(
        [
            "The Verge RSS 获取失败：timeout",
            "BBC News RSS 获取失败：timeout",
        ],
        _NEWS_SOURCES,
        rss_candidate_count=50,
    )
    assert material == ["BBC News RSS 获取失败：timeout"]
    assert {d["source"]: d["material"] for d in diagnostics} == {
        "The Verge": False, "BBC News": True,
    }


def test_no_rss_warnings_yields_empty_diagnostics_and_material():
    diagnostics, material = _classify_rss_warnings([], _NEWS_SOURCES, rss_candidate_count=50)
    assert diagnostics == []
    assert material == []


def test_end_to_end_supplementary_rss_failure_does_not_flip_status_or_show_banner_warning(
    tmp_path, monkeypatch
):
    """A single P1 RSS source failing, with the rest of the pool healthy, must
    not flip report.status to 'partial' or add a banner-worthy warning, but
    must still be kept in news_source_diagnostics for troubleshooting."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, market_warnings = main._offline_market(
        now, market_config["core"], market_config["context"]
    )
    healthy_candidates = [
        {
            "candidate_id": f"c{i}",
            "title": f"Story {i}",
            "summary": "Summary",
            "source": "BBC News",
            "priority": "P0",
            "published_at": "2026-08-12T01:00:00+00:00",
            "url": f"https://example.com/{i}",
        }
        for i in range(20)
    ]

    # Use [] rather than `market_warnings`: `_offline_market` includes a fixed
    # disclaimer warning unrelated to this test's RSS-classification behavior.
    monkeypatch.setattr(main, "fetch_market", lambda *args: (snapshots, histories, []))
    monkeypatch.setattr(main, "fetch_market_context", lambda *args: (contexts, []))
    monkeypatch.setattr(
        main, "build_market_breadth",
        lambda *args: main.build_offline_market_breadth(
            breadth_config, "2026-08-11", snapshots["sp500"]["daily_return"]
        ),
    )
    monkeypatch.setattr(
        main, "fetch_candidates",
        lambda sources, now: (
            healthy_candidates,
            ["SEC Press Releases RSS 获取失败：EOF occurred in violation of protocol"],
        ),
    )
    monkeypatch.setattr(main, "filter_final_candidates", lambda candidates, now: candidates)
    monkeypatch.setattr(main, "dedupe_candidates", lambda candidates: candidates)
    monkeypatch.setattr(main, "cluster_candidates_local", lambda candidates: [])
    monkeypatch.setattr(main, "build_event_representatives", lambda events, candidates: [])
    monkeypatch.setattr(main, "event_selection_candidates", lambda events: [])
    monkeypatch.setattr(main, "score_candidates", lambda *args, **kwargs: {})
    monkeypatch.setattr(main, "generate_market_summary", lambda *args, **kwargs: {"degraded": True})
    monkeypatch.setattr(main, "render_site", lambda *args, **kwargs: None)

    generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    report = json.loads((tmp_path / "data" / "reports" / "2026-08-12.json").read_text(encoding="utf-8"))
    assert not any("SEC" in warning for warning in report["warnings"])
    assert report["status"] != "partial"
    assert report["news_source_diagnostics"] == [{
        "source": "SEC Press Releases", "priority": "P1",
        "warning": "SEC Press Releases RSS 获取失败：EOF occurred in violation of protocol",
        "material": False,
    }]


def _patch_common_pipeline(monkeypatch, snapshots, histories, contexts, breadth_config, selection_candidates):
    monkeypatch.setattr(main, "fetch_market", lambda *args: (snapshots, histories, []))
    monkeypatch.setattr(main, "fetch_market_context", lambda *args: (contexts, []))
    monkeypatch.setattr(
        main, "build_market_breadth",
        lambda *args: main.build_offline_market_breadth(
            breadth_config, "2026-08-11", snapshots["sp500"]["daily_return"]
        ),
    )
    monkeypatch.setattr(main, "fetch_candidates", lambda sources, now: (selection_candidates, []))
    monkeypatch.setattr(main, "filter_final_candidates", lambda candidates, now: candidates)
    monkeypatch.setattr(main, "dedupe_candidates", lambda candidates: candidates)
    monkeypatch.setattr(main, "cluster_candidates_local", lambda candidates: [])
    monkeypatch.setattr(main, "build_event_representatives", lambda events, candidates: [])
    monkeypatch.setattr(main, "event_selection_candidates", lambda events: selection_candidates)
    monkeypatch.setattr(main, "generate_market_summary", lambda *args, **kwargs: {"degraded": True})
    monkeypatch.setattr(main, "render_site", lambda *args, **kwargs: None)


def test_scoring_total_failure_falls_back_to_rule_based_top_news_and_flags_degraded(tmp_path, monkeypatch):
    """Every scoring batch failing must never leave "今日重要新闻" empty -- the
    pure-code rank fills it in, and the page is flagged as degraded rather than
    silently looking like a fully healthy AI-scored report."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, _ = main._offline_market(now, market_config["core"], market_config["context"])
    selection_candidates = [
        {"candidate_id": "c1", "title": "Fed holds rates", "summary": "Summary 1",
         "source": "Reuters", "priority": "P0", "url": "https://example.com/c1",
         "published_at": "2026-08-12T01:00:00+00:00", "topic_group": "US_MARKET_MACRO"},
    ]
    _patch_common_pipeline(monkeypatch, snapshots, histories, contexts, breadth_config, selection_candidates)
    monkeypatch.setattr(main, "score_candidates", lambda *args, **kwargs: {})

    generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    report = json.loads((tmp_path / "data" / "reports" / "2026-08-12.json").read_text(encoding="utf-8"))
    assert report["news_degraded"] is True
    assert any("评分" in warning for warning in report["warnings"])
    assert len(report["news"]) == 1
    assert report["news"][0]["candidate_id"] == "c1"
    assert report["news"][0]["selection_mode"] == "rule_based"
    assert report["news"][0]["title_zh"] == "Fed holds rates"
    assert report["news"][0]["score"] is None


def test_report_news_candidates_covers_the_full_scored_pool_with_selected_flag(tmp_path, monkeypatch):
    """The review-drawer candidate pool covers every scored candidate, not just
    the ones that made "今日重要新闻" -- Layer 1 scores (and translates) the
    whole pool, so there is no separate review-filter/translation step left."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, _ = main._offline_market(now, market_config["core"], market_config["context"])
    selection_candidates = [
        {"candidate_id": "c1", "title": "Fed holds rates", "summary": "Summary 1",
         "source": "Reuters", "priority": "P0", "url": "https://example.com/c1",
         "published_at": "2026-08-12T01:00:00+00:00", "topic_group": "US_MARKET_MACRO"},
        {"candidate_id": "c2", "title": "Nvidia launches chip", "summary": "Summary 2",
         "source": "TechCrunch", "priority": "P0", "url": "https://example.com/c2",
         "published_at": "2026-08-12T02:00:00+00:00", "topic_group": "AI_CHIPS"},
        {"candidate_id": "c3", "title": "Local festival draws crowds", "summary": "Summary 3",
         "source": "BBC News", "priority": "P2", "url": "https://example.com/c3",
         "published_at": "2026-08-12T03:00:00+00:00", "topic_group": "OTHER_SYSTEMIC"},
    ]
    _patch_common_pipeline(monkeypatch, snapshots, histories, contexts, breadth_config, selection_candidates)
    scores = {
        "c1": {"score": 92, "category": "美联储 / 利率", "title_zh": "美联储维持利率", "summary_zh": "摘要", "reason": ""},
        "c2": {"score": 60, "category": "半导体", "title_zh": "英伟达推出新芯片", "summary_zh": "摘要2中文", "reason": ""},
        "c3": {"score": 5, "category": "", "title_zh": "本地节日吸引人群", "summary_zh": "摘要3中文", "reason": ""},
    }
    monkeypatch.setattr(main, "score_candidates", lambda *args, **kwargs: scores)

    generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    report = json.loads((tmp_path / "data" / "reports" / "2026-08-12.json").read_text(encoding="utf-8"))
    candidates_by_id = {item["candidate_id"]: item for item in report["news_candidates"]}
    assert set(candidates_by_id) == {"c1", "c2", "c3"}
    assert candidates_by_id["c1"]["selected"] is True
    assert candidates_by_id["c1"]["category"] == "宏观 / 利率"
    assert candidates_by_id["c1"]["title_zh"] == "美联储维持利率"
    assert candidates_by_id["c2"]["selected"] is True
    assert candidates_by_id["c2"]["category"] == "AI / 科技"
    assert candidates_by_id["c3"]["selected"] is True
    # All three scored candidates fit within NEWS_TOP_N, so all three are "selected".
    assert [item["candidate_id"] for item in report["news"]] == ["c1", "c2", "c3"]


def test_report_generation_falls_back_to_english_when_scoring_omits_a_candidate(tmp_path, monkeypatch):
    """A candidate score_candidates() has no result for (dropped as invalid, or
    its batch failed/was skipped) must degrade to the original English
    title/summary in the "more news" pool, never break report generation."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, _ = main._offline_market(now, market_config["core"], market_config["context"])
    selection_candidates = [{
        "candidate_id": "c1", "title": "Federal Reserve holds interest rates steady", "summary": "Summary 1",
        "source": "BBC News", "priority": "P0", "url": "https://example.com/c1",
        "published_at": "2026-08-12T01:00:00+00:00", "topic_group": "US_MARKET_MACRO",
    }]
    _patch_common_pipeline(monkeypatch, snapshots, histories, contexts, breadth_config, selection_candidates)
    monkeypatch.setattr(main, "score_candidates", lambda *args, **kwargs: {})

    report_path = generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["news_candidates"][0]["title_zh"] == "Federal Reserve holds interest rates steady"
    assert report["news_candidates"][0]["summary_zh"] == "Summary 1"


def test_full_run_makes_at_most_5_llm_calls(tmp_path, monkeypatch):
    """PR 3's core regression, extended for the TopDedup stage added after real
    2026-09-09 production data showed unmerged same-event duplicates crowding
    out "今日重要新闻": a normal run must be 3 scoring batches (100 candidates
    -> 40+40+20) + 1 bounded TopDedup call + 1 Layer 2 call = 5 total, down
    from the old architecture's Stage A/B/translation fan-out (confirmed live
    to reach 40+ calls on an ordinary day, worse on failure). Exercises the
    real score_candidates()/dedupe_top_candidates()/generate_market_summary()
    batching and validation logic end to end, with only the transport faked."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, _ = main._offline_market(now, market_config["core"], market_config["context"])
    selection_candidates = [
        {"candidate_id": f"c{i}", "title": f"Story {i}", "summary": "Summary",
         "source": "BBC News", "priority": "P0", "url": f"https://example.com/{i}",
         "published_at": "2026-08-12T01:00:00+00:00", "topic_group": "US_MARKET_MACRO"}
        for i in range(100)
    ]

    calls = []

    def fake_call_model(system_prompt, user_payload, api_key, **kwargs):
        calls.append(1)
        payload = json.loads(user_payload)
        if "candidates" not in payload:
            return json.dumps({"summary": "标普500当日下跌0.5%，纳指100当日下跌0.5%。市场同时关注上述新闻。"})
        ids = [item["candidate_id"] for item in payload["candidates"]]
        if "duplicate_of" in system_prompt:
            return json.dumps({"items": [{"candidate_id": cid, "duplicate_of": None} for cid in ids]})
        return json.dumps({"scores": [
            {"candidate_id": cid, "score": 60, "category": "美国经济", "title_zh": "标题", "summary_zh": "摘要", "reason": ""}
            for cid in ids
        ]})

    monkeypatch.setattr(main, "fetch_market", lambda *args: (snapshots, histories, []))
    monkeypatch.setattr(main, "fetch_market_context", lambda *args: (contexts, []))
    monkeypatch.setattr(
        main, "build_market_breadth",
        lambda *args: main.build_offline_market_breadth(
            breadth_config, "2026-08-11", snapshots["sp500"]["daily_return"]
        ),
    )
    monkeypatch.setattr(main, "fetch_candidates", lambda sources, now: (selection_candidates, []))
    monkeypatch.setattr(main, "filter_final_candidates", lambda candidates, now: candidates)
    monkeypatch.setattr(main, "dedupe_candidates", lambda candidates: candidates)
    monkeypatch.setattr(main, "cluster_candidates_local", lambda candidates: [])
    monkeypatch.setattr(main, "build_event_representatives", lambda events, candidates: [])
    monkeypatch.setattr(main, "event_selection_candidates", lambda events: selection_candidates)
    monkeypatch.setattr(main, "render_site", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main, "score_candidates",
        lambda candidates, api_key, focus_rules, **kwargs: _real_score_candidates(
            candidates, api_key, focus_rules, call_model=fake_call_model, sleep_fn=lambda _: None,
            usage_tracker=kwargs.get("usage_tracker"), observability=kwargs.get("observability"),
        ),
    )
    monkeypatch.setattr(
        main, "generate_market_summary",
        lambda *args, **kwargs: _real_generate_market_summary(*args, call_model=fake_call_model),
    )
    monkeypatch.setattr(
        main, "dedupe_top_candidates",
        lambda ranked_candidates, api_key, **kwargs: _real_dedupe_top_candidates(
            ranked_candidates, api_key, call_model=fake_call_model, usage_tracker=kwargs.get("usage_tracker"),
        ),
    )

    generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    assert len(calls) <= 5
