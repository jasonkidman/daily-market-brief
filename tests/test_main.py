import json

import pytest

import src.main as main
from src.main import _classify_rss_warnings, _log_news_pipeline, assess_market_validity, generate_daily_report


def test_news_pipeline_logs_distinct_stage_counts(capsys):
    _log_news_pipeline(89, 85, 79, 79, 50, [], [], 6, 5, [], [])

    output = capsys.readouterr().out
    assert "rss_raw_count: 89" in output
    assert "within_24h_count: 85" in output
    assert "deduplicated_count: 79" in output
    assert "stage_a_pre_cap_count: 79" in output
    assert "stage_a_actual_input_count: 50" in output
    assert "stage_a_cap_dropped: 29" in output
    assert "stage_a_output_event_count: 0" in output
    assert "clustering_collapsed: 50" in output
    assert "stage_b_input_count: 0" in output
    assert "stage_b_raw_count: 6" in output
    assert "stage_b_validated_count: 5" in output
    assert "Duplicates collapsed" not in output


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
        "US_MARKET_MACRO", "AI_CHIPS", "GEOPOLITICS",
    ]
    assert all(item["event_summary"] for item in report["news"])
    for item in report["news"]:
        assert item["focus"]
        assert item["tags"]
        assert 50 <= item["investment_relevance_score"] <= 100
        assert "不代表真实新闻或投资信息" in item["summary_zh"]
    assert report["portfolio_action"] == "hold"
    assert report["market_summary"]["degraded"] is True
    assert "标普500当日下跌0.5%" in report["market_summary"]["market"]
    assert report["market_summary"]["drivers"].startswith("市场同时关注")
    assert report["market_summary"]["action"] == "未触发额外回撤加仓，维持正常定投，备用金保持不动。"
    assert "Daily Market Brief" in index_path.read_text(encoding="utf-8")
    assert result == report_path


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
    monkeypatch.setattr(main, "stage_a_input_counts", lambda candidates: (len(candidates), len(candidates)))
    monkeypatch.setattr(main, "cluster_news_events", lambda *args, **kwargs: ([], None))
    monkeypatch.setattr(main, "build_event_representatives", lambda events, candidates: [])
    monkeypatch.setattr(main, "event_selection_candidates", lambda events: [])
    monkeypatch.setattr(main, "select_news_two_pass", lambda *args, **kwargs: ([], None))
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


def test_event_clustering_fallback_does_not_flip_status_or_show_banner_warning(tmp_path, monkeypatch):
    """Stage A clustering falling back to one-event-per-candidate always keeps
    basic (URL-level) dedup intact and never empties the pipeline, so it must
    stay diagnostics-only -- never a page-top banner or status flip."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, _ = main._offline_market(
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
        for i in range(5)
    ]
    fallback_events = [
        {"event_id": f"fallback_{i:03d}", "candidate_ids": [c["candidate_id"]],
         "event_summary": c["title"], "topic_group": "OTHER_SYSTEMIC", "event_category": "other"}
        for i, c in enumerate(healthy_candidates, 1)
    ]
    clustering_reason = (
        "⚠️ 新闻事件级去重暂时失败，已使用基础去重结果继续生成日报。 原因：所有 candidate 必须恰好被一个 event 覆盖。"
    )

    monkeypatch.setattr(main, "fetch_market", lambda *args: (snapshots, histories, []))
    monkeypatch.setattr(main, "fetch_market_context", lambda *args: (contexts, []))
    monkeypatch.setattr(
        main, "build_market_breadth",
        lambda *args: main.build_offline_market_breadth(
            breadth_config, "2026-08-11", snapshots["sp500"]["daily_return"]
        ),
    )
    monkeypatch.setattr(main, "fetch_candidates", lambda sources, now: (healthy_candidates, []))
    monkeypatch.setattr(main, "filter_final_candidates", lambda candidates, now: candidates)
    monkeypatch.setattr(main, "dedupe_candidates", lambda candidates: candidates)
    monkeypatch.setattr(main, "stage_a_input_counts", lambda candidates: (len(candidates), len(candidates)))
    monkeypatch.setattr(
        main, "cluster_news_events",
        lambda *args, **kwargs: (fallback_events, clustering_reason),
    )
    monkeypatch.setattr(main, "build_event_representatives", lambda events, candidates: [])
    monkeypatch.setattr(main, "event_selection_candidates", lambda events: [])
    monkeypatch.setattr(main, "select_news_two_pass", lambda *args, **kwargs: ([], None))
    monkeypatch.setattr(main, "generate_market_summary", lambda *args, **kwargs: {"degraded": True})
    monkeypatch.setattr(main, "render_site", lambda *args, **kwargs: None)

    generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    report = json.loads((tmp_path / "data" / "reports" / "2026-08-12.json").read_text(encoding="utf-8"))
    assert not any("事件级去重" in warning for warning in report["warnings"])
    assert report["status"] != "partial"
    assert report["event_clustering_diagnostics"] == {
        "fallback_used": True, "reason": clustering_reason,
    }
