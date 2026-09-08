import json

import pytest

import src.main as main
from src.news_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    load_stage_b_snapshot,
    write_stage_b_snapshot,
)


def sample_snapshot():
    candidates = [
        {
            "candidate_id": "event-1",
            "title": "Fed holds rates",
            "summary": "The Federal Reserve held rates unchanged.",
            "source": "Reuters",
            "published_at": "2026-08-26T01:00:00+00:00",
            "event_summary": "The Fed held rates unchanged.",
            "topic_group": "US_MARKET_MACRO",
            "event_category": "other",
        }
    ]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "report_date": "2026-08-26",
        "run_at": "2026-08-26T10:00:00+08:00",
        "candidate_counts": {
            "rss_raw": 2,
            "within_24h": 2,
            "deduplicated": 1,
            "stage_a_pre_cap": 1,
            "stage_a_actual_input": 1,
            "stage_a_events": 1,
            "stage_b_input": 1,
        },
        "stage_a_events": [{"event_id": "event-1", "candidate_ids": ["event-1"]}],
        "stage_b": {
            "candidates": candidates,
            "recent_7_days_events": [],
            "market_context": {"core_market": {"sp500": {"valid": True}}},
        },
    }


def test_stage_b_snapshot_is_written_with_replayable_schema(tmp_path):
    path = write_stage_b_snapshot(tmp_path, sample_snapshot())

    assert path == tmp_path / "data" / "news_snapshots" / "2026-08-26.json"
    loaded = load_stage_b_snapshot(path)
    assert loaded["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert loaded["stage_b"]["candidates"] == sample_snapshot()["stage_b"]["candidates"]
    assert loaded["stage_a_events"] == sample_snapshot()["stage_a_events"]


def test_replay_restores_same_candidates_without_rss_or_clustering(tmp_path, monkeypatch):
    path = write_stage_b_snapshot(tmp_path, sample_snapshot())
    calls = {}

    def fail(name):
        def inner(*args, **kwargs):
            raise AssertionError(f"{name} must not run during replay")
        return inner

    monkeypatch.setattr(main, "fetch_candidates", fail("RSS"))
    monkeypatch.setattr(main, "cluster_candidates_local", fail("clustering"))

    def fake_score(candidates, api_key, focus_rules, **kwargs):
        calls["candidates"] = candidates
        calls["focus_rules"] = focus_rules
        return {}

    monkeypatch.setattr(main, "score_candidates", fake_score)

    result = main.replay_scoring_snapshot(path, api_key="test-key")

    # score_candidates() returned no results, so the one candidate surfaces
    # unscored (score=None) rather than being dropped -- see _merge_scores.
    assert [item["candidate_id"] for item in result] == ["event-1"]
    assert result[0]["score"] is None
    assert calls["candidates"] == sample_snapshot()["stage_b"]["candidates"]
    assert calls["focus_rules"]
    assert not (tmp_path / "data" / "reports").exists()
    assert not (tmp_path / "site").exists()


def test_snapshot_replay_preserves_old_report_schema(tmp_path, monkeypatch):
    report = {
        "report_date": "2026-08-25",
        "status": "ok",
        "market": {},
        "news": [],
        "warnings": [],
    }
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    report_path = reports_dir / "2026-08-25.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    path = write_stage_b_snapshot(tmp_path, sample_snapshot())

    monkeypatch.setattr(main, "score_candidates", lambda *args, **kwargs: {})
    main.replay_scoring_snapshot(path, api_key="test-key")

    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_production_pipeline_writes_snapshot_before_scoring(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    now = main.datetime.fromisoformat("2026-08-12T10:00:00").replace(tzinfo=main.SHANGHAI)
    market_config = main._load_yaml(main.ROOT / "config" / "market.yaml")
    breadth_config = main._load_yaml(main.ROOT / "config" / "market_breadth.yaml")
    snapshots, histories, contexts, market_warnings = main._offline_market(
        now, market_config["core"], market_config["context"]
    )
    candidate = sample_snapshot()["stage_b"]["candidates"][0]
    events = [{"event_id": "event-1", "candidate_ids": ["event-1"], "event_summary": candidate["event_summary"]}]
    selection_candidates = [candidate]

    monkeypatch.setattr(main, "fetch_market", lambda *args: (snapshots, histories, market_warnings))
    monkeypatch.setattr(main, "fetch_market_context", lambda *args: (contexts, []))
    monkeypatch.setattr(
        main, "build_market_breadth",
        lambda *args: main.build_offline_market_breadth(
            breadth_config, "2026-08-11", snapshots["sp500"]["daily_return"]
        ),
    )
    monkeypatch.setattr(main, "fetch_candidates", lambda *args: ([candidate], []))
    monkeypatch.setattr(main, "filter_final_candidates", lambda candidates, now: candidates)
    monkeypatch.setattr(main, "dedupe_candidates", lambda candidates: candidates)
    monkeypatch.setattr(main, "cluster_candidates_local", lambda candidates: events)
    event_representatives = [{**candidate, "candidate_ids": ["event-1"]}]
    monkeypatch.setattr(main, "build_event_representatives", lambda events, candidates: event_representatives)
    monkeypatch.setattr(main, "event_selection_candidates", lambda events: selection_candidates)
    monkeypatch.setattr(main, "generate_market_summary", lambda *args, **kwargs: {"degraded": True})
    monkeypatch.setattr(main, "render_site", lambda *args, **kwargs: None)

    def fake_score(candidates, *args, **kwargs):
        path = tmp_path / "data" / "news_snapshots" / "2026-08-12.json"
        assert path.exists()
        assert load_stage_b_snapshot(path)["stage_b"]["candidates"] == candidates
        return {}

    monkeypatch.setattr(main, "score_candidates", fake_score)
    main.generate_daily_report(base_dir=tmp_path, report_date="2026-08-12")

    snapshot = load_stage_b_snapshot(tmp_path / "data" / "news_snapshots" / "2026-08-12.json")
    assert snapshot["stage_b"]["candidates"] == selection_candidates
    assert snapshot["stage_a_events"] == events
