from datetime import datetime

import pytest

from src.news_events import (
    NewsEventError,
    build_event_representatives,
    cluster_news_events,
    event_selection_candidates,
    select_event_representative,
    validate_event_clusters,
)


def candidate(cid, priority="P1", summary="summary", published_at="2026-08-12T10:00:00+00:00"):
    return {
        "candidate_id": cid,
        "source": "Source",
        "priority": priority,
        "title": f"Title {cid}",
        "summary": summary,
        "published_at": published_at,
        "url": f"https://example.com/{cid}",
    }


def cluster_payload(*events):
    return {"events": list(events)}


def event(event_id, candidate_ids, summary="A factual event.", topic_group="US_MARKET_MACRO"):
    return {
        "event_id": event_id,
        "candidate_ids": candidate_ids,
        "event_summary": summary,
        "topic_group": topic_group,
    }


def test_validates_complete_non_overlapping_event_clusters():
    candidates = [candidate("a"), candidate("b"), candidate("c")]

    events = validate_event_clusters(cluster_payload(
        event("event_001", ["a", "b"]),
        event("event_002", ["c"], topic_group="AI_CHIPS"),
    ), candidates)

    assert [item["event_id"] for item in events] == ["event_001", "event_002"]


@pytest.mark.parametrize("payload", [
    cluster_payload(event("event_001", ["missing"])),
    cluster_payload(event("event_001", ["a"]), event("event_002", ["a", "b"])),
    cluster_payload(event("event_001", ["a"])),
    cluster_payload(event("event_001", ["a", "b"], topic_group="SPORTS")),
    cluster_payload(event("event_001", ["a", "b"], summary="   ")),
    cluster_payload(event("event_001", ["a"]), event("event_001", ["b"])),
])
def test_rejects_invalid_event_cluster_contract(payload):
    with pytest.raises(NewsEventError):
        validate_event_clusters(payload, [candidate("a"), candidate("b")])


def test_representative_prefers_priority_then_summary_completeness_then_newness():
    pool = [
        candidate("p1", priority="P1", summary="A much longer summary than P0."),
        candidate("p0-short", priority="P0", summary="short"),
        candidate("p0-full-old", priority="P0", summary="A full detailed summary", published_at="2026-08-12T09:00:00+00:00"),
        candidate("p0-full-new", priority="P0", summary="A full detailed summary", published_at="2026-08-12T11:00:00+00:00"),
    ]

    selected = select_event_representative({"candidate_ids": [item["candidate_id"] for item in pool]}, pool)

    assert selected["candidate_id"] == "p0-full-new"


def test_build_event_representatives_keeps_program_owned_event_metadata():
    pool = [candidate("a", priority="P0"), candidate("b", priority="P1")]
    events = validate_event_clusters(cluster_payload(event("event_001", ["a", "b"], "Fed held rates", "US_MARKET_MACRO")), pool)

    representatives = build_event_representatives(events, pool)

    assert representatives == [{
        "event_id": "event_001",
        "event_summary": "Fed held rates",
        "topic_group": "US_MARKET_MACRO",
        "candidate_ids": ["a", "b"],
        "representative": pool[0],
    }]


def test_event_selection_candidates_expose_only_the_representative_with_event_metadata():
    pool = [candidate("a", priority="P0"), candidate("b", priority="P1")]
    event_representatives = build_event_representatives(
        validate_event_clusters(cluster_payload(event("event_001", ["a", "b"], "Fed held rates", "US_MARKET_MACRO")), pool),
        pool,
    )

    selected = event_selection_candidates(event_representatives)

    assert selected[0]["candidate_id"] == "a"
    assert selected[0]["event_summary"] == "Fed held rates"
    assert selected[0]["topic_group"] == "US_MARKET_MACRO"


def test_cluster_calls_shared_transport_without_urls_and_returns_validated_events():
    captured = {}
    pool = [candidate("a"), candidate("b")]

    def model(system_prompt, user_payload, api_key):
        captured["prompt"] = system_prompt
        captured["payload"] = __import__("json").loads(user_payload)
        return cluster_payload(event("event_001", ["a", "b"], "Fed held rates"))

    events, warning = cluster_news_events(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert events[0]["candidate_ids"] == ["a", "b"]
    assert captured["payload"] == {"candidates": [{
        key: item[key] for key in ("candidate_id", "source", "priority", "title", "summary", "published_at")
    } for item in pool]}
    assert "https://example.com" not in __import__("json").dumps(captured["payload"])
    assert "现实世界事件聚类" in captured["prompt"]


def test_cluster_skips_model_when_at_most_one_candidate():
    calls = []
    events, warning = cluster_news_events(
        [candidate("a")], "key", call_model=lambda *args: calls.append(args), sleep_fn=lambda _: None
    )

    assert calls == []
    assert warning is None
    assert events[0]["candidate_ids"] == ["a"]
    assert events[0]["topic_group"] == "OTHER_SYSTEMIC"


def test_cluster_caps_input_to_fifty_by_priority_and_recency():
    captured = {}
    pool = [candidate(f"p2-{index}", priority="P2", published_at=f"2026-08-12T{index % 24:02d}:00:00+00:00") for index in range(55)]
    pool.append(candidate("p0", priority="P0", published_at="2026-08-11T00:00:00+00:00"))

    def model(system_prompt, user_payload, api_key):
        captured["payload"] = __import__("json").loads(user_payload)
        ids = [item["candidate_id"] for item in captured["payload"]["candidates"]]
        return cluster_payload(event("event_001", ids))

    events, warning = cluster_news_events(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert len(captured["payload"]["candidates"]) == 50
    assert "p0" in events[0]["candidate_ids"]


def test_cluster_three_failures_uses_candidate_per_event_fallback():
    calls, sleeps = [], []

    def failing(*args):
        calls.append(1)
        return "not json"

    events, warning = cluster_news_events(
        [candidate("a"), candidate("b")], "key", call_model=failing, sleep_fn=sleeps.append
    )

    assert len(calls) == 3
    assert sleeps == [5, 10]
    assert [item["candidate_ids"] for item in events] == [["a"], ["b"]]
    assert all(item["topic_group"] == "OTHER_SYSTEMIC" for item in events)
    assert "事件级去重暂时失败" in warning
