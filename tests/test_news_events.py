from datetime import datetime

import pytest

from src.news_events import (
    NewsEventError,
    _cluster_candidate_input,
    _importance_signal,
    _rank_stage_a_candidates,
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


def event(event_id, candidate_ids, summary="A factual event.", topic_group="US_MARKET_MACRO", event_category="other"):
    return {
        "event_id": event_id,
        "candidate_ids": candidate_ids,
        "event_summary": summary,
        "topic_group": topic_group,
        "event_category": event_category,
    }


def test_validates_complete_non_overlapping_event_clusters():
    candidates = [candidate("a"), candidate("b"), candidate("c")]

    events = validate_event_clusters(cluster_payload(
        event("event_001", ["a", "b"]),
        event("event_002", ["c"], topic_group="AI_CHIPS"),
    ), candidates)

    assert [item["event_id"] for item in events] == ["event_001", "event_002"]


def test_event_category_is_factual_stage_a_metadata_and_survives_program_mapping():
    pool = [candidate("a")]
    clusters = cluster_payload({
        **event("event_001", ["a"]),
        "event_category": "financial_markets",
    })

    events = validate_event_clusters(clusters, pool)
    flattened = event_selection_candidates(build_event_representatives(events, pool))

    assert flattened[0]["event_category"] == "financial_markets"


def test_rejects_invalid_factual_event_category():
    with pytest.raises(NewsEventError):
        validate_event_clusters(cluster_payload({
            **event("event_001", ["a"]),
            "event_category": "world_news",
        }), [candidate("a")])


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
        "event_category": "other",
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

    def model(system_prompt, user_payload, api_key, **kwargs):
        captured["prompt"] = system_prompt
        captured["payload"] = __import__("json").loads(user_payload)
        captured["kwargs"] = kwargs
        return cluster_payload(event("event_001", ["a", "b"], "Fed held rates"))

    events, warning = cluster_news_events(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert events[0]["candidate_ids"] == ["a", "b"]
    assert captured["payload"] == {"candidates": [{
        key: item[key] for key in ("candidate_id", "source", "priority", "title", "summary", "published_at")
    } for item in pool]}
    assert "https://example.com" not in __import__("json").dumps(captured["payload"])
    assert "现实世界事件聚类" in captured["prompt"]
    assert captured["kwargs"] == {"thinking_enabled": False, "reasoning_effort": None}


def test_stage_a_logs_input_and_complete_output_events(capsys):
    pool = [candidate("a", summary="Fed held rates."), candidate("b", summary="Fed held rates unchanged.")]

    def model(system_prompt, user_payload, api_key):
        return cluster_payload(event("event_001", ["a", "b"], "Fed held rates", "US_MARKET_MACRO", "macro_policy"))

    events, warning = cluster_news_events(pool, "secret-api-key", call_model=model, sleep_fn=lambda _: None)

    output = capsys.readouterr().out
    assert warning is None
    assert events[0]["event_id"] == "event_001"
    assert "[NEWS STAGE A] input candidates: 2" in output
    assert "event_id=event_001 | category=macro_policy | title=Fed held rates" in output
    assert "secret-api-key" not in output


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


def test_pre_cap_ranking_reserves_same_priority_high_value_candidates_from_source_flood():
    ordinary = [candidate(f"bbc-{index}", priority="P0", published_at=f"2026-08-24T{12 + index % 10:02d}:00:00+00:00")
                for index in range(50)]
    valuable = [
        {**candidate("treasury", priority="P0", published_at="2026-08-23T01:00:00+00:00"),
         "source": "The Guardian Business", "title": "US Treasury bond yields jump after policy signal"},
        {**candidate("nvidia", priority="P0", published_at="2026-08-23T02:00:00+00:00"),
         "source": "Ars Technica", "title": "Nvidia unveils new AI semiconductor for data centers"},
        {**candidate("apple", priority="P0", published_at="2026-08-23T03:00:00+00:00"),
         "source": "The Verge", "title": "Apple announces major iPhone strategy change"},
    ]

    ranked = _cluster_candidate_input(ordinary + valuable)
    ids = [item["candidate_id"] for item in ranked]

    assert {"treasury", "nvidia", "apple"}.issubset(ids)


def test_pre_cap_ranking_keeps_priority_for_equal_importance():
    pool = [
        {**candidate("p0-ordinary", priority="P0"), "title": "Routine company update"},
        {**candidate("p1-ordinary", priority="P1"), "title": "Routine company update"},
        {**candidate("p2-ordinary", priority="P2"), "title": "Routine company update"},
    ]

    ranked = _cluster_candidate_input(pool)

    assert [item["candidate_id"] for item in ranked] == ["p0-ordinary", "p1-ordinary", "p2-ordinary"]


def test_high_importance_p1_can_beat_low_importance_p0():
    pool = [
        {**candidate("p0-low", priority="P0"), "title": "Routine local company update"},
        {**candidate("p1-high", priority="P1"),
         "title": "Treasury yields surge after inflation and jobs data shift Fed rate outlook"},
    ]

    ranked = _cluster_candidate_input(pool)

    assert [item["candidate_id"] for item in ranked] == ["p1-high", "p0-low"]


def test_equal_importance_still_orders_p0_before_p1_before_p2():
    pool = [
        {**candidate("p2", priority="P2"), "title": "Routine local company update"},
        {**candidate("p1", priority="P1"), "title": "Routine local company update"},
        {**candidate("p0", priority="P0"), "title": "Routine local company update"},
    ]

    ranked = _cluster_candidate_input(pool)

    assert [item["candidate_id"] for item in ranked] == ["p0", "p1", "p2"]


def test_high_value_p1_enters_cap_when_ordinary_p0_candidates_exceed_cap():
    ordinary = [
        {**candidate(f"p0-{index}", priority="P0"), "title": f"Routine local company update {index}"}
        for index in range(55)
    ]
    valuable = [
        {**candidate("p1-treasury", priority="P1"),
         "title": "Treasury yields surge after inflation and jobs data shift Fed rate outlook"},
        {**candidate("p1-trade", priority="P1"),
         "title": "US Canada trade talks collapse as retaliatory tariffs are announced"},
    ]

    ranked = _cluster_candidate_input(ordinary + valuable)

    assert {"p1-treasury", "p1-trade"}.issubset({item["candidate_id"] for item in ranked})


def test_low_value_p1_does_not_beat_clearly_important_p0():
    pool = [
        {**candidate("p0-high", priority="P0"),
         "title": "Treasury yields surge after inflation and jobs data shift Fed rate outlook"},
        {**candidate("p1-low", priority="P1"), "title": "Routine local company update"},
    ]

    ranked = _cluster_candidate_input(pool)

    assert [item["candidate_id"] for item in ranked] == ["p0-high", "p1-low"]


def test_importance_keyword_requires_event_significance_for_mega_cap_promotion():
    promo = {**candidate("apple-promo"), "title": "Apple's four-pack of second-gen AirTags is $20 off"}
    major = {**candidate("nvidia-major"), "title": "Nvidia faces major export controls on AI semiconductors"}

    promo_score, _ = _importance_signal(promo)
    major_score, _ = _importance_signal(major)

    assert major_score > promo_score


def test_pre_cap_ranking_softly_diversifies_quality_matched_sources():
    pool = [
        {**candidate(f"bbc-{index}", priority="P0", published_at="2026-08-24T12:00:00+00:00"),
         "source": "BBC News", "title": f"Market update {index}"}
        for index in range(30)
    ] + [
        {**candidate(f"ars-{index}", priority="P0", published_at="2026-08-24T12:00:00+00:00"),
         "source": "Ars Technica", "title": f"Market update {index}"}
        for index in range(30)
    ]

    ranked = _cluster_candidate_input(pool)
    first_ten_sources = [item["source"] for item in ranked[:10]]

    assert first_ten_sources.count("BBC News") < 9
    assert first_ten_sources.count("Ars Technica") > 1


def test_pre_cap_ranking_does_not_penalize_later_high_importance_candidate():
    pool = [
        {**candidate(f"tech-{index}", priority="P0", published_at="2026-08-24T12:00:00+00:00"),
         "source": "TechCrunch", "title": f"AI product update {index}"}
        for index in range(50)
    ]
    pool.append({**candidate("late-macro", priority="P0", published_at="2026-08-23T01:00:00+00:00"),
                 "source": "TechCrunch", "title": "US inflation report shifts Treasury bond yield outlook"})

    ranked = _cluster_candidate_input(pool)

    assert "late-macro" in [item["candidate_id"] for item in ranked]


def test_cluster_logs_cap_counts_and_each_cap_drop(capsys):
    pool = [candidate(f"candidate-{index}", priority="P2", published_at=f"2026-08-12T{index % 24:02d}:00:00+00:00")
            for index in range(79)]

    def model(system_prompt, user_payload, api_key):
        ids = [item["candidate_id"] for item in __import__("json").loads(user_payload)["candidates"]]
        return cluster_payload(event("event_001", ids))

    cluster_news_events(pool, "key", call_model=model, sleep_fn=lambda _: None)

    output = capsys.readouterr().out
    assert "[NEWS STAGE A CAP] pre_cap=79 actual_input=50 cap_dropped=29" in output
    assert output.count("stage=stage_a_cap | action=drop | reason=input_cap_50") == 29
    assert output.count("stage=stage_a_cap | action=keep") == 50
    assert "[NEWS STAGE A MAPPING] candidate_id=candidate-" in output


def test_cluster_two_failures_uses_candidate_per_event_fallback():
    calls, sleeps = [], []

    def failing(*args):
        calls.append(1)
        return "not json"

    events, warning = cluster_news_events(
        [candidate("a"), candidate("b")], "key", call_model=failing, sleep_fn=sleeps.append
    )

    assert len(calls) == 2
    assert sleeps == [5]
    assert [item["candidate_ids"] for item in events] == [["a"], ["b"]]
    assert all(item["topic_group"] == "OTHER_SYSTEMIC" for item in events)
    assert "事件级去重暂时失败" in warning
