from datetime import datetime

import pytest

from src import deepseek_client
from src.news_events import (
    STAGE_A_MAX_INPUT,
    STAGE_B_MAX_INPUT,
    NewsEventError,
    _classify_fallback_topic,
    _cluster_candidate_input,
    _importance_signal,
    _merge_cross_batch_duplicate_events,
    _rank_stage_a_candidates,
    batch_stage_a_input,
    batch_stage_b_input,
    build_deterministic_fallback_selection,
    build_event_representatives,
    cluster_news_events,
    cluster_news_events_batched,
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
    assert captured["kwargs"] == {
        "thinking_enabled": False,
        "reasoning_effort": deepseek_client.NEWS_REASONING_EFFORT,
    }


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


def test_importance_signal_recognizes_spacex_as_mega_cap_tech():
    minor = {**candidate("spacex-minor"), "title": "SpaceX shares a new wallpaper for fans"}
    major = {**candidate("spacex-major"), "title": "SpaceX announces major new launch facility capital expenditure"}

    minor_score, minor_reason = _importance_signal(minor)
    major_score, major_reason = _importance_signal(major)

    assert "mega_cap_tech" in major_reason
    assert major_score > minor_score


def test_importance_signal_keyword_match_is_word_bounded_not_substring():
    """Regression for a real 2026-09-06 production false positive: "hikers"
    contains "hike" as a literal substring, which used to score an unrelated
    human-interest rescue story as an interest-rate-hike macro signal. Word-
    boundary matching must not treat "hike" as present inside "hikers"."""
    hikers_story = {
        **candidate("hikers"),
        "title": "Hikers rescued after using Google Gemini for planning",
        "summary": "The sheriff's office said the hikers were advised by Gemini "
                   "to bring far less food and water than their group required.",
    }

    score, reason = _importance_signal(hikers_story)

    assert "macro_rates" not in reason.split(",")


def test_importance_signal_keyword_match_still_allows_simple_plurals():
    """The word-boundary fix must not regress legitimate plural forms of a
    singular keyword (e.g. "semiconductor" matching "semiconductors")."""
    major = {**candidate("nvidia-major"), "title": "Nvidia faces major export controls on AI semiconductors"}

    score, reason = _importance_signal(major)

    assert score > 0
    assert "ai_chips" in reason


def test_geopolitics_policy_requires_both_region_and_economic_signal():
    """Regression for a real 2026-09-06 production false positive: a purely
    local/cultural policy story (a Ukrainian city weighing a Russian-language
    arts ban) matched the old flat keyword list via a bare region name plus a
    generic policy word ("ban"), even though it has no US-market relevance.
    geopolitics_policy must require a real economic/market-transmission signal
    (sanctions, tariffs, energy, war, export controls, ...) alongside the
    region -- not just region + ban/regulation/policy/law."""
    local_policy = {
        **candidate("odesa-language-ban"),
        "title": "Majority Russian-speaking city in Ukraine considers language ban in the arts",
        "summary": "Ukraine's third-biggest city Odesa decides this week whether to ban "
                   "Russian-language content in music and literature in public.",
    }
    real_sanctions = {
        **candidate("russia-sanctions"),
        "title": "New sanctions target Russian energy exports",
    }
    export_controls = {
        **candidate("china-export-controls"),
        "title": "US tightens export controls on chip sales to China",
    }

    local_score, local_reason = _importance_signal(local_policy)
    sanctions_score, sanctions_reason = _importance_signal(real_sanctions)
    export_score, export_reason = _importance_signal(export_controls)

    assert "geopolitics_policy" not in local_reason.split(",")
    assert "geopolitics_policy" in sanctions_reason
    assert "geopolitics_policy" in export_reason


def test_geopolitics_policy_still_recognizes_us_market_events():
    """Real economic/market-transmission geopolitical events (an oil-tanker
    strike, the Russia/Ukraine war, energy-supply disruption, and tariffs) must
    still be recognized after the two-tier tightening."""
    oil_tanker_strike = {
        **candidate("iran-oil-strike"),
        "title": "US strikes Iranian oil tankers amid rising Gulf tensions",
    }
    russia_ukraine_war = {
        **candidate("russia-ukraine-war"),
        "title": "Russia launches new missile strikes as Ukraine war escalates",
    }
    energy_disruption = {
        **candidate("energy-disruption"),
        "title": "Middle East conflict disrupts oil shipping through key strait",
    }
    tariffs = {
        **candidate("us-china-tariffs"),
        "title": "US announces new tariffs on Chinese semiconductor exports",
    }

    for item in (oil_tanker_strike, russia_ukraine_war, energy_disruption, tariffs):
        score, reason = _importance_signal(item)
        assert "geopolitics_policy" in reason, f"{item['title']!r} should still match: {reason}"


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


def test_cluster_exhausts_all_retries_then_uses_candidate_per_event_fallback():
    """Stage A gets one extra attempt beyond the shared DEEPSEEK_MAX_ATTEMPTS
    (3 total by default) because observed failures here are LLM structural-
    output-contract violations, not network errors, and are plausibly
    transient given LLM output non-determinism."""
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


def test_cluster_fallback_preserves_topic_classification_instead_of_all_other():
    """Regression for the real 2026-09-07 incident: when Stage A clustering
    fails entirely, every one of the 50 fallback events used to collapse to
    OTHER_SYSTEMIC/other, discarding all semantic topic information for the
    whole day. A mixed pool of real-signal and generic content must not all
    come back as OTHER_SYSTEMIC."""
    pool = [
        {**candidate("chips"), "title": "New GPU shortage squeezes AI data center buildout"},
        {**candidate("fed"), "title": "Fed holds interest rates steady amid inflation concerns"},
        {**candidate("oil"), "title": "Oil prices rise after OPEC production cut"},
        {**candidate("local"), "title": "Local town festival draws record crowds this weekend"},
    ]

    events, warning = cluster_news_events(
        pool, "key", call_model=lambda *args: "not json", sleep_fn=lambda _: None,
    )

    topic_groups = {event["candidate_ids"][0]: event["topic_group"] for event in events}
    assert topic_groups["chips"] == "AI_CHIPS"
    assert topic_groups["fed"] == "US_MARKET_MACRO"
    assert topic_groups["oil"] == "ENERGY_COMMODITIES"
    assert topic_groups["local"] == "OTHER_SYSTEMIC"
    assert not all(tg == "OTHER_SYSTEMIC" for tg in topic_groups.values())
    assert "事件级去重暂时失败" in warning


@pytest.mark.parametrize("company_title", [
    "Nvidia unveils new AI chip for data centers",
    "SpaceX Starship completes key orbital test flight",
    "Apple Bets on Foldable iPhone Under New CEO",
    "Microsoft raises AI data center capital expenditure guidance",
    "Alphabet's Google unveils new search feature",
    "Amazon expands same-day delivery network",
    "Tesla issues software recall for autopilot feature",
    "Meta announces new AI model amid heavy compute investment",
])
def test_classify_fallback_topic_recognizes_mega_cap_companies(company_title):
    """Mega-cap company names take priority over generic AI/chip language --
    a company's own AI product news still reads as MEGA_CAP_TECH."""
    topic_group, event_category = _classify_fallback_topic({**candidate("c"), "title": company_title})

    assert topic_group == "MEGA_CAP_TECH"
    assert event_category == "high_tech"


def test_classify_fallback_topic_recognizes_ai_chips_without_a_named_company():
    topic_group, event_category = _classify_fallback_topic(
        {**candidate("c"), "title": "New GPU shortage squeezes AI data center buildout"}
    )

    assert topic_group == "AI_CHIPS"
    assert event_category == "high_tech"


def test_classify_fallback_topic_recognizes_us_macro():
    topic_group, event_category = _classify_fallback_topic(
        {**candidate("c"), "title": "US jobs report shows hiring slowdown, unemployment ticks up"}
    )

    assert topic_group == "US_MARKET_MACRO"
    assert event_category == "macro_policy"


def test_classify_fallback_topic_recognizes_energy_commodities():
    topic_group, event_category = _classify_fallback_topic(
        {**candidate("c"), "title": "Oil prices rise after OPEC agrees to production cut"}
    )

    assert topic_group == "ENERGY_COMMODITIES"
    assert event_category == "financial_markets"


def test_classify_fallback_topic_recognizes_geopolitics_with_market_transmission():
    topic_group, event_category = _classify_fallback_topic(
        {**candidate("c"), "title": "New sanctions target Russian energy exports"}
    )

    assert topic_group == "GEOPOLITICS"
    assert event_category == "geopolitics"


def test_classify_fallback_topic_defaults_to_other_systemic_without_a_signal():
    topic_group, event_category = _classify_fallback_topic(
        {**candidate("c"), "title": "Local town festival draws record crowds this weekend"}
    )

    assert topic_group == "OTHER_SYSTEMIC"
    assert event_category == "other"


def test_classify_fallback_topic_does_not_reintroduce_substring_false_positives():
    """Must not regress the "hikers" (contains "hike") / "clearance rates"
    (contains "rates") substring bugs already fixed in the ranking pipeline --
    the fallback classifier reuses the same word-boundary/context-gated
    matching (_keyword_present)."""
    hikers_story = {
        **candidate("hikers"),
        "title": "Hikers rescued after using Google Gemini for planning",
        "summary": "The sheriff's office said the hikers were advised by Gemini "
                   "to bring far less food and water than their group required.",
    }
    local_property_story = {
        **candidate("property"),
        "title": "Australia Spring Property Season Begins Sharply Lower Than 2025",
        "summary": "Auction clearance rates held broadly steady, according to data from property researcher Cotality.",
    }

    hikers_topic, _ = _classify_fallback_topic(hikers_story)
    property_topic, _ = _classify_fallback_topic(local_property_story)

    # "google" is a recognized mega-cap company keyword, so this correctly
    # reads as MEGA_CAP_TECH -- the point is it must NOT be misclassified as
    # US_MARKET_MACRO via "hike" being a substring of "hikers".
    assert hikers_topic == "MEGA_CAP_TECH"
    assert property_topic == "OTHER_SYSTEMIC"


def test_cluster_recovers_on_final_extra_retry_attempt():
    """A third attempt succeeding (after two structural-contract failures) must
    return real clustered events, not the fallback, and no warning."""
    calls = []

    def flaky(system_prompt, user_payload, api_key):
        calls.append(1)
        if len(calls) < 3:
            return "not json"
        ids = [item["candidate_id"] for item in __import__("json").loads(user_payload)["candidates"]]
        return cluster_payload(event("event_001", ids))

    events, warning = cluster_news_events(
        [candidate("a"), candidate("b")], "key", call_model=flaky, sleep_fn=lambda _: None
    )

    assert len(calls) == 3
    assert warning is None
    assert events == [event("event_001", ["a", "b"])]


def test_cluster_max_attempts_is_configurable_independent_of_shared_stage_b_constant():
    calls = []

    def failing(*args):
        calls.append(1)
        return "not json"

    events, warning = cluster_news_events(
        [candidate("a"), candidate("b")], "key", call_model=failing, sleep_fn=lambda _: None, max_attempts=1,
    )

    assert len(calls) == 1
    assert [item["candidate_ids"] for item in events] == [["a"], ["b"]]
    assert "事件级去重暂时失败" in warning


def test_batch_stage_a_input_splits_large_pool_without_dropping_any_candidate():
    """Regression for the real 2026-09-07 incident: 72 dedup'd candidates were
    silently capped to the top 50 by composite score before ever reaching
    Stage A, permanently losing the other 22."""
    pool = [candidate(f"c{i}", priority="P2") for i in range(72)]

    batches = batch_stage_a_input(pool)
    all_batched_ids = {item["candidate_id"] for batch in batches for item in batch}

    assert len(batches) == 2
    assert all(len(batch) <= STAGE_A_MAX_INPUT for batch in batches)
    assert all_batched_ids == {item["candidate_id"] for item in pool}
    assert sum(len(batch) for batch in batches) == 72


def test_batch_stage_a_input_on_empty_pool_returns_no_batches():
    assert batch_stage_a_input([]) == []


def test_cluster_news_events_batched_covers_every_candidate_across_batches():
    """Every dedup'd candidate must reach some Stage A batch -- nothing
    silently dropped past a single-batch cap."""
    pool = [candidate(f"c{i}") for i in range(72)]

    def model(system_prompt, user_payload, api_key):
        ids = [item["candidate_id"] for item in __import__("json").loads(user_payload)["candidates"]]
        return cluster_payload(*[event(f"event_{cid}", [cid]) for cid in ids])

    observability = {}
    events, warning = cluster_news_events_batched(
        pool, "key", call_model=model, sleep_fn=lambda _: None, observability=observability,
    )

    assert observability["stage_a_total_candidate_count"] == 72
    assert observability["stage_a_batch_count"] == 2
    assert sum(observability["stage_a_batch_sizes"]) == 72
    assert observability["stage_a_uncovered_candidate_count"] == 0
    all_covered_ids = {cid for evt in events for cid in evt["candidate_ids"]}
    assert all_covered_ids == {item["candidate_id"] for item in pool}
    assert warning is None


def test_cluster_news_events_batched_one_batch_timeout_does_not_affect_other_batch():
    """One batch exhausting its retries (and degrading to its own local
    per-candidate fallback classification) must not prevent a different batch
    from completing its own real AI clustering."""
    pool = [candidate(f"c{i}") for i in range(72)]
    first_batch_ids = {item["candidate_id"] for item in batch_stage_a_input(pool)[0]}

    def model(system_prompt, user_payload, api_key):
        payload = __import__("json").loads(user_payload)
        ids = [item["candidate_id"] for item in payload["candidates"]]
        if set(ids) == first_batch_ids:
            raise TimeoutError("simulated batch 1 timeout")
        return cluster_payload(*[event(f"event_{cid}", [cid]) for cid in ids])

    observability = {}
    events, warning = cluster_news_events_batched(
        pool, "key", call_model=model, sleep_fn=lambda _: None, observability=observability,
    )

    assert observability["stage_a_batch_count"] == 2
    assert observability["stage_a_batch_fallback_used"] == [True, False]
    assert observability["stage_a_uncovered_candidate_count"] == 0
    all_covered_ids = {cid for evt in events for cid in evt["candidate_ids"]}
    assert all_covered_ids == {item["candidate_id"] for item in pool}
    # batch 2's candidates still went through real AI clustering (not fallback)
    second_batch_ids = {item["candidate_id"] for item in batch_stage_a_input(pool)[1]}
    second_batch_events = [evt for evt in events if evt["candidate_ids"][0] in second_batch_ids]
    assert all(not evt["event_id"].startswith("fallback_") for evt in second_batch_events)


def test_merge_cross_batch_duplicate_events_combines_near_identical_summaries():
    """Two different batches independently clustering two different articles
    about the same real event must be merged into one event before reaching
    Stage B, not duplicated."""
    events = [
        {"event_id": "fallback_001", "candidate_ids": ["a"],
         "event_summary": "Fed holds interest rates steady after September meeting",
         "topic_group": "OTHER_SYSTEMIC", "event_category": "other"},
        {"event_id": "fallback_014", "candidate_ids": ["b"],
         "event_summary": "Fed holds interest rate steady after the September meeting",
         "topic_group": "US_MARKET_MACRO", "event_category": "macro_policy"},
    ]

    merged = _merge_cross_batch_duplicate_events(events)

    assert len(merged) == 1
    assert set(merged[0]["candidate_ids"]) == {"a", "b"}
    # prefers the successfully-classified topic_group over the fallback one
    assert merged[0]["topic_group"] == "US_MARKET_MACRO"


def test_merge_cross_batch_duplicate_events_keeps_distinct_events_separate():
    events = [
        {"event_id": "e1", "candidate_ids": ["a"], "event_summary": "Fed holds interest rates steady",
         "topic_group": "US_MARKET_MACRO", "event_category": "macro_policy"},
        {"event_id": "e2", "candidate_ids": ["b"], "event_summary": "Nvidia unveils new AI chip for data centers",
         "topic_group": "AI_CHIPS", "event_category": "high_tech"},
    ]

    merged = _merge_cross_batch_duplicate_events(events)

    assert len(merged) == 2


def stage_b_candidate(cid, priority="P1", title=None, topic_group="OTHER_SYSTEMIC",
                      event_summary=None, published_at="2026-08-12T10:00:00+00:00"):
    """Shape one Stage B input item -- i.e. what event_selection_candidates()
    actually hands to batch_stage_b_input / Stage B, not a raw RSS candidate."""
    return {
        "candidate_id": cid,
        "source": "Source",
        "priority": priority,
        "title": title or f"Title {cid}",
        "summary": "summary",
        "published_at": published_at,
        "url": f"https://example.com/{cid}",
        "topic_group": topic_group,
        "event_category": "other",
        "event_summary": event_summary or f"事件 {cid} 的中文摘要。",
    }


def test_batch_stage_b_input_keeps_pool_in_one_batch_at_or_below_size():
    pool = [stage_b_candidate(f"c{i}") for i in range(STAGE_B_MAX_INPUT)]

    batches = batch_stage_b_input(pool)

    assert len(batches) == 1
    assert {item["candidate_id"] for item in batches[0]} == {item["candidate_id"] for item in pool}


def test_batch_stage_b_input_splits_large_pool_without_dropping_any_candidate():
    """Regression for the real 2026-09-07 incident: Stage A produced 50 events,
    but the old select_stage_b_input silently dropped everything past the
    first 28 by composite score, so 22 candidates (including a real Nvidia/
    Hugging Face acquisition story) never got a Stage B judgment at all."""
    pool = [stage_b_candidate(f"c{i}", priority="P2") for i in range(50)]

    batches = batch_stage_b_input(pool)
    all_batched_ids = {item["candidate_id"] for batch in batches for item in batch}

    assert len(batches) == 2
    assert all(len(batch) <= STAGE_B_MAX_INPUT for batch in batches)
    assert all_batched_ids == {item["candidate_id"] for item in pool}
    assert sum(len(batch) for batch in batches) == 50


def test_batch_stage_b_input_orders_high_importance_events_into_the_first_batch():
    """The highest-composite-score candidates should land in batch 1 -- the
    batch guaranteed to run even under the tightest time budget -- mirroring
    the Stage A cap's own priority-vs-importance ranking."""
    ordinary = [
        stage_b_candidate(f"ordinary-{i}", priority="P0", title="Routine local company update")
        for i in range(STAGE_B_MAX_INPUT + 10)
    ]
    important = [
        stage_b_candidate("fed-rates", priority="P1",
                          title="Fed signals rate cut path after inflation and jobs data surprise",
                          topic_group="US_MARKET_MACRO"),
        stage_b_candidate("nvidia-chip", priority="P1",
                          title="Nvidia unveils new AI chip for data centers",
                          topic_group="AI_CHIPS"),
    ]

    batches = batch_stage_b_input(ordinary + important)
    first_batch_ids = {item["candidate_id"] for item in batches[0]}

    assert {"fed-rates", "nvidia-chip"}.issubset(first_batch_ids)
    # still nothing dropped overall
    all_ids = {item["candidate_id"] for batch in batches for item in batch}
    assert all_ids == {item["candidate_id"] for item in ordinary + important}


def test_batch_stage_b_input_on_empty_pool_returns_no_batches():
    assert batch_stage_b_input([]) == []


def test_build_deterministic_fallback_selection_on_empty_pool_returns_empty():
    assert build_deterministic_fallback_selection([]) == []


def test_build_deterministic_fallback_selection_returns_empty_when_nothing_is_high_confidence():
    """Quality over quantity: a batch of ordinary, ineligible news must return
    [] -- no padding to any minimum count. Real 2026-09-07 examples that must
    NOT be fallback-eligible."""
    pool = [
        stage_b_candidate("turkey-gdp", priority="P0", title="Turkey Cuts 2027 GDP Growth Forecast as Elections Beckon"),
        stage_b_candidate("germany-election", priority="P0", title="Germany's far-right AfD set for big win in eastern state"),
        stage_b_candidate("parenting", priority="P0", title="Parents who are really good at handling tantrums do 5 things"),
        stage_b_candidate("sugar", priority="P0", title="Sugar is outperforming the stock market this year"),
    ]

    result = build_deterministic_fallback_selection(pool)

    assert result == []


def test_build_deterministic_fallback_selection_rejects_ordinary_foreign_gdp_news():
    story = stage_b_candidate("gdp", title="Turkey Cuts 2027 GDP Growth Forecast as Elections Beckon")

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_rejects_ordinary_local_politics():
    story = stage_b_candidate("election", title="Germany's far-right AfD set for big win in eastern state")

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_rejects_ordinary_commodity_price_news():
    story = stage_b_candidate("sugar", title="Sugar is outperforming the stock market this year")

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_ignores_incidental_region_mention_in_summary():
    """Regression for a real false positive found via 2026-09-07 replay: an
    ordinary Turkey-domestic-GDP story's raw article `summary` happened to
    mention "fallout from the Iran war reverberates" purely as passing
    background color, which alone used to satisfy the geopolitics region+
    signal check even though the story is not about Iran or the war. Matching
    is restricted to title+event_summary (Stage A's own distilled factual
    restatement) specifically to avoid this."""
    story = stage_b_candidate(
        "turkey-gdp-full", title="Turkey Cuts 2027 GDP Growth Forecast as Elections Beckon",
        event_summary="Turkey cut its 2027 GDP growth forecast.",
    )
    story["summary"] = (
        "Turkey trimmed its 2027 gross domestic product growth forecast, signaling a delicate "
        "balancing act between price stability and growth as fallout from the Iran war reverberates "
        "and a potential election beckons."
    )

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_rejects_non_spacex_company_crash_news():
    """Regression for a real false positive found via 2026-09-07 replay: "Five
    dead after Amazon cargo plane crashes at Miami airport" matched the old
    flat major-event list via "Amazon" + "crash", even though a cargo-flight
    accident is not a material Amazon business event. "crash"/"explosion"/
    "mission failure" only count for SpaceX, whose entire business is
    spaceflight."""
    story = stage_b_candidate("amazon-crash", title="Five dead after Amazon cargo plane crashes at Miami airport")

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_keeps_fed_and_rejects_bare_rates_jobs():
    fed_story = stage_b_candidate("fed", title="Federal Reserve holds interest rates steady after September meeting")
    bare_words_story = stage_b_candidate(
        "retailer", title="Retailer reports strong jobs growth and raises hourly rates for staff",
    )

    fed_result = build_deterministic_fallback_selection([fed_story])
    bare_result = build_deterministic_fallback_selection([bare_words_story])

    assert len(fed_result) == 1
    assert fed_result[0]["selection_reason"].find("us_macro_fed_treasury") != -1
    assert bare_result == []


def test_build_deterministic_fallback_selection_keeps_nonfarm_payroll_and_treasury():
    payroll_story = stage_b_candidate("payroll", title="US nonfarm payroll report shows hiring slowdown")
    treasury_story = stage_b_candidate("treasury", title="US Treasury yield jumps after auction")

    assert len(build_deterministic_fallback_selection([payroll_story])) == 1
    assert len(build_deterministic_fallback_selection([treasury_story])) == 1


def test_build_deterministic_fallback_selection_keeps_mega_cap_major_acquisition():
    story = stage_b_candidate("nvidia-acq", title="Nvidia announces major acquisition of AI infrastructure startup")

    result = build_deterministic_fallback_selection([story])

    assert len(result) == 1
    assert "mega_cap_major_event" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_rejects_mega_cap_company_name_alone():
    """A company name alone -- with no major event type -- must not qualify,
    mirroring the same 2026-09-07 "Model fatigue"/Stage-A lesson applied here."""
    story = stage_b_candidate("nvidia-mention", title="Analysts discuss Nvidia's role in the broader AI trade")

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_rejects_ordinary_lawsuit_mention():
    """Regression for the real 2026-09-07 Seattle Times/Newsday copyright suit:
    bare "sue"/"lawsuit" is deliberately not a qualifying major-event signal
    here (this deterministic filter cannot judge lawsuit materiality the way
    the Stage B prompt's "重大诉讼或和解" standard can)."""
    story = stage_b_candidate(
        "lawsuit", title="Seattle Times and Newsday sue OpenAI and Microsoft for infringement",
    )

    assert build_deterministic_fallback_selection([story]) == []


def test_build_deterministic_fallback_selection_keeps_spacex_mission_failure():
    story = stage_b_candidate("spacex-failure", title="SpaceX Starship suffers mission failure during test flight")

    result = build_deterministic_fallback_selection([story])

    assert len(result) == 1
    assert "mega_cap_major_event" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_rejects_ceo_mention_without_leadership_change():
    """Regression for a real false positive found via 2026-09-07 replay:
    "Apple's New CEO-Like Cook Pay Package Shows He's Going Nowhere" matched
    the old bare "ceo" term even though the story is explicitly about Cook
    *staying*, not any leadership change. A compensation-package writeup,
    interview, or routine appearance must not qualify just because the CEO is
    mentioned."""
    pay_package = stage_b_candidate(
        "apple-ceo-pay", title="Apple's New CEO-Like Cook Pay Package Shows He's Going Nowhere; Sept. 9 Event",
    )
    interview = stage_b_candidate("tesla-ceo-interview", title="Tesla CEO discusses strategy in new interview")

    assert build_deterministic_fallback_selection([pay_package]) == []
    assert build_deterministic_fallback_selection([interview]) == []


def test_build_deterministic_fallback_selection_keeps_real_ceo_change():
    resignation = stage_b_candidate("ceo-resigns", title="Meta CEO announces resignation effective next quarter")
    replacement = stage_b_candidate("ceo-replaced", title="Microsoft board replaced its CEO after strategy dispute")
    succession = stage_b_candidate("ceo-succession", title="Apple names new CEO in succession plan announcement")

    for story in (resignation, replacement, succession):
        result = build_deterministic_fallback_selection([story])
        assert len(result) == 1, story["title"]
        assert "mega_cap_major_event" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_geopolitics_requires_both_region_and_market_signal():
    region_only = stage_b_candidate("region-only", title="Ukraine considers new language policy for public arts")
    signal_only = stage_b_candidate("signal-only", title="Global shipping firms brace for new tariff rules")
    both = stage_b_candidate("both", title="New sanctions target Russian energy exports")

    assert build_deterministic_fallback_selection([region_only]) == []
    assert build_deterministic_fallback_selection([signal_only]) == []
    result = build_deterministic_fallback_selection([both])
    assert len(result) == 1
    assert "geopolitics_region_and_market_transmission" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_rejects_tariff_commentary_without_policy_action():
    """Regression for a real false positive found via 2026-09-07 replay:
    "Rep. Stevens Says Canadian Tariffs Are Squeezing Michigan" matched the
    base region+signal geopolitics check, but the story is a politician's
    commentary on an *existing* tariff, not a new policy action. Deliberately
    not naming the real politician/state in the test title, per instructions
    not to hardcode this specific case."""
    commentary = stage_b_candidate(
        "tariff-commentary", title="Lawmaker says existing Canadian tariffs are squeezing local manufacturers",
    )
    concern = stage_b_candidate("tariff-concern", title="Analyst warns of concern about Chinese trade tariffs")

    assert build_deterministic_fallback_selection([commentary]) == []
    assert build_deterministic_fallback_selection([concern]) == []


def test_build_deterministic_fallback_selection_keeps_real_tariff_policy_action():
    imposed = stage_b_candidate("tariff-imposed", title="US imposed new tariffs on Chinese semiconductor imports")
    announced = stage_b_candidate("sanctions-announced", title="US announced new sanctions on Russian banks")
    removed = stage_b_candidate("export-control-removed", title="US removed export controls on Canadian chip exports")

    for story in (imposed, announced, removed):
        result = build_deterministic_fallback_selection([story])
        assert len(result) == 1, story["title"]
        assert "geopolitics_region_and_market_transmission" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_geopolitics_kinetic_signal_needs_no_policy_action():
    """A kinetic/already-happened event (a strike, a shipping disruption) needs
    no extra "announced/imposed" confirmation -- the verb itself already
    describes something that happened; only policy-type signals (tariff/
    sanction/export control/trade/banking) need that extra bar."""
    story = stage_b_candidate("military-strike", title="Iran launches missile strikes amid escalating conflict")

    result = build_deterministic_fallback_selection([story])

    assert len(result) == 1
    assert "geopolitics_region_and_market_transmission" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_energy_requires_both_core_and_disruption():
    oil_only = stage_b_candidate("oil-only", title="Latest oil market news and analysis for the week")
    disruption_only = stage_b_candidate("disruption-only", title="Major supply disruption hits regional factories")
    both = stage_b_candidate("both", title="Oil prices surge after major pipeline attack disrupts supply")

    assert build_deterministic_fallback_selection([oil_only]) == []
    assert build_deterministic_fallback_selection([disruption_only]) == []
    result = build_deterministic_fallback_selection([both])
    assert len(result) == 1
    assert "energy_supply_disruption" in result[0]["selection_reason"]


def test_build_deterministic_fallback_selection_is_deterministic_and_repeatable():
    pool = [
        stage_b_candidate("fed", title="Federal Reserve holds interest rates steady"),
        stage_b_candidate("nvidia", title="Nvidia announces major acquisition of AI startup"),
        stage_b_candidate("sanctions", title="New sanctions target Russian energy exports"),
    ]

    first = build_deterministic_fallback_selection(pool)
    second = build_deterministic_fallback_selection(pool)

    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert len(first) == 3


def test_build_deterministic_fallback_selection_never_calls_ai_or_fabricates_fields():
    pool = [stage_b_candidate(
        "fed", title="Federal Reserve holds interest rates steady", event_summary="美联储维持利率不变。",
    )]

    result = build_deterministic_fallback_selection(pool)

    assert len(result) == 1
    item = result[0]
    assert item["title_zh"] == "美联储维持利率不变。"
    assert item["summary_zh"] == "美联储维持利率不变。"
    assert item["fallback"] is True
    assert item["selection_mode"] == "deterministic_fallback"
    assert item["investment_relevance_score"] is None
    assert item["tags"] == []
    assert "确定性降级选取" in item["selection_reason"]


def test_build_deterministic_fallback_selection_falls_back_to_english_title_without_event_summary():
    pool = [{**stage_b_candidate("nvidia", title="Nvidia announces major acquisition of AI startup"), "event_summary": ""}]

    result = build_deterministic_fallback_selection(pool)

    assert result[0]["title_zh"] == "Nvidia announces major acquisition of AI startup"
