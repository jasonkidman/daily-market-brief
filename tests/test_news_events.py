import pytest

from src.news_events import (
    _classify_fallback_topic,
    _importance_signal,
    _rank_stage_a_candidates,
    build_event_representatives,
    event_selection_candidates,
    select_event_representative,
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


def event(event_id, candidate_ids, summary="A factual event.", topic_group="US_MARKET_MACRO", event_category="other"):
    return {
        "event_id": event_id,
        "candidate_ids": candidate_ids,
        "event_summary": summary,
        "topic_group": topic_group,
        "event_category": event_category,
    }


def _ranked_ids(candidates):
    return [item["candidate_id"] for item, *_ in _rank_stage_a_candidates(candidates)]


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
    events = [event("event_001", ["a", "b"], "Fed held rates", "US_MARKET_MACRO")]

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
        [event("event_001", ["a", "b"], "Fed held rates", "US_MARKET_MACRO")], pool,
    )

    selected = event_selection_candidates(event_representatives)

    assert selected[0]["candidate_id"] == "a"
    assert selected[0]["event_summary"] == "Fed held rates"
    assert selected[0]["topic_group"] == "US_MARKET_MACRO"


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

    ids = _ranked_ids(ordinary + valuable)

    assert {"treasury", "nvidia", "apple"}.issubset(ids)


def test_pre_cap_ranking_keeps_priority_for_equal_importance():
    pool = [
        {**candidate("p0-ordinary", priority="P0"), "title": "Routine company update"},
        {**candidate("p1-ordinary", priority="P1"), "title": "Routine company update"},
        {**candidate("p2-ordinary", priority="P2"), "title": "Routine company update"},
    ]

    assert _ranked_ids(pool) == ["p0-ordinary", "p1-ordinary", "p2-ordinary"]


def test_high_importance_p1_can_beat_low_importance_p0():
    pool = [
        {**candidate("p0-low", priority="P0"), "title": "Routine local company update"},
        {**candidate("p1-high", priority="P1"),
         "title": "Treasury yields surge after inflation and jobs data shift Fed rate outlook"},
    ]

    assert _ranked_ids(pool) == ["p1-high", "p0-low"]


def test_equal_importance_still_orders_p0_before_p1_before_p2():
    pool = [
        {**candidate("p2", priority="P2"), "title": "Routine local company update"},
        {**candidate("p1", priority="P1"), "title": "Routine local company update"},
        {**candidate("p0", priority="P0"), "title": "Routine local company update"},
    ]

    assert _ranked_ids(pool) == ["p0", "p1", "p2"]


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

    ids = _ranked_ids(ordinary + valuable)

    assert {"p1-treasury", "p1-trade"}.issubset(set(ids))


def test_low_value_p1_does_not_beat_clearly_important_p0():
    pool = [
        {**candidate("p0-high", priority="P0"),
         "title": "Treasury yields surge after inflation and jobs data shift Fed rate outlook"},
        {**candidate("p1-low", priority="P1"), "title": "Routine local company update"},
    ]

    assert _ranked_ids(pool) == ["p0-high", "p1-low"]


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

    ranked = [item for item, *_ in _rank_stage_a_candidates(pool)]
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

    assert "late-macro" in _ranked_ids(pool)


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
