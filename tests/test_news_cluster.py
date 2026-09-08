from src.news_events import (
    TITLE_MERGE_THRESHOLD,
    build_event_representatives,
    cluster_candidates_local,
    event_selection_candidates,
)


def candidate(cid, title, summary="summary", priority="P1", published_at="2026-08-12T10:00:00+00:00", source="Source"):
    return {
        "candidate_id": cid,
        "source": source,
        "priority": priority,
        "title": title,
        "summary": summary,
        "published_at": published_at,
        "url": f"https://example.com/{cid}",
    }


def test_cluster_merges_same_event_across_sources():
    candidates = [
        candidate("a", "Fed's Waller Says Rate Cut Still on the Table This Year", source="Bloomberg"),
        candidate("b", "Fed's Waller says rate cut still on the table this year", source="Reuters"),
    ]
    events = cluster_candidates_local(candidates)
    assert len(events) == 1
    assert sorted(events[0]["candidate_ids"]) == ["a", "b"]


def test_cluster_does_not_merge_similar_headlines_different_subjects():
    candidates = [
        candidate("a", "Fed Governor Barr Says Higher Rates Needed to Curb Inflation"),
        candidate("b", "Bank of England Governor Bailey Says Higher Rates Needed to Tame Prices"),
    ]
    events = cluster_candidates_local(candidates)
    assert len(events) == 2
    assert {frozenset(e["candidate_ids"]) for e in events} == {frozenset(["a"]), frozenset(["b"])}


def test_cluster_threshold_is_conservative():
    # Two different countries' inflation stories that happen to share a lot of
    # generic vocabulary -- real near-miss from PR 2's calibration data (see
    # experiments/calibrate_clustering.py) that must NOT merge at the shipped
    # threshold, even though a looser threshold would merge it.
    candidates = [
        candidate("a", "UK Shop Price Inflation Hits Two-Year High After Energy Spike"),
        candidate("b", "Euro-Zone Inflation Jumps to Highest in Almost Three Years"),
    ]
    events = cluster_candidates_local(candidates, title_merge_threshold=TITLE_MERGE_THRESHOLD)
    assert len(events) == 2
    # Sanity check the fixture actually straddles the threshold as intended:
    # a much looser threshold does merge it, showing this is a real threshold
    # boundary test and not just two obviously-unrelated titles.
    looser_events = cluster_candidates_local(candidates, title_merge_threshold=0.40)
    assert len(looser_events) == 1


def test_cluster_singleton_events_keep_original_title_and_topic():
    candidates = [candidate("a", "Nvidia Unveils New AI Chip")]
    events = cluster_candidates_local(candidates)
    assert len(events) == 1
    event = events[0]
    assert event["candidate_ids"] == ["a"]
    assert event["event_summary"] == "Nvidia Unveils New AI Chip"
    assert event["topic_group"] == "MEGA_CAP_TECH"
    assert event["event_category"] == "high_tech"


def test_cluster_merge_prefers_non_other_systemic_topic_group_and_longer_title():
    candidates = [
        candidate("a", "Markets React"),
        candidate("b", "Markets React to Nvidia's Blowout AI Chip Earnings Report"),
    ]
    events = cluster_candidates_local(candidates, title_merge_threshold=0.4)
    assert len(events) == 1
    event = events[0]
    assert sorted(event["candidate_ids"]) == ["a", "b"]
    # "Markets React" alone classifies as OTHER_SYSTEMIC; the merge should
    # prefer the more specific MEGA_CAP_TECH classification from the longer
    # title, mirroring _merge_cross_batch_duplicate_events' existing behavior.
    assert event["topic_group"] == "MEGA_CAP_TECH"
    assert event["event_category"] == "high_tech"
    assert event["event_summary"] == "Markets React to Nvidia's Blowout AI Chip Earnings Report"


def test_cluster_output_is_compatible_with_event_representative_helpers():
    candidates = [
        candidate("a", "Fed's Waller Says Rate Cut Still on the Table This Year", priority="P2", summary="short"),
        candidate("b", "Fed's Waller says rate cut still on the table this year", priority="P0", summary="much longer and more informative summary"),
        candidate("c", "Unrelated Story About Oil Prices"),
    ]
    events = cluster_candidates_local(candidates)
    representatives = build_event_representatives(events, candidates)
    selection_candidates = event_selection_candidates(representatives)

    merged_event = next(e for e in selection_candidates if e["candidate_id"] == "b")
    assert merged_event["priority"] == "P0"  # representative picked by priority/completeness
    assert merged_event["event_summary"]
    assert merged_event["topic_group"]
    assert "event_category" in merged_event

    assert {c["candidate_id"] for c in selection_candidates} == {"b", "c"}


def test_cluster_does_not_merge_across_many_unrelated_candidates():
    candidates = [
        candidate("a", "Apple Reports Record iPhone Sales"),
        candidate("b", "Oil Prices Surge on OPEC Cut"),
        candidate("c", "Fed Holds Rates Steady Amid Inflation Concerns"),
        candidate("d", "Tesla Recalls Vehicles Over Software Bug"),
    ]
    events = cluster_candidates_local(candidates)
    assert len(events) == 4
