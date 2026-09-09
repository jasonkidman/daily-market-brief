"""Deterministic off-topic pre-filter: narrow, high-precision rules only.
A false-positive drop here is silent and unrecoverable (unlike a low score,
which still shows up in "更多新闻"), so these tests lean on making sure the
exempt-company escape hatch and non-matching content both pass through."""

from __future__ import annotations

from src.news_prefilter import filter_off_topic_candidates


RULES = [
    {
        "reason": "content_moderation_child_safety",
        "phrases": ["child sexual abuse", "csam"],
    },
    {
        "reason": "crypto_theft_unrelated_to_tracked_companies",
        "phrases": ["crypto heist", "hackers stole"],
        "exempt_if_mentions": ["coinbase"],
    },
]


def candidate(cid, title="", summary=""):
    return {"candidate_id": cid, "title": title, "summary": summary}


def test_matching_phrase_drops_the_candidate():
    candidates = [candidate("a", title="Report says Meta ran ads with child sexual abuse content")]

    assert filter_off_topic_candidates(candidates, RULES) == []


def test_matching_phrase_in_summary_also_drops():
    candidates = [candidate("a", title="Platform faces scrutiny", summary="Ads promoted CSAM material.")]

    assert filter_off_topic_candidates(candidates, RULES) == []


def test_exempt_company_mention_keeps_the_candidate():
    candidates = [candidate("a", title="Hackers stole funds from a bridge tied to Coinbase")]

    assert filter_off_topic_candidates(candidates, RULES) == candidates


def test_non_matching_candidate_passes_through():
    candidates = [candidate("a", title="Fed holds interest rates steady")]

    assert filter_off_topic_candidates(candidates, RULES) == candidates


def test_matching_is_case_insensitive():
    candidates = [candidate("a", title="HACKERS STOLE crypto from an exchange")]

    assert filter_off_topic_candidates(candidates, RULES) == []


def test_mixed_pool_only_drops_matching_candidates():
    candidates = [
        candidate("keep", title="Fed holds interest rates steady"),
        candidate("drop", title="Crypto heist nets $10 million"),
    ]

    result = filter_off_topic_candidates(candidates, RULES)

    assert [item["candidate_id"] for item in result] == ["keep"]


def test_real_config_file_loads_and_has_expected_shape():
    import yaml
    from pathlib import Path

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "news_prefilter.yaml").read_text())

    assert "rules" in config
    for rule in config["rules"]:
        assert "reason" in rule
        assert isinstance(rule["phrases"], list) and rule["phrases"]
