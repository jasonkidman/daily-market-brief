"""Deterministic, zero-LLM-cost early relevance filter.

Runs on raw English title+summary, before clustering/scoring, so a candidate
matching a rule in config/news_prefilter.yaml never reaches (and never
costs) a scoring call and never appears in "更多新闻". This is a narrow,
high-precision list (content-moderation and unrelated-crypto-theft genres,
see the config file), not a general relevance classifier -- anything not an
exact match here still flows through to Scoring, which is where the real
relevance judgment happens.
"""

from __future__ import annotations


def _candidate_text(candidate: dict) -> str:
    return f"{candidate.get('title', '')} {candidate.get('summary', '')}".lower()


def _rule_matches(text: str, rule: dict) -> bool:
    if not any(phrase.lower() in text for phrase in rule.get("phrases", [])):
        return False
    exempt = rule.get("exempt_if_mentions", [])
    return not any(name.lower() in text for name in exempt)


def filter_off_topic_candidates(candidates: list[dict], rules: list[dict]) -> list[dict]:
    """Drop candidates matching a narrow, high-precision off-topic rule.

    Returns only the kept candidates. Each drop is logged in the same
    [NEWS CANDIDATE] format the rest of the pipeline uses (stage=prefilter),
    so it shows up in run logs the same way dedup/24h-filter drops do.
    """
    kept = []
    for candidate in candidates:
        text = _candidate_text(candidate)
        matched = next((rule for rule in rules if _rule_matches(text, rule)), None)
        if matched is None:
            kept.append(candidate)
            continue
        print(
            f"[NEWS CANDIDATE] candidate_id={candidate.get('candidate_id', '')} "
            f"| title={candidate.get('title', '')} | source={candidate.get('source', '')} "
            f"| published_at={candidate.get('published_at', '')} | stage=prefilter "
            f"| action=drop | reason={matched['reason']}"
        )
    return kept
