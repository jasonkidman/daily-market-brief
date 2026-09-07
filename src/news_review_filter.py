"""Deterministic Review Filter: picks which Stage-B-unselected candidates are
worth a human's time in the "more news" (更多新闻) drawer.

This sits between Stage B final selection and Candidate Pool Translation:

    all Stage B candidates -> selected (homepage) -> unselected
        -> Review Filter -> review candidates -> translation -> "more news"

It never calls an LLM and never changes what Stage A/Stage B select or how
many candidates they see -- Stage A/Stage B coverage of the full candidate
pool is completely untouched by this module. It only decides, after the fact,
which of the candidates Stage B already declined to show on the homepage are
worth surfacing for manual review (so the user can spot near-misses and tune
the selection rules), versus which are obviously out of scope and should stay
hidden. There is deliberately no minimum pool size -- an ordinary day with
few review-worthy candidates should show few, not be padded to a quota -- and
a `MAX_REVIEW_POOL_SIZE` cap only guards against an extreme high-volume day.
"""

from __future__ import annotations

import re

from .news_events import (
    FALLBACK_ENERGY_CORE_TERMS,
    FALLBACK_ENERGY_DISRUPTION_TERMS,
    FALLBACK_MACRO_STRICT_TERMS,
    FALLBACK_MEGA_CAP_COMPANIES,
    FALLBACK_SPACEX_ONLY_EVENT_TERMS,
    PRIORITY_ORDER,
    _fallback_matches_geopolitics,
    _importance_signal,
    _keyword_present,
)

# No minimum: a quiet day may legitimately have zero review-worthy candidates.
# The cap only exists to bound an extreme high-volume day's drawer size --
# see _select_with_diversity for how candidates are prioritized within it.
MAX_REVIEW_POOL_SIZE = 30

# See _review_text for why this exists.
_CJK_LATIN_BOUNDARY = re.compile(r"(?<=[一-鿿])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[一-鿿])")

# Extra US-macro phrases beyond news_events.FALLBACK_MACRO_STRICT_TERMS (which
# _fallback_high_confidence_rule already checks): dollar/DXY and a couple of
# explicitly US-qualified terms the stricter Stage B fallback rule doesn't need.
# Every entry keeps the same "explicit US context, never a bare generic word
# like rates/jobs/trade" discipline as the rest of this module -- seeded by
# the same 2026-09 production lessons captured in news_events.py.
REVIEW_MACRO_EXTRA_TERMS = (
    "us gdp", "us consumption", "dollar index", "dxy", "us dollar",
)

# Broader than news_events.FALLBACK_MEGA_CAP_MAJOR_EVENT_TERMS on purpose: the
# Review Pool's bar is lower than Stage B's ("即使事件最终没有达到首页重大新闻门槛，
# 只要存在合理人工复核价值，也可以保留"). Deliberately still excludes bare
# "unveil"/"launch" -- those alone would let ordinary consumer-product trivia
# (a new phone color, a minor OS update) back in just for naming a tracked
# company; a real AI/chip product launch is still caught via
# REVIEW_AI_CHIP_TERMS below regardless of which company is behind it.
REVIEW_MEGA_CAP_EXTRA_TERMS = (
    "acquisition", "acquire", "acquires", "merger", "investment", "invests", "investing",
    "capital expenditure", "capex",
    "partnership", "collaboration", "collaborate", "joint venture", "alliance", "deal", "agreement",
    "antitrust", "doj", "ftc", "regulation", "regulator", "regulatory",
    "lawsuit", "sue", "sues", "sued",
    "earnings", "quarterly results", "guidance",
    "government contract", "nasa contract",
)

# On top of news_events.FALLBACK_SPACEX_ONLY_EVENT_TERMS (starship/starlink/
# explosion/crash/mission failure/anomaly): "Starbase" is SpaceX's Texas
# launch-site name, not a generic word, and infrastructure progress there is a
# meaningful SpaceX-specific update Review Pool's lower bar should catch.
# Confirmed via real 2026-09-07 replay: "Starbase Infrastructure Advances
# Toward Flight 14" (NASASpaceflight, P0) was otherwise filtered.
REVIEW_SPACEX_EXTRA_TERMS = ("starbase",)

# Structural AI/semiconductor/data-center signal, independent of company --
# lower bar than Stage B on purpose ("Review Pool 的门槛应低于首页 Stage B"), but
# still a curated phrase list (not a bare "ai") to avoid matching incidental
# mentions in otherwise-unrelated articles.
REVIEW_AI_CHIP_TERMS = (
    "artificial intelligence", "semiconductor", "chip", "data center", "datacenter",
    "gpu", "machine learning", "large language model",
)

# Sort-order tiers for the "topic relevance" ranking key (see
# select_review_pool). Lower is more important. A candidate whose only
# qualifying signal is Stage B's own borderline/reserve/topic-cap history (no
# keyword match at all) gets the best tier, per "这是最有人工反馈价值的一类新闻".
_TIER_BORDERLINE_ONLY = 0
_TIER_MACRO = 1
_TIER_MEGA_CAP = 2
_TIER_AI_CHIPS = 3
_TIER_GEOPOLITICS_ENERGY = 4

_RULE_TIER = {
    "us_macro_fed_treasury": _TIER_MACRO,
    "us_macro_extra": _TIER_MACRO,
    "mega_cap_review": _TIER_MEGA_CAP,
    "ai_semiconductor_structural": _TIER_AI_CHIPS,
    "geopolitics_market_transmission": _TIER_GEOPOLITICS_ENERGY,
    "energy_supply_disruption": _TIER_GEOPOLITICS_ENERGY,
}


def _review_text(candidate: dict) -> str:
    """Title + Stage A's event_summary only -- deliberately excludes the raw
    article `summary` field, matching news_events._fallback_high_confidence_rule's
    own reasoning: a long, noisy article summary can mention an unrelated region
    or company purely as background color (confirmed via real 2026-09-07 data,
    see news_events.py), while event_summary is a distilled, on-topic statement
    of what the story is actually about."""
    raw = " ".join(str(candidate.get(field) or "") for field in ("title", "event_summary"))
    # Stage A's Chinese event_summary often glues an English company name
    # directly onto adjacent Chinese characters with no space (e.g.
    # "SpaceX继续推进..."). Python's \w (and therefore news_events._keyword_pattern's
    # \b boundary) treats CJK ideographs as word characters, so "spacex" right
    # up against "继" has no boundary on that side and silently fails to match.
    # Confirmed via real 2026-09-07 replay: "Starbase Infrastructure Advances
    # Toward Flight 14" (a real SpaceX update) was wrongly filtered this way.
    # Inserting a space at every CJK/Latin-or-digit transition fixes this
    # locally, without touching the shared keyword matcher Stage A/B also use.
    normalized = _CJK_LATIN_BOUNDARY.sub(" ", raw)
    return normalized.lower()


def _energy_supply_disruption_matches(text: str) -> bool:
    """Mirrors news_events.py's Rule D (energy): a core oil/OPEC term AND a real
    supply/shipping-disruption term, both required -- ordinary commodity
    coverage (sugar, a routine oil-market roundup) does not qualify just for
    mentioning oil."""
    has_core = any(_keyword_present(text, term) for term in FALLBACK_ENERGY_CORE_TERMS)
    has_disruption = any(_keyword_present(text, term) for term in FALLBACK_ENERGY_DISRUPTION_TERMS)
    return has_core and has_disruption


def _review_signal(candidate: dict) -> str | None:
    """Return the name of the review-worthiness rule `candidate` clears, or
    None if it clears none of them. OTHER_SYSTEMIC (or any other topic_group)
    gets no special treatment here -- only the text content decides, so an
    OTHER_SYSTEMIC candidate with no qualifying signal is filtered by default,
    per the "OTHER_SYSTEMIC 不应该默认进入 Review Pool" requirement.

    Deliberately does NOT reuse news_events._fallback_high_confidence_rule
    wholesale: that rule's mega-cap branch also accepts bare "unveil"/"launch"
    (a legitimate Stage-B-failure recovery signal there), which would let
    ordinary consumer-product trivia -- a new Kindle color, a minor phone
    update -- back in just for naming a tracked company. This module reuses
    its macro/geopolitics/energy building blocks directly, but defines its own,
    narrower mega-cap event-term check (REVIEW_MEGA_CAP_EXTRA_TERMS)."""
    text = _review_text(candidate)
    if any(_keyword_present(text, term) for term in FALLBACK_MACRO_STRICT_TERMS):
        return "us_macro_fed_treasury"
    if any(_keyword_present(text, term) for term in REVIEW_MACRO_EXTRA_TERMS):
        return "us_macro_extra"
    matched_companies = [word for word in FALLBACK_MEGA_CAP_COMPANIES if _keyword_present(text, word)]
    if matched_companies:
        has_extra_event = any(_keyword_present(text, term) for term in REVIEW_MEGA_CAP_EXTRA_TERMS)
        has_spacex_event = "spacex" in matched_companies and any(
            _keyword_present(text, term) for term in FALLBACK_SPACEX_ONLY_EVENT_TERMS + REVIEW_SPACEX_EXTRA_TERMS
        )
        if has_extra_event or has_spacex_event:
            return "mega_cap_review"
    if any(_keyword_present(text, term) for term in REVIEW_AI_CHIP_TERMS):
        return "ai_semiconductor_structural"
    if _fallback_matches_geopolitics(text):
        return "geopolitics_market_transmission"
    if _energy_supply_disruption_matches(text):
        return "energy_supply_disruption"
    return None


def _sort_key(scored: dict) -> tuple:
    return (
        0 if scored["is_near_miss"] else 1,
        scored["tier"],
        -scored["importance_score"],
        PRIORITY_ORDER.get(scored["priority"], 2),
        -scored["published_ts"],
    )


def _published_ts(candidate: dict) -> float:
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(candidate.get("published_at", "")).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return float("-inf")


def _select_with_diversity(scored: list[dict], max_pool: int) -> list[dict]:
    """Sort best-first, then -- only if truncation is actually needed -- round
    robin across (near-miss, tier) groups so the top `max_pool` isn't
    accidentally dominated by a single busy category, per "尽量保持宏观 / 科技 /
    地缘政治等主题多样性"."""
    ordered = sorted(scored, key=_sort_key)
    if len(ordered) <= max_pool:
        return ordered
    groups: dict[tuple, list[dict]] = {}
    group_order: list[tuple] = []
    for item in ordered:
        key = (0 if item["is_near_miss"] else 1, item["tier"])
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(item)
    pointers = {key: 0 for key in group_order}
    result: list[dict] = []
    while len(result) < max_pool:
        progressed = False
        for key in group_order:
            pointer = pointers[key]
            bucket = groups[key]
            if pointer < len(bucket):
                result.append(bucket[pointer])
                pointers[key] = pointer + 1
                progressed = True
                if len(result) >= max_pool:
                    break
        if not progressed:
            break
    return result


def select_review_pool(unselected_candidates: list[dict], stage_b_diagnostics: dict | None = None,
                       max_pool: int = MAX_REVIEW_POOL_SIZE) -> dict:
    """Split `unselected_candidates` (Stage B input pool minus whatever Stage B
    selected) into a review pool and a filtered-out remainder.

    `stage_b_diagnostics` carries pure bookkeeping Stage B already computed
    about its own decision (never reused for re-selection): `borderline_ids`
    (candidates one two-pass sample picked and the other didn't, then review
    dropped), `reserve_ids` (Stage B's own runner-up picks), and
    `topic_cap_dropped_ids` (candidates that scored well enough but lost to the
    per-topic display cap). A candidate in any of these is treated as a
    Stage-B-acknowledged near-miss and always prioritized for review.

    Returns {"review_candidates", "filtered_candidates" (each entry
    {"candidate_id", "filter_reason"}), "unselected_candidate_count",
    "review_candidate_count", "review_filtered_count"}.
    """
    diagnostics = stage_b_diagnostics or {}
    near_miss_ids = (
        set(diagnostics.get("borderline_ids", ()))
        | set(diagnostics.get("reserve_ids", ()))
        | set(diagnostics.get("topic_cap_dropped_ids", ()))
    )

    qualifying: list[dict] = []
    filtered: list[dict] = []
    for candidate in unselected_candidates:
        candidate_id = candidate.get("candidate_id")
        is_near_miss = candidate_id in near_miss_ids
        rule = _review_signal(candidate)
        if not is_near_miss and rule is None:
            filtered.append({
                "candidate_id": candidate_id,
                "filter_reason": f"no_qualifying_signal:topic_group={candidate.get('topic_group')}",
            })
            continue
        importance_score, _ = _importance_signal(candidate)
        qualifying.append({
            "candidate": candidate,
            "is_near_miss": is_near_miss,
            "tier": _RULE_TIER.get(rule, _TIER_BORDERLINE_ONLY),
            "importance_score": importance_score,
            "priority": candidate.get("priority", "P2"),
            "published_ts": _published_ts(candidate),
        })

    kept = _select_with_diversity(qualifying, max_pool)
    kept_ids = {item["candidate"]["candidate_id"] for item in kept}
    for item in qualifying:
        if item["candidate"]["candidate_id"] not in kept_ids:
            filtered.append({"candidate_id": item["candidate"]["candidate_id"], "filter_reason": "capped_at_max_pool_size"})

    review_candidates = [item["candidate"] for item in kept]
    return {
        "review_candidates": review_candidates,
        "filtered_candidates": filtered,
        "unselected_candidate_count": len(unselected_candidates),
        "review_candidate_count": len(review_candidates),
        "review_filtered_count": len(filtered),
    }
