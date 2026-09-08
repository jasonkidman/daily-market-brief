"""Event-level RSS news clustering and deterministic representative selection."""

from __future__ import annotations

import difflib
import re
from datetime import datetime
from functools import lru_cache

from .news_dedupe import normalize_title


TOPIC_GROUPS = {
    "US_MARKET_MACRO",
    "AI_CHIPS",
    "MEGA_CAP_TECH",
    "ENERGY_COMMODITIES",
    "GEOPOLITICS",
    "CORPORATE_EARNINGS",
    "OTHER_SYSTEMIC",
}
EVENT_CATEGORIES = {"macro_policy", "financial_markets", "high_tech", "geopolitics", "other"}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
PRIORITY_BASE_WEIGHT = {"P0": 6, "P1": 3, "P2": 0}
IMPORTANCE_WEIGHT = 2
SOURCE_PENALTY_WEIGHT = 0.25
MAX_SOURCE_PENALTY = 2.0

# These signals are intentionally small and explicit: they improve the cap
# ranking without asking an LLM to classify candidates or changing source data.
IMPORTANCE_SIGNAL_GROUPS = (
    ("macro_rates", 3, (
        "treasury", "federal reserve", "fed ", "interest rate", "interest rates",
        "policy rates", "treasury yields", "rates", "yield", "bond",
        "inflation", "employment", "jobs",
    )),
    ("mega_cap_tech", 3, ("apple", "microsoft", "amazon", "tesla", "nvidia", "alphabet", "google", "meta", "spacex")),
    ("ai_chips", 3, ("artificial intelligence", " ai ", "semiconductor", "chip", "data center", "datacenter", "gpu")),
    ("geopolitics_policy", 2, ()),  # see GEOPOLITICS_REGION_WORDS / GEOPOLITICS_ECONOMIC_SIGNALS below
)

# geopolitics_policy requires BOTH a geopolitical region/subject AND a keyword
# with real economic/market transmission -- a bare region name plus a generic
# policy word (ban/regulation/policy/law) is not enough on its own. Confirmed
# on real 2026-09-06 production data: a Ukrainian city's Russian-language arts
# policy matched the old flat keyword list purely via "Ukraine" + "ban", despite
# having no discernible US-market relevance. Local/cultural/social policy tied
# to one of these regions must NOT gain importance just from the region name.
GEOPOLITICS_REGION_WORDS = (
    "iran", "iranian", "russia", "russian", "ukraine", "ukrainian",
    "china", "chinese", "middle east", "middle eastern", "canada", "canadian",
)
GEOPOLITICS_ECONOMIC_SIGNALS = (
    "sanction", "tariff", "trade",
    "oil", "energy", "gas",
    "shipping", "strait", "supply chain",
    "war", "military strike", "missile", "conflict",
    "export control", "chip restriction",
    "financial sanction", "banking",
    "commodity", "market disruption",
)

# A handful of EVENT_SIGNIFICANCE_TERMS / IMPORTANCE_SIGNAL_GROUPS entries are
# themselves ordinary English words ("raise", "cut", "rates", "hike") that show
# up constantly in text having nothing to do with interest rates -- confirmed
# on real 2026-09-06 production data: "auction clearance rates" (Australian
# real estate) and "raises ethical questions" (an AI/animal-communication
# science piece) both scored as Fed/rate-policy signals purely because the bare
# word was present. Rather than dropping them (real "rate cut"/"Fed hike"
# stories do sometimes only use the bare verb), they now only count when a real
# monetary-policy context word also appears in the same text.
RATE_CONTEXT_WORDS = (
    "fed", "federal reserve", "central bank", "interest", "policy", "yield", "inflation", "monetary", "treasury",
)
AMBIGUOUS_TERMS_REQUIRING_CONTEXT = {"raise", "cut", "rates", "hike"}

EVENT_SIGNIFICANCE_TERMS = (
    "surge", "soar", "jump", "plunge", "collapse", "crash", "breakout",
    "rate hike", "interest rate hike", "rate cut", "interest rate cut", "fed hike", "fed cut",
    "raise", "cut", "hike",
    "inflation", "employment", "jobs", "yield", "rates", "outlook", "policy", "guidance", "earnings",
    "revenue", "profit", "loss", "lawsuit", "fine", "recall", "sanction", "tariff", "retaliatory",
    "trade war", "export control", "regulation", "regulator", "acquisition", "merger", "launch", "unveil",
    "announced", "announces", "ban", "restriction", "crisis", "war",
)

# Deterministic topic/category classification, originally written as Stage A's
# LLM-clustering-failure fallback (so a failed clustering call didn't collapse
# every event to OTHER_SYSTEMIC/other, discarding all semantic topic
# information for the whole day -- confirmed live on 2026-09-07, 50/50 fallback
# events all OTHER_SYSTEMIC) and now cluster_candidates_local's only topic
# classifier, since there is no LLM clustering left to have a "real"
# classification to fall back from. Reuses the same word-boundary/ambiguous-
# term-gated matching as the rest of the ranking pipeline (_keyword_present,
# _geopolitics_policy_matches) so it doesn't reintroduce the "hikers"/
# "clearance rates" substring bugs.
FALLBACK_MEGA_CAP_COMPANIES = (
    "apple", "microsoft", "amazon", "tesla", "nvidia", "alphabet", "google", "meta", "spacex",
)
FALLBACK_AI_CHIPS_TERMS = (
    "artificial intelligence", "ai", "gpu", "semiconductor", "chip", "data center", "datacenter", "cloud", "model",
)
FALLBACK_EARNINGS_TERMS = (
    "earnings", "quarterly results", "quarterly profit", "quarterly revenue", "revenue guidance", "profit warning",
)
FALLBACK_MACRO_TERMS = (
    "fed", "federal reserve", "treasury", "inflation", "jobs", "employment", "gdp", "tariff", "fiscal",
)
FALLBACK_ENERGY_TERMS = ("oil", "opec", "crude", "natural gas", "gas price", "energy price")
FALLBACK_TOPIC_TO_CATEGORY = {
    "US_MARKET_MACRO": "macro_policy",
    "MEGA_CAP_TECH": "high_tech",
    "AI_CHIPS": "high_tech",
    "ENERGY_COMMODITIES": "financial_markets",
    "GEOPOLITICS": "geopolitics",
    "CORPORATE_EARNINGS": "financial_markets",
    "OTHER_SYSTEMIC": "other",
}


def _classify_fallback_topic(candidate: dict) -> tuple[str, str]:
    """Return (topic_group, event_category) for one candidate.

    Order matters and is deliberate: a recognized mega-cap company name takes
    priority over generic AI/chip language (a company's own AI product news
    still reads as MEGA_CAP_TECH, per explicit product requirement), which in
    turn takes priority over generic earnings/macro/energy/geopolitics terms.
    """
    text = " ".join(str(candidate.get(field, "")) for field in ("title", "summary")).lower()
    if any(_keyword_present(text, word) for word in FALLBACK_MEGA_CAP_COMPANIES):
        topic_group = "MEGA_CAP_TECH"
    elif any(_keyword_present(text, word) for word in FALLBACK_AI_CHIPS_TERMS):
        topic_group = "AI_CHIPS"
    elif any(_keyword_present(text, word) for word in FALLBACK_EARNINGS_TERMS):
        topic_group = "CORPORATE_EARNINGS"
    elif any(_keyword_present(text, word) for word in FALLBACK_MACRO_TERMS):
        topic_group = "US_MARKET_MACRO"
    elif any(_keyword_present(text, word) for word in FALLBACK_ENERGY_TERMS):
        topic_group = "ENERGY_COMMODITIES"
    elif _geopolitics_policy_matches(text):
        topic_group = "GEOPOLITICS"
    else:
        topic_group = "OTHER_SYSTEMIC"
    return topic_group, FALLBACK_TOPIC_TO_CATEGORY[topic_group]


def _published_at_value(item: dict) -> float:
    try:
        return datetime.fromisoformat(item.get("published_at", "").replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return float("-inf")


@lru_cache(maxsize=None)
def _keyword_pattern(keyword: str) -> re.Pattern:
    """Word-boundary match with tolerance for a simple trailing plural (s/es),
    so e.g. "semiconductor" still matches "semiconductors" and "export control"
    still matches "export controls", but "hike" no longer matches inside an
    unrelated word like "hikers" and "chip" no longer matches inside "chipper".
    A bare substring check (the previous behavior) matched keywords anywhere
    inside another word -- confirmed live on the 2026-09-06 production data,
    where "hikers rescued..." was scored as an interest-rate-hike signal purely
    because "hike" is a substring of "hikers"."""
    return re.compile(r"\b" + re.escape(keyword.strip()) + r"(?:s|es)?\b")


def _keyword_present(text: str, keyword: str) -> bool:
    """Like a plain word-boundary match, except a small set of ambiguous single
    words (AMBIGUOUS_TERMS_REQUIRING_CONTEXT) only count as present when a real
    monetary-policy context word (RATE_CONTEXT_WORDS) also appears in the same
    text -- otherwise they match on ordinary English usage that has nothing to
    do with interest rates (see the module comment above those constants)."""
    if not _keyword_pattern(keyword).search(text):
        return False
    if keyword.strip().lower() not in AMBIGUOUS_TERMS_REQUIRING_CONTEXT:
        return True
    return any(_keyword_pattern(context).search(text) for context in RATE_CONTEXT_WORDS)


def _geopolitics_policy_matches(text: str) -> bool:
    """geopolitics_policy's own two-tier check (see GEOPOLITICS_REGION_WORDS /
    GEOPOLITICS_ECONOMIC_SIGNALS): a region word alone, or a generic policy word
    (ban/regulation/policy/law) alone, is not enough -- both a region and a
    keyword with real economic/market transmission must be present."""
    has_region = any(_keyword_pattern(word).search(text) for word in GEOPOLITICS_REGION_WORDS)
    has_signal = any(_keyword_pattern(word).search(text) for word in GEOPOLITICS_ECONOMIC_SIGNALS)
    return has_region and has_signal


def _importance_signal(item: dict) -> tuple[int, str]:
    """Return an explainable deterministic importance score for cap ranking."""
    text = " ".join(str(item.get(field, "")) for field in ("title", "summary", "category_hint")).lower()
    significant = any(_keyword_present(text, term) for term in EVENT_SIGNIFICANCE_TERMS)
    matches = []
    score = 0
    for name, weight, keywords in IMPORTANCE_SIGNAL_GROUPS:
        if name == "geopolitics_policy":
            # The region+economic-signal pair is already its own self-contained
            # significance test (see _geopolitics_policy_matches); it does not
            # additionally require a hit in the generic EVENT_SIGNIFICANCE_TERMS
            # list, which does not cover terms like "strike" or "tanker".
            group_matches = _geopolitics_policy_matches(text)
        else:
            group_matches = any(_keyword_present(text, keyword) for keyword in keywords) and significant
        if group_matches:
            matches.append(name)
            score += weight
    return min(score, 6), ",".join(matches) or "none"


def _rank_stage_a_candidates(candidates: list[dict]) -> list[tuple[dict, int, str, float, int, float]]:
    """Rank with a strong priority base, weighted importance, and soft diversity."""
    remaining = list(enumerate(candidates))
    ranked = []
    source_counts: dict[str, int] = {}
    while remaining:
        def sort_key(entry: tuple[int, dict]):
            index, item = entry
            importance_score, _ = _importance_signal(item)
            source = item.get("source", "")
            priority_base = PRIORITY_BASE_WEIGHT.get(item.get("priority", "P2"), 0)
            source_penalty = min(source_counts.get(source, 0) * SOURCE_PENALTY_WEIGHT, MAX_SOURCE_PENALTY)
            composite_score = priority_base + importance_score * IMPORTANCE_WEIGHT - source_penalty
            return (
                -composite_score,
                -_published_at_value(item),
                index,
            )

        index, item = min(remaining, key=sort_key)
        remaining.remove((index, item))
        importance_score, reason = _importance_signal(item)
        source = item.get("source", "")
        source_count = source_counts.get(source, 0)
        source_penalty = min(source_count * SOURCE_PENALTY_WEIGHT, MAX_SOURCE_PENALTY)
        priority_base = PRIORITY_BASE_WEIGHT.get(item.get("priority", "P2"), 0)
        composite_score = priority_base + importance_score * IMPORTANCE_WEIGHT - source_penalty
        ranked.append((item, importance_score, reason, source_penalty, priority_base, composite_score))
        source_counts[source] = source_count + 1
    return ranked


def select_event_representative(event: dict, candidate_pool: list[dict]) -> dict:
    """Choose source article by priority, information completeness, then recency."""
    by_id = {item["candidate_id"]: item for item in candidate_pool}
    articles = [by_id[candidate_id] for candidate_id in event["candidate_ids"]]
    return max(articles, key=lambda item: (
        -PRIORITY_ORDER.get(item.get("priority", "P2"), 2),
        len(item.get("summary", "")),
        _published_at_value(item),
    ))


def build_event_representatives(events: list[dict], candidate_pool: list[dict]) -> list[dict]:
    """Attach a local representative while retaining model-independent event metadata."""
    return [{
        **event,
        "representative": select_event_representative(event, candidate_pool),
    } for event in events]


def event_selection_candidates(event_representatives: list[dict]) -> list[dict]:
    """Flatten one program-selected article per event for the news selection stage."""
    return [{
        **event["representative"],
        "event_summary": event["event_summary"],
        "topic_group": event["topic_group"],
        "event_category": event.get("event_category", "other"),
    } for event in event_representatives]


# Calibrated offline against 7 days of real Stage A output (2026-08-31 through
# 2026-09-06; see experiments/calibrate_clustering.py and IMPLEMENTATION_PLAN.md
# section 6.2 for the method, and PR 2's writeup for why this is title-only).
#
# The calibration's ground truth (which raw candidates Stage A considered the
# same real-world event) had to be reconstructed from GitHub Actions job logs,
# because the snapshot artifact only persists one representative article per
# event, not the full pre-clustering pool -- so every candidate an event's
# LLM-run merged away (i.e. didn't pick as representative) has no summary text
# recorded anywhere, ever. That rules out validating a summary-corroborated
# "soft" title-match band against real data, so this function deliberately
# does not have one: only a single hard title-similarity threshold, decided
# entirely by raw title text.
#
# Against that reconstructed ground truth (127 real same-event pairs, restricted
# to the subset of candidates Stage A actually evaluated -- the old per-request
# input cap silently dropped the rest before clustering, so "not merged" for
# those isn't a real negative), title similarity produces exactly one over-merge
# candidate at every threshold from 0.56 up through 0.88 -- a Bloomberg live-blog
# and its own later "key takeaways" wrap-up of the same jobs report, confirmed by
# hand to be the same real event, i.e. not a real error. Below 0.54 real
# over-merges (e.g. a UK inflation story merged with a Eurozone one) start
# appearing and grow quickly. 0.84 sits well clear of that observed edge --
# recall is nearly flat across the whole 0.66-0.88 safe range (this data set's
# cross-outlet title-similarity recall for genuine duplicates is low either way,
# ~6-8%; see PR 2 writeup), so there is little to trade away for the extra margin.
TITLE_MERGE_THRESHOLD = 0.84


def _candidates_are_likely_duplicates(a: dict, b: dict, *, title_merge_threshold: float = TITLE_MERGE_THRESHOLD) -> bool:
    """Deterministic same-real-event check on raw article titles (as opposed
    to _events_are_likely_duplicates, which compares already-LLM-normalized
    event_summary text -- raw titles vary far more in wording across outlets,
    which is why this threshold is calibrated separately, see above)."""
    title_ratio = difflib.SequenceMatcher(
        None, normalize_title(a["title"]), normalize_title(b["title"])
    ).ratio()
    return title_ratio >= title_merge_threshold


def cluster_candidates_local(candidates: list[dict], *, title_merge_threshold: float = TITLE_MERGE_THRESHOLD) -> list[dict]:
    """Pure-code replacement for Stage A's LLM clustering: greedily merges
    candidates describing the same real-world event based on raw title
    similarity, conservative by design (see threshold comment above).

    Output is shaped to drop into build_event_representatives()/
    event_selection_candidates() unchanged: each event has event_id /
    candidate_ids / event_summary / topic_group / event_category.
    event_summary has no LLM to author it, so it is just the representative
    article's own title.
    """
    events: list[dict] = []
    for index, candidate in enumerate(candidates, 1):
        topic_group, event_category = _classify_fallback_topic(candidate)
        title = candidate.get("title", "")
        match = next((
            event for event in events
            if _candidates_are_likely_duplicates(
                {"title": event["_rep_title"]}, candidate, title_merge_threshold=title_merge_threshold,
            )
        ), None)
        if match is None:
            events.append({
                "event_id": f"local_{index:03d}",
                "candidate_ids": [candidate["candidate_id"]],
                "event_summary": title or "新闻事件",
                "topic_group": topic_group,
                "event_category": event_category,
                "_rep_title": title,
            })
            continue
        match["candidate_ids"].append(candidate["candidate_id"])
        if match["topic_group"] == "OTHER_SYSTEMIC" and topic_group != "OTHER_SYSTEMIC":
            match["topic_group"] = topic_group
            match["event_category"] = event_category
        if len(title) > len(match["_rep_title"]):
            match["_rep_title"] = title
            match["event_summary"] = title
    for event in events:
        del event["_rep_title"]
    return events


