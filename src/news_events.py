"""Event-level RSS news clustering and deterministic representative selection."""

from __future__ import annotations

import difflib
import json
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Optional

from .deepseek_client import (
    DEEPSEEK_MAX_ATTEMPTS, NEWS_REASONING_EFFORT, NonRetryableAPIError, SUMMARY_ZH_LIMIT, TITLE_ZH_LIMIT,
    DeepSeekUsageTracker, call_deepseek, invoke_model,
)
from .news_event_prompt import SYSTEM_PROMPT


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

# Maximum candidates per single Stage A clustering request (see
# batch_stage_a_input). Bounds one request's size -- and therefore its odds of
# finishing inside its timeout -- but is NOT a total cap on how many dedup'd
# candidates Stage A ever clusters: a pool larger than this is split into
# multiple batches, each run through clustering in full, so nothing is
# silently dropped on a heavy news day. Confirmed live on 2026-09-07: 72
# dedup'd candidates were silently capped to the top 50 by composite score
# before ever reaching Stage A, permanently losing the other 22.
STAGE_A_MAX_INPUT = 50

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

# Maximum candidates per single Stage B request (see batch_stage_b_input).
# Bounds one request's size -- and therefore its odds of finishing inside its
# timeout -- but is NOT a total cap on how many candidates Stage B ever judges:
# a pool larger than this is split into multiple batches, each run through
# Stage B in full, so nothing is silently dropped regardless of how many
# events Stage A produces on a heavy news day.
STAGE_B_MAX_INPUT = 28

# Used only by the deterministic Stage B fallback below, never by the real AI
# selection: a coarse, static bucket from Stage A's topic_group to one of Stage
# B's own category labels, so a fallback item still renders with a category
# instead of leaving the field blank. This is a fixed lookup, not a judgement
# call, so it is not "fabricating an AI field".
TOPIC_GROUP_TO_CATEGORY = {
    "US_MARKET_MACRO": "美国经济",
    "AI_CHIPS": "半导体",
    "MEGA_CAP_TECH": "大型科技",
    "ENERGY_COMMODITIES": "美国经济",
    "GEOPOLITICS": "地缘政治",
    "CORPORATE_EARNINGS": "金融市场",
    "OTHER_SYSTEMIC": "美国经济",
}

EVENT_SIGNIFICANCE_TERMS = (
    "surge", "soar", "jump", "plunge", "collapse", "crash", "breakout",
    "rate hike", "interest rate hike", "rate cut", "interest rate cut", "fed hike", "fed cut",
    "raise", "cut", "hike",
    "inflation", "employment", "jobs", "yield", "rates", "outlook", "policy", "guidance", "earnings",
    "revenue", "profit", "loss", "lawsuit", "fine", "recall", "sanction", "tariff", "retaliatory",
    "trade war", "export control", "regulation", "regulator", "acquisition", "merger", "launch", "unveil",
    "announced", "announces", "ban", "restriction", "crisis", "war",
)

# Deterministic topic/category classification used ONLY when Stage A event
# clustering itself fails and falls back to one event per candidate (see
# _fallback_events). Without this, every fallback event used to collapse to
# OTHER_SYSTEMIC/other, discarding all semantic topic information for the
# whole day whenever the clustering call timed out or the model returned a
# malformed contract -- confirmed live on 2026-09-07 (50/50 fallback events
# all OTHER_SYSTEMIC). This does not aim to match Stage A's own LLM-quality
# classification, only to be meaningfully better than "everything is other";
# it reuses the same word-boundary/ambiguous-term-gated matching as the rest
# of the ranking pipeline (_keyword_present, _geopolitics_policy_matches) so
# it doesn't reintroduce the "hikers"/"clearance rates" substring bugs.
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
    """Return (topic_group, event_category) for one Stage-A-fallback event.

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


class NewsEventError(ValueError):
    """Raised when the event-clustering output violates its contract."""


def _parse_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise NewsEventError("事件聚类输出不是 JSON 对象。")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise NewsEventError("事件聚类输出无法解析为 JSON。") from exc


def validate_event_clusters(payload: Any, candidates: list[dict]) -> list[dict]:
    """Validate total, non-overlapping candidate coverage from the clustering model."""
    data = _parse_payload(payload)
    events = data.get("events")
    if not isinstance(events, list):
        raise NewsEventError("events 必须是数组。")
    pool_ids = {item["candidate_id"] for item in candidates}
    event_ids, assigned, validated = set(), set(), []
    for item in events:
        if not isinstance(item, dict):
            raise NewsEventError("event 条目必须是对象。")
        event_id = str(item.get("event_id", "")).strip()
        candidate_ids = item.get("candidate_ids")
        summary = str(item.get("event_summary", "")).strip()
        topic_group = item.get("topic_group")
        event_category = item.get("event_category", "other")
        if not event_id or event_id in event_ids:
            raise NewsEventError("event_id 不能为空且不得重复。")
        if not isinstance(candidate_ids, list) or not candidate_ids:
            raise NewsEventError("candidate_ids 必须是非空数组。")
        if not summary:
            raise NewsEventError("event_summary 不能为空。")
        if topic_group not in TOPIC_GROUPS:
            raise NewsEventError("topic_group 不合法。")
        if event_category not in EVENT_CATEGORIES:
            raise NewsEventError("event_category 不合法。")
        for candidate_id in candidate_ids:
            if candidate_id not in pool_ids:
                raise NewsEventError("candidate_id 不在候选池。")
            if candidate_id in assigned:
                raise NewsEventError("candidate 不得同时属于多个 event。")
            assigned.add(candidate_id)
        event_ids.add(event_id)
        validated.append({
            "event_id": event_id,
            "candidate_ids": candidate_ids,
            "event_summary": summary,
            "topic_group": topic_group,
            "event_category": event_category,
        })
    if assigned != pool_ids:
        raise NewsEventError("所有 candidate 必须恰好被一个 event 覆盖。")
    return validated


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


def _cluster_candidate_input(candidates: list[dict]) -> list[dict]:
    """Bound one clustering *request's* size without exposing article URLs to
    the model. `candidates` is expected to already be at most one batch (see
    batch_stage_a_input) -- the [:STAGE_A_MAX_INPUT] slice here is a no-op
    safety bound for that case, not the total-pool cap it used to be."""
    limited = [item for item, _, _, _, _, _ in _rank_stage_a_candidates(candidates)[:STAGE_A_MAX_INPUT]]
    fields = ("candidate_id", "source", "priority", "title", "summary", "published_at")
    return [{field: item.get(field, "") for field in fields} for item in limited]


def stage_a_input_counts(candidates: list[dict]) -> tuple[int, int]:
    """Return (pre-cap, actual Stage A input) counts. Batching (see
    cluster_news_events_batched) guarantees every dedup'd candidate reaches
    some Stage A batch, so these are always equal now; both are still reported
    for the snapshot/report schema's sake."""
    return len(candidates), len(candidates)


def _log_stage_a_cap(candidates: list[dict], cluster_input: list[dict]) -> None:
    actual_ids = {item["candidate_id"] for item in cluster_input}
    ranking = _rank_stage_a_candidates(candidates)
    diagnostics = {
        item["candidate_id"]: (rank, importance_score, reason, source_penalty, priority_base, composite_score)
        for rank, (item, importance_score, reason, source_penalty, priority_base, composite_score)
        in enumerate(ranking, 1)
    }
    print(
        f"[NEWS STAGE A CAP] pre_cap={len(candidates)} actual_input={len(cluster_input)} "
        f"cap_dropped={len(candidates) - len(cluster_input)}"
    )
    for candidate in candidates:
        action = "keep" if candidate["candidate_id"] in actual_ids else "drop"
        reason = "" if action == "keep" else "input_cap_50"
        suffix = f" | reason={reason}" if reason else ""
        rank, importance_score, importance_reason, source_penalty, priority_base, composite_score = diagnostics[candidate["candidate_id"]]
        print(
            f"[NEWS CANDIDATE] candidate_id={candidate.get('candidate_id', '')} "
            f"| source={candidate.get('source', '')} | priority={candidate.get('priority', 'P2')} "
            f"| importance_score={importance_score} | importance_reason={importance_reason} "
            f"| priority_base={priority_base} | source_penalty={source_penalty:g} "
            f"| composite_score={composite_score:g} | pre_cap_rank={rank} "
            f"| stage=stage_a_cap "
            f"| action={action}{suffix}"
        )


def _log_stage_a_mapping(events: list[dict]) -> None:
    for event in events:
        for candidate_id in event.get("candidate_ids", []):
            print(
                f"[NEWS STAGE A MAPPING] candidate_id={candidate_id} "
                f"-> event_id={event.get('event_id', '')}"
            )


def _fallback_events(candidates: list[dict]) -> list[dict]:
    events = []
    for index, item in enumerate(candidates, 1):
        topic_group, event_category = _classify_fallback_topic(item)
        events.append({
            "event_id": f"fallback_{index:03d}",
            "candidate_ids": [item["candidate_id"]],
            "event_summary": item.get("title", "新闻事件"),
            "topic_group": topic_group,
            "event_category": event_category,
        })
    return events


def _log_stage_a_events(events: list[dict]) -> None:
    print(f"[NEWS STAGE A] output events: {len(events)}")
    for event in events:
        print(
            "[NEWS STAGE A] event_id={event_id} | category={category} | title={title}".format(
                event_id=event.get("event_id", ""),
                category=event.get("event_category", "other"),
                title=event.get("event_summary", ""),
            )
        )


def cluster_news_events(candidates: list[dict], api_key: str,
                        call_model: Callable = call_deepseek,
                        sleep_fn: Callable = time.sleep,
                        usage_tracker: DeepSeekUsageTracker | None = None,
                        max_attempts: int = DEEPSEEK_MAX_ATTEMPTS + 1) -> tuple[list[dict], Optional[str]]:
    """Cluster deterministic candidates, falling back safely if Stage A is unavailable.

    Uses its own `max_attempts` (one more than the shared DEEPSEEK_MAX_ATTEMPTS
    used by Stage B/Layer 2) rather than the shared constant directly: the
    observed failures here are LLM structural-output-contract violations
    (e.g. "candidate_id 不在候选池", "所有 candidate 必须恰好被一个 event 覆盖"),
    not network errors, and are plausibly transient given LLM output
    non-determinism -- one extra attempt is a cheap, isolated mitigation that
    does not change Stage B/Layer 2 retry behavior or any selection rule.
    """
    cluster_input = _cluster_candidate_input(candidates)
    _log_stage_a_cap(candidates, cluster_input)
    print(f"[NEWS STAGE A] input candidates: {len(cluster_input)}")
    if len(cluster_input) <= 1:
        events = _fallback_events(cluster_input)
        _log_stage_a_events(events)
        _log_stage_a_mapping(events)
        return events, None
    user_payload = json.dumps({"candidates": cluster_input}, ensure_ascii=False)
    last_error = None
    started = time.monotonic()
    for attempt in range(max_attempts):
        try:
            raw = invoke_model(
                call_model, SYSTEM_PROMPT, user_payload, api_key,
                thinking_enabled=False, reasoning_effort=NEWS_REASONING_EFFORT,
                stage="Stage A", attempt=attempt + 1, usage_tracker=usage_tracker,
            )
            try:
                events = validate_event_clusters(raw, cluster_input)
            except Exception as exc:
                if usage_tracker is not None:
                    usage_tracker.record_validation_failure("Stage A", attempt + 1, exc)
                raise
            _log_stage_a_events(events)
            _log_stage_a_mapping(events)
            print(f"[NEWS AI] event clustering succeeded in {time.monotonic() - started:.1f}s")
            return events, None
        except NonRetryableAPIError:
            # Handled by cluster_news_events_batched: no further attempt here and
            # no further batch there.
            print("[NEWS AI] event clustering aborted on non-retryable API error, no further attempts")
            raise
        except Exception as exc:
            last_error = exc
            print(
                f"[NEWS AI] event clustering attempt {attempt + 1}/{max_attempts} failed "
                f"after {time.monotonic() - started:.1f}s: {exc}"
            )
            if attempt < max_attempts - 1:
                sleep_fn((5, 10, 10)[min(attempt, 2)])
    events = _fallback_events(cluster_input)
    _log_stage_a_events(events)
    _log_stage_a_mapping(events)
    return events, (
        "⚠️ 新闻事件级去重暂时失败，已使用基础去重结果继续生成日报。"
        f" 原因：{last_error}"
    )


def batch_stage_a_input(candidates: list[dict], batch_size: int = STAGE_A_MAX_INPUT) -> list[list[dict]]:
    """Split the FULL dedup'd candidate pool into batches of at most
    `batch_size`, so every candidate reaches at least one real Stage A
    clustering request -- unlike the old single [:STAGE_A_MAX_INPUT] slice
    inside _cluster_candidate_input, which permanently dropped anything past
    the top 50 by composite score (confirmed live on 2026-09-07: 72 dedup'd
    candidates, only 50 ever reached Stage A).

    Candidates are ranked with the same priority/importance/composite-score
    logic used everywhere else in this module (_rank_stage_a_candidates), then
    sliced into batches in that order, mirroring batch_stage_b_input: the
    highest-value candidates land in the first (always-run) batch.
    """
    if not candidates:
        return []
    ranked = [item for item, _, _, _, _, _ in _rank_stage_a_candidates(candidates)]
    return [ranked[start:start + batch_size] for start in range(0, len(ranked), batch_size)]


def _events_are_likely_duplicates(a: dict, b: dict) -> bool:
    """Lightweight, deterministic (no LLM) same-real-event check used only to
    merge across independently-clustered Stage A batches -- within one batch,
    Stage A's own clustering already handles this. Two different batches can
    each cluster a different article about the same real event without ever
    seeing each other's candidates, so a plain text-similarity check on the
    event_summary catches the obvious cases (near-identical wording) without
    a second LLM request."""
    text_a = (a.get("event_summary") or "").strip().lower()
    text_b = (b.get("event_summary") or "").strip().lower()
    if not text_a or not text_b:
        return False
    return difflib.SequenceMatcher(None, text_a, text_b).ratio() >= 0.72


def _merge_cross_batch_duplicate_events(events: list[dict]) -> list[dict]:
    """Merge events from different Stage A batches that likely describe the
    same real-world event, so the same story doesn't reach Stage B twice just
    because its two source articles landed in different batches. Prefers a
    successfully-classified topic_group over a fallback OTHER_SYSTEMIC one,
    and keeps the longer (more informative) event_summary."""
    merged: list[dict] = []
    for event in events:
        match = next((existing for existing in merged if _events_are_likely_duplicates(existing, event)), None)
        if match is None:
            merged.append({**event, "candidate_ids": list(event["candidate_ids"])})
            continue
        match["candidate_ids"] = list(dict.fromkeys(match["candidate_ids"] + event["candidate_ids"]))
        if match["topic_group"] == "OTHER_SYSTEMIC" and event["topic_group"] != "OTHER_SYSTEMIC":
            match["topic_group"] = event["topic_group"]
            match["event_category"] = event["event_category"]
        if len(event.get("event_summary") or "") > len(match.get("event_summary") or ""):
            match["event_summary"] = event["event_summary"]
    return merged


def cluster_news_events_batched(candidates: list[dict], api_key: str,
                                call_model: Callable = call_deepseek,
                                sleep_fn: Callable = time.sleep,
                                usage_tracker: DeepSeekUsageTracker | None = None,
                                max_attempts: int = DEEPSEEK_MAX_ATTEMPTS + 1,
                                batch_size: int | None = None,
                                observability: dict | None = None) -> tuple[list[dict], Optional[str]]:
    """Run Stage A clustering (cluster_news_events) over the FULL dedup'd
    candidate pool, split into batches so no single request grows large enough
    to risk the timeout that used to motivate a hard, permanently-dropping cap
    (the old bare [:50] inside _cluster_candidate_input). Unlike that cap, no
    candidate is silently skipped: batch_stage_a_input partitions every
    candidate into some batch, and every batch is run through the same
    cluster_news_events pipeline (its own retries, then its own deterministic
    per-candidate fallback classification) a single un-batched pool used to
    get -- so a batch that fails only ever costs that batch's own candidates
    their AI clustering, never any other batch's.

    Events from different batches that likely describe the same real-world
    event are merged (see _merge_cross_batch_duplicate_events) so the same
    story doesn't reach Stage B twice just because its source articles landed
    in different batches -- a lightweight text-similarity check, not a second
    LLM request.
    """
    if observability is not None:
        observability["stage_a_total_candidate_count"] = len(candidates)
    if not candidates:
        if observability is not None:
            observability.update({
                "stage_a_batch_count": 0, "stage_a_batch_sizes": [], "stage_a_batch_fallback_used": [],
                "stage_a_uncovered_candidate_count": 0, "stage_a_final_event_count": 0,
                "stage_a_non_retryable_abort": False, "stage_a_aborted_at_batch": None,
            })
        return [], None

    batches = batch_stage_a_input(candidates, batch_size=batch_size or STAGE_A_MAX_INPUT)
    if observability is not None:
        observability["stage_a_batch_count"] = len(batches)
        observability["stage_a_batch_sizes"] = [len(batch) for batch in batches]
        observability["stage_a_batch_fallback_used"] = []
        observability["stage_a_batch_output_counts"] = []
        observability["stage_a_non_retryable_abort"] = False
        observability["stage_a_aborted_at_batch"] = None

    all_events: list[dict] = []
    batch_warnings: list[str] = []
    covered_ids: set[str] = set()
    aborted = False
    for index, batch in enumerate(batches, start=1):
        if aborted:
            # Same deterministic one-event-per-candidate fallback a failed batch
            # already used -- just without sending a request that cannot succeed.
            events = _fallback_events(_cluster_candidate_input(batch))
            covered_ids.update(item["candidate_id"] for item in batch)
            all_events.extend(events)
            print(
                f"[NEWS STAGE A BATCH] batch={index}/{len(batches)} input={len(batch)} "
                f"output_events={len(events)} skipped=non_retryable_abort fallback_used=True"
            )
            if observability is not None:
                observability["stage_a_batch_fallback_used"].append(True)
                observability["stage_a_batch_output_counts"].append(len(events))
            continue
        if True:
            try:
                events, warning = cluster_news_events(
                    batch, api_key, call_model, sleep_fn, usage_tracker, max_attempts
                )
            except NonRetryableAPIError as exc:
                aborted = True
                events = _fallback_events(_cluster_candidate_input(batch))
                warning = (
                    "⚠️ 新闻事件级去重已中止，已使用基础去重结果继续生成日报。"
                    f" 原因：api_error_{exc.status_code or 'auth'}: {exc}"
                )
                if observability is not None:
                    observability["stage_a_non_retryable_abort"] = True
                    observability["stage_a_aborted_at_batch"] = index
        covered_ids.update(item["candidate_id"] for item in batch)
        all_events.extend(events)
        fallback_used = warning is not None
        print(
            f"[NEWS STAGE A BATCH] batch={index}/{len(batches)} input={len(batch)} "
            f"output_events={len(events)} fallback_used={fallback_used}"
        )
        if observability is not None:
            observability["stage_a_batch_fallback_used"].append(fallback_used)
            observability["stage_a_batch_output_counts"].append(len(events))
        if warning:
            batch_warnings.append(f"batch {index}/{len(batches)}: {warning}")

    all_ids = {item["candidate_id"] for item in candidates}
    uncovered = all_ids - covered_ids
    if uncovered:
        print(f"[NEWS STAGE A BATCH] WARNING uncovered candidates (should never happen): {sorted(uncovered)}")
    if observability is not None:
        observability["stage_a_uncovered_candidate_count"] = len(uncovered)

    merged_events = _merge_cross_batch_duplicate_events(all_events)
    print(
        f"[NEWS STAGE A BATCH] final_event_count={len(merged_events)} (pre_merge={len(all_events)}) "
        f"from {len(batches)} batches over {len(candidates)} candidates | uncovered={len(uncovered)}"
    )
    if observability is not None:
        observability["stage_a_final_event_count"] = len(merged_events)

    warning = "; ".join(batch_warnings) if batch_warnings else None
    return merged_events, warning


def _log_stage_b_batches(pool: list[dict], batches: list[list[dict]]) -> None:
    ranking = _rank_stage_a_candidates(pool)
    batch_by_candidate = {
        item["candidate_id"]: index
        for index, batch in enumerate(batches, 1)
        for item in batch
    }
    print(
        f"[NEWS STAGE B BATCH PLAN] total_candidates={len(pool)} batch_count={len(batches)} "
        f"batch_sizes={[len(batch) for batch in batches]}"
    )
    for rank, (item, importance_score, reason, source_penalty, priority_base, composite_score) in enumerate(ranking, 1):
        candidate_id = item.get("candidate_id", "")
        print(
            f"[NEWS STAGE B BATCH PLAN ITEM] candidate_id={candidate_id} "
            f"| source={item.get('source', '')} | composite_score={composite_score:g} "
            f"| importance_reason={reason} | rank={rank} | batch={batch_by_candidate.get(candidate_id, '<none>')}"
        )


def batch_stage_b_input(candidates: list[dict], batch_size: int = STAGE_B_MAX_INPUT) -> list[list[dict]]:
    """Split the FULL Stage B input pool into batches of at most `batch_size`,
    so every candidate is guaranteed to reach at least one real Stage B request
    -- unlike the old select_stage_b_input, which permanently dropped anything
    past the first `batch_size` (confirmed live on 2026-09-07: Stage A produced
    50 events but only the top 28 by composite score were ever sent to Stage B;
    the other 22, including a real Nvidia/Hugging Face acquisition story, were
    silently never judged by any model).

    Candidates are first ranked with the same priority/importance/composite-score
    logic used everywhere else in this module (`_rank_stage_a_candidates`), then
    sliced into batches in that order -- so the highest-value candidates land in
    the first batch (the one guaranteed to run even under the tightest time
    budget), and each batch is still small enough to fit the same per-request
    timeout budget a single un-batched call already had to meet. `batch_size`
    bounds a single request's size, not the total candidate pool: unlike the
    old cap, nothing here is ever dropped -- see the caller (select_news_multi_batch
    in deepseek_client.py) for how every batch is then actually sent to Stage B.
    """
    if not candidates:
        return []
    ranked = [item for item, _, _, _, _, _ in _rank_stage_a_candidates(candidates)]
    batches = [ranked[start:start + batch_size] for start in range(0, len(ranked), batch_size)]
    _log_stage_b_batches(candidates, batches)
    return batches


# Stage B's deterministic fallback (build_deterministic_fallback_selection) is
# an ELIGIBILITY FILTER, not a ranking cut: a candidate is only ever included
# if it clears one of these four high-confidence rules. There is deliberately
# no minimum or maximum count -- quality over quantity. Confirmed necessary
# via real 2026-09-07 replay: the previous composite-score top-8 cut surfaced
# Turkey's GDP forecast, a German state election, parenting advice, and sugar
# prices onto the homepage the moment two of three real Stage B batches timed
# out, none of which belong in front of a long-term SPY/Nasdaq-100 investor.
#
# Rule A -- US macro/Fed/Treasury: specific phrases only, never bare generic
# words like "rates" or "jobs" (which show up in unrelated contexts -- see the
# AMBIGUOUS_TERMS_REQUIRING_CONTEXT precedent above for the same lesson).
FALLBACK_MACRO_STRICT_TERMS = (
    "federal reserve", "fed", "fomc",
    "us inflation", "cpi", "core cpi", "pce",
    "us employment", "nonfarm payroll", "payrolls", "unemployment rate",
    "us treasury", "treasury yield", "us bond market",
    "us fiscal", "us tariff", "us trade policy",
)

# Rule B -- a tracked mega-cap company AND a major event type. A company name
# alone is never enough (that was the exact 2026-09-07 "Model fatigue" lesson
# applied to Stage A; the same principle holds here). "lawsuit"/"sue" are
# deliberately NOT in the major-event list: an LLM can judge whether a lawsuit
# is actually material (see news_prompt.py's "「重大诉讼或和解」的入选门槛"),
# but this deterministic filter cannot, and bare "lawsuit" is exactly what let
# an ordinary Seattle Times/Newsday copyright suit slip through before that
# prompt rule existed -- so a lawsuit only qualifies here via a stronger,
# less ambiguous signal (antitrust/DOJ/FTC involvement).
FALLBACK_MEGA_CAP_COMPANIES = (
    "apple", "microsoft", "alphabet", "google", "amazon", "meta", "nvidia", "tesla", "spacex",
)
FALLBACK_MEGA_CAP_MAJOR_EVENT_TERMS = (
    "earnings", "quarterly results", "guidance",
    "acquisition", "acquire", "acquires", "merger", "investment",
    "capital expenditure", "capex",
    "unveil", "launch",
    "government contract", "nasa contract",
    "antitrust", "doj", "ftc",
)
# "crash"/"explosion"/"mission failure"/"anomaly"/"starship"/"starlink" only
# count for SpaceX specifically -- SpaceX's entire business is spaceflight, so
# a launch/mission accident is a core business event; the same words applied
# to any of the other eight companies is not (confirmed via real 2026-09-07
# data: "Five dead after Amazon cargo plane crashes at Miami airport" is a
# tragic accident involving an Amazon-branded cargo flight, not a material
# Amazon business event, and must not qualify just because "Amazon" + "crash"
# both appear).
FALLBACK_SPACEX_ONLY_EVENT_TERMS = ("starship", "starlink", "explosion", "crash", "mission failure", "anomaly")

# Bare "CEO"/"chief executive" is deliberately NOT a qualifying major-event
# term (removed from the list above): a company mentioning its CEO at all --
# an interview, a pay-package writeup, a routine appearance -- is not itself a
# major event. Confirmed via real 2026-09-07 data: "Apple's New CEO-Like Cook
# Pay Package Shows He's Going Nowhere" matched the old bare "ceo" term even
# though the story is explicitly about Cook *staying*, not any leadership
# change. Only real executive-change language qualifies.
FALLBACK_EXECUTIVE_CHANGE_TERMS = (
    "resign", "resignation", "step down", "steps down", "stepping down",
    "appointed ceo", "appoints ceo", "named ceo", "names ceo",
    "replaced", "succession", "successor",
    "fired", "removed", "departure",
    "leadership reorganization", "leadership shakeup",
)

# Rule C reuses _geopolitics_policy_matches (region + real economic/market-
# transmission signal, both required) as its base gate, plus one more
# condition below (_fallback_matches_geopolitics) for policy-type signals
# specifically (tariff/sanction/export control/trade/banking): those need an
# actual policy-action state change, not just commentary about an existing
# policy. A kinetic/already-happened signal (war, a military strike, an oil-
# shipping disruption, etc.) needs no such extra confirmation -- the verb
# itself already describes something that happened.
#
# Confirmed via real 2026-09-07 data: "Rep. Stevens Says Canadian Tariffs Are
# Squeezing Michigan" matched the base region ("Canadian") + signal ("tariff")
# check, but the story is a politician's commentary on an *existing* tariff
# ("raising costs"), not a new policy action -- deterministic fallback cannot
# tell "a new tariff was just imposed" from "a politician commented on tariffs
# that already exist" the way the Stage B AI prompt can.
FALLBACK_GEOPOLITICS_KINETIC_SIGNALS = (
    "oil", "energy", "gas", "shipping", "strait", "supply chain",
    "war", "military strike", "missile", "conflict", "commodity", "market disruption",
    "tanker attack", "tanker strike",
)
FALLBACK_GEOPOLITICS_POLICY_SIGNALS = (
    "sanction", "tariff", "trade", "export control", "chip restriction", "financial sanction", "banking",
)
FALLBACK_POLICY_ACTION_TERMS = (
    "announced", "announces", "imposed", "imposes", "enacted", "effective",
    "raised", "raises", "cut", "cuts", "removed", "removes", "suspended", "suspends",
    "retaliatory tariff", "export control introduced", "export controls tightened",
)


def _fallback_matches_geopolitics(text: str) -> bool:
    if not _geopolitics_policy_matches(text):
        return False
    if any(_keyword_present(text, word) for word in FALLBACK_GEOPOLITICS_KINETIC_SIGNALS):
        return True
    if any(_keyword_present(text, word) for word in FALLBACK_GEOPOLITICS_POLICY_SIGNALS):
        return any(_keyword_present(text, term) for term in FALLBACK_POLICY_ACTION_TERMS)
    return False

# Rule D -- energy: a core oil/OPEC term AND a real supply/shipping disruption
# signal, both required, mirroring geopolitics's two-tier structure. Ordinary
# commodity coverage (sugar, agricultural prices, a routine "oil market
# roundup" with no disruption) does not qualify just for mentioning oil.
FALLBACK_ENERGY_CORE_TERMS = ("oil", "crude", "opec")
FALLBACK_ENERGY_DISRUPTION_TERMS = (
    "supply disruption", "shipping disruption", "strait", "pipeline attack",
    "production cut", "supply cut", "export ban", "blockade", "tanker attack", "tanker strike",
)


def _fallback_high_confidence_rule(item: dict) -> Optional[str]:
    """Return the name of the one high-confidence rule `item` clears, or None
    if it clears none of them.

    Text is drawn from title + event_summary ONLY -- deliberately excluding
    the raw article `summary` field. Confirmed necessary via real 2026-09-07
    data: "Turkey Cuts 2027 GDP Growth Forecast as Elections Beckon" (an
    ordinary foreign-country GDP story) has a summary that happens to mention
    "fallout from the Iran war reverberates" purely as passing background
    color, which alone satisfied the geopolitics rule's region+signal check --
    even though the story is not about Iran or the war at all. Stage A's own
    event_summary is a distilled, on-topic factual restatement (here: "Turkey
    cut its 2027 GDP growth forecast.") and does not carry that kind of
    incidental cross-reference, making title+event_summary a much more
    reliable signal of what a story is actually about than the original
    (often noisy) article summary.
    """
    text = " ".join(str(item.get(field) or "") for field in ("title", "event_summary")).lower()
    if any(_keyword_present(text, term) for term in FALLBACK_MACRO_STRICT_TERMS):
        return "us_macro_fed_treasury"
    matched_companies = [word for word in FALLBACK_MEGA_CAP_COMPANIES if _keyword_present(text, word)]
    if matched_companies:
        has_major_event = any(_keyword_present(text, word) for word in FALLBACK_MEGA_CAP_MAJOR_EVENT_TERMS)
        has_executive_change = any(_keyword_present(text, word) for word in FALLBACK_EXECUTIVE_CHANGE_TERMS)
        has_spacex_event = "spacex" in matched_companies and any(
            _keyword_present(text, word) for word in FALLBACK_SPACEX_ONLY_EVENT_TERMS
        )
        if has_major_event or has_executive_change or has_spacex_event:
            return "mega_cap_major_event"
    if _fallback_matches_geopolitics(text):
        return "geopolitics_region_and_market_transmission"
    has_energy_core = any(_keyword_present(text, word) for word in FALLBACK_ENERGY_CORE_TERMS)
    has_disruption = any(_keyword_present(text, word) for word in FALLBACK_ENERGY_DISRUPTION_TERMS)
    if has_energy_core and has_disruption:
        return "energy_supply_disruption"
    return None


def build_deterministic_fallback_selection(candidates: list[dict]) -> list[dict]:
    """Strict, high-confidence-only Stage B replacement for when the two-pass
    AI selection actually fails (see select_news_with_fallback for exactly
    when this runs -- never merely because the AI legitimately selected zero
    items). Never calls the LLM.

    This is a pure ELIGIBILITY FILTER (see the four rules above), not a
    ranking cut: every candidate is checked against all four rules, and only
    those that clear one are kept, in composite-score order for display.
    Returning [] is the correct, expected result when nothing in this batch
    clears the bar -- there is no padding to reach any target count.

    title_zh/summary_zh reuse Stage A's own Chinese event_summary (falling back
    to the original English title/summary only when no event_summary exists)
    instead of fabricating an AI translation; investment_relevance_score and
    tags are left unset/empty rather than invented, and selection_reason names
    the specific rule that qualified the item so it is never mistaken for one
    the AI itself vetted.
    """
    if not candidates:
        return []
    fallback = []
    for item, importance_score, signal_reason, source_penalty, priority_base, composite_score in _rank_stage_a_candidates(candidates):
        rule = _fallback_high_confidence_rule(item)
        if rule is None:
            continue
        original_title = item.get("title", "")
        event_summary = (item.get("event_summary") or "").strip()
        zh_text = event_summary or original_title
        fallback.append({
            "rank": len(fallback) + 1,
            "candidate_id": item["candidate_id"],
            "category": TOPIC_GROUP_TO_CATEGORY.get(item.get("topic_group"), "美国经济"),
            "title_zh": zh_text[:TITLE_ZH_LIMIT],
            "summary_zh": zh_text[:SUMMARY_ZH_LIMIT],
            "focus": "",
            "tags": [],
            "investment_relevance_score": None,
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "published_at": item.get("published_at"),
            "original_title": original_title,
            "selection_reason": (
                f"确定性降级选取（AI 筛选失败）：命中高置信规则「{rule}」 "
                f"| composite_score={composite_score:g} (priority_base={priority_base:g}, "
                f"importance_score={importance_score}, source_penalty={source_penalty:g}, signals={signal_reason})"
            ),
            "event_summary": item.get("event_summary", original_title),
            "topic_group": item.get("topic_group"),
            "event_category": item.get("event_category", "other"),
            "source_channel": item.get("source_channel"),
            "fallback": True,
            "selection_mode": "deterministic_fallback",
        })
    return fallback
