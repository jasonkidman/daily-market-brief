"""Event-level RSS news clustering and deterministic representative selection."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Optional

from .deepseek_client import (
    DEEPSEEK_MAX_ATTEMPTS, SUMMARY_ZH_LIMIT, TITLE_ZH_LIMIT, DeepSeekUsageTracker, call_deepseek, invoke_model,
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

# Stage B's own input pool is capped the same way Stage A's clustering input is:
# a pure ranking cut, no extra LLM call. Keeps Stage B's per-request payload (and
# therefore its odds of finishing inside its timeout) bounded regardless of how
# many events Stage A produces on a heavy news day.
STAGE_B_MAX_INPUT = 28

# Used only by the deterministic Stage B fallback below, never by the real AI
# selection: a coarse, static bucket from Stage A's topic_group to one of Stage
# B's own category labels, so a fallback item still renders with a category
# instead of leaving the field blank. This is a fixed lookup, not a judgement
# call, so it is not "fabricating an AI field".
FALLBACK_MAX_ITEMS = 8
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
    """Bound the clustering prompt without exposing article URLs to the model."""
    limited = [item for item, _, _, _, _, _ in _rank_stage_a_candidates(candidates)[:50]]
    fields = ("candidate_id", "source", "priority", "title", "summary", "published_at")
    return [{field: item.get(field, "") for field in fields} for item in limited]


def stage_a_input_counts(candidates: list[dict]) -> tuple[int, int]:
    """Return pre-cap and actual Stage A input counts without changing selection."""
    return len(candidates), len(_cluster_candidate_input(candidates))


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
    return [{
        "event_id": f"fallback_{index:03d}",
        "candidate_ids": [item["candidate_id"]],
        "event_summary": item.get("title", "新闻事件"),
        "topic_group": "OTHER_SYSTEMIC",
        "event_category": "other",
    } for index, item in enumerate(candidates, 1)]


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
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=False, reasoning_effort=None,
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


def _log_stage_b_input_cap(pool: list[dict], capped: list[dict]) -> None:
    capped_ids = {item["candidate_id"] for item in capped}
    ranking = _rank_stage_a_candidates(pool)
    print(
        f"[NEWS STAGE B CAP] pre_filter_pool={len(pool)} stage_b_input={len(capped)} "
        f"cap_dropped={len(pool) - len(capped)}"
    )
    for rank, (item, importance_score, reason, source_penalty, priority_base, composite_score) in enumerate(ranking, 1):
        action = "keep" if item["candidate_id"] in capped_ids else "drop"
        print(
            f"[NEWS STAGE B CAP ITEM] candidate_id={item.get('candidate_id', '')} "
            f"| source={item.get('source', '')} | composite_score={composite_score:g} "
            f"| importance_reason={reason} | pre_filter_rank={rank} | action={action}"
        )


def select_stage_b_input(candidates: list[dict], max_input: int = STAGE_B_MAX_INPUT) -> list[dict]:
    """Deterministically bound Stage B's LLM input before it is ever called.

    Reuses the exact same priority/importance/composite-score/source-diversity/
    recency ranking as the Stage A clustering input cap (`_rank_stage_a_candidates`)
    rather than adding a second LLM call: a heavy news day used to hand Stage B all
    40+ Stage A events in one request, which is what pushed it past its own timeout
    budget. `candidates` is expected to be the Stage B input pool (one representative
    per Stage A event, from `event_selection_candidates`) -- its priority/title/
    summary/published_at/source fields are exactly what the ranking function reads.
    """
    if len(candidates) <= max_input:
        _log_stage_b_input_cap(candidates, candidates)
        return candidates
    ranked = _rank_stage_a_candidates(candidates)
    capped = [item for item, _, _, _, _, _ in ranked[:max_input]]
    _log_stage_b_input_cap(candidates, capped)
    return capped


def build_deterministic_fallback_selection(candidates: list[dict], max_items: int = FALLBACK_MAX_ITEMS) -> list[dict]:
    """Rule-based Stage B replacement for when the two-pass AI selection comes back
    with nothing (e.g. both samples time out after their own retries). Never calls
    the LLM: candidates are ranked with the same priority/importance/composite-score
    logic as the Stage A and Stage B input caps (`_rank_stage_a_candidates`), and
    only a small, high-confidence prefix is kept -- an empty or short result here is
    expected behavior (favor recall precision over padding to `max_items`), not a bug.

    title_zh/summary_zh reuse Stage A's own Chinese event_summary (falling back to
    the original English title/summary only when no event_summary exists) instead of
    fabricating an AI translation; investment_relevance_score and tags are left
    unset/empty rather than invented, and selection_reason spells out the ranking
    signals that were actually used so a fallback item is never mistaken for one the
    AI itself vetted.
    """
    if not candidates:
        return []
    ranked = _rank_stage_a_candidates(candidates)[:max_items]
    fallback = []
    for rank, (item, importance_score, signal_reason, source_penalty, priority_base, composite_score) in enumerate(
        ranked, start=1
    ):
        original_title = item.get("title", "")
        event_summary = (item.get("event_summary") or "").strip()
        zh_text = event_summary or original_title
        fallback.append({
            "rank": rank,
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
                f"确定性降级选取（AI 筛选超时未产出结果）：composite_score={composite_score:g} "
                f"(priority_base={priority_base:g}, importance_score={importance_score}, "
                f"source_penalty={source_penalty:g}, signals={signal_reason})"
            ),
            "event_summary": item.get("event_summary", original_title),
            "topic_group": item.get("topic_group"),
            "event_category": item.get("event_category", "other"),
            "source_channel": item.get("source_channel"),
            "fallback": True,
        })
    return fallback
