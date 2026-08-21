"""Event-level RSS news clustering and deterministic representative selection."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable, Optional

from .deepseek_client import DEEPSEEK_MAX_ATTEMPTS, call_deepseek, invoke_model
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
    """Flatten one program-selected article per event for the Top 8 stage."""
    return [{
        **event["representative"],
        "event_summary": event["event_summary"],
        "topic_group": event["topic_group"],
        "event_category": event.get("event_category", "other"),
    } for event in event_representatives]


def _cluster_candidate_input(candidates: list[dict]) -> list[dict]:
    """Bound the clustering prompt without exposing article URLs to the model."""
    limited = sorted(candidates, key=lambda item: (
        PRIORITY_ORDER.get(item.get("priority", "P2"), 2),
        -_published_at_value(item),
    ))[:50]
    fields = ("candidate_id", "source", "priority", "title", "summary", "published_at")
    return [{field: item.get(field, "") for field in fields} for item in limited]


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
                        sleep_fn: Callable = time.sleep) -> tuple[list[dict], Optional[str]]:
    """Cluster deterministic candidates, falling back safely if Stage A is unavailable."""
    cluster_input = _cluster_candidate_input(candidates)
    print(f"[NEWS STAGE A] input candidates: {len(cluster_input)}")
    if len(cluster_input) <= 1:
        events = _fallback_events(cluster_input)
        _log_stage_a_events(events)
        return events, None
    user_payload = json.dumps({"candidates": cluster_input}, ensure_ascii=False)
    last_error = None
    started = time.monotonic()
    for attempt in range(DEEPSEEK_MAX_ATTEMPTS):
        try:
            raw = invoke_model(
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=False, reasoning_effort=None
            )
            events = validate_event_clusters(raw, cluster_input)
            _log_stage_a_events(events)
            print(f"[NEWS AI] event clustering succeeded in {time.monotonic() - started:.1f}s")
            return events, None
        except Exception as exc:
            last_error = exc
            print(
                f"[NEWS AI] event clustering attempt {attempt + 1}/{DEEPSEEK_MAX_ATTEMPTS} failed "
                f"after {time.monotonic() - started:.1f}s: {exc}"
            )
            if attempt < DEEPSEEK_MAX_ATTEMPTS - 1:
                sleep_fn((5, 10)[attempt])
    events = _fallback_events(cluster_input)
    _log_stage_a_events(events)
    return events, (
        "⚠️ 新闻事件级去重暂时失败，已使用基础去重结果继续生成日报。"
        f" 原因：{last_error}"
    )
