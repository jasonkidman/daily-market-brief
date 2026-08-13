"""DeepSeek transport, strict selection validation, and graceful degradation."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from .news_prompt import SYSTEM_PROMPT


ALLOWED_CATEGORIES = {"市场 / 宏观", "AI / 科技", "全球事件"}


class NewsSelectionError(ValueError):
    """Raised when model output violates the candidate-only contract."""


def _parse_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise NewsSelectionError("DeepSeek 输出不是 JSON 对象。")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise NewsSelectionError("DeepSeek 输出无法解析为 JSON。") from exc


def validate_selection(payload: Any, candidates: list[dict]) -> list[dict]:
    data = _parse_payload(payload)
    news = data.get("news")
    if not isinstance(news, list):
        raise NewsSelectionError("news 必须是数组。")
    if len(news) > 8:
        raise NewsSelectionError("news 数量不得超过 8。")
    pool = {item["candidate_id"]: item for item in candidates}
    seen, validated = set(), []
    for item in news:
        if not isinstance(item, dict):
            raise NewsSelectionError("news 条目必须是对象。")
        candidate_id = item.get("candidate_id")
        rank = item.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= len(news):
            raise NewsSelectionError("rank 不合法。")
        if candidate_id not in pool:
            raise NewsSelectionError("candidate_id 不在候选池。")
        if candidate_id in seen:
            raise NewsSelectionError("candidate_id 不得重复。")
        if item.get("category") not in ALLOWED_CATEGORIES:
            raise NewsSelectionError("category 不合法。")
        if not str(item.get("title_zh", "")).strip() or not str(item.get("summary_zh", "")).strip():
            raise NewsSelectionError("中文标题和摘要不能为空。")
        seen.add(candidate_id)
        source = pool[candidate_id]
        validated.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "category": item["category"],
            "title_zh": str(item["title_zh"]).strip(),
            "summary_zh": str(item["summary_zh"]).strip(),
            "source": source["source"],
            "url": source["url"],
            "published_at": source["published_at"],
            "original_title": source["title"],
            "selection_reason": str(item.get("selection_reason", "")).strip(),
            "event_summary": source.get("event_summary", source["title"]),
            "topic_group": source.get("topic_group"),
        })
    if sorted(item["rank"] for item in validated) != list(range(1, len(validated) + 1)):
        raise NewsSelectionError("rank 必须从 1 连续排列。")
    return sorted(validated, key=lambda item: item["rank"])


def call_deepseek(system_prompt: str, user_payload: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def select_news(candidates: list[dict], api_key: str, recent_selected: list[dict] = None,
                market_context: dict = None,
                call_model: Callable = call_deepseek, sleep_fn: Callable = time.sleep
                ) -> tuple[list[dict], Optional[str]]:
    if not candidates:
        return [], None
    event_fields = (
        "candidate_id", "event_summary", "topic_group", "source", "title", "summary", "published_at",
    )
    events = [{
        **{field: candidate.get(field) for field in event_fields},
        "event_summary": candidate.get("event_summary", candidate.get("title", "")),
        "topic_group": candidate.get("topic_group", "OTHER_SYSTEMIC"),
    } for candidate in candidates]
    payload = {"events": events, "recent_7_days_events": recent_selected or []}
    if market_context:
        payload.update(market_context)
    user_payload = json.dumps(payload, ensure_ascii=False)
    last_error = None
    for attempt in range(3):
        try:
            raw = call_model(SYSTEM_PROMPT, user_payload, api_key)
            return validate_selection(raw, candidates), None
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                sleep_fn((5, 10)[attempt])
    return [], f"⚠️ 新闻 AI 处理暂时失败；RSS 数据已获取，等待下一次更新。原因：{last_error}"
