"""DeepSeek transport, strict selection validation, and graceful degradation."""

from __future__ import annotations

import json
import inspect
import time
from typing import Any, Callable, Optional

from .news_prompt import SYSTEM_PROMPT


ALLOWED_CATEGORIES = {
    "美联储 / 利率",
    "就业 / 通胀",
    "美国经济",
    "美债 / 美元",
    "金融市场",
    "AI / 资本开支",
    "半导体",
    "地缘政治",
    "政策 / 监管",
}
TITLE_ZH_LIMIT = 70
SUMMARY_ZH_LIMIT = 180
INVESTMENT_IMPACT_LIMIT = 220
FOCUS_LIMIT = 80
SELECTION_REASON_LIMIT = 120
TAG_LIMIT = 16


class NewsSelectionError(ValueError):
    """Raised when model output violates the candidate-only contract."""


def invoke_model(call_model: Callable, system_prompt: str, user_payload: str, api_key: str, *,
                 thinking_enabled: bool, reasoning_effort: str | None) -> str:
    """Call production transport with reasoning settings while supporting legacy test injectables."""
    parameters = inspect.signature(call_model).parameters.values()
    accepts_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_keywords or {"thinking_enabled", "reasoning_effort"}.issubset(inspect.signature(call_model).parameters):
        return call_model(
            system_prompt, user_payload, api_key,
            thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort,
        )
    return call_model(system_prompt, user_payload, api_key)


def _parse_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise NewsSelectionError("DeepSeek 输出不是 JSON 对象。")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise NewsSelectionError("DeepSeek 输出无法解析为 JSON。") from exc


def _required_text(item: dict, key: str, limit: int) -> str:
    raw_value = item.get(key)
    if not isinstance(raw_value, str):
        raise NewsSelectionError(f"{key} 不合法。")
    value = raw_value.strip()
    if not value or len(value) > limit:
        raise NewsSelectionError(f"{key} 不合法。")
    return value


def _has_impact_path(value: str) -> bool:
    if value == "短期资产价格影响有限，暂以观察为主。":
        return True
    variables = (
        "SPY", "Nasdaq", "纳指", "科技股", "美债", "收益率", "美元", "信用", "AI", "半导体",
        "估值", "通胀", "油价", "利率", "金融条件", "就业", "盈利", "流动性", "风险偏好",
    )
    connectors = ("→", "若", "如果", "导致", "使得", "进而", "从而", "有利于", "压制")
    return any(term in value for term in variables) and any(term in value for term in connectors)


def _validated_tags(item: dict) -> list[str]:
    tags = item.get("tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= 4:
        raise NewsSelectionError("tags 不合法。")
    if any(not isinstance(tag, str) or not tag.strip() or len(tag.strip()) > TAG_LIMIT for tag in tags):
        raise NewsSelectionError("tags 不合法。")
    return [tag.strip() for tag in tags]


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
        title_zh = _required_text(item, "title_zh", TITLE_ZH_LIMIT)
        summary_zh = _required_text(item, "summary_zh", SUMMARY_ZH_LIMIT)
        investment_impact = _required_text(item, "investment_impact", INVESTMENT_IMPACT_LIMIT)
        if not _has_impact_path(investment_impact):
            raise NewsSelectionError("investment_impact 不合法。")
        focus = _required_text(item, "focus", FOCUS_LIMIT)
        tags = _validated_tags(item)
        score = item.get("investment_relevance_score")
        if not isinstance(score, int) or isinstance(score, bool) or not 50 <= score <= 100:
            raise NewsSelectionError("investment_relevance_score 不合法。")
        selection_reason = _required_text(item, "selection_reason", SELECTION_REASON_LIMIT)
        seen.add(candidate_id)
        source = pool[candidate_id]
        validated.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "category": item["category"],
            "title_zh": title_zh,
            "summary_zh": summary_zh,
            "investment_impact": investment_impact,
            "focus": focus,
            "tags": tags,
            "investment_relevance_score": score,
            "source": source["source"],
            "url": source["url"],
            "published_at": source["published_at"],
            "original_title": source["title"],
            "selection_reason": selection_reason,
            "event_summary": source.get("event_summary", source["title"]),
            "topic_group": source.get("topic_group"),
        })
    if sorted(item["rank"] for item in validated) != list(range(1, len(validated) + 1)):
        raise NewsSelectionError("rank 必须从 1 连续排列。")
    validated.sort(key=lambda item: item["rank"])
    if any(
        item["investment_relevance_score"] < next_item["investment_relevance_score"]
        for item, next_item in zip(validated, validated[1:])
    ):
        raise NewsSelectionError("investment_relevance_score 必须按 rank 非递增排列。")
    topic_counts: dict[str, int] = {}
    for item in validated:
        topic_group = item["topic_group"]
        if not topic_group:
            continue
        topic_counts[topic_group] = topic_counts.get(topic_group, 0) + 1
        if topic_counts[topic_group] > 2 and (
            item["investment_relevance_score"] < 85 or "主题上限例外" not in item["selection_reason"]
        ):
            raise NewsSelectionError("topic_group 超过主题上限。")
    return validated


def call_deepseek(system_prompt: str, user_payload: str, api_key: str, *,
                  thinking_enabled: bool = False, reasoning_effort: str | None = None) -> str:
    """Call DeepSeek and return only final content, never reasoning content."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    request = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**request)
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
            raw = invoke_model(
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=True, reasoning_effort="high"
            )
            return validate_selection(raw, candidates), None
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                sleep_fn((5, 10)[attempt])
    return [], f"⚠️ 新闻 AI 处理暂时失败；RSS 数据已获取，等待下一次更新。原因：{last_error}"
