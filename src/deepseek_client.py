"""DeepSeek transport, strict selection validation, and graceful degradation."""

from __future__ import annotations

import json
import inspect
import time
from typing import Any, Callable, Optional

import httpx

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
DEEPSEEK_TIMEOUT = httpx.Timeout(connect=5.0, read=25.0, write=15.0, pool=5.0)
DEEPSEEK_MAX_RETRIES = 0
DEEPSEEK_MAX_ATTEMPTS = 2


class NewsSelectionError(ValueError):
    """Raised when model output violates the candidate-only contract."""


def _error_kind(exc: Exception) -> str:
    """Classify transport failures for concise operational logs."""
    error_type = type(exc).__name__.lower()
    error_module = type(exc).__module__.lower()
    if "timeout" in error_type or "timeout" in error_module:
        return "timeout"
    if "connection" in error_type or "network" in error_type or "connect" in error_module:
        return "connection_error"
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return f"api_error_{status_code}"
    return "error"


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
            "event_category": source.get("event_category", "other"),
            "source_channel": source.get("source_channel"),
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

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=DEEPSEEK_TIMEOUT,
        max_retries=DEEPSEEK_MAX_RETRIES,
    )
    request = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
        "response_format": {"type": "json_object"},
        "temperature": 0.15,
        "extra_body": {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**request)
    return response.choices[0].message.content


def _log_stage_b_input(candidates: list[dict]) -> None:
    print(f"[NEWS STAGE B] input events: {len(candidates)}")
    for candidate in candidates:
        print(
            "[NEWS STAGE B] candidate_id={candidate_id} | category={category} | title={title}".format(
                candidate_id=candidate.get("candidate_id", ""),
                category=candidate.get("event_category", "other"),
                title=candidate.get("title", ""),
            )
        )


def _log_stage_b_raw_response(raw: Any) -> None:
    try:
        data = raw if isinstance(raw, dict) else json.loads(raw)
        items = data.get("news") if isinstance(data, dict) else None
    except (TypeError, json.JSONDecodeError):
        print("[NEWS STAGE B] DeepSeek raw return count: <unparseable>")
        return
    if not isinstance(items, list):
        print("[NEWS STAGE B] DeepSeek raw return count: <missing news array>")
        return
    print(f"[NEWS STAGE B] DeepSeek raw return count: {len(items)}")
    for item in items:
        if not isinstance(item, dict):
            print("[NEWS STAGE B] raw item: <non-object>")
            continue
        print(
            "[NEWS STAGE B] raw item | rank={rank} | candidate_id={candidate_id} | title={title} | "
            "importance={importance} | us_relevance={us_relevance} | novelty={novelty} | "
            "persistence={persistence} | investment_relevance_score={score}".format(
                rank=item.get("rank", "<missing>"),
                candidate_id=item.get("candidate_id", "<missing>"),
                title=item.get("title_zh", item.get("title", "<missing>")),
                importance=item.get("importance", "<missing>"),
                us_relevance=item.get("us_relevance", "<missing>"),
                novelty=item.get("novelty", "<missing>"),
                persistence=item.get("persistence", "<missing>"),
                score=item.get("investment_relevance_score", "<missing>"),
            )
        )


def select_news(candidates: list[dict], api_key: str, recent_selected: list[dict] = None,
                market_context: dict = None,
                call_model: Callable = call_deepseek, sleep_fn: Callable = time.sleep
                ) -> tuple[list[dict], Optional[str]]:
    if not candidates:
        return [], None
    event_fields = (
        "candidate_id", "event_summary", "topic_group", "event_category", "source_channel", "source", "title", "summary", "published_at",
    )
    events = [{
        **{field: candidate.get(field) for field in event_fields},
        "event_summary": candidate.get("event_summary", candidate.get("title", "")),
        "topic_group": candidate.get("topic_group", "OTHER_SYSTEMIC"),
        "event_category": candidate.get("event_category", "other"),
    } for candidate in candidates]
    _log_stage_b_input(candidates)
    payload = {"events": events, "recent_7_days_events": recent_selected or []}
    if market_context:
        payload.update(market_context)
    user_payload = json.dumps(payload, ensure_ascii=False)
    last_error = None
    started = time.monotonic()
    for attempt in range(DEEPSEEK_MAX_ATTEMPTS):
        try:
            raw = invoke_model(
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=False, reasoning_effort=None
            )
            _log_stage_b_raw_response(raw)
            try:
                selected = validate_selection(raw, candidates)
            except Exception as exc:
                print(f"[NEWS STAGE B] validate_selection: failed | reason={exc}")
                raise
            print("[NEWS STAGE B] validate_selection: passed")
            print(f"[NEWS AI] selection succeeded in {time.monotonic() - started:.1f}s")
            return selected, None
        except Exception as exc:
            last_error = exc
            print(
                f"[NEWS AI] attempt {attempt + 1}/{DEEPSEEK_MAX_ATTEMPTS} failed "
                f"after {time.monotonic() - started:.1f}s ({_error_kind(exc)}): {exc}"
            )
            if attempt < DEEPSEEK_MAX_ATTEMPTS - 1:
                sleep_fn((5, 10)[attempt])
    return [], f"⚠️ 新闻 AI 处理暂时失败；RSS 数据已获取，等待下一次更新。原因：{_error_kind(last_error)}: {last_error}"
