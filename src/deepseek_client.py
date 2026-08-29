"""DeepSeek transport, strict selection validation, and graceful degradation."""

from __future__ import annotations

import json
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from .news_prompt import SYSTEM_PROMPT, BORDERLINE_REVIEW_PROMPT


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
FOCUS_LIMIT = 80
SELECTION_REASON_LIMIT = 120
TAG_LIMIT = 16
LLM_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=15.0, pool=5.0)
DEEPSEEK_MAX_RETRIES = 0
DEEPSEEK_MAX_ATTEMPTS = 2
DEEPSEEK_MODEL = "gpt-5.6-terra"
DEEPSEEK_BASE_URL = "https://api.lmuai.com/v1"

# RMB per one million tokens. terra (灵眸) has no entry yet — real pricing is unknown,
# so estimate_cost_cny() falls through to None for it rather than guessing at a rate.
# Kept aliases below are for historical DeepSeek usage records already logged.
DEEPSEEK_PRICE_CNY_PER_MILLION = {
    "deepseek-chat": {"cache_hit": 0.02, "cache_miss": 1.0, "completion": 2.0},
    "deepseek-reasoner": {"cache_hit": 0.02, "cache_miss": 1.0, "completion": 2.0},
    "deepseek-v4-flash": {"cache_hit": 0.02, "cache_miss": 1.0, "completion": 2.0},
    "deepseek-v4-pro": {"cache_hit": 0.025, "cache_miss": 3.0, "completion": 6.0},
}
OBSERVED_STAGES = ("Stage A", "Stage B", "Stage B (sample A)", "Stage B (sample B)", "Stage B Review", "Layer 2")


class NewsSelectionError(ValueError):
    """Raised when model output violates the candidate-only contract."""


class _ItemValidationError(NewsSelectionError):
    def __init__(self, field: str, reason: str):
        super().__init__(reason)
        self.field = field


class DeepSeekModelResult(str):
    """String-compatible model content with response metadata for observability."""

    def __new__(cls, content: str, *, model: str | None, usage: Any):
        instance = super().__new__(cls, content or "")
        instance.model = model
        instance.usage = usage
        return instance


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def extract_usage(usage: Any) -> dict[str, int | None]:
    """Normalize OpenAI-compatible usage objects without inferring missing fields."""
    completion_details = _field(usage, "completion_tokens_details")
    prompt_details = _field(usage, "prompt_tokens_details")
    return {
        "prompt_tokens": _field(usage, "prompt_tokens"),
        "completion_tokens": _field(usage, "completion_tokens"),
        "total_tokens": _field(usage, "total_tokens"),
        "prompt_cache_hit_tokens": (
            _field(usage, "prompt_cache_hit_tokens")
            if _field(usage, "prompt_cache_hit_tokens") is not None
            else _field(prompt_details, "cached_tokens")
        ),
        "prompt_cache_miss_tokens": _field(usage, "prompt_cache_miss_tokens"),
        "reasoning_tokens": (
            _field(usage, "reasoning_tokens")
            if _field(usage, "reasoning_tokens") is not None
            else _field(completion_details, "reasoning_tokens")
        ),
    }


def _format_value(value: Any) -> str:
    return "unavailable" if value is None else str(value)


def estimate_cost_cny(model: str | None, usage: dict[str, int | None]) -> float | None:
    """Calculate only when the response provides the full cache-aware token split."""
    prices = DEEPSEEK_PRICE_CNY_PER_MILLION.get(model or "")
    required = ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens")
    if prices is None or any(usage.get(key) is None for key in required):
        return None
    return (
        usage["prompt_cache_hit_tokens"] * prices["cache_hit"]
        + usage["prompt_cache_miss_tokens"] * prices["cache_miss"]
        + usage["completion_tokens"] * prices["completion"]
    ) / 1_000_000


@dataclass
class DeepSeekUsageTracker:
    """Request-safe Actions log aggregation for one daily-report process."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record_success(self, *, stage: str, attempt: int, model: str | None,
                       thinking_enabled: bool, reasoning_effort: str | None,
                       elapsed_ms: int, usage: dict[str, int | None]) -> None:
        cost = estimate_cost_cny(model, usage)
        record = {
            "stage": stage, "attempt": attempt, "model": model,
            "thinking_enabled": thinking_enabled, "reasoning_effort": reasoning_effort,
            "elapsed_ms": elapsed_ms, "success": True, "usage": usage, "cost": cost,
        }
        self.records.append(record)
        fields = " ".join(f"{key}={_format_value(usage[key])}" for key in usage)
        print(
            f"[DEEPSEEK REQUEST] stage={stage} attempt={attempt} model={_format_value(model)} "
            f"thinking_enabled={thinking_enabled} reasoning_effort={_format_value(reasoning_effort)} "
            f"elapsed_ms={elapsed_ms} success=true {fields} "
            f"estimated_cost_cny={_format_value(f'{cost:.6f}' if cost is not None else None)}"
        )

    def record_failure(self, *, stage: str, attempt: int, model: str | None,
                       thinking_enabled: bool, reasoning_effort: str | None,
                       elapsed_ms: int, exc: Exception) -> None:
        status_code = getattr(exc, "status_code", None)
        self.records.append({"stage": stage, "attempt": attempt, "model": model, "cost": None})
        print(
            f"[DEEPSEEK REQUEST] stage={stage} attempt={attempt} model={_format_value(model)} "
            f"thinking_enabled={thinking_enabled} reasoning_effort={_format_value(reasoning_effort)} "
            f"elapsed_ms={elapsed_ms} success=false exception_type={type(exc).__name__} "
            f"http_status={_format_value(status_code)} estimated_cost_cny=unavailable"
        )

    def record_validation_failure(self, stage: str, attempt: int, exc: Exception) -> None:
        print(
            f"[DEEPSEEK VALIDATION] stage={stage} attempt={attempt} "
            f"validation_failure={type(exc).__name__}: {exc} retry=true"
        )

    def log_summary(self) -> None:
        print("[DEEPSEEK COST SUMMARY]")
        for stage in OBSERVED_STAGES:
            records = [record for record in self.records if record["stage"] == stage]
            costs = [record["cost"] for record in records]
            cost = sum(costs) if costs and all(item is not None for item in costs) else (0.0 if not costs else None)
            print(
                f"[DEEPSEEK COST SUMMARY] stage={stage} actual_api_requests={len(records)} "
                f"retry_count={max(len(records) - 1, 0)} "
                f"estimated_cost_cny={_format_value(f'{cost:.6f}' if cost is not None else None)}"
            )
        costs = [record["cost"] for record in self.records]
        total = sum(costs) if costs and all(item is not None for item in costs) else (0.0 if not costs else None)
        print(
            f"[DEEPSEEK COST SUMMARY] actual_api_requests={len(self.records)} "
            f"total_estimated_cost_cny={_format_value(f'{total:.6f}' if total is not None else None)}"
        )


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
                 thinking_enabled: bool, reasoning_effort: str | None,
                 stage: str | None = None, attempt: int | None = None,
                 usage_tracker: DeepSeekUsageTracker | None = None) -> str:
    """Call production transport with reasoning settings while supporting legacy test injectables."""
    parameters = inspect.signature(call_model).parameters.values()
    accepts_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    started = time.monotonic()
    try:
        if accepts_keywords or {"thinking_enabled", "reasoning_effort"}.issubset(inspect.signature(call_model).parameters):
            raw = call_model(
                system_prompt, user_payload, api_key,
                thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort,
            )
        else:
            raw = call_model(system_prompt, user_payload, api_key)
    except Exception as exc:
        if usage_tracker is not None and stage is not None and attempt is not None:
            usage_tracker.record_failure(
                stage=stage, attempt=attempt, model=DEEPSEEK_MODEL,
                thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort,
                elapsed_ms=round((time.monotonic() - started) * 1000), exc=exc,
            )
        raise
    if usage_tracker is not None and stage is not None and attempt is not None:
        usage_tracker.record_success(
            stage=stage, attempt=attempt, model=getattr(raw, "model", DEEPSEEK_MODEL),
            thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            usage=extract_usage(getattr(raw, "usage", None)),
        )
    return raw


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


def _validated_tags(item: dict) -> tuple[list[str], bool]:
    tags = item.get("tags")
    if not isinstance(tags, list):
        raise _ItemValidationError("tags", "tags 必须是数组")
    normalized_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        value = tag.strip()
        if not value or len(value) > TAG_LIMIT:
            continue
        if value not in normalized_tags:
            normalized_tags.append(value)
    normalized_tags = normalized_tags[:4]
    if not normalized_tags:
        raise _ItemValidationError("tags", "tags 清理后为空")
    return normalized_tags, normalized_tags != tags


def _validate_item(item: Any, pool: dict[str, dict], raw_count: int) -> tuple[dict, list[str]]:
    if not isinstance(item, dict):
        raise _ItemValidationError("item", "news 条目必须是对象")
    candidate_id = item.get("candidate_id")
    rank = item.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= raw_count:
        raise _ItemValidationError("rank", "rank 不合法")
    if candidate_id not in pool:
        raise _ItemValidationError("candidate_id", "candidate_id 不在候选池")
    if item.get("category") not in ALLOWED_CATEGORIES:
        raise _ItemValidationError("category", "category 不合法")
    try:
        title_zh = _required_text(item, "title_zh", TITLE_ZH_LIMIT)
        summary_zh = _required_text(item, "summary_zh", SUMMARY_ZH_LIMIT)
    except NewsSelectionError as exc:
        raise _ItemValidationError("text", str(exc)) from exc
    try:
        focus = _required_text(item, "focus", FOCUS_LIMIT)
        selection_reason = _required_text(item, "selection_reason", SELECTION_REASON_LIMIT)
    except NewsSelectionError as exc:
        raise _ItemValidationError("text", str(exc)) from exc
    tags, tags_normalized = _validated_tags(item)
    score = item.get("investment_relevance_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 50 <= score <= 100:
        raise _ItemValidationError("investment_relevance_score", "investment_relevance_score 不合法")
    source = pool[candidate_id]
    validated = {
        "rank": rank,
        "candidate_id": candidate_id,
        "category": item["category"],
        "title_zh": title_zh,
        "summary_zh": summary_zh,
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
    }
    normalized_fields = []
    if tags_normalized:
        normalized_fields.append("tags")
    return validated, normalized_fields


def _validate_selection_items(payload: Any, candidates: list[dict]) -> dict[str, Any]:
    data = _parse_payload(payload)
    news = data.get("news")
    if not isinstance(news, list):
        raise NewsSelectionError("news 必须是数组。")
    pool = {item["candidate_id"]: item for item in candidates}
    seen = set()
    validated = []
    issues = []
    normalized_count = 0
    for item in news:
        candidate_id = item.get("candidate_id", "<missing>") if isinstance(item, dict) else "<non-object>"
        if candidate_id in seen:
            issues.append({"candidate_id": candidate_id, "field": "candidate_id", "reason": "candidate_id 重复"})
            continue
        try:
            selected, normalized_fields = _validate_item(item, pool, len(news))
        except _ItemValidationError as exc:
            issues.append({"candidate_id": candidate_id, "field": exc.field, "reason": str(exc)})
            continue
        seen.add(candidate_id)
        validated.append(selected)
        if normalized_fields:
            normalized_count += 1
            for field in normalized_fields:
                issues.append({"candidate_id": candidate_id, "field": field, "action": "normalized"})
    validated.sort(key=lambda item: (-item["investment_relevance_score"], item["rank"]))
    validated, cap_issues = _apply_topic_cap(validated)
    issues.extend(cap_issues)
    for rank, item in enumerate(validated, start=1):
        item["rank"] = rank
    return {
        "raw_count": len(news),
        "validated": validated,
        "issues": issues,
        "normalized_count": normalized_count,
    }


def _apply_topic_cap(validated_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop items beyond the per-topic_group cap (>4), unless the item scores >=85 and
    its selection_reason explicitly claims the '主题上限例外'. Order-sensitive: call with
    items already sorted by descending priority so the cap keeps the strongest ones."""
    kept = []
    issues = []
    topic_counts: dict[str, int] = {}
    for item in validated_items:
        topic_group = item["topic_group"]
        if topic_group:
            topic_counts[topic_group] = topic_counts.get(topic_group, 0) + 1
            if topic_counts[topic_group] > 4 and (
                item["investment_relevance_score"] < 85
                or "主题上限例外" not in item["selection_reason"]
            ):
                issues.append({
                    "candidate_id": item["candidate_id"],
                    "field": "topic_group",
                    "reason": "topic_group 超过主题上限",
                })
                topic_counts[topic_group] -= 1
                continue
        kept.append(item)
    return kept, issues


def _stage_b_contract(payload: Any) -> tuple[list[dict], list[dict]]:
    data = _parse_payload(payload)
    if "selected" in data or "reserve" in data:
        selected = data.get("selected")
        reserve = data.get("reserve")
        if not isinstance(selected, list) or not isinstance(reserve, list):
            raise NewsSelectionError("selected 和 reserve 必须是数组。")
        return selected, reserve
    legacy_news = data.get("news")
    if not isinstance(legacy_news, list):
        raise NewsSelectionError("news 必须是数组。")
    return legacy_news, []


def validate_selection(payload: Any, candidates: list[dict]) -> list[dict]:
    selected, reserve = _stage_b_contract(payload)
    payload = {"news": selected}
    result = _validate_selection_items(payload, candidates)
    if result["raw_count"] != len(result["validated"]):
        issue = next((item for item in result["issues"] if "reason" in item), None)
        raise NewsSelectionError(issue["reason"] if issue else "news 条目不合法。")
    validated = result["validated"]
    if sorted(item["rank"] for item in validated) != list(range(1, len(validated) + 1)):
        raise NewsSelectionError("rank 必须从 1 连续排列。")
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
        if topic_counts[topic_group] > 4 and (
            item["investment_relevance_score"] < 85 or "主题上限例外" not in item["selection_reason"]
        ):
            raise NewsSelectionError("topic_group 超过主题上限。")
    return validated


def call_deepseek(system_prompt: str, user_payload: str, api_key: str, *,
                  thinking_enabled: bool = False, reasoning_effort: str | None = None) -> str:
    """Call the terra reasoning model (via 灵眸's OpenAI-compatible API) and return final content.

    terra is always a reasoning model: there is no separate "thinking" toggle like
    DeepSeek's extra_body param, and it does not accept `temperature`. Reasoning depth
    is controlled solely via `reasoning_effort`; `thinking_enabled` is accepted for
    call-site compatibility but has no effect here.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=DEEPSEEK_MAX_RETRIES,
    )
    request = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**request)
    return DeepSeekModelResult(
        response.choices[0].message.content,
        model=getattr(response, "model", DEEPSEEK_MODEL),
        usage=getattr(response, "usage", None),
    )


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
        items = data.get("selected") if isinstance(data, dict) and "selected" in data else data.get("news") if isinstance(data, dict) else None
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


def _log_stage_b_selection(candidates: list[dict], raw_items: list[dict], validated: list[dict], issues: list[dict]) -> None:
    raw_ids = {item.get("candidate_id") for item in raw_items}
    selected_ids = {item.get("candidate_id") for item in validated}
    issue_reasons = {
        item.get("candidate_id"): item.get("reason", "validation_drop")
        for item in issues if item.get("reason")
    }
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", "")
        if candidate_id in selected_ids:
            action, reason = "selected", "validated"
        elif candidate_id in raw_ids:
            action, reason = "not_selected", issue_reasons.get(candidate_id, "validation_drop")
        else:
            action, reason = "not_selected", "model_not_selected"
        print(
            f"[NEWS STAGE B TRACE] candidate_id={candidate_id} | title={candidate.get('title', '')} "
            f"| source={candidate.get('source', '')} | stage=stage_b | action={action} | reason={reason}"
        )


def _log_stage_b_contract_trace(raw_selected: list[dict], selected: list[dict], final: list[dict]) -> None:
    selected_ids = {item.get("candidate_id") for item in selected}
    for item in raw_selected:
        candidate_id = item.get("candidate_id", "<missing>")
        action = "pass" if candidate_id in selected_ids else "drop"
        print(f"[NEWS STAGE B EVENT] stage_b_selected candidate_id={candidate_id} action={action}")
        if action == "drop":
            print(f"[NEWS STAGE B EVENT] stage_b_selected_validation_drop candidate_id={candidate_id}")
        print(f"[NEWS STAGE B TRACE] candidate_id={candidate_id} | pool=selected | model_rank={item.get('rank', '<missing>')} | validation_action={action} | final_rank={next((x.get('rank') for x in final if x.get('candidate_id') == candidate_id), '<none>')}")
def select_news(candidates: list[dict], api_key: str, recent_selected: list[dict] = None,
                market_context: dict = None,
                call_model: Callable = call_deepseek, sleep_fn: Callable = time.sleep,
                usage_tracker: DeepSeekUsageTracker | None = None,
                observability: dict | None = None,
                stage_label: str = "Stage B",
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
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=False, reasoning_effort=None,
                stage=stage_label, attempt=attempt + 1, usage_tracker=usage_tracker,
            )
            _log_stage_b_raw_response(raw)
            contract_selected, contract_reserve = _stage_b_contract(raw)
            raw_items = [item for item in contract_selected if isinstance(item, dict)]
            raw_reserve = [item for item in contract_reserve if isinstance(item, dict)]
            target_count = len(raw_items)
            if observability is not None:
                observability["raw_count"] = len(raw_items)
                observability["stage_b_selected_count"] = len(raw_items)
                observability["stage_b_reserve_count"] = len(raw_reserve)
                observability["stage_b_target_count"] = target_count
            try:
                validation = _validate_selection_items({"news": raw_items}, candidates)
                for issue in validation["issues"]:
                    if issue.get("action") == "normalized":
                        print(
                            f"[NEWS STAGE B VALIDATION] candidate_id={issue['candidate_id']} "
                            f"field={issue['field']} action=normalized"
                        )
                    else:
                        print(
                            f"[NEWS STAGE B VALIDATION] candidate_id={issue['candidate_id']} "
                            f"field={issue['field']} action=dropped reason={issue['reason']}"
                        )
                selected = validation["validated"]
                for rank, item in enumerate(selected, start=1):
                    item["rank"] = rank
                if observability is not None:
                    observability["validated_count"] = len(selected)
                    observability["stage_b_selected_valid_count"] = len(validation["validated"])
                    observability["stage_b_reserve_validation_pass_count"] = 0
                    observability["stage_b_reserve_validation_drop_count"] = 0
                    observability["stage_b_backfilled_count"] = 0
                    observability["stage_b_final_count"] = len(selected)
                _log_stage_b_selection(candidates, raw_items, validation["validated"], validation["issues"])
                _log_stage_b_contract_trace(raw_items, validation["validated"], selected)
                print(
                    f"[NEWS STAGE B VALIDATION] raw_count={validation['raw_count']} "
                    f"valid_count={len(selected)} normalized_count={validation['normalized_count']} "
                    f"dropped_count={validation['raw_count'] - len(validation['validated'])}"
                )
                if validation["raw_count"] > 0 and not selected:
                    raise NewsSelectionError("所有新闻条目均未通过 validation。")
            except Exception as exc:
                print(f"[NEWS STAGE B] validate_selection: failed | reason={exc}")
                if usage_tracker is not None:
                    usage_tracker.record_validation_failure(stage_label, attempt + 1, exc)
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


def _review_borderline(borderline_items: list[dict], recent_selected: list[dict] | None,
                       market_context: dict | None, api_key: str, *,
                       call_model: Callable = call_deepseek,
                       usage_tracker: DeepSeekUsageTracker | None = None) -> dict[str, tuple[bool, str]]:
    """One batched LLM call asking keep/drop for every borderline candidate. Does not
    re-score, re-rank, or regenerate title/summary/selection_reason — those fields are
    reused unchanged from whichever sample originally selected the candidate.

    Returns {candidate_id: (keep, reason)} only for candidate_ids the model actually
    answered about with a valid boolean `keep`; a candidate_id outside the input set is
    ignored (illegal candidate_id), and one the model never mentions is simply absent
    from the result. Callers must treat "absent" the same as "not kept" — a missing
    verdict is not evidence the story belongs in the report.
    """
    review_candidates = [{
        "candidate_id": item["candidate_id"],
        "event_summary": item.get("event_summary", ""),
        "title_zh": item.get("title_zh"),
        "summary_zh": item.get("summary_zh"),
        "topic_group": item.get("topic_group"),
        "source": item.get("source"),
        "original_title": item.get("original_title"),
        "published_at": item.get("published_at"),
    } for item in borderline_items]
    valid_ids = {item["candidate_id"] for item in borderline_items}
    payload = {"candidates": review_candidates, "recent_7_days_events": recent_selected or []}
    if market_context:
        payload.update(market_context)
    user_payload = json.dumps(payload, ensure_ascii=False)
    raw = invoke_model(
        call_model, BORDERLINE_REVIEW_PROMPT, user_payload, api_key,
        thinking_enabled=False, reasoning_effort=None,
        stage="Stage B Review", attempt=1, usage_tracker=usage_tracker,
    )
    data = _parse_payload(raw)
    reviews = data.get("reviews")
    if not isinstance(reviews, list):
        raise NewsSelectionError("reviews 必须是数组。")
    result: dict[str, tuple[bool, str]] = {}
    for entry in reviews:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        if candidate_id not in valid_ids:
            continue
        keep = entry.get("keep")
        if not isinstance(keep, bool):
            continue
        reason = entry.get("reason") if isinstance(entry.get("reason"), str) else ""
        result[candidate_id] = (keep, reason)
    return result


def select_news_two_pass(candidates: list[dict], api_key: str, recent_selected: list[dict] = None,
                         market_context: dict = None,
                         call_model: Callable = call_deepseek,
                         review_call_model: Callable | None = None,
                         sleep_fn: Callable = time.sleep,
                         usage_tracker: DeepSeekUsageTracker | None = None,
                         observability: dict | None = None,
                         ) -> tuple[list[dict], Optional[str]]:
    """Stage B with two independent samples over the identical input, to reduce how much
    a single LLM sampling draw can swing the final news set.

    sample_A and sample_B are two ordinary select_news() calls with identical arguments,
    run concurrently — no prompt or parameter difference between them; any divergence
    comes only from the model's own sampling variance. Candidates both samples selected
    (the intersection) are kept outright. Candidates only one sample selected (the
    symmetric difference, "borderline") go through one batched review call that answers
    a single keep/drop question per candidate — it does not re-run full Stage B scoring.
    The merged result is re-sorted and passed through the existing topic_group cap
    (_apply_topic_cap) before ranks are reassigned, matching how a single-pass selection
    is finalized.

    Degrades to a single sample (or to one plain select_news call) rather than failing
    the whole report if a sample, the review call, or the orchestration itself errors —
    see the observability["two_pass_degraded"] reason recorded in each branch below.
    """
    if not candidates:
        return [], None
    if review_call_model is None:
        review_call_model = call_model
    if observability is not None:
        observability.update({
            "stage_b_sample_a_count": 0, "stage_b_sample_b_count": 0,
            "stage_b_intersection_count": 0, "stage_b_borderline_count": 0,
            "stage_b_review_keep_count": 0, "two_pass_degraded": False,
        })

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                select_news, candidates, api_key, recent_selected, market_context,
                call_model, sleep_fn, usage_tracker, None, "Stage B (sample A)",
            )
            future_b = executor.submit(
                select_news, candidates, api_key, recent_selected, market_context,
                call_model, sleep_fn, usage_tracker, None, "Stage B (sample B)",
            )
            sample_a, warning_a = future_a.result()
            sample_b, warning_b = future_b.result()
    except Exception as exc:
        print(f"[NEWS STAGE B TWO-PASS] unexpected orchestration error, falling back to single pass | reason={exc}")
        if observability is not None:
            observability["two_pass_degraded"] = "unexpected_error"
        return select_news(
            candidates, api_key, recent_selected, market_context,
            call_model, sleep_fn, usage_tracker, observability,
        )

    ok_a, ok_b = warning_a is None, warning_b is None
    print(
        f"[NEWS STAGE B TWO-PASS] sample_a_ok={ok_a} sample_a_count={len(sample_a)} "
        f"sample_b_ok={ok_b} sample_b_count={len(sample_b)}"
    )
    if observability is not None:
        observability["stage_b_sample_a_count"] = len(sample_a)
        observability["stage_b_sample_b_count"] = len(sample_b)

    if not ok_a and not ok_b:
        print("[NEWS STAGE B TWO-PASS] both samples failed, preserving existing Stage B failure behavior")
        if observability is not None:
            observability["two_pass_degraded"] = "both_samples_failed"
        return [], warning_a or warning_b

    if ok_a and not ok_b:
        print(f"[NEWS STAGE B TWO-PASS] sample_b failed after its own retries, degrading to sample_a only | reason={warning_b}")
        if observability is not None:
            observability["two_pass_degraded"] = "sample_b_failed"
            observability["raw_count"] = len(sample_a)
            observability["validated_count"] = len(sample_a)
            observability["stage_b_final_count"] = len(sample_a)
        return sample_a, None

    if ok_b and not ok_a:
        print(f"[NEWS STAGE B TWO-PASS] sample_a failed after its own retries, degrading to sample_b only | reason={warning_a}")
        if observability is not None:
            observability["two_pass_degraded"] = "sample_a_failed"
            observability["raw_count"] = len(sample_b)
            observability["validated_count"] = len(sample_b)
            observability["stage_b_final_count"] = len(sample_b)
        return sample_b, None

    ids_a = {item["candidate_id"] for item in sample_a}
    ids_b = {item["candidate_id"] for item in sample_b}
    intersection_ids = ids_a & ids_b
    borderline_ids = ids_a ^ ids_b
    intersection_items = [item for item in sample_a if item["candidate_id"] in intersection_ids]
    borderline_items = (
        [item for item in sample_a if item["candidate_id"] in borderline_ids]
        + [item for item in sample_b if item["candidate_id"] in borderline_ids]
    )
    print(
        f"[NEWS STAGE B TWO-PASS] intersection_count={len(intersection_items)} "
        f"borderline_count={len(borderline_items)} borderline_ids={sorted(borderline_ids)}"
    )
    if observability is not None:
        observability["stage_b_intersection_count"] = len(intersection_items)
        observability["stage_b_borderline_count"] = len(borderline_items)

    kept_borderline: list[dict] = []
    if borderline_items:
        try:
            review_result = _review_borderline(
                borderline_items, recent_selected, market_context, api_key,
                call_model=review_call_model, usage_tracker=usage_tracker,
            )
        except Exception as exc:
            print(f"[NEWS STAGE B TWO-PASS] borderline review failed, dropping all borderline | reason={exc}")
            review_result = {}
            if observability is not None:
                observability["two_pass_degraded"] = "borderline_review_failed"
        for item in borderline_items:
            decision = review_result.get(item["candidate_id"])
            if decision is None:
                print(f"[NEWS STAGE B TWO-PASS TRACE] candidate_id={item['candidate_id']} | review=missing | action=drop")
                continue
            keep, reason = decision
            print(f"[NEWS STAGE B TWO-PASS TRACE] candidate_id={item['candidate_id']} | review_keep={keep} | reason={reason}")
            if keep:
                kept_borderline.append(item)

    if observability is not None:
        observability["stage_b_review_keep_count"] = len(kept_borderline)

    merged = intersection_items + kept_borderline
    merged.sort(key=lambda item: (-item["investment_relevance_score"], item["candidate_id"]))
    merged, cap_issues = _apply_topic_cap(merged)
    for issue in cap_issues:
        print(f"[NEWS STAGE B TWO-PASS] dropped on merge by topic cap | candidate_id={issue['candidate_id']}")
    for rank, item in enumerate(merged, start=1):
        item["rank"] = rank

    print(f"[NEWS STAGE B TWO-PASS] final_count={len(merged)}")
    if observability is not None:
        observability["raw_count"] = len(sample_a) + len(sample_b)
        observability["validated_count"] = len(merged)
        observability["stage_b_final_count"] = len(merged)
    return merged, None
