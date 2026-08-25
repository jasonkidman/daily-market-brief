"""DeepSeek transport, strict selection validation, and graceful degradation."""

from __future__ import annotations

import json
import inspect
import time
from dataclasses import dataclass, field
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
DEEPSEEK_MODEL = "deepseek-chat"

# RMB per one million tokens.  Keep aliases because deployed workflows may use
# either the compatibility names or the current V4 model names.
DEEPSEEK_PRICE_CNY_PER_MILLION = {
    "deepseek-chat": {"cache_hit": 0.02, "cache_miss": 1.0, "completion": 2.0},
    "deepseek-reasoner": {"cache_hit": 0.02, "cache_miss": 1.0, "completion": 2.0},
    "deepseek-v4-flash": {"cache_hit": 0.02, "cache_miss": 1.0, "completion": 2.0},
    "deepseek-v4-pro": {"cache_hit": 0.025, "cache_miss": 3.0, "completion": 6.0},
}
OBSERVED_STAGES = ("Stage A", "Stage B", "Layer 2")


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


def _normalize_investment_impact(item: dict) -> tuple[str, bool]:
    raw_value = item.get("investment_impact")
    if not isinstance(raw_value, str):
        raise _ItemValidationError("investment_impact", "investment_impact 类型不合法")
    value = raw_value.strip()
    normalized = value
    for arrow in ("->", "=>", "➡", "⟶"):
        normalized = normalized.replace(arrow, "→")
    normalized = normalized.strip()
    if not normalized or len(normalized) > INVESTMENT_IMPACT_LIMIT:
        raise _ItemValidationError("investment_impact", "investment_impact 超出长度或为空")
    if not _has_impact_path(normalized):
        raise _ItemValidationError("investment_impact", "缺少资产变量及因果/条件传导路径")
    return normalized, normalized != value


def _has_impact_path(value: str) -> bool:
    if "短期资产价格影响有限，暂以观察为主" in value:
        return True
    variables = (
        "SPY", "Nasdaq", "纳指", "科技股", "美债", "收益率", "美元", "信用", "AI", "半导体",
        "估值", "通胀", "油价", "利率", "金融条件", "就业", "盈利", "流动性", "风险偏好",
    )
    connectors = ("→", "若", "如果", "导致", "使得", "进而", "从而", "有利于", "压制")
    return any(term in value for term in variables) and any(term in value for term in connectors)


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
    investment_impact, impact_normalized = _normalize_investment_impact(item)
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
    }
    normalized_fields = []
    if tags_normalized:
        normalized_fields.append("tags")
    if impact_normalized:
        normalized_fields.append("investment_impact")
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
    validated.sort(key=lambda item: item["rank"])
    batch_validated = []
    topic_counts: dict[str, int] = {}
    for item in validated:
        if batch_validated and item["investment_relevance_score"] > batch_validated[-1]["investment_relevance_score"]:
            issues.append({
                "candidate_id": item["candidate_id"],
                "field": "investment_relevance_score",
                "reason": "分数未按 rank 非递增",
            })
            continue
        topic_group = item["topic_group"]
        if topic_group:
            topic_counts[topic_group] = topic_counts.get(topic_group, 0) + 1
            if topic_counts[topic_group] > 2 and (
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
        batch_validated.append(item)
    validated = batch_validated
    for rank, item in enumerate(validated, start=1):
        item["rank"] = rank
    return {
        "raw_count": len(news),
        "validated": validated,
        "issues": issues,
        "normalized_count": normalized_count,
    }


def validate_selection(payload: Any, candidates: list[dict]) -> list[dict]:
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
                call_model: Callable = call_deepseek, sleep_fn: Callable = time.sleep,
                usage_tracker: DeepSeekUsageTracker | None = None
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
                stage="Stage B", attempt=attempt + 1, usage_tracker=usage_tracker,
            )
            _log_stage_b_raw_response(raw)
            try:
                validation = _validate_selection_items(raw, candidates)
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
                print(
                    f"[NEWS STAGE B VALIDATION] raw_count={validation['raw_count']} "
                    f"valid_count={len(selected)} normalized_count={validation['normalized_count']} "
                    f"dropped_count={validation['raw_count'] - len(selected)}"
                )
                if validation["raw_count"] > 0 and not selected:
                    raise NewsSelectionError("所有新闻条目均未通过 validation。")
            except Exception as exc:
                print(f"[NEWS STAGE B] validate_selection: failed | reason={exc}")
                if usage_tracker is not None:
                    usage_tracker.record_validation_failure("Stage B", attempt + 1, exc)
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
