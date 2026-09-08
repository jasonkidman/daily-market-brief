"""DeepSeek transport: shared retry classification, usage tracking, and the
raw call_deepseek/invoke_model transport functions every news-AI stage uses."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


# The ten categories Scoring (news_scoring.py) assigns to every candidate;
# news_candidates.py buckets them down to six display groups.
ALLOWED_CATEGORIES = {
    "美联储 / 利率",
    "就业 / 通胀",
    "美国经济",
    "美债 / 美元",
    "金融市场",
    "大型科技",
    "AI / 资本开支",
    "半导体",
    "地缘政治",
    "政策 / 监管",
}

# Single source of truth for a call site's per-request read timeout, so
# individual callers never hardcode their own magic number. LLM_TIMEOUT is the
# default every call_deepseek() invocation gets unless it explicitly overrides
# `timeout` -- currently only Layer 2 (market_summary.py) relies on this
# default; Candidate Pool Translation gets a shorter one on purpose (see
# CANDIDATE_TRANSLATION_TIMEOUT) so a stuck translation call can't eat into the
# rest of the workflow's time, and Scoring builds its own from
# news_scoring.SCORING_TIMEOUT (a plain float, easy to tune) rather than a
# derived constant here. SDK-level retries stay at 0 (DEEPSEEK_MAX_RETRIES) --
# every stage controls its own retry/backoff/fallback behavior instead.
STAGE_TIMEOUT_SECONDS = {
    "default": 90.0,
    # Raised from 45s after real 2026-09-07 production data showed multiple
    # successful translation-batch requests taking 36-44s -- dangerously close
    # to the old ceiling, which caused ~2 of 5 real batches to time out
    # outright (see TRANSLATION_BATCH_SIZE in news_candidate_translation.py).
    "candidate_translation": 60.0,
}


def _llm_timeout(read_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=read_seconds, write=15.0, pool=5.0)


LLM_TIMEOUT = _llm_timeout(STAGE_TIMEOUT_SECONDS["default"])
CANDIDATE_TRANSLATION_TIMEOUT = _llm_timeout(STAGE_TIMEOUT_SECONDS["candidate_translation"])
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
OBSERVED_STAGES = ("Scoring", "Layer 2")


# Every news-side stage (Stage A / Stage B / Stage B Review / Candidate Pool
# Translation) sends this explicitly. terra is always a reasoning model and
# these stages used to pass reasoning_effort=None, which left the depth at the
# provider default -- the single largest contributor to both per-request
# latency and completion-token cost. Layer 2 keeps its own explicit "high":
# it is one request per run and it writes the page's top-level conclusion.
NEWS_REASONING_EFFORT = "low"


class NonRetryableAPIError(Exception):
    """A transport failure that retrying cannot fix -- bad credentials, or an
    account with no balance. Confirmed live on 2026-09-08: a 403
    INSUFFICIENT_BALANCE was handled as an ordinary exception, so every Stage A
    batch, both Stage B samples of every batch, their retries, and every
    translation batch each re-sent a request that could not possibly succeed,
    with 5-10s backoff sleeps in between.

    Raised by invoke_model and handled by each stage's own retry loop: the loop
    stops immediately (no further attempt, no backoff) and the enclosing batch
    loop stops dispatching new batches, degrading the remaining ones through the
    same deterministic, non-LLM path a failed batch already used.
    """

    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original
        self.status_code = getattr(original, "status_code", None)


_NON_RETRYABLE_STATUS = {401, 402, 403}
_NON_RETRYABLE_MARKERS = (
    "insufficient_quota",
    "INSUFFICIENT_BALANCE",
    "Insufficient account balance",
    "账户余额不足",
)


def is_non_retryable(exc: Exception) -> bool:
    """Credential/balance failures only. Timeouts, connection errors, 429 and
    5xx are all still retryable -- this must never widen to those."""
    if isinstance(exc, NonRetryableAPIError):
        return True
    if getattr(exc, "status_code", None) in _NON_RETRYABLE_STATUS:
        return True
    message = str(exc)
    return any(marker in message for marker in _NON_RETRYABLE_MARKERS)


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
        # Single choke point: every stage reaches the provider through here, so
        # classifying once means no stage can accidentally retry a credential or
        # balance failure. The usage record above is still written first -- an
        # aborted run must still show what it attempted.
        if is_non_retryable(exc):
            print(
                f"[DEEPSEEK NON-RETRYABLE] stage={_format_value(stage)} attempt={_format_value(attempt)} "
                f"http_status={_format_value(getattr(exc, 'status_code', None))} "
                f"exception_type={type(exc).__name__} -- aborting without retry"
            )
            raise NonRetryableAPIError(exc) from exc
        raise
    if usage_tracker is not None and stage is not None and attempt is not None:
        usage_tracker.record_success(
            stage=stage, attempt=attempt, model=getattr(raw, "model", DEEPSEEK_MODEL),
            thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            usage=extract_usage(getattr(raw, "usage", None)),
        )
    return raw


def call_deepseek(system_prompt: str, user_payload: str, api_key: str, *,
                  thinking_enabled: bool = False, reasoning_effort: str | None = None,
                  timeout: httpx.Timeout | None = None, max_tokens: int | None = None) -> str:
    """Call the terra reasoning model (via 灵眸's OpenAI-compatible API) and return final content.

    terra is always a reasoning model: there is no separate "thinking" toggle like
    DeepSeek's extra_body param, and it does not accept `temperature`. Reasoning depth
    is controlled solely via `reasoning_effort`; `thinking_enabled` is accepted for
    call-site compatibility but has no effect here. `timeout` defaults to the shared
    Stage A/B budget (LLM_TIMEOUT); callers with a different budget (e.g. Scoring's
    SCORING_TIMEOUT) bind it via functools.partial rather than passing a raw number
    down through invoke_model's generic call-site dispatch. `max_tokens` is left
    unset (server default) unless a caller explicitly needs a higher cap -- Scoring
    does, since a truncated JSON response fails the whole batch (see news_scoring.py).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout or LLM_TIMEOUT,
        max_retries=DEEPSEEK_MAX_RETRIES,
    )
    request = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    response = client.chat.completions.create(**request)
    return DeepSeekModelResult(
        response.choices[0].message.content,
        model=getattr(response, "model", DEEPSEEK_MODEL),
        usage=getattr(response, "usage", None),
    )


