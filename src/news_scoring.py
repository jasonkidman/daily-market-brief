"""Layer 1 scoring: one absolute-scale score per candidate, no filtering.

Replaces the old Stage A (LLM clustering) + Stage B (LLM selection) pipeline's
selection step. Every candidate that reaches this module gets scored
independently -- the model is never asked to choose a subset, only to rate
importance on a fixed 0-100 scale (see news_score_prompt.py). Selecting the
final "today's news" set from these scores is main.py's job (pure code: sort
by score, take the top N -- see IMPLEMENTATION_PLAN.md section 1/section 5.8).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event
from typing import Callable, Optional

from .deepseek_client import (
    ALLOWED_CATEGORIES, DeepSeekUsageTracker, NEWS_REASONING_EFFORT, NonRetryableAPIError,
    _llm_timeout, call_deepseek, invoke_model,
)
from .news_events import _rank_stage_a_candidates
from .news_score_prompt import build_system_prompt


SCORING_BATCH_SIZE = 40
SCORING_MAX_WORKERS = 3
SCORING_MAX_ATTEMPTS = 2
SCORING_TIMEOUT = 240.0
# A truncated JSON response fails the whole batch's parse, so this must be
# generous: A1's per-item output (title_zh + summary_zh + reason, ~280 tokens)
# times a 40-item batch is roughly 11k output tokens. See IMPLEMENTATION_PLAN.md
# section 4.2 -- do not rely on the SDK/server default here.
SCORING_MAX_TOKENS = 16000

TITLE_ZH_LIMIT = 40
SUMMARY_ZH_LIMIT = 80
REASON_LIMIT = 60

_call_deepseek_scoring = partial(
    call_deepseek, timeout=_llm_timeout(SCORING_TIMEOUT), max_tokens=SCORING_MAX_TOKENS,
)


def _batch_candidates(candidates: list[dict], batch_size: int) -> list[list[dict]]:
    """Split the full candidate pool into batches, ranked highest-value-first
    (_rank_stage_a_candidates) so the first, always-run batch carries the most
    important candidates if later batches are ever skipped (non-retryable abort)."""
    if not candidates:
        return []
    ranked = [item for item, *_ in _rank_stage_a_candidates(candidates)]
    return [ranked[start:start + batch_size] for start in range(0, len(ranked), batch_size)]


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text[:limit] if len(text) > limit else text


def _validate_score_item(item, pool_by_id: dict[str, dict]) -> tuple[Optional[dict], Optional[str]]:
    """One candidate's raw scoring output -> a validated record, or (None, reason).

    Never raises: an invalid item is the caller's signal to drop just this one
    entry, not to retry the batch (see IMPLEMENTATION_PLAN.md section 4.3/5.7).
    `category` is intentionally NOT normalized to a fixed fallback value here --
    an unrecognized string already buckets to CATEGORY_OTHER downstream via
    news_candidates._bucket_category's dict.get(..., CATEGORY_OTHER), so passing
    it through unchanged already satisfies "非法 category 落到 其他展示分组"
    without this module needing to know that display-layer constant.
    """
    if not isinstance(item, dict):
        return None, "not_a_dict"
    candidate_id = item.get("candidate_id")
    if candidate_id not in pool_by_id:
        return None, "unknown_candidate_id"
    score = item.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not (0 <= score <= 100):
        return None, "invalid_score"
    candidate = pool_by_id[candidate_id]
    category = item.get("category") if isinstance(item.get("category"), str) else ""
    title_zh = _clip(item["title_zh"], TITLE_ZH_LIMIT) if isinstance(item.get("title_zh"), str) else ""
    title_zh = title_zh or candidate.get("title", "")
    summary_zh = _clip(item["summary_zh"], SUMMARY_ZH_LIMIT) if isinstance(item.get("summary_zh"), str) else ""
    summary_zh = summary_zh or candidate.get("summary", "")
    reason = _clip(item["reason"], REASON_LIMIT) if isinstance(item.get("reason"), str) else ""
    return {
        "candidate_id": candidate_id,
        "score": score,
        "category": category,
        "title_zh": title_zh,
        "summary_zh": summary_zh,
        "reason": reason,
    }, None


def _score_batch(batch: list[dict], api_key: str, system_prompt: str, *,
                 call_model: Callable, sleep_fn: Callable,
                 usage_tracker: DeepSeekUsageTracker | None, batch_label: str,
                 ) -> tuple[dict[str, dict], bool]:
    """Run one batch through Scoring, retrying only when the whole batch
    produced zero valid results (transport failure, unparseable output, or an
    empty/all-invalid `scores` array) -- never because of a single bad item.
    `NonRetryableAPIError` is never retried and propagates to the caller.

    Returns (results_by_candidate_id, ai_failed). `ai_failed=True` means every
    attempt was exhausted with zero usable output -- the caller treats those
    candidates as unscored (score=None), not as an empty/zero score.
    """
    pool_by_id = {item["candidate_id"]: item for item in batch}
    fields = ("candidate_id", "title", "summary", "source", "published_at")
    user_payload = json.dumps(
        {"candidates": [{field: item.get(field, "") for field in fields} for item in batch]},
        ensure_ascii=False,
    )
    last_error: Exception | None = None
    for attempt in range(1, SCORING_MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            raw = invoke_model(
                call_model, system_prompt, user_payload, api_key,
                thinking_enabled=False, reasoning_effort=NEWS_REASONING_EFFORT,
                stage="Scoring", attempt=attempt, usage_tracker=usage_tracker,
            )
        except NonRetryableAPIError:
            raise
        except Exception as exc:
            last_error = exc
            print(f"[SCORING BATCH] {batch_label} attempt={attempt}/{SCORING_MAX_ATTEMPTS} transport failed: {exc}")
            if attempt < SCORING_MAX_ATTEMPTS:
                sleep_fn((5, 10)[attempt - 1])
            continue

        try:
            payload = json.loads(raw) if not isinstance(raw, dict) else raw
            raw_scores = payload.get("scores")
            if not isinstance(raw_scores, list):
                raise ValueError("scores 不是数组。")
        except Exception as exc:
            last_error = exc
            print(f"[SCORING BATCH] {batch_label} attempt={attempt}/{SCORING_MAX_ATTEMPTS} unparseable output: {exc}")
            if attempt < SCORING_MAX_ATTEMPTS:
                sleep_fn((5, 10)[attempt - 1])
            continue

        results: dict[str, dict] = {}
        invalid_count = 0
        for raw_item in raw_scores:
            validated, reason = _validate_score_item(raw_item, pool_by_id)
            if validated is None:
                invalid_count += 1
                print(f"[SCORING BATCH] {batch_label} dropped invalid item | reason={reason}")
                continue
            results[validated["candidate_id"]] = validated

        if not results:
            last_error = ValueError("整批 0 条有效评分。")
            print(f"[SCORING BATCH] {batch_label} attempt={attempt}/{SCORING_MAX_ATTEMPTS} zero valid items")
            if attempt < SCORING_MAX_ATTEMPTS:
                sleep_fn((5, 10)[attempt - 1])
            continue

        missing = [item["candidate_id"] for item in batch if item["candidate_id"] not in results]
        scores = sorted(item["score"] for item in results.values())
        elapsed = time.monotonic() - started
        print(
            f"[SCORING BATCH] {batch_label} input={len(batch)} scored={len(results)} "
            f"invalid={invalid_count} missing={len(missing)} "
            f"score_min={scores[0]} score_max={scores[-1]} elapsed_s={elapsed:.1f}"
        )
        return results, False

    print(f"[SCORING BATCH] {batch_label} exhausted {SCORING_MAX_ATTEMPTS} attempts, zero usable output | last_error={last_error}")
    return {}, True


def score_candidates(
    candidates: list[dict], api_key: str, focus_rules: str, *,
    call_model: Callable = _call_deepseek_scoring,
    sleep_fn: Callable = time.sleep,
    usage_tracker: DeepSeekUsageTracker | None = None,
    observability: dict | None = None,
) -> dict[str, dict]:
    """Score every candidate. Returns {candidate_id: {score, category, title_zh,
    summary_zh, reason}} -- only for candidates that got a usable result.
    A candidate_id absent from the return value was either dropped as invalid
    by its batch, or belonged to a batch that failed entirely or was skipped
    after a non-retryable abort; the caller treats all of those as score=None,
    never retries them, and never re-dispatches a skipped batch.

    Batches run CONCURRENTLY (up to SCORING_MAX_WORKERS at once), all submitted
    up front -- not one after another. A NonRetryableAPIError from any batch
    sets a shared abort flag: batches already running are allowed to finish (or
    fail) on their own, but any batch not yet started skips its API call
    entirely and contributes nothing (never a fabricated score).
    """
    if observability is not None:
        observability.update({
            "scoring_total_candidate_count": len(candidates),
            "scoring_batch_count": 0, "scoring_batch_sizes": [],
            "scoring_non_retryable_abort": False, "scoring_failed_batch_count": 0,
        })
    if not candidates:
        return {}

    batches = _batch_candidates(candidates, SCORING_BATCH_SIZE)
    if observability is not None:
        observability["scoring_batch_count"] = len(batches)
        observability["scoring_batch_sizes"] = [len(batch) for batch in batches]

    system_prompt = build_system_prompt(focus_rules, sorted(ALLOWED_CATEGORIES))
    aborted = Event()
    failed_batch_count = 0

    def run_batch(index: int, batch: list[dict]) -> dict[str, dict]:
        nonlocal failed_batch_count
        label = f"batch={index}/{len(batches)}"
        if aborted.is_set():
            print(f"[SCORING BATCH] {label} input={len(batch)} skipped=non_retryable_abort")
            failed_batch_count += 1
            return {}
        try:
            results, ai_failed = _score_batch(
                batch, api_key, system_prompt,
                call_model=call_model, sleep_fn=sleep_fn,
                usage_tracker=usage_tracker, batch_label=label,
            )
        except NonRetryableAPIError as exc:
            aborted.set()
            if observability is not None:
                observability["scoring_non_retryable_abort"] = True
            print(f"[SCORING BATCH] {label} aborted=non_retryable_api_error | {exc}")
            failed_batch_count += 1
            return {}
        if ai_failed:
            failed_batch_count += 1
        return results

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=SCORING_MAX_WORKERS) as executor:
        futures = [executor.submit(run_batch, index, batch) for index, batch in enumerate(batches, start=1)]
        for future in futures:
            results.update(future.result())

    if observability is not None:
        observability["scoring_failed_batch_count"] = failed_batch_count
        observability["scoring_scored_count"] = len(results)

    print(
        f"[SCORING] total_candidates={len(candidates)} batches={len(batches)} "
        f"scored={len(results)} failed_batches={failed_batch_count} "
        f"non_retryable_abort={aborted.is_set()}"
    )
    return results


def rule_based_top_news(ranked_candidates: list[dict], limit: int) -> list[dict]:
    """Zero-API fallback for when every scoring batch fails. `ranked_candidates`
    is expected pre-sorted best-first (e.g. by _rank_stage_a_candidates'
    composite score). Keeps the original English title/summary -- there is no
    AI translation to fall back on -- and marks selection_mode='rule_based' so
    the caller can flag the report as degraded. Deliberately simple: no topic
    diversity cap, no eligibility rules beyond the pre-existing rank order --
    the old, much more elaborate deterministic fallback is not being revived.
    """
    top = ranked_candidates[:limit]
    return [{
        **item,
        "rank": rank,
        "score": None,
        "category": None,
        "title_zh": item.get("title", ""),
        "summary_zh": item.get("summary", ""),
        "reason": "",
        "selection_mode": "rule_based",
    } for rank, item in enumerate(top, start=1)]
