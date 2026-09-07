"""Batched, translation-only Chinese titles/summaries for the candidate-pool drawer.

Only candidates that Stage B never selected reach this stage (selected candidates
already carry an AI-generated title_zh/summary_zh from Stage B and are reused as-is
-- see `build_news_candidates` in news_candidates.py). This never re-selects, re-ranks,
or re-scores anything, and never touches Stage A/Stage B: a failure here degrades to
the original English title/summary and never fails report generation.
"""

from __future__ import annotations

import json
import time
from functools import partial
from typing import Any, Callable

from .deepseek_client import (
    CANDIDATE_TRANSLATION_TIMEOUT, DEEPSEEK_MAX_ATTEMPTS, DeepSeekUsageTracker, call_deepseek, invoke_model,
)
from .news_candidate_translation_prompt import SYSTEM_PROMPT


TITLE_ZH_LIMIT = 70
SUMMARY_ZH_LIMIT = 180

# Small enough that one batch's failure only ever costs that batch's own
# candidates their Chinese translation, never the whole pool -- confirmed live
# on 2026-09-07, where a single failed request (of ~46 unselected candidates
# sent in one shot) fell every one of them back to English at once. Lowered
# from 10 to 8 alongside the CANDIDATE_TRANSLATION_TIMEOUT increase (see
# deepseek_client.py): a smaller per-request payload plus a longer timeout
# together, rather than either alone, is what real 2026-09-07 replay data
# showed was needed to get a real translation success rate above ~90%.
TRANSLATION_BATCH_SIZE = 8

# Shorter than Stage A/B's timeout on purpose: a stuck candidate-pool translation
# must not eat into the rest of the workflow's time budget, and a failure here
# always degrades to the original English text (see translate_candidates) rather
# than affecting the main news selection.
_call_deepseek_translation = partial(call_deepseek, timeout=CANDIDATE_TRANSLATION_TIMEOUT)


class NewsCandidateTranslationError(ValueError):
    """Raised when the translation stage output violates its contract."""


def _parse_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise NewsCandidateTranslationError("翻译输出不是 JSON 对象。")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise NewsCandidateTranslationError("翻译输出无法解析为 JSON。") from exc


def _clip(text: str, limit: int) -> str:
    """Clip to `limit` chars rather than discard -- a truncated Chinese translation
    is still far more useful in the review drawer than falling all the way back to
    a long English original just because the model ran a little over budget."""
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def validate_translations(payload: Any, candidates: list[dict]) -> dict[str, dict]:
    """Return {candidate_id: {"title_zh":..., "summary_zh":...}} for well-formed entries.

    Unlike Stage A/B's strict contracts, a malformed or missing entry is simply
    dropped rather than failing the whole batch -- callers fall back to the
    candidate's original English title/summary for anything not present here.
    Present-but-overlong text is clipped rather than dropped: some source articles
    (e.g. long Bloomberg summaries) are well over the length budget even after a
    faithful translation, and losing the whole entry to English over that is worse
    for review than a clipped Chinese translation.
    """
    data = _parse_payload(payload)
    translations = data.get("translations")
    if not isinstance(translations, list):
        raise NewsCandidateTranslationError("translations 必须是数组。")
    pool_ids = {item["candidate_id"] for item in candidates}
    result: dict[str, dict] = {}
    for item in translations:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        title_zh = str(item.get("title_zh") or "").strip()
        summary_zh = str(item.get("summary_zh") or "").strip()
        if candidate_id not in pool_ids or candidate_id in result:
            continue
        if not title_zh or not summary_zh:
            continue
        result[candidate_id] = {
            "title_zh": _clip(title_zh, TITLE_ZH_LIMIT),
            "summary_zh": _clip(summary_zh, SUMMARY_ZH_LIMIT),
        }
    return result


def _translate_batch(candidates: list[dict], api_key: str, call_model: Callable, sleep_fn: Callable,
                     usage_tracker: DeepSeekUsageTracker | None, max_attempts: int) -> tuple[dict[str, dict], str | None]:
    """Translate ONE batch, with its own independent request/validate/retry.
    Returns (translations, failure_reason) -- failure_reason is None whenever
    the batch got a usable response (even if validate_translations dropped
    some malformed individual entries from it), and a short diagnostic string
    only when every attempt for this batch failed outright.
    """
    translation_input = [
        {"candidate_id": item["candidate_id"], "title": item.get("title", ""), "summary": item.get("summary", "")}
        for item in candidates
    ]
    user_payload = json.dumps({"candidates": translation_input}, ensure_ascii=False)
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            raw = invoke_model(
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=False, reasoning_effort=None,
                stage="Candidate Pool Translation", attempt=attempt + 1, usage_tracker=usage_tracker,
            )
            translations = validate_translations(raw, translation_input)
            print(
                f"[NEWS CANDIDATE TRANSLATION] batch translated {len(translations)}/{len(translation_input)} "
                f"candidates in {time.monotonic() - started:.1f}s"
            )
            return translations, None
        except Exception as exc:
            last_error = exc
            print(
                f"[NEWS CANDIDATE TRANSLATION] batch attempt {attempt + 1}/{max_attempts} failed "
                f"after {time.monotonic() - started:.1f}s: {exc}"
            )
            if usage_tracker is not None:
                usage_tracker.record_validation_failure("Candidate Pool Translation", attempt + 1, exc)
            if attempt < max_attempts - 1:
                sleep_fn((5, 10)[min(attempt, 1)])
    reason = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown"
    print(f"[NEWS CANDIDATE TRANSLATION] batch all attempts failed, falling back to English | reason={reason}")
    return {}, reason


def translate_candidates(candidates: list[dict], api_key: str,
                         call_model: Callable = _call_deepseek_translation,
                         sleep_fn: Callable = time.sleep,
                         usage_tracker: DeepSeekUsageTracker | None = None,
                         max_attempts: int = DEEPSEEK_MAX_ATTEMPTS,
                         batch_size: int = TRANSLATION_BATCH_SIZE,
                         observability: dict | None = None) -> dict[str, dict]:
    """Translate title/summary for candidates Stage B never looked at, split
    into independent batches of at most `batch_size` so one batch's failure
    can never fall the whole pool back to English at once (see
    TRANSLATION_BATCH_SIZE). Only title/summary are translated -- this never
    re-selects, re-ranks, re-scores, or re-categorizes anything, staying
    decoupled from Stage B.

    Returns {candidate_id: {"title_zh", "summary_zh"}} merged across every
    batch; never raises. A candidate_id absent from the result means either
    its batch failed after all retries, or the model didn't return a
    well-formed entry for it -- callers must treat that the same as "keep the
    original English", never as an error. Pass `observability` to record
    translation_requested_count/translation_batch_count/translation_success_count/
    translation_failed_count and each failed batch's reason.
    """
    if observability is not None:
        observability.update({
            "translation_requested_count": len(candidates),
            "translation_batch_count": 0,
            "translation_success_count": 0,
            "translation_failed_count": 0,
            "translation_batch_failures": [],
        })
    if not candidates or not api_key:
        return {}
    batches = [candidates[start:start + batch_size] for start in range(0, len(candidates), batch_size)]
    if observability is not None:
        observability["translation_batch_count"] = len(batches)
    merged: dict[str, dict] = {}
    for index, batch in enumerate(batches, start=1):
        translations, failure_reason = _translate_batch(batch, api_key, call_model, sleep_fn, usage_tracker, max_attempts)
        merged.update(translations)
        print(
            f"[NEWS CANDIDATE TRANSLATION BATCH] batch={index}/{len(batches)} input={len(batch)} "
            f"translated={len(translations)} failure_reason={failure_reason}"
        )
        if observability is not None and failure_reason is not None:
            observability["translation_batch_failures"].append({"batch": index, "reason": failure_reason})
    if observability is not None:
        observability["translation_success_count"] = len(merged)
        observability["translation_failed_count"] = len(candidates) - len(merged)
    print(
        f"[NEWS CANDIDATE TRANSLATION] requested={len(candidates)} translated={len(merged)} "
        f"failed={len(candidates) - len(merged)} batches={len(batches)}"
    )
    return merged
