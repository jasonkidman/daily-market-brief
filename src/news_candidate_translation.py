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
from typing import Any, Callable

from .deepseek_client import DEEPSEEK_MAX_ATTEMPTS, DeepSeekUsageTracker, call_deepseek, invoke_model
from .news_candidate_translation_prompt import SYSTEM_PROMPT


TITLE_ZH_LIMIT = 70
SUMMARY_ZH_LIMIT = 180


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


def translate_candidates(candidates: list[dict], api_key: str,
                         call_model: Callable = call_deepseek,
                         sleep_fn: Callable = time.sleep,
                         usage_tracker: DeepSeekUsageTracker | None = None,
                         max_attempts: int = DEEPSEEK_MAX_ATTEMPTS) -> dict[str, dict]:
    """Translate title/summary for candidates Stage B never looked at.

    Returns {} (never raises) on any failure -- callers must treat a missing
    candidate_id as "keep the original English", never as an error.
    """
    if not candidates or not api_key:
        return {}
    translation_input = [
        {"candidate_id": item["candidate_id"], "title": item.get("title", ""), "summary": item.get("summary", "")}
        for item in candidates
    ]
    user_payload = json.dumps({"candidates": translation_input}, ensure_ascii=False)
    started = time.monotonic()
    for attempt in range(max_attempts):
        try:
            raw = invoke_model(
                call_model, SYSTEM_PROMPT, user_payload, api_key, thinking_enabled=False, reasoning_effort=None,
                stage="Candidate Pool Translation", attempt=attempt + 1, usage_tracker=usage_tracker,
            )
            translations = validate_translations(raw, translation_input)
            print(
                f"[NEWS CANDIDATE TRANSLATION] translated {len(translations)}/{len(translation_input)} "
                f"candidates in {time.monotonic() - started:.1f}s"
            )
            return translations
        except Exception as exc:
            print(
                f"[NEWS CANDIDATE TRANSLATION] attempt {attempt + 1}/{max_attempts} failed "
                f"after {time.monotonic() - started:.1f}s: {exc}"
            )
            if usage_tracker is not None:
                usage_tracker.record_validation_failure("Candidate Pool Translation", attempt + 1, exc)
            if attempt < max_attempts - 1:
                sleep_fn((5, 10)[min(attempt, 1)])
    print("[NEWS CANDIDATE TRANSLATION] all attempts failed, falling back to English for untranslated candidates")
    return {}
