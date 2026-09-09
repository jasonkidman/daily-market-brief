"""Semantic near-duplicate detection among the highest-scored candidates.

Pure text-similarity (character-sequence or keyword-overlap) cannot reliably
tell that "Qualcomm Lands Amazon as Customer for Push Into AI Chips" and
"Qualcomm issues warrants to Amazon to acquire $4 billion worth of
chipmaker's stock" describe the same real-world announcement -- confirmed
against real 2026-09-09 production data, where known-duplicate title pairs
scored 0.23-0.36 on character similarity and 0.00-0.12 on keyword overlap,
no better than pairs describing genuinely different events. This needs
semantic judgment, so it is the one place after Scoring where an LLM call is
deliberately reintroduced -- scoped to only the top-ranked candidates
(DEDUP_POOL_SIZE), not the full pool, so the cost is one bounded call, not a
return to full-pool multi-stage orchestration.
"""

from __future__ import annotations

import json
from functools import partial
from typing import Callable

from .deepseek_client import (
    DeepSeekUsageTracker, NonRetryableAPIError, _llm_timeout, call_deepseek, invoke_model,
)


DEDUP_POOL_SIZE = 25
DEDUP_TIMEOUT = 90.0
DEDUP_MAX_TOKENS = 4000
DEDUP_REASONING_EFFORT = "low"

_call_deepseek_dedup = partial(call_deepseek, timeout=_llm_timeout(DEDUP_TIMEOUT), max_tokens=DEDUP_MAX_TOKENS)

SYSTEM_PROMPT = """你会收到一批已经过重要性打分、按分数排序的新闻候选（同一天）。

任务：找出其中属于同一条新闻主线的候选——同一场冲突、同一个正在演变的市场
事态，即使具体触发细节、报道角度、用词完全不同，也算同一条主线。例如：
- "高通拿下亚马逊代工订单"和"高通向亚马逊发认股权证换股"是同一笔交易的
  两个角度，属于同一条主线。
- "美国袭击伊朗油轮后油价逼近100美元"、"美伊互袭令布伦特原油逼近100美元"、
  "伊朗据报再次袭击美国海军舰艇，油价上涨"、"沙特称将回应胡塞袭击城市和能源
  设施"，虽然具体触发点、涉及方不同，但都是同一场中东冲突升级、推高油价这
  一持续事态下的连续报道，属于同一条主线。

每条主线只保留其中信息最完整、最新、最有价值的 1 条作为代表（该条
duplicate_of 填 null）；主线内其余候选的 duplicate_of 填代表的
candidate_id。不属于任何主线、独立存在的候选，duplicate_of 也填 null。

判断标准是"是否在报道同一件正在发生/演变的事情"，而不是"是否属于同一大类
新闻"。例如"美联储议息决定"和"最新非农就业数据"都属于宏观/利率大类，但
报道的是两件独立的事情，不应合并。输入有多少条，输出就必须有多少条。

严格 JSON 输出，无 markdown 代码块，格式：
{"items": [{"candidate_id": "...", "duplicate_of": null}]}
"""


def _validate_item(item, valid_ids: set[str]) -> tuple[str, str | None] | None:
    if not isinstance(item, dict):
        return None
    candidate_id = item.get("candidate_id")
    if candidate_id not in valid_ids:
        return None
    duplicate_of = item.get("duplicate_of")
    if duplicate_of is not None and (
        not isinstance(duplicate_of, str) or duplicate_of not in valid_ids or duplicate_of == candidate_id
    ):
        duplicate_of = None
    return candidate_id, duplicate_of


def dedupe_top_candidates(
    ranked_candidates: list[dict], api_key: str, *,
    call_model: Callable = _call_deepseek_dedup,
    usage_tracker: DeepSeekUsageTracker | None = None,
    pool_size: int = DEDUP_POOL_SIZE,
) -> dict[str, str | None]:
    """Return {candidate_id: duplicate_of} for the top `pool_size` ranked
    candidates (assumed pre-sorted best-first, e.g. by score descending).

    A candidate_id absent from the return value -- outside the pool, or this
    call failed or was unparseable entirely -- must be treated by the caller
    as not-a-duplicate. This stage degrades to a no-op on any failure: it
    never raises, never blocks report generation, and never removes a
    candidate from scoring output or the "更多新闻" pool -- it only affects
    who gets a "今日重要新闻" slot.
    """
    pool = ranked_candidates[:pool_size]
    if len(pool) < 2:
        return {}
    valid_ids = {item["candidate_id"] for item in pool}
    fields = ("candidate_id", "title", "summary", "source", "published_at")
    user_payload = json.dumps(
        {"candidates": [{field: item.get(field, "") for field in fields} for item in pool]},
        ensure_ascii=False,
    )
    try:
        raw = invoke_model(
            call_model, SYSTEM_PROMPT, user_payload, api_key,
            thinking_enabled=False, reasoning_effort=DEDUP_REASONING_EFFORT,
            stage="TopDedup", attempt=1, usage_tracker=usage_tracker,
        )
    except NonRetryableAPIError as exc:
        print(f"[TOP DEDUP] aborted on non-retryable API error, skipping dedup: {exc}")
        return {}
    except Exception as exc:
        print(f"[TOP DEDUP] failed, skipping dedup: {exc}")
        return {}

    try:
        payload = json.loads(raw) if not isinstance(raw, dict) else raw
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("items 不是数组。")
    except Exception as exc:
        print(f"[TOP DEDUP] unparseable output, skipping dedup: {exc}")
        return {}

    result: dict[str, str | None] = {}
    for raw_item in raw_items:
        validated = _validate_item(raw_item, valid_ids)
        if validated is None:
            continue
        candidate_id, duplicate_of = validated
        result[candidate_id] = duplicate_of

    # A representative must itself have duplicate_of=None. Guard against a
    # chain or cycle (A -> B -> C, or A <-> B) by only trusting a one-hop
    # pointer to an actual representative: if duplicate_of points at another
    # entry that is *also* marked as a duplicate, treat this one as
    # not-a-duplicate instead of trying to re-chain it -- worst case is one
    # extra near-duplicate shows up in the top list, not a legitimate story
    # silently disappearing.
    for candidate_id, duplicate_of in list(result.items()):
        if duplicate_of is not None and result.get(duplicate_of) is not None:
            result[candidate_id] = None

    dropped = sum(1 for value in result.values() if value is not None)
    print(f"[TOP DEDUP] pool={len(pool)} marked_duplicate={dropped}")
    return result
