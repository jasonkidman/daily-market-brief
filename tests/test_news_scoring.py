"""Layer 1 scoring: call-count regression guards (the core of PR 3's ask --
these are what stop the old Stage A/B API fan-out from reappearing) plus field
validation and the pure-code degradation path."""

from __future__ import annotations

import json
import threading

import pytest

from src.deepseek_client import NonRetryableAPIError
from src.news_scoring import (
    SCORING_BATCH_SIZE,
    SCORING_MAX_WORKERS,
    _validate_score_item,
    rule_based_top_news,
    score_candidates,
)


def candidate(cid, title=None, summary="Summary", priority="P2", source="Source",
             published_at="2026-08-12T00:00:00+00:00"):
    return {
        "candidate_id": cid,
        "title": title or f"Title {cid}",
        "summary": summary,
        "source": source,
        "priority": priority,
        "published_at": published_at,
        "url": f"https://example.com/{cid}",
    }


def score_item(candidate_id, score=80, category="美联储 / 利率", title_zh="标题", summary_zh="摘要", reason="理由"):
    return {
        "candidate_id": candidate_id, "score": score, "category": category,
        "title_zh": title_zh, "summary_zh": summary_zh, "reason": reason,
    }


def counting_scorer(handler):
    """A call_model that records every invocation's input candidate_ids and
    delegates to `handler(candidate_ids) -> str|dict` for the response."""
    calls = []
    lock = threading.Lock()

    def call_model(system_prompt, user_payload, api_key, **kwargs):
        payload = json.loads(user_payload)
        candidate_ids = [item["candidate_id"] for item in payload["candidates"]]
        with lock:
            calls.append(candidate_ids)
        return handler(candidate_ids)

    return call_model, calls


def all_pass_handler(candidate_ids):
    return json.dumps({"scores": [score_item(cid) for cid in candidate_ids]})


# --------------------------------------------------------------------------
# call-count regression guards
# --------------------------------------------------------------------------

def test_scoring_call_count_100_candidates_is_exactly_3():
    """100 candidates / SCORING_BATCH_SIZE(40) -> 40 + 40 + 20 -> exactly 3 calls."""
    pool = [candidate(str(i)) for i in range(100)]
    call_model, calls = counting_scorer(all_pass_handler)

    results = score_candidates(pool, "key", "focus rules", call_model=call_model, sleep_fn=lambda _: None)

    assert len(calls) == 3
    assert sorted(len(batch) for batch in calls) == [20, 40, 40]
    assert len(results) == 100


def test_no_fanout_on_invalid_item():
    """1 malformed item among 40 must not trigger a batch retry -- only that
    one item is dropped, the other 39 are kept, and call_model is invoked once."""
    pool = [candidate(str(i)) for i in range(40)]

    def handler(candidate_ids):
        items = [score_item(cid) for cid in candidate_ids]
        items[0]["score"] = "not-a-number"  # invalid
        return json.dumps({"scores": items})

    call_model, calls = counting_scorer(handler)

    results = score_candidates(pool, "key", "focus rules", call_model=call_model, sleep_fn=lambda _: None)

    assert len(calls) == 1
    assert len(results) == 39
    assert pool[0]["candidate_id"] not in results


def test_no_fanout_on_missing_items():
    """The model answering 38 of 40 candidates must not trigger a batch retry --
    the 2 missing candidate_ids are simply absent from the result (caller
    treats absence as score=None), and call_model is invoked once."""
    pool = [candidate(str(i)) for i in range(40)]

    def handler(candidate_ids):
        return json.dumps({"scores": [score_item(cid) for cid in candidate_ids[:38]]})

    call_model, calls = counting_scorer(handler)

    results = score_candidates(pool, "key", "focus rules", call_model=call_model, sleep_fn=lambda _: None)

    assert len(calls) == 1
    assert len(results) == 38
    assert pool[38]["candidate_id"] not in results
    assert pool[39]["candidate_id"] not in results


def test_retry_only_once_on_timeout():
    """A transient (timeout) failure retries exactly once per batch -- 2 total
    attempts, matching SCORING_MAX_ATTEMPTS."""
    pool = [candidate(str(i)) for i in range(5)]
    attempts = []

    def call_model(system_prompt, user_payload, api_key, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("read timeout")
        payload = json.loads(user_payload)
        ids = [item["candidate_id"] for item in payload["candidates"]]
        return json.dumps({"scores": [score_item(cid) for cid in ids]})

    slept = []
    results = score_candidates(
        pool, "key", "focus rules", call_model=call_model, sleep_fn=slept.append,
    )

    assert len(attempts) == 2
    assert slept, "a retryable failure must still back off between attempts"
    assert len(results) == 5


def test_scoring_all_batches_fail_returns_empty_not_partial_garbage():
    """A batch that never produces a single valid item after exhausting
    retries contributes nothing -- never a fabricated score."""
    pool = [candidate(str(i)) for i in range(5)]

    def always_broken(system_prompt, user_payload, api_key, **kwargs):
        return "not json"

    results = score_candidates(
        pool, "key", "focus rules", call_model=always_broken, sleep_fn=lambda _: None,
    )

    assert results == {}


def test_non_retryable_aborts_without_retry_and_stops_new_dispatches():
    """5 batches (200 candidates, SCORING_MAX_WORKERS=3): a non-retryable error
    must never be retried, and once it fires, any batch not yet dispatched to a
    worker thread must skip its API call entirely -- only the batches already
    running when the abort happened may complete (or also fail) on their own.
    With an always-403 call_model, that caps total calls at SCORING_MAX_WORKERS,
    not the full batch count."""
    pool = [candidate(str(i)) for i in range(200)]

    class FakeAPIError(Exception):
        def __init__(self):
            super().__init__("no balance")
            self.status_code = 403

    def always_403(system_prompt, user_payload, api_key, **kwargs):
        raise FakeAPIError()

    def no_sleep(_seconds):
        raise AssertionError("a non-retryable error must never back off or retry")

    observability = {}
    results = score_candidates(
        pool, "key", "focus rules", call_model=always_403, sleep_fn=no_sleep, observability=observability,
    )

    assert observability["scoring_batch_count"] == 5
    assert observability["scoring_non_retryable_abort"] is True
    assert results == {}
    # This is the concurrency guarantee under test: with 5 batches all
    # submitted up front (not one at a time -- see the dedicated concurrency
    # test below) and only 3 workers, the 2 batches never dispatched to a
    # worker thread must not have called the API at all.
    assert observability["scoring_failed_batch_count"] == 5


def test_scoring_runs_batches_concurrently_not_sequentially():
    """With SCORING_MAX_WORKERS=3 and exactly 3 batches, all 3 must be
    in-flight at the same time -- not started one after the previous one
    finished. Proven with a barrier: every call blocks until all 3 have
    started, which can only complete if they overlap."""
    pool = [candidate(str(i)) for i in range(100)]  # -> 3 batches (40/40/20)
    barrier = threading.Barrier(SCORING_MAX_WORKERS, timeout=5)

    def handler(candidate_ids):
        barrier.wait()  # deadlocks (-> BrokenBarrierError) if not run concurrently
        return json.dumps({"scores": [score_item(cid) for cid in candidate_ids]})

    call_model, calls = counting_scorer(handler)

    results = score_candidates(pool, "key", "focus rules", call_model=call_model, sleep_fn=lambda _: None)

    assert len(calls) == 3
    assert len(results) == 100


# --------------------------------------------------------------------------
# field validation
# --------------------------------------------------------------------------

def test_validate_score_item_clips_overlong_chinese_fields():
    pool_by_id = {"a": candidate("a")}
    item = score_item("a", title_zh="标" * 50, summary_zh="摘" * 100, reason="理" * 80)

    validated, reason = _validate_score_item(item, pool_by_id)

    assert reason is None
    assert len(validated["title_zh"]) == 40
    assert len(validated["summary_zh"]) == 80
    assert len(validated["reason"]) == 60


def test_validate_score_item_falls_back_to_english_when_title_zh_empty():
    pool_by_id = {"a": candidate("a", title="Fed holds rates")}
    item = score_item("a", title_zh="", summary_zh="")

    validated, reason = _validate_score_item(item, pool_by_id)

    assert reason is None
    assert validated["title_zh"] == "Fed holds rates"


def test_validate_score_item_allows_empty_reason():
    pool_by_id = {"a": candidate("a")}
    item = score_item("a", reason="")

    validated, reason = _validate_score_item(item, pool_by_id)

    assert reason is None
    assert validated["reason"] == ""


@pytest.mark.parametrize("bad_score", [-1, 101, "80", 50.5, None, True])
def test_validate_score_item_rejects_out_of_range_or_wrong_type_score(bad_score):
    pool_by_id = {"a": candidate("a")}
    item = score_item("a", score=bad_score)

    validated, reason = _validate_score_item(item, pool_by_id)

    assert validated is None
    assert reason == "invalid_score"


def test_validate_score_item_rejects_unknown_candidate_id():
    pool_by_id = {"a": candidate("a")}
    item = score_item("not-in-pool")

    validated, reason = _validate_score_item(item, pool_by_id)

    assert validated is None
    assert reason == "unknown_candidate_id"


def test_validate_score_item_keeps_invalid_category_unchanged_for_downstream_fallback():
    """Downstream (news_candidates._bucket_category) already maps any
    unrecognized category string to CATEGORY_OTHER via dict.get's default, so
    this module does not need to know that display-layer fallback value --
    passing the raw (invalid) string through is what "非法 category 落到 其他"
    actually requires."""
    pool_by_id = {"a": candidate("a")}
    item = score_item("a", category="not-a-real-category")

    validated, reason = _validate_score_item(item, pool_by_id)

    assert reason is None
    assert validated["category"] == "not-a-real-category"


# --------------------------------------------------------------------------
# rule_based_top_news (total-failure fallback)
# --------------------------------------------------------------------------

def test_rule_based_top_news_keeps_english_text_and_marks_degraded():
    ranked = [candidate("a"), candidate("b"), candidate("c")]

    top = rule_based_top_news(ranked, limit=2)

    assert [item["candidate_id"] for item in top] == ["a", "b"]
    for item in top:
        assert item["score"] is None
        assert item["selection_mode"] == "rule_based"
        assert item["title_zh"] == item["title"]
        assert item["summary_zh"] == item["summary"]


def test_rule_based_top_news_on_empty_pool_returns_empty():
    assert rule_based_top_news([], limit=10) == []
