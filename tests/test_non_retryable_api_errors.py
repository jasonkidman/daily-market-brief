"""PR 1 regression guards: a credential/balance failure must cost exactly one
request per already-dispatched call site, and must never fan out into retries,
single-pass fallbacks, or further batches.

Confirmed live on 2026-09-08 (run #88): a 403 INSUFFICIENT_BALANCE was handled
as an ordinary exception, so every Stage A batch, both Stage B samples of every
batch, their retries and every translation batch each re-sent a request that
could not possibly succeed, with 5-10s backoff sleeps in between.
"""

import pytest

from src import deepseek_client, market_summary, news_candidate_translation, news_events
from src.deepseek_client import NonRetryableAPIError, is_non_retryable


class FakeAPIError(Exception):
    """Shaped like the openai SDK's error objects: carries .status_code."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


# Deliberately unrelated wording: the cross-batch duplicate merge is a real
# text-similarity check and would otherwise collapse a synthetic pool of
# near-identical titles into a single event.
_TITLE_STEMS = (
    "Fed holds policy rates steady",
    "Nvidia unveils new datacenter accelerator",
    "Oil slips on demand worries",
    "Treasury yields climb after auction",
    "Apple faces EU antitrust probe",
    "Payrolls beat expectations sharply",
    "Tesla cuts prices across Europe",
    "Bank regulators propose capital rules",
)


def candidate(cid, title=None, priority="P0"):
    stem = _TITLE_STEMS[int(cid) % len(_TITLE_STEMS)] if str(cid).isdigit() else "Fed holds policy rates steady"
    title = title or f"{stem} ({cid})"
    return {
        "candidate_id": cid, "title": title, "url": f"https://x/{cid}",
        "source": "BBC News", "summary": "full summary",
        "published_at": "2026-08-12T00:00:00+00:00", "priority": priority,
    }


def counting_raiser(exc_factory):
    """A call_model that records every invocation and always fails."""
    calls = []

    def call_model(system_prompt, user_payload, api_key, **kwargs):
        calls.append(kwargs)
        raise exc_factory()

    return call_model, calls


def no_sleep(_seconds):
    raise AssertionError("backoff sleep must not run for a non-retryable error")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    FakeAPIError("unauthorized", 401),
    FakeAPIError("payment required", 402),
    FakeAPIError("forbidden", 403),
    FakeAPIError("Error code: 403 - {'code': 'INSUFFICIENT_BALANCE'}"),
    FakeAPIError("{'type': 'insufficient_quota'}"),
    FakeAPIError("Insufficient account balance"),
    FakeAPIError("【账户余额不足】当前可用余额 $-0.01"),
])
def test_credential_and_balance_failures_are_non_retryable(exc):
    assert is_non_retryable(exc) is True


@pytest.mark.parametrize("exc", [
    FakeAPIError("rate limited", 429),
    FakeAPIError("bad gateway", 502),
    FakeAPIError("service unavailable", 503),
    TimeoutError("read timeout"),
    ConnectionError("connection reset"),
    ValueError("news 必须是数组。"),
])
def test_transient_failures_stay_retryable(exc):
    """The classification must never widen to timeouts, 429 or 5xx -- those are
    exactly the cases a limited retry is for."""
    assert is_non_retryable(exc) is False


def test_invoke_model_wraps_non_retryable_and_still_records_usage():
    tracker = deepseek_client.DeepSeekUsageTracker()

    def call_model(system_prompt, user_payload, api_key, **kwargs):
        raise FakeAPIError("no balance", 403)

    with pytest.raises(NonRetryableAPIError) as excinfo:
        deepseek_client.invoke_model(
            call_model, "system", "payload", "key",
            thinking_enabled=False, reasoning_effort="low",
            stage="Stage B", attempt=1, usage_tracker=tracker,
        )
    assert excinfo.value.status_code == 403
    # An aborted run must still show what it attempted.
    assert len(tracker.records) == 1


# --------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------

def test_stage_b_single_pass_makes_one_call_and_does_not_sleep():
    call_model, calls = counting_raiser(lambda: FakeAPIError("no balance", 403))
    with pytest.raises(NonRetryableAPIError):
        deepseek_client.select_news(
            [candidate("a")], "key", call_model=call_model, sleep_fn=no_sleep,
        )
    assert len(calls) == 1


def test_stage_b_two_pass_does_not_fall_back_to_another_single_pass():
    """The generic orchestration handler re-runs the whole selection as a single
    pass -- that path must not be reachable for a non-retryable error."""
    call_model, calls = counting_raiser(lambda: FakeAPIError("no balance", 403))
    with pytest.raises(NonRetryableAPIError):
        deepseek_client.select_news_two_pass(
            [candidate("a")], "key", call_model=call_model, sleep_fn=no_sleep,
        )
    # Two concurrent samples are already in flight when the error lands; nothing
    # beyond them (no retry, no review, no single-pass rerun) may be dispatched.
    assert len(calls) == 2


def test_stage_b_multi_batch_stops_dispatching_after_non_retryable():
    """Three batches, failure on the first: only the first batch's two samples
    may ever reach the API. Under the old behaviour this was
    3 batches x 2 samples x 2 attempts = 12 requests."""
    pool = [candidate(str(index)) for index in range(84)]
    call_model, calls = counting_raiser(lambda: FakeAPIError("no balance", 403))
    observability = {}

    news, warning = deepseek_client.select_news_multi_batch(
        pool, "key", call_model=call_model, sleep_fn=no_sleep,
        observability=observability, batch_size=28,
    )

    assert observability["stage_b_batch_count"] == 3
    assert len(calls) == 2
    assert observability["stage_b_non_retryable_abort"] is True
    assert observability["stage_b_aborted_at_batch"] == 1
    assert warning is not None and "已中止" in warning
    # Every candidate still reaches a deterministic decision -- the page degrades
    # exactly as it already did for a failed batch.
    assert observability["stage_b_uncovered_candidate_count"] == 0


def test_stage_b_multi_batch_still_retries_a_timeout_once_per_sample():
    """The abort path must not have removed ordinary retry behaviour."""
    slept = []
    call_model, calls = counting_raiser(lambda: TimeoutError("read timeout"))

    deepseek_client.select_news_multi_batch(
        [candidate("a")], "key", call_model=call_model, sleep_fn=slept.append,
        batch_size=28,
    )

    # 2 samples x DEEPSEEK_MAX_ATTEMPTS(2)
    assert len(calls) == 2 * deepseek_client.DEEPSEEK_MAX_ATTEMPTS
    assert slept, "a retryable failure must still back off between attempts"


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------

def test_stage_a_batched_stops_dispatching_after_non_retryable():
    pool = [candidate(str(index)) for index in range(120)]
    call_model, calls = counting_raiser(lambda: FakeAPIError("no balance", 403))
    observability = {}

    events, warning = news_events.cluster_news_events_batched(
        pool, "key", call_model=call_model, sleep_fn=no_sleep,
        observability=observability, batch_size=50,
    )

    assert observability["stage_a_batch_count"] == 3
    assert len(calls) == 1
    assert observability["stage_a_non_retryable_abort"] is True
    # Deterministic one-event-per-candidate fallback still covers everything.
    assert observability["stage_a_uncovered_candidate_count"] == 0
    covered = {cid for event in events for cid in event["candidate_ids"]}
    assert covered == {item["candidate_id"] for item in pool}
    assert warning is not None


# --------------------------------------------------------------------------
# Candidate Pool Translation
# --------------------------------------------------------------------------

def test_translation_stops_dispatching_after_non_retryable():
    pool = [candidate(str(index)) for index in range(40)]
    call_model, calls = counting_raiser(lambda: FakeAPIError("no balance", 403))
    observability = {}

    translations = news_candidate_translation.translate_candidates(
        pool, "key", call_model=call_model, sleep_fn=no_sleep,
        observability=observability,
    )

    assert observability["translation_batch_count"] == 5
    assert len(calls) == 1
    assert observability["translation_non_retryable_abort"] is True
    # Everything degrades to the original English text, as a failed batch already did.
    assert translations == {}


# --------------------------------------------------------------------------
# Layer 2
# --------------------------------------------------------------------------

def test_layer2_uses_deterministic_summary_without_retrying():
    call_model, calls = counting_raiser(lambda: FakeAPIError("no balance", 403))
    market_data = {"sp500": {"name": "标普 500", "valid": False}}

    summary = market_summary.generate_market_summary(
        market_data, {}, {"health": {}}, [], "hold", "key",
        call_model=call_model, sleep_fn=no_sleep,
    )

    assert len(calls) == 1
    assert summary["degraded"] is True
