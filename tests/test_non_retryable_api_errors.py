"""PR 1 regression guards: a credential/balance failure must cost exactly one
request per already-dispatched call site, and must never fan out into retries,
single-pass fallbacks, or further batches.

Confirmed live on 2026-09-08 (run #88): a 403 INSUFFICIENT_BALANCE was handled
as an ordinary exception, so every Stage A batch, both Stage B samples of every
batch, their retries and every translation batch each re-sent a request that
could not possibly succeed, with 5-10s backoff sleeps in between.
"""

import pytest

from src import deepseek_client, market_summary, news_candidate_translation
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
# Scoring (news_scoring.score_candidates) -- see tests/test_news_scoring.py
# for the full call-count regression suite (batching, invalid/missing items,
# concurrency). The non-retryable-abort and retry-once cases specifically are
# also covered there (test_non_retryable_aborts_without_retry_and_stops_new_dispatches,
# test_retry_only_once_on_timeout); not duplicated here.
# --------------------------------------------------------------------------

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
