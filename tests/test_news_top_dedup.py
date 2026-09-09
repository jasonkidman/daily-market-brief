"""TopDedup: the bounded, top-ranked-only semantic near-duplicate call added
after real 2026-09-09 data showed pure text-similarity cannot catch a same
event reported under very differently worded headlines. Every failure mode
must degrade to a no-op (treat as not-a-duplicate), never raise, and never
touch a candidate outside the ranked pool it was given."""

from __future__ import annotations

import json

from src.deepseek_client import NonRetryableAPIError
from src.news_top_dedup import dedupe_top_candidates


def candidate(cid):
    return {
        "candidate_id": cid, "title": f"Title {cid}", "summary": "Summary",
        "source": "Source", "published_at": "2026-08-12T00:00:00+00:00",
    }


def fixed_response(items):
    def call_model(system_prompt, user_payload, api_key, **kwargs):
        return json.dumps({"items": items})
    return call_model


def test_marks_non_representative_duplicates():
    pool = [candidate("a"), candidate("b"), candidate("c")]
    call_model = fixed_response([
        {"candidate_id": "a", "duplicate_of": None},
        {"candidate_id": "b", "duplicate_of": "a"},
        {"candidate_id": "c", "duplicate_of": None},
    ])

    result = dedupe_top_candidates(pool, "key", call_model=call_model)

    assert result == {"a": None, "b": "a", "c": None}


def test_unknown_candidate_id_in_response_is_dropped():
    pool = [candidate("a"), candidate("b")]
    call_model = fixed_response([
        {"candidate_id": "a", "duplicate_of": None},
        {"candidate_id": "not-in-pool", "duplicate_of": None},
    ])

    result = dedupe_top_candidates(pool, "key", call_model=call_model)

    assert result == {"a": None}


def test_duplicate_of_pointing_outside_the_pool_is_treated_as_not_a_duplicate():
    pool = [candidate("a"), candidate("b")]
    call_model = fixed_response([
        {"candidate_id": "a", "duplicate_of": "ghost-id"},
        {"candidate_id": "b", "duplicate_of": None},
    ])

    result = dedupe_top_candidates(pool, "key", call_model=call_model)

    assert result == {"a": None, "b": None}


def test_self_referencing_duplicate_of_is_treated_as_not_a_duplicate():
    pool = [candidate("a"), candidate("b")]
    call_model = fixed_response([
        {"candidate_id": "a", "duplicate_of": "a"},
        {"candidate_id": "b", "duplicate_of": None},
    ])

    result = dedupe_top_candidates(pool, "key", call_model=call_model)

    assert result["a"] is None


def test_chained_duplicate_of_resolves_to_not_a_duplicate_rather_than_re_chaining():
    """A -> B -> C (B itself marked as a duplicate) is ambiguous about who the
    real representative is; the safer failure mode is showing one extra
    near-duplicate (A kept) rather than guessing a chain and risking the
    genuine representative (C) being excluded from everything."""
    pool = [candidate("a"), candidate("b"), candidate("c")]
    call_model = fixed_response([
        {"candidate_id": "a", "duplicate_of": "b"},
        {"candidate_id": "b", "duplicate_of": "c"},
        {"candidate_id": "c", "duplicate_of": None},
    ])

    result = dedupe_top_candidates(pool, "key", call_model=call_model)

    assert result["a"] is None
    assert result["b"] == "c"
    assert result["c"] is None


def test_pool_size_limits_which_candidates_are_sent():
    pool = [candidate(str(i)) for i in range(5)]
    sent_ids = []

    def call_model(system_prompt, user_payload, api_key, **kwargs):
        payload = json.loads(user_payload)
        sent_ids.extend(item["candidate_id"] for item in payload["candidates"])
        return json.dumps({"items": [{"candidate_id": cid, "duplicate_of": None} for cid in sent_ids]})

    dedupe_top_candidates(pool, "key", call_model=call_model, pool_size=3)

    assert sent_ids == ["0", "1", "2"]


def test_fewer_than_two_candidates_skips_the_call_entirely():
    calls = []

    def call_model(*args, **kwargs):
        calls.append(1)
        return json.dumps({"items": []})

    result = dedupe_top_candidates([candidate("a")], "key", call_model=call_model)

    assert result == {}
    assert calls == []


def test_non_retryable_error_degrades_to_no_op():
    def call_model(*args, **kwargs):
        raise NonRetryableAPIError(RuntimeError("no balance"))

    result = dedupe_top_candidates([candidate("a"), candidate("b")], "key", call_model=call_model)

    assert result == {}


def test_transport_failure_degrades_to_no_op():
    def call_model(*args, **kwargs):
        raise RuntimeError("boom")

    result = dedupe_top_candidates([candidate("a"), candidate("b")], "key", call_model=call_model)

    assert result == {}


def test_unparseable_output_degrades_to_no_op():
    def bad_call_model(system_prompt, user_payload, api_key, **kwargs):
        return json.dumps({"items": "not-a-list"})  # wrong shape on purpose

    result = dedupe_top_candidates([candidate("a"), candidate("b")], "key", call_model=bad_call_model)

    assert result == {}
