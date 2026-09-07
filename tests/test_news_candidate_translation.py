import json

from src.news_candidate_translation import TRANSLATION_BATCH_SIZE, translate_candidates, validate_translations


def candidate(candidate_id, title="Title", summary="Summary"):
    return {"candidate_id": candidate_id, "title": title, "summary": summary}


def test_validate_translations_keeps_only_known_well_formed_entries():
    candidates = [candidate("a"), candidate("b")]
    payload = {"translations": [
        {"candidate_id": "a", "title_zh": "标题A", "summary_zh": "摘要A"},
        {"candidate_id": "unknown", "title_zh": "标题", "summary_zh": "摘要"},
        {"candidate_id": "b", "title_zh": "", "summary_zh": "摘要B"},
    ]}

    result = validate_translations(payload, candidates)

    assert result == {"a": {"title_zh": "标题A", "summary_zh": "摘要A"}}


def test_validate_translations_clips_overlong_text_instead_of_dropping_the_entry():
    """A long source summary (e.g. a lengthy Bloomberg writeup) can produce a
    faithful translation that runs over budget; losing the whole entry back to
    raw English is worse for review than a clipped Chinese translation."""
    candidates = [candidate("a")]
    payload = {"translations": [{"candidate_id": "a", "title_zh": "x" * 71, "summary_zh": "摘" * 200}]}

    result = validate_translations(payload, candidates)

    assert result["a"]["title_zh"] == "x" * 69 + "…"
    assert len(result["a"]["title_zh"]) == 70
    assert result["a"]["summary_zh"] == "摘" * 179 + "…"
    assert len(result["a"]["summary_zh"]) == 180


def test_validate_translations_drops_duplicate_candidate_ids():
    candidates = [candidate("a")]
    payload = {"translations": [
        {"candidate_id": "a", "title_zh": "第一次", "summary_zh": "摘要"},
        {"candidate_id": "a", "title_zh": "第二次", "summary_zh": "摘要"},
    ]}

    result = validate_translations(payload, candidates)

    assert result == {"a": {"title_zh": "第一次", "summary_zh": "摘要"}}


def test_translate_candidates_returns_empty_dict_without_api_key():
    assert translate_candidates([candidate("a")], api_key=None) == {}


def test_translate_candidates_returns_empty_dict_for_empty_input():
    assert translate_candidates([], api_key="test-key") == {}


def test_translate_candidates_parses_successful_model_response():
    def fake_call(system_prompt, user_payload, api_key):
        payload = json.loads(user_payload)
        return json.dumps({"translations": [
            {"candidate_id": c["candidate_id"], "title_zh": f"中文-{c['title']}", "summary_zh": f"摘要-{c['summary']}"}
            for c in payload["candidates"]
        ]}, ensure_ascii=False)

    result = translate_candidates(
        [candidate("a", title="Fed holds rates", summary="Fed summary")],
        api_key="test-key", call_model=fake_call,
    )

    assert result == {"a": {"title_zh": "中文-Fed holds rates", "summary_zh": "摘要-Fed summary"}}


def test_translate_candidates_falls_back_to_empty_dict_without_raising_on_failure(monkeypatch):
    def failing_call(system_prompt, user_payload, api_key):
        raise RuntimeError("network error")

    result = translate_candidates(
        [candidate("a")], api_key="test-key", call_model=failing_call,
        sleep_fn=lambda seconds: None,
    )

    assert result == {}


def test_translate_candidates_falls_back_to_empty_dict_on_malformed_json():
    def bad_call(system_prompt, user_payload, api_key):
        return "not json"

    result = translate_candidates(
        [candidate("a")], api_key="test-key", call_model=bad_call, sleep_fn=lambda seconds: None,
    )

    assert result == {}


def _translating_model(system_prompt, user_payload, api_key):
    payload = json.loads(user_payload)
    return json.dumps({"translations": [
        {"candidate_id": c["candidate_id"], "title_zh": f"中文-{c['title']}", "summary_zh": f"摘要-{c['summary']}"}
        for c in payload["candidates"]
    ]}, ensure_ascii=False)


def test_translate_candidates_splits_large_pool_into_multiple_batches():
    """Regression for the real 2026-09-07 incident: ~46 unselected candidates
    were all sent to Candidate Pool Translation in one request."""
    candidates = [candidate(f"c{i}") for i in range(46)]
    observability = {}

    result = translate_candidates(
        candidates, api_key="test-key", call_model=_translating_model, observability=observability,
    )

    assert observability["translation_requested_count"] == 46
    assert observability["translation_batch_count"] == 6  # ceil(46 / 8)
    assert all(len(result[c["candidate_id"]]) == 2 for c in candidates)
    assert len(result) == 46


def test_translate_candidates_one_batch_failure_does_not_fall_back_other_batches():
    """A failing batch must only cost its own candidates their translation --
    other batches must still come back in Chinese, not all-English."""
    candidates = [candidate(f"c{i}") for i in range(20)]  # 2 batches of 10
    first_batch_ids = {c["candidate_id"] for c in candidates[:TRANSLATION_BATCH_SIZE]}

    def flaky_model(system_prompt, user_payload, api_key):
        payload = json.loads(user_payload)
        ids = {c["candidate_id"] for c in payload["candidates"]}
        if ids == first_batch_ids:
            raise TimeoutError("simulated batch timeout")
        return _translating_model(system_prompt, user_payload, api_key)

    observability = {}
    result = translate_candidates(
        candidates, api_key="test-key", call_model=flaky_model,
        sleep_fn=lambda seconds: None, observability=observability,
    )

    second_batch_ids = {c["candidate_id"] for c in candidates[TRANSLATION_BATCH_SIZE:]}
    assert second_batch_ids.issubset(result.keys())
    assert not (first_batch_ids & result.keys())
    assert len(observability["translation_batch_failures"]) == 1
    assert observability["translation_batch_failures"][0]["batch"] == 1


def test_translate_candidates_success_plus_failed_equals_requested():
    candidates = [candidate(f"c{i}") for i in range(25)]  # 3 batches
    second_batch_ids = {c["candidate_id"] for c in candidates[TRANSLATION_BATCH_SIZE:2 * TRANSLATION_BATCH_SIZE]}

    def flaky_model(system_prompt, user_payload, api_key):
        payload = json.loads(user_payload)
        ids = {c["candidate_id"] for c in payload["candidates"]}
        if ids == second_batch_ids:
            raise TimeoutError("simulated batch timeout")
        return _translating_model(system_prompt, user_payload, api_key)

    observability = {}
    translate_candidates(
        candidates, api_key="test-key", call_model=flaky_model,
        sleep_fn=lambda seconds: None, observability=observability,
    )

    assert (
        observability["translation_success_count"] + observability["translation_failed_count"]
        == observability["translation_requested_count"]
        == 25
    )
    assert observability["translation_failed_count"] == TRANSLATION_BATCH_SIZE
