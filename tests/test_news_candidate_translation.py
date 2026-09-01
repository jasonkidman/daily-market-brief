import json

from src.news_candidate_translation import translate_candidates, validate_translations


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
