from datetime import datetime
import json
import os
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import yaml

from src.deepseek_client import NewsSelectionError, select_news, validate_selection
import src.deepseek_client as deepseek_client
from src.news_dedupe import dedupe_candidates
from src.news_events import build_event_representatives, cluster_news_events, event_selection_candidates
from src.main import _recent_news_events
from src.rss_news import fetch_candidates, filter_final_candidates


def candidate(cid, title, url, source="BBC News", summary="full summary", priority="P0"):
    return {"candidate_id": cid, "title": title, "url": url, "source": source,
            "summary": summary, "published_at": "2026-08-12T00:00:00+00:00", "priority": priority}


def enriched_selection(cid="1", rank=1, score=92, category="美联储 / 利率"):
    return {
        "rank": rank,
        "candidate_id": cid,
        "category": category,
        "title_zh": "美联储维持政策利率不变",
        "summary_zh": "委员会维持政策利率不变，并继续关注通胀和就业数据。",
        "investment_impact": "通胀回落 → 降息空间增加 → 长端利率压力缓解 → 成长股估值获得支撑。",
        "focus": "FOMC · 官员讲话 · 10Y 美债",
        "tags": ["Fed", "10Y 美债", "成长股估值"],
        "investment_relevance_score": score,
        "selection_reason": "美国利率路径直接影响股票折现率。",
    }


def test_validates_enriched_investment_fields_and_keeps_source_metadata_program_owned():
    pool = [candidate("1", "Fed holds rates", "https://trusted.example/fed", source="Reuters")]
    item = {**enriched_selection(), "source": "Fake", "url": "https://fake.example"}

    news = validate_selection({"news": [item]}, pool)

    assert news[0]["source"] == "Reuters"
    assert news[0]["url"] == "https://trusted.example/fed"
    assert news[0]["investment_relevance_score"] == 92
    assert news[0]["tags"] == ["Fed", "10Y 美债", "成长股估值"]


def test_stage_b_accepts_important_event_without_investment_impact():
    pool = [candidate("tesla", "Tesla recalls vehicles", "https://x/tesla")]
    item = enriched_selection("tesla", category="政策 / 监管")
    item.pop("investment_impact")
    item["title_zh"] = "特斯拉在华大规模召回车辆"
    item["summary_zh"] = "特斯拉在华启动大规模车辆召回，事实来源和事件信息完整。"

    news = validate_selection({"news": [item]}, pool)

    assert news[0]["title_zh"] == "特斯拉在华大规模召回车辆"
    assert "investment_impact" not in news[0]


def test_stage_b_accepts_empty_investment_impact_for_important_event():
    pool = [candidate("tesla", "Tesla recalls vehicles", "https://x/tesla")]
    item = enriched_selection("tesla", category="政策 / 监管")
    item["investment_impact"] = ""

    news = validate_selection({"news": [item]}, pool)

    assert news[0]["candidate_id"] == "tesla"
    assert "investment_impact" not in news[0]


@pytest.mark.parametrize("patch", [
    {"investment_relevance_score": 49},
    {"investment_relevance_score": 92.5},
    {"tags": []},
    {"title_zh": "标" * 71},
    {"summary_zh": "摘" * 181},
    {"focus": "关" * 81},
    {"selection_reason": "理" * 121},
])
def test_rejects_invalid_investment_contract(patch):
    pool = [candidate("1", "Fed holds rates", "https://x/1")]

    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**enriched_selection(), **patch}]}, pool)


def test_rejects_boolean_investment_relevance_score():
    pool = [candidate("1", "Fed holds rates", "https://x/1")]

    with pytest.raises(NewsSelectionError):
        validate_selection(
            {"news": [{**enriched_selection(), "investment_relevance_score": True}]}, pool
        )


@pytest.mark.parametrize("field,value", [
    ("title_zh", 123),
    ("summary_zh", {"text": "摘要"}),
    ("focus", None),
    ("selection_reason", True),
])
def test_rejects_non_string_text_fields(field, value):
    pool = [candidate("1", "Fed holds rates", "https://x/1")]

    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**enriched_selection(), field: value}]}, pool)


def test_rejects_tag_longer_than_sixteen_characters():
    pool = [candidate("1", "Fed holds rates", "https://x/1")]

    with pytest.raises(NewsSelectionError):
        validate_selection(
            {"news": [{**enriched_selection(), "tags": ["标" * 17]}]}, pool
        )


def test_stage_b_keeps_other_items_when_one_tags_item_is_invalid(capsys):
    pool = [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(10)]

    def model(system_prompt, user_payload, api_key):
        items = [enriched_selection(str(i), i + 1, 100 - i) for i in range(10)]
        items[3]["tags"] = "not-a-list"
        return json.dumps({"news": items}, ensure_ascii=False)

    selected, warning = select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    output = capsys.readouterr().out
    assert warning is None
    assert len(selected) == 9
    assert [item["rank"] for item in selected] == list(range(1, 10))
    assert "candidate_id=3" in output
    assert "field=tags" in output
    assert "action=dropped" in output
    assert "raw_count=10 valid_count=9" in output


def test_stage_b_keeps_other_items_when_investment_impact_is_missing(capsys):
    pool = [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(10)]

    def model(system_prompt, user_payload, api_key):
        items = [enriched_selection(str(i), i + 1, 100 - i) for i in range(10)]
        items[6].pop("investment_impact")
        return json.dumps({"news": items}, ensure_ascii=False)

    selected, warning = select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    output = capsys.readouterr().out
    assert warning is None
    assert len(selected) == 10
    assert "field=investment_impact" not in output
    assert "raw_count=10 valid_count=10" in output


def test_stage_b_normalizes_tags_without_inventing_tags():
    pool = [candidate("1", "Fed holds rates", "https://x/1")]

    def model(system_prompt, user_payload, api_key):
        item = enriched_selection()
        item["tags"] = [" Fed ", "", "Fed", " 美债 ", "多余标签"]
        return json.dumps({"news": [item]}, ensure_ascii=False)

    selected, warning = select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert selected[0]["tags"] == ["Fed", "美债", "多余标签"]


def test_stage_b_preserves_dynamic_news_count():
    pool = [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(12)]

    def model(system_prompt, user_payload, api_key):
        return json.dumps({
            "news": [enriched_selection(str(i), i + 1, 100 - i) for i in range(12)]
        }, ensure_ascii=False)

    selected, warning = select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert len(selected) == 12


def test_stage_b_retries_only_when_all_items_fail_validation():
    attempts = []

    def model(system_prompt, user_payload, api_key):
        attempts.append(1)
        items = [enriched_selection(str(i), i + 1, 100 - i) for i in range(10)]
        for item in items:
            item.pop("investment_impact")
        return json.dumps({"news": items}, ensure_ascii=False)

    selected, warning = select_news(
        [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(10)],
        "key", call_model=model, sleep_fn=lambda _: None,
    )

    assert len(selected) == 10
    assert warning is None
    assert len(attempts) == 1


@pytest.mark.parametrize("field", [
    "focus", "tags", "investment_relevance_score",
])
def test_rejects_missing_required_enriched_field(field):
    pool = [candidate("1", "Fed holds rates", "https://x/1")]
    item = enriched_selection()
    item.pop(field)

    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [item]}, pool)


def test_requires_non_increasing_scores():
    pool = [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(1, 3)]
    payload = {"news": [enriched_selection("1", 1, 70), enriched_selection("2", 2, 90)]}

    with pytest.raises(NewsSelectionError):
        validate_selection(payload, pool)


def test_rejects_third_same_topic_without_high_score_exception_reason():
    pool = [{**candidate(str(i), f"Title {i}", f"https://x/{i}"), "topic_group": "AI_CHIPS"}
            for i in range(1, 4)]
    payload = {"news": [enriched_selection(str(i), i, 90 - i) for i in range(1, 4)]}

    with pytest.raises(NewsSelectionError):
        validate_selection(payload, pool)


def test_allows_third_same_topic_with_high_score_and_exception_reason():
    pool = [{**candidate(str(i), f"Title {i}", f"https://x/{i}"), "topic_group": "AI_CHIPS"}
            for i in range(1, 4)]
    payload = {"news": [
        enriched_selection("1", 1, 95),
        enriched_selection("2", 2, 90),
        enriched_selection("3", 3, 85),
    ]}
    payload["news"][2]["selection_reason"] = "主题上限例外：该事件具有独立系统性影响。"

    news = validate_selection(payload, pool)

    assert [item["candidate_id"] for item in news] == ["1", "2", "3"]


def test_dedupes_canonical_url_and_normalized_title():
    items = [
        candidate("1", "Fed holds rates", "https://example.com/a?utm_source=x"),
        candidate("2", "Other title", "https://example.com/a"),
        candidate("3", " FED holds rates! ", "https://example.com/b"),
    ]
    assert [x["candidate_id"] for x in dedupe_candidates(items)] == ["1"]


def test_similar_title_prefers_higher_priority_or_more_complete_item():
    items = [
        candidate("1", "Major chipmaker unveils new AI accelerator", "https://a/1", "The Verge", "short", "P2"),
        candidate("2", "Major chipmaker unveils a new AI accelerator", "https://b/2", "Reuters", "a much fuller summary", "P0"),
    ]
    assert [x["candidate_id"] for x in dedupe_candidates(items)] == ["2"]


def test_rejects_invalid_candidate_category_duplicates_and_accepts_dynamic_count():
    pool = [candidate(str(i), f"title {i}", f"https://x/{i}") for i in range(12)]
    base = enriched_selection("0")
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**base, "candidate_id": "missing"}]}, pool)
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**base, "category": "体育"}]}, pool)
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [base, {**base, "rank": 2}]}, pool)
    selected = validate_selection(
        {"news": [{**enriched_selection(str(i), rank=i + 1), "candidate_id": str(i)} for i in range(12)]}, pool
    )
    assert len(selected) == 12


def test_rss_failure_does_not_block_other_sources():
    now = datetime(2026, 8, 12, tzinfo=ZoneInfo("UTC"))

    def parser(url):
        if "bad" in url:
            raise RuntimeError("down")
        entry = SimpleNamespace(title="Good story", link="https://good/story", summary="Summary",
                                published_parsed=now.timetuple())
        return SimpleNamespace(entries=[entry])

    candidates, warnings = fetch_candidates(
        [{"name": "Bad", "url": "https://bad/rss", "priority": "P0"},
         {"name": "Good", "url": "https://good/rss", "priority": "P1"}],
        now,
        parser=parser,
    )
    assert len(candidates) == 1
    assert candidates[0]["source"] == "Good"
    assert len(warnings) == 1


def test_rss_logs_source_counts_and_received_candidate(capsys):
    now = datetime(2026, 8, 12, tzinfo=ZoneInfo("UTC"))
    entry = SimpleNamespace(title="Fed holds rates", link="https://good/story", summary="Summary",
                            published_parsed=now.timetuple())

    candidates, warnings = fetch_candidates(
        [{"name": "Federal Reserve - Monetary Policy", "url": "https://good/rss", "priority": "P0"}],
        now,
        parser=lambda url: SimpleNamespace(entries=[entry]),
    )

    output = capsys.readouterr().out
    assert warnings == []
    assert len(candidates) == 1
    assert "[NEWS RSS SOURCE] source=Federal Reserve - Monetary Policy raw=1 accepted=1 warning=<none>" in output
    assert "stage=rss_fetch | action=received" in output
    assert "title=Fed holds rates" in output


def test_final_eligibility_logs_drop_reasons(capsys):
    now = datetime(2026, 8, 12, 12, 0, tzinfo=ZoneInfo("UTC"))
    candidates = [
        {**candidate("old", "Old", "https://x/old"), "published_at": "2026-08-11T11:59:00+00:00"},
        {**candidate("future", "Future", "https://x/future"), "published_at": "2026-08-12T12:01:00+00:00"},
        {**candidate("missing", "Missing", "https://x/missing"), "published_at": ""},
        {**candidate("invalid", "Invalid", "https://x/invalid"), "published_at": "not-a-time"},
    ]

    assert filter_final_candidates(candidates, now) == []
    output = capsys.readouterr().out
    assert "candidate_id=old" in output and "reason=too_old" in output
    assert "candidate_id=future" in output and "reason=future_timestamp" in output
    assert "candidate_id=missing" in output and "reason=missing_timestamp" in output
    assert "candidate_id=invalid" in output and "reason=invalid_timestamp" in output


def test_disabled_rss_source_is_not_requested_and_produces_no_warning():
    now = datetime(2026, 8, 12, tzinfo=ZoneInfo("UTC"))
    requested = []

    def parser(url):
        requested.append(url)
        raise RuntimeError("disabled source must never be called")

    candidates, warnings = fetch_candidates(
        [{"name": "Disabled", "url": "https://disabled/rss", "priority": "P0", "enabled": False}],
        now,
        parser=parser,
    )

    assert requested == []
    assert candidates == []
    assert warnings == []


def test_news_source_config_keeps_disabled_sources_and_adds_macro_feeds():
    config_path = __import__("pathlib").Path(__file__).parents[1] / "config" / "news_sources.yaml"
    sources = yaml.safe_load(config_path.read_text(encoding="utf-8"))["sources"]
    by_name = {source["name"]: source for source in sources}

    assert by_name["Reuters"]["enabled"] is False
    assert by_name["AP News"]["enabled"] is False
    assert {name for name, source in by_name.items() if source.get("enabled", True)} == {
        "BBC News", "BBC Business", "Federal Reserve - Monetary Policy", "TechCrunch",
        "Ars Technica", "The Guardian Business", "SEC Press Releases", "The Verge",
    }
    assert by_name["Federal Reserve - Monetary Policy"]["category_hint"] == "市场 / 宏观"
    assert by_name["SEC Press Releases"]["category_hint"] == "市场 / 宏观"


def test_category_hint_is_preserved_without_forcing_selection():
    now = datetime(2026, 8, 12, tzinfo=ZoneInfo("UTC"))
    entry = SimpleNamespace(title="Policy release", link="https://official/policy", summary="Summary",
                            published_parsed=now.timetuple())

    candidates, warnings = fetch_candidates(
        [{"name": "Official", "url": "https://official/rss", "priority": "P1", "enabled": True,
          "category_hint": "市场 / 宏观"}],
        now,
        parser=lambda url: SimpleNamespace(entries=[entry]),
    )

    assert warnings == []
    assert candidates[0]["category_hint"] == "市场 / 宏观"
    assert "selected" not in candidates[0]


def test_rss_30_hour_window_uses_feed_gmt_not_runner_timezone(monkeypatch):
    now = datetime(2026, 8, 12, 2, 0, tzinfo=ZoneInfo("UTC"))
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()

    def parser(url):
        old = datetime(2026, 8, 10, 19, 0, tzinfo=ZoneInfo("UTC"))  # 31 hours old
        entry = SimpleNamespace(title="Old story", link="https://old/story", summary="Summary",
                                published_parsed=old.timetuple())
        return SimpleNamespace(entries=[entry])

    try:
        candidates, _ = fetch_candidates(
            [{"name": "Feed", "url": "https://feed/rss", "priority": "P0"}], now, parser=parser
        )
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()
    assert candidates == []


def test_final_eligibility_keeps_30_hour_fetch_buffer_out_of_stage_a_and_b():
    now = datetime(2026, 8, 12, 12, 0, tzinfo=ZoneInfo("UTC"))
    candidates = [
        {**candidate("fresh", "Fresh", "https://x/fresh"), "published_at": "2026-08-11T13:00:00+00:00"},
        {**candidate("buffer-only", "Buffered", "https://x/buffer"), "published_at": "2026-08-11T11:00:00+00:00"},
    ]

    eligible = filter_final_candidates(candidates, now)

    assert [item["candidate_id"] for item in eligible] == ["fresh"]


def test_final_eligibility_uses_inclusive_24_hour_cutoff():
    now = datetime(2026, 8, 12, 12, 0, tzinfo=ZoneInfo("UTC"))
    candidates = [
        {**candidate("just-inside", "Just inside", "https://x/just-inside"),
         "published_at": "2026-08-11T12:01:00+00:00"},
        {**candidate("exact-cutoff", "Exact cutoff", "https://x/exact-cutoff"),
         "published_at": "2026-08-11T12:00:00+00:00"},
        {**candidate("just-outside", "Just outside", "https://x/just-outside"),
         "published_at": "2026-08-11T11:59:00+00:00"},
    ]

    eligible = filter_final_candidates(candidates, now)

    assert [item["candidate_id"] for item in eligible] == ["just-inside", "exact-cutoff"]


def test_final_eligibility_rejects_future_and_invalid_timestamps():
    now = datetime(2026, 8, 12, 12, 0, tzinfo=ZoneInfo("UTC"))
    candidates = [
        {**candidate("future", "Future", "https://x/future"),
         "published_at": "2026-08-12T12:01:00+00:00"},
        {**candidate("missing", "Missing", "https://x/missing"), "published_at": ""},
        {**candidate("invalid", "Invalid", "https://x/invalid"), "published_at": "not-a-time"},
        {**candidate("none", "None", "https://x/none"), "published_at": None},
    ]

    assert filter_final_candidates(candidates, now) == []


def test_dedupe_logs_drop_and_replace_reasons(capsys):
    items = [
        candidate("first", "Same title", "https://x/one", source="BBC News", summary="short", priority="P1"),
        candidate("second", "Same title", "https://x/two", source="Reuters", summary="long summary", priority="P0"),
    ]

    assert [item["candidate_id"] for item in dedupe_candidates(items)] == ["second"]
    output = capsys.readouterr().out
    assert "candidate_id=first" in output and "action=drop" in output
    assert "reason=replaced_by_higher_quality_candidate" in output
    assert "retained_candidate_id=second" in output
    assert "candidate_id=second" in output and "action=replace" in output


def test_source_channel_is_source_metadata_not_event_category():
    now = datetime(2026, 8, 12, tzinfo=ZoneInfo("UTC"))
    entry = SimpleNamespace(title="Treasury yields rise", link="https://example.test/yields", summary="Summary",
                            published_parsed=now.timetuple())

    candidates, warnings = fetch_candidates(
        [{"name": "CNBC", "url": "https://example.test/rss", "priority": "P1",
          "source_channel": "world_news"}],
        now,
        parser=lambda url: SimpleNamespace(entries=[entry]),
    )

    assert warnings == []
    assert candidates[0]["source_channel"] == "world_news"
    assert "event_category" not in candidates[0]


def test_two_ai_failures_degrade_without_raising():
    attempts = []
    sleeps = []

    def failing(*args, **kwargs):
        attempts.append(1)
        return "not json"

    news, warning = select_news([candidate("1", "Title", "https://x/1")], "key", call_model=failing,
                                sleep_fn=sleeps.append)
    assert news == []
    assert "新闻 AI 处理暂时失败" in warning
    assert len(attempts) == 2
    assert sleeps == [5]


def test_deepseek_client_uses_bounded_timeout_and_disables_sdk_retries(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    assert deepseek_client.call_deepseek("system", "user", "key") == "{}"
    assert captured["max_retries"] == 0
    assert captured["timeout"].connect == 5.0
    assert captured["timeout"].read == 25.0
    assert captured["timeout"].write == 15.0
    assert captured["timeout"].pool == 5.0
    assert captured["request"]["model"] == "deepseek-chat"
    assert captured["request"]["temperature"] == 0.15
    assert captured["request"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in captured["request"]


def test_news_selection_timeout_retries_once_then_returns_timeout_reason():
    attempts = []
    sleeps = []

    def timing_out(*args, **kwargs):
        attempts.append(1)
        raise TimeoutError("read timed out")

    news, warning = select_news(
        [candidate("1", "Title", "https://x/1")], "key", call_model=timing_out,
        sleep_fn=sleeps.append,
    )

    assert news == []
    assert len(attempts) == 2
    assert sleeps == [5]
    assert "timeout" in warning.lower()


def test_deepseek_payload_includes_market_driven_context_and_selection_reason():
    captured = {}

    def model(system_prompt, user_payload, api_key, **kwargs):
        captured["prompt"] = system_prompt
        captured["payload"] = json.loads(user_payload)
        captured["kwargs"] = kwargs
        return json.dumps({"news": [{
            **enriched_selection(),
            "selection_reason": "与利率明显上升相关",
        }]})

    market_context = {
        "core_market": {"sp500": {"daily_return": -0.003}},
        "market_context": {"us10y": {"close": 4.32, "yield_change_bp": 7}},
        "market_signals": {"us10y_bp_change": 7, "signals": ["利率明显上升"]},
        "market_context_text": "【市场环境】\n10Y 美债 4.32% +7bp",
    }
    news, warning = select_news(
        [candidate("1", "Title", "https://x/1")], "key", market_context=market_context,
        call_model=model, sleep_fn=lambda _: None,
    )

    assert warning is None
    assert captured["payload"]["core_market"] == market_context["core_market"]
    assert captured["payload"]["market_context"] == market_context["market_context"]
    assert captured["payload"]["market_signals"] == market_context["market_signals"]
    assert "不得根据时间共现" in captured["prompt"]
    assert news[0]["selection_reason"] == "与利率明显上升相关"
    assert captured["kwargs"] == {"thinking_enabled": False, "reasoning_effort": None}


def test_stage_b_logs_input_raw_return_and_validation(capsys):
    def model(system_prompt, user_payload, api_key):
        return json.dumps({"news": [{
            **enriched_selection(),
            "title_zh": "美联储维持利率",
            "summary_zh": "摘要",
            "selection_reason": "重大宏观事件",
        }]})

    selected, warning = select_news(
        [candidate("1", "Fed holds rates", "https://x/1")],
        "secret-api-key",
        call_model=model,
        sleep_fn=lambda _: None,
    )

    output = capsys.readouterr().out
    assert warning is None
    assert selected[0]["title_zh"] == "美联储维持利率"
    assert "[NEWS STAGE B] input events: 1" in output
    assert "candidate_id=1 | category=other | title=Fed holds rates" in output
    assert "[NEWS STAGE B] DeepSeek raw return count: 1" in output
    assert "rank=1 | candidate_id=1 | title=美联储维持利率" in output
    assert "importance=<missing>" in output
    assert "investment_relevance_score=92" in output
    assert "validate_selection: passed" in output
    assert "secret-api-key" not in output


def test_deepseek_payload_accepts_breadth_context_and_prompt_has_sector_rotation_rule():
    captured = {}

    def model(system_prompt, user_payload, api_key):
        captured["prompt"] = system_prompt
        captured["payload"] = json.loads(user_payload)
        return json.dumps({"news": []})

    market_context = {
        "market_breadth": {"stocks": {"advance_ratio": 0.65}, "health": {"level": "healthy"}},
        "market_breadth_text": "【市场宽度】\n领先板块：科技 +1.2%",
    }
    news, warning = select_news(
        [candidate("1", "Title", "https://x/1")], "key", market_context=market_context,
        call_model=model, sleep_fn=lambda _: None,
    )

    assert news == [] and warning is None
    assert captured["payload"]["market_breadth"] == market_context["market_breadth"]
    assert captured["payload"]["market_breadth_text"] == market_context["market_breadth_text"]
    assert "市场宽度" in captured["prompt"]
    assert "板块轮动" in captured["prompt"]
    assert "不得无依据建立因果关系" in captured["prompt"]


def test_deepseek_still_runs_without_market_context():
    calls = []

    def model(system_prompt, user_payload, api_key):
        calls.append(json.loads(user_payload))
        return json.dumps({"news": []})

    news, warning = select_news(
        [candidate("1", "Title", "https://x/1")], "key",
        call_model=model, sleep_fn=lambda _: None,
    )
    assert news == []
    assert warning is None
    assert "market_context" not in calls[0]


def test_selection_uses_program_owned_event_metadata_and_excludes_urls_from_stage_b_payload():
    captured = {}
    event_candidate = {
        **candidate("1", "Fed holds rates", "https://secret.example/fed"),
        "event_summary": "Fed held rates after its meeting.",
        "topic_group": "US_MARKET_MACRO",
    }

    def model(system_prompt, user_payload, api_key):
        captured["payload"] = json.loads(user_payload)
        return json.dumps({"news": [{
            **enriched_selection(),
            "title_zh": "美联储维持利率",
            "summary_zh": "摘要",
            "selection_reason": "重大宏观事件",
            "event_summary": "模型伪造摘要", "topic_group": "AI_CHIPS",
        }]})

    news, warning = select_news(
        [event_candidate], "key", recent_selected=[{"event_summary": "历史事件", "topic_group": "US_MARKET_MACRO"}],
        call_model=model, sleep_fn=lambda _: None,
    )

    assert warning is None
    assert captured["payload"]["events"][0]["event_summary"] == "Fed held rates after its meeting."
    assert captured["payload"]["recent_7_days_events"][0]["event_summary"] == "历史事件"
    assert "url" not in captured["payload"]["events"][0]
    assert news[0]["event_summary"] == "Fed held rates after its meeting."
    assert news[0]["topic_group"] == "US_MARKET_MACRO"


def test_recent_news_events_supports_legacy_and_v2_reports():
    events = _recent_news_events([
        {"report_date": "2026-08-12", "news": [{
            "original_title": "Legacy Fed headline", "url": "https://legacy.example",
        }]},
        {"report_date": "2026-08-11", "news": [{
            "original_title": "New headline", "event_summary": "Fed held rates", "topic_group": "US_MARKET_MACRO",
        }]},
    ])

    assert events == [
        {"report_date": "2026-08-12", "event_summary": "Legacy Fed headline", "topic_group": None,
         "original_title": "Legacy Fed headline"},
        {"report_date": "2026-08-11", "event_summary": "Fed held rates", "topic_group": "US_MARKET_MACRO",
         "original_title": "New headline"},
    ]


def test_event_selection_prompt_has_topic_concentration_contract():
    from src.news_prompt import SYSTEM_PROMPT

    assert "同一 topic_group 通常最多2条" in SYSTEM_PROMPT
    assert "重大独立事件允许突破" in SYSTEM_PROMPT
    assert "突破必须说明理由" in SYSTEM_PROMPT


def test_investment_priority_prompt_has_selection_contract():
    from src.news_prompt import SYSTEM_PROMPT

    for phrase in (
        "长期持有 SPY 与 Nasdaq-100",
        "美国资产定价的重要程度",
        "investment_relevance_score",
        "importance*0.35 + us_relevance*0.30 + novelty*0.20 + persistence*0.15",
        "低于50分不得入选",
        "普通产品更新",
        "普通公司融资",
        "不要返回URL",
    ):
        assert phrase in SYSTEM_PROMPT


def test_stage_a_fallback_still_feeds_stage_b_selection():
    pool = [candidate("a", "Fed holds rates", "https://x/a"), candidate("b", "Oil rises", "https://x/b")]
    events, clustering_warning = cluster_news_events(
        pool, "key", call_model=lambda *args: "not json", sleep_fn=lambda _: None
    )

    def editor(system_prompt, user_payload, api_key):
        return json.dumps({"news": [{
            **enriched_selection("a"),
            "title_zh": "美联储维持利率",
            "summary_zh": "摘要",
        }]})

    selected, selection_warning = select_news(
        event_selection_candidates(build_event_representatives(events, pool)), "key",
        call_model=editor, sleep_fn=lambda _: None,
    )

    assert "事件级去重暂时失败" in clustering_warning
    assert selection_warning is None
    assert selected[0]["candidate_id"] == "a"


def test_event_selection_prompt_strengthens_market_structure_relevance():
    from src.news_prompt import SYSTEM_PROMPT

    assert "市场结构相关性" in SYSTEM_PROMPT
    assert "不得把相关性写成确定因果" in SYSTEM_PROMPT


def test_selection_rules_v1_are_explicit_and_dynamic_count():
    from src.news_prompt import SYSTEM_PROMPT

    for phrase in (
        "Selection Rules v1",
        "政策状态或制度环境",
        "系统性金融变化",
        "产业结构变化",
        "投资传导路径",
        "普通产品更新",
        "不得根据当天市场涨跌反向寻找新闻",
        "来源质量不等于事件重要性",
        "不得通过夸大 why_it_matters",
        "数量动态",
        "不为凑数选入低价值事件",
    ):
        assert phrase in SYSTEM_PROMPT


def test_stage_b_prompt_matches_validated_demo_selection_contract():
    from src.news_prompt import SYSTEM_PROMPT

    for phrase in (
        "macro_policy",
        "financial_markets",
        "high_tech",
        "geopolitics",
        "重大事件优先展示",
        "rank 仅作为展示顺序编号",
        "同一事件的不同媒体报道只能占1条",
        "分析师观点",
        "评论或观点文章",
        "不要把“某只股票、某个板块、债券、商品或其他资产上涨/下跌”本身作为新闻事件",
        "importance*0.35 + us_relevance*0.30 + novelty*0.20 + persistence*0.15",
    ):
        assert phrase in SYSTEM_PROMPT
