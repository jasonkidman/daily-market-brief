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


CONTRACT_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "news_selection_eval",
    "2026-08-25-run-32820597633-stage-b-contract-fixture.json",
)


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


def stage_b_item(cid, rank=1, score=92, category="美联储 / 利率"):
    return enriched_selection(cid, rank=rank, score=score, category=category)


def stage_b_pool(ids, topic_group=None):
    return [
        {**candidate(cid, f"Title {cid}", f"https://x/{cid}"),
         **({"topic_group": topic_group} if topic_group else {})}
        for cid in ids
    ]


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


def test_stage_b_selected_all_valid_does_not_consume_reserve():
    ids = [str(i) for i in range(9)]
    pool = stage_b_pool(ids)

    def model(system_prompt, user_payload, api_key):
        return json.dumps({
            "selected": [stage_b_item(str(i), i + 1, 100 - i) for i in range(6)],
            "reserve": [stage_b_item(str(i), i + 1, 90 - i) for i in range(6, 9)],
        }, ensure_ascii=False)

    observability = {}
    selected, warning = select_news(pool, "key", call_model=model,
                                    sleep_fn=lambda _: None,
                                    observability=observability)

    assert warning is None
    assert len(selected) == 6
    assert observability["stage_b_target_count"] == 6
    assert observability["stage_b_backfilled_count"] == 0
    assert observability["stage_b_reserve_validation_pass_count"] == 0


def test_stage_b_does_not_backfill_reserve_after_selected_item_is_dropped():
    pool = stage_b_pool(["0", "1", "2"])

    def model(system_prompt, user_payload, api_key):
        invalid = stage_b_item("1", 2, 90)
        invalid["tags"] = []
        return json.dumps({
            "selected": [stage_b_item("0", 1, 92), invalid],
            "reserve": [stage_b_item("2", 1, 80)],
        }, ensure_ascii=False)

    observability = {}
    selected, warning = select_news(pool, "key", call_model=model,
                                    sleep_fn=lambda _: None,
                                    observability=observability)

    assert warning is None
    assert [item["candidate_id"] for item in selected] == ["0"]
    assert observability["stage_b_target_count"] == 2
    assert observability["stage_b_backfilled_count"] == 0
    assert observability["stage_b_reserve_validation_pass_count"] == 0


def test_stage_b_does_not_backfill_selected_topic_drop():
    ids = [str(i) for i in range(8)]
    pool = stage_b_pool(ids[:7], topic_group="AI_CHIPS") + stage_b_pool(ids[7:])
    pool[0]["topic_group"] = "US_MARKET_MACRO"
    pool[1]["topic_group"] = "US_MARKET_MACRO"
    pool[2]["topic_group"] = "MEGA_CAP_TECH"
    pool[3]["topic_group"] = "MEGA_CAP_TECH"
    pool[4]["topic_group"] = "AI_CHIPS"
    pool[5]["topic_group"] = "AI_CHIPS"
    pool[6]["topic_group"] = "AI_CHIPS"
    pool[7]["topic_group"] = "AI_CHIPS"

    def model(system_prompt, user_payload, api_key):
        selected = [stage_b_item(str(i), i + 1, 92 - i * 2) for i in range(7)]
        selected[6]["investment_relevance_score"] = 80
        reserve = [stage_b_item("7", 1, 79)]
        pool[7]["topic_group"] = "OTHER_SYSTEMIC"
        return json.dumps({"selected": selected, "reserve": reserve}, ensure_ascii=False)

    observability = {}
    selected, warning = select_news(pool, "key", call_model=model,
                                    sleep_fn=lambda _: None,
                                    observability=observability)

    assert warning is None
    assert [item["candidate_id"] for item in selected] == [str(i) for i in range(6)]
    assert observability["stage_b_target_count"] == 7
    assert observability["stage_b_backfilled_count"] == 0


def test_stage_b_reserve_is_not_used_to_restore_selected_count():
    pool = stage_b_pool([str(i) for i in range(5)], topic_group="AI_CHIPS")

    def model(system_prompt, user_payload, api_key):
        selected = [stage_b_item(str(i), i + 1, 92 - i * 2) for i in range(3)]
        selected[2]["investment_relevance_score"] = 80
        reserve = [stage_b_item("3", 1, 78), stage_b_item("4", 2, 77)]
        pool[3]["topic_group"] = "AI_CHIPS"
        pool[4]["topic_group"] = "OTHER_SYSTEMIC"
        return json.dumps({"selected": selected, "reserve": reserve}, ensure_ascii=False)

    selected, warning = select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert [item["candidate_id"] for item in selected] == ["0", "1"]


def test_stage_b_reserve_exhaustion_does_not_overfill_or_lower_target():
    pool = stage_b_pool([str(i) for i in range(4)], topic_group="AI_CHIPS")

    def model(system_prompt, user_payload, api_key):
        selected = [stage_b_item(str(i), i + 1, 92 - i * 2) for i in range(3)]
        selected[2]["investment_relevance_score"] = 80
        reserve = [stage_b_item("3", 1, 77)]
        reserve[0]["tags"] = []
        return json.dumps({"selected": selected, "reserve": reserve}, ensure_ascii=False)

    observability = {}
    selected, warning = select_news(pool, "key", call_model=model,
                                    sleep_fn=lambda _: None,
                                    observability=observability)

    assert warning is None
    assert len(selected) == 2
    assert observability["stage_b_target_count"] == 3
    assert observability["stage_b_final_count"] == 2


def test_stage_b_dynamic_target_is_selected_count_not_eight():
    for selected_count in (5, 7):
        ids = [str(i) for i in range(selected_count)]
        pool = stage_b_pool(ids)

        def model(system_prompt, user_payload, api_key, count=selected_count):
            return json.dumps({"selected": [stage_b_item(str(i), i + 1, 100 - i) for i in range(count)],
                               "reserve": []}, ensure_ascii=False)

        observability = {}
        selected, warning = select_news(pool, "key", call_model=model,
                                        sleep_fn=lambda _: None,
                                        observability=observability)
        assert warning is None
        assert len(selected) == selected_count
        assert observability["stage_b_target_count"] == selected_count


def test_stage_b_backfill_does_not_make_an_additional_llm_call():
    calls = []
    pool = stage_b_pool(["0", "1", "2"], topic_group="AI_CHIPS")
    pool[0]["topic_group"] = "US_MARKET_MACRO"

    def model(system_prompt, user_payload, api_key):
        calls.append(1)
        selected = [stage_b_item("0", 1, 92), stage_b_item("1", 2, 80)]
        return json.dumps({"selected": selected, "reserve": [stage_b_item("2", 1, 79)]}, ensure_ascii=False)

    select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert len(calls) == 1


def test_stage_b_duplicate_reserve_cannot_enter_final():
    pool = stage_b_pool(["0", "1"], topic_group="AI_CHIPS")
    pool[0]["topic_group"] = "US_MARKET_MACRO"

    def model(system_prompt, user_payload, api_key):
        return json.dumps({"selected": [stage_b_item("0", 1, 92)],
                           "reserve": [stage_b_item("0", 1, 91), stage_b_item("1", 2, 90)]}, ensure_ascii=False)

    selected, warning = select_news(pool, "key", call_model=model, sleep_fn=lambda _: None)

    assert warning is None
    assert [item["candidate_id"] for item in selected] == ["0"]


def test_run_32822389426_fixture_keeps_amazon_drop_without_reserve_backfill(capsys):
    fixture = json.load(open(CONTRACT_FIXTURE, encoding="utf-8"))
    contract = fixture["stage_b_response_fixture"]
    ids = contract["selected"] + contract["reserve"]
    pool = stage_b_pool(ids)
    topic_map = {
        "Treasury": "US_MARKET_MACRO", "US-Canada": "GEOPOLITICS", "Iran": "ENERGY_COMMODITIES",
        "Hugging Face acquisition": "MEGA_CAP_TECH",
        "Hugging Face/OpenAI investigation": "MEGA_CAP_TECH", "AI jobs": "AI_CHIPS",
        "Amazon": "MEGA_CAP_TECH",
    }
    for item in pool:
        item["topic_group"] = topic_map.get(item["candidate_id"], "OTHER_SYSTEMIC")

    def model(system_prompt, user_payload, api_key):
        selected = [stage_b_item(cid, index + 1, 92 - index * 2)
                    for index, cid in enumerate(contract["selected"])]
        selected[-1]["investment_relevance_score"] = 80
        reserve = [stage_b_item(cid, index + 1, 78 - index)
                   for index, cid in enumerate(contract["reserve"])]
        return json.dumps({"selected": selected, "reserve": reserve}, ensure_ascii=False)

    observability = {}
    selected, warning = select_news(pool, "key", call_model=model,
                                    sleep_fn=lambda _: None,
                                    observability=observability)

    output = capsys.readouterr().out
    assert warning is None
    assert len(selected) == 6
    assert "Amazon" not in [item["candidate_id"] for item in selected]
    assert "SEC probe" not in [item["candidate_id"] for item in selected]
    assert observability["stage_b_target_count"] == 7
    assert observability["stage_b_backfilled_count"] == 0
    assert "stage_b_backfill candidate_id=SEC probe" not in output


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


def test_normalizes_non_increasing_scores_without_dropping_valid_news():
    pool = [candidate(str(i), f"Title {i}", f"https://x/{i}") for i in range(1, 3)]
    payload = {"news": [enriched_selection("1", 1, 70), enriched_selection("2", 2, 90)]}

    news = validate_selection(payload, pool)

    assert [item["candidate_id"] for item in news] == ["2", "1"]
    assert [item["investment_relevance_score"] for item in news] == [90, 70]


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
        "不得用50-69分新闻填补数量",
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


def test_stage_b_prompt_has_us_market_hard_gate_and_no_padding_rules():
    from src.news_prompt import SYSTEM_PROMPT

    for phrase in (
        "If only a small number of stories meet the importance threshold, return only those stories.",
        "Do not fill remaining slots with lower-priority financial, technology, funding, local economic, or industry news.",
        'Would a U.S. equity / rates / macro investor reasonably care about this development today?',
        "If the answer is no, exclude it.",
        "Local economic developments outside the U.S. should normally be excluded unless they have a clear and material transmission channel to U.S. markets.",
        "Ordinary Series A/B/C/D financing",
        "AI-related news is not automatically important.",
        "event → market / macro / industry variable → U.S. asset prices or major listed companies",
        "Before finalizing the list, apply a marginal-value test:",
        "If this story were removed, would the investor materially lose understanding of today's market, macro, technology, or geopolitical environment?",
        "Before returning JSON, ensure selected rank and investment_relevance_score are in non-increasing order; reorder items instead of dropping a qualifying story.",
        "No story with a score below 70 may be selected.",
        "A local fusion demonstration project",
        "$200M autonomous-driving startup round",
        "UK household energy-price forecast without global spillover",
        "A SpaceX-scale capital expenditure or infrastructure event, including a $100B-class launch facility, should be retained when confirmed.",
        "Confirmed major OpenAI product or infrastructure events and $100B-class SpaceX infrastructure events should not be dropped as indirect.",
    ):
        assert phrase in SYSTEM_PROMPT

    assert "50-69分只能在高质量事件不足时补充" not in SYSTEM_PROMPT
    assert "select approximately 8-10 stories" not in SYSTEM_PROMPT


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


def test_stage_b_prompt_prioritizes_major_single_company_events_and_explicit_regulatory_action():
    from src.news_prompt import SYSTEM_PROMPT

    fixture = json.load(open(CONTRACT_FIXTURE, encoding="utf-8"))
    assert fixture["expected_contract"]["A"].startswith("major_company_action")
    assert fixture["expected_contract"]["B"].startswith("explicit_regulatory_action")
    for phrase in (
        "单一公司事件不能因为只影响一家企业自动降级",
        "大规模召回",
        "重大监管调查",
        "重大诉讼/反垄断",
        "明确的 SEC / FTC / DOJ / federal investigation / enforcement",
        "高于普通 AI 产品发布、创业融资、传闻类新闻",
    ):
        assert phrase in SYSTEM_PROMPT


def test_stage_b_prompt_prioritizes_confirmed_policy_state_changes_over_threats():
    from src.news_prompt import SYSTEM_PROMPT

    fixture = json.load(open(CONTRACT_FIXTURE, encoding="utf-8"))
    assert fixture["expected_contract"]["D"] == "policy_state_change > future_threat"
    for phrase in (
        "外交/政策状态变化",
        "sanctions",
        "terrorism designation",
        "export controls",
        "tariffs",
        "diplomatic recognition/status",
        "已经发生的政策状态变化",
        "优先于纯预测、口头威胁",
    ):
        assert phrase in SYSTEM_PROMPT


def test_stage_b_prompt_handles_structural_employment_news_without_equating_it_to_official_data():
    from src.news_prompt import SYSTEM_PROMPT

    assert "结构性宏观/就业变化" in SYSTEM_PROMPT
    assert "可靠研究或数据支持" in SYSTEM_PROMPT
    assert "不要强行等同于官方就业数据" in SYSTEM_PROMPT


def test_stage_b_prompt_has_dual_track_and_big_tech_independent_standard():
    from src.news_prompt import SYSTEM_PROMPT

    for phrase in (
        "两条独立主线",
        "主线 A（美国市场重要新闻）",
        "主线 B（大型科技公司重要动态）",
        "不需要证明其对美股大盘、利率、汇率或其他宏观变量存在明确影响",
        "重点科技公司名单：Apple、Microsoft、Alphabet / Google、Amazon、Meta、NVIDIA、Tesla、SpaceX",
        "大型科技公司独立重要性标准（主线 B）",
        "普通产品小更新、一般性营销活动、小型合作、人物花边等低重要度信息即使提到这些公司也必须排除",
        "此硬门槛适用于主线 A",
        "此硬判断同样只适用于主线 A",
    ):
        assert phrase in SYSTEM_PROMPT


def test_stage_b_prompt_has_default_exclude_categories_for_geopolitics_cyber_and_china_policy():
    from src.news_prompt import SYSTEM_PROMPT

    for phrase in (
        "默认排除类别",
        "泛地缘政治：一般外交冲突、军事行动、国家间政治对抗、普通制裁或外交表态",
        "台海、俄乌、中东等事件的常规进展",
        "泛网络安全事件：黑客组织活动、政府网站攻击、国家间网络攻击、僵尸网络、网络安全执法行动默认排除",
        "中国国内政策新闻：继续默认排除",
        "泛国际新闻：与美国宏观经济、美国金融市场、重点科技公司均没有明确关系的新闻默认排除",
    ):
        assert phrase in SYSTEM_PROMPT


def test_stage_b_prompt_bans_vague_market_correlation_phrases():
    from src.news_prompt import SYSTEM_PROMPT

    assert "禁止强行建立市场关联" in SYSTEM_PROMPT
    for banned_phrase in (
        "可能影响市场风险偏好",
        "可能提升科技股风险溢价",
        "可能影响投资者情绪",
        "值得关注后续市场影响",
        "可能加剧市场不确定性",
        "可能对科技板块产生影响",
    ):
        assert banned_phrase in SYSTEM_PROMPT


def test_stage_b_prompt_requires_verifiable_selection_reason_format():
    from src.news_prompt import SYSTEM_PROMPT

    assert "selection_reason 必须是可验证的具体事实依据，并以所属主线开头" in SYSTEM_PROMPT
    for example in ("美国宏观：关键通胀数据", "大型科技：NVIDIA 财报", "大型科技：Amazon AI资本开支",
                    "市场影响：关税可能直接推高美国消费品价格"):
        assert example in SYSTEM_PROMPT


def test_stage_b_prompt_concentrates_by_company_and_incremental_information_without_hard_quota():
    from src.news_prompt import SYSTEM_PROMPT

    fixture = json.load(open(CONTRACT_FIXTURE, encoding="utf-8"))
    assert "higher-information" in fixture["expected_contract"]["C"]
    for phrase in (
        "同一公司或高度相邻主题",
        "信息增量更高",
        "只有当两条事件彼此独立且都达到明显高重要性时",
        "不做硬 company quota",
    ):
        assert phrase in SYSTEM_PROMPT
