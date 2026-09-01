from src.news_candidates import CATEGORY_ORDER, build_news_candidates


def selection_candidate(candidate_id, topic_group, title="Title", summary="Summary",
                        source="BBC News", url=None, published_at="2026-08-28T09:00:00+00:00"):
    return {
        "candidate_id": candidate_id,
        "title": title,
        "summary": summary,
        "source": source,
        "url": url or f"https://example.com/{candidate_id}",
        "published_at": published_at,
        "topic_group": topic_group,
    }


def selected_news_item(candidate_id, category, title_zh="中文标题", summary_zh="中文摘要"):
    return {"candidate_id": candidate_id, "category": category, "title_zh": title_zh, "summary_zh": summary_zh}


def test_marks_selected_and_unselected_candidates_from_final_news_list():
    candidates = [
        selection_candidate("a", "US_MARKET_MACRO"),
        selection_candidate("b", "MEGA_CAP_TECH"),
    ]
    news = [selected_news_item("a", "美联储 / 利率")]

    result = build_news_candidates(candidates, news)

    assert {item["candidate_id"]: item["selected"] for item in result} == {"a": True, "b": False}


def test_selected_candidate_uses_stage_b_ai_category_bucketed_to_display_group():
    candidates = [selection_candidate("a", "OTHER_SYSTEMIC")]
    news = [selected_news_item("a", "半导体")]

    result = build_news_candidates(candidates, news)

    assert result[0]["category"] == "AI / 科技"


def test_unselected_candidate_falls_back_to_topic_group_bucket():
    candidates = [selection_candidate("a", "ENERGY_COMMODITIES")]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "能源 / 大宗商品"
    assert result[0]["selected"] is False


def test_unmapped_topic_group_falls_back_to_other_without_fabricating_a_category():
    candidates = [selection_candidate("a", None)]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "其他"


def test_every_bucket_is_a_known_display_category():
    topic_groups = [
        "US_MARKET_MACRO", "AI_CHIPS", "MEGA_CAP_TECH", "ENERGY_COMMODITIES",
        "GEOPOLITICS", "CORPORATE_EARNINGS", "OTHER_SYSTEMIC", "UNKNOWN_GROUP",
    ]
    candidates = [selection_candidate(str(i), tg) for i, tg in enumerate(topic_groups)]

    result = build_news_candidates(candidates, [])

    assert all(item["category"] in CATEGORY_ORDER for item in result)


def test_preserves_article_fields_needed_for_display():
    candidates = [selection_candidate(
        "a", "AI_CHIPS", title="Nvidia unveils chip", summary="A new chip.",
        source="TechCrunch", url="https://techcrunch.example/a",
        published_at="2026-08-28T01:00:00+00:00",
    )]

    result = build_news_candidates(candidates, [])

    assert result[0] == {
        "candidate_id": "a",
        "title": "Nvidia unveils chip",
        "title_zh": "Nvidia unveils chip",
        "source": "TechCrunch",
        "published_at": "2026-08-28T01:00:00+00:00",
        "summary": "A new chip.",
        "summary_zh": "A new chip.",
        "url": "https://techcrunch.example/a",
        "category": "AI / 科技",
        "selected": False,
    }


def test_selected_candidate_reuses_stage_b_chinese_title_and_summary_verbatim():
    candidates = [selection_candidate("a", "US_MARKET_MACRO", title="Fed holds rates", summary="Fed summary.")]
    news = [selected_news_item("a", "美联储 / 利率", title_zh="美联储维持利率不变", summary_zh="委员会维持政策利率不变。")]

    result = build_news_candidates(candidates, news)

    assert result[0]["title_zh"] == "美联储维持利率不变"
    assert result[0]["summary_zh"] == "委员会维持政策利率不变。"
    # Original English is kept alongside for troubleshooting, not shown by default.
    assert result[0]["title"] == "Fed holds rates"
    assert result[0]["summary"] == "Fed summary."


def test_unselected_candidate_uses_translation_map_when_available():
    candidates = [selection_candidate("a", "AI_CHIPS", title="Nvidia chip", summary="New chip.")]
    translations = {"a": {"title_zh": "英伟达推出新芯片", "summary_zh": "新款芯片发布。"}}

    result = build_news_candidates(candidates, [], translations)

    assert result[0]["title_zh"] == "英伟达推出新芯片"
    assert result[0]["summary_zh"] == "新款芯片发布。"
    assert result[0]["title"] == "Nvidia chip"


def test_unselected_candidate_falls_back_to_english_when_translation_missing():
    candidates = [selection_candidate("a", "AI_CHIPS", title="Nvidia chip", summary="New chip.")]

    result = build_news_candidates(candidates, [], translations={})

    assert result[0]["title_zh"] == "Nvidia chip"
    assert result[0]["summary_zh"] == "New chip."


def test_empty_pool_yields_empty_candidate_list():
    assert build_news_candidates([], []) == []
