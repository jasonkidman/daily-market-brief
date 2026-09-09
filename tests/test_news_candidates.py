from src.news_candidates import CATEGORY_ORDER, build_news_candidates


def scored_candidate(candidate_id, topic_group, category=None, title="Title", summary="Summary",
                     title_zh="中文标题", summary_zh="中文摘要", score=80, reason="",
                     source="BBC News", url=None, published_at="2026-08-28T09:00:00+00:00"):
    return {
        "candidate_id": candidate_id,
        "title": title,
        "title_zh": title_zh,
        "summary": summary,
        "summary_zh": summary_zh,
        "source": source,
        "url": url or f"https://example.com/{candidate_id}",
        "published_at": published_at,
        "topic_group": topic_group,
        "category": category,
        "score": score,
        "reason": reason,
    }


def test_marks_selected_and_unselected_candidates_from_final_news_list():
    candidates = [
        scored_candidate("a", "US_MARKET_MACRO", category="美联储 / 利率"),
        scored_candidate("b", "MEGA_CAP_TECH", category="大型科技"),
    ]
    news = [{"candidate_id": "a"}]

    result = build_news_candidates(candidates, news)

    assert {item["candidate_id"]: item["selected"] for item in result} == {"a": True, "b": False}


def test_scored_category_is_bucketed_to_display_group():
    candidates = [scored_candidate("a", "OTHER_SYSTEMIC", category="半导体")]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "AI / 科技"


def test_policy_regulation_category_maps_to_macro_not_geopolitics():
    """Regression: "政策 / 监管" (policy/regulation) was previously mismapped
    to the "地缘政治与风险事件" (geopolitics/risk) display bucket, making
    routine US financial-market/regulatory news display as a geopolitical
    risk event. Scoring is instructed to reserve "政策 / 监管" for system-wide
    macro/financial regulation (Fed/Treasury/market-structure rules), not
    geopolitics, so it belongs in the macro bucket."""
    candidates = [scored_candidate("a", "OTHER_SYSTEMIC", category="政策 / 监管")]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "宏观 / 利率"


def test_unscored_candidate_falls_back_to_topic_group_bucket():
    """A candidate score_candidates() has no result for (dropped, or its batch
    failed/was skipped) carries no scoring category at all -- news_candidates
    still buckets it via topic_group rather than leaving it uncategorized."""
    candidates = [scored_candidate("a", "ENERGY_COMMODITIES", category=None)]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "能源 / 大宗商品"
    assert result[0]["selected"] is False


def test_company_news_other_category_buckets_to_other():
    """Regression: before this category existed, Scoring had no legitimate
    bucket for single-company news outside the tracked mega-cap list (e.g.
    "礼来表现抗跌，波音交付数据喜忧参半"), so it got miscategorized as
    macro/rates content instead."""
    candidates = [scored_candidate("a", "OTHER_SYSTEMIC", category="公司新闻 / 其他")]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "其他"


def test_unmapped_topic_group_falls_back_to_other_without_fabricating_a_category():
    candidates = [scored_candidate("a", None, category=None)]

    result = build_news_candidates(candidates, [])

    assert result[0]["category"] == "其他"


def test_every_bucket_is_a_known_display_category():
    topic_groups = [
        "US_MARKET_MACRO", "AI_CHIPS", "MEGA_CAP_TECH", "ENERGY_COMMODITIES",
        "GEOPOLITICS", "CORPORATE_EARNINGS", "OTHER_SYSTEMIC", "UNKNOWN_GROUP",
    ]
    candidates = [scored_candidate(str(i), tg, category=None) for i, tg in enumerate(topic_groups)]

    result = build_news_candidates(candidates, [])

    assert all(item["category"] in CATEGORY_ORDER for item in result)


def test_preserves_article_fields_needed_for_display():
    candidates = [scored_candidate(
        "a", "AI_CHIPS", category=None, title="Nvidia unveils chip", summary="A new chip.",
        title_zh=None, summary_zh=None, score=None, reason="",
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
        "score": None,
        "reason": "",
        "selected": False,
    }


def test_scored_candidate_reuses_scoring_chinese_title_and_summary_verbatim():
    candidates = [scored_candidate(
        "a", "US_MARKET_MACRO", category="美联储 / 利率", title="Fed holds rates", summary="Fed summary.",
        title_zh="美联储维持利率不变", summary_zh="委员会维持政策利率不变。",
    )]
    news = [{"candidate_id": "a"}]

    result = build_news_candidates(candidates, news)

    assert result[0]["title_zh"] == "美联储维持利率不变"
    assert result[0]["summary_zh"] == "委员会维持政策利率不变。"
    # Original English is kept alongside for troubleshooting, not shown by default.
    assert result[0]["title"] == "Fed holds rates"
    assert result[0]["summary"] == "Fed summary."


def test_unscored_candidate_falls_back_to_english_title_and_summary():
    candidates = [scored_candidate(
        "a", "AI_CHIPS", category=None, title="Nvidia chip", summary="New chip.",
        title_zh=None, summary_zh=None, score=None,
    )]

    result = build_news_candidates(candidates, [])

    assert result[0]["title_zh"] == "Nvidia chip"
    assert result[0]["summary_zh"] == "New chip."


def test_empty_pool_yields_empty_candidate_list():
    assert build_news_candidates([], []) == []


def test_more_news_drawer_sorts_by_score_descending_unscored_last():
    """Regression: the "更多新闻" drawer used to preserve raw clustering/fetch
    order, so a 79-scored candidate and a 1-scored one appeared in arbitrary
    order within the same category -- reads as noise once the pool is large."""
    candidates = [
        scored_candidate("low", "OTHER_SYSTEMIC", category=None, score=5),
        scored_candidate("unscored", "OTHER_SYSTEMIC", category=None, score=None),
        scored_candidate("high", "OTHER_SYSTEMIC", category=None, score=79),
        scored_candidate("mid", "OTHER_SYSTEMIC", category=None, score=40),
    ]

    result = build_news_candidates(candidates, [])

    assert [item["candidate_id"] for item in result] == ["high", "mid", "low", "unscored"]
