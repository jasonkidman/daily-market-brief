"""Build the human-review news candidate pool for the "more news" drawer.

This is a read-only view over data the pipeline already produces: it does not
issue new RSS fetches or model calls, and it does not change which articles
Stage A/Stage B select. See `build_news_candidates` for the exact source stage.
"""

from __future__ import annotations


CATEGORY_OTHER = "其他"
CATEGORY_ORDER = (
    "宏观 / 利率",
    "大型科技",
    "AI / 科技",
    "地缘政治与风险事件",
    "能源 / 大宗商品",
    CATEGORY_OTHER,
)

# Stage B assigns one of these (finer-grained) categories only to the articles
# it actually selects; bucket them down to the six display groups above.
_CATEGORY_BY_ALLOWED_CATEGORY = {
    "美联储 / 利率": "宏观 / 利率",
    "就业 / 通胀": "宏观 / 利率",
    "美国经济": "宏观 / 利率",
    "美债 / 美元": "宏观 / 利率",
    "金融市场": "宏观 / 利率",
    "大型科技": "大型科技",
    "AI / 资本开支": "AI / 科技",
    "半导体": "AI / 科技",
    "地缘政治": "地缘政治与风险事件",
    "政策 / 监管": "地缘政治与风险事件",
}

# Unselected candidates only carry Stage A's topic_group (no AI category was
# ever assigned to them), so they are bucketed through this coarser table.
_CATEGORY_BY_TOPIC_GROUP = {
    "US_MARKET_MACRO": "宏观 / 利率",
    "MEGA_CAP_TECH": "大型科技",
    "AI_CHIPS": "AI / 科技",
    "GEOPOLITICS": "地缘政治与风险事件",
    "ENERGY_COMMODITIES": "能源 / 大宗商品",
    "CORPORATE_EARNINGS": CATEGORY_OTHER,
    "OTHER_SYSTEMIC": CATEGORY_OTHER,
}


def _bucket_category(selected_category: str | None, topic_group: str | None) -> str:
    if selected_category:
        return _CATEGORY_BY_ALLOWED_CATEGORY.get(selected_category, CATEGORY_OTHER)
    return _CATEGORY_BY_TOPIC_GROUP.get(topic_group, CATEGORY_OTHER)


def build_news_candidates(selection_candidates: list[dict], selected_news: list[dict],
                          translations: dict[str, dict] | None = None) -> list[dict]:
    """Flatten the Stage B input pool into the review-drawer candidate list.

    `selection_candidates` is the same pool persisted as the Stage B snapshot's
    `stage_b.candidates` (one representative article per Stage A event, i.e.
    already deduplicated and event-clustered). Each item is flagged `selected`
    based on whether its candidate_id survived into the final, post-topic-cap
    `selected_news` list that is shown on the page.

    Chinese title/summary are resolved with no new translation for anything Stage
    B already translated: selected candidates reuse Stage B's own title_zh/
    summary_zh outright. `translations` (candidate_id -> {"title_zh", "summary_zh"})
    covers only candidates Stage B never selected; any candidate_id absent from it
    (translation skipped, failed, or never attempted) falls back to the original
    English title/summary, which are always kept alongside for reference.
    """
    translations = translations or {}
    selected_by_id = {item["candidate_id"]: item for item in selected_news}
    candidates = []
    for item in selection_candidates:
        candidate_id = item.get("candidate_id")
        selected_item = selected_by_id.get(candidate_id)
        title = item.get("title", "")
        summary = item.get("summary", "")
        if selected_item is not None:
            title_zh = selected_item.get("title_zh") or title
            summary_zh = selected_item.get("summary_zh") or summary
        else:
            translated = translations.get(candidate_id, {})
            title_zh = translated.get("title_zh") or title
            summary_zh = translated.get("summary_zh") or summary
        candidates.append({
            "candidate_id": candidate_id,
            "title": title,
            "title_zh": title_zh,
            "source": item.get("source", ""),
            "published_at": item.get("published_at", ""),
            "summary": summary,
            "summary_zh": summary_zh,
            "url": item.get("url", ""),
            "category": _bucket_category(
                selected_item.get("category") if selected_item else None,
                item.get("topic_group"),
            ),
            "selected": selected_item is not None,
        })
    return candidates
