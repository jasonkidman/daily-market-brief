"""Build the human-review news candidate pool for the "more news" drawer.

This is a read-only view over data the pipeline already produces: it does not
issue new RSS fetches or model calls, and it does not change which articles
Layer 1 scoring rates highest. See `build_news_candidates` for the exact
source stage.
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

# Scoring assigns one of these (finer-grained) categories to every candidate;
# bucket them down to the six display groups above.
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
    # Scoring is instructed to only use "政策 / 监管" for system-wide/macro
    # regulatory changes (financial stability, market-structure/trading rules,
    # broad multi-company impact, or major macro/trade/Fed/Treasury policy) --
    # a regulatory event specific to one of the tracked mega-cap companies is
    # instructed to use "大型科技" instead (see news_score_prompt.py). So
    # "政策 / 监管" belongs with the macro bucket, not geopolitics; it was
    # previously mismapped to "地缘政治与风险事件", which made routine US
    # financial-market/regulatory news display as a geopolitical risk event.
    "政策 / 监管": "宏观 / 利率",
}

# A candidate scoring never returned a result for (dropped as invalid, or its
# batch failed/was skipped -- see build_news_candidates) only carries the
# clustering step's topic_group, no scoring category, so it is bucketed
# through this coarser table instead.
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


def build_news_candidates(scored_candidates: list[dict], selected_news: list[dict]) -> list[dict]:
    """Flatten the scored candidate pool into the review-drawer candidate list.

    `scored_candidates` is the full event-clustered pool (event_selection_candidates'
    output) with each item's Layer 1 scoring result merged in -- every candidate
    already carries its own title_zh/summary_zh/category/score/reason from
    score_candidates(), since Layer 1 scores (and translates) the whole pool, not
    just what ends up selected. A candidate score_candidates() never returned a
    result for (dropped as invalid, or its batch failed/was skipped) has score=None
    and no category; its title_zh/summary_zh fall back to the original English.

    Each item is flagged `selected` based on whether its candidate_id survived
    into the final, ranked `selected_news` list shown as "今日重要新闻".
    """
    selected_ids = {item["candidate_id"] for item in selected_news}
    candidates = []
    for item in scored_candidates:
        candidate_id = item.get("candidate_id")
        title = item.get("title", "")
        summary = item.get("summary", "")
        candidates.append({
            "candidate_id": candidate_id,
            "title": title,
            "title_zh": item.get("title_zh") or title,
            "source": item.get("source", ""),
            "published_at": item.get("published_at", ""),
            "summary": summary,
            "summary_zh": item.get("summary_zh") or summary,
            "url": item.get("url", ""),
            "category": _bucket_category(item.get("category"), item.get("topic_group")),
            "score": item.get("score"),
            "reason": item.get("reason", ""),
            "selected": candidate_id in selected_ids,
        })
    return candidates
