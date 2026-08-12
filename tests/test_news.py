from datetime import datetime
import os
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.deepseek_client import NewsSelectionError, select_news, validate_selection
from src.news_dedupe import dedupe_candidates
from src.rss_news import fetch_candidates


def candidate(cid, title, url, source="BBC News", summary="full summary", priority="P0"):
    return {"candidate_id": cid, "title": title, "url": url, "source": source,
            "summary": summary, "published_at": "2026-08-12T00:00:00+00:00", "priority": priority}


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


def test_rejects_invalid_candidate_category_duplicates_and_more_than_eight():
    pool = [candidate(str(i), f"title {i}", f"https://x/{i}") for i in range(9)]
    base = {"rank": 1, "candidate_id": "0", "category": "市场 / 宏观", "title_zh": "标题", "summary_zh": "摘要"}
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**base, "candidate_id": "missing"}]}, pool)
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**base, "category": "体育"}]}, pool)
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [base, {**base, "rank": 2}]}, pool)
    with pytest.raises(NewsSelectionError):
        validate_selection({"news": [{**base, "rank": i + 1, "candidate_id": str(i)} for i in range(9)]}, pool)


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


def test_three_ai_failures_degrade_without_raising():
    attempts = []
    sleeps = []

    def failing(*args, **kwargs):
        attempts.append(1)
        return "not json"

    news, warning = select_news([candidate("1", "Title", "https://x/1")], "key", call_model=failing,
                                sleep_fn=sleeps.append)
    assert news == []
    assert "新闻 AI 处理暂时失败" in warning
    assert len(attempts) == 3
    assert sleeps == [5, 10]
