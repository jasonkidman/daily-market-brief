"""Fault-isolated RSS candidate acquisition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import calendar
import hashlib
import html
import re
from typing import Any, Optional

import feedparser

from .news_dedupe import canonical_url


RSS_USER_AGENT = "daily-market-brief/1.0 (github.com/jasonkidman/daily-market-brief)"


def _parse_feed(url: str):
    return feedparser.parse(url, agent=RSS_USER_AGENT)


def _entry_datetime(entry: Any) -> Optional[datetime]:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def fetch_candidates(sources: list[dict], now: datetime, hours: int = 30,
                     parser=_parse_feed) -> tuple[list[dict], list[str]]:
    now_utc = now.astimezone(timezone.utc)
    cutoff = now_utc - timedelta(hours=hours)
    candidates, warnings = [], []
    for source in sources:
        if source.get("enabled", True) is False:
            continue
        try:
            feed = parser(source["url"])
            if getattr(feed, "bozo", False) and not getattr(feed, "entries", []):
                raise RuntimeError(str(getattr(feed, "bozo_exception", "RSS 解析失败")))
            for entry in getattr(feed, "entries", []):
                published = _entry_datetime(entry)
                if published is None or published < cutoff or published > now_utc + timedelta(minutes=5):
                    continue
                title = _plain_text(getattr(entry, "title", ""))
                url = getattr(entry, "link", "").strip()
                if not title or not url:
                    continue
                summary = _plain_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))
                identity = canonical_url(url) or title
                candidate = {
                    "candidate_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                    "source": source["name"],
                    "title": title,
                    "summary": summary,
                    "published_at": published.isoformat(),
                    "url": url,
                    "priority": source["priority"],
                }
                if source.get("category_hint"):
                    candidate["category_hint"] = source["category_hint"]
                candidates.append(candidate)
        except Exception as exc:
            warnings.append(f"{source['name']} RSS 获取失败：{exc}")
    candidates.sort(key=lambda item: item["published_at"], reverse=True)
    return candidates, warnings
