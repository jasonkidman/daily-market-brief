"""Fault-isolated RSS candidate acquisition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import calendar
import hashlib
import html
import re
import time
from typing import Any, Optional

import feedparser

from .news_dedupe import canonical_url


RSS_USER_AGENT = "daily-market-brief/1.0 (github.com/jasonkidman/daily-market-brief)"
FINAL_NEWS_WINDOW_HOURS = 24

# RSS fetches are read-only and idempotent, and real-world observation shows
# transient TLS/network drops (e.g. SSLEOFError mid-handshake) that succeed
# on a bare retry -- not certificate or configuration problems (the same
# client config succeeds on most attempts against the same host). Retrying a
# few times with a short backoff before recording a failure is safe and
# resolves most of these without masking a genuinely broken source.
RSS_FETCH_MAX_ATTEMPTS = 3
RSS_FETCH_RETRY_DELAY_SECONDS = 0.5


def _parse_feed(url: str):
    return feedparser.parse(url, agent=RSS_USER_AGENT)


def _entry_datetime(entry: Any) -> Optional[datetime]:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def _log_candidate(candidate: dict, stage: str, action: str, reason: str = "") -> None:
    suffix = f" | reason={reason}" if reason else ""
    print(
        f"[NEWS CANDIDATE] candidate_id={candidate.get('candidate_id', '')} "
        f"| title={candidate.get('title', '')} | source={candidate.get('source', '')} "
        f"| published_at={candidate.get('published_at', '')} | stage={stage} "
        f"| action={action}{suffix}"
    )


def _fetch_feed_with_retry(parser, url: str, max_attempts: int, retry_delay_seconds: float,
                           sleep=time.sleep):
    """Call `parser(url)` with a bounded retry for transient fetch failures.

    Raises the last exception (or a RuntimeError built from a bozo
    exception with no entries) if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            feed = parser(url)
            if getattr(feed, "bozo", False) and not getattr(feed, "entries", []):
                raise RuntimeError(str(getattr(feed, "bozo_exception", "RSS 解析失败")))
            return feed
        except Exception as exc:  # noqa: BLE001 - any fetch/parse failure is retry-eligible
            last_exc = exc
            if attempt < max_attempts:
                sleep(retry_delay_seconds * attempt)
    raise last_exc


def fetch_candidates(sources: list[dict], now: datetime, hours: int = 30,
                     parser=_parse_feed, max_attempts: int = RSS_FETCH_MAX_ATTEMPTS,
                     retry_delay_seconds: float = RSS_FETCH_RETRY_DELAY_SECONDS,
                     sleep=time.sleep) -> tuple[list[dict], list[str]]:
    now_utc = now.astimezone(timezone.utc)
    cutoff = now_utc - timedelta(hours=hours)
    candidates, warnings = [], []
    for source in sources:
        if source.get("enabled", True) is False:
            continue
        raw_count = 0
        accepted_count = 0
        source_warning = "<none>"
        try:
            feed = _fetch_feed_with_retry(parser, source["url"], max_attempts, retry_delay_seconds, sleep)
            entries = getattr(feed, "entries", [])
            raw_count = len(entries)
            for entry in entries:
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
                if source.get("source_channel"):
                    candidate["source_channel"] = source["source_channel"]
                candidates.append(candidate)
                accepted_count += 1
                _log_candidate(candidate, "rss_fetch", "received")
        except Exception as exc:
            source_warning = str(exc)
            warnings.append(f"{source['name']} RSS 获取失败：{exc}")
        print(
            f"[NEWS RSS SOURCE] source={source['name']} raw={raw_count} "
            f"accepted={accepted_count} warning={source_warning}"
        )
    candidates.sort(key=lambda item: item["published_at"], reverse=True)
    return candidates, warnings


def filter_final_candidates(candidates: list[dict], now: datetime) -> list[dict]:
    """Keep candidates published in the inclusive final 24-hour window.

    RSS acquisition intentionally uses a 30-hour buffer. This second pass is
    the strict eligibility boundary before event clustering and selection.
    Missing, malformed, and future timestamps are not eligible.
    """
    now_utc = now.astimezone(timezone.utc)
    cutoff = now_utc - timedelta(hours=FINAL_NEWS_WINDOW_HOURS)
    eligible = []
    for candidate in candidates:
        raw_published_at = candidate.get("published_at")
        if not isinstance(raw_published_at, str) or not raw_published_at.strip():
            _log_candidate(candidate, "24h_filter", "drop", "missing_timestamp")
            continue
        try:
            published_at = datetime.fromisoformat(raw_published_at.strip().replace("Z", "+00:00"))
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            published_at = published_at.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            _log_candidate(candidate, "24h_filter", "drop", "invalid_timestamp")
            continue
        if cutoff <= published_at <= now_utc:
            eligible.append(candidate)
            _log_candidate(candidate, "24h_filter", "keep")
        elif published_at > now_utc:
            _log_candidate(candidate, "24h_filter", "drop", "future_timestamp")
        else:
            _log_candidate(candidate, "24h_filter", "drop", "too_old")
    return eligible
