"""Deterministic first-pass news deduplication."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


PRIORITY = {"P0": 0, "P1": 1, "P2": 2}
TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith(TRACKING_PREFIXES)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title.casefold())


def _quality(item: dict) -> tuple:
    return (-PRIORITY.get(item.get("priority", "P2"), 2), len(item.get("summary", "")))


def _duplicate(a: dict, b: dict) -> bool:
    if canonical_url(a["url"]) == canonical_url(b["url"]):
        return True
    title_a, title_b = normalize_title(a["title"]), normalize_title(b["title"])
    if title_a == title_b:
        return True
    return SequenceMatcher(None, title_a, title_b).ratio() >= 0.90


def dedupe_candidates(candidates: list[dict], recent_selected: list[dict] = None) -> list[dict]:
    recent_selected = recent_selected or []
    exact_recent_urls = {canonical_url(item["url"]) for item in recent_selected if item.get("url")}
    exact_recent_titles = {normalize_title(item["title"]) for item in recent_selected if item.get("title")}
    kept = []
    for candidate in candidates:
        if canonical_url(candidate["url"]) in exact_recent_urls or normalize_title(candidate["title"]) in exact_recent_titles:
            continue
        match_index = next((i for i, existing in enumerate(kept) if _duplicate(candidate, existing)), None)
        if match_index is None:
            kept.append(candidate)
        elif _quality(candidate) > _quality(kept[match_index]):
            kept[match_index] = candidate
    return kept
