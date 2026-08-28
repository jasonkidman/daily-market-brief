"""Market history retrieval, validation, and close-based calculations."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any, Iterable


class MarketDataError(ValueError):
    """Raised when market history cannot safely support calculations."""


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _clean_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for row in rows:
        try:
            close = float(row["close"])
            row_date = _as_date(row["date"])
        except (KeyError, TypeError, ValueError):
            raise MarketDataError("行情记录缺少有效日期或收盘价。")
        if not math.isfinite(close) or close <= 0:
            raise MarketDataError("Close 必须为大于零的有限数值。")
        cleaned.append({"date": row_date, "close": close})
    cleaned.sort(key=lambda item: item["date"])
    return cleaned


def validate_close_rows(rows: Iterable[dict[str, Any]], now: datetime) -> dict[str, Any]:
    try:
        cleaned = _clean_rows(rows)
        if len(cleaned) < 2:
            raise MarketDataError("至少需要两个有效 Close。")
        today = now.date()
        latest = cleaned[-1]["date"]
        if latest > today:
            raise MarketDataError("当前交易日不得使用未来日期。")
        if latest < today - timedelta(days=10):
            raise MarketDataError("最新行情日期超出合理范围。")
        ath = max(item["close"] for item in cleaned)
        if ath <= 0 or ath < cleaned[-1]["close"]:
            raise MarketDataError("ATH 校验失败。")
        return {"valid": True, "error": None, "rows": cleaned}
    except MarketDataError as exc:
        return {"valid": False, "error": str(exc), "rows": []}


def calculate_market_snapshot(rows: Iterable[dict[str, Any]], now: datetime) -> dict[str, Any]:
    validation = validate_close_rows(rows, now)
    if not validation["valid"]:
        raise MarketDataError(validation["error"])
    cleaned = validation["rows"]
    latest, previous = cleaned[-1], cleaned[-2]
    previous_year = latest["date"].year - 1
    prior_year_rows = [item for item in cleaned if item["date"].year == previous_year]
    if not prior_year_rows:
        raise MarketDataError("缺少上一自然年度最后一个有效交易日 Close。")
    year_end = prior_year_rows[-1]
    ath_row = max(cleaned, key=lambda item: item["close"])
    return {
        "market_date": latest["date"].isoformat(),
        "close": latest["close"],
        "daily_return": latest["close"] / previous["close"] - 1,
        "ytd_return": latest["close"] / year_end["close"] - 1,
        "ath": ath_row["close"],
        "ath_date": ath_row["date"].isoformat(),
        "drawdown": 1 - latest["close"] / ath_row["close"],
    }


def calculate_context_snapshot(rows: Iterable[dict[str, Any]], now: datetime,
                               is_yield: bool = False) -> dict[str, Any]:
    """Calculate a compact context snapshot without requiring YTD history."""
    validation = validate_close_rows(rows, now)
    if not validation["valid"]:
        raise MarketDataError(validation["error"])
    latest, previous = validation["rows"][-1], validation["rows"][-2]
    snapshot = {
        "market_date": latest["date"].isoformat(),
        "close": latest["close"],
    }
    if is_yield:
        snapshot["yield_change_bp"] = (latest["close"] - previous["close"]) * 100
    else:
        snapshot["daily_return"] = latest["close"] / previous["close"] - 1
    return snapshot


def build_sparkline(history: Iterable[dict[str, Any]], points: int = 30,
                     width: int = 420, height: int = 52) -> dict[str, str] | None:
    """Build SVG path `d` strings for a sparkline from real historical closes.

    Uses the same row-cleaning rules as the rest of the market pipeline (sorted,
    de-duplicated, validated closes) so the sparkline never plots fabricated data.
    Returns None when fewer than 2 usable points are available.
    """
    try:
        cleaned = _clean_rows(history)
    except MarketDataError:
        return None
    if len(cleaned) < 2:
        return None
    recent = cleaned[-points:]
    closes = [item["close"] for item in recent]
    count = len(closes)
    low, high = min(closes), max(closes)
    span = high - low
    top_pad, bottom_pad = 4, 4
    plot_height = max(height - top_pad - bottom_pad, 1)

    def _x(index: int) -> float:
        if count == 1:
            return 0.0
        return round(index * width / (count - 1), 2)

    def _y(close: float) -> float:
        if span <= 0:
            return round(top_pad + plot_height / 2, 2)
        return round(top_pad + (1 - (close - low) / span) * plot_height, 2)

    coords = [(_x(i), _y(close)) for i, close in enumerate(closes)]
    line = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(coords))
    area = f"{line} L{coords[-1][0]},{height} L{coords[0][0]},{height} Z"
    return {"line": line, "area": area}


def fetch_close_history(ticker: str, period: str = "max") -> list[dict[str, Any]]:
    """Fetch daily close history for one ticker via yfinance.

    `period` controls how far back the request goes. Core indices need the
    full history (ATH-since-inception, YTD-vs-prior-year-end), so they must
    use "max". Context indicators only ever read the latest and previous
    close (see `calculate_context_snapshot`), so they should use a short,
    bounded period instead of "max": some tickers' full history legitimately
    contains an old, real, unusual print (e.g. WTI crude futures traded at
    -$37.63 on 2020-04-20, the widely reported real-world negative oil price
    event) that `validate_close_rows` correctly rejects as a bad *current*
    close but that is irrelevant noise for a same-day snapshot. Fetching a
    bounded window avoids failing today's snapshot over years-old data.
    """
    import yfinance as yf

    frame = yf.Ticker(ticker).history(period=period, auto_adjust=False, actions=False)
    if frame is None or frame.empty or "Close" not in frame:
        raise MarketDataError(f"{ticker} 未返回有效历史 Close。")
    rows = []
    for index, value in frame["Close"].items():
        if value is None:
            continue
        rows.append({"date": index.date().isoformat(), "close": float(value)})
    return rows


def fetch_market(config: dict[str, Any], now: datetime) -> tuple[dict[str, Any], dict[str, list], list[str]]:
    snapshots, histories, warnings = {}, {}, []
    for key, item in config.items():
        try:
            history = fetch_close_history(item["ticker"])
            snapshot = calculate_market_snapshot(history, now)
            snapshot.update({
                "name": item["name"], "ticker": item["ticker"], "valid": True,
                "sparkline": build_sparkline(history),
            })
            snapshots[key], histories[key] = snapshot, history
        except Exception as exc:
            warnings.append(f"{item['name']} 行情获取或校验失败：{exc}")
            snapshots[key] = {"name": item["name"], "ticker": item["ticker"], "valid": False, "error": str(exc)}
    return snapshots, histories, warnings


def fetch_market_context(config: dict[str, Any], now: datetime) -> tuple[dict[str, Any], list[str]]:
    """Fetch each context indicator independently so one failure cannot block peers.

    Uses a bounded 3-month lookback rather than "max": context snapshots only
    need the latest and previous close (see `calculate_context_snapshot`), and
    a short window avoids failing today's value over an unrelated, real, old
    print elsewhere in a ticker's full history (see `fetch_close_history`).
    """
    snapshots, warnings = {}, []
    for key, item in config.items():
        try:
            history = fetch_close_history(item["ticker"], period="3mo")
            snapshot = calculate_context_snapshot(history, now, is_yield=(key == "us10y"))
            snapshot.update({"name": item["name"], "ticker": item["ticker"], "valid": True})
            snapshots[key] = snapshot
        except Exception as exc:
            warnings.append(f"{item['name']} 行情获取或校验失败：{exc}")
            snapshots[key] = {
                "name": item["name"], "ticker": item["ticker"], "valid": False, "error": str(exc)
            }
    return snapshots, warnings
