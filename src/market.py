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


def fetch_close_history(ticker: str) -> list[dict[str, Any]]:
    """Fetch the complete daily close history for one ticker via yfinance."""
    import yfinance as yf

    frame = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=False)
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
            snapshot.update({"name": item["name"], "ticker": item["ticker"], "valid": True})
            snapshots[key], histories[key] = snapshot, history
        except Exception as exc:
            warnings.append(f"{item['name']} 行情获取或校验失败：{exc}")
            snapshots[key] = {"name": item["name"], "ticker": item["ticker"], "valid": False, "error": str(exc)}
    return snapshots, histories, warnings
