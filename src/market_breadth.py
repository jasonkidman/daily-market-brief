"""Daily S&P 500 constituent and sector breadth calculations."""

from __future__ import annotations

from datetime import date, datetime
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from .constituents import load_constituents
from .market_health import calculate_market_health


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _default_downloader(tickers: list[str], start: date, end: date):
    import yfinance as yf

    return yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _rows_from_download(frame, ticker: str, batch_size: int) -> list[dict]:
    if frame is None or frame.empty or "Close" not in frame:
        return []
    close = frame["Close"]
    if hasattr(close, "columns"):
        if ticker not in close.columns:
            return []
        close = close[ticker]
    rows = []
    for index, value in close.items():
        try:
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                continue
            rows.append({"date": _as_date(index).isoformat(), "close": number})
        except (TypeError, ValueError):
            continue
    return rows


def fetch_batched_close_rows(tickers: Iterable[str], start_date: date, end_date: date,
                             batch_size: int, downloader: Callable = None) -> dict[str, list[dict]]:
    """Download daily prices in bounded yfinance batches and return Close rows by ticker."""
    tickers = list(dict.fromkeys(ticker for ticker in tickers if ticker))
    download = downloader or _default_downloader
    fetched = {}
    for batch in _chunks(tickers, batch_size):
        frame = download(batch, start_date, end_date)
        for ticker in batch:
            fetched[ticker] = _rows_from_download(frame, ticker, len(batch))
    return fetched


def _return_on_target(rows: Iterable[dict], target: date) -> float | None:
    cleaned = []
    for row in rows:
        try:
            row_date = _as_date(row["date"])
            close = float(row["close"])
            if math.isfinite(close) and close > 0:
                cleaned.append((row_date, close))
        except (KeyError, TypeError, ValueError):
            continue
    cleaned.sort()
    target_close = next((close for row_date, close in cleaned if row_date == target), None)
    previous = [close for row_date, close in cleaned if row_date < target]
    if target_close is None or not previous:
        return None
    return target_close / previous[-1] - 1


def _coverage_status(ratio: float, config: dict) -> str:
    if ratio >= float(config["normal_coverage"]):
        return "ok"
    if ratio >= float(config["minimum_coverage"]):
        return "partial"
    return "invalid"


def calculate_stock_breadth(constituents: Iterable[dict], close_rows_by_ticker: dict[str, list],
                            target_market_date: str, config: dict) -> dict:
    """Calculate S&P 500 daily breadth strictly on the supplied market date."""
    target = _as_date(target_market_date)
    constituents = list(constituents)
    returns = []
    for constituent in constituents:
        ticker = constituent.get("yahoo_ticker")
        daily_return = _return_on_target(close_rows_by_ticker.get(ticker, []), target)
        if daily_return is not None:
            returns.append(daily_return)
    valid_count = len(returns)
    total = len(constituents)
    advancers = sum(value > 0 for value in returns)
    decliners = sum(value < 0 for value in returns)
    unchanged = sum(value == 0 for value in returns)
    coverage = valid_count / total if total else 0.0
    denominator = valid_count or None
    dates = {item.get("as_of_date") for item in constituents if item.get("as_of_date")}
    return {
        "market_date": target.isoformat(),
        "constituent_date": dates.pop() if len(dates) == 1 else None,
        "total_constituents": total,
        "valid_count": valid_count,
        "invalid_count": total - valid_count,
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "advance_ratio": advancers / denominator if denominator else None,
        "decline_ratio": decliners / denominator if denominator else None,
        "unchanged_ratio": unchanged / denominator if denominator else None,
        "coverage_ratio": coverage,
        "status": _coverage_status(coverage, config),
    }


def calculate_sector_breadth(sector_config: dict[str, dict], close_rows_by_ticker: dict[str, list],
                             target_market_date: str, full_scale: float = 0.03) -> dict:
    """Calculate daily returns and breadth for the eleven S&P sector ETFs."""
    target = _as_date(target_market_date)
    items = []
    for key, sector in sector_config.items():
        ticker = sector["ticker"]
        daily_return = _return_on_target(close_rows_by_ticker.get(ticker, []), target)
        if daily_return is None:
            direction = None
            bar_strength = None
        elif daily_return > 0:
            direction = "up"
            bar_strength = min(daily_return / full_scale, 1.0)
        elif daily_return < 0:
            direction = "down"
            bar_strength = min(abs(daily_return) / full_scale, 1.0)
        else:
            direction = "flat"
            bar_strength = 0.0
        items.append({
            "key": key,
            "name": sector["name"],
            "ticker": ticker,
            "valid": daily_return is not None,
            "daily_return": daily_return,
            "direction": direction,
            "bar_strength": bar_strength,
        })

    items.sort(key=lambda item: (not item["valid"], -(item["daily_return"] or 0)))
    valid_items = [item for item in items if item["valid"]]
    valid_count = len(valid_items)
    advancers = sum(item["direction"] == "up" for item in valid_items)
    decliners = sum(item["direction"] == "down" for item in valid_items)
    unchanged = sum(item["direction"] == "flat" for item in valid_items)
    return {
        "market_date": target.isoformat(),
        "items": items,
        "valid_count": valid_count,
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "advance_ratio": advancers / valid_count if valid_count else None,
    }


def unavailable_market_breadth(target_market_date: str | None = None) -> dict:
    """Shape-preserving degraded payload; intentionally independent of drawdown validity."""
    return {
        "market_date": target_market_date,
        "stocks": {
            "market_date": target_market_date, "constituent_date": None,
            "total_constituents": 0, "valid_count": 0, "invalid_count": 0,
            "advancers": 0, "decliners": 0, "unchanged": 0,
            "advance_ratio": None, "decline_ratio": None, "unchanged_ratio": None,
            "coverage_ratio": 0.0, "status": "invalid",
        },
        "sectors": {
            "market_date": target_market_date, "items": [], "valid_count": 0,
            "advancers": 0, "decliners": 0, "unchanged": 0, "advance_ratio": None,
        },
        "health": {
            "valid": False, "score": None, "level": "unavailable", "label": "数据不足",
            "divergence": None, "summary": "市场宽度数据暂不可用。",
        },
    }


def build_market_breadth(reference_path: Path, config: dict, target_market_date: str,
                         sp500_daily_return: float | None, downloader: Callable = None) -> dict:
    """Fetch daily breadth from the local reference with target-date-aligned closes only."""
    target = _as_date(target_market_date)
    constituent_config = config["constituents"]
    lookback_days = int(constituent_config["lookback_calendar_days"])
    start_date = date.fromordinal(target.toordinal() - lookback_days)
    end_date = date.fromordinal(target.toordinal() + 1)
    constituents = load_constituents(Path(reference_path))
    stock_rows = fetch_batched_close_rows(
        [item["yahoo_ticker"] for item in constituents], start_date, end_date,
        int(constituent_config["batch_size"]), downloader,
    )
    stocks = calculate_stock_breadth(constituents, stock_rows, target.isoformat(), constituent_config)
    sectors = calculate_sector_breadth(
        config["sectors"],
        fetch_batched_close_rows(
            [sector["ticker"] for sector in config["sectors"].values()], start_date, end_date,
            int(constituent_config["batch_size"]), downloader,
        ),
        target.isoformat(),
        float(config["visual"]["sector_full_scale"]),
    )
    health = calculate_market_health(stocks, sectors, sp500_daily_return, config)
    return {"market_date": target.isoformat(), "stocks": stocks, "sectors": sectors, "health": health}


def build_offline_market_breadth(config: dict, target_market_date: str,
                                 sp500_daily_return: float | None) -> dict:
    """Small deterministic fixture: ten stocks and all eleven sector ETFs, never live data."""
    target = _as_date(target_market_date)
    previous = date.fromordinal(target.toordinal() - 1).isoformat()
    constituents = [
        {"source_ticker": f"FIX{index}", "yahoo_ticker": f"FIX{index}",
         "name": f"Fixture {index}", "as_of_date": "2026-08-10"}
        for index in range(10)
    ]
    stock_rows = {
        item["yahoo_ticker"]: [
            {"date": previous, "close": 100.0},
            {"date": target.isoformat(), "close": 101.0 if index < 6 else 99.0 if index < 9 else 100.0},
        ]
        for index, item in enumerate(constituents)
    }
    sector_rows = {}
    for index, sector in enumerate(config["sectors"].values()):
        sector_rows[sector["ticker"]] = [
            {"date": previous, "close": 100.0},
            {"date": target.isoformat(), "close": 101.0 if index < 7 else 99.0},
        ]
    stocks = calculate_stock_breadth(constituents, stock_rows, target.isoformat(), config["constituents"])
    sectors = calculate_sector_breadth(
        config["sectors"], sector_rows, target.isoformat(), float(config["visual"]["sector_full_scale"])
    )
    health = calculate_market_health(stocks, sectors, sp500_daily_return, config)
    return {"market_date": target.isoformat(), "stocks": stocks, "sectors": sectors, "health": health}
