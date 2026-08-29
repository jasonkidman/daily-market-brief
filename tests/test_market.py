from datetime import datetime
from zoneinfo import ZoneInfo

import math
import pytest

from src.market import (
    MarketDataError,
    build_sparkline,
    calculate_context_snapshot,
    calculate_market_snapshot,
    fetch_close_history,
    fetch_market,
    fetch_market_context,
    validate_close_rows,
)


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def rows(*pairs):
    return [{"date": date, "close": close} for date, close in pairs]


def test_calculates_daily_ytd_and_closing_ath():
    result = calculate_market_snapshot(
        rows(
            ("2025-12-31", 100.0),
            ("2026-01-02", 102.0),
            ("2026-08-10", 119.0),
            ("2026-08-11", 120.0),
        ),
        NOW,
    )

    assert result["market_date"] == "2026-08-11"
    assert result["close"] == 120.0
    assert result["daily_return"] == pytest.approx(120 / 119 - 1)
    assert result["ytd_return"] == pytest.approx(0.20)
    assert result["ath"] == 120.0
    assert result["ath_date"] == "2026-08-11"


@pytest.mark.parametrize("bad_close", [math.nan, 0, -1])
def test_rejects_nan_and_non_positive_close(bad_close):
    result = validate_close_rows(
        rows(("2025-12-31", 100.0), ("2026-08-11", bad_close)), NOW
    )
    assert result["valid"] is False


def test_rejects_future_market_date():
    result = validate_close_rows(
        rows(("2026-08-11", 100.0), ("2026-08-13", 101.0)), NOW
    )
    assert result["valid"] is False


def test_requires_two_valid_closes():
    with pytest.raises(MarketDataError):
        calculate_market_snapshot(rows(("2026-08-11", 100.0)), NOW)


@pytest.mark.parametrize(
    ("previous", "latest", "expected"),
    [(3000, 3030, 0.01), (15, 16.5, 0.10), (100, 100.5, 0.005)],
)
def test_context_daily_return_calculation(previous, latest, expected):
    result = calculate_context_snapshot(
        rows(("2026-08-10", previous), ("2026-08-11", latest)), NOW
    )
    assert result["daily_return"] == pytest.approx(expected)
    assert "yield_change_bp" not in result


@pytest.mark.parametrize(
    ("previous", "latest", "expected_bp"),
    [(4.25, 4.32, 7), (4.32, 4.25, -7)],
)
def test_us10y_context_uses_basis_point_change(previous, latest, expected_bp):
    result = calculate_context_snapshot(
        rows(("2026-08-10", previous), ("2026-08-11", latest)), NOW, is_yield=True
    )
    assert result["yield_change_bp"] == pytest.approx(expected_bp)
    assert "daily_return" not in result


def test_one_context_fetch_failure_does_not_block_other_indicators(monkeypatch):
    def fake_fetch(ticker, period="max"):
        if ticker == "^VIX":
            raise RuntimeError("feed unavailable")
        return rows(("2026-08-10", 100), ("2026-08-11", 101))

    monkeypatch.setattr("src.market.fetch_close_history", fake_fetch)
    config = {
        "russell2000": {"name": "Russell 2000", "ticker": "^RUT"},
        "vix": {"name": "VIX", "ticker": "^VIX"},
        "dxy": {"name": "美元指数", "ticker": "DX-Y.NYB"},
        "us10y": {"name": "10Y 美债", "ticker": "^TNX"},
    }

    snapshots, warnings = fetch_market_context(config, NOW)

    assert snapshots["vix"]["valid"] is False
    assert all(snapshots[key]["valid"] for key in ("russell2000", "dxy", "us10y"))
    assert len(warnings) == 1
    assert "VIX" in warnings[0]


def test_context_fetch_uses_bounded_recent_period_not_full_history(monkeypatch):
    """Regression: context snapshots only need latest+previous close, so they must
    request a short window. Requesting "max" would fail today's WTI snapshot
    forever because CL=F's full history contains the real 2020-04-20 negative
    print, which is irrelevant to a same-day close/previous-close read."""
    captured_periods = []

    def fake_fetch(ticker, period="max"):
        captured_periods.append(period)
        return rows(("2026-08-10", 100), ("2026-08-11", 101))

    monkeypatch.setattr("src.market.fetch_close_history", fake_fetch)
    config = {"wti": {"name": "原油 (WTI)", "ticker": "CL=F"}}

    fetch_market_context(config, NOW)

    assert captured_periods == ["3mo"]


def test_core_fetch_still_uses_full_history_for_ath_and_ytd(monkeypatch):
    captured_periods = []

    def fake_fetch(ticker, period="max"):
        captured_periods.append(period)
        return rows(("2025-12-31", 100), ("2026-08-10", 110), ("2026-08-11", 111))

    monkeypatch.setattr("src.market.fetch_close_history", fake_fetch)
    config = {"sp500": {"name": "S&P 500", "ticker": "^GSPC"}}

    snapshots, histories, warnings = fetch_market(config, NOW)

    assert captured_periods == ["max"]
    assert snapshots["sp500"]["valid"] is True
    assert warnings == []


def test_old_out_of_window_negative_print_does_not_fail_current_context_snapshot():
    """The real WTI 2020-04-20 negative print, replayed as if it were still inside
    a fetched window, must still be correctly rejected by validation -- the fix
    is a narrower fetch window, not weaker validation of what is fetched."""
    result = validate_close_rows(
        rows(("2020-04-20", -37.63), ("2026-08-11", 83.53), ("2026-08-12", 82.55)), NOW
    )
    assert result["valid"] is False
    assert "大于零" in result["error"]


def test_build_sparkline_returns_none_with_fewer_than_two_points():
    assert build_sparkline([]) is None
    assert build_sparkline(rows(("2026-08-11", 100.0))) is None


def test_build_sparkline_returns_none_on_invalid_rows():
    assert build_sparkline([{"date": "2026-08-11", "close": -1}]) is None
    assert build_sparkline([{"close": 100.0}]) is None


def test_build_sparkline_produces_monotonic_path_for_monotonic_input():
    history = rows(
        ("2026-08-01", 100.0), ("2026-08-02", 105.0), ("2026-08-03", 110.0), ("2026-08-04", 120.0)
    )
    result = build_sparkline(history, width=420, height=52)

    assert result is not None
    assert result["line"].startswith("M0.0,")
    # Rising closes should map to strictly decreasing (higher-on-screen) y values.
    y_values = [float(part.split(",")[1]) for part in result["line"].replace("M", "L").split("L") if part]
    assert y_values == sorted(y_values, reverse=True)
    assert result["area"].startswith(result["line"])
    assert result["area"].endswith("Z")


def test_build_sparkline_handles_flat_price_series_without_dividing_by_zero():
    history = rows(("2026-08-01", 100.0), ("2026-08-02", 100.0), ("2026-08-03", 100.0))
    result = build_sparkline(history)

    assert result is not None
    y_values = [float(part.split(",")[1]) for part in result["line"].replace("M", "L").split("L") if part]
    assert len(set(y_values)) == 1


def test_build_sparkline_uses_only_the_most_recent_points():
    history = rows(*[(f"2026-07-{day:02d}", float(day)) for day in range(1, 31)])
    result = build_sparkline(history, points=5)

    assert result is not None
    coords = [part for part in result["line"].replace("M", "L").split("L") if part]
    assert len(coords) == 5


# --- fetch_close_history NaN handling (regression for the 2026-08-29 all-core-invalid incident) ---
#
# Real incident: yfinance served the most recent trading day's row with
# Close = NaN for S&P 500 / Nasdaq-100 / Dow / Russell 2000 / DXY / Gold / WTI
# simultaneously for several hours (Yahoo backend backfill lag, not a real
# market condition), while `value is None` never matches a pandas NaN, so the
# NaN flowed straight into `_clean_rows` and failed the *entire* ticker.

def _fake_close_frame(pairs):
    """Build a minimal object mimicking yf.Ticker(...).history()'s return:
    anything with a pandas-Series-shaped `frame["Close"]` supporting
    `.items()` and an `.index` with `.max()` / per-row `.date()`."""
    import pandas as pd

    dates = pd.to_datetime([date for date, _ in pairs])
    closes = [close for _, close in pairs]
    return pd.DataFrame({"Close": closes}, index=dates)


def _patch_yfinance(monkeypatch, frame):
    import yfinance

    class _FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period="max", auto_adjust=False, actions=False):
            return frame

    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)


def test_fetch_close_history_skips_nan_in_middle_row(monkeypatch):
    _patch_yfinance(monkeypatch, _fake_close_frame([
        ("2026-08-24", 100.0),
        ("2026-08-25", float("nan")),
        ("2026-08-26", 102.0),
        ("2026-08-27", 103.0),
    ]))

    result = fetch_close_history("^GSPC")

    assert [row["date"] for row in result] == ["2026-08-24", "2026-08-26", "2026-08-27"]
    assert [row["close"] for row in result] == [100.0, 102.0, 103.0]


def test_fetch_close_history_skips_inf_in_middle_row(monkeypatch):
    _patch_yfinance(monkeypatch, _fake_close_frame([
        ("2026-08-24", 100.0),
        ("2026-08-25", float("inf")),
        ("2026-08-26", 102.0),
    ]))

    result = fetch_close_history("^GSPC")

    assert [row["date"] for row in result] == ["2026-08-24", "2026-08-26"]


def test_fetch_close_history_fails_when_latest_row_is_nan(monkeypatch):
    _patch_yfinance(monkeypatch, _fake_close_frame([
        ("2026-08-24", 100.0),
        ("2026-08-25", 101.0),
        ("2026-08-26", float("nan")),
    ]))

    with pytest.raises(MarketDataError):
        fetch_close_history("^GSPC")


def test_fetch_close_history_does_not_fall_back_to_prior_day_when_latest_is_nan(monkeypatch):
    """A NaN latest row must fail outright, not silently return the prior
    valid day's close as if it were current."""
    _patch_yfinance(monkeypatch, _fake_close_frame([
        ("2026-08-25", 101.0),
        ("2026-08-26", float("nan")),
    ]))

    with pytest.raises(MarketDataError, match="最新交易日"):
        fetch_close_history("^GSPC")


def test_fetch_close_history_leaves_fewer_than_two_rows_after_dropping_nan(monkeypatch):
    """End-to-end: fetch_close_history drops a non-latest NaN row, leaving
    only one valid row, and the existing >=2-rows gate in
    validate_close_rows/calculate_market_snapshot must still catch it."""
    _patch_yfinance(monkeypatch, _fake_close_frame([
        ("2026-08-24", float("nan")),
        ("2026-08-25", 101.0),
    ]))

    result = fetch_close_history("^GSPC")

    assert result == [{"date": "2026-08-25", "close": 101.0}]
    validation = validate_close_rows(result, NOW)
    assert validation["valid"] is False
    assert "至少需要两个有效" in validation["error"]
    with pytest.raises(MarketDataError):
        calculate_market_snapshot(result, NOW)


def test_fetch_close_history_normal_data_is_unchanged(monkeypatch):
    """No NaN present: behavior (dates, closes, order) must be identical to
    before this change."""
    _patch_yfinance(monkeypatch, _fake_close_frame([
        ("2025-12-31", 100.0),
        ("2026-08-10", 119.0),
        ("2026-08-11", 120.0),
    ]))

    result = fetch_close_history("^GSPC")

    assert result == [
        {"date": "2025-12-31", "close": 100.0},
        {"date": "2026-08-10", "close": 119.0},
        {"date": "2026-08-11", "close": 120.0},
    ]

    snapshot = calculate_market_snapshot(result, NOW)
    assert snapshot["close"] == 120.0
    assert snapshot["ytd_return"] == pytest.approx(0.20)
    assert snapshot["ath"] == 120.0
