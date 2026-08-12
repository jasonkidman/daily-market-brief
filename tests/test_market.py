from datetime import datetime
from zoneinfo import ZoneInfo

import math
import pytest

from src.market import MarketDataError, calculate_market_snapshot, validate_close_rows


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
