from datetime import datetime
from zoneinfo import ZoneInfo

import math
import pytest

from src.market import (
    MarketDataError,
    calculate_context_snapshot,
    calculate_market_snapshot,
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
    def fake_fetch(ticker):
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
