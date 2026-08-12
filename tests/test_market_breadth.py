from datetime import date

import pandas as pd
import pytest

from src.market_breadth import (
    calculate_sector_breadth,
    calculate_stock_breadth,
    fetch_batched_close_rows,
)


TARGET = "2026-08-11"
CONFIG = {"normal_coverage": 0.95, "minimum_coverage": 0.90}


def constituents(count):
    return [
        {"source_ticker": f"S{index}", "yahoo_ticker": f"T{index}", "name": f"Stock {index}",
         "as_of_date": "2026-08-10"}
        for index in range(count)
    ]


def rows(previous, latest, target=TARGET):
    return [{"date": "2026-08-10", "close": previous}, {"date": target, "close": latest}]


def test_stock_breadth_uses_valid_target_date_closes_and_valid_count_denominator():
    stocks = constituents(10)
    histories = {
        **{f"T{index}": rows(100, 101) for index in range(6)},
        **{f"T{index}": rows(100, 99) for index in range(6, 9)},
        "T9": rows(100, 100),
    }

    result = calculate_stock_breadth(stocks, histories, TARGET, CONFIG)

    assert result["total_constituents"] == 10
    assert result["valid_count"] == 10
    assert result["invalid_count"] == 0
    assert result["advancers"] == 6
    assert result["decliners"] == 3
    assert result["unchanged"] == 1
    assert result["advance_ratio"] == pytest.approx(0.60)
    assert result["decline_ratio"] == pytest.approx(0.30)
    assert result["unchanged_ratio"] == pytest.approx(0.10)
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["status"] == "ok"


def test_stock_without_target_market_date_is_invalid_not_stale():
    stocks = constituents(2)
    histories = {
        "T0": rows(100, 101),
        "T1": [{"date": "2026-08-07", "close": 90}, {"date": "2026-08-10", "close": 100}],
    }

    result = calculate_stock_breadth(stocks, histories, TARGET, CONFIG)

    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
    assert result["advancers"] == 1
    assert result["advance_ratio"] == pytest.approx(1.0)
    assert result["coverage_ratio"] == pytest.approx(0.5)
    assert result["status"] == "invalid"


@pytest.mark.parametrize(
    ("total", "valid", "status"),
    [(20, 19, "ok"), (20, 18, "partial"), (20, 17, "invalid")],
)
def test_stock_breadth_coverage_boundaries(total, valid, status):
    stocks = constituents(total)
    histories = {f"T{index}": rows(100, 101) for index in range(valid)}

    result = calculate_stock_breadth(stocks, histories, TARGET, CONFIG)

    assert result["coverage_ratio"] == pytest.approx(valid / total)
    assert result["status"] == status


def test_batch_price_fetch_uses_bounded_batches_and_returns_close_rows():
    calls = []

    def downloader(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        columns = pd.MultiIndex.from_product([["Close"], tickers])
        return pd.DataFrame(
            [[100 + index for index in range(len(tickers))], [101 + index for index in range(len(tickers))]],
            index=pd.to_datetime(["2026-08-10", "2026-08-11"]),
            columns=columns,
        )

    fetched = fetch_batched_close_rows(
        ["T0", "T1", "T2", "T3", "T4"], date(2026, 8, 1), date(2026, 8, 12), 2, downloader
    )

    assert [call[0] for call in calls] == [("T0", "T1"), ("T2", "T3"), ("T4",)]
    assert fetched["T0"] == [{"date": "2026-08-10", "close": 100.0}, {"date": "2026-08-11", "close": 101.0}]
    assert fetched["T4"][-1] == {"date": "2026-08-11", "close": 101.0}


def test_sector_breadth_counts_sorts_and_uses_target_market_date():
    sector_config = {
        "technology": {"name": "科技", "ticker": "XLK"},
        "financials": {"name": "金融", "ticker": "XLF"},
        "utilities": {"name": "公用事业", "ticker": "XLU"},
        "real_estate": {"name": "房地产", "ticker": "XLRE"},
    }
    histories = {
        "XLK": rows(100, 103),
        "XLF": rows(100, 101.5),
        "XLU": rows(100, 99.25),
        # A stale close must not be reused as the target date.
        "XLRE": [{"date": "2026-08-07", "close": 100}, {"date": "2026-08-10", "close": 99}],
    }

    result = calculate_sector_breadth(sector_config, histories, TARGET)

    assert [item["ticker"] for item in result["items"]] == ["XLK", "XLF", "XLU", "XLRE"]
    assert result["valid_count"] == 3
    assert result["advancers"] == 2
    assert result["decliners"] == 1
    assert result["unchanged"] == 0
    assert result["advance_ratio"] == pytest.approx(2 / 3)
    assert result["items"][-1]["valid"] is False
    assert result["items"][-1]["bar_strength"] is None


@pytest.mark.parametrize(
    ("latest", "expected_strength"),
    [(103, 1.0), (101.5, 0.5), (97, 1.0), (96, 1.0)],
)
def test_sector_bar_strength_uses_fixed_three_percent_scale_and_caps(latest, expected_strength):
    result = calculate_sector_breadth(
        {"technology": {"name": "科技", "ticker": "XLK"}},
        {"XLK": rows(100, latest)},
        TARGET,
    )

    item = result["items"][0]
    assert item["daily_return"] == pytest.approx(latest / 100 - 1)
    assert item["bar_strength"] == pytest.approx(expected_strength)
