from pathlib import Path

import pytest

from src.constituents import (
    ConstituentReferenceError,
    load_constituents,
    normalize_yahoo_ticker,
    update_reference,
    validate_constituents,
)


def rows(*tickers):
    return [
        {
            "source_ticker": ticker,
            "yahoo_ticker": normalize_yahoo_ticker(ticker),
            "name": f"Company {ticker}",
            "as_of_date": "2026-08-11",
        }
        for ticker in tickers
    ]


def test_normalizes_state_street_class_share_tickers_for_yahoo():
    assert normalize_yahoo_ticker(" BRK.B ") == "BRK-B"
    assert normalize_yahoo_ticker("BF/B") == "BF-B"
    assert normalize_yahoo_ticker("AAPL") == "AAPL"


def test_reads_complete_constituent_reference_csv(tmp_path):
    path = tmp_path / "constituents.csv"
    path.write_text(
        "source_ticker,yahoo_ticker,name,as_of_date\nBRK.B,BRK-B,Berkshire Hathaway,2026-08-11\n",
        encoding="utf-8",
    )

    assert load_constituents(path) == [
        {
            "source_ticker": "BRK.B",
            "yahoo_ticker": "BRK-B",
            "name": "Berkshire Hathaway",
            "as_of_date": "2026-08-11",
        }
    ]


def test_rejects_duplicate_yahoo_tickers():
    with pytest.raises(ConstituentReferenceError, match="重复"):
        validate_constituents(rows("BRK.B", "BRK-B"), minimum_count=2)


def test_invalid_update_never_overwrites_existing_reference(tmp_path):
    reference = tmp_path / "sp500_constituents.csv"
    original = b"source_ticker,yahoo_ticker,name,as_of_date\nAAPL,AAPL,Apple,2026-08-11\n"
    reference.write_bytes(original)

    def invalid_fetcher():
        return [{"source_ticker": "CASH", "name": "Cash", "as_of_date": "2026-08-12"}]

    with pytest.raises(ConstituentReferenceError):
        update_reference(reference, invalid_fetcher, minimum_count=1)

    assert reference.read_bytes() == original
