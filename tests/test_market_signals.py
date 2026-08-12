import pytest

from src.market_signals import build_market_context_for_ai, calculate_market_signals


def snapshot(daily_return=None, close=100, yield_change_bp=None, valid=True):
    result = {"valid": valid, "close": close}
    if daily_return is not None:
        result["daily_return"] = daily_return
    if yield_change_bp is not None:
        result["yield_change_bp"] = yield_change_bp
    return result


def calculate(relative=0.005, small_relative=0.005, vix=0.10, dxy=0.004, bp=5):
    core = {
        "sp500": snapshot(0),
        "nasdaq100": snapshot(relative),
        "dow": snapshot(0),
    }
    context = {
        "russell2000": snapshot(small_relative),
        "vix": snapshot(vix),
        "dxy": snapshot(dxy),
        "us10y": snapshot(close=4.32, yield_change_bp=bp),
    }
    return calculate_market_signals(core, context)


def test_market_signal_values_are_program_calculated():
    result = calculate(relative=-0.008, small_relative=0.008, vix=0.126, dxy=0.0051, bp=8)
    assert result["tech_relative"] == pytest.approx(-0.008)
    assert result["small_cap_relative"] == pytest.approx(0.008)
    assert result["vix_daily_return"] == pytest.approx(0.126)
    assert result["dxy_daily_return"] == pytest.approx(0.0051)
    assert result["us10y_bp_change"] == pytest.approx(8)
    assert {item["key"] for item in result["signals"]} == {
        "tech_relative", "small_cap_relative", "vix_daily_return",
        "dxy_daily_return", "us10y_bp_change",
    }


@pytest.mark.parametrize(
    ("kwargs", "key"),
    [
        ({"relative": 0.0049}, "tech_relative"),
        ({"small_relative": 0.0049}, "small_cap_relative"),
        ({"vix": 0.099}, "vix_daily_return"),
        ({"dxy": 0.0039}, "dxy_daily_return"),
        ({"bp": 4}, "us10y_bp_change"),
    ],
)
def test_market_signal_just_below_threshold_is_not_significant(kwargs, key):
    defaults = {"relative": 0, "small_relative": 0, "vix": 0, "dxy": 0, "bp": 0}
    defaults.update(kwargs)
    result = calculate(**defaults)
    assert key not in {item["key"] for item in result["signals"]}


def test_market_signal_threshold_boundaries_are_significant():
    result = calculate(relative=0.005, small_relative=-0.005, vix=-0.10, dxy=0.004, bp=-5)
    assert {item["key"] for item in result["signals"]} == {
        "tech_relative", "small_cap_relative", "vix_daily_return",
        "dxy_daily_return", "us10y_bp_change",
    }


def test_market_context_text_omits_missing_data():
    core = {"sp500": snapshot(-0.0032), "nasdaq100": snapshot(-0.0078), "dow": snapshot(valid=False)}
    context = {
        "russell2000": snapshot(0.0044, close=2946.1),
        "vix": snapshot(valid=False),
        "dxy": snapshot(0.0051, close=99.83),
        "us10y": snapshot(close=4.34, yield_change_bp=8),
    }
    signals = calculate_market_signals(core, context)
    text = build_market_context_for_ai(core, context, signals)
    assert "S&P 500" in text
    assert "Russell 2000" in text
    assert "VIX" not in text
    assert "10Y 美债" in text
