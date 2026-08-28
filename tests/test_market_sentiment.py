from src.market_sentiment import (
    calculate_market_sentiment,
    label_for_score,
    score_breadth,
    score_momentum,
    score_risk_appetite,
    score_vix,
)


def _vix(close, valid=True):
    return {"valid": valid, "close": close}


def _snapshot(daily_return, valid=True):
    return {"valid": valid, "daily_return": daily_return}


def _history(closes):
    return [{"date": f"2026-08-{i + 1:02d}", "close": close} for i, close in enumerate(closes)]


# ---- score_vix ----

def test_score_vix_low_level_is_high_score():
    assert score_vix(_vix(8.0)) == 100.0


def test_score_vix_extreme_high_level_is_zero():
    assert score_vix(_vix(80.0)) == 0.0


def test_score_vix_midpoint_interpolates():
    assert score_vix(_vix(20.0)) == 50.0


def test_score_vix_invalid_or_missing_returns_none():
    assert score_vix(None) is None
    assert score_vix(_vix(15.0, valid=False)) is None
    assert score_vix({"valid": True, "close": None}) is None


# ---- score_breadth ----

def test_score_breadth_scales_health_score_to_100():
    assert score_breadth({"valid": True, "score": 0.702}) == 70.2


def test_score_breadth_clips_out_of_range_score():
    assert score_breadth({"valid": True, "score": 1.5}) == 100.0
    assert score_breadth({"valid": True, "score": -0.5}) == 0.0


def test_score_breadth_missing_or_invalid_returns_none():
    assert score_breadth(None) is None
    assert score_breadth({"valid": False, "score": 0.9}) is None
    assert score_breadth({"valid": True, "score": None}) is None


# ---- score_momentum ----

def test_score_momentum_positive_trend_scores_above_50():
    histories = {"sp500": _history([100, 102, 104, 106, 108, 110]),
                 "nasdaq100": _history([200, 204, 208, 212, 216, 220])}
    score = score_momentum(histories, lookback=5)
    assert score > 50.0


def test_score_momentum_flat_series_scores_50():
    histories = {"sp500": _history([100, 100, 100]), "nasdaq100": _history([200, 200, 200])}
    assert score_momentum(histories, lookback=5) == 50.0


def test_score_momentum_extreme_drop_clips_to_zero():
    histories = {"sp500": _history([100, 50]), "nasdaq100": _history([200, 100])}
    assert score_momentum(histories, lookback=5) == 0.0


def test_score_momentum_extreme_rally_clips_to_100():
    histories = {"sp500": _history([100, 200]), "nasdaq100": _history([200, 400])}
    assert score_momentum(histories, lookback=5) == 100.0


def test_score_momentum_insufficient_history_returns_none():
    assert score_momentum({"sp500": _history([100]), "nasdaq100": _history([200])}) is None
    assert score_momentum({}) is None
    assert score_momentum(None) is None


def test_score_momentum_uses_only_available_index_when_one_is_missing():
    histories = {"sp500": _history([100, 110]), "nasdaq100": _history([200])}
    score = score_momentum(histories, lookback=5)
    assert score is not None and score > 50.0


# ---- score_risk_appetite ----

def test_score_risk_appetite_small_cap_outperformance_scores_above_50():
    context = {"russell2000": _snapshot(0.02)}
    core = {"sp500": _snapshot(0.005)}
    assert score_risk_appetite(context, core) > 50.0


def test_score_risk_appetite_small_cap_underperformance_scores_below_50():
    context = {"russell2000": _snapshot(-0.01)}
    core = {"sp500": _snapshot(0.01)}
    assert score_risk_appetite(context, core) < 50.0


def test_score_risk_appetite_equal_returns_scores_exactly_50():
    context = {"russell2000": _snapshot(0.01)}
    core = {"sp500": _snapshot(0.01)}
    assert score_risk_appetite(context, core) == 50.0


def test_score_risk_appetite_missing_or_invalid_returns_none():
    assert score_risk_appetite({}, {}) is None
    assert score_risk_appetite({"russell2000": _snapshot(0.01, valid=False)},
                               {"sp500": _snapshot(0.01)}) is None
    assert score_risk_appetite({"russell2000": _snapshot(0.01)},
                               {"sp500": _snapshot(None)}) is None


# ---- label_for_score ----

def test_label_for_score_boundaries():
    assert label_for_score(0) == "恐慌"
    assert label_for_score(19.9) == "恐慌"
    assert label_for_score(20) == "偏谨慎"
    assert label_for_score(39.9) == "偏谨慎"
    assert label_for_score(40) == "中性"
    assert label_for_score(59.9) == "中性"
    assert label_for_score(60) == "偏乐观"
    assert label_for_score(74.9) == "偏乐观"
    assert label_for_score(75) == "亢奋"
    assert label_for_score(89.9) == "亢奋"
    assert label_for_score(90) == "极度亢奋"
    assert label_for_score(100) == "极度亢奋"


# ---- calculate_market_sentiment (composite) ----

def test_calculate_market_sentiment_full_real_data_produces_weighted_composite():
    context = {"vix": _vix(20.0), "russell2000": _snapshot(0.01)}
    core = {"sp500": _snapshot(0.01)}
    histories = {"sp500": _history([100, 100]), "nasdaq100": _history([200, 200])}
    breadth_health = {"valid": True, "score": 0.6}
    result = calculate_market_sentiment(context, core, histories, breadth_health)
    # vix_score=50, breadth_score=60, momentum_score=50, risk_appetite_score=50
    expected = round(50 * 0.35 + 60 * 0.30 + 50 * 0.20 + 50 * 0.15, 1)
    assert result["vix_score"] == 50.0
    assert result["breadth_score"] == 60.0
    assert result["momentum_score"] == 50.0
    assert result["risk_appetite_score"] == 50.0
    assert result["market_sentiment_score"] == expected
    assert result["market_sentiment_label"] == label_for_score(expected)


def test_calculate_market_sentiment_missing_any_subscore_yields_none_composite_not_estimated():
    context = {"vix": _vix(20.0, valid=False), "russell2000": _snapshot(0.01)}
    core = {"sp500": _snapshot(0.01)}
    histories = {"sp500": _history([100, 100]), "nasdaq100": _history([200, 200])}
    breadth_health = {"valid": True, "score": 0.6}
    result = calculate_market_sentiment(context, core, histories, breadth_health)
    assert result["vix_score"] is None
    assert result["breadth_score"] == 60.0
    assert result["market_sentiment_score"] is None
    assert result["market_sentiment_label"] is None


def test_calculate_market_sentiment_all_missing_returns_all_none():
    result = calculate_market_sentiment({}, {}, {}, {})
    assert result == {
        "vix_score": None,
        "breadth_score": None,
        "momentum_score": None,
        "risk_appetite_score": None,
        "market_sentiment_score": None,
        "market_sentiment_label": None,
    }


def test_calculate_market_sentiment_extreme_inputs_clip_within_bounds():
    context = {"vix": _vix(5.0), "russell2000": _snapshot(0.5)}
    core = {"sp500": _snapshot(-0.5)}
    histories = {"sp500": _history([100, 1000]), "nasdaq100": _history([200, 2000])}
    breadth_health = {"valid": True, "score": 1.0}
    result = calculate_market_sentiment(context, core, histories, breadth_health)
    assert result["market_sentiment_score"] is not None
    assert 0.0 <= result["market_sentiment_score"] <= 100.0
