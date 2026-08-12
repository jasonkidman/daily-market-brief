import pytest

from src.market_health import build_market_breadth_text, calculate_market_health


CONFIG = {
    "sector_minimum_valid": 10,
    "health": {"stock_weight": 0.60, "sector_weight": 0.40, "very_healthy": 0.80,
               "healthy": 0.65, "mixed": 0.45, "weak": 0.30},
    "divergence": {"index_move_threshold": 0.003, "narrow_rally_stock_threshold": 0.45,
                   "narrow_rally_sector_threshold": 0.45,
                   "positive_breadth_stock_threshold": 0.60,
                   "positive_breadth_sector_threshold": 0.60},
}


def breadth(stock_ratio, sector_ratio, coverage=1.0, sector_valid=11):
    return (
        {"advance_ratio": stock_ratio, "coverage_ratio": coverage, "status": "ok"},
        {"advance_ratio": sector_ratio, "valid_count": sector_valid, "items": []},
    )


@pytest.mark.parametrize(
    ("score", "level"),
    [(0.799, "healthy"), (0.80, "very_healthy"), (0.649, "mixed"), (0.65, "healthy"),
     (0.449, "weak"), (0.45, "mixed"), (0.299, "very_weak"), (0.30, "weak")],
)
def test_health_level_boundaries(score, level):
    stocks, sectors = breadth(score, score)

    result = calculate_market_health(stocks, sectors, 0.0, CONFIG)

    assert result["valid"] is True
    assert result["score"] == pytest.approx(score)
    assert result["level"] == level


def test_health_is_unavailable_when_stock_or_sector_quality_is_insufficient():
    stocks, sectors = breadth(0.8, 0.8, coverage=0.89)
    assert calculate_market_health(stocks, sectors, 0.01, CONFIG)["valid"] is False

    stocks, sectors = breadth(0.8, 0.8, sector_valid=9)
    assert calculate_market_health(stocks, sectors, 0.01, CONFIG)["valid"] is False


def test_narrow_rally_detects_low_stock_or_sector_participation():
    stocks, sectors = breadth(0.44, 0.70)
    result = calculate_market_health(stocks, sectors, 0.0031, CONFIG)
    assert result["divergence"] == "narrow_rally"

    stocks, sectors = breadth(0.70, 0.44)
    result = calculate_market_health(stocks, sectors, 0.0031, CONFIG)
    assert result["divergence"] == "narrow_rally"

    stocks, sectors = breadth(0.44, 0.44)
    assert calculate_market_health(stocks, sectors, 0.0029, CONFIG)["divergence"] is None


def test_positive_breadth_divergence_requires_both_participation_measures():
    stocks, sectors = breadth(0.61, 0.61)
    result = calculate_market_health(stocks, sectors, -0.0031, CONFIG)
    assert result["divergence"] == "positive_breadth"

    stocks, sectors = breadth(0.60, 0.61)
    assert calculate_market_health(stocks, sectors, -0.0031, CONFIG)["divergence"] is None


def test_market_breadth_text_limits_sector_context_to_three_leaders_and_laggers():
    stocks, sectors = breadth(0.65, 0.73)
    sectors["advancers"] = 8
    sectors["decliners"] = 3
    sectors["unchanged"] = 0
    sectors["items"] = [
        {"name": "科技", "daily_return": 0.012, "valid": True},
        {"name": "金融", "daily_return": 0.008, "valid": True},
        {"name": "工业", "daily_return": 0.006, "valid": True},
        {"name": "能源", "daily_return": 0.004, "valid": True},
        {"name": "公用事业", "daily_return": -0.011, "valid": True},
        {"name": "非必需消费", "daily_return": -0.007, "valid": True},
        {"name": "房地产", "daily_return": -0.004, "valid": True},
    ]
    stocks.update({"advancers": 322, "decliners": 169, "unchanged": 6})
    health = calculate_market_health(stocks, sectors, 0.01, CONFIG)

    text = build_market_breadth_text(stocks, sectors, health)

    assert "322上涨 / 169下跌 / 6平盘" in text
    assert "科技 +1.2%" in text
    assert "工业 +0.6%" in text
    assert "能源 +0.4%" not in text
    assert "公用事业 -1.1%" in text
