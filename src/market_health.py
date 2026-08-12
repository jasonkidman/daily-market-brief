"""Rule-based Market Health and market-breadth news context."""

from __future__ import annotations

from typing import Any


LEVELS = (
    ("very_healthy", "广泛上涨", "大多数股票与板块共同上涨，市场参与度非常广泛。"),
    ("healthy", "市场健康", "多数股票与板块共同上涨，市场参与度良好。"),
    ("mixed", "市场分化", "上涨与下跌分布较为均衡，市场内部存在明显分化。"),
    ("weak", "市场偏弱", "多数股票或板块表现偏弱，市场参与度下降。"),
    ("very_weak", "广泛走弱", "多数股票与板块共同走弱，市场宽度明显恶化。"),
)


def _unavailable() -> dict[str, Any]:
    return {
        "valid": False,
        "score": None,
        "level": "unavailable",
        "label": "数据不足",
        "divergence": None,
        "summary": "市场宽度数据暂不可用。",
    }


def calculate_market_health(stocks: dict, sectors: dict, sp500_daily_return: float | None,
                            config: dict) -> dict[str, Any]:
    """Return an observation-only health assessment; it never affects drawdown state."""
    minimum_coverage = float(config.get("constituents", {}).get("minimum_coverage", 0.90))
    if stocks.get("coverage_ratio", 0) < minimum_coverage:
        return _unavailable()
    if sectors.get("valid_count", 0) < int(config["sector_minimum_valid"]):
        return _unavailable()
    stock_ratio = stocks.get("advance_ratio")
    sector_ratio = sectors.get("advance_ratio")
    if stock_ratio is None or sector_ratio is None:
        return _unavailable()

    health = config["health"]
    score = stock_ratio * float(health["stock_weight"]) + sector_ratio * float(health["sector_weight"])
    if score >= float(health["very_healthy"]):
        level_index = 0
    elif score >= float(health["healthy"]):
        level_index = 1
    elif score >= float(health["mixed"]):
        level_index = 2
    elif score >= float(health["weak"]):
        level_index = 3
    else:
        level_index = 4
    level, label, summary = LEVELS[level_index]

    divergence = None
    if sp500_daily_return is not None:
        divergence_config = config["divergence"]
        if (
            sp500_daily_return > float(divergence_config["index_move_threshold"])
            and (
                stock_ratio < float(divergence_config["narrow_rally_stock_threshold"])
                or sector_ratio < float(divergence_config["narrow_rally_sector_threshold"])
            )
        ):
            divergence = "narrow_rally"
            summary = "指数上涨但市场参与度不足，上涨较集中于少数股票或板块。"
        elif (
            sp500_daily_return < -float(divergence_config["index_move_threshold"])
            and stock_ratio > float(divergence_config["positive_breadth_stock_threshold"])
            and sector_ratio > float(divergence_config["positive_breadth_sector_threshold"])
        ):
            divergence = "positive_breadth"
            summary = "指数偏弱但内部市场宽度良好，多数股票与板块仍保持上涨。"
    return {
        "valid": True,
        "score": score,
        "level": level,
        "label": label,
        "divergence": divergence,
        "summary": summary,
    }


def _format_return(item: dict) -> str:
    return f"{item['name']} {item['daily_return']:+.1%}"


def build_market_breadth_text(stocks: dict, sectors: dict, health: dict) -> str:
    """Build bounded AI context: counts plus at most three leading and lagging sectors."""
    if not health.get("valid"):
        return "【市场宽度】\n市场宽度数据暂不可用"
    valid_items = [item for item in sectors.get("items", []) if item.get("valid")]
    leading = sorted(valid_items, key=lambda item: item["daily_return"], reverse=True)[:3]
    lagging = sorted(valid_items, key=lambda item: item["daily_return"])[:3]
    lines = [
        "【市场宽度】",
        "S&P 500 成分股：",
        f"{stocks.get('advancers', 0)}上涨 / {stocks.get('decliners', 0)}下跌 / {stocks.get('unchanged', 0)}平盘",
        f"上涨占比 {stocks['advance_ratio']:.1%}",
        "板块：",
        f"{sectors.get('advancers', 0)}上涨 / {sectors.get('decliners', 0)}下跌",
    ]
    if leading:
        lines.extend(["领先板块：", *(_format_return(item) for item in leading)])
    if lagging:
        lines.extend(["落后板块：", *(_format_return(item) for item in lagging)])
    lines.extend(["市场健康：", health["level"]])
    if health.get("divergence"):
        lines.append(health["summary"])
    return "\n".join(lines)
