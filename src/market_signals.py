"""Program-calculated market signals used only as news editorial context."""

from __future__ import annotations

import math
from typing import Any, Optional


SIGNIFICANT = {
    "tech_relative": 0.005,
    "small_cap_relative": 0.005,
    "vix_daily_return": 0.10,
    "dxy_daily_return": 0.004,
    "us10y_bp_change": 5.0,
}

STRONG = {
    "tech_relative": 0.01,
    "small_cap_relative": 0.01,
    "vix_daily_return": 0.20,
    "dxy_daily_return": 0.008,
    "us10y_bp_change": 10.0,
}


def _number(snapshot: dict[str, Any], field: str) -> Optional[float]:
    if not snapshot or not snapshot.get("valid"):
        return None
    try:
        value = float(snapshot[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _signal(key: str, value: float, message: str) -> Optional[dict[str, Any]]:
    if abs(value) + 1e-12 < SIGNIFICANT[key]:
        return None
    return {
        "key": key,
        "value": value,
        "level": "strong" if abs(value) + 1e-12 >= STRONG[key] else "significant",
        "message": message,
    }


def calculate_market_signals(core: dict[str, dict], context: dict[str, dict]) -> dict[str, Any]:
    sp500 = _number(core.get("sp500", {}), "daily_return")
    nasdaq = _number(core.get("nasdaq100", {}), "daily_return")
    russell = _number(context.get("russell2000", {}), "daily_return")
    vix = _number(context.get("vix", {}), "daily_return")
    dxy = _number(context.get("dxy", {}), "daily_return")
    us10y = _number(context.get("us10y", {}), "yield_change_bp")
    tech_relative = nasdaq - sp500 if nasdaq is not None and sp500 is not None else None
    small_relative = russell - sp500 if russell is not None and sp500 is not None else None
    values = {
        "tech_relative": tech_relative,
        "small_cap_relative": small_relative,
        "vix_daily_return": vix,
        "dxy_daily_return": dxy,
        "us10y_bp_change": us10y,
    }
    messages = {}
    if tech_relative is not None:
        verb = "领先" if tech_relative >= 0 else "落后"
        messages["tech_relative"] = (
            f"Nasdaq-100 比 S&P 500 {verb} {abs(tech_relative) * 100:.2f} 个百分点，"
            f"科技/成长相对{'占优' if tech_relative >= 0 else '承压'}"
        )
    if small_relative is not None:
        verb = "领先" if small_relative >= 0 else "落后"
        messages["small_cap_relative"] = (
            f"Russell 2000 比 S&P 500 {verb} {abs(small_relative) * 100:.2f} 个百分点，"
            f"小盘相对{'占优' if small_relative >= 0 else '落后'}"
        )
    if vix is not None:
        messages["vix_daily_return"] = f"VIX 明显{'上升，波动率风险升温' if vix >= 0 else '下降，风险情绪趋缓'}"
    if dxy is not None:
        messages["dxy_daily_return"] = f"美元明显{'走强' if dxy >= 0 else '走弱'}"
    if us10y is not None:
        messages["us10y_bp_change"] = f"10Y 美债收益率明显{'上升' if us10y >= 0 else '下降'}"
    signals = []
    for key, value in values.items():
        if value is not None:
            item = _signal(key, value, messages[key])
            if item:
                signals.append(item)
    return {**values, "signals": signals}


def build_market_context_for_ai(core: dict[str, dict], context: dict[str, dict],
                                market_signals: dict[str, Any]) -> str:
    core_names = {"sp500": "S&P 500", "nasdaq100": "Nasdaq-100", "dow": "Dow Jones"}
    context_names = {"russell2000": "Russell 2000", "vix": "VIX", "dxy": "美元指数"}
    lines = ["【核心市场】"]
    for key, name in core_names.items():
        value = _number(core.get(key, {}), "daily_return")
        if value is not None:
            lines.extend([name, f"{value:+.2%}"])
    lines.append("【市场环境】")
    for key, name in context_names.items():
        snapshot = context.get(key, {})
        change = _number(snapshot, "daily_return")
        if change is not None:
            lines.extend([name, f"{change:+.2%}"])
    us10y = context.get("us10y", {})
    yield_value = _number(us10y, "close")
    yield_bp = _number(us10y, "yield_change_bp")
    if yield_value is not None and yield_bp is not None:
        lines.extend(["10Y 美债", f"{yield_value:.2f}%", f"日变化 {yield_bp:+.0f}bp"])
    lines.append("【结构信号】")
    lines.extend(f"- {item['message']}（{item['level']}）" for item in market_signals.get("signals", []))
    return "\n".join(lines)
