"""Evidence-bounded, action-safe executive market-summary generation."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .deepseek_client import call_deepseek
from .market_summary_prompt import SYSTEM_PROMPT


ACTION_COPY = {
    "hold": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
    "pending_drawdown_buy": "已触发回撤加仓条件，等待人工确认。",
    "drawdown_buy_executed": "对应回撤档位已经人工确认执行。",
}
HOLD_CONFLICTS = ("暂停定投", "停止定投", "减仓", "清仓", "卖出", "提前加仓", "提前买入", "建议抄底", "建议卖出")
MAX_SUMMARY_LENGTH = 220


class MarketSummaryError(ValueError):
    """Raised when the summary model output is not safe or contract-valid."""


def derive_portfolio_action(drawdown: dict) -> str:
    """Derive the sole permitted action strictly from persisted drawdown state."""
    indices = list((drawdown or {}).values())
    if any(item.get("status") == "pending" or item.get("pending_tiers") for item in indices):
        return "pending_drawdown_buy"
    if any(item.get("status") == "executed" or item.get("executed_tiers") for item in indices):
        return "drawdown_buy_executed"
    return "hold"


def _valid_snapshot(snapshot: dict, field: str):
    if not snapshot or not snapshot.get("valid"):
        return None
    value = snapshot.get(field)
    return value if isinstance(value, (int, float)) else None


def _summary_payload(market_data: dict, market_context: dict, market_breadth: dict,
                     news: list[dict], drawdown_action: str) -> dict:
    final_news = []
    for item in news[:8]:
        final_news.append({
            "event_summary": item.get("event_summary") or item.get("original_title") or item.get("title"),
            "title": item.get("original_title") or item.get("title"),
            "summary": item.get("summary_zh") or item.get("summary"),
            "topic_group": item.get("topic_group"),
        })
    return {
        "market_data": market_data,
        "market_context": market_context,
        "market_breadth": market_breadth,
        "final_news": final_news,
        "portfolio_action": drawdown_action,
        "portfolio_action_meaning": ACTION_COPY[drawdown_action],
    }


def _parse_model_summary(raw: Any, drawdown_action: str) -> dict:
    try:
        payload = raw if isinstance(raw, dict) else json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MarketSummaryError("市场摘要输出无法解析为 JSON。") from exc
    if not isinstance(payload, dict):
        raise MarketSummaryError("市场摘要输出不是 JSON 对象。")
    fields = {}
    for key in ("market", "drivers", "action"):
        value = str(payload.get(key, "")).strip()
        if not value:
            raise MarketSummaryError(f"{key} 不能为空。")
        fields[key] = value
    text = "".join(fields.values())
    if len(text) > MAX_SUMMARY_LENGTH:
        raise MarketSummaryError("市场摘要超过长度上限。")
    if drawdown_action == "hold" and any(term in text for term in HOLD_CONFLICTS):
        raise MarketSummaryError("hold 状态下出现冲突的投资指令。")
    return {
        "market": fields["market"],
        "drivers": fields["drivers"],
        "action": ACTION_COPY[drawdown_action],
        "degraded": False,
    }


def _movement(label: str, value: float | None) -> str:
    if value is None:
        return f"{label}数据暂不可用"
    if value > 0:
        return f"{label}当日上涨{value:.1%}"
    if value < 0:
        return f"{label}当日下跌{abs(value):.1%}"
    return f"{label}当日持平"


def deterministic_market_summary(market_data: dict, market_breadth: dict, news: list[dict],
                                 drawdown_action: str) -> dict:
    """Produce a compact, evidence-only result when AI is unavailable or unsafe."""
    sp500 = _valid_snapshot(market_data.get("sp500", {}), "daily_return")
    nasdaq = _valid_snapshot(market_data.get("nasdaq100", {}), "daily_return")
    market = f"{_movement('标普500', sp500)}，{_movement('纳指100', nasdaq)}。"
    health = (market_breadth or {}).get("health", {})
    if health.get("valid"):
        market = market[:-1] + f"，当前市场宽度为{health.get('label', health.get('level', '数据不足'))}。"
    if news:
        item = news[0]
        fact = (
            item.get("title_zh") or item.get("event_summary") or item.get("original_title") or item.get("title")
        )
        if fact:
            ending = "" if str(fact).endswith(("。", "！", "？", ".", "!", "?")) else "。"
            drivers = f"市场同时关注{fact}{ending}"
        else:
            drivers = "新闻解释数据暂不可用。"
    else:
        drivers = "新闻解释数据暂不可用。"
    return {
        "market": market,
        "drivers": drivers,
        "action": ACTION_COPY[drawdown_action],
        "degraded": True,
    }


def generate_market_summary(market_data: dict, market_context: dict, market_breadth: dict,
                            news: list[dict], drawdown_action: str, api_key: str | None,
                            call_model: Callable = call_deepseek, sleep_fn: Callable = time.sleep) -> dict:
    """Generate a validated summary, degrading to program-owned facts after three failures."""
    if drawdown_action not in ACTION_COPY:
        raise ValueError("portfolio_action 不合法。")
    if not api_key:
        return deterministic_market_summary(market_data, market_breadth, news, drawdown_action)
    user_payload = json.dumps(
        _summary_payload(market_data, market_context, market_breadth, news, drawdown_action),
        ensure_ascii=False,
    )
    for attempt in range(3):
        try:
            return _parse_model_summary(call_model(SYSTEM_PROMPT, user_payload, api_key), drawdown_action)
        except Exception:
            if attempt < 2:
                sleep_fn((5, 10)[attempt])
    return deterministic_market_summary(market_data, market_breadth, news, drawdown_action)
