import json

from src.market_summary import derive_portfolio_action, generate_market_summary


def market_data():
    return {
        "sp500": {"name": "S&P 500", "valid": True, "daily_return": 0.004},
        "nasdaq100": {"name": "Nasdaq-100", "valid": True, "daily_return": 0.008},
        "dow": {"name": "Dow Jones", "valid": True, "daily_return": 0.001},
    }


def context():
    return {
        "vix": {"name": "VIX", "valid": True, "daily_return": 0.08},
        "us10y": {"name": "10Y 美债", "valid": True, "yield_change_bp": 7},
    }


def breadth(valid=True):
    return {
        "stocks": {"advancers": 322, "decliners": 169, "unchanged": 6, "advance_ratio": 0.648},
        "sectors": {"advancers": 8, "decliners": 3, "items": []},
        "health": {
            "valid": valid,
            "level": "mixed" if valid else "unavailable",
            "label": "市场分化" if valid else "数据不足",
            "divergence": None,
        },
    }


def news():
    return [{
        "event_summary": "Federal Reserve held interest rates unchanged.",
        "original_title": "Fed holds rates",
        "summary_zh": "美联储维持利率不变。",
        "topic_group": "US_MARKET_MACRO",
    }]


def drawdown(status="normal"):
    return {
        "sp500": {"status": status, "pending_tiers": [], "executed_tiers": []},
        "nasdaq100": {"status": "normal", "pending_tiers": [], "executed_tiers": []},
    }


def model_response(market="标普500小幅上涨，纳指100相对更强，市场内部仍有分化。",
                   drivers="市场关注利率预期，并同时关注人工智能相关事件。",
                   action="维持正常定投与备用金计划。"):
    return json.dumps({"market": market, "drivers": drivers, "action": action})


def test_derives_hold_pending_and_executed_actions_from_drawdown_state():
    assert derive_portfolio_action(drawdown()) == "hold"
    assert derive_portfolio_action(drawdown("pending")) == "pending_drawdown_buy"
    executed = drawdown()
    executed["sp500"]["status"] = "executed"
    executed["sp500"]["executed_tiers"] = [{"id": "tier_1"}]
    assert derive_portfolio_action(executed) == "drawdown_buy_executed"


def test_generates_valid_summary_from_existing_inputs_and_program_owned_action():
    captured = {}

    def model(system_prompt, user_payload, api_key, **kwargs):
        captured["prompt"] = system_prompt
        captured["payload"] = json.loads(user_payload)
        captured["kwargs"] = kwargs
        return model_response()

    result = generate_market_summary(
        market_data(), context(), breadth(), news(), "hold", "key", call_model=model, sleep_fn=lambda _: None,
    )

    assert result == {
        "market": "标普500小幅上涨，纳指100相对更强，市场内部仍有分化。",
        "drivers": "市场关注利率预期，并同时关注人工智能相关事件。",
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
        "degraded": False,
    }
    assert captured["payload"]["market_breadth"]["health"]["level"] == "mixed"
    assert captured["payload"]["final_news"][0]["event_summary"] == "Federal Reserve held interest rates unchanged."
    assert "url" not in json.dumps(captured["payload"], ensure_ascii=False)
    assert "不得无依据建立因果关系" in captured["prompt"]
    assert captured["kwargs"] == {"thinking_enabled": True, "reasoning_effort": "high"}


def test_summary_payload_passes_all_dynamic_news_without_eight_item_truncation():
    captured = {}

    def model(system_prompt, user_payload, api_key, **kwargs):
        captured["payload"] = json.loads(user_payload)
        return model_response()

    dynamic_news = [{**news()[0], "original_title": f"Event {index}"} for index in range(12)]
    generate_market_summary(
        market_data(), context(), breadth(), dynamic_news, "hold", "key",
        call_model=model, sleep_fn=lambda _: None,
    )

    assert len(captured["payload"]["final_news"]) == 12


def test_pending_and_executed_actions_use_only_program_owned_action_copy():
    pending = generate_market_summary(
        market_data(), context(), breadth(), news(), "pending_drawdown_buy", "key",
        call_model=lambda *args: model_response(action="可以立即买入。"), sleep_fn=lambda _: None,
    )
    executed = generate_market_summary(
        market_data(), context(), breadth(), news(), "drawdown_buy_executed", "key",
        call_model=lambda *args: model_response(action="继续操作。"), sleep_fn=lambda _: None,
    )

    assert pending["action"] == "已触发回撤加仓条件，等待人工确认。"
    assert executed["action"] == "对应回撤档位已经人工确认执行。"


def test_rejects_hold_summary_with_conflicting_investment_instruction_then_retries():
    attempts = []

    def model(*args):
        attempts.append(1)
        if len(attempts) == 1:
            return model_response(action="建议暂停定投并提前加仓。")
        return model_response()

    result = generate_market_summary(
        market_data(), context(), breadth(), news(), "hold", "key", call_model=model, sleep_fn=lambda _: None,
    )

    assert len(attempts) == 2
    assert result["degraded"] is False


def test_three_failures_use_deterministic_fallback_without_inventing_news_or_breadth():
    attempts, sleeps = [], []

    def failing(*args):
        attempts.append(1)
        return "not json"

    result = generate_market_summary(
        market_data(), context(), breadth(valid=False), [], "hold", "key",
        call_model=failing, sleep_fn=sleeps.append,
    )

    assert len(attempts) == 3
    assert sleeps == [5, 10]
    assert result["degraded"] is True
    assert "标普500当日上涨0.4%" in result["market"]
    assert "市场宽度" not in result["market"]
    assert result["drivers"] == "新闻解释数据暂不可用。"
    assert result["action"] == "未触发额外回撤加仓，维持正常定投，备用金保持不动。"


def test_rejects_overlong_model_output_and_falls_back_after_three_attempts():
    result = generate_market_summary(
        market_data(), context(), breadth(), news(), "hold", "key",
        call_model=lambda *args: model_response(market="市场" * 221), sleep_fn=lambda _: None,
    )

    assert result["degraded"] is True
    assert len("".join(result[key] for key in ("market", "drivers", "action"))) <= 220


def test_fallback_prefers_final_chinese_title_and_normalizes_sentence_punctuation():
    final_news = [{
        **news()[0],
        "title_zh": "美联储维持利率不变",
        "event_summary": "Federal Reserve held interest rates unchanged.",
    }]

    result = generate_market_summary(
        market_data(), context(), breadth(), final_news, "hold", None,
    )

    assert result["drivers"] == "市场同时关注美联储维持利率不变。"
