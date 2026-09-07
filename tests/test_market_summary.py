import json

from src.market_summary import MAX_SUMMARY_LENGTH, derive_portfolio_action, generate_market_summary


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


def model_response(summary="美国就业数据强于预期，市场重新评估9月降息概率；美债收益率与美元同步走高，"
                           "S&P 500小幅下跌，Nasdaq-100在科技股支撑下上涨，科技股相对韧性仍强。整体来看，"
                           "当前市场核心仍是经济数据与利率预期之间的重新定价，指数处于高位震荡，尚未进入明显回撤阶段。"):
    return json.dumps({"summary": summary}, ensure_ascii=False)


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
        "summary": json.loads(model_response())["summary"],
        "action": "未触发额外回撤加仓，维持正常定投，备用金保持不动。",
        "degraded": False,
    }
    assert captured["payload"]["market_breadth"]["health"]["level"] == "mixed"
    assert captured["payload"]["final_news"][0]["event_summary"] == "Federal Reserve held interest rates unchanged."
    assert "url" not in json.dumps(captured["payload"], ensure_ascii=False)
    assert "因果" in captured["prompt"]
    assert "今日结论" in captured["prompt"]
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


def test_portfolio_action_is_always_program_owned_regardless_of_model_output():
    """The model no longer outputs an `action` field at all (V2); `action` is
    derived purely from `drawdown_action`, so a model response is not even
    capable of overriding it."""
    pending = generate_market_summary(
        market_data(), context(), breadth(), news(), "pending_drawdown_buy", "key",
        call_model=lambda *args: model_response(), sleep_fn=lambda _: None,
    )
    executed = generate_market_summary(
        market_data(), context(), breadth(), news(), "drawdown_buy_executed", "key",
        call_model=lambda *args: model_response(), sleep_fn=lambda _: None,
    )

    assert pending["action"] == "已触发回撤加仓条件，等待人工确认。"
    assert executed["action"] == "对应回撤档位已经人工确认执行。"


def test_rejects_hold_summary_with_conflicting_investment_instruction_then_retries():
    attempts = []

    def model(*args):
        attempts.append(1)
        if len(attempts) == 1:
            return model_response(summary="建议暂停定投并提前加仓，等待更好的入场点。")
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
    assert "标普500当日上涨0.4%" in result["summary"]
    assert "市场宽度" not in result["summary"]
    assert "新闻解释数据暂不可用" in result["summary"]
    assert result["action"] == "未触发额外回撤加仓，维持正常定投，备用金保持不动。"


def test_rejects_overlong_model_output_and_falls_back_after_three_attempts():
    result = generate_market_summary(
        market_data(), context(), breadth(), news(), "hold", "key",
        call_model=lambda *args: model_response(summary="市场" * (MAX_SUMMARY_LENGTH + 1)),
        sleep_fn=lambda _: None,
    )

    assert result["degraded"] is True
    assert len(result["summary"]) <= MAX_SUMMARY_LENGTH


def test_max_summary_length_is_raised_well_above_the_old_220_cap():
    """V2 must support a genuine 3-4 sentence Executive Summary, not the old
    single-sentence 220-char limit."""
    assert MAX_SUMMARY_LENGTH >= 350
    long_but_valid = "市场分析。" * 70  # 350 chars, under the new cap
    assert len(long_but_valid) > 220
    result = generate_market_summary(
        market_data(), context(), breadth(), news(), "hold", "key",
        call_model=lambda *args: model_response(summary=long_but_valid), sleep_fn=lambda _: None,
    )
    assert result["degraded"] is False
    assert result["summary"] == long_but_valid


def test_fallback_prefers_final_chinese_title_and_normalizes_sentence_punctuation():
    final_news = [{
        **news()[0],
        "title_zh": "美联储维持利率不变",
        "event_summary": "Federal Reserve held interest rates unchanged.",
    }]

    result = generate_market_summary(
        market_data(), context(), breadth(), final_news, "hold", None,
    )

    assert "市场同时关注美联储维持利率不变。" in result["summary"]


def test_summary_never_invents_facts_not_in_the_input_payload():
    """The prompt must not instruct the model to search, guess, or supplement
    facts beyond the provided market/news payload."""
    from src.market_summary_prompt import SYSTEM_PROMPT

    assert "不得搜索新闻、补充宏观或公司信息、修改数字、预测市场" in SYSTEM_PROMPT


def test_prompt_no_longer_caps_numbers_to_two_to_four():
    from src.market_summary_prompt import SYSTEM_PROMPT

    assert "仅保留真正重要的 2 至 4 个数字" not in SYSTEM_PROMPT
    assert "不设固定上限" in SYSTEM_PROMPT


def test_prompt_requires_a_closing_market_phase_judgment():
    from src.market_summary_prompt import SYSTEM_PROMPT

    assert "最后一句必须给出一个当前市场阶段判断" in SYSTEM_PROMPT


def test_prompt_does_not_force_daily_portfolio_action_restatement():
    from src.market_summary_prompt import SYSTEM_PROMPT

    assert "不必每天重复" in SYSTEM_PROMPT
    assert "action 只能表达输入的 portfolio_action" not in SYSTEM_PROMPT
