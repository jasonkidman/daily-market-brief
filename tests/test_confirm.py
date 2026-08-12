from datetime import datetime
from zoneinfo import ZoneInfo

import json
import pytest

from src.confirm_drawdown import ConfirmationError, confirm_and_render, confirm_tier
from src.drawdown import create_index_state


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def state_with(status):
    return {
        "indices": {
            "sp500": {
                "cycle_id": "sp500-2026-01-01-1000000",
                "tiers": {"tier_1": {"status": status, "amount": 28000, "executed_at": None}},
            }
        },
        "executions": [],
    }


def test_not_triggered_cannot_be_confirmed():
    with pytest.raises(ConfirmationError, match="该档位尚未触发，禁止标记为已执行。"):
        confirm_tier(state_with("not_triggered"), "sp500", "tier_1", NOW)


def test_pending_can_be_confirmed_and_records_execution():
    updated, changed = confirm_tier(state_with("pending"), "sp500", "tier_1", NOW)
    assert changed is True
    assert updated["indices"]["sp500"]["tiers"]["tier_1"]["status"] == "executed"
    assert updated["executions"] == [{
        "executed_at": NOW.isoformat(), "amount": 28000, "index": "sp500",
        "tier": "tier_1", "cycle_id": "sp500-2026-01-01-1000000"
    }]


def test_executed_confirmation_is_idempotent():
    original = state_with("executed")
    updated, changed = confirm_tier(original, "sp500", "tier_1", NOW)
    assert changed is False
    assert updated == original


def test_confirm_and_render_updates_latest_report_without_ai(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "data" / "reports").mkdir(parents=True)
    rules = {"name": "S&P 500", "pool": 140000, "tiers": [
        {"id": "tier_1", "threshold": 0.10, "allocation": 0.20},
        {"id": "tier_2", "threshold": 0.15, "allocation": 0.30},
        {"id": "tier_3", "threshold": 0.20, "allocation": 0.30},
        {"id": "tier_4", "threshold": 0.25, "allocation": 0.20},
    ]}
    index_state = create_index_state("sp500", rules, 100, "2026-01-01", NOW)
    index_state["tiers"]["tier_1"]["status"] = "pending"
    current = {"indices": {"sp500": index_state}, "executions": []}
    (tmp_path / "state" / "drawdown_state.json").write_text(json.dumps(current), encoding="utf-8")
    report = {"report_date": "2026-08-12", "drawdown": {}, "news": [{"title_zh": "保持不变"}]}
    report_path = tmp_path / "data" / "reports" / "2026-08-12.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    changed = confirm_and_render(tmp_path, "sp500", "tier_1", NOW, render=False)

    updated_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert changed is True
    assert updated_report["drawdown"]["sp500"]["executed_amount"] == 28000
    assert updated_report["news"] == [{"title_zh": "保持不变"}]
