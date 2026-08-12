from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.drawdown import create_index_state, summarize_index_state, update_index_state


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

SP_RULES = {
    "name": "S&P 500",
    "pool": 140000,
    "tiers": [
        {"id": "tier_1", "threshold": 0.10, "allocation": 0.20},
        {"id": "tier_2", "threshold": 0.15, "allocation": 0.30},
        {"id": "tier_3", "threshold": 0.20, "allocation": 0.30},
        {"id": "tier_4", "threshold": 0.25, "allocation": 0.20},
    ],
}
NDX_RULES = {
    "name": "Nasdaq-100",
    "pool": 60000,
    "tiers": [
        {"id": "tier_1", "threshold": 0.15, "allocation": 0.20},
        {"id": "tier_2", "threshold": 0.20, "allocation": 0.30},
        {"id": "tier_3", "threshold": 0.30, "allocation": 0.30},
        {"id": "tier_4", "threshold": 0.40, "allocation": 0.20},
    ],
}


def close_for(drawdown):
    return 100 * (1 - drawdown)


@pytest.mark.parametrize(
    ("drawdown", "pending"),
    [(0.09, []), (0.10, ["tier_1"]), (0.15, ["tier_1", "tier_2"]),
     (0.20, ["tier_1", "tier_2", "tier_3"]),
     (0.25, ["tier_1", "tier_2", "tier_3", "tier_4"])],
)
def test_sp500_thresholds_trigger_all_reached_tiers(drawdown, pending):
    state = create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW)
    updated, archived = update_index_state(
        state, [{"date": "2026-08-11", "close": close_for(drawdown)}], True, SP_RULES, NOW
    )
    assert [key for key, tier in updated["tiers"].items() if tier["status"] == "pending"] == pending
    assert archived == []


@pytest.mark.parametrize(
    ("drawdown", "count"), [(0.14, 0), (0.15, 1), (0.20, 2), (0.30, 3), (0.40, 4)]
)
def test_nasdaq_thresholds(drawdown, count):
    state = create_index_state("nasdaq100", NDX_RULES, 100, "2026-01-01", NOW)
    updated, _ = update_index_state(
        state, [{"date": "2026-08-11", "close": close_for(drawdown)}], True, NDX_RULES, NOW
    )
    assert sum(t["status"] == "pending" for t in updated["tiers"].values()) == count


def test_pending_survives_rebound_and_executed_never_retriggers():
    state = create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW)
    state, _ = update_index_state(state, [{"date": "2026-08-10", "close": 85}], True, SP_RULES, NOW)
    state["tiers"]["tier_1"]["status"] = "executed"
    state["tiers"]["tier_1"]["executed_at"] = NOW.isoformat()
    state, _ = update_index_state(state, [{"date": "2026-08-11", "close": 92}], True, SP_RULES, NOW)
    assert state["tiers"]["tier_1"]["status"] == "executed"
    assert state["tiers"]["tier_2"]["status"] == "pending"


def test_invalid_market_data_does_not_mutate_state():
    state = create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW)
    before = deepcopy(state)
    updated, archived = update_index_state(
        state, [{"date": "2026-08-11", "close": 50}], False, SP_RULES, NOW
    )
    assert updated == before
    assert archived == []


def test_new_ath_archives_cycle_and_resets_tiers():
    state = create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW)
    state, _ = update_index_state(state, [{"date": "2026-08-09", "close": 85}], True, SP_RULES, NOW)
    updated, archived = update_index_state(
        state,
        [{"date": "2026-08-10", "close": 105}, {"date": "2026-08-11", "close": 104}],
        True,
        SP_RULES,
        NOW,
    )
    assert len(archived) == 1
    assert archived[0]["cycle_id"] == state["cycle_id"]
    assert updated["ath"] == 105
    assert updated["ath_date"] == "2026-08-10"
    assert all(t["status"] == "not_triggered" for t in updated["tiers"].values())


def test_missed_run_intervening_ath_is_detected_then_latest_drawdown_applied():
    state = create_index_state("sp500", SP_RULES, 100, "2026-08-10", NOW)
    updated, archived = update_index_state(
        state,
        [{"date": "2026-08-11", "close": 105}, {"date": "2026-08-12", "close": 82}],
        True,
        SP_RULES,
        NOW,
    )
    assert len(archived) == 1
    assert updated["ath"] == 105
    assert updated["tiers"]["tier_1"]["status"] == "pending"
    assert updated["tiers"]["tier_2"]["status"] == "pending"


def test_summary_has_non_negative_next_tier_distance_and_correct_amounts():
    state = create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW)
    state, _ = update_index_state(state, [{"date": "2026-08-11", "close": 78}], True, SP_RULES, NOW)
    summary = summarize_index_state(state, 78)
    assert summary["pending_amount"] == 112000
    assert summary["remaining_amount"] == 28000
    assert summary["next_threshold"] == 0.25
    assert summary["distance_to_next"] == pytest.approx(0.03)


def test_pending_summary_keeps_original_tier_label_after_earlier_execution():
    state = create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW)
    state, _ = update_index_state(state, [{"date": "2026-08-11", "close": 84}], True, SP_RULES, NOW)
    state["tiers"]["tier_1"]["status"] = "executed"
    summary = summarize_index_state(state, 84)
    assert summary["pending_tiers"][0]["id"] == "tier_2"
    assert summary["pending_tiers"][0]["label"] == "第二档"
