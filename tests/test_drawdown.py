from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.drawdown import (
    compute_suggested_topup,
    create_index_state,
    reserve_used_total,
    summarize_index_state,
    update_drawdown_state,
    update_index_state,
)


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


@pytest.mark.parametrize(
    ("invalid_key", "valid_key", "valid_close", "expected_pending"),
    [
        ("sp500", "nasdaq100", 80, ["tier_1", "tier_2"]),
        ("nasdaq100", "sp500", 80, ["tier_1", "tier_2", "tier_3"]),
    ],
)
def test_one_invalid_drawdown_index_does_not_block_the_other_index(
    invalid_key, valid_key, valid_close, expected_pending
):
    state = {
        "version": 1,
        "indices": {
            "sp500": create_index_state("sp500", SP_RULES, 100, "2026-01-01", NOW),
            "nasdaq100": create_index_state("nasdaq100", NDX_RULES, 100, "2026-01-01", NOW),
        },
        "executions": [],
    }
    histories = {invalid_key: [{"date": "2026-08-11", "close": 50}],
                 valid_key: [{"date": "2026-08-11", "close": valid_close}]}
    snapshots = {
        invalid_key: {"valid": False, "close": 50},
        valid_key: {"valid": True, "close": valid_close},
    }
    rules = {"sp500": SP_RULES, "nasdaq100": NDX_RULES}

    updated, archived = update_drawdown_state(
        state,
        histories,
        {invalid_key: False, valid_key: True},
        rules,
        snapshots,
        NOW,
    )

    assert updated["indices"][invalid_key] == state["indices"][invalid_key]
    pending = [key for key, tier in updated["indices"][valid_key]["tiers"].items()
               if tier["status"] == "pending"]
    assert pending == expected_pending
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


# --- reserve_used_total / compute_suggested_topup (2026-08 reserve restructure) ---

def test_reserve_used_total_sums_executions_across_indices():
    executions = [
        {"index": "nasdaq100", "amount": 7500, "tier": None},
        {"index": "sp500", "amount": 28000, "tier": "tier_1"},
    ]
    assert reserve_used_total(executions) == 35500


def test_reserve_used_total_empty_is_zero():
    assert reserve_used_total([]) == 0


def test_suggested_topup_subtracts_historical_investment_for_that_index():
    """The real 2026-08-29 scenario: Nasdaq-100 has $7,500 already deployed via a
    manually backfilled executions entry (pre-dating the tier system), before any
    tier has actually triggered under the current 200,000-pool plan."""
    summary = {"pending_amount": 0, "executed_amount": 0}
    executions = [{"index": "nasdaq100", "amount": 7500, "tier": None}]
    assert compute_suggested_topup(summary, executions, "nasdaq100", reserve_remaining=192500) == 0


def test_suggested_topup_nets_against_cumulative_tier_target_once_triggered():
    """Once nasdaq100 tier_1 (12,000) triggers, the suggestion must be the gap to
    the 7,500 already spent -- not the full tier amount again."""
    summary = {"pending_amount": 12000, "executed_amount": 0}
    executions = [{"index": "nasdaq100", "amount": 7500, "tier": None}]
    assert compute_suggested_topup(summary, executions, "nasdaq100", reserve_remaining=192500) == 4500


def test_suggested_topup_never_negative():
    summary = {"pending_amount": 0, "executed_amount": 0}
    executions = [{"index": "nasdaq100", "amount": 50000, "tier": None}]
    assert compute_suggested_topup(summary, executions, "nasdaq100", reserve_remaining=150000) == 0


def test_suggested_topup_capped_by_reserve_remaining():
    summary = {"pending_amount": 42000, "executed_amount": 0}
    executions = []
    assert compute_suggested_topup(summary, executions, "sp500", reserve_remaining=1000) == 1000


def test_suggested_topup_ignores_other_indices_executions():
    summary = {"pending_amount": 28000, "executed_amount": 0}
    executions = [{"index": "nasdaq100", "amount": 7500, "tier": None}]
    assert compute_suggested_topup(summary, executions, "sp500", reserve_remaining=192500) == 28000


def test_suggested_topup_is_deterministic_across_repeated_calls():
    """Repeated report generation re-reads the same persisted executions ledger and
    must get the same suggested amount every time -- the $7,500 backfill must never
    be counted twice just because the daily report runs again."""
    summary = {"pending_amount": 12000, "executed_amount": 0}
    executions = [{"index": "nasdaq100", "amount": 7500, "tier": None}]
    first = compute_suggested_topup(summary, executions, "nasdaq100", reserve_remaining=192500)
    second = compute_suggested_topup(summary, executions, "nasdaq100", reserve_remaining=192500)
    assert first == second == 4500
