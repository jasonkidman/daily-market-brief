"""Persistent, independent drawdown-cycle state machines."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Any, Optional


def _cycle_id(index_key: str, ath_date: str, ath: float) -> str:
    return f"{index_key}-{ath_date}-{round(ath * 10000)}"


def create_index_state(index_key: str, rules: dict[str, Any], ath: float, ath_date: str,
                       now: datetime) -> dict[str, Any]:
    tiers = {}
    for tier in rules["tiers"]:
        tiers[tier["id"]] = {
            "threshold": float(tier["threshold"]),
            "allocation": float(tier["allocation"]),
            "amount": int(round(float(rules["pool"]) * float(tier["allocation"]))),
            "status": "not_triggered",
            "triggered_at": None,
            "executed_at": None,
        }
    return {
        "index": index_key,
        "name": rules["name"],
        "cycle_id": _cycle_id(index_key, ath_date, ath),
        "started_at": now.isoformat(),
        "ath": float(ath),
        "ath_date": ath_date,
        "last_market_date": ath_date,
        "last_updated_at": now.isoformat(),
        "current_close": float(ath),
        "current_drawdown": 0.0,
        "pool": int(rules["pool"]),
        "tiers": tiers,
    }


def _normalize_rows(close_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in close_rows:
        row_date = row["date"]
        if isinstance(row_date, datetime):
            row_date = row_date.date()
        if isinstance(row_date, date):
            row_date = row_date.isoformat()
        normalized.append({"date": str(row_date)[:10], "close": float(row["close"])})
    return sorted(normalized, key=lambda item: item["date"])


def update_index_state(existing: dict[str, Any], close_rows: list[dict[str, Any]],
                       market_data_valid: bool, rules: dict[str, Any], now: datetime
                       ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not market_data_valid:
        return deepcopy(existing), []
    rows = _normalize_rows(close_rows)
    if not rows:
        return deepcopy(existing), []
    state = deepcopy(existing)
    archives = []
    last_seen = state.get("last_market_date", state["ath_date"])
    unseen = [row for row in rows if row["date"] > last_seen]
    new_ath_rows = [row for row in unseen if row["close"] > float(state["ath"])]
    if new_ath_rows:
        highest = max(new_ath_rows, key=lambda row: row["close"])
        archived = deepcopy(state)
        archived["ended_at"] = now.isoformat()
        archived["ended_by_new_ath"] = {"ath": highest["close"], "date": highest["date"]}
        archives.append(archived)
        state = create_index_state(state["index"], rules, highest["close"], highest["date"], now)
    latest = rows[-1]
    drawdown = max(0.0, 1 - latest["close"] / float(state["ath"]))
    for tier in state["tiers"].values():
        if tier["status"] == "not_triggered" and drawdown + 1e-12 >= tier["threshold"]:
            tier["status"] = "pending"
            tier["triggered_at"] = now.isoformat()
    state.update({
        "current_close": latest["close"],
        "current_drawdown": drawdown,
        "last_market_date": latest["date"],
        "last_updated_at": now.isoformat(),
    })
    return state, archives


def summarize_index_state(state: dict[str, Any], current_close: Optional[float] = None) -> dict[str, Any]:
    close = float(current_close if current_close is not None else state["current_close"])
    drawdown = max(0.0, 1 - close / float(state["ath"]))
    tier_labels = {"tier_1": "第一档", "tier_2": "第二档", "tier_3": "第三档", "tier_4": "第四档"}
    enriched = {key: dict({"id": key, "label": tier_labels[key]}, **tier) for key, tier in state["tiers"].items()}
    pending = [tier for tier in enriched.values() if tier["status"] == "pending"]
    executed = [tier for tier in enriched.values() if tier["status"] == "executed"]
    not_triggered = [tier for tier in enriched.values() if tier["status"] == "not_triggered"]
    next_tier = not_triggered[0] if not_triggered else None
    pending_amount = sum(tier["amount"] for tier in pending)
    executed_amount = sum(tier["amount"] for tier in executed)
    if pending:
        status = "pending"
        status_label = "🔴 加仓信号已触发 · 待执行"
    elif next_tier and max(0.0, next_tier["threshold"] - drawdown) <= 0.02 + 1e-12:
        status = "near"
        status_label = "🟠 接近触发"
    elif executed and not next_tier:
        status = "executed"
        status_label = "✅ 已执行"
    else:
        status = "normal"
        status_label = "🟢 未触发"
    return {
        **state,
        "current_close": close,
        "current_drawdown": drawdown,
        "pending_tiers": pending,
        "executed_tiers": executed,
        "pending_amount": pending_amount,
        "executed_amount": executed_amount,
        "remaining_amount": max(0, int(state["pool"]) - pending_amount - executed_amount),
        "next_threshold": next_tier["threshold"] if next_tier else None,
        "distance_to_next": max(0.0, next_tier["threshold"] - drawdown) if next_tier else None,
        "status": status,
        "status_label": status_label,
    }


def reserve_used_total(executions: list[dict[str, Any]]) -> int:
    """Total reserve money actually deployed so far, from the single source of truth:
    the `executions` ledger. This covers both formal tier confirmations (appended by
    `confirm_drawdown.confirm_tier`) and any manually backfilled historical deployment
    (an executions entry with `tier: None`, e.g. money spent before this state machine
    existed) -- there is deliberately no second, separately-maintained "used" number."""
    return sum(int(item.get("amount", 0)) for item in executions)


def compute_suggested_topup(index_summary: dict[str, Any], executions: list[dict[str, Any]],
                            index_key: str, reserve_remaining: int) -> int:
    """Suggested new deployment for one index right now.

    cumulative_target is what the drawdown tiers reached so far call for in total
    (pending + executed tier amounts, still computed from the original 200,000-pool
    tier plan -- unaffected by how much of the reserve remains). already_invested is
    this index's own share of the executions ledger, which already includes any
    manually backfilled historical deployment. The gap between the two is what's left
    to invest for this index's current tier progress; it is never suggested twice
    (repeated report runs re-read the same persisted executions ledger and get the
    same gap), never negative, and never more than what the reserve actually has left.
    """
    cumulative_target = int(index_summary.get("pending_amount", 0)) + int(index_summary.get("executed_amount", 0))
    already_invested = sum(
        int(item.get("amount", 0)) for item in executions if item.get("index") == index_key
    )
    suggested = cumulative_target - already_invested
    return max(0, min(suggested, max(0, int(reserve_remaining))))


def update_drawdown_state(state: dict[str, Any], histories: dict[str, list],
                          market_validity: dict[str, bool], rules: dict[str, Any],
                          snapshots: dict[str, Any], now: datetime
                          ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated = deepcopy(state)
    updated.setdefault("version", 1)
    updated.setdefault("indices", {})
    updated.setdefault("executions", [])
    archives = []
    for key in ("sp500", "nasdaq100"):
        if key not in updated["indices"]:
            if not market_validity.get(key) or key not in snapshots:
                continue
            snap = snapshots[key]
            updated["indices"][key] = create_index_state(key, rules[key], snap["ath"], snap["ath_date"], now)
        new_state, closed = update_index_state(
            updated["indices"][key], histories.get(key, []), market_validity.get(key, False), rules[key], now
        )
        updated["indices"][key] = new_state
        archives.extend(closed)
    updated["updated_at"] = now.isoformat()
    return updated, archives
