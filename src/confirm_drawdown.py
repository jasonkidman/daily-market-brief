"""Manual pending-tier confirmation and CLI entry point."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .drawdown import summarize_index_state
from .renderer import render_site


class ConfirmationError(ValueError):
    """Raised when a tier cannot be marked executed."""


def confirm_tier(state: dict, index_key: str, tier_key: str, now: datetime) -> tuple[dict, bool]:
    updated = deepcopy(state)
    try:
        index_state = updated["indices"][index_key]
        tier = index_state["tiers"][tier_key]
    except KeyError as exc:
        raise ConfirmationError("指数或档位不存在。") from exc
    if tier["status"] == "not_triggered":
        raise ConfirmationError("该档位尚未触发，禁止标记为已执行。")
    if tier["status"] == "executed":
        return updated, False
    tier["status"] = "executed"
    tier["executed_at"] = now.isoformat()
    updated.setdefault("executions", []).append({
        "executed_at": now.isoformat(),
        "amount": tier["amount"],
        "index": index_key,
        "tier": tier_key,
        "cycle_id": index_state["cycle_id"],
    })
    updated["updated_at"] = now.isoformat()
    return updated, True


def confirm_and_render(base_dir: Path, index_key: str, tier_key: str, now: datetime,
                       render: bool = True) -> bool:
    base_dir = Path(base_dir)
    state_path = base_dir / "state" / "drawdown_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    updated, changed = confirm_tier(state, index_key, tier_key, now)
    if not changed:
        return False
    state_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reports_dir = base_dir / "data" / "reports"
    report_paths = sorted(reports_dir.glob("????-??-??.json"))
    if not report_paths:
        raise ConfirmationError("没有可更新的日报。")
    latest_path = report_paths[-1]
    report = json.loads(latest_path.read_text(encoding="utf-8"))
    report["drawdown"] = {
        key: summarize_index_state(value) for key, value in updated.get("indices", {}).items()
    }
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if render:
        root = Path(__file__).resolve().parents[1]
        render_site(reports_dir, root / "templates" / "report.html", root / "static" / "style.css", base_dir / "site")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", choices=("sp500", "nasdaq100"))
    parser.add_argument("tier", choices=("tier_1", "tier_2", "tier_3", "tier_4"))
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    changed = confirm_and_render(
        args.base_dir, args.index, args.tier, datetime.now(ZoneInfo("Asia/Shanghai")), render=True
    )
    if changed:
        print(f"已确认 {args.index} {args.tier}。")
    else:
        print("该档位已经执行，无需重复修改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
