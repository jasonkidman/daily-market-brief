"""Persistence and validation for production Stage B replay snapshots."""

from __future__ import annotations

import json
from pathlib import Path


SNAPSHOT_SCHEMA_VERSION = 1


def write_stage_b_snapshot(base_dir: Path, snapshot: dict) -> Path:
    """Write a dated Stage B snapshot under the repository data directory."""
    validate_stage_b_snapshot(snapshot)
    path = Path(base_dir) / "data" / "news_snapshots" / f"{snapshot['report_date']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_stage_b_snapshot(path: Path) -> dict:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_stage_b_snapshot(snapshot)
    return snapshot


def validate_stage_b_snapshot(snapshot: dict) -> None:
    required = {"schema_version", "report_date", "run_at", "candidate_counts", "stage_a_events", "stage_b"}
    missing = required - set(snapshot)
    if missing:
        raise ValueError(f"Stage B snapshot missing fields: {sorted(missing)}")
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage B snapshot schema: {snapshot['schema_version']}")
    if not isinstance(snapshot["report_date"], str) or not isinstance(snapshot["run_at"], str):
        raise ValueError("Stage B snapshot report_date and run_at must be strings")
    counts = snapshot["candidate_counts"]
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(key), int) or counts[key] < 0
        for key in (
            "rss_raw", "within_24h", "deduplicated", "stage_a_pre_cap",
            "stage_a_actual_input", "stage_a_events", "stage_b_input",
        )
    ):
        raise ValueError("Stage B snapshot candidate_counts is incomplete or invalid")
    if not isinstance(snapshot["stage_a_events"], list):
        raise ValueError("Stage B snapshot stage_a_events must be a list")
    stage_b = snapshot["stage_b"]
    if not isinstance(stage_b, dict) or not isinstance(stage_b.get("candidates"), list):
        raise ValueError("Stage B snapshot stage_b.candidates must be a list")
    if not isinstance(stage_b.get("recent_7_days_events", []), list):
        raise ValueError("Stage B snapshot recent_7_days_events must be a list")
    if stage_b.get("market_context") is not None and not isinstance(stage_b["market_context"], dict):
        raise ValueError("Stage B snapshot market_context must be an object or null")
