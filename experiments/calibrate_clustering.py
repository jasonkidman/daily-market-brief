"""Offline grid search for cluster_candidates_local's TITLE_MERGE_THRESHOLD.

DATA SOURCE NOTE (important, see IMPLEMENTATION_PLAN.md section 6.1 vs. reality):
the plan assumed the `stage-b-snapshot-*` artifact's `stage_b.candidates` was the
raw deduplicated candidate pool that Stage A clustered. It is not -- it is only
one representative article per Stage-A-produced event (event_selection_candidates
flattens one winner per event before Stage B ever sees it), so any candidate that
got merged away by the LLM and was NOT chosen as its event's representative has
no title/summary persisted in the artifact at all.

The raw per-candidate titles ARE recoverable, though: news_dedupe.py prints a
`[NEWS CANDIDATE] ... stage=dedup ... action=keep` line for every surviving
deduplicated candidate, and those lines are preserved in the GitHub Actions job
log (not the artifact). This script reconstructs the full raw title pool from
`experiments/run_logs/<date>-<run-id>.log` (fetched via `gh run view <id> --log`)
and cross-references it against the artifact's `stage_a_events` (candidate_ids
groupings) for ground truth.

Two things this script does NOT have real data for, both documented findings
from running this calibration rather than assumptions going in:

1. Raw `summary` text for merged-away candidates is never logged anywhere
   (only `title`, `source`, `published_at` are printed at the dedup stage), so
   a summary-corroborated "soft" title-match band can't be validated. Decision
   (confirmed with the user): cluster_candidates_local() does not have one --
   title similarity alone decides merges. This script reflects that.

2. STAGE_A_MAX_INPUT=50 caps each Stage A batch by composite score, so on a
   heavy news day a chunk of the deduplicated pool never reaches Stage A at
   all. For those candidates, "not grouped with X" does not mean "judged to be
   a different event from X" -- it means "never judged". Counting them as
   ground-truth negatives manufactures fake over-merges, so this script
   restricts scoring to the subset of candidates that appear in some
   stage_a_events group (the real cluster_candidates_local(), unlike the old
   LLM path, does not drop anything -- this restriction only affects what this
   script can validate against real ground truth).

Does not call any API. Run with:  python -m experiments.calibrate_clustering
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.news_events import _candidates_are_likely_duplicates  # noqa: E402

SNAPSHOTS_DIR = ROOT / "experiments" / "snapshots"
RUN_LOGS_DIR = ROOT / "experiments" / "run_logs"

TITLE_MERGE_GRID = (0.40, 0.44, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64,
                    0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88)

DEDUP_LINE = re.compile(
    r"\[NEWS CANDIDATE\] candidate_id=(?P<id>\S+) \| title=(?P<title>.*?) \| "
    r"source=(?P<source>.*?) \| published_at=(?P<published_at>\S+) \| "
    r"stage=dedup \| action=(?P<action>\w+)"
)


def reconstruct_raw_pool(log_path: Path) -> dict[str, dict]:
    """candidate_id -> {title} for every candidate that survived dedup,
    reconstructed from the job log's dedup-stage print lines."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pool: dict[str, dict] = {}
    for m in DEDUP_LINE.finditer(text):
        if m.group("action") in ("keep", "replace"):
            pool[m.group("id")] = {"candidate_id": m.group("id"), "title": m.group("title")}
    return pool


def load_days() -> list[dict]:
    """One entry per day: candidates Stage A actually evaluated (id -> title),
    plus the ground-truth same-event pairs among them from stage_a_events."""
    days = []
    for snap_path in sorted(SNAPSHOTS_DIR.glob("*/*.json")):
        date = snap_path.stem
        snap = json.loads(snap_path.read_text())
        log_matches = list(RUN_LOGS_DIR.glob(f"{date}-*.log"))
        if not log_matches:
            print(f"WARNING: no run log found for {date}, skipping")
            continue
        full_raw_pool = reconstruct_raw_pool(log_matches[0])
        if len(full_raw_pool) != snap["candidate_counts"]["deduplicated"]:
            print(
                f"WARNING: {date} reconstructed pool size {len(full_raw_pool)} != "
                f"snapshot deduplicated count {snap['candidate_counts']['deduplicated']}, "
                "ground truth for this day may be incomplete"
            )
        evaluated_ids = {cid for event in snap["stage_a_events"] for cid in event["candidate_ids"]}
        raw_pool = {cid: item for cid, item in full_raw_pool.items() if cid in evaluated_ids}
        baseline_pairs = set()
        for event in snap["stage_a_events"]:
            ids = [i for i in event["candidate_ids"] if i in raw_pool]
            for a, b in combinations(ids, 2):
                baseline_pairs.add(frozenset((a, b)))
        days.append({
            "date": date,
            "raw_pool": raw_pool,
            "capped_out_count": len(full_raw_pool) - len(raw_pool),
            "baseline_pairs": baseline_pairs,
        })
    return days


def cluster_by_title(pool: dict[str, dict], title_merge_threshold: float) -> set[frozenset]:
    """Same greedy single-linkage strategy as cluster_candidates_local."""
    clusters: list[dict] = []
    for item in pool.values():
        match = next((
            c for c in clusters
            if _candidates_are_likely_duplicates(
                {"title": c["_rep_title"]}, item, title_merge_threshold=title_merge_threshold,
            )
        ), None)
        if match is None:
            clusters.append({"_rep_title": item["title"], "ids": [item["candidate_id"]]})
            continue
        match["ids"].append(item["candidate_id"])
        if len(item["title"]) > len(match["_rep_title"]):
            match["_rep_title"] = item["title"]
    pairs = set()
    for c in clusters:
        for a, b in combinations(c["ids"], 2):
            pairs.add(frozenset((a, b)))
    return pairs


def main() -> None:
    days = load_days()
    if not days:
        print(f"No usable (snapshot, log) pairs found under {SNAPSHOTS_DIR} / {RUN_LOGS_DIR}. Nothing to calibrate.")
        return

    total_pool = sum(len(d["raw_pool"]) for d in days)
    total_capped = sum(d["capped_out_count"] for d in days)
    total_baseline_pairs = sum(len(d["baseline_pairs"]) for d in days)
    print(f"Loaded {len(days)} days: {[d['date'] for d in days]}")
    print(f"Candidates actually judged by Stage A (post STAGE_A_MAX_INPUT cap): {total_pool}")
    print(f"Candidates excluded (capped out before Stage A, no ground truth): {total_capped}")
    print(f"Ground-truth same-event pairs (from stage_a_events): {total_baseline_pairs}\n")

    results = []
    for title_merge in TITLE_MERGE_GRID:
        over_merge_total = 0
        under_merge_total = 0
        examples = []
        for d in days:
            predicted = cluster_by_title(d["raw_pool"], title_merge)
            over = predicted - d["baseline_pairs"]
            under = d["baseline_pairs"] - predicted
            over_merge_total += len(over)
            under_merge_total += len(under)
            for pair in over:
                a, b = tuple(pair)
                examples.append((d["date"], d["raw_pool"][a]["title"], d["raw_pool"][b]["title"]))
        results.append({"title_merge": title_merge, "over_merge": over_merge_total,
                         "under_merge": under_merge_total, "examples": examples})

    print(f"{'title_merge':>12} {'over_merge':>11} {'under_merge':>12}")
    for r in results:
        print(f"{r['title_merge']:>12} {r['over_merge']:>11} {r['under_merge']:>12}")

    print("\n[OVER-MERGE SUSPECTS] (first occurrence per pair, for manual review)")
    seen = set()
    for r in results:
        for date, title_a, title_b in r["examples"]:
            key = (title_a, title_b)
            if key in seen:
                continue
            seen.add(key)
            print(f"[OVER-MERGE] threshold={r['title_merge']} | date={date}")
            print(f"    A: \"{title_a}\"")
            print(f"    B: \"{title_b}\"")

    zero = [r for r in results if r["over_merge"] == 0]
    if zero:
        loosest = min(zero, key=lambda r: (r["title_merge"], r["under_merge"]))
        print(f"\n[RECOMMENDATION] Loosest zero-over-merge title_merge_threshold: {loosest['title_merge']}")
    else:
        smallest_over = min(r["over_merge"] for r in results)
        print(f"\n[RECOMMENDATION] No threshold achieves literal zero over-merge; smallest is "
              f"{smallest_over}. Read the [OVER-MERGE SUSPECTS] above by hand before trusting any "
              f"threshold -- do not pick one without confirming those pairs are real duplicates.")


if __name__ == "__main__":
    main()
