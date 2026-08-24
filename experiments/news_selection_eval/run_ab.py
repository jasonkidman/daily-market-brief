from __future__ import annotations

import json
import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.deepseek_client import DEEPSEEK_MAX_ATTEMPTS, DEEPSEEK_TIMEOUT, validate_selection
from src.news_dedupe import dedupe_candidates
from src.news_events import build_event_representatives, cluster_news_events, event_selection_candidates
from src.rss_news import fetch_candidates, filter_final_candidates
from src.news_prompt import SYSTEM_PROMPT as NEW_PROMPT


OUT = ROOT / "experiments" / "news_selection_eval"


def old_prompt() -> str:
    return subprocess.check_output(["git", "show", "HEAD:src/news_prompt.py"], cwd=ROOT, text=True).split('SYSTEM_PROMPT = """', 1)[1].split('"""', 1)[0]


def chat(prompt: str, payload: str, key: str) -> str:
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=DEEPSEEK_TIMEOUT, max_retries=0)
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.15,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": payload}],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def run_group(label: str, prompt: str, candidates: list[dict], key: str) -> dict:
    events = [{field: candidate.get(field) for field in (
        "candidate_id", "event_summary", "topic_group", "event_category", "source_channel", "source", "title", "summary", "published_at",
    )} for candidate in candidates]
    payload = json.dumps({"events": events, "recent_7_days_events": []}, ensure_ascii=False)
    results = []
    started = time.monotonic()
    for attempt in range(DEEPSEEK_MAX_ATTEMPTS):
        try:
            raw = chat(prompt, payload, key)
            raw_path = OUT / f"2026-08-24-{label}-raw-{attempt + 1}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            parsed = json.loads(raw)
            selected = validate_selection(parsed, candidates)
            result = {"label": label, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2), "status": "success", "raw": parsed, "selected": selected}
            (OUT / f"2026-08-24-{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result
        except Exception as exc:
            results.append({"attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2), "error": f"{type(exc).__name__}: {exc}"})
    result = {"label": label, "status": "failed", "attempts": results, "selected": []}
    (OUT / f"2026-08-24-{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    key = os.environ["DEEPSEEK_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        candidates = snapshot["events"]
    else:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        import yaml
        sources = yaml.safe_load((ROOT / "config" / "news_sources.yaml").read_text(encoding="utf-8"))["sources"]
        raw, warnings = fetch_candidates(sources, now, hours=30)
        window = filter_final_candidates(raw, now)
        deduped = dedupe_candidates(window)
        events, stage_a_warning = cluster_news_events(deduped, key)
        candidates = event_selection_candidates(build_event_representatives(events, deduped))
        snapshot = {
            "run_at": now.isoformat(), "raw_candidate_count": len(raw), "final_24h_candidate_count": len(window),
            "deduped_candidate_count": len(deduped), "stage_a_event_count": len(events), "warnings": warnings,
            "stage_a_warning": stage_a_warning, "events": candidates,
        }
        (OUT / "2026-08-24-stage-a-snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a = run_group("a-old-prompt", old_prompt(), candidates, key)
    b = run_group("b-new-prompt", NEW_PROMPT, candidates, key)
    print(json.dumps({"snapshot": snapshot, "a": a, "b": b}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
