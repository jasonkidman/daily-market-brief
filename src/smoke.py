"""Small dependency-free smoke validation for generated artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Matches API-key-shaped tokens (e.g. "sk-abcDEF0123456789..."), not any
# incidental "sk-" substring -- real news URLs/slugs (e.g. ".../risk-label/")
# legitimately contain "sk-" as part of an ordinary word.
_SECRET_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")


REQUIRED_HTML = (
    "Daily Market Brief",
    "今日市场一句话",
    "今日重要新闻",
    "市场环境",
    "市场广度",
    "长期投资与风险管理",
    "本报告仅供参考，不构成投资建议。",
)


def validate_generated(base_dir: Path) -> None:
    base_dir = Path(base_dir)
    reports = sorted((base_dir / "data" / "reports").glob("????-??-??.json"))
    if not reports:
        raise SystemExit("Smoke check failed: no report JSON.")
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    required_keys = {"report_date", "generated_at", "market_date", "status", "market", "drawdown", "news", "market_summary", "warnings"}
    if not required_keys.issubset(report):
        raise SystemExit("Smoke check failed: report JSON schema is incomplete.")
    html = (base_dir / "site" / "index.html").read_text(encoding="utf-8")
    missing = [text for text in REQUIRED_HTML if text not in html]
    if missing:
        raise SystemExit(f"Smoke check failed: missing HTML content {missing}.")
    if "DEEPSEEK_API_KEY" in html or _SECRET_KEY_PATTERN.search(html):
        raise SystemExit("Smoke check failed: possible secret material in HTML.")
    print(f"Smoke check passed: {reports[-1].name} and site/index.html")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    validate_generated(args.base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
