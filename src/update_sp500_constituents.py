"""Low-frequency command for refreshing the local State Street SPY reference."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

from .constituents import parse_state_street_holdings, update_reference


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "daily-market-brief/1.0 (github.com/jasonkidman/daily-market-brief)"


def fetch_state_street_rows(url: str) -> list[dict]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return parse_state_street_holdings(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config" / "market_breadth.yaml").read_text(encoding="utf-8"))
    constituents = config["constituents"]
    reference_path = args.base_dir / constituents["reference_file"]
    changed = update_reference(
        reference_path,
        lambda: fetch_state_street_rows(constituents["download_url"]),
        minimum_count=constituents.get("minimum_constituents", 480),
    )
    print(f"Updated {reference_path}" if changed else f"No change for {reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
