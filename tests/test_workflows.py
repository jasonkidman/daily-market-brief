from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def load_workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_daily_workflow_has_timezone_concurrency_pages_and_secret():
    workflow = load_workflow("daily-report.yml")
    text = (ROOT / ".github" / "workflows" / "daily-report.yml").read_text(encoding="utf-8")
    assert 'timezone: "Asia/Shanghai"' in text
    assert workflow["concurrency"]["group"] == "investment-report-state"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"] == {"contents": "write", "pages": "write", "id-token": "write"}
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in text
    assert "path: site" in text


def test_confirm_workflow_uses_choices_same_lock_and_never_passes_deepseek_secret():
    workflow = load_workflow("confirm-drawdown.yml")
    text = (ROOT / ".github" / "workflows" / "confirm-drawdown.yml").read_text(encoding="utf-8")
    assert workflow["concurrency"]["group"] == "investment-report-state"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "type: choice" in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "python -m src.confirm_drawdown" in text
