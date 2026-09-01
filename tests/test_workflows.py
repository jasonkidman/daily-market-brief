from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def load_workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_daily_workflow_commits_then_deploys_site_to_github_pages():
    workflow = load_workflow("daily-report.yml")
    text = (ROOT / ".github" / "workflows" / "daily-report.yml").read_text(encoding="utf-8")
    assert 'cron: "0 1 * * *"' in text
    assert "timezone:" not in text
    assert workflow["concurrency"]["group"] == "investment-report-state"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"] == {
        "contents": "write",
        "pages": "write",
        "id-token": "write",
    }
    assert workflow["jobs"]["build"]["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in text
    assert "git add data/reports state site" in text
    assert "actions/upload-artifact@v4" in text
    assert "stage-b-snapshot-${{ steps.snapshot.outputs.date }}-${{ github.run_id }}" in text
    assert 'path: ${{ steps.snapshot.outputs.path }}' in text
    assert "EDGEONE" not in text
    assert "edgeone" not in text

    # Pages deployment must run only after the report is committed, and only
    # target site/ (never repo root), gated on the same core-market guard
    # that already decides whether the run's output is durable.
    commit_idx = text.index("Commit durable report and state changes")
    configure_idx = text.index("actions/configure-pages@v5")
    upload_idx = text.index("actions/upload-pages-artifact@v3")
    deploy_idx = text.index("actions/deploy-pages@v4")
    assert commit_idx < configure_idx < upload_idx < deploy_idx
    assert "path: ./site" in text

    steps = workflow["jobs"]["build"]["steps"]
    pages_steps = [
        s for s in steps
        if s.get("uses", "").startswith((
            "actions/configure-pages",
            "actions/upload-pages-artifact",
            "actions/deploy-pages",
        ))
    ]
    assert len(pages_steps) == 3
    assert all(s.get("if") == "steps.commit.outputs.core_valid == 'true'" for s in pages_steps)


def test_confirm_workflow_uses_choices_same_lock_and_edgeone_commit_flow():
    workflow = load_workflow("confirm-drawdown.yml")
    text = (ROOT / ".github" / "workflows" / "confirm-drawdown.yml").read_text(encoding="utf-8")
    assert workflow["concurrency"]["group"] == "investment-report-state"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"] == {"contents": "write"}
    assert "type: choice" in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "python -m src.confirm_drawdown" in text
    assert "git add state data/reports site" in text
    assert "configure-pages" not in text
    assert "upload-pages-artifact" not in text
    assert "deploy-pages" not in text
    assert "github-pages" not in text


def test_constituent_reference_workflow_is_weekly_manual_and_does_not_deploy_pages():
    path = __import__("pathlib").Path(__file__).parents[1] / ".github/workflows/update-sp500-constituents.yml"
    text = path.read_text(encoding="utf-8")

    assert 'cron: "30 0 * * 1"' in text
    assert "workflow_dispatch:" in text
    assert "python -m src.update_sp500_constituents" in text
    assert "python -m pytest -v tests/test_constituents.py" in text
    assert "data/reference/sp500_constituents.csv" in text
    assert "deploy-pages" not in text
    assert "upload-pages-artifact" not in text
