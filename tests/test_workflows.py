from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def load_workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_daily_workflow_commits_then_deploys_site_to_github_pages():
    workflow = load_workflow("daily-report.yml")
    text = (ROOT / ".github" / "workflows" / "daily-report.yml").read_text(encoding="utf-8")
    assert 'cron: "0 22 * * *"' in text
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


def test_daily_workflow_has_a_wall_clock_timeout():
    """Without this the job inherits GitHub's 6-hour default: run #89 spent
    1557s inside "Generate daily report" before being cancelled by hand."""
    workflow = load_workflow("daily-report.yml")
    timeout = workflow["jobs"]["build"]["timeout-minutes"]
    assert isinstance(timeout, int)
    assert 0 < timeout <= 30


def test_snapshot_step_resolves_todays_shanghai_date_not_arbitrary_find_order():
    """`find -print -quit` returns whatever file the filesystem yields first.
    It only worked because the directory held exactly one file; once snapshots
    are committed and retained it would pick an arbitrary historical date and
    make the commit step read the wrong report."""
    text = (ROOT / ".github" / "workflows" / "daily-report.yml").read_text(encoding="utf-8")
    step = text[text.index("Resolve generated snapshot"):text.index("Run tests")]
    # Comment lines legitimately name the old form when explaining why it went.
    executable = "\n".join(
        line for line in step.splitlines() if not line.strip().startswith("#")
    )
    assert "-print -quit" not in executable
    assert 'TZ=Asia/Shanghai date +%F' in step
    assert 'data/news_snapshots/${date}.json' in step
    # The midnight-straddle fallback must still be deterministic (newest file),
    # never an arbitrary one.
    assert "ls -t data/news_snapshots/*.json" in step


def test_snapshot_directory_is_no_longer_git_ignored():
    """Prerequisite for committing snapshots in PR 4; harmless on its own."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/news_snapshots/" not in ignored
