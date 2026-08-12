import json

import pytest

import src.main as main
from src.main import assess_market_validity, generate_daily_report


def test_offline_fixture_runs_complete_pipeline(tmp_path):
    result = generate_daily_report(base_dir=tmp_path, offline_fixture=True, report_date="2026-08-12")
    report_path = tmp_path / "data" / "reports" / "2026-08-12.json"
    index_path = tmp_path / "site" / "index.html"
    assert report_path.exists()
    assert index_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_date"] == "2026-08-12"
    assert report["market_date"] == "2026-08-11"
    assert set(report["market"]) == {"sp500", "nasdaq100", "dow"}
    assert set(report["market_context"]) == {"russell2000", "vix", "dxy", "us10y"}
    assert report["market_context"]["russell2000"]["daily_return"] == pytest.approx(0.01)
    assert report["market_context"]["vix"]["daily_return"] == pytest.approx(0.10)
    assert report["market_context"]["dxy"]["daily_return"] == pytest.approx(0.005)
    assert report["market_context"]["us10y"]["yield_change_bp"] == pytest.approx(8)
    assert report["market_signals"]["vix_daily_return"] == pytest.approx(0.10)
    assert report["market_signals"]["us10y_bp_change"] == pytest.approx(8)
    assert report["market_breadth"]["stocks"]["advance_ratio"] == pytest.approx(0.60)
    assert report["market_breadth"]["sectors"]["advance_ratio"] == pytest.approx(7 / 11)
    assert report["market_breadth"]["health"]["score"] == pytest.approx(0.61454545)
    assert report["market_breadth"]["health"]["level"] == "mixed"
    assert "Daily Market Brief" in index_path.read_text(encoding="utf-8")
    assert result == report_path


def test_drawdown_validity_ignores_dow_and_context_failures():
    core = {key: {"valid": True} for key in ("sp500", "nasdaq100", "dow")}
    context = {key: {"valid": True} for key in ("russell2000", "vix", "dxy", "us10y")}

    core["dow"]["valid"] = False
    for key in context:
        broken_context = {name: dict(value) for name, value in context.items()}
        broken_context[key]["valid"] = False
        validity = assess_market_validity(core, broken_context)
        assert validity["drawdown_by_index"] == {"sp500": True, "nasdaq100": True}
        assert validity["drawdown_market_valid"] is True


def test_drawdown_validity_is_independent_for_sp500_and_nasdaq100():
    context = {key: {"valid": True} for key in ("russell2000", "vix", "dxy", "us10y")}
    core = {"sp500": {"valid": False}, "nasdaq100": {"valid": True}, "dow": {"valid": True}}
    assert assess_market_validity(core, context)["drawdown_by_index"] == {
        "sp500": False, "nasdaq100": True,
    }
    core["sp500"]["valid"], core["nasdaq100"]["valid"] = True, False
    assert assess_market_validity(core, context)["drawdown_by_index"] == {
        "sp500": True, "nasdaq100": False,
    }


def test_breadth_failure_does_not_block_drawdown_state_updates(tmp_path, monkeypatch):
    def fail_breadth(*args, **kwargs):
        raise RuntimeError("breadth source unavailable")

    monkeypatch.setattr(main, "build_offline_market_breadth", fail_breadth)
    generate_daily_report(base_dir=tmp_path, offline_fixture=True, report_date="2026-08-12")

    report = json.loads((tmp_path / "data" / "reports" / "2026-08-12.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "state" / "drawdown_state.json").read_text(encoding="utf-8"))
    assert report["market_breadth"]["health"]["valid"] is False
    assert set(state["indices"]) == {"sp500", "nasdaq100"}
