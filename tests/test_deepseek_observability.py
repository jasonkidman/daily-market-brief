from types import SimpleNamespace

import pytest

from src.deepseek_client import DeepSeekUsageTracker, estimate_cost_cny, extract_usage


def test_usage_tracker_prices_a_request_from_cache_split_and_completion_tokens(capsys):
    tracker = DeepSeekUsageTracker()
    usage = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        prompt_cache_hit_tokens=200,
        prompt_cache_miss_tokens=1000,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=80),
    )

    tracker.record_success(
        stage="Stage A",
        attempt=1,
        model="deepseek-chat",
        thinking_enabled=False,
        reasoning_effort=None,
        elapsed_ms=125,
        usage=extract_usage(usage),
    )
    tracker.log_summary()

    output = capsys.readouterr().out
    assert "stage=Stage A" in output
    assert "attempt=1" in output
    assert "prompt_tokens=1200" in output
    assert "prompt_cache_hit_tokens=200" in output
    assert "reasoning_tokens=80" in output
    assert "estimated_cost_cny=0.001604" in output
    assert "total_estimated_cost_cny=0.001604" in output


def test_usage_tracker_marks_cost_unavailable_without_cache_split(capsys):
    tracker = DeepSeekUsageTracker()
    usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=300, total_tokens=1500)

    tracker.record_success(
        stage="Layer 2",
        attempt=1,
        model="deepseek-chat",
        thinking_enabled=False,
        reasoning_effort=None,
        elapsed_ms=125,
        usage=extract_usage(usage),
    )
    tracker.log_summary()

    output = capsys.readouterr().out
    assert "prompt_cache_hit_tokens=unavailable" in output
    assert "estimated_cost_cny=unavailable" in output
    assert "total_estimated_cost_cny=unavailable" in output


def test_usage_tracker_marks_unknown_model_cost_unavailable_but_keeps_usage(capsys):
    tracker = DeepSeekUsageTracker()
    usage = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        prompt_cache_hit_tokens=200,
        prompt_cache_miss_tokens=1000,
    )

    tracker.record_success(
        stage="Stage B",
        attempt=1,
        model="future-model-without-price",
        thinking_enabled=True,
        reasoning_effort="high",
        elapsed_ms=125,
        usage=extract_usage(usage),
    )

    output = capsys.readouterr().out
    assert "model=future-model-without-price" in output
    assert "prompt_tokens=1200" in output
    assert "estimated_cost_cny=unavailable" in output


def test_cost_uses_the_price_mapping_for_the_response_model():
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "total_tokens": 1500,
        "prompt_cache_hit_tokens": 200,
        "prompt_cache_miss_tokens": 1000,
        "reasoning_tokens": None,
    }

    assert estimate_cost_cny("deepseek-v4-flash", usage) == pytest.approx(0.001604)
    assert estimate_cost_cny("deepseek-v4-pro", usage) == pytest.approx(0.004805)
    assert estimate_cost_cny("unmapped-model", usage) is None


def test_usage_tracker_records_validation_failure_and_retry_reason(capsys):
    tracker = DeepSeekUsageTracker()

    tracker.record_validation_failure("Stage B", 2, ValueError("rank 不合法"))

    output = capsys.readouterr().out
    assert "stage=Stage B" in output
    assert "attempt=2" in output
    assert "validation_failure=ValueError: rank 不合法" in output


@pytest.mark.parametrize("value,expected", [
    ({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, 1),
    (SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3), 1),
])
def test_extract_usage_reads_mapping_and_sdk_usage_objects(value, expected):
    assert extract_usage(value)["prompt_tokens"] == expected
