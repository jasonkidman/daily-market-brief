"""Market Sentiment 0-100 composite score.

Fixed weights:
    VIX 情绪        35%
    市场广度        30%
    指数动量        20%
    风险偏好        15%

Every sub-score is derived from real, already-fetched market data:
    - vix_score            VIX close level (absolute, not daily change) mapped
                            through a fixed fear/greed piecewise curve.
    - breadth_score         market_breadth.health.score (0-1) rescaled to 0-100.
    - momentum_score        average close-over-close return for S&P 500 and
                            Nasdaq-100 over the available lookback window,
                            mapped through a clipped linear scale.
    - risk_appetite_score   Russell 2000 daily return relative to S&P 500
                            (small-cap vs large-cap relative strength) mapped
                            through a clipped linear scale.

If any sub-score cannot be computed from real data, `market_sentiment_score`
and `market_sentiment_label` are returned as None ("--") rather than being
estimated from the remaining sub-scores.
"""

from __future__ import annotations

from typing import Any, Optional


WEIGHTS = {
    "vix_score": 0.35,
    "breadth_score": 0.30,
    "momentum_score": 0.20,
    "risk_appetite_score": 0.15,
}

# (lower_bound_inclusive, label), checked from highest to lowest.
LABEL_BANDS = (
    (90, "极度亢奋"),
    (75, "亢奋"),
    (60, "偏乐观"),
    (40, "中性"),
    (20, "偏谨慎"),
    (0, "恐慌"),
)

# VIX absolute-level anchor points (vix_close -> score), industry-standard
# fear/greed style thresholds: low VIX = complacency/greed (high score),
# high VIX = fear (low score).
_VIX_ANCHORS = ((10.0, 100.0), (20.0, 50.0), (30.0, 20.0), (40.0, 5.0), (60.0, 0.0))

_MOMENTUM_LOOKBACK_DAYS = 20
_MOMENTUM_RANGE = 0.10  # +/-10% over the lookback window maps to 0..100
_RISK_APPETITE_RANGE = 0.01  # +/-1% relative daily return maps to 0..100


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _linear_score(value: float, low_value: float, high_value: float) -> float:
    """Map value in [low_value, high_value] linearly to [0, 100], clipped."""
    if high_value == low_value:
        return 50.0
    ratio = (value - low_value) / (high_value - low_value)
    return _clip(ratio, 0.0, 1.0) * 100.0


def _piecewise_linear(x: float, anchors: tuple[tuple[float, float], ...]) -> float:
    """Interpolate x through ascending-x anchors, clamped at both ends."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t
    return anchors[-1][1]


def score_vix(vix_snapshot: Optional[dict[str, Any]]) -> Optional[float]:
    if not vix_snapshot or not vix_snapshot.get("valid"):
        return None
    close = vix_snapshot.get("close")
    if close is None:
        return None
    return round(_piecewise_linear(float(close), _VIX_ANCHORS), 1)


def score_breadth(breadth_health: Optional[dict[str, Any]]) -> Optional[float]:
    if not breadth_health or not breadth_health.get("valid"):
        return None
    score = breadth_health.get("score")
    if score is None:
        return None
    return round(_clip(float(score) * 100.0, 0.0, 100.0), 1)


def score_momentum(core_histories: Optional[dict[str, list[dict[str, Any]]]],
                    lookback: int = _MOMENTUM_LOOKBACK_DAYS) -> Optional[float]:
    if not core_histories:
        return None
    changes = []
    for key in ("sp500", "nasdaq100"):
        rows = core_histories.get(key) or []
        cleaned = sorted(
            ({"date": row["date"], "close": float(row["close"])} for row in rows
             if row.get("close") is not None),
            key=lambda row: row["date"],
        )
        if len(cleaned) < 2:
            continue
        base_index = max(0, len(cleaned) - 1 - lookback)
        base_close = cleaned[base_index]["close"]
        latest_close = cleaned[-1]["close"]
        if base_close <= 0:
            continue
        changes.append(latest_close / base_close - 1.0)
    if not changes:
        return None
    average_change = sum(changes) / len(changes)
    return round(_linear_score(average_change, -_MOMENTUM_RANGE, _MOMENTUM_RANGE), 1)


def score_risk_appetite(context_snapshots: Optional[dict[str, Any]],
                        core_snapshots: Optional[dict[str, Any]]) -> Optional[float]:
    russell = (context_snapshots or {}).get("russell2000") or {}
    sp500 = (core_snapshots or {}).get("sp500") or {}
    if not russell.get("valid") or not sp500.get("valid"):
        return None
    russell_return = russell.get("daily_return")
    sp500_return = sp500.get("daily_return")
    if russell_return is None or sp500_return is None:
        return None
    relative = float(russell_return) - float(sp500_return)
    return round(_linear_score(relative, -_RISK_APPETITE_RANGE, _RISK_APPETITE_RANGE), 1)


def label_for_score(score: float) -> str:
    for lower_bound, label in LABEL_BANDS:
        if score >= lower_bound:
            return label
    return LABEL_BANDS[-1][1]


def calculate_market_sentiment(
    context_snapshots: Optional[dict[str, Any]],
    core_snapshots: Optional[dict[str, Any]],
    core_histories: Optional[dict[str, list[dict[str, Any]]]],
    breadth_health: Optional[dict[str, Any]],
) -> dict[str, Any]:
    vix_score = score_vix((context_snapshots or {}).get("vix"))
    breadth_score = score_breadth(breadth_health)
    momentum_score = score_momentum(core_histories)
    risk_appetite_score = score_risk_appetite(context_snapshots, core_snapshots)

    result = {
        "vix_score": vix_score,
        "breadth_score": breadth_score,
        "momentum_score": momentum_score,
        "risk_appetite_score": risk_appetite_score,
        "market_sentiment_score": None,
        "market_sentiment_label": None,
    }
    if any(sub is None for sub in (vix_score, breadth_score, momentum_score, risk_appetite_score)):
        return result

    composite = (
        vix_score * WEIGHTS["vix_score"]
        + breadth_score * WEIGHTS["breadth_score"]
        + momentum_score * WEIGHTS["momentum_score"]
        + risk_appetite_score * WEIGHTS["risk_appetite_score"]
    )
    composite = round(_clip(composite, 0.0, 100.0), 1)
    result["market_sentiment_score"] = composite
    result["market_sentiment_label"] = label_for_score(composite)
    return result
