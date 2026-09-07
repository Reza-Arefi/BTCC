"""Market regime labels at entry — diagnostic only, no lookahead.

Uses factor outputs available at decision time (ADX, NATR, BB width).
"""

from __future__ import annotations

from typing import Any

REGIME_RANGE = "RANGE"
REGIME_LOW_VOL_TREND = "LOW_VOL_TREND"
REGIME_NORMAL_TREND = "NORMAL_TREND"
REGIME_STRONG_TREND = "STRONG_TREND"
REGIME_HIGH_VOL = "HIGH_VOLATILITY"

REGIME_CLASSES = (
    REGIME_RANGE,
    REGIME_LOW_VOL_TREND,
    REGIME_NORMAL_TREND,
    REGIME_STRONG_TREND,
    REGIME_HIGH_VOL,
)


def classify_regime(
    factors: dict[str, Any],
    *,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify regime from as-of-entry factors."""
    r = rules or {}
    high_vol_natr = float(r.get("high_vol_natr", 0.035))
    range_adx = float(r.get("range_adx", 20.0))
    low_adx = float(r.get("low_trend_adx", 30.0))
    normal_adx = float(r.get("normal_trend_adx", 40.0))

    trend = factors.get("trend") or {}
    vol = factors.get("volatility") or {}
    adx = float(trend.get("adx") or 20.0)
    natr = vol.get("natr")
    natr_f = float(natr) if natr is not None else 0.015
    bb_bw = vol.get("bandwidth")
    bb_bw_f = float(bb_bw) if bb_bw is not None else None

    if natr_f >= high_vol_natr:
        label = REGIME_HIGH_VOL
    elif adx < range_adx:
        label = REGIME_RANGE
    elif adx < low_adx:
        label = REGIME_LOW_VOL_TREND
    elif adx < normal_adx:
        label = REGIME_NORMAL_TREND
    else:
        label = REGIME_STRONG_TREND

    return {
        "regime": label,
        "adx": adx,
        "natr": natr_f,
        "bb_bandwidth": bb_bw_f,
    }
