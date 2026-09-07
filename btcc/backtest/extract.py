"""Flatten factor outputs into indicator/factor columns for research CSV."""

from __future__ import annotations

from typing import Any


def flatten_indicators_and_factors(factors: dict[str, Any]) -> dict[str, float | None]:
    """Extract individual indicator + factor group scores (0–1 scale where applicable)."""
    mom = factors["momentum"]
    trend = factors["trend"]
    regime = factors["btc_regime"]
    vol = factors["volume"]
    volat = factors["volatility"]
    rsi = factors["rsi"]
    struct = factors["structure"]

    def _v(x: Any) -> float | None:
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    return {
        # Individual indicators
        "indicator_momentum": _v(mom.get("score")),
        "indicator_ema": _v(trend.get("ema_score")),
        "indicator_macd": _v(trend.get("macd_score")),
        "indicator_ichimoku": _v(trend.get("ichimoku_score")),
        "indicator_adx": _v(trend.get("adx_score")),
        "indicator_btc_regime": _v(regime.get("regime_indicator_score", regime.get("score"))),
        "indicator_btc_dominance": _v(regime.get("dominance_indicator_score")),
        "indicator_rvol": _v(vol.get("rvol_score")),
        "indicator_bollinger": _v(volat.get("bollinger_score")),
        "indicator_atr": _v(volat.get("atr_score")),
        "indicator_natr": _v(volat.get("natr_score")),
        "indicator_rsi": _v(rsi.get("score")),
        "indicator_structure": _v(struct.get("score")),
        # Factor groups (aggregated)
        "factor_momentum": _v(mom.get("score")),
        "factor_trend": _v(trend.get("score")),
        "factor_btc_regime": _v(regime.get("score")),
        "factor_volume": _v(vol.get("score")),
        "factor_volatility": _v(volat.get("score")),
        "factor_rsi": _v(rsi.get("score")),
        "factor_structure": _v(struct.get("score")),
    }
