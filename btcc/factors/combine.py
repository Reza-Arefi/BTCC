"""Combine factor groups with configurable weights → SIGNAL SCORE (not probability)."""

from __future__ import annotations

from typing import Any

from btcc.factors.btc_regime import btc_regime_factor
from btcc.factors.momentum import momentum_factor
from btcc.factors.rsi_factor import rsi_factor
from btcc.factors.structure import structure_factor
from btcc.factors.trend import trend_factor
from btcc.factors.volatility import volatility_factor
from btcc.factors.volume import volume_factor


def compute_all_factors(
    alt_btc,
    alt_usdt,
    btc_usdt,
    dominance_pct: float | None,
    dom_changes: dict,
    cfg: dict[str, Any],
    interval: str = "15m",
    factor_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute factors → signal score.

    ``factor_weights`` overrides config weights when a Champion model is active.
    Indicator architecture is unchanged; only the factor-group mix may adapt.
    """
    mom_profile = str((cfg.get("factors") or {}).get("momentum_profile") or "base")
    mom = momentum_factor(alt_btc, interval, profile=mom_profile)
    trend = trend_factor(alt_btc)
    regime = btc_regime_factor(btc_usdt, dominance_pct, dom_changes, interval)
    volu = volume_factor(alt_usdt, alt_btc)
    volat = volatility_factor(alt_btc)
    rsi_f = rsi_factor(alt_btc)
    struct = structure_factor(alt_btc)

    w = dict(factor_weights) if factor_weights is not None else dict(cfg["factors"]["weights"])
    # Isolation: BTC.D-linked btc_regime group is contextual only (w=0), then renormalize.
    w["btc_regime"] = 0.0
    active = ("momentum", "trend", "volume", "volatility", "rsi", "structure")
    s = sum(max(0.0, float(w.get(k, 0.0))) for k in active)
    if s <= 0:
        w = {k: (1.0 / len(active) if k in active else 0.0) for k in w}
        w["btc_regime"] = 0.0
    else:
        for k in active:
            w[k] = max(0.0, float(w.get(k, 0.0))) / s
        w["btc_regime"] = 0.0
    signal_score = (
        w["momentum"] * mom["score"]
        + w["trend"] * trend["score"]
        + w["btc_regime"] * regime["score"]
        + w["volume"] * volu["score"]
        + w["volatility"] * volat["score"]
        + w["rsi"] * rsi_f["score"]
        + w["structure"] * struct["score"]
    )

    return {
        "signal_score": float(signal_score),
        "momentum": mom,
        "trend": trend,
        "btc_regime": regime,
        "volume": volu,
        "volatility": volat,
        "rsi": rsi_f,
        "structure": struct,
        "weights": w,
    }
