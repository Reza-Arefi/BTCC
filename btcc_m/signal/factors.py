"""Absolute-BTC factor computation (reuse btcc factor math, not ALT/BTC relative series)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.factors.momentum import momentum_factor
from btcc.factors.rsi_factor import rsi_factor
from btcc.factors.structure import structure_factor
from btcc.factors.trend import trend_factor
from btcc.factors.volatility import volatility_factor
from btcc.factors.volume import volume_factor

BASELINE_WEIGHTS: dict[str, float] = {
    "momentum": 0.25,
    "trend": 0.25,
    "volume": 0.15,
    "volatility": 0.125,
    "rsi": 0.10,
    "structure": 0.125,
    "btc_regime": 0.0,
}


def compute_btc_factors(
    btc: pd.DataFrame,
    *,
    interval: str,
    momentum_profile: str = "e2",
) -> dict[str, Any]:
    """Compute six active factors on absolute BTCUSDT OHLCV.

    ``btc`` must contain only information available at the signal bar close
    (caller slices ``timestamp <= t``).
    """
    mom = momentum_factor(btc, interval, profile=momentum_profile)
    trend = trend_factor(btc)
    # Absolute volume on BTC: pass BTCUSDT as both volume and price legs.
    volu = volume_factor(btc, btc)
    volat = volatility_factor(btc)
    rsi_f = rsi_factor(btc)
    struct = structure_factor(btc)
    return {
        "momentum": mom,
        "trend": trend,
        "btc_regime": {"score": 0.5},
        "volume": volu,
        "volatility": volat,
        "rsi": rsi_f,
        "structure": struct,
    }
