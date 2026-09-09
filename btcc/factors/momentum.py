"""Momentum factor on ALT/BTC.

Profiles:
  base — frozen research/default horizons (1/4/8/12/24h) + 85/15 accel mix
  e2   — faster live profile: adds 15m, reweights, 80/20 accel mix
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.factors.helpers import clamp01, sigmoid
from btcc.series.relative import horizon_bars, relative_return


def momentum_factor(
    alt_btc: pd.DataFrame,
    interval: str = "15m",
    *,
    profile: str = "base",
) -> dict[str, Any]:
    c = alt_btc["close"]
    profile_key = str(profile or "base").strip().lower()

    b4 = horizon_bars(4, interval)
    cur4 = relative_return(c, b4)
    prev4 = None
    if len(c) > 2 * b4:
        prev4 = float(c.iloc[-1 - b4] / c.iloc[-1 - 2 * b4] - 1.0)
    accel = 0.0
    if cur4 is not None and prev4 is not None:
        accel = float(cur4) - float(prev4)

    if profile_key == "e2":
        rets: dict[Any, float | None] = {}
        b15 = 1
        rets["15m"] = float(c.iloc[-1] / c.iloc[-1 - b15] - 1.0) if len(c) > b15 else None
        for h in (1, 4, 8, 12, 24):
            rets[h] = relative_return(c, horizon_bars(h, interval))
        parts = []
        for key, w in (("15m", 0.20), (1, 0.20), (4, 0.30), (8, 0.15), (12, 0.10), (24, 0.05)):
            r = rets.get(key)
            if r is None:
                parts.append(0.5 * w)
            else:
                parts.append(sigmoid(float(r), 0.0, 40.0) * w)
        mom = sum(parts)
        mom = 0.80 * mom + 0.20 * sigmoid(accel, 0.0, 50.0)
        return {
            "score": clamp01(mom),
            "return_15m": rets.get("15m"),
            "return_1h": rets.get(1),
            "return_4h": rets.get(4),
            "return_8h": rets.get(8),
            "return_12h": rets.get(12),
            "return_24h": rets.get(24),
            "accel_4h": accel if cur4 is not None else None,
            "momentum_decay": bool(cur4 is not None and prev4 is not None and cur4 > 0 and accel < 0),
            "profile": "e2",
        }

    # base (frozen)
    rets_h: dict[int, float | None] = {}
    for h in (1, 4, 8, 12, 24):
        rets_h[h] = relative_return(c, horizon_bars(h, interval))
    parts = []
    for h, w in ((1, 0.15), (4, 0.35), (8, 0.20), (12, 0.15), (24, 0.15)):
        r = rets_h.get(h)
        if r is None:
            parts.append(0.5 * w)
        else:
            parts.append(sigmoid(float(r), 0.0, 40.0) * w)
    mom = sum(parts)
    mom = 0.85 * mom + 0.15 * sigmoid(accel, 0.0, 50.0)
    return {
        "score": clamp01(mom),
        "return_1h": rets_h.get(1),
        "return_4h": rets_h.get(4),
        "return_8h": rets_h.get(8),
        "return_12h": rets_h.get(12),
        "return_24h": rets_h.get(24),
        "accel_4h": accel if cur4 is not None else None,
        "momentum_decay": bool(cur4 is not None and prev4 is not None and cur4 > 0 and accel < 0),
        "profile": "base",
    }
