"""Momentum factor on ALT/BTC."""

from __future__ import annotations

import pandas as pd

from btcc.factors.helpers import clamp01, last, sigmoid
from btcc.series.relative import horizon_bars, relative_return


def momentum_factor(alt_btc: pd.DataFrame, interval: str = "15m") -> dict:
    c = alt_btc["close"]
    rets = {}
    for h in (1, 4, 8, 12, 24):
        rets[h] = relative_return(c, horizon_bars(h, interval))

    # Acceleration: current 4h vs previous 4h
    b4 = horizon_bars(4, interval)
    cur4 = rets.get(4)
    prev4 = None
    if len(c) > 2 * b4:
        prev4 = float(c.iloc[-1 - b4] / c.iloc[-1 - 2 * b4] - 1.0)

    accel = 0.0
    if cur4 is not None and prev4 is not None:
        accel = cur4 - prev4

    # Score: blend multi-horizon returns + acceleration
    parts = []
    for h, w in ((1, 0.15), (4, 0.35), (8, 0.20), (12, 0.15), (24, 0.15)):
        r = rets.get(h)
        if r is None:
            parts.append(0.5 * w)
        else:
            parts.append(sigmoid(r, 0.0, 40.0) * w)
    mom = sum(parts)
    mom = 0.85 * mom + 0.15 * sigmoid(accel, 0.0, 50.0)

    return {
        "score": clamp01(mom),
        "return_1h": rets.get(1),
        "return_4h": rets.get(4),
        "return_8h": rets.get(8),
        "return_12h": rets.get(12),
        "return_24h": rets.get(24),
        "accel_4h": accel if cur4 is not None else None,
        "momentum_decay": bool(cur4 is not None and prev4 is not None and cur4 > 0 and accel < 0),
    }
