"""RSI as state/context factor (not RSI>70 sell)."""

from __future__ import annotations

import pandas as pd

from btcc.factors.helpers import clamp01, last, rsi


def rsi_factor(alt_btc: pd.DataFrame) -> dict:
    r = rsi(alt_btc["close"], 14)
    rv = last(r)
    slope = last(r.diff())
    score = 0.5
    if rv is not None:
        # constructive 50–70 rising; overheated >75 reduces unless slope strongly up
        if 50 <= rv <= 70:
            base = 0.65 + 0.2 * (rv - 50) / 20
        elif rv < 50:
            base = 0.35 + 0.3 * (rv / 50)
        elif rv <= 75:
            base = 0.55
        else:
            base = 0.35
        if slope is not None and slope > 0:
            base += 0.08
        if slope is not None and slope < 0 and rv > 70:
            base -= 0.10
        score = clamp01(base)

    # simple divergence: price HH, RSI LH over last 20 bars
    bear_div = False
    if len(alt_btc) >= 20 and rv is not None:
        px = alt_btc["close"].iloc[-20:]
        rs = r.iloc[-20:]
        if px.iloc[-1] >= px.max() * 0.995 and rs.iloc[-1] < rs.max() * 0.97:
            bear_div = True
            score = clamp01(score - 0.08)

    return {
        "score": score,
        "rsi14": rv,
        "rsi_slope": slope,
        "bearish_divergence": bear_div,
        "overheated": bool(rv is not None and rv > 75),
    }
