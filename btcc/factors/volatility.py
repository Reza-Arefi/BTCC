"""Volatility factor: Bollinger + ATR/NATR → context score (not auto bullish/bearish)."""

from __future__ import annotations

import pandas as pd

from btcc.factors.helpers import atr, bollinger, clamp01, last


def volatility_factor(alt_btc: pd.DataFrame) -> dict:
    c = alt_btc["close"]
    mid, upper, lower, pct_b, bw = bollinger(c)
    a = atr(alt_btc, 14)
    natr = (last(a) / last(c)) if last(a) and last(c) else None
    pb = last(pct_b)
    bwv = last(bw)
    bw_chg = last(bw.diff()) if bw is not None else None

    # Expansion with trend-friendly mid location scores higher; extreme extension alone ≠ bullish
    score = 0.5
    bollinger_score = None
    atr_score = None
    natr_score = None
    if pb is not None and bwv is not None:
        # prefer expansion from compression with price in upper half but not absurd
        expand = clamp01(((bw_chg or 0) * 50) + 0.5)
        loc = clamp01(pb)  # 0 lower band … 1 upper
        # sweet spot ~0.55-0.85
        loc_score = 1.0 - abs(loc - 0.7) / 0.7
        natr_score = clamp01(1.0 - min(abs((natr or 0.015) - 0.015) / 0.05, 1.0))
        bollinger_score = clamp01(0.55 * expand + 0.45 * clamp01(loc_score))
        atr_score = clamp01(min((natr or 0) / 0.05, 1.0)) if natr is not None else None
        score = clamp01(0.45 * expand + 0.35 * clamp01(loc_score) + 0.20 * natr_score)

    return {
        "score": score,
        "bollinger_score": bollinger_score,
        "atr_score": atr_score,
        "natr_score": natr_score,
        "percent_b": pb,
        "bandwidth": bwv,
        "bandwidth_change": bw_chg,
        "natr": natr,
        "above_upper_band": bool(pb is not None and pb > 1.0),
    }
