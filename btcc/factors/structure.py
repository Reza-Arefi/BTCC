"""Market structure: HH/HL, breakout, distance to support/resistance."""

from __future__ import annotations

import pandas as pd

from btcc.factors.helpers import clamp01, last


def structure_factor(alt_btc: pd.DataFrame, lookback: int = 48) -> dict:
    df = alt_btc.tail(lookback)
    if len(df) < 10:
        return {"score": 0.5, "hh": False, "hl": False, "breakout": False,
                "dist_to_resistance": None, "dist_to_support": None}

    c = df["close"]
    h, l = df["high"], df["low"]
    # swing proxies: rolling extrema
    res = float(h.iloc[:-3].max())
    sup = float(l.iloc[:-3].min())
    px = float(c.iloc[-1])
    dist_res = (res - px) / px if px else None
    dist_sup = (px - sup) / px if px else None

    # HH/HL on last two halves
    mid = len(df) // 2
    hh = float(h.iloc[mid:].max()) > float(h.iloc[:mid].max())
    hl = float(l.iloc[mid:].min()) > float(l.iloc[:mid].min())
    breakout = px > res * 0.999

    score = 0.4
    if hh:
        score += 0.15
    if hl:
        score += 0.15
    if breakout:
        score += 0.20
    if dist_res is not None and dist_res > 0:
        # farther from resistance slightly less urgent; near resistance lower score for new thrust
        score += 0.05 if dist_res > 0.01 else -0.05
    score = clamp01(score)

    return {
        "score": score,
        "hh": hh,
        "hl": hl,
        "breakout": breakout,
        "dist_to_resistance": dist_res,
        "dist_to_support": dist_sup,
        "resistance": res,
        "support": sup,
    }
