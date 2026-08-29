"""BTC regime factor: BTC returns, vol, trend, dominance changes."""

from __future__ import annotations

import pandas as pd

from btcc.factors.helpers import atr, clamp01, ema, last, sigmoid
from btcc.series.relative import horizon_bars, relative_return


def btc_regime_factor(
    btc: pd.DataFrame,
    dominance_pct: float | None,
    dom_changes: dict[int, float | None],
    interval: str = "15m",
) -> dict:
    c = btc["close"]
    rets = {h: relative_return(c, horizon_bars(h, interval)) for h in (1, 4, 8, 12, 24)}

    # BTC trend
    e20, e50 = ema(c, 20), ema(c, 50)
    trend_bull = 1.0 if last(e20) and last(e50) and last(e20) > last(e50) else 0.0

    # Volatility (NATR)
    a = atr(btc, 14)
    natr = (last(a) / last(c)) if last(a) and last(c) else None
    # mild vol preferred for alts; extreme vol slightly lower
    vol_score = 0.5
    if natr is not None:
        vol_score = clamp01(1.0 - abs(natr - 0.01) / 0.04)

    ret_score = 0.5
    parts = []
    for h, w in ((1, 0.15), (4, 0.35), (8, 0.2), (12, 0.15), (24, 0.15)):
        r = rets.get(h)
        parts.append((sigmoid(r, 0.0, 30.0) if r is not None else 0.5) * w)
    ret_score = sum(parts)

    # Dominance: falling dominance → more favorable for alt RS (continuous)
    # Missing changes → neutral contribution (INSUFFICIENT_DATA), never invented
    dom_score = 0.5
    if dominance_pct is not None:
        level = clamp01(1.0 - abs(dominance_pct - 50.0) / 40.0)
        chg_parts = []
        known = 0
        for h, w in ((1, 0.2), (4, 0.4), (12, 0.2), (24, 0.2)):
            ch = dom_changes.get(h)
            if ch is None:
                chg_parts.append(0.5 * w)
            else:
                known += 1
                chg_parts.append(sigmoid(-ch, 0.0, 2.0) * w)
        if known == 0:
            dom_score = clamp01(0.5 * level + 0.5 * 0.5)
        else:
            dom_score = clamp01(0.3 * level + 0.7 * sum(chg_parts))

    score = clamp01(0.35 * ret_score + 0.25 * trend_bull + 0.15 * vol_score + 0.25 * dom_score)

    # Display-only sub-scores (aggregate `score` unchanged)
    regime_indicator_score = clamp01(
        (0.35 * ret_score + 0.25 * trend_bull + 0.15 * vol_score) / 0.75
    )
    dominance_indicator_score = dom_score if dominance_pct is not None else None

    return {
        "score": score,
        "regime_indicator_score": regime_indicator_score,
        "dominance_indicator_score": dominance_indicator_score,
        "btc_return_1h": rets.get(1),
        "btc_return_4h": rets.get(4),
        "btc_return_8h": rets.get(8),
        "btc_return_12h": rets.get(12),
        "btc_return_24h": rets.get(24),
        "btc_natr": natr,
        "btc_ema_bull": bool(trend_bull),
        "btc_dominance_pct": dominance_pct,
        "dom_change_1h": dom_changes.get(1),
        "dom_change_4h": dom_changes.get(4),
        "dom_change_12h": dom_changes.get(12),
        "dom_change_24h": dom_changes.get(24),
        "dominance_source_note": (
            "slow_moving_macro_factor; missing horizon changes are INSUFFICIENT_DATA "
            "(never interpolated)"
        ),
    }
