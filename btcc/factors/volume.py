"""Volume factor from ALTUSDT volume (RVOL + confirmation/divergence)."""

from __future__ import annotations

import pandas as pd

from btcc.factors.helpers import clamp01, last, sigmoid


def volume_factor(alt_usdt: pd.DataFrame, alt_btc: pd.DataFrame, sma: int = 20) -> dict:
    vol = alt_usdt["volume"]
    v_sma = vol.rolling(sma).mean()
    rvol = (last(vol) / last(v_sma)) if last(v_sma) else None

    # price change vs volume change (last bar)
    px = alt_btc["close"].pct_change()
    vv = vol.pct_change()
    px_up = (last(px) or 0) > 0
    vol_up = (last(vv) or 0) > 0
    confirmation = px_up and vol_up
    divergence = px_up and not vol_up and (last(vv) or 0) < 0

    rvol_score = sigmoid((rvol or 1.0) - 1.0, 0.0, 1.5) if rvol is not None else 0.5
    conf = 0.7 if confirmation else (0.35 if divergence else 0.5)
    score = clamp01(0.65 * rvol_score + 0.35 * conf)

    return {
        "score": score,
        "rvol_score": rvol_score if rvol is not None else None,
        "rvol": rvol,
        "volume_confirmation": confirmation,
        "volume_divergence": divergence,
    }
