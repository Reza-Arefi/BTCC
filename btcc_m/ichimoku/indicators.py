"""Standard Ichimoku computation (Phase I)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcc.factors.helpers import atr


def compute_ichimoku(
    df: pd.DataFrame,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """OHLCV-aligned Ichimoku + research features (no look-ahead on cloud)."""
    out = df.copy().reset_index(drop=True)
    h, l, c = out["high"], out["low"], out["close"]

    tenkan = (h.rolling(tenkan_period).max() + l.rolling(tenkan_period).min()) / 2.0
    kijun = (h.rolling(kijun_period).max() + l.rolling(kijun_period).min()) / 2.0
    spa_raw = (tenkan + kijun) / 2.0
    spb_raw = (h.rolling(senkou_b_period).max() + l.rolling(senkou_b_period).min()) / 2.0

    senkou_a = spa_raw.shift(displacement)
    senkou_b = spb_raw.shift(displacement)

    kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    kumo_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    atr14 = atr(out, 14).replace(0, np.nan)

    above = c > kumo_top
    below = c < kumo_bot
    inside = (~above) & (~below) & kumo_top.notna()
    cloud_pos = np.where(above, "above", np.where(below, "below", np.where(inside, "inside", "na")))

    hist_price = c.shift(displacement)
    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["senkou_a"] = senkou_a
    out["senkou_b"] = senkou_b
    out["kumo_top"] = kumo_top
    out["kumo_bot"] = kumo_bot
    out["atr14"] = atr14
    out["cloud_pos"] = cloud_pos
    out["kumo_bullish"] = senkou_a > senkou_b
    out["kumo_width_atr"] = (senkou_a - senkou_b).abs() / atr14
    out["tk_spread_atr"] = (tenkan - kijun) / atr14
    out["tenkan_slope"] = tenkan.diff()
    out["kijun_slope"] = kijun.diff()
    out["price_kijun_atr"] = (c - kijun) / atr14
    out["price_cloud_atr"] = (c - kumo_top) / atr14
    out["chikou_above"] = c > hist_price
    out["chikou_clearance_atr"] = (c - hist_price) / atr14
    return out


def attach_forward_returns(
    frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 6, 12, 24)
) -> pd.DataFrame:
    out = frame.copy()
    c = out["close"].astype(float)
    for h in horizons:
        out[f"fwd_{h}"] = c.shift(-h) / c - 1.0
    return out
