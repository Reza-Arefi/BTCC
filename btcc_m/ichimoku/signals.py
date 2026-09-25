"""Phase I pure Ichimoku long signal generators."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


SignalFn = Callable[[pd.DataFrame], pd.Series]


def _valid(df: pd.DataFrame) -> pd.Series:
    return df["tenkan"].notna() & df["kijun"].notna() & df["kumo_top"].notna() & df["kumo_bot"].notna()


def tk_cross(df: pd.DataFrame) -> pd.Series:
    """I1 — Tenkan crosses above Kijun."""
    prev_le = df["tenkan"].shift(1) <= df["kijun"].shift(1)
    now_gt = df["tenkan"] > df["kijun"]
    return prev_le & now_gt & _valid(df)


def tk_cross_above_kumo(df: pd.DataFrame) -> pd.Series:
    """I1A — TK cross with close above cloud."""
    return tk_cross(df) & (df["cloud_pos"] == "above")


def tk_cross_inside_kumo(df: pd.DataFrame) -> pd.Series:
    """I1B — TK cross inside cloud."""
    return tk_cross(df) & (df["cloud_pos"] == "inside")


def tk_cross_below_kumo(df: pd.DataFrame) -> pd.Series:
    """I1C — TK cross below cloud."""
    return tk_cross(df) & (df["cloud_pos"] == "below")


def kumo_breakout(df: pd.DataFrame) -> pd.Series:
    """I2 — Close crosses from <= KumoTop to > KumoTop."""
    prev = df["close"].shift(1) <= df["kumo_top"].shift(1)
    now = df["close"] > df["kumo_top"]
    return prev & now & _valid(df)


def kijun_reclaim(df: pd.DataFrame) -> pd.Series:
    """I3 — Touch/pierce Kijun then close back above, preferably in bullish cloud regime."""
    touched = df["low"] <= df["kijun"]
    close_above = df["close"] > df["kijun"]
    # Prefer existing bullish structure: price above cloud bot or cloud bullish
    regime = (df["cloud_pos"] == "above") | (df["kumo_bullish"].fillna(False))
    # Was at/below kijun recently (prior bar close <= kijun or prior not reclaiming)
    was_below = df["close"].shift(1) <= df["kijun"].shift(1)
    return touched & close_above & was_below & regime & _valid(df)


def kumo_breakout_retest(df: pd.DataFrame, lookback: int = 12) -> pd.Series:
    """I4 — Break above Kumo, pullback toward cloud, bullish close without losing cloud."""
    n = len(df)
    sig = np.zeros(n, dtype=bool)
    c = df["close"].values
    low = df["low"].values
    top = df["kumo_top"].values
    bot = df["kumo_bot"].values
    atrv = df["atr14"].values
    valid = _valid(df).values

    # Breakout bars
    prev_c = df["close"].shift(1).values
    prev_top = df["kumo_top"].shift(1).values
    breakout = valid & np.isfinite(prev_c) & np.isfinite(prev_top) & (prev_c <= prev_top) & (c > top)

    for i in range(n):
        if not breakout[i]:
            continue
        # Search retest window after breakout
        for j in range(i + 1, min(n, i + 1 + lookback)):
            if not valid[j] or not np.isfinite(atrv[j]) or atrv[j] <= 0:
                continue
            # Pullback toward cloud: low within 0.5 ATR of kumo top or dips into cloud
            near = low[j] <= top[j] + 0.5 * atrv[j]
            into = low[j] <= top[j]
            # Did not decisively lose cloud
            held = c[j] >= bot[j]
            bullish_close = c[j] > df["open"].values[j]
            reclaim = c[j] > top[j]
            if near and into and held and bullish_close and reclaim:
                # Only first retest entry per breakout
                sig[j] = True
                break
    return pd.Series(sig, index=df.index)


def trend_continuation_immediate(df: pd.DataFrame) -> pd.Series:
    """I5a — Enter when trend-state newly becomes true."""
    state = (
        (df["cloud_pos"] == "above")
        & (df["tenkan"] > df["kijun"])
        & (df["kijun_slope"] > 0)
        & _valid(df)
    )
    return state & (~state.shift(1).fillna(False))


def trend_continuation_pullback(df: pd.DataFrame) -> pd.Series:
    """I5b — In trend-state, pullback to Kijun then close back above."""
    state = (
        (df["cloud_pos"] == "above")
        & (df["tenkan"] > df["kijun"])
        & (df["kijun_slope"] > 0)
        & _valid(df)
    )
    touch = df["low"] <= df["kijun"]
    reclaim = df["close"] > df["kijun"]
    was_not = df["close"].shift(1) <= df["kijun"].shift(1)
    return state & touch & reclaim & was_not


PHASE1_SIGNALS: dict[str, SignalFn] = {
    "I1_tk_cross": tk_cross,
    "I1A_tk_above_kumo": tk_cross_above_kumo,
    "I1B_tk_inside_kumo": tk_cross_inside_kumo,
    "I1C_tk_below_kumo": tk_cross_below_kumo,
    "I2_kumo_breakout": kumo_breakout,
    "I3_kijun_reclaim": kijun_reclaim,
    "I4_kumo_retest": kumo_breakout_retest,
    "I5a_trend_immediate": trend_continuation_immediate,
    "I5b_trend_pullback": trend_continuation_pullback,
}


def classify_failure_features(row: pd.Series) -> list[str]:
    """Heuristic failure tags from signal-bar features (for losing trades)."""
    tags: list[str] = []
    pos = str(row.get("cloud_pos", "na"))
    if pos == "inside":
        tags.append("Failure_B_inside_cloud")
    if pos == "below":
        tags.append("Failure_B_below_cloud")
    pk = row.get("price_kijun_atr")
    if pk is not None and np.isfinite(pk) and float(pk) > 2.0:
        tags.append("Failure_C_extended_from_kijun")
    if row.get("chikou_above") is False or (isinstance(row.get("chikou_above"), float) and not row.get("chikou_above")):
        tags.append("Failure_chikou_disagree")
    kw = row.get("kumo_width_atr")
    if kw is not None and np.isfinite(kw) and float(kw) < 0.3:
        tags.append("Failure_thin_kumo")
    if not tags:
        tags.append("Failure_A_unclassified_reversal")
    return tags
