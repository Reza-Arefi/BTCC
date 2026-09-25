"""Phase II Ichimoku-only structural filters (F1–F6).

F1 and F5 overlap conceptually (both require Close > Kumo top).
Test separately for diagnostics; never combine F1+F5 in a model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Fixed exploratory extension cap (Phase I Failure_C used >2.0). Not optimized.
F6_EXTENSION_ATR = 2.0

FilterFn = Callable[[pd.DataFrame], pd.Series]


def f1_above_kumo(df: pd.DataFrame) -> pd.Series:
    """F1 — Close > Kumo top."""
    return df["close"] > df["kumo_top"]


def f2_bullish_kumo(df: pd.DataFrame) -> pd.Series:
    """F2 — Senkou A > Senkou B."""
    return df["kumo_bullish"].fillna(False).astype(bool)


def f3_kijun_slope(df: pd.DataFrame) -> pd.Series:
    """F3 — ΔKijun > 0."""
    return df["kijun_slope"] > 0


def f4_chikou(df: pd.DataFrame) -> pd.Series:
    """F4 — Chikou above historical price."""
    return df["chikou_above"].fillna(False).astype(bool)


def f5_avoid_interior_below(df: pd.DataFrame) -> pd.Series:
    """F5 — Reject Close ≤ Kumo top (i.e. require above cloud)."""
    return df["close"] > df["kumo_top"]


def f6_avoid_extension(df: pd.DataFrame, max_atr: float = F6_EXTENSION_ATR) -> pd.Series:
    """F6 — Reject excessively large (Close−Kijun)/ATR."""
    pk = df["price_kijun_atr"]
    return pk.notna() & (pk <= float(max_atr))


FILTERS: dict[str, FilterFn] = {
    "F1": f1_above_kumo,
    "F2": f2_bullish_kumo,
    "F3": f3_kijun_slope,
    "F4": f4_chikou,
    "F5": f5_avoid_interior_below,
    "F6": f6_avoid_extension,
}

# Overlap: F1 ≡ F5 for long (Close > Kumo top). Never pair them.
OVERLAP_PAIRS = {frozenset({"F1", "F5"})}


def apply_filters(df: pd.DataFrame, fire: pd.Series, filter_keys: list[str]) -> pd.Series:
    out = fire.copy().astype(bool)
    for k in filter_keys:
        if k not in FILTERS:
            raise KeyError(f"Unknown filter {k}")
        out = out & FILTERS[k](df)
    return out


def valid_combination(keys: list[str]) -> bool:
    s = frozenset(keys)
    for bad in OVERLAP_PAIRS:
        if bad.issubset(s):
            return False
    return True
