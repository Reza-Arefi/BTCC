"""Shared indicator helpers — scores normalized to [0, 1]."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.5
    return float(max(0.0, min(1.0, x)))


def sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    z = (x - center) * scale
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    bw = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b, bw


def percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) < 20:
        return 0.5
    return float((s <= value).mean())


def last(s: pd.Series) -> float | None:
    if s is None or len(s) == 0 or pd.isna(s.iloc[-1]):
        return None
    return float(s.iloc[-1])
