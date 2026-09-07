"""Trend factor: EMA + MACD + Ichimoku + ADX → one TREND_SCORE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcc.factors.helpers import atr, clamp01, ema, last, macd, sigmoid


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, 1)
    atr_n = tr.rolling(n).mean()
    pdi = pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr_n * 100
    mdi = pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr_n * 100
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    return dx.rolling(n).mean()


def _ichimoku(close: pd.Series):
    high = close  # approx using close path if high/low passed separately — caller passes df
    return None


def trend_factor(alt_btc: pd.DataFrame) -> dict:
    c = alt_btc["close"]
    h, l = alt_btc["high"], alt_btc["low"]

    e20 = ema(c, 20)
    e50 = ema(c, 50)
    e20_s = e20.diff()
    e50_s = e50.diff()
    spread = e20 - e50
    spread_s = spread.diff()

    ema_score = 0.5
    if last(e20) is not None and last(e50) is not None:
        bull = 1.0 if last(e20) > last(e50) else 0.0
        slope = sigmoid(float(last(e20_s) or 0) / (last(c) or 1), 0, 500)
        widen = sigmoid(float(last(spread_s) or 0) / (last(c) or 1), 0, 800)
        ema_score = clamp01(0.5 * bull + 0.25 * slope + 0.25 * (0.5 + 0.5 * (1 if last(e50_s) and last(e50_s) > 0 else 0) + 0.5 * widen))

    line, sig, hist = macd(c)
    macd_score = 0.5
    if last(line) is not None and last(sig) is not None:
        bull = 1.0 if last(line) > last(sig) else 0.0
        hpos = 1.0 if (last(hist) or 0) > 0 else 0.0
        hs = hist.diff()
        rising = sigmoid(float(last(hs) or 0) / (abs(last(c) or 1)), 0, 2000)
        macd_score = clamp01(0.4 * bull + 0.3 * hpos + 0.3 * rising)

    # Ichimoku
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(26)
    spb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    # future cloud orientation: spa/spb without shift for "future" view using current calc
    spa_f = (tenkan + kijun) / 2
    spb_f = (h.rolling(52).max() + l.rolling(52).min()) / 2
    ichi = 0.5
    if last(c) is not None and last(spa) is not None and last(spb) is not None:
        cloud_top = max(last(spa), last(spb))
        cloud_bot = min(last(spa), last(spb))
        above = 1.0 if last(c) > cloud_top else (0.5 if last(c) > cloud_bot else 0.0)
        tk = 1.0 if last(tenkan) and last(kijun) and last(tenkan) > last(kijun) else 0.0
        fut = 1.0 if last(spa_f) and last(spb_f) and last(spa_f) > last(spb_f) else 0.0
        ichi = clamp01(0.45 * above + 0.30 * tk + 0.25 * fut)

    adx_s = _adx(alt_btc)
    adx_v = last(adx_s) or 20.0
    # ADX strength scaled; direction from EMA
    strength = clamp01((adx_v - 15) / 35)
    direction = 1.0 if last(e20) and last(e50) and last(e20) > last(e50) else 0.0
    adx_score = clamp01(0.4 * strength + 0.6 * (0.5 * strength + 0.5 * direction * strength + 0.25))

    # Documented weights within trend group
    w_ema, w_macd, w_ichi, w_adx = 0.30, 0.30, 0.25, 0.15
    trend = w_ema * ema_score + w_macd * macd_score + w_ichi * ichi + w_adx * adx_score

    return {
        "score": clamp01(trend),
        "ema_score": ema_score,
        "macd_score": macd_score,
        "ichimoku_score": ichi,
        "adx_score": adx_score,
        "ema20_gt_ema50": bool(last(e20) and last(e50) and last(e20) > last(e50)),
        "adx": adx_v,
        "components_weights": {"ema": w_ema, "macd": w_macd, "ichimoku": w_ichi, "adx": w_adx},
    }
