"""Synthetic ALT/BTC = ALTUSDT / BTCUSDT relative-strength series.

Also supports native BTC-quoted markets (e.g. CKBTCBTC) when no USDT pair exists.
"""

from __future__ import annotations

import pandas as pd


def build_alt_btc(alt: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame | None:
    """Construct ALT/BTC OHLCV from USDT legs. Volume from ALTUSDT.

    Conservative high/low:
      high ≈ alt_high / btc_low
      low  ≈ alt_low / btc_high
    """
    if alt is None or btc is None or alt.empty or btc.empty:
        return None
    a = alt.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
    a.columns = ["ao", "ah", "al", "ac", "volume"]
    b = btc.set_index("timestamp")[["open", "high", "low", "close"]].copy()
    b.columns = ["bo", "bh", "bl", "bc"]
    m = a.join(b, how="inner").dropna()
    if m.empty:
        return None
    o = m["ao"] / m["bo"]
    c = m["ac"] / m["bc"]
    h = (m["ah"] / m["bl"]).combine(o, max).combine(c, max)
    l = (m["al"] / m["bh"]).combine(o, min).combine(c, min)
    return pd.DataFrame({
        "timestamp": m.index,
        "open": o.values,
        "high": h.values,
        "low": l.values,
        "close": c.values,
        "volume": m["volume"].values,
    }).reset_index(drop=True)


def native_btc_as_relative(btc_quoted: pd.DataFrame) -> pd.DataFrame | None:
    """Use a native *BTC market as the ALT/BTC relative series (already in BTC terms)."""
    if btc_quoted is None or btc_quoted.empty:
        return None
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return btc_quoted[cols].copy().reset_index(drop=True)


def relative_return(close: pd.Series, bars: int) -> float | None:
    if len(close) <= bars:
        return None
    a, b = float(close.iloc[-1]), float(close.iloc[-1 - bars])
    if b == 0:
        return None
    return a / b - 1.0


def horizon_bars(hours: int, interval: str = "15m") -> int:
    per = {"15m": 4, "1h": 1, "5m": 12}
    return max(1, hours * per.get(interval, 4))
