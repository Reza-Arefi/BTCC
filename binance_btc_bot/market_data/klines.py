"""Public kline helpers for production scoring (closed 15m bars only).

Uses Binance Vision / public REST — no signed credentials required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


def interval_to_ms(interval: str) -> int:
    key = str(interval).strip().lower()
    if key not in INTERVAL_MS:
        raise ValueError(f"unsupported kline interval: {interval}")
    return INTERVAL_MS[key]


def klines_to_dataframe(raw: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Convert Binance REST kline rows to OHLCV with UTC timestamps (open time)."""
    if not raw:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    rows = [
        {
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
        for k in raw
    ]
    return (
        pd.DataFrame(rows)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def drop_incomplete_candle(
    df: pd.DataFrame,
    *,
    interval: str = "15m",
    now: datetime | None = None,
) -> pd.DataFrame:
    """Keep only closed bars (open_time + interval <= now).

    Matches the live SignalEngine intent: never decide on a forming candle.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    last_open = out["timestamp"].iloc[-1]
    last_open_dt = (
        last_open.to_pydatetime()
        if isinstance(last_open, pd.Timestamp)
        else pd.Timestamp(last_open, tz="UTC").to_pydatetime()
    )
    if last_open_dt.tzinfo is None:
        last_open_dt = last_open_dt.replace(tzinfo=timezone.utc)
    close_dt = last_open_dt + timedelta(milliseconds=interval_to_ms(interval))
    if close_dt > now_utc:
        out = out.iloc[:-1].copy()
    return out.reset_index(drop=True)


def closed_candle_age_sec(
    df: pd.DataFrame,
    *,
    interval: str = "15m",
    now: datetime | None = None,
) -> float:
    """Seconds since the last closed candle's close time. Inf if empty."""
    if df is None or df.empty:
        return float("inf")
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    last_open = pd.to_datetime(df["timestamp"].iloc[-1], utc=True).to_pydatetime()
    if last_open.tzinfo is None:
        last_open = last_open.replace(tzinfo=timezone.utc)
    close_dt = last_open + timedelta(milliseconds=interval_to_ms(interval))
    return max(0.0, (now_utc - close_dt).total_seconds())
