"""Lazy 1-second ALT/BTC exit tape (day-partitioned parquet cache)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.data.binance_vision import load_candles_range_1s, partitioned_1s_dir
from btcc.series.relative import build_alt_btc

logger = logging.getLogger(__name__)


class ExitTape1s:
    """On-demand 1s relative bars — avoids loading full 90d × universe into RAM."""

    def __init__(self, candle_dir: Path | str, bases: list[str]):
        self.candle_dir = Path(candle_dir)
        self.available: set[str] = set()
        unavailable: list[str] = []
        for base in bases:
            root = partitioned_1s_dir(self.candle_dir, f"{base}USDT")
            parts = list(root.glob("????-??-??.parquet")) if root.exists() else []
            if parts:
                self.available.add(base)
            else:
                unavailable.append(base)
        btc_root = partitioned_1s_dir(self.candle_dir, "BTCUSDT")
        if not btc_root.exists() or not list(btc_root.glob("????-??-??.parquet")):
            raise RuntimeError("Missing BTCUSDT 1s partitioned candles")
        logger.info(
            "1s exit tape ready coins=%d unavailable=%s",
            len(self.available),
            unavailable[:12],
        )
        self.unavailable = unavailable
        # tiny cache: last slice per base to avoid re-reading identical windows
        self._slice_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._btc_close_cache: dict[tuple[str, str], pd.Series] = {}

    def has(self, base: str) -> bool:
        return base in self.available

    def rel_and_btc_close(
        self,
        base: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return (relative OHLC, btc_usdt close series) for (start, end]."""
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        else:
            s = s.tz_convert("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        else:
            e = e.tz_convert("UTC")
        # Include a 1s pad before start so joins are stable near boundaries
        fetch_start = s - pd.Timedelta(seconds=2)
        key = (base, str(fetch_start), str(e))
        if key in self._slice_cache:
            rel = self._slice_cache[key]
            btc_key = (str(fetch_start), str(e))
            btc_close = self._btc_close_cache[btc_key]
            bars = rel[(rel["timestamp"] > s) & (rel["timestamp"] <= e)]
            return bars, btc_close

        alt = load_candles_range_1s(self.candle_dir, f"{base}USDT", fetch_start, e)
        btc = load_candles_range_1s(self.candle_dir, "BTCUSDT", fetch_start, e)
        if alt.empty or btc.empty:
            empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            return empty, pd.Series(dtype=float)

        rel = build_alt_btc(alt, btc)
        if rel is None or rel.empty:
            empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            return empty, pd.Series(dtype=float)
        rel = rel.copy()
        rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
        btc_close = btc.set_index("timestamp")["close"]
        # Keep a small LRU-ish cache
        if len(self._slice_cache) > 64:
            self._slice_cache.clear()
            self._btc_close_cache.clear()
        self._slice_cache[key] = rel
        self._btc_close_cache[(str(fetch_start), str(e))] = btc_close
        bars = rel[(rel["timestamp"] > s) & (rel["timestamp"] <= e)].reset_index(drop=True)
        return bars, btc_close

    def btc_close_near(self, ts: pd.Timestamp, default: float) -> float:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        btc = load_candles_range_1s(self.candle_dir, "BTCUSDT", t - pd.Timedelta(minutes=2), t + pd.Timedelta(seconds=1))
        if btc is None or btc.empty:
            return float(default)
        sub = btc[btc["timestamp"] <= t]
        if sub.empty:
            return float(btc["close"].iloc[0])
        return float(sub["close"].iloc[-1])
