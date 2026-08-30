"""MEXC REST + WebSocket market data (public only — no trading keys)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import requests

from btcc.data.candles import COLS, candle_path, load_candles, save_candles, upsert_candle

logger = logging.getLogger(__name__)

INTERVAL_MAP = {"15m": "15m", "1h": "60m", "4h": "4h", "1d": "1d"}
INTERVAL_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


class MexcPublicREST:
    def __init__(self, base_url: str = "https://api.mexc.com"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BTCC-signal/0.1 (public-data-only)"})

    def _get(self, path: str, params: dict | None = None) -> Any:
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(min(60.0, 1.5 * (2 ** attempt)))
                    continue
                # Invalid symbol / bad request — do not retry forever
                if r.status_code == 400:
                    logger.error("DATA_UNAVAILABLE (HTTP 400): %s params=%s body=%s",
                                 path, params, r.text[:200])
                    return None
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_err = e
                sleep_s = min(60.0, 1.0 * (2 ** attempt))
                logger.warning("REST retry %s (sleep %.1fs): %s", attempt, sleep_s, e)
                time.sleep(sleep_s)
        raise RuntimeError(f"REST failed: {path} ({last_err})")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        mexc_i = INTERVAL_MAP.get(interval, interval)
        params: dict[str, Any] = {"symbol": symbol, "interval": mexc_i, "limit": min(limit, 1000)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        raw = self._get("/api/v3/klines", params)
        if raw is None:
            return pd.DataFrame(columns=COLS)
        rows = []
        for k in raw:
            rows.append({
                "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=COLS)

    def _fetch_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[pd.DataFrame]:
        step = INTERVAL_MS[interval] * 1000
        cursor = int(start_ms)
        end_ms = int(end_ms)
        frames: list[pd.DataFrame] = []
        while cursor < end_ms:
            chunk_end = min(cursor + step, end_ms)
            df = self.fetch_klines(symbol, interval, start_ms=cursor, end_ms=chunk_end)
            if not df.empty:
                frames.append(df)
                last = int(df["timestamp"].max().timestamp() * 1000)
                nxt = last + INTERVAL_MS[interval]
                if nxt <= cursor:
                    break
                cursor = nxt
            else:
                cursor = chunk_end + INTERVAL_MS[interval]
            time.sleep(0.05)
        return frames

    def bootstrap_symbol(
        self,
        symbol: str,
        interval: str,
        lookback_bars: int,
        candle_dir: str,
        force: bool = False,
    ) -> pd.DataFrame | None:
        """Load/extend local candle cache for ``symbol``.

        Prefer incremental forward extension of an existing valid cache.
        Never fabricates missing history. Short listing histories are kept as-is.
        ``force=True`` still preserves existing cache rows and only re-fetches
        to extend / fill toward ``lookback_bars`` when the exchange provides more.
        """
        path = candle_path(candle_dir, symbol, interval)
        cached = load_candles(path)
        min_ok = max(int(lookback_bars * 0.95), lookback_bars - 50)

        # Probe latest candles first — fail fast on invalid symbols
        probe = self.fetch_klines(symbol, interval, limit=5)
        if probe.empty:
            logger.error("DATA_UNAVAILABLE: %s (no klines)", symbol)
            return None

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        frames: list[pd.DataFrame] = []
        if cached is not None and not cached.empty:
            frames.append(cached)
            tmax = cached["timestamp"].max()
            if getattr(tmax, "tzinfo", None) is None:
                tmax = tmax.tz_localize("UTC")
            age_h = (datetime.now(timezone.utc) - tmax.to_pydatetime()).total_seconds() / 3600
            if not force and len(cached) >= min_ok and age_h < 1.0:
                return cached.reset_index(drop=True)
            # Incremental forward extend from last cached bar
            start_ms = int(tmax.timestamp() * 1000) + INTERVAL_MS[interval]
            if start_ms < end_ms:
                frames.extend(self._fetch_range(symbol, interval, start_ms, end_ms))
            # Optional backfill only when cache is shorter than requested and
            # we may still obtain older exchange history (never invent gaps).
            if len(cached) < min_ok:
                need_ms = INTERVAL_MS[interval] * lookback_bars
                want_start = end_ms - need_ms
                tmin = cached["timestamp"].min()
                if getattr(tmin, "tzinfo", None) is None:
                    tmin = tmin.tz_localize("UTC")
                have_start = int(tmin.timestamp() * 1000)
                if want_start < have_start:
                    frames.extend(self._fetch_range(symbol, interval, want_start, have_start))
        else:
            need_ms = INTERVAL_MS[interval] * lookback_bars
            cursor = end_ms - need_ms
            frames.extend(self._fetch_range(symbol, interval, cursor, end_ms))

        if not frames:
            logger.error("DATA_UNAVAILABLE: %s", symbol)
            return None
        out = (
            pd.concat(frames)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
        )
        # NEVER truncate the on-disk cache to the caller's lookback window.
        # Short backtests (smoke) used to call save with out.tail(lookback_bars)
        # and permanently destroyed ~1y history. Keep all valid history; callers
        # that need a window must slice in-memory after load.
        save_candles(out, path)
        return out.reset_index(drop=True)


class MexcWebSocketFeed:
    """Subscribe to mini-ticker / deal updates; fold into 15m candles locally.

    Trading endpoints are intentionally absent.
    """

    def __init__(self, ws_url: str, symbols: list[str], on_tick: Callable[[str, dict], None] | None = None):
        self.ws_url = ws_url
        self.symbols = [s.upper() for s in symbols]
        self.on_tick = on_tick
        self._running = False

    async def run(self) -> None:
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("Install websockets: pip install websockets") from e

        self._running = True
        # MEXC spot public: subscribe miniTicker for each symbol
        subs = []
        for sym in self.symbols:
            # JSON protocol channel style used by MEXC public WS
            subs.append({"method": "SUBSCRIPTION", "params": [f"wendy@{sym}@ticker"]})

        while self._running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                    for sub in subs:
                        await ws.send(json.dumps(sub))
                        await asyncio.sleep(0.05)
                    logger.info("WS connected; subscribed %d symbols", len(self.symbols))
                    async for msg in ws:
                        if not self._running:
                            break
                        self._handle(msg)
            except Exception as e:
                logger.warning("WS reconnect in 3s: %s", e)
                await asyncio.sleep(3)

    def _handle(self, msg: str) -> None:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return
        # Flexible parse — MEXC payloads vary by channel version
        if isinstance(data, dict):
            sym = data.get("s") or data.get("symbol")
            if not sym and "d" in data and isinstance(data["d"], dict):
                sym = data["d"].get("s") or data["d"].get("symbol")
                payload = data["d"]
            else:
                payload = data
            if sym and self.on_tick:
                self.on_tick(str(sym).upper(), payload)

    def stop(self) -> None:
        self._running = False


def apply_ticker_to_candle(df: pd.DataFrame, price: float, volume_inc: float, interval: str) -> pd.DataFrame:
    """Update / create current incomplete 15m candle from live price (for features at close we use completed bars only)."""
    ms = INTERVAL_MS[interval]
    now = datetime.now(timezone.utc)
    bucket = int(now.timestamp() * 1000) // ms * ms
    ts = pd.to_datetime(bucket, unit="ms", utc=True)
    if df is None or df.empty:
        return pd.DataFrame([{
            "timestamp": ts, "open": price, "high": price, "low": price, "close": price, "volume": volume_inc
        }])
    last = df.iloc[-1]
    if pd.Timestamp(last["timestamp"]) == ts:
        df = df.copy()
        i = len(df) - 1
        df.at[i, "high"] = max(float(df.at[i, "high"]), price)
        df.at[i, "low"] = min(float(df.at[i, "low"]), price)
        df.at[i, "close"] = price
        df.at[i, "volume"] = float(df.at[i, "volume"]) + volume_inc
        return df
    return upsert_candle(df, {
        "timestamp": ts, "open": price, "high": price, "low": price, "close": price, "volume": volume_inc
    })
