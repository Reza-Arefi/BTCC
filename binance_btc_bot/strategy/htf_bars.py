"""4h ALT/BTC bars built from 1m USDT legs — the exact construction used by the HTF4H research.

``build_alt_btc`` takes high = alt_high / btc_low and low = alt_low / btc_high. On 4h legs that spread is
far wider than the real intrabar range of the ratio, so the ratio is formed per minute and then
aggregated to 4h (open=first, high=max, low=min, close=last, volume=sum), left-labelled on Binance 4h
boundaries (00/04/08/12/16/20 UTC).

Bars are cached per base on disk; after the first backfill only the newest window is fetched
(one 1m request per leg per 4h bar). The newest bar is only accepted once both legs have their final
minute, so a lagging data feed can never freeze a truncated bar into the cache.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from btcc.series.relative import build_alt_btc

from binance_btc_bot.market_data.klines import klines_to_dataframe

logger = logging.getLogger(__name__)

BAR_MS = 14_400_000
MIN_MS = 60_000
PAGE = 1000
COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# fetch(symbol, start_ms, end_ms_inclusive) -> raw 1m kline rows (max PAGE)
OneMinuteFetcher = Callable[[str, int, int], list]


class BarNotReady(RuntimeError):
    """The newest 4h window is not complete in the 1m feed yet."""


def last_closed_open_ms(now_ms: int) -> int:
    return (int(now_ms) // BAR_MS) * BAR_MS - BAR_MS


def aggregate_4h(rel_1m: pd.DataFrame) -> pd.DataFrame:
    if rel_1m is None or rel_1m.empty:
        return pd.DataFrame(columns=COLS)
    df = rel_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    g = df.groupby(df["timestamp"].dt.floor("4h"))
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    }).dropna(subset=["open", "high", "low", "close"])
    return out.rename_axis("timestamp").reset_index()[COLS]


class Rel4hBarStore:
    """Rolling cache of the newest ``bars`` closed 4h ALT/BTC bars per base."""

    def __init__(
        self,
        fetch_1m: OneMinuteFetcher,
        *,
        bars: int,
        cache_dir: Path | str | None = None,
        request_pause_sec: float = 0.05,
        reference: str = "BTCUSDT",
    ) -> None:
        self.fetch_1m = fetch_1m
        self.bars_needed = int(bars)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.request_pause_sec = float(request_pause_sec)
        self.reference = reference.upper()
        self._bars: dict[str, pd.DataFrame] = {}
        self._empty: dict[str, set[int]] = {}
        self._ref_1m: dict[int, pd.DataFrame] = {}

    # ------------------------------------------------------------------ API
    def bars(self, base: str, last_open_ms: int) -> pd.DataFrame:
        """Newest ``bars`` closed bars ending at ``last_open_ms`` (inclusive). Raises BarNotReady."""
        base = base.upper()
        have = self._load(base)
        first = int(last_open_ms) - (self.bars_needed - 1) * BAR_MS
        have_ms = set((have["timestamp"].astype("int64") // 1_000_000).tolist()) if len(have) else set()
        empty = self._empty.setdefault(base, set())
        last = int(last_open_ms)
        missing = [t for t in range(first, last + 1, BAR_MS) if t not in have_ms and t not in empty]
        not_ready: BarNotReady | None = None
        if missing:
            parts = []
            older = [t for t in missing if t != last]
            if older:
                parts.append(self._build(base, older))
            if last in missing:
                try:
                    parts.append(self._build(base, [last], require_complete=True))
                except BarNotReady as e:
                    not_ready = e
                    self._ref_1m.pop(last, None)
            new = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else None
            got = set((new["timestamp"].astype("int64") // 1_000_000).tolist()) if new is not None else set()
            empty.update(t for t in older if t not in got)
            if new is not None:
                have = (pd.concat([have, new], ignore_index=True).drop_duplicates("timestamp", keep="last")
                        .sort_values("timestamp").reset_index(drop=True))
            have = have[have["timestamp"] >= pd.Timestamp(first, unit="ms", tz="UTC")].reset_index(drop=True)
            self._bars[base] = have
            if new is not None:
                self._save(base, have)
            self._prune_ref(first)
        if not_ready is not None:
            raise not_ready
        out = have[have["timestamp"] <= pd.Timestamp(last, unit="ms", tz="UTC")]
        return out.tail(self.bars_needed).reset_index(drop=True)

    # -------------------------------------------------------------- internals
    def _build(self, base: str, missing: list[int], *, require_complete: bool = False) -> pd.DataFrame:
        frames = []
        for start, end in _runs(missing):
            ref = self._ref_range(start, end)
            alt = self._fetch_range(f"{base}USDT", start, end)
            if require_complete:
                self._require_complete(alt, ref, end - BAR_MS, base)
            if alt.empty or ref.empty:
                continue
            rel = build_alt_btc(alt, ref)
            if rel is not None and not rel.empty:
                frames.append(aggregate_4h(rel))
        if not frames:
            return pd.DataFrame(columns=COLS)
        return pd.concat(frames, ignore_index=True)

    def _require_complete(self, alt: pd.DataFrame, ref: pd.DataFrame, open_ms: int, base: str) -> None:
        """Final minute present, or the feed already has a minute after the close (a genuine 1m gap)."""
        close_ms = open_ms + BAR_MS
        last_min = pd.Timestamp(close_ms - MIN_MS, unit="ms", tz="UTC")
        for name, df in ((f"{base}USDT", alt), (self.reference, ref)):
            if not df.empty and df["timestamp"].max() >= last_min:
                continue
            if self.fetch_1m(name, close_ms, close_ms + 5 * MIN_MS - 1):
                continue
            raise BarNotReady(f"{name} 1m feed not complete for 4h bar {pd.Timestamp(open_ms, unit='ms', tz='UTC')}")

    def _ref_range(self, start: int, end: int) -> pd.DataFrame:
        need = [t for t in range(start, end, BAR_MS) if t not in self._ref_1m]
        for s, e in _runs(need):
            df = self._fetch_range(self.reference, s, e)
            for t in range(s, e, BAR_MS):
                lo, hi = pd.Timestamp(t, unit="ms", tz="UTC"), pd.Timestamp(t + BAR_MS, unit="ms", tz="UTC")
                self._ref_1m[t] = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)]
        parts = [self._ref_1m[t] for t in range(start, end, BAR_MS) if len(self._ref_1m.get(t, ()))]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)

    def _prune_ref(self, first: int) -> None:
        for t in [t for t in self._ref_1m if t < first]:
            del self._ref_1m[t]

    def _fetch_range(self, symbol: str, start: int, end: int) -> pd.DataFrame:
        rows: list = []
        cur = start
        while cur < end:
            page = self.fetch_1m(symbol, cur, min(end, cur + PAGE * MIN_MS) - 1)
            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)
            if not page:
                cur += PAGE * MIN_MS
                continue
            rows.extend(page)
            cur = max(cur + MIN_MS, int(page[-1][0]) + MIN_MS)
        df = klines_to_dataframe(rows)
        if df.empty:
            return df
        lo, hi = pd.Timestamp(start, unit="ms", tz="UTC"), pd.Timestamp(end, unit="ms", tz="UTC")
        return df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].reset_index(drop=True)

    def _path(self, base: str) -> Path | None:
        return self.cache_dir / f"{base}.csv" if self.cache_dir else None

    def _load(self, base: str) -> pd.DataFrame:
        if base in self._bars:
            return self._bars[base]
        p = self._path(base)
        df = pd.DataFrame(columns=COLS)
        if p is not None and p.exists():
            try:
                raw = pd.read_csv(p)
                raw["timestamp"] = pd.to_datetime(raw["timestamp"].astype("int64"), unit="ms", utc=True)
                df = raw[COLS].astype({c: float for c in COLS[1:]})
            except Exception as e:  # noqa: BLE001
                logger.warning("rel4h cache unreadable for %s (%s) — rebuilding", base, type(e).__name__)
        self._bars[base] = df
        return df

    def _save(self, base: str, df: pd.DataFrame) -> None:
        p = self._path(base)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["timestamp"] = out["timestamp"].astype("int64") // 1_000_000
        tmp = p.with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(p)


def _runs(opens: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of bar opens -> [(start_ms, end_ms_exclusive)]."""
    out: list[tuple[int, int]] = []
    for t in sorted(opens):
        if out and out[-1][1] == t:
            out[-1] = (out[-1][0], t + BAR_MS)
        else:
            out.append((t, t + BAR_MS))
    return out
