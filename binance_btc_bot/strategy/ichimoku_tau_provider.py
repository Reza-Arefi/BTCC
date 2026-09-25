"""Ichimoku I2 + first τ-crossing live entry (ALT/BTC relative).

Matches research: causal Ichimoku (9/26/52/disp26), I2 bullish episode,
first close where ext_pct_close crosses τ while still above kumo top.

Returns a binary score for LiveEntryEngine / LIVE-3 gates:
  1.0 = fire this closed 15m bar (first τ cross in episode)
  0.0 = no fire
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from btcc.series.relative import build_alt_btc
from btcc_m.ichimoku.cloud_extension import (
    attach_extension_features,
    detect_i2_episodes,
    threshold_crossing_events,
)
from btcc_m.ichimoku.indicators import compute_ichimoku

from binance_btc_bot.market_data.klines import (
    closed_candle_age_sec,
    drop_incomplete_candle,
    klines_to_dataframe,
)
from binance_btc_bot.strategy.relative_price import base_from_btc_pair, usdt_pair_for_base
from binance_btc_bot.strategy.score_provider import ScoreSnapshot

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = "15m"
DEFAULT_LOOKBACK = 1000
MIN_HISTORY_BARS = 120
DEFAULT_MAX_CLOSED_AGE_SEC = 1800.0
DEFAULT_TAU = 0.03
FIRE_SCORE = 1.0
IDLE_SCORE = 0.0

KlineFetcher = Callable[[str, str, int], list]


@dataclass
class IchimokuTauProvider:
    """Callable score provider: 1.0 on first τ-cross bar, else 0.0."""

    kline_fetcher: KlineFetcher
    tau: float = DEFAULT_TAU
    interval: str = DEFAULT_INTERVAL
    lookback_bars: int = DEFAULT_LOOKBACK
    min_history_bars: int = MIN_HISTORY_BARS
    long_threshold: float = 0.5  # binary fire uses score 0/1
    max_closed_candle_age_sec: float = DEFAULT_MAX_CLOSED_AGE_SEC
    diagnostics: bool = False
    now_fn: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self._btc_cache: tuple[str | None, pd.DataFrame | None] = (None, None)
        self._last: dict[str, ScoreSnapshot] = {}
        self._last_entry_snapshot: dict[str, dict[str, Any]] = {}
        self._fired: dict[str, set[str]] = {}  # symbol -> episode_ids already armed

    def __call__(self, symbol: str, rel: Mapping[str, Any] | None = None) -> float | None:
        snap = self.evaluate(symbol, rel=rel)
        if snap.S_current is None or not math.isfinite(float(snap.S_current)):
            return None
        return float(snap.S_current)

    def last_entry_snapshot(self, symbol: str) -> dict[str, Any] | None:
        return self._last_entry_snapshot.get(symbol.upper())

    def last_snapshot(self, symbol: str) -> ScoreSnapshot | None:
        return self._last.get(symbol.upper())

    def evaluate(
        self,
        symbol: str,
        *,
        rel: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ScoreSnapshot:
        sym = symbol.upper()
        now_utc = now or self.now_fn()
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        ts_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            rel_df = self._load_relative(sym, now=now_utc)
        except Exception as e:  # noqa: BLE001
            return self._fail(sym, ts_iso, f"LOAD:{e}")

        if rel_df is None or len(rel_df) < self.min_history_bars:
            return self._fail(sym, ts_iso, "INSUFFICIENT_HISTORY")

        age = closed_candle_age_sec(rel_df["timestamp"].iloc[-1], now=now_utc)
        if age is not None and age > self.max_closed_candle_age_sec:
            return self._fail(sym, ts_iso, "STALE_CANDLE", rel_df=rel_df)

        try:
            ichi = attach_extension_features(compute_ichimoku(rel_df))
            # Episodes over full history so open episodes are visible
            t0 = pd.Timestamp(ichi["timestamp"].iloc[0])
            t1 = pd.Timestamp(ichi["timestamp"].iloc[-1])
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            if t1.tzinfo is None:
                t1 = t1.tz_localize("UTC")
            episodes, _ = detect_i2_episodes(ichi, interval=self.interval, eval_start=t0, eval_end=t1)
            ev = threshold_crossing_events(ichi, episodes, [float(self.tau)], interval=self.interval)
        except Exception as e:  # noqa: BLE001
            logger.warning("ichimoku evaluate %s failed: %s", sym, e)
            return self._fail(sym, ts_iso, f"ICHI:{e}", rel_df=rel_df)

        last_ts = pd.Timestamp(rel_df["timestamp"].iloc[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")

        fire = False
        event_id = None
        episode_id = None
        ext = float(ichi["ext_pct_close"].iloc[-1]) if pd.notna(ichi["ext_pct_close"].iloc[-1]) else float("nan")
        fired = self._fired.setdefault(sym, set())

        if ev is not None and not ev.empty:
            # Signal candle = closed bar where τ first crossed; live fires when that bar is the latest closed.
            for _, row in ev.iterrows():
                sig_ts = pd.Timestamp(row["signal_ts"])
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.tz_localize("UTC")
                else:
                    sig_ts = sig_ts.tz_convert("UTC")
                eid = str(row["episode_id"])
                if eid in fired:
                    continue
                # Fire when latest closed bar is the signal bar (exact match on timestamp)
                if abs((sig_ts - last_ts).total_seconds()) < 1.0:
                    fire = True
                    event_id = str(row["event_id"])
                    episode_id = eid
                    ext = float(row["ext_pct_close"])
                    fired.add(eid)
                    break

        s_curr = FIRE_SCORE if fire else IDLE_SCORE
        s_prev = IDLE_SCORE if fire else (FIRE_SCORE if False else IDLE_SCORE)
        # For NEW_CROSS gate: when firing, pretend prior was idle
        if fire:
            s_prev = IDLE_SCORE

        entry_snap = {
            "timestamp": ts_iso,
            "symbol": sym,
            "S_previous": s_prev,
            "S_current": s_curr,
            "cross_detected": fire,
            "long_threshold": float(self.long_threshold),
            "decision_candle_ts": str(last_ts),
            "relative_price": float(rel_df["close"].iloc[-1]),
            "history_bars": len(rel_df),
            "interval": self.interval,
            "entry_mode": "ichimoku_i2_tau",
            "tau": float(self.tau),
            "ext_pct_close": ext,
            "event_id": event_id,
            "episode_id": episode_id,
            "n_episodes": len(episodes),
            "n_tau_events": int(len(ev)) if ev is not None else 0,
        }
        self._last_entry_snapshot[sym] = entry_snap
        snap = ScoreSnapshot(
            timestamp=ts_iso,
            symbol=sym,
            relative_price=float(rel_df["close"].iloc[-1]),
            S_previous=s_prev,
            S_current=s_curr,
            cross_detected=fire,
            decision_candle_ts=str(last_ts),
            reason=None if fire else "NO_TAU_CROSS",
            interval=self.interval,
            long_threshold=self.long_threshold,
            history_bars=len(rel_df),
        )
        self._last[sym] = snap
        if self.diagnostics and fire:
            logger.info("Ichimoku τ-cross FIRE %s ext=%.4f event=%s", sym, ext, event_id)
        return snap

    def _fail(
        self,
        sym: str,
        ts_iso: str,
        reason: str,
        *,
        rel_df: pd.DataFrame | None = None,
    ) -> ScoreSnapshot:
        snap = ScoreSnapshot(
            timestamp=ts_iso,
            symbol=sym,
            relative_price=float(rel_df["close"].iloc[-1]) if rel_df is not None and len(rel_df) else None,
            S_previous=None,
            S_current=None,
            cross_detected=False,
            decision_candle_ts=str(rel_df["timestamp"].iloc[-1]) if rel_df is not None and len(rel_df) else None,
            reason=reason,
            interval=self.interval,
            long_threshold=self.long_threshold,
            history_bars=len(rel_df) if rel_df is not None else 0,
        )
        self._last[sym] = snap
        return snap

    def _load_relative(self, sym: str, *, now: datetime) -> pd.DataFrame:
        base = base_from_btc_pair(sym)
        alt_usdt = usdt_pair_for_base(base)
        alt_raw = self.kline_fetcher(alt_usdt, self.interval, self.lookback_bars)
        btc_raw = self.kline_fetcher("BTCUSDT", self.interval, self.lookback_bars)
        alt_df = drop_incomplete_candle(klines_to_dataframe(alt_raw), interval=self.interval, now=now)
        btc_df = drop_incomplete_candle(klines_to_dataframe(btc_raw), interval=self.interval, now=now)
        if alt_df is None or btc_df is None or alt_df.empty or btc_df.empty:
            raise RuntimeError("empty klines")
        # Prefer cache for BTC
        cache_key = str(btc_df["timestamp"].iloc[-1])
        if self._btc_cache[0] == cache_key and self._btc_cache[1] is not None:
            btc_df = self._btc_cache[1]
        else:
            self._btc_cache = (cache_key, btc_df)
        rel = build_alt_btc(alt_df, btc_df)
        if rel is None or rel.empty:
            raise RuntimeError("build_alt_btc failed")
        return rel.reset_index(drop=True)


def build_ichimoku_tau_provider(
    cfg: dict[str, Any],
    *,
    exchange: Any,
    diagnostics: bool | None = None,
) -> IchimokuTauProvider:
    signal = dict(cfg.get("signal") or {})
    entry = cfg.get("entry") or {}
    tau = float(entry.get("tau") or signal.get("tau") or DEFAULT_TAU)
    thr = float(entry.get("long_threshold") or 0.5)
    diag = bool(signal.get("diagnostics", False)) if diagnostics is None else bool(diagnostics)

    def _fetch(symbol: str, interval: str, limit: int) -> list:
        return list(exchange.get_klines(symbol, interval=interval, limit=limit))

    return IchimokuTauProvider(
        kline_fetcher=_fetch,
        tau=tau,
        interval=str(signal.get("interval") or DEFAULT_INTERVAL),
        lookback_bars=int(signal.get("lookback_bars") or DEFAULT_LOOKBACK),
        min_history_bars=int(signal.get("min_history_bars") or MIN_HISTORY_BARS),
        long_threshold=thr,
        max_closed_candle_age_sec=float(
            signal.get("max_closed_candle_age_sec") or DEFAULT_MAX_CLOSED_AGE_SEC
        ),
        diagnostics=diag,
    )
