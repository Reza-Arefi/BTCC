"""Shared data + scoring helpers for the HTF4H investigation.

Data: local 1m USDT candles (data/backtest_candles_binance/1m), 2026-03-01 -> 2026-09-24, 38 live coins.
ALT/BTC is built from the USDT legs (same as all S65 research), then resampled to 4h bars aligned to
00/04/08/12/16/20 UTC (Binance 4h boundaries). A 4h bar labelled t closes at t+4h; the signal is known
then and the entry fills at the next bar's open (= time t+4h).

Score variants (same factors/weights as live, E2 momentum):
  literal  score_variant(interval="4h"): E2 momentum horizons stay in HOURS (4h..24h -> 1..6 bars)
  scaled   score_variant(interval="15m") on 4h bars: every horizon x16 in time (4h..16 days), i.e. the
           exact 15m score shape stretched to 4h bars
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from btcc.research.altbtc_breakout.data import DEFAULT_CANDLE_DIR, _load_symbol_1m, live_universe_bases
from btcc.research.early_breakout_entry import score_variant
from btcc.series.relative import build_alt_btc
from btcc.sim.score import static_factor_weights
from btcc_m.data.load import resample_ohlcv

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "htf_4h"
SIGNAL_CFG = REPO / "configs" / "signal_config.yaml"

DATA_START = pd.Timestamp("2026-03-01 00:00", tz="UTC")
DATA_END = pd.Timestamp("2026-09-24 23:00", tz="UTC")
BAR = pd.Timedelta(hours=4)
FEE_PER_SIDE = 0.001
THRESHOLD = 0.65
VARIANTS = {"literal": "4h", "scaled": "15m"}
FACTOR_KEYS = ("momentum", "trend", "volume", "volatility", "rsi", "structure")

# 4h portfolio: 10 concurrent positions at 10% of realized equity each; SL may go as wide as 25%
SLOTS = 10
SLOT_FRACTION = 0.10
MAX_SL = 0.25


def bases() -> list[str]:
    return list(live_universe_bases())


def weights() -> dict[str, float]:
    return static_factor_weights(yaml.safe_load(SIGNAL_CFG.read_text(encoding="utf-8")))


def _load_1m(candle_dir: Path | tuple[Path, ...], symbol: str) -> pd.DataFrame | None:
    dirs = candle_dir if isinstance(candle_dir, tuple) else (candle_dir,)
    parts = [p for p in (_load_symbol_1m(d, symbol) for d in dirs) if p is not None and not p.empty]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_coin(base: str, candle_dir: Path | tuple[Path, ...] = DEFAULT_CANDLE_DIR, start: pd.Timestamp = DATA_START,
              end: pd.Timestamp = DATA_END) -> dict[str, pd.DataFrame] | None:
    """1m ALT/BTC plus aligned 4h ALT/BTC, ALTUSDT and BTCUSDT bars (complete bars only).
    ``candle_dir`` may be several stores (e.g. 2025 OOS + 2026 main), joined on timestamp."""
    btc = _load_1m(candle_dir, "BTCUSDT")
    alt = _load_1m(candle_dir, f"{base}USDT")
    if btc is None or alt is None:
        return None
    btc = btc[(btc["timestamp"] >= start) & (btc["timestamp"] <= end)].reset_index(drop=True)
    alt = alt[(alt["timestamp"] >= start) & (alt["timestamp"] <= end)].reset_index(drop=True)
    if btc.empty or alt.empty:
        return None
    rel = build_alt_btc(alt, btc)
    if rel is None or rel.empty:
        return None
    rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
    rel4 = resample_ohlcv(rel, "4h")
    rel4 = rel4[rel4["timestamp"] + BAR <= end + pd.Timedelta(minutes=1)].reset_index(drop=True)
    idx = pd.DatetimeIndex(rel4["timestamp"])
    alt4 = resample_ohlcv(alt, "4h").set_index("timestamp").reindex(idx).ffill().reset_index()
    btc4 = resample_ohlcv(btc, "4h").set_index("timestamp").reindex(idx).ffill().reset_index()
    return {"rel1m": rel[["timestamp", "open", "high", "low", "close"]], "rel4": rel4, "alt4": alt4, "btc4": btc4}


def score_bar(d: dict[str, pd.DataFrame], i: int, history: int, variant: str, w: dict[str, float]) -> dict[str, Any]:
    lo = max(0, i + 1 - history)
    sc = score_variant(
        alt_btc=d["rel4"].iloc[lo : i + 1],
        alt_usdt=d["alt4"].iloc[lo : i + 1],
        btc_usdt=d["btc4"].iloc[lo : i + 1],
        weights=w,
        interval=VARIANTS[variant],
        use_e2=True,
    )
    return {"S": float(sc["S"]), **{f"f_{k}": float(sc["factor_scores"][k]) for k in FACTOR_KEYS}}


def score_coin_4h(base: str, variant: str, history: int, candle_dir: Path | tuple[Path, ...] = DEFAULT_CANDLE_DIR,
                  start: pd.Timestamp = DATA_START, end: pd.Timestamp = DATA_END) -> pd.DataFrame:
    """S on every complete 4h bar that has ``history`` bars behind it."""
    d = load_coin(base, candle_dir, start, end)
    if d is None:
        return pd.DataFrame()
    w = weights()
    r4 = d["rel4"]
    rows = []
    for i in range(history - 1, len(r4)):
        ts = r4["timestamp"].iloc[i]
        nxt = i + 1 < len(r4) and r4["timestamp"].iloc[i + 1] - ts == BAR
        rows.append({
            "base": base, "variant": variant, "timestamp": ts, "close_ts": ts + BAR,
            "close": float(r4["close"].iloc[i]),
            "next_open": float(r4["open"].iloc[i + 1]) if nxt else np.nan,
            **score_bar(d, i, history, variant, w),
        })
    return pd.DataFrame(rows)


def new_cross_signals(s: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """NEW_CROSS into S >= threshold on a closed 4h bar; requires the previous bar to be contiguous."""
    s = s.sort_values(["variant", "base", "timestamp"]).reset_index(drop=True)
    key = [s["variant"], s["base"]]
    zone = s["S"] >= threshold
    contiguous = s.groupby(["variant", "base"])["timestamp"].diff() == BAR
    prev = zone.groupby(key).shift(1, fill_value=False) & contiguous
    m = zone & ~prev & contiguous & s["next_open"].notna()
    sig = s.loc[m].copy()
    sig["entry_utc"] = sig["close_ts"]
    sig["entry_px"] = sig["next_open"]
    return sig.reset_index(drop=True)
