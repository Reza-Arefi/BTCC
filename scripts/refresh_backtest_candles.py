#!/usr/bin/env python3
"""Incrementally refresh backtest candle caches to latest MEXC data.

Preserves existing valid history. Does not fabricate candles.
Uses shared resolve_data_market (CKBTC → CKBTCBTC, never CKBTCUSDT).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("refresh_backtest_candles")

from btcc.config import load_config
from btcc.backtest.config import load_backtest_config
from btcc.backtest.data_loader import required_bars
from btcc.data.candles import load_candles, candle_path
from btcc.data.websocket import MexcPublicREST
from btcc.universe import resolve_data_market
import pandas as pd


def _audit(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"n": 0, "gaps": 0, "dups": 0, "bad": 0, "first": None, "last": None, "days": 0.0}
    ts = pd.to_datetime(df["timestamp"], utc=True)
    first, last = ts.min(), ts.max()
    expected = pd.date_range(first, last, freq="15min", tz="UTC")
    present = set(ts.drop_duplicates())
    gaps = sum(1 for t in expected if t not in present)
    dups = int(ts.duplicated().sum())
    bad = int(
        ((df["high"] < df["low"]) | (df["open"] <= 0) | (df["close"] <= 0) | (df["high"] <= 0) | (df["low"] < 0)).sum()
        + df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    )
    chrono = bool(ts.is_monotonic_increasing)
    return {
        "n": len(df),
        "gaps": gaps,
        "dups": dups,
        "bad": bad,
        "first": str(first),
        "last": str(last),
        "days": float((last - first).total_seconds() / 86400.0),
        "chrono_ok": chrono,
    }


def main() -> int:
    cfg = load_config()
    bt = load_backtest_config()
    for k in ("backtest", "backtest_data", "backtest_output", "_root"):
        if k in bt:
            cfg[k] = bt[k]
    days = 365
    warmup = int(cfg["backtest"]["min_warmup_bars"])
    lookback = required_bars(days, warmup)
    interval = cfg["backtest"]["interval"]
    candle_dir = cfg["backtest_data"]["candle_dir"]
    rest = MexcPublicREST(cfg["data"]["mexc_rest"])

    symbols: list[tuple[str, str]] = [("BTC", cfg["universe"]["btc_symbol"])]
    for base in cfg["universe"]["bases"]:
        meta = resolve_data_market(base, cfg)
        symbols.append((base, meta["resolved_market"]))
        assert meta["resolved_market"] != "CKBTCUSDT" or base != "CKBTC"

    print(f"Incremental refresh lookback_bars={lookback} dir={candle_dir}")
    print(f"{'PAIR':<8} {'MARKET':<12} {'FIRST':<22} {'LAST':<22} {'DAYS':>7} {'N':>7} {'GAPS':>5} {'DUPS':>5} {'BAD':>4} {'CHRONO'}")

    failed = 0
    for logical, market in symbols:
        if market == "CKBTCUSDT":
            raise RuntimeError("BUG: attempted CKBTCUSDT refresh")
        before = load_candles(candle_path(candle_dir, market, interval))
        df = rest.bootstrap_symbol(market, interval, lookback, candle_dir, force=False)
        a = _audit(df)
        ok = a["n"] > 0 and a["dups"] == 0 and a["bad"] == 0 and a.get("chrono_ok", False)
        if not ok:
            failed += 1
        print(
            f"{logical:<8} {market:<12} {str(a['first'] or '-'):<22} {str(a['last'] or '-'):<22} "
            f"{a['days']:7.1f} {a['n']:7d} {a['gaps']:5d} {a['dups']:5d} {a['bad']:4d} "
            f"{'OK' if a.get('chrono_ok') else 'FAIL'}"
        )
        if before is not None and df is not None:
            bmin = pd.to_datetime(before["timestamp"], utc=True).min()
            amin = pd.to_datetime(df["timestamp"], utc=True).min()
            if amin > bmin:
                logger.warning("%s lost older history (%s → %s)", market, bmin, amin)
                failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
