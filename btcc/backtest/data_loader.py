"""Download historical 15m candles for backtest window."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from btcc.data.websocket import MexcPublicREST
from btcc.series.relative import build_alt_btc, native_btc_as_relative
from btcc.universe import resolve_btc_market, resolve_usdt_symbol

logger = logging.getLogger(__name__)

BARS_PER_DAY_15M = 96


def compute_window(days: int, warmup_bars: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (data_start, eval_start, eval_end) pinned at run time UTC."""
    eval_end = pd.Timestamp(datetime.now(timezone.utc))
    eval_start = eval_end - pd.Timedelta(days=days)
    data_start = eval_start - pd.Timedelta(minutes=15 * warmup_bars)
    return data_start, eval_start, eval_end


def required_bars(days: int, warmup_bars: int) -> int:
    return days * BARS_PER_DAY_15M + warmup_bars + 100


def download_panels(cfg: dict, days: int, warmup_bars: int, force: bool = False) -> dict:
    """Download BTC + universe symbols; build relative panels."""
    data_start, eval_start, eval_end = compute_window(days, warmup_bars)
    lookback = required_bars(days, warmup_bars)
    interval = cfg["backtest"]["interval"]
    candle_dir = cfg["backtest_data"]["candle_dir"]
    rest = MexcPublicREST(cfg["data"]["mexc_rest"])

    btc_sym = cfg["universe"]["btc_symbol"]
    btc_df = rest.bootstrap_symbol(btc_sym, interval, lookback, candle_dir, force=force)
    if btc_df is None or btc_df.empty:
        raise RuntimeError(f"DATA_UNAVAILABLE: {btc_sym}")

    panels: dict = {
        "btc": btc_df,
        "coins": {},
        "unavailable": [],
        "construction": {},
    }

    for base in cfg["universe"]["bases"]:
        usdt = resolve_usdt_symbol(base, cfg)
        btc_mkt = resolve_btc_market(base, cfg)
        alt_df = rest.bootstrap_symbol(usdt, interval, lookback, candle_dir, force=force)
        mode = f"{usdt} / BTCUSDT"
        rel = None
        alt_for_vol = None

        if alt_df is not None and not alt_df.empty:
            rel = build_alt_btc(alt_df, btc_df)
            alt_for_vol = alt_df
            sym_key = usdt
        else:
            native = rest.bootstrap_symbol(btc_mkt, interval, lookback, candle_dir, force=force)
            if native is None or native.empty:
                panels["unavailable"].append(base)
                panels["construction"][base] = "DATA_UNAVAILABLE"
                logger.error("DATA_UNAVAILABLE: %s and %s", usdt, btc_mkt)
                continue
            rel = native_btc_as_relative(native)
            alt_for_vol = native
            sym_key = btc_mkt
            mode = f"native {btc_mkt}"

        if rel is None or rel.empty:
            panels["unavailable"].append(base)
            panels["construction"][base] = "INSUFFICIENT_DATA"
            continue

        panels["coins"][base] = {
            "base": base,
            "symbol": sym_key,
            "usdt": usdt,
            "btc_market": btc_mkt,
            "construction": mode,
            "alt_usdt": alt_df if alt_df is not None else alt_for_vol,
            "rel": rel,
            "alt_for_volume": alt_for_vol,
        }
        panels["construction"][base] = mode

    panels["window"] = {
        "data_start": data_start,
        "eval_start": eval_start,
        "eval_end": eval_end,
    }
    return panels
